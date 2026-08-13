# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
KVConnectorBase_V1 Class for Distributed KV Cache & Hidden State
communication in vLLM v1

The class provides the following primitives:
    Scheduler-side: runs in the scheduler, binds metadata, which
    is used by the worker-side to load/save KV cache.
        get_num_new_matched_tokens() - get number of new tokens
            that exist in the remote KV cache. Might be called multiple
            times for a given request and should be side-effect free.
        update_state_after_alloc() - update KVConnector state after
            temporary buffer alloc by the CacheManager.
        update_connector_output() - update KVConnector state after
            output is received from worker-side connectors.
        request_finished() - called once when a request is finished,
            with the computed kv cache blocks for the request.
            Returns whether KV cache should be freed now or if the
            connector now assumes responsibility for freeing the
            the blocks asynchronously. Also optionally returns KV
            transfer params.
        take_events() - returns new KV events that were collected
            by the connector since the last call.

    Worker-side: runs in each worker, loads/saves KV cache to/from
    the Connector based on the metadata.
        handle_preemptions() - called for handling preempted requests
            or request evicted blocks before they are overwritten

        start_load_kv() - starts loading all KVs (maybe async)
        wait_for_layer_load() - blocks until layer i load is done

        save_kv_layer() - starts saving KV for layer i (maybe async)
        wait_for_save() - blocks until all saves are done

        get_finished() - called with ids of finished requests, returns
            ids of requests that have completed async sending/recving.
        build_connector_worker_meta() - builds metadata to be sent
            back to the scheduler-side connector
"""

import enum
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any, Literal

import torch

from vllm.logger import init_logger
from vllm.v1.attention.backend import AttentionBackend, AttentionMetadata
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.outputs import KVConnectorOutput

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.distributed.kv_events import KVCacheEvent, KVConnectorKVEvents
    from vllm.distributed.kv_transfer.kv_connector.v1.metrics import (
        KVConnectorPromMetrics,
        KVConnectorStats,
        PromMetric,
        PromMetricT,
    )
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.block_pool import BlockPool
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

# s_tensor_list, d_tensor_list, s_indices, d_indices, direction
CopyBlocksOp = Callable[
    [
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
        list[int],
        list[int],
        Literal["h2d", "d2h"],
    ],
    None,
]

logger = init_logger(__name__)


class SupportsHMA(ABC):
    """
    The class that indicates the corresponding connector supports hybrid memory
    allocator (HMA).
    This is required to use the connector together with hybrid memory allocator.
    """

    @abstractmethod
    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        """
        Called exactly once when a request has finished for all kv cache groups,
        before its blocks are freed for each group.

        NOTE(Kuntai): This function is only supported by connectors that support HMA.

        The connector may assumes responsibility for freeing the blocks
        asynchronously by returning True.

        Returns:
            True if the request is being saved/sent asynchronously and blocks
            should not be freed until the request_id is returned from
            get_finished().
            Optional KVTransferParams to be included in the request outputs
            returned by the engine.
        """
        # ------【PD 分离】抽象方法：请求在所有 KV 组都结束时触发一次，决定是否异步接管块释放 ------
        raise NotImplementedError


def supports_hma(connector: Any) -> bool:
    # ------【PD 分离】判断 connector 是类还是实例，据此判定是否支持 HMA 混合显存分配 ------
    if isinstance(connector, type):
        return issubclass(connector, SupportsHMA)
    else:
        return isinstance(connector, SupportsHMA)


class KVConnectorRole(enum.Enum):
    # Connector running in the scheduler process
    SCHEDULER = 0

    # Connector running in the worker process
    WORKER = 1


class KVConnectorHandshakeMetadata(ABC):  # noqa: B024
    """
    Metadata used for out of band connector handshake between
    P/D workers. This needs to serializable.
    """

    # ------【PD 分离】空基类仅作类型标记，具体握手元数据须可序列化以跨进程传递 ------
    pass


class KVConnectorMetadata(ABC):  # noqa: B024
    """
    Abstract Metadata used to communicate
    Scheduler KVConnector -> Worker KVConnector.
    """

    # ------【PD 分离】空基类标记调度器侧下发到 worker 侧的连接器元数据 ------
    pass


class KVConnectorWorkerMetadata(ABC):
    """
    Abstract Metadata used to communicate back
    Worker KVConnector -> Scheduler KVConnector.

    Each worker can output its own metadata.
    For a single engine step, all metadata objects returned by workers
    will be aggregated using the `aggregate` method below, before
    being passed to the Scheduler KVConnector.
    """

    @abstractmethod
    def aggregate(
        self, other: "KVConnectorWorkerMetadata"
    ) -> "KVConnectorWorkerMetadata":
        """
        Aggregate metadata with another `KVConnectorWorkerMetadata` object.
        """
        # ------【TP+PP】抽象方法：把多个 worker 回传的元数据聚合为一份再交调度器 ------
        pass


class KVConnectorBase_V1(ABC):
    """
    Base class for KV connectors.
    """

    @property
    def prefer_cross_layer_blocks(self) -> bool:
        """
        Indicates whether this connector prefers KV blocks that hold KV data for all
        layers, which can speed up KV data transfers. Defaults to False.
        """
        # ------【PD 分离】默认不偏好跨层共享块；开启后可减少逐层传输次数 ------
        return False

    @property
    def requires_kv_delivery(self) -> bool:
        """Whether this connector hands off KV that must be reliably delivered.

        If True, a request preempted while its hand-off is still pending is
        recomputed rather than allowed to finish and hand off blocks that the
        preemption already freed. Defaults to the producer role, since only a
        producer hands KV off when a request completes. Best-effort caches
        return False, as a dropped save is just a future cache miss.
        """
        # ------【PD 分离】仅生产者需可靠投递 KV；尽力而为缓存丢了也只是未来一次 miss ------
        return self._kv_transfer_config.is_kv_producer

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig",
    ):
        # ------【核心逻辑】打日志提示该 KV 传输 API 仍属实验性质，接口可能变动 ------
        logger.warning(
            "Initializing KVConnectorBase_V1. This API is experimental and "
            "subject to change in the future as we iterate the design."
        )
        # ------【核心逻辑】初始化连接器元数据为空，待调度器经 bind 下发给 worker ------
        self._connector_metadata: KVConnectorMetadata | None = None
        self._vllm_config = vllm_config
        # ------【核心逻辑】强制要求 kv_transfer_config 存在，否则无传输参数可用 ------
        if vllm_config.kv_transfer_config is not None:
            self._kv_transfer_config = vllm_config.kv_transfer_config
        else:
            raise ValueError("kv_transfer_config must be set for KVConnectorBase_V1")
        # ------【核心逻辑】缓存 KV cache 配置与进程角色，供后续 load/save 分支使用 ------
        self._kv_cache_config = kv_cache_config
        self._role = role

    @property
    def role(self) -> KVConnectorRole:
        # ------【核心逻辑】暴露进程角色，便于区分调度器侧与 worker 侧的行为 ------
        return self._role

    # ==============================
    # Worker-side methods
    # ==============================

    def bind_connector_metadata(self, connector_metadata: KVConnectorMetadata) -> None:
        """Set the connector metadata from the scheduler.

        This function should be called by the model runner every time
        before the model execution. The metadata will be used for runtime
        KV cache loading and saving.

        Args:
            connector_metadata (dict): the connector metadata.
        """
        # ------【PD 分离】模型执行前由 model runner 写入调度器下发的元数据，供运行时 load/save 使用 ------
        self._connector_metadata = connector_metadata

    def clear_connector_metadata(self) -> None:
        """Clear the connector metadata.

        This function should be called by the model runner every time
        after the model execution.
        """
        # ------【PD 分离】模型执行后清空元数据，避免下次 step 误用上一次的加载指令 ------
        self._connector_metadata = None

    def _get_connector_metadata(self) -> KVConnectorMetadata:
        """Get the connector metadata.

        This function should only be called inside the connector.

        Returns:
            ConnectorMetadata: the connector metadata.
        """
        # Should only be called while set to valid metadata.
        # ------【核心逻辑】断言元数据已被绑定，防止在未下发时访问空元数据 ------
        assert self._connector_metadata is not None
        return self._connector_metadata

    def has_connector_metadata(self) -> bool:
        """Check whether the connector metadata is currently set.

        Returns:
            bool: True if connector metadata exists, False otherwise.
        """
        # ------【核心逻辑】查询当前 step 是否有待执行的连接器元数据 ------
        return self._connector_metadata is not None

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        """
        Initialize with the KV caches. Useful for pre-registering the
        KV Caches in the KVConnector (e.g. for NIXL).

        Args:
            kv_caches: dictionary of layer names, kv cache
        """
        # ------【核心逻辑】默认无操作；子类可预注册 KV cache 加速后续传输（如 NIXL） ------
        return

    def register_cross_layers_kv_cache(
        self, kv_cache: torch.Tensor, attn_backend: type["AttentionBackend"]
    ):
        """
        Initialize with a single KV cache tensor used by all layers.
        The first dimension should be num_layers.
        This function will only be called for models with uniform layers,
        and only if the prefers_cross_layer_blocks is set to True.
        Only one of the functions
        {register_kv_caches, register_cross_layers_kv_cache} will be called.

        Args:
            kv_cache: a cross-layers kv cache tensor
            attn_backend: The attention backend that corresponds to all layers
        """
        # ------【PD 分离】默认无操作；跨层共享 KV 块可减少传输次数，需子类重写 ------
        return

    def set_host_xfer_buffer_ops(self, copy_operation: CopyBlocksOp):
        """
        Set the xPU-specific ops for copying KV between host and device.
        Needed when host buffer is used for kv transfer (e.g., in NixlConnector)
        """
        # ------【异步 RPC】默认无操作；子类注入 host/device 间拷贝算子以支持主机缓冲传输 ------
        return

    def handle_preemptions(self, kv_connector_metadata: KVConnectorMetadata):
        """
        Handle preempted requests or evicted blocks BEFORE they are overwritten.
        Needed for connectors which use async saves (e.g., OffloadingConnector)
        """
        # ------【PD 分离+异步 RPC】默认无操作；在块被覆盖前处理被抢占/驱逐请求的异步保存 ------
        return

    @abstractmethod
    def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:
        """
        Start loading the KV cache from the connector to vLLM's paged
        KV buffer. This is called from the forward context before the
        forward pass to enable async loading during model execution.

        Args:
            forward_context (ForwardContext): the forward context.
            **kwargs: additional arguments for the load operation

        Note:
            The number of elements in kv_caches and layer_names should be
            the same.

        """
        # ------【PD 分离+异步 RPC】抽象方法：前向开始前启动异步加载 KV，与计算重叠隐藏传输延迟 ------
        pass

    @abstractmethod
    def wait_for_layer_load(self, layer_name: str) -> None:
        """
        Block until the KV for a specific layer is loaded into vLLM's
        paged buffer. This is called from within attention layer to ensure
        async copying from start_load_kv is complete.

        This interface will be useful for layer-by-layer pipelining.

        Args:
            layer_name: the name of that layer
        """
        # ------【PD 分离+异步 RPC】抽象方法：阻塞等待某层 KV 到位，实现逐层流水线加载 ------
        pass

    @abstractmethod
    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: "AttentionMetadata",
        **kwargs: Any,
    ) -> None:
        """
        Start saving a layer of KV cache from vLLM's paged buffer
        to the connector. This is called from within attention layer to
        enable async copying during execution.

        Args:
            layer_name (str): the name of the layer.
            kv_layer (torch.Tensor): the paged KV buffer of the current
                layer in vLLM.
            attn_metadata (AttentionMetadata): the attention metadata.
            **kwargs: additional arguments for the save operation.
        """
        # ------【PD 分离+异步 RPC】抽象方法：注意力层内启动单层 KV 异步保存，与执行重叠 ------
        pass

    @abstractmethod
    def wait_for_save(self):
        """
        Block until all the save operations is done. This is called
        as the forward context exits to ensure that the async saving
        from save_kv_layer is complete before finishing the forward.

        This prevents overwrites of paged KV buffer before saving done.
        """
        # ------【PD 分离+异步 RPC】抽象方法：前向退出时等待所有异步保存完成，防止 KV 缓冲被覆盖 ------
        pass

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[set[str] | None, set[str] | None]:
        """
        Notifies worker-side connector ids of requests that have
        finished generating tokens on the worker.
        The scheduler process (via the Executors) will use this output
        to track which workers are done.

        Returns:
            ids of requests that have finished asynchronous transfer
            (requests that previously returned True from request_finished()),
            tuple of (sending/saving ids, recving/loading ids).
            The finished saves/sends req ids must belong to a set provided in a
            call to this method (this call or a prior one).
        """
        # ------【异步 RPC】默认无已完成传输；返回 (保存完成集合, 加载完成集合) 供调度器跟踪 ------
        return None, None

    def get_block_ids_with_load_errors(self) -> set[int]:
        """
        Get the set of block IDs that failed to load.

        Returns:
            Set of block IDs that encountered load errors.
            Empty set if no load errors occurred.

        Notes:
            - Applies to both sync- and async-loading requests.
            - Async loading: failed blocks may be reported in any forward pass
              up to and including the pass where the request ID is returned by
              `get_finished()`. Even if failures occur, the request must still
              be reported via `get_finished()`, and the failed block IDs must
              appear here no later than that same pass.
            - Sync loading: failed blocks should be reported in the forward
              pass in which they are detected.
        """
        # ------【PD 分离】默认无加载失败块；上报加载失败块 ID 供调度器降级/重算处理 ------
        return set()

    def shutdown(self):
        """
        Shutdown the connector. This is called when the worker process
        is shutting down to ensure that all the async operations are
        completed and the connector is cleaned up properly.
        """
        # ------【异步 RPC】默认无操作；worker 退出时需等待所有异步传输完成并清理资源 ------
        return None

    def get_kv_connector_stats(self) -> "KVConnectorStats | None":
        """
        Get the KV connector stats collected during the last interval.
        """
        # ------【显存 profiling】默认无统计；返回上个区间的传输统计用于观测吞吐与延迟 ------
        return None

    def get_kv_connector_kv_cache_events(self) -> "KVConnectorKVEvents | None":
        """
        Get the KV connector kv cache events collected during the last interval.
        This function should be called by the model runner every time after the
        model execution and before cleanup.
        """
        # ------【核心逻辑】默认无事件；返回上个区间的 KV cache 事件供前缀缓存等消费 ------
        return None

    def get_handshake_metadata(self) -> KVConnectorHandshakeMetadata | None:
        """
        Get the KVConnector handshake metadata for this connector.
        This metadata is used for out-of-band connector handshake
        between P/D workers.

        Returns:
            KVConnectorHandshakeMetadata: the handshake metadata.
            None if no handshake metadata is available.
        """
        # ------【PD 分离】默认无握手元数据；P/D worker 间带外握手交换传输参数 ------
        return None

    def build_connector_worker_meta(self) -> KVConnectorWorkerMetadata | None:
        """
        Build the KVConnector worker metadata for this engine step.

        Returns:
            KVConnectorWorkerMetadata: the worker metadata.
            None if no worker metadata is available.
        """
        # ------【PD 分离】默认无回传元数据；worker 侧结果聚合后回传调度器侧连接器 ------
        return None

    # ==============================
    # Scheduler-side methods
    # ==============================

    def bind_gpu_block_pool(self, gpu_block_pool: "BlockPool") -> None:
        """
        Bind the GPU block pool to the connector for per-GPU block status tracking.
        For example, inc/dec ref counts, or iterate over the prefix cache blocks.

        Args:
            gpu_block_pool: the GPU block pool.
        """
        # ------【前缀缓存】默认无操作；绑定 GPU 块池以跟踪引用计数/迭代前缀缓存块 ------
        return

    @abstractmethod
    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        """
        Get number of new tokens that can be loaded from the
        external KV cache beyond the num_computed_tokens.

        Args:
            request (Request): the request object.
            num_computed_tokens (int): the number of locally
                computed tokens for this request

        Returns:
            A tuple with the following elements:
                - An optional number of tokens that can be loaded from the
                  external KV cache beyond what is already computed.
                  If None, it means that the connector needs more time to
                  determine the number of matched tokens, and the scheduler
                  should query for this request again later.
                - `True` if external KV cache tokens will be loaded
                  asynchronously (between scheduler steps). Must be
                  'False' if the first element is 0.

        Notes:
            The connector should only consider the largest prefix of prompt-
            tokens for which KV cache is actually available at the time of the
            call. If the cache cannot be loaded for some tokens (e.g., due to
            connectivity issues or eviction), those tokens must not be taken
            into account.
        """
        # ------【PD 分离+前缀缓存】抽象方法：查询外部 KV cache 可复用的新 token 数，供调度器决定加载量 ------
        pass

    @abstractmethod
    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        """
        Update KVConnector state after block allocation.

        If get_num_new_matched_tokens previously returned True for a
        request, this function may be called twice for that same request -
        first when blocks are allocated for the connector tokens to be
        asynchronously loaded into, and second when any additional blocks
        are allocated, after the load/transfer is complete.

        Decide whether to load based on ``num_external_tokens``, not on
        whether ``blocks`` is empty: ``blocks`` may be non-empty even when
        ``num_external_tokens == 0`` (e.g. a non-chosen sub-connector of
        MultiConnector still receives the request's real blocks).

        Args:
            request (Request): the request object.
            blocks (KVCacheBlocks): the blocks allocated for the request.
            num_external_tokens (int): the number of tokens to load from the
                external KV cache. 0 means nothing should be loaded.
        """
        # ------【PD 分离】抽象方法：块分配后更新状态，记录本次需从外部加载的 token 数 ------
        pass

    @abstractmethod
    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        """
        Build the connector metadata for this step.

        This function should NOT modify fields in the scheduler_output.
        Also, calling this function will reset the state of the connector.

        Args:
            scheduler_output (SchedulerOutput): the scheduler output object.
        """
        # ------【PD 分离】抽象方法：依据调度输出构建下发元数据，并重置本步连接器状态 ------
        pass

    def on_new_request(self, request: "Request") -> None:
        """Called by the scheduler when a new request is added.

        Connectors can override this to inspect the request and perform
        bookkeeping. The default implementation is a no-op.
        """
        # ------【核心逻辑】默认无操作；新请求加入时子类可做登记/记账 ------
        return

    def update_connector_output(self, connector_output: KVConnectorOutput):
        """
        Update KVConnector state from worker-side connectors output.

        Args:
            connector_output (KVConnectorOutput): the worker-side
                connectors output.
        """
        # ------【PD 分离】默认无操作；消费 worker 侧回传结果以更新调度器侧状态 ------
        return

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        """
        Called exactly once when a request has finished, before its blocks are
        freed.

        The connector may assumes responsibility for freeing the blocks
        asynchronously by returning True.

        Returns:
            True if the request is being saved/sent asynchronously and blocks
            should not be freed until the request_id is returned from
            get_finished().
            Optional KVTransferParams to be included in the request outputs
            returned by the engine.
        """
        # ------【PD 分离】默认同步释放；返回 False 表示不接管块、由调度器立即释放 ------
        return False, None

    def take_events(self) -> Iterable["KVCacheEvent"]:
        """
        Take the KV cache events from the connector.

        Yields:
            New KV cache events since the last call.
        """
        # ------【核心逻辑】默认无事件；拉取自上次调用以来新产生的 KV cache 事件 ------
        return ()

    def has_pending_push_work(self) -> bool:
        """Return True if the connector has push-mode work that requires
        the engine main loop to keep stepping (e.g. a P-side request whose
        KV blocks are waiting to be WRITTEN to a D node).

        Connectors that don't implement push-based KV transfer should
        leave this as False.
        """
        # TODO: replace with a more general connector hook for keeping the
        # scheduler alive (e.g. extend has_unfinished_requests).
        # ------【PD 分离】默认无推送任务；push 模式需主循环持续步进以写出 KV ------
        return False

    @classmethod
    def get_required_kvcache_layout(cls, vllm_config: "VllmConfig") -> str | None:
        """
        Get the required KV cache layout for this connector.
        Args:
            vllm_config (VllmConfig): the vllm config.

        Returns:
            str: the required KV cache layout. e.g. HND, or NHD.
            None if the connector does not require a specific layout.
        """

        # ------【核心逻辑】抽象基类禁止调用，防止基类被当作具体布局声明 ------
        if cls is KVConnectorBase_V1:
            raise TypeError(
                "get_required_kvcache_layout should not be called "
                "on the abstract base class"
            )
        # ------【核心逻辑】默认不要求特定 KV cache 布局（如 HND/NHD），子类可重写 ------
        return None

    @classmethod
    def requires_piecewise_for_cudagraph(cls, extra_config: dict[str, Any]) -> bool:
        """
        Check if this connector requires PIECEWISE CUDA graph mode.

        Connectors that use asynchronous layer-by-layer operations
        (wait_for_layer_load/save_kv_layer) should override this method
        to return True when those operations are enabled. These operations
        cannot be captured in CUDA graphs and will be skipped during replay,
        causing data races. PIECEWISE mode allows Python code to execute
        between graph pieces, ensuring proper synchronization.

        Args:
            extra_config: The kv_connector_extra_config dict from
                KVTransferConfig.

        Returns:
            True if this connector requires PIECEWISE CUDA graph mode,
            False otherwise.
        """
        # ------【CUDA Graph】默认不要求分片图；逐层异步操作无法被捕获需启用 PIECEWISE ------
        return False

    def get_finished_count(self) -> int | None:
        """
        Get the count of requests expected to complete send/receive operations
        via this connector. This method is used to initialize the
        KVOutputAggregator, overwriting the default world_size.

        Returns:
            int: expected sending or receiving completion count.
        """

        # ------【异步 RPC】默认返回 None；用于初始化输出聚合器覆盖默认 world_size ------
        return None

    @classmethod
    def build_kv_connector_stats(
        cls, data: dict[str, Any] | None = None
    ) -> "KVConnectorStats | None":
        """
        KVConnectorStats resolution method. This method allows dynamically
        registered connectors to return their own KVConnectorStats object,
        which can implement custom aggregation logic on the data dict.
        """
        # ------【显存 profiling】默认无统计对象；动态注册的连接器可返回自定义聚合逻辑 ------
        return None

    def set_xfer_handshake_metadata(
        self, metadata: dict[int, KVConnectorHandshakeMetadata]
    ) -> None:
        """
        Set the KV connector handshake metadata for this connector.

        Args:
            metadata (KVConnectorHandshakeMetadata): the handshake metadata to set.
        """
        # ------【PD 分离】默认无操作；接收对端 P/D worker 的握手元数据供传输协商 ------
        return None

    def set_xfer_handshake_metadata_pp_aware(
        self, metadata: dict[tuple[int, int], KVConnectorHandshakeMetadata]
    ) -> None:
        """
        Set handshake metadata keyed by (pp_rank, tp_rank).
        - Default implementation assumes pp_rank is always 0
        - PP-aware connectors override this to consume all PP producer shards.
        """
        # ------【PP】检测是否存在 pp_rank>0 的分片，若不支持 PP 分离则拒绝 ------
        if any(pp_rank != 0 for pp_rank, _ in metadata):
            raise ValueError(
                f"{type(self).__name__} received pp_rank > 0 handshake metadata "
                "but does not support PP-disaggregated KV transfer."
            )
        # ------【PP+TP】默认忽略 PP 维度，仅按 tp_rank 重键并下发给子类 ------
        self.set_xfer_handshake_metadata(
            {tp_rank: meta for (_, tp_rank), meta in metadata.items()}
        )

    @classmethod
    def build_prom_metrics(
        cls,
        vllm_config: "VllmConfig",
        metric_types: dict[type["PromMetric"], type["PromMetricT"]],
        labelnames: list[str],
        per_engine_labelvalues: dict[int, list[object]],
    ) -> "KVConnectorPromMetrics | None":
        """
        Create a KVConnectorPromMetrics subclass which should register
        per-connector Prometheus metrics and implement observe() to
        expose connector transfer stats via Prometheus.
        """
        # ------【显存 profiling】默认不建指标；子类注册 Prometheus 指标以暴露传输统计 ------
        return None

    def reset_cache(self) -> bool | None:
        """
        Reset the connector's internal cache.

        Returns:
            bool: True if the cache was successfully reset, False otherwise.
        """
        # ------【前缀缓存】默认未实现重置，仅打日志并返回 None 表示结果未知 ------
        logger.debug(
            "Connector cache reset requested, but %s does not implement reset_cache().",
            type(self).__name__,
        )

        return None
