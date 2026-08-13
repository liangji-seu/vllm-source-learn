# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
KV cache helper for store.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast

import torch

from vllm.config import (
    VllmConfig,
    get_current_vllm_config,
    get_layers_from_vllm_config,
    set_current_vllm_config,
)
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.logger import init_logger
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.platforms import current_platform
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.outputs import KVConnectorOutput, ModelRunnerOutput

if TYPE_CHECKING:
    from vllm.distributed.kv_transfer.kv_connector.base import KVConnectorBase

logger = init_logger(__name__)

EngineId = str
# block ids as returned by the hybrid KV cache manager. list[list[int]] are allow
# mutability and are for connector internal use only.
BlockIds = tuple[list[int], ...] | list[list[int]]


def get_kv_connector_cache_layout():
    # NOTE (NickLucche) When running disaggregated PD with NIXL, HND layout is
    # used for faster transfer.
    # ------【PD 分离】从当前线程上下文取出 vllm 配置，再取 KV transfer 配置 ------
    vllm_config = get_current_vllm_config()
    kv_config = vllm_config.kv_transfer_config
    # ------【PD 分离】仅当配置了 connector 时，才向 connector 查询其要求的 cache 布局 ------
    if kv_config is not None:
        connector_cls = KVConnectorFactory.get_connector_class(kv_config)
        required_kvcache_layout = connector_cls.get_required_kvcache_layout(vllm_config)
        # ------【PD 分离】connector 指定了布局(如 NIXL 的 HND)则直接用，加速传输 ------
        if required_kvcache_layout is not None:
            return required_kvcache_layout
        logger.info_once(
            "Connectors do not specify a kv cache layout, defaulting to NHD."
        )
    # ------【PD 分离】connector 未指定或无 connector 时，回退默认 NHD 布局 ------
    return "NHD"


class KVOutputAggregator:
    """Utility class to aggregate the output of all workers into a single
    output corresponding to Rank 0 for scheduler."""

    def __init__(self, expected_finished_count: int):
        # Complete transfer tracker. Used to track finished requests
        # [req_id -> n_remaining_workers]
        # ------【PD 分离+异步 RPC】维护收/发两个倒计数表：req_id 到剩余 worker 数的映射 ------
        self._recv_remaining_count = dict[str, int]()
        self._send_remaining_count = dict[str, int]()
        self._expected_finished_count = expected_finished_count

    @classmethod
    def from_connector(cls, connector: "KVConnectorBase", world_size: int):
        # ------【PD 分离】优先用 connector 声明的完成计数，未声明则回退 world_size ------
        return cls(connector.get_finished_count() or world_size)

    def aggregate(
        self, outputs: list[ModelRunnerOutput | None], output_rank: int = 0
    ) -> ModelRunnerOutput | None:
        # ------【PD 分离】output_rank 对应的输出为空则直接返回 None，无聚合对象 ------
        if not outputs[output_rank]:
            return None

        # Aggregate kv_connector_output from all workers

        # ------【PD 分离】内部工具：对每个完成请求把剩余 worker 数减一，减到 0 则标记完成 ------
        def update_finished_set(
            req_ids: set[str] | None,
            remaining_count_dict: dict[str, int],
            finished_set: set[str],
        ) -> None:
            # ------【PD 分离】遍历本轮上报完成的 req_id，缺省用期望完成数初始化倒计数 ------
            for req_id in req_ids or ():
                remaining_count = remaining_count_dict.get(
                    req_id, self._expected_finished_count
                )
                # ------【PD 分离】每收到一个 worker 的完成信号就减一 ------
                remaining_count_dict[req_id] = remaining_count - 1
                # ------【PD 分离】倒计数归零说明所有 worker 都完成，加入 finished 并清理表项 ------
                if remaining_count_dict[req_id] == 0:
                    finished_set.add(req_id)
                    del remaining_count_dict[req_id]

        # ------【PD 分离】初始化聚合容器：完成集合、统计累加器、事件合并器、无效块集合 ------
        finished_sending = set[str]()
        finished_recving = set[str]()
        aggregated_kv_connector_stats = None
        aggregated_kv_connector_worker_meta = None
        combined_kv_cache_events = None
        invalid_block_ids = set[int]()
        # ------【PD 分离+TP】遍历所有 worker 的模型输出，逐份聚合 KV connector 结果 ------
        for model_runner_output in outputs:
            assert model_runner_output is not None
            kv_output = model_runner_output.kv_connector_output
            # ------【PD 分离】该 worker 无 KV 输出则跳过，不参与聚合 ------
            if not kv_output:
                continue
            # Allow the worker to dynamically update the expected number of
            # finished sending/recving for new requests.
            # ------【PD 分离】允许 worker 动态上调期望完成数(新请求加入)，并同步到聚合器 ------
            if (
                kv_output.expected_finished_count > 0
                and kv_output.expected_finished_count != self._expected_finished_count
            ):
                logger.debug(
                    "Expected finished requests updated from %d to %d",
                    self._expected_finished_count,
                    kv_output.expected_finished_count,
                )
                self._expected_finished_count = kv_output.expected_finished_count

            # ------【PD 分离】把该 worker 上报的发送/接收完成集合并入倒计数聚合 ------
            update_finished_set(
                kv_output.finished_sending, self._send_remaining_count, finished_sending
            )
            update_finished_set(
                kv_output.finished_recving, self._recv_remaining_count, finished_recving
            )

            # Aggregate kv_connector_stats from all workers.
            if aggregated_kv_connector_stats is None:
                # Use the first worker's kv_connector_stats as accumulator.
                # ------【PD 分离】首个 worker 的 stats 直接作为累加器起点 ------
                aggregated_kv_connector_stats = kv_output.kv_connector_stats
            elif kv_connector_stats := kv_output.kv_connector_stats:
                assert isinstance(
                    aggregated_kv_connector_stats, type(kv_connector_stats)
                )
                # ------【PD 分离】后续 worker 的 stats 逐个调用 aggregate 叠加到累加器 ------
                aggregated_kv_connector_stats = aggregated_kv_connector_stats.aggregate(
                    kv_connector_stats
                )

            # Aggregate kv_connector_worker_meta from all workers.
            if aggregated_kv_connector_worker_meta is None:
                # Use the first worker's kv_connector_worker_meta as accumulator.
                # ------【PD 分离】首个 worker 的 worker_meta 作为累加器起点 ------
                aggregated_kv_connector_worker_meta = kv_output.kv_connector_worker_meta
            elif kv_connector_worker_meta := kv_output.kv_connector_worker_meta:
                # ------【PD 分离】后续 worker 的 worker_meta 逐个 aggregate 叠加 ------
                aggregated_kv_connector_worker_meta = (
                    aggregated_kv_connector_worker_meta.aggregate(
                        kv_connector_worker_meta
                    )
                )

            # Combine kv_cache_events from all workers.
            if combined_kv_cache_events is None:
                # Use the first worker's kv_cache events as start event list.
                # ------【PD 分离】首个 worker 的 cache 事件列表作为合并起点 ------
                combined_kv_cache_events = kv_output.kv_cache_events
            elif kv_cache_events := kv_output.kv_cache_events:
                assert isinstance(
                    combined_kv_cache_events,
                    type(kv_cache_events),
                )
                # ------【PD 分离】取出该 worker 的事件并追加到合并列表，同时递增 worker 计数 ------
                worker_kv_cache_events = kv_cache_events.get_all_events()
                combined_kv_cache_events.add_events(worker_kv_cache_events)
                combined_kv_cache_events.increment_workers(1)

            # ------【PD 分离】把该 worker 的无效块 id 并入全局集合(并集) ------
            invalid_block_ids |= kv_output.invalid_block_ids

        # select output of the worker specified by output_rank
        output = outputs[output_rank]

        assert output is not None
        # ------【PD 分离】用聚合结果重建 KVConnectorOutput，回填到指定 rank 的输出对象 ------
        output.kv_connector_output = KVConnectorOutput(
            finished_sending=finished_sending or None,
            finished_recving=finished_recving or None,
            kv_connector_stats=aggregated_kv_connector_stats or None,
            kv_cache_events=combined_kv_cache_events or None,
            kv_connector_worker_meta=aggregated_kv_connector_worker_meta or None,
            invalid_block_ids=invalid_block_ids,
            expected_finished_count=self._expected_finished_count,
        )

        return output


def _make_src_and_dst_indices(
    src_block_ids: list[int],
    dst_block_ids: list[int],
    src_device: torch.device | str,
    dst_device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor]:
    # ------【PD 分离】把源/目标 block id 列表转成对应设备上的 int64 索引张量，供设备侧拷贝用 ------
    src_indices = torch.tensor(src_block_ids, device=src_device, dtype=torch.int64)
    dst_indices = torch.tensor(dst_block_ids, device=dst_device, dtype=torch.int64)
    return src_indices, dst_indices


def copy_kv_blocks(
    src_kv_caches: dict[str, torch.Tensor],
    dst_kv_caches: dict[str, torch.Tensor],
    src_block_ids: list[int],
    dst_block_ids: list[int],
    direction: Literal["h2d", "d2h"],
) -> None:
    """Copy kv blocks between different buffers."""
    # ------【PD 分离】空缓存或 block id 数量不匹配则直接返回，避免非法拷贝 ------
    if (
        not src_kv_caches
        or not dst_kv_caches
        or not src_block_ids
        or not dst_block_ids
        or len(src_block_ids) != len(dst_block_ids)
    ):
        return

    # ------【PD 分离】从第一个层的张量推断源/目标设备(host/device) ------
    src_device = next(iter(src_kv_caches.values())).device
    dst_device = next(iter(dst_kv_caches.values())).device

    # ------【PD 分离】生成放在各自设备上的 block 索引张量 ------
    src_indices, dst_indices = _make_src_and_dst_indices(
        src_block_ids=src_block_ids,
        dst_block_ids=dst_block_ids,
        src_device=src_device,
        dst_device=dst_device,
    )

    # ------【PD 分离】按方向选择平台拷贝原语：h2d 用 insert、d2h 用 swap_out ------
    if direction == "h2d":
        copy_fn = current_platform.insert_blocks_to_device
    else:
        copy_fn = current_platform.swap_out_blocks_to_host
    # ------【PD 分离】逐层拷贝 KV cache，src/dst 按同名 layer 一一对应 ------
    for layer_name in src_kv_caches:
        src_tensor = src_kv_caches[layer_name]
        dst_tensor = dst_kv_caches[layer_name]
        copy_fn(src_tensor, dst_tensor, src_indices, dst_indices)


def kv_postprocess_blksize_on_receive(cache, indices, block_size_ratio):
    """
    Transforms the layout of received KV cache blocks to the local block_size.
    (Only works for local blocksize > remote blocksize)

    example:
    local blocksize = 16 tokens, remote blocksize = 4 tokens
    local block[0] = remote block[0, 1, 2, 3]
    remote is |h0-b0|h1-b0|h2-b0|h3-b0|h0-b1|h1-b1|h2-b1|h3-b1|...
    local is  |h0-b0..................|h1-b0..................|...
    permute is to:
    1. view => view remote as n_blocks * remote_shape(H,remoteN,D)
    2. permute => (H, nblocks, remoteN, D)
    3. flatten => (H, localN, D)
    """
    # ------【PD 分离】按 block 索引取出要重排的块 ------
    blocks_to_update = cache.index_select(0, indices)
    # use physical order
    # ------【PD 分离】先转成物理顺序(HND)，以便按远程小块边界 reshape ------
    blocks_to_update = blocks_to_update.permute(0, 2, 1, 3)
    n_kv_heads, block_size, head_size = blocks_to_update.shape[1:]
    # ------【PD 分离】根据 block_size_ratio 反推远程小块大小与每个本地块含的远程块数 ------
    remote_block_size = block_size // block_size_ratio
    n_blocks = block_size_ratio

    # ------【PD 分离】reshape+permute+flatten 把多个远程小块重排成一个本地大块的布局 ------
    permuted_blocks = (
        blocks_to_update.reshape(-1, n_blocks, n_kv_heads, remote_block_size, head_size)
        .permute(0, 2, 1, 3, 4)
        .flatten(2, 3)
    )
    # ------【PD 分离】再 permute 回本地 HND 顺序，对齐 head 与 token 维度 ------
    permuted_blocks = permuted_blocks.permute(0, 2, 1, 3)
    # ------【PD 分离】原地回写重排后的块，完成本地 block_size 的转换 ------
    cache.index_copy_(0, indices, permuted_blocks)


def kv_postprocess_layout_on_receive(cache, indices):
    """Transforms the layout of received KV cache blocks to the local format.

    This method corrects layout mismatches from direct memory copies by
    permuting the tensor dimensions.

    4D cache:
    - **Source Layout:** `[num_blocks, n_kv_head, block_size, head_dim]`
    - **Target Layout:** `[num_blocks, block_size, n_kv_head, head_dim]`
    5D cache:
    - **Source Layout:** `[num_blocks, kv_dim, n_kv_head, block_size, head_dim]`
    - **Target Layout:** `[num_blocks, kv_dim, block_size, n_kv_head, head_dim]`

    Implementation:
    - x = blocks_to_update.reshape(src_shape) # view local kv with sender layout
    - permuted_blocks = x.permute(*inv_order) # transpose n_kv_heads, block_size
    - cache.index_copy_(0, indices, permuted_blocks) # copy permuted kv back

    """
    # ------【PD 分离】取出要重排的块，并记录本地目标 shape(首维设为 -1 自动推导) ------
    blocks_to_update = cache.index_select(0, indices)
    target_shape = list(blocks_to_update.shape)
    target_shape[0] = -1
    # ------【PD 分离】按维度数(4D/5D)确定逆置换顺序，用于把发送端布局还原成本地布局 ------
    inv_order = [0, 1, 3, 2, 4] if blocks_to_update.ndim == 5 else [0, 2, 1, 3]
    src_shape = tuple(target_shape[i] for i in inv_order)
    # ------【PD 分离】先 reshape 成发送端布局再 permute 回本地布局 ------
    blocks_to_update = cache.index_select(0, indices)
    permuted_blocks = blocks_to_update.reshape(src_shape).permute(*inv_order)
    # ------【PD 分离】原地回写重排结果，修正直接内存拷贝带来的布局差异 ------
    cache.index_copy_(0, indices, permuted_blocks)


def kv_postprocess_blksize_and_layout_on_receive(cache, indices, block_size_ratio):
    """
    Transforms the layout of received KV cache to the local block_size and HND.
    (Only works for local blocksize > remote blocksize)

    prefill is HND, smaller block_size
    decode(local) is NHD, larger block_size
    """
    # ------【PD 分离】取出要转换的块，并解析本地(NHD)的 block_size/头数/头维度 ------
    blocks_to_update = cache.index_select(0, indices)

    block_size, n_kv_heads, head_size = blocks_to_update.shape[1:]
    # ------【PD 分离】反推远程小块大小与每个本地块含的远程块数 ------
    remote_block_size = block_size // block_size_ratio
    n_blocks = block_size_ratio

    # ------【PD 分离】reshape+permute+flatten 把远程 HND 小块合并成本地大块并切到 NHD 布局 ------
    permuted_blocks = (
        blocks_to_update.reshape(-1, n_blocks, n_kv_heads, remote_block_size, head_size)
        .permute(0, 1, 3, 2, 4)
        .flatten(1, 2)
    )
    # ------【PD 分离】原地回写，同时完成 block_size 与 HND 布局的双重转换 ------
    cache.index_copy_(0, indices, permuted_blocks)


def yield_req_data(
    scheduler_output,
) -> Iterator[tuple[str, tuple[list[int], ...] | None, bool]]:
    """
    Yields:
        (req_id, new_block_id_groups, preempted)
    """
    # new requests
    # ------【PD 分离+前缀缓存】新请求逐个 yield：req_id、分配的 block 组、未抢占(False) ------
    for req_data in scheduler_output.scheduled_new_reqs:
        yield req_data.req_id, req_data.block_ids, False

    # cached requests
    cached_reqs = scheduler_output.scheduled_cached_reqs
    # ------【PD 分离+前缀缓存】缓存请求：yield 新 block 组，并用 resumed_req_ids 标记是否被抢占恢复 ------
    yield from zip(
        cached_reqs.req_ids,
        cached_reqs.new_block_ids,
        (req_id in cached_reqs.resumed_req_ids for req_id in cached_reqs.req_ids),
    )


def get_current_attn_backends(
    vllm_config: VllmConfig, layer_names: list[str] | None = None
) -> list[type[AttentionBackend]]:
    """Get all distinct attention backends for the given layers.

    Args:
        vllm_config: The current vLLM configuration.
        layer_names: Optional list of layer names to scope the lookup.
            When None, all attention layers are considered.

    Returns:
        Deduplicated list of attention backend classes.
    """
    layer_type = cast(type[Any], AttentionLayerBase)
    # ------【核心逻辑】从 vllm 配置里提取注意力层(可按 layer_names 过滤) ------
    layers = get_layers_from_vllm_config(vllm_config, layer_type, layer_names)
    if layers:
        seen: dict[str, type[AttentionBackend]] = {}
        # ------【核心逻辑】逐层取 attention backend，用完整类名去重，保证返回无重复类 ------
        for layer in layers.values():
            backend = layer.get_attn_backend()
            seen[backend.full_cls_name()] = backend
        return list(seen.values())

    # Fallback for tests, when static_forward_context is empty.
    # ------【核心逻辑】没有层时的回退路径(测试场景)，改用 selector 按模型参数推导默认 backend ------
    logger.debug(
        "No layers found in the vLLM config. Falling back to default attention backend."
    )
    from vllm.v1.attention.selector import get_attn_backend

    # ------【核心逻辑】用临时 vllm 配置上下文调用 selector，避免污染线程级全局配置 ------
    with set_current_vllm_config(vllm_config):
        return [
            get_attn_backend(
                head_size=vllm_config.model_config.get_head_size(),
                dtype=vllm_config.model_config.dtype,
                kv_cache_dtype=vllm_config.cache_config.cache_dtype,
                use_mla=vllm_config.model_config.use_mla,
            )
        ]


def get_current_attn_backend(
    vllm_config: VllmConfig, layer_names: list[str] | None = None
) -> type[AttentionBackend]:
    """Get the first attention backend for the given layers."""
    return get_current_attn_backends(vllm_config, layer_names)[0]


# ---- Per-engine transfer info ----


@dataclass(frozen=True)
class EngineTransferInfo:
    """Common per-remote-engine transfer state, computed at handshake.

    Stored per ``(engine_id, pp_rank)`` inside ``TransferTopology._engines``.
    """

    remote_tp_size: int

    remote_block_len: int
    """Block length (bytes)"""

    remote_block_size: int
    """Tokens per block."""

    remote_physical_blocks_per_logical: int
    """Physical blocks per logical block."""

    remote_pp_rank: int = 0
    """Remote producer PP rank for this engine."""

    start_layer: int = 0
    """Global index of the first layer owned by this PP rank."""

    end_layer: int = 0
    """Exclusive global index after the last layer owned by this PP rank."""


# ---- Transfer topology ----


@dataclass
class TransferTopology:
    """Single source of truth for local TP identity and per-engine remote info."""

    tp_rank: int
    tp_size: int
    block_size: int
    engine_id: EngineId
    is_mla: bool
    is_mamba: bool
    total_num_kv_heads: int
    attn_backends: list[type[AttentionBackend]]
    tensor_shape: torch.Size | None = None

    def __post_init__(self):
        # ------【TP】本地每个 TP rank 分到的物理 KV 头数，至少为 1(头数小于 TP 数时复制) ------
        self.local_physical_heads = max(1, self.total_num_kv_heads // self.tp_size)

        # ------【PD 分离】初始化远程引擎信息字典，键为 (engine_id, pp_rank) ------
        self._engines: dict[tuple[EngineId, int], EngineTransferInfo] = {}

        # Figure out whether the first dimension of the cache is K/V
        # or num_blocks.
        attn_backend = self.attn_backends[0]
        # ------【核心逻辑】用 mock 参数向 backend 查询 KV cache shape，探测首维是否为 block 数 ------
        if not self.is_mamba:
            _MOCK_BLOCK_SIZE = 16
            kv_cache_shape: tuple[int, ...] = attn_backend.get_kv_cache_shape(
                num_blocks=1,
                block_size=_MOCK_BLOCK_SIZE,
                num_kv_heads=1,
                head_size=1,
            )
            logger.debug("Test kv_cache_shape: %s", kv_cache_shape)
            # ------【核心逻辑】校验布局是 blocks-first(首维为 num_blocks=1) ------
            assert kv_cache_shape[0] == 1, (
                "KV cache layout must be blocks-first; expected mocked "
                f"num_blocks=1 in leading dim, got shape {kv_cache_shape}."
            )
            # ------【核心逻辑】非 MLA 时校验 cache 为标准 4D 布局 [blocks, heads, bs, content] ------
            if not self.is_mla:
                assert len(kv_cache_shape) == 4, (
                    "Attention KV cache layout must be standardized as "
                    "[num_blocks, num_kv_heads, block_size, content_size], "
                    f"got shape {kv_cache_shape}."
                )

        # ------【核心逻辑】tensor_shape 比 cache 多一维时判定为跨层(per-layer) block 布局 ------
        self._cross_layers_blocks = False
        if self.tensor_shape is not None:
            self._cross_layers_blocks = (
                len(self.tensor_shape) == len(kv_cache_shape) + 1
            )

        # ------【核心逻辑】跨层布局：补一层数维度，并按 backend 的 stride 顺序重排 shape ------
        if self._cross_layers_blocks:
            logger.debug("Using cross-layer KV cache")
            _MOCK_NUM_LAYERS = 80
            kv_cache_shape = (_MOCK_NUM_LAYERS,) + kv_cache_shape
            try:
                # ------【核心逻辑】查询 backend 提供的跨层 stride 顺序 ------
                kv_cache_stride_order = attn_backend.get_kv_cache_stride_order(
                    include_num_layers_dimension=self._cross_layers_blocks
                )
            except (AttributeError, NotImplementedError):
                # ------【核心逻辑】backend 不支持时回退恒等 stride，直接用 tensor_shape 的自然顺序 ------
                assert self.tensor_shape is not None
                kv_cache_stride_order = tuple(range(len(self.tensor_shape)))
            kv_cache_shape = tuple(kv_cache_shape[i] for i in kv_cache_stride_order)

    # ============================================================
    # Engine registration
    # ============================================================

    def register_remote_engine(
        self,
        remote_engine_id: EngineId,
        info: EngineTransferInfo,
    ) -> EngineTransferInfo:
        """Register a remote engine, unifying worker dicts state.

        The caller (worker) is responsible for computing the info via
        the transfer policy.  This method only stores and deduplicates.
        """
        # ------【PD 分离】禁止把本地 engine 注册成远程引擎，本地身份由 __init__ 参数决定 ------
        assert remote_engine_id != self.engine_id, (
            f"Cannot register local engine {self.engine_id} as remote. "
            f"Local identity is set via __init__ params."
        )
        # ------【PD 分离】用 (engine_id, pp_rank) 作为唯一键构造条目 ------
        engine_key = (remote_engine_id, info.remote_pp_rank)
        # ------【PD 分离】已注册则直接返回既有 info，去重并统一各 worker 字典状态 ------
        if engine_key in self._engines:
            return self._engines[engine_key]
        self._engines[engine_key] = info
        return info

    def get_engine_info(
        self, remote_engine_id: EngineId, remote_pp_rank: int = 0
    ) -> EngineTransferInfo:
        # ------【PD 分离】按 (engine_id, pp_rank) 键直接查表返回远程引擎传输信息 ------
        return self._engines[(remote_engine_id, remote_pp_rank)]

    def unregister_remote_engine(self, remote_engine_id: EngineId) -> None:
        # Remove all pp_rank entries for the remote engine.
        # ------【PD 分离+PP】删除该 engine 的所有 pp_rank 条目(遍历键首元素匹配的键) ------
        for key in [k for k in self._engines if k[0] == remote_engine_id]:
            del self._engines[key]

    # ============================================================
    # Layout properties
    # ============================================================

    @property
    def cross_layers_blocks(self) -> bool:
        # ------【核心逻辑】返回是否使用跨层(per-layer) block 布局的标志位 ------
        return self._cross_layers_blocks

    @property
    def virtually_split_kv_in_blocks(self) -> bool:
        # Whether to logically split each block into two separately-indexable
        # sub-regions. With K and V packed into the content dim, an attention
        # block transfers as a single unit — no K/V sub-split is needed. Only
        # Mamba still needs this, to index its two state regions (conv/ssm)
        # separately. Not applicable to cross-layer blocks (per-layer
        # interleaving means a simple half-split does not separate the parts).
        # ------【核心逻辑】仅 Mamba 且非跨层布局时才需要把块逻辑拆成两个可独立索引的子区域 ------
        return self.is_mamba and not self._cross_layers_blocks

    # ============================================================
    # Common methods
    # ============================================================

    def tp_ratio(self, remote_tp_size: int) -> int:
        """Calculate the tensor parallel ratio between local and remote TP.

        Positive when local_tp >= remote_tp (local workers read from the
        same remote worker in groups of size ``tp_ratio``).  Negative when
        remote_tp > local_tp (ratio is flipped).
        """
        # ------【TP】本地 TP 更大：先校验可整除，再返回正比(本地 worker 分组共享远程 worker) ------
        if self.tp_size >= remote_tp_size:
            assert self.tp_size % remote_tp_size == 0, (
                f"Local tensor parallel size {self.tp_size} is not divisible "
                f"by remote tensor parallel size {remote_tp_size}."
            )
            return self.tp_size // remote_tp_size
        # ------【TP】远程 TP 更大：校验可整除后返回负比(用负数表示比例方向反转) ------
        assert remote_tp_size % self.tp_size == 0, (
            f"Remote tensor parallel size {remote_tp_size} is not divisible "
            f"by local tensor parallel size {self.tp_size}."
        )
        return -(remote_tp_size // self.tp_size)

    def block_size_ratio(self, remote_block_size: int) -> int:
        """Calculate the block size ratio between local and remote."""
        # ------【PD 分离】校验本地 block_size 可被远程 block_size 整除，再返回两者比例 ------
        assert self.block_size % remote_block_size == 0, (
            f"Local block size {self.block_size} is not divisible "
            f"by remote block size {remote_block_size} or vice versa."
        )
        return self.block_size // remote_block_size

    def is_kv_replicated(
        self, remote_engine_id: EngineId, remote_pp_rank: int = 0
    ) -> bool:
        """Whether the KV cache is replicated across TP workers due to the
        number of TP workers being greater than the number of KV heads.
        """
        # ------【TP】远程 TP worker 数大于 KV 头数时，KV cache 会在 TP 间复制而非切分 ------
        return (
            self._engines[(remote_engine_id, remote_pp_rank)].remote_tp_size
            > self.total_num_kv_heads
        )

    def replicates_kv_cache(
        self, remote_engine_id: EngineId, remote_pp_rank: int = 0
    ) -> bool:
        # MLA is always replicated as the hidden dim can't be split.
        # ------【TP】MLA(隐维不可切)或头数不足时，远程 KV cache 按复制方式处理 ------
        return self.is_mla or self.is_kv_replicated(remote_engine_id, remote_pp_rank)

    @property
    def local_replicates_kv_cache(self) -> bool:
        """Whether the local engine's KV cache is replicated."""
        # ------【TP】本地 MLA 或 TP 数超过头数时，本地 KV cache 同样按复制方式处理 ------
        return self.is_mla or self.tp_size > self.total_num_kv_heads

    def handshake_target_ranks(self, remote_tp_size: int) -> list[int]:
        """Pre-registration: compute which remote TP ranks to handshake with.

        Pure math based on local/remote TP sizes — does not require
        the remote engine to be registered yet.
        """
        # ------【TP】本地 TP 更大时，多个本地 rank 映射到同一个远程 rank(整除分组) ------
        tp_ratio = self.tp_ratio(remote_tp_size)
        if tp_ratio > 0:
            return [self.tp_rank // tp_ratio]
        # ------【TP】远程 TP 更大时，本地 rank 需要与多个远程 rank 握手(逐个展开) ------
        abs_ratio = -tp_ratio
        return [self.tp_rank * abs_ratio + i for i in range(abs_ratio)]

    def target_remote_ranks(
        self, remote_engine_id: EngineId, remote_pp_rank: int = 0
    ) -> list[int]:
        """Get the remote TP rank(s) that the current local TP rank will
        read from.  When remote tp_size > local tp_size, reads from
        multiple remote ranks.
        """
        info = self._engines[(remote_engine_id, remote_pp_rank)]
        # ------【TP】本地 TP 更大时，本地 rank 读取对应分组映射到的唯一远程 rank ------
        tp_ratio = self.tp_ratio(info.remote_tp_size)
        if tp_ratio > 0:
            return [self.tp_rank // tp_ratio]
        # remote TP > local TP: read from |tp_ratio| remote workers
        # ------【TP】远程 TP 更大时，本地 rank 从 |tp_ratio| 个远程 worker 读取，返回全部目标 rank ------
        abs_ratio = -tp_ratio
        return [self.tp_rank * abs_ratio + i for i in range(abs_ratio)]

    def describe(self, remote_engine_id: EngineId, remote_pp_rank: int = 0) -> str:
        """One-line summary of transfer config for logging."""
        info = self._engines[(remote_engine_id, remote_pp_rank)]
        # ------【核心逻辑】拼出一行摘要字符串，便于日志打印传输拓扑关键参数 ------
        return (
            f"TransferTopology("
            f"tp_ratio={self.tp_ratio(info.remote_tp_size)}, "
            f"num_kv_heads={self.total_num_kv_heads if not self.is_mla else 1}, "
            f"local_tp={self.tp_size}, "
            f"remote_tp={info.remote_tp_size}, "
            f"remote_pp={remote_pp_rank}, "
            f"local_rank={self.tp_rank}, "
            f"remote_block_len={info.remote_block_len})"
        )
