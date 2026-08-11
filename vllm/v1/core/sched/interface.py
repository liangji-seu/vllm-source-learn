# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import enum
from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import TYPE_CHECKING

from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.config.kv_events import KVEventsConfig
    from vllm.distributed.ec_transfer.ec_connector.base import ECConnectorBase
    from vllm.distributed.kv_transfer.kv_connector.v1 import KVConnectorBase_V1
    from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
    from vllm.v1.engine import EngineCoreOutputs
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.metrics.stats import SchedulerStats
    from vllm.v1.outputs import DraftTokenIds, ModelRunnerOutput
    from vllm.v1.request import Request, RequestStatus
    from vllm.v1.structured_output import StructuredOutputManager


class PauseState(enum.IntEnum):
    """Scheduler pause state.

    - UNPAUSED: Normal operation
    - PAUSE_NEW: No new requests are scheduled, requests already in
                 running state are scheduled.
    - PAUSE_ALL: No requests are scheduled
    """

    UNPAUSED = 0
    PAUSED_NEW = 1
    PAUSED_ALL = 2


class SchedulerInterface(ABC):
    """
    === 类说明 ===
        继承: ABC
        职责: 调度器抽象接口。定义引擎后端调度器的完整契约，
              Scheduler 类实现此接口，SchedulerClient 通过此接口操作调度器。

    === 抽象方法 (15个) ===
        __init__()               — 构造：注入配置、KV cache、结构化输出管理器等
        schedule()               — 核心：选取请求 + 分配 KV cache，每次前向调用一次
        update_from_output()     — 根据模型输出更新请求状态，返回 EngineCoreOutputs
        add_request()            — 新请求加入调度器内部队列
        finish_requests()        — 中止/停止请求（客户端中止 or 检测到 stop string）
        get_num_unfinished_requests() — 返回未完成请求数量
        has_finished_requests()  — 是否有已完成待清理的请求（非 has_unfinished 的反义）
        update_draft_token_ids() — 更新请求的草稿 token（投机解码）
        update_draft_token_ids_in_output() — 同上 + 同步更新 SchedulerOutput
        get_grammar_bitmask()    — 获取结构化输出语法掩码
        get_request_counts()     — 返回 (运行中, 等待中) 数量
        reset_prefix_cache()     — 重置 KV 前缀缓存（模型热更新时）
        reset_encoder_cache()    — 重置编码器缓存（模型热更新时使视觉嵌入失效）
        make_stats()             — 生成 SchedulerStats 用于日志
        shutdown()               — 关闭调度器

    === 具体方法 (7个) ===
        has_unfinished_requests()  — 基于 get_num_unfinished_requests() > 0
        has_requests()             — has_unfinished OR has_finished
        pause_state / set_pause_state — 暂停/恢复调度
        get_kv_cache_usage()       — KV cache 使用率 (0.0-1.0)，默认返回 0.0
        get_kv_connector()         — 获取 KV connector（P/D 分离），默认 None
        get_ec_connector()         — 获取 EC connector（弹性扩容），默认 None
        get_kv_event_publisher_config() — 获取 KV 事件发布配置，默认 None
    """

    @abstractmethod
    def __init__(
        self,
        vllm_config: "VllmConfig",
        kv_cache_config: "KVCacheConfig",
        structured_output_manager: "StructuredOutputManager",
        block_size: int,
        hash_block_size: int,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
        include_finished_set: bool = False,
        log_stats: bool = False,
    ) -> None:
        raise NotImplementedError




    '''
    每个调度步骤对应模型的单次前向传递。因此，这个方法会被引擎里的忙循环反复调用
    '''
    @abstractmethod
    def schedule(self, throttle_prefills: bool = False) -> "SchedulerOutput":
        """Schedule the requests to process in this scheduling step.

        The scheduling decision is made at the iteration level. Each scheduling
        step corresponds to a single forward pass of the model. Therefore, this
        method is called repeatedly by a busy loop in the engine.

        Essentially, the scheduler produces a dictionary of {req_id: num_tokens}
        that specifies how many tokens to process for each request in this
        scheduling step. For example, num_tokens can be as large as the number
        of prompt tokens for new requests, or it can be 1 for the requests that
        are auto-regressively generating new tokens one by one. Otherwise, it
        can be somewhere in between in case of chunked prefills, prefix caching,
        speculative decoding, etc.

        Additionally, the scheduler also returns useful data about each request
        or the batch as a whole. The model runner will use this information in
        preparing inputs to the model.

        Args:
            throttle_prefills: DP prefill balancing. When True (set by the DP
                engine core on non-cadence-aligned steps), new prefill compute is
                deferred to a later step so prefills stay aligned across DP ranks;
                automatically overridden when the rank is saturated.

        Returns:
            A SchedulerOutput object containing information about the scheduled
            requests.
        """
        raise NotImplementedError

    @abstractmethod
    def get_grammar_bitmask(
        self, scheduler_output: "SchedulerOutput"
    ) -> "GrammarOutput | None":
        raise NotImplementedError




    '''
        根据模型运行器的输出更新调度器状态。

        此方法在模型运行器处理完预定的请求后调用。
        模型运行器的输出包括生成的令牌ID、下一步的草稿令牌ID等。调度器使用这些信息来更新其状态，检查已完成的请求，并返回每个请求的输出。
        返回：一个从客户端索引到EngineCoreOutputs对象的字典，其中包含来自该客户端的每个请求的输出。
    
    '''
    @abstractmethod
    def update_from_output(
        self,
        scheduler_output: "SchedulerOutput",
        model_runner_output: "ModelRunnerOutput",
    ) -> dict[int, "EngineCoreOutputs"]:
        """Update the scheduler state based on the model runner output.

        This method is called after the model runner has processed the scheduled
        requests. The model runner output includes generated token ids, draft
        token ids for next step, etc. The scheduler uses this information to
        update its states, checks the finished requests, and returns the output
        for each request.

        Returns:
            A dict of client index to EngineCoreOutputs object containing the
            outputs for each request originating from that client.
        """
        raise NotImplementedError





    '''
    使用新生成的草稿令牌ID更新请求，如果需要，应用结构化输出语法验证。

    参数：draft_token_ids：每个请求的输入草稿令牌ID。
    '''
    @abstractmethod
    def update_draft_token_ids(self, draft_token_ids: "DraftTokenIds") -> None:
        """Update requests with newly generated draft token ids, applying
        structured output grammar validation if needed.

        Args:
            draft_token_ids: The input draft token ids for each request.
        """
        raise NotImplementedError





    '''
    使用新生成的草稿令牌ID更新调度程序输出，如有需要，应用结构化输出语法验证。

    参数：draft_token_ids：每个请求的输入草稿令牌ID。
    调度器输出：使用相应的草稿令牌ID更新给定的调度器输出。
    '''
    @abstractmethod
    def update_draft_token_ids_in_output(
        self, draft_token_ids: "DraftTokenIds", scheduler_output: "SchedulerOutput"
    ) -> None:
        """Update scheduler output with newly generated draft token ids, applying
        structured output grammar validation if needed.

        Args:
            draft_token_ids: The input draft token ids for each request.
            scheduler_output: Update the given scheduler_output
                with the corresponding draft token ids.
        """
        raise NotImplementedError




    '''
    将新请求添加到调度程序的内部队列中。

    参数：
    请求：正在添加的新请求。
    '''
    @abstractmethod
    def add_request(self, request: "Request") -> None:
        """Add a new request to the scheduler's internal queue.

        Args:
            request: The new request being added.
        """
        raise NotImplementedError



    '''
    处理调度器内部队列中的请求。如果请求不在队列中，则此方法对该请求不执行任何操作。

    此方法在两种情况下被调用：1. 当客户端中止请求时。
    2. 前端进程在对其生成的标记进行去标记化处理后，检测到请求中的停止字符串。

    参数：request_ids：单个或多个请求ID，或为None表示完成所有请求。
    finished_status：给定请求的完成状态。

    返回：已中止的请求列表。不包括任何已完成的请求。
    '''
    @abstractmethod
    def finish_requests(
        self,
        request_ids: str | Iterable[str] | None,
        finished_status: "RequestStatus",
    ) -> "list[Request]":
        """Finish the requests in the scheduler's internal queue. If the request
        is not in the queue, this method will do nothing for that request.

        This method is called in two cases:
        1. When the request is aborted by the client.
        2. When the frontend process detects a stop string of the request after
           de-tokenizing its generated tokens.

        Args:
            request_ids: A single or a list of request IDs, or None to finish all.
            finished_status: The finished status of the given requests.

        Returns:
            List of requests that were aborted. Will not include any that were
            already finished.
        """
        raise NotImplementedError



    '''
    调度器内部队列中未完成的请求数量。
    '''
    @abstractmethod
    def get_num_unfinished_requests(self) -> int:
        """Number of unfinished requests in the scheduler's internal queue."""
        raise NotImplementedError



    '''
    如果调度程序的内部队列中有未完成的请求，则返回True。
    '''
    def has_unfinished_requests(self) -> bool:
        """Returns True if there are unfinished requests in the scheduler's
        internal queue."""
        return self.get_num_unfinished_requests() > 0

    '''
    如果存在需要清理的已完成请求，则返回True。
    注意：这与 `not self.has_unfinished_requests()` 不同。

    调度器维护一个内部列表，记录上一步中已完成的请求。
    此列表在下一次调用 schedule() 时返回，发送给模型运行器，
    以便在下一步中清除这些已完成请求的缓存状态。

    此方法检查这个已完成请求的内部列表是否非空。
    此信息对 DP attention 有用。
    '''
    @abstractmethod
    def has_finished_requests(self) -> bool:
        """Returns True if there are finished requests that need to be cleared.
        NOTE: This is different from `not self.has_unfinished_requests()`.

        The scheduler maintains an internal list of the requests finished in the
        previous step. This list is returned from the next call to schedule(),
        to be sent to the model runner in the next step to clear cached states
        for these finished requests.

        This method checks if this internal list of finished requests is
        non-empty. This information is useful for DP attention.
        """
        raise NotImplementedError

    '''
    如果存在未完成的请求，或已完成但尚未在 SchedulerOutputs 中返回的请求，则返回True。
    '''
    def has_requests(self) -> bool:
        """Returns True if there are unfinished requests, or finished requests
        not yet returned in SchedulerOutputs."""
        return self.has_unfinished_requests() or self.has_finished_requests()

    '''
    调度器当前的暂停状态。
    '''
    @property
    @abstractmethod
    def pause_state(self) -> PauseState:
        """Current pause state of the scheduler."""
        raise NotImplementedError

    @abstractmethod
    def set_pause_state(self, pause_state: PauseState) -> None:
        raise NotImplementedError

    '''
    重置 KV cache 的前缀缓存。

    当模型权重被热更新时，这尤其必要。

    参数：
        reset_running_requests: 如果为True，所有正在运行的请求将被抢占并移回等待队列。
            否则，此方法仅在没有正在运行的请求占用 KV cache 时才重置 KV 前缀缓存。
    '''
    @abstractmethod
    def reset_prefix_cache(
        self, reset_running_requests: bool = False, reset_connector: bool = False
    ) -> bool:
        """Reset the prefix cache for KV cache.

        This is particularly required when the model weights are live-updated.

        Args:
            reset_running_requests: If True, all the running requests will be
                preempted and moved to the waiting queue. Otherwise, this method
                will only reset the KV prefix cache when there is no running request
                taking KV cache.
        """
        raise NotImplementedError

    '''
    重置编码器缓存，使所有缓存的编码器输出失效。

    当模型权重更新时应调用此方法，以确保旧的视觉嵌入不会被复用。
    '''
    @abstractmethod
    def reset_encoder_cache(self) -> None:
        """Reset the encoder cache to invalidate all cached encoder outputs.

        This should be called when model weights are updated to ensure
        stale vision embeddings are not reused.
        """
        raise NotImplementedError

    '''
    返回 (正在运行的请求数, 等待中的请求数)。
    '''
    @abstractmethod
    def get_request_counts(self) -> tuple[int, int]:
        """Returns (num_running_reqs, num_waiting_reqs)."""
        raise NotImplementedError

    '''
    返回 KV cache 当前使用比例 (0.0-1.0)。
    '''
    def get_kv_cache_usage(self) -> float:
        """Returns the fraction of the KV cache currently in use (0.0-1.0)."""
        return 0.0







    '''
    创建 SchedulerStats 对象用于日志记录。

    每个调度步骤都会创建一个 SchedulerStats 对象。
    '''
    @abstractmethod
    def make_stats(self) -> "SchedulerStats | None":
        """Make a SchedulerStats object for logging.

        The SchedulerStats object is created for every scheduling step.
        """
        raise NotImplementedError






    '''
    关闭调度器。
    '''
    @abstractmethod
    def shutdown(self) -> None:
        """Shutdown the scheduler."""
        raise NotImplementedError

    def get_kv_connector(self) -> "KVConnectorBase_V1 | None":
        return None

    def get_ec_connector(self) -> "ECConnectorBase | None":
        return None

    def get_kv_event_publisher_config(self) -> "KVEventsConfig | None":
        return None
