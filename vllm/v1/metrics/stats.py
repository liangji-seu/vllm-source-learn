# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import vllm.envs as envs
from vllm.compilation.cuda_graph import CUDAGraphStat
from vllm.v1.metrics.perf import PerfStats
from vllm.v1.spec_decode.metrics import SpecDecodingStats

if TYPE_CHECKING:
    from vllm.v1.engine import EngineCoreEvent, EngineCoreOutput, FinishReason


@dataclass
class BaseCacheStats:
    # ------【前缀缓存】缓存命中统计基类：聚合命中/查询计数 ------
    """Stores cache hit statistics."""

    # ------【前缀缓存】reset：本轮是否重置缓存（reset_prefix_cache 调用） ------
    reset: bool = False
    """Whether the cache was reset."""

    # ------【前缀缓存】requests：本轮统计覆盖的请求数 ------
    requests: int = 0
    """The number of requests in this update."""

    # ------【前缀缓存】queries：这些请求产生的查询数（token/数据项） ------
    queries: int = 0
    """The number of queries in these requests."""

    # ------【前缀缓存】hits：这些查询中命中的数量 ------
    hits: int = 0
    """The number of hits in these requests."""


class CachingMetrics:
    # ------【前缀缓存】缓存命中率滑动窗口聚合器 ------
    """Metrics for caching with a hit rate of the most recent N requests.
    Args:
        interval: The number of the most recent requests to aggregate.
            Defaults to 1000.
    """

    def __init__(self, max_recent_requests: int = 1000) -> None:
        super().__init__()

        # ------【前缀缓存】max_recent_requests：滑动窗口请求数上限 ------
        self.max_recent_requests = max_recent_requests
        # The current aggregated values.
        # ------【前缀缓存】aggregated_requests：当前窗口累计请求数 ------
        self.aggregated_requests = 0
        # ------【前缀缓存】aggregated_query_total：当前窗口累计查询总数 ------
        self.aggregated_query_total = 0
        # ------【前缀缓存】aggregated_query_hit：当前窗口累计命中总数 ------
        self.aggregated_query_hit = 0

        # A deque of (requests, queries, hits) for the most recent requests.
        # ------【前缀缓存】query_queue：保存每批命中计数的双端队列 ------
        self.query_queue = deque[tuple[int, int, int]]()

    def observe(self, stats: BaseCacheStats):
        # ------【前缀缓存】observe：并入新批次命中统计并淘汰最旧批次 ------
        """Observe the prefix caching for a set of requests.

        This function is called with information gathered when new requests
        are being scheduled and are looking for computed blocks.

        When there are more than `max_recent_requests` requests, the oldest set
        of requests are removed from the metrics.

        Args:
            stats: The prefix cache stats.
        """
        # reset_prefix_cache was invoked before the current update.
        # Reset the metrics before aggregating the current stats.
        if stats.reset:
            self.reset()

        # DO NOT appending empty stats to avoid helpful info get kicked out
        # due to sliding window.
        if stats.requests == 0:
            return

        # Update the metrics.
        self.query_queue.append((stats.requests, stats.queries, stats.hits))
        self.aggregated_requests += stats.requests
        self.aggregated_query_total += stats.queries
        self.aggregated_query_hit += stats.hits

        # Remove the oldest stats until number of requests does not exceed
        # the limit.
        # NOTE: We preserve the latest added stats regardless.
        while (
            len(self.query_queue) > 1
            and self.aggregated_requests > self.max_recent_requests
        ):
            old_requests, old_queries, old_hits = self.query_queue.popleft()
            self.aggregated_requests -= old_requests
            self.aggregated_query_total -= old_queries
            self.aggregated_query_hit -= old_hits

    def reset(self):
        # ------【前缀缓存】reset：清空累计计数与滑动窗口 ------
        """Reset the metrics."""
        self.aggregated_requests = 0
        self.aggregated_query_total = 0
        self.aggregated_query_hit = 0
        self.query_queue.clear()

    @property
    def empty(self) -> bool:
        # ------【前缀缓存】empty：是否尚未观察到任何请求 ------
        """Return true if no requests have been observed."""
        return self.aggregated_requests == 0

    @property
    def hit_rate(self) -> float:
        # ------【前缀缓存】hit_rate：计算最近 N 个请求的命中率 ------
        """Calculate the hit rate for the past N requests."""
        if self.aggregated_query_total == 0:
            return 0.0
        return self.aggregated_query_hit / self.aggregated_query_total


@dataclass
class PrefixCacheStats(BaseCacheStats):
    # ------【前缀缓存】前缀缓存命中统计：区分新请求与被抢占请求 ------
    """
    Stores prefix cache hit statistics.
    - `reset`: Whether `reset_prefix_cache` was invoked.
    - `queries`: Refers to the number of tokens that were queried.
    """

    # ------【前缀缓存】preempted_requests：本轮被抢占后重新调度的请求数 ------
    preempted_requests: int = 0
    """The number of previously preempted requests in this update."""

    # ------【前缀缓存】preempted_queries：被抢占请求的查询 token 数 ------
    preempted_queries: int = 0
    """The `queries` number for preempted requests."""

    # ------【前缀缓存】preempted_hits：被抢占请求的命中 token 数 ------
    preempted_hits: int = 0
    """The `hits` number for preempted requests."""

    def record(self, num_tokens: int, num_hits: int, preempted: bool) -> None:
        # ------【前缀缓存】record：按 preempted 标记聚合到对应计数器 ------
        """Aggregate request information into the stats."""
        if preempted:
            # Previously preempted request
            self.preempted_requests += 1
            self.preempted_queries += num_tokens
            self.preempted_hits += num_hits
        else:
            # New request
            self.requests += 1
            self.queries += num_tokens
            self.hits += num_hits


@dataclass
class MultiModalCacheStats(BaseCacheStats):
    # ------【前缀缓存】多模态缓存命中统计：数据项查询/命中 ------
    """
    Stores multi-modal cache hit statistics.
    - `reset`: Whether `reset_mm_cache` was invoked.
    - `queries`: Refers to the number of multi-modal data items
      that were queried.
    """

    def record(self, num_queries: int, num_hits: int) -> None:
        # ------【前缀缓存】record：把一次查询聚合到请求/查询/命中计数 ------
        """Aggregate request information into the stats."""
        self.requests += 1
        self.queries += num_queries
        self.hits += num_hits


@dataclass
class KVCacheEvictionEvent:
    # ------【前缀缓存】KV 块逐出样本：记录块生存/空闲/复用间隔 ------
    """Single KV cache block eviction sample."""

    # ------【前缀缓存】lifetime_seconds：块从分配到被逐出的存活时长 ------
    lifetime_seconds: float
    # ------【前缀缓存】idle_seconds：块逐出前最后一次访问后的空闲时长 ------
    idle_seconds: float
    # ------【前缀缓存】reuse_gaps_seconds：块各次复用之间的时间间隔序列 ------
    reuse_gaps_seconds: tuple[float, ...]


@dataclass
class SchedulerIterationDetails:
    # ------【核心逻辑】单次调度迭代明细：记录本轮各阶段负载与耗时 ------
    """Scheduler-side details for one engine iteration."""

    # ------【核心逻辑】iteration_index：本轮迭代序号 ------
    iteration_index: int
    # ------【chunked prefill】num_ctx_requests：本轮 prefill 阶段处理的请求数 ------
    num_ctx_requests: int
    # ------【chunked prefill】num_ctx_tokens：本轮 prefill 阶段处理的 token 数 ------
    num_ctx_tokens: int
    # ------【核心逻辑】num_generation_requests：本轮 decode 处理的请求数 ------
    num_generation_requests: int
    # ------【核心逻辑】num_generation_tokens：本轮 decode 阶段生成的 token 数 ------
    num_generation_tokens: int
    # ------【核心逻辑】elapsed_ms：本轮调度耗时（毫秒） ------
    elapsed_ms: float
    # ------【核心逻辑】num_encoder_inputs：本轮编码器输入数量（多模态） ------
    num_encoder_inputs: int = 0
    # ------【核心逻辑】num_encoder_output_tokens：本轮编码器输出 token 数 ------
    num_encoder_output_tokens: int = 0
    # ------【核心逻辑】is_dummy：是否为空转/占位迭代 ------
    is_dummy: bool = False


@dataclass
class SchedulerStats:
    # ------【核心逻辑】调度器统计汇总 DTO：聚合运行时指标 ------
    """Stats associated with the scheduler."""

    # ------【核心逻辑】num_running_reqs：当前正在运行的请求数 ------
    num_running_reqs: int = 0

    # ------【核心逻辑】num_waiting_reqs：等待队列长度 ------
    num_waiting_reqs: int = 0  # length of the "waiting" request queue
    # ------【核心逻辑】num_skipped_waiting_reqs：被跳过等待队列的长度 ------
    num_skipped_waiting_reqs: int = 0  # length of the "skipped waiting" queue

    # These are used for internal DP load-balancing.
    # ------【DP】step_counter：DP 负载均衡的步计数，用于轮转调度 ------
    step_counter: int = 0
    # ------【DP】current_wave：当前 DP 调度轮次 ------
    current_wave: int = 0

    # ------【显存 profiling】kv_cache_usage：KV 缓存使用率（0~1） ------
    kv_cache_usage: float = 0.0
    # ------【核心逻辑】iteration_details：本轮调度迭代明细 ------
    iteration_details: SchedulerIterationDetails | None = None

    # ------【前缀缓存】prefix_cache_stats：前缀缓存命中统计 ------
    prefix_cache_stats: PrefixCacheStats = field(default_factory=PrefixCacheStats)
    # ------【前缀缓存】connector_prefix_cache_stats：KV connector 侧前缀缓存 ------
    connector_prefix_cache_stats: PrefixCacheStats | None = None

    # ------【前缀缓存】kv_cache_eviction_events：KV 块逐出事件列表 ------
    kv_cache_eviction_events: list[KVCacheEvictionEvent] = field(default_factory=list)

    # ------【投机解码】spec_decoding_stats：投机解码统计 ------
    spec_decoding_stats: SpecDecodingStats | None = None
    # ------【核心逻辑】kv_connector_stats：KV connector 自定义统计 ------
    kv_connector_stats: dict[str, Any] | None = None

    # ------【LoRA】waiting_lora_adapters：各 LoRA 等待中的请求数 ------
    waiting_lora_adapters: dict[str, int] = field(default_factory=dict)
    # ------【LoRA】running_lora_adapters：各 LoRA 运行中的请求数 ------
    running_lora_adapters: dict[str, int] = field(default_factory=dict)

    # ------【CUDA Graph】cudagraph_stats：CUDA Graph 捕获/回放统计 ------
    cudagraph_stats: CUDAGraphStat | None = None

    # ------【核心逻辑】perf_stats：性能剖析统计 ------
    perf_stats: PerfStats | None = None


@dataclass
class RequestStateStats:
    # ------【核心逻辑】单请求运行状态：跨多轮累积时间戳/计数 ------
    """Stats that need to be tracked across delta updates."""

    # ------【核心逻辑】num_generation_tokens：该请求累计生成的 token 数 ------
    num_generation_tokens: int = 0

    # This is an engine frontend timestamp (wall-clock)
    # ------【核心逻辑】arrival_time：请求到达前端的时间戳（挂钟时间） ------
    arrival_time: float = 0.0

    # These are engine core timestamps (monotonic)
    # ------【核心逻辑】queued_ts：入队时间戳（单调时钟） ------
    queued_ts: float = 0.0
    # ------【核心逻辑】scheduled_ts：首次被调度时间戳（单调时钟） ------
    scheduled_ts: float = 0.0
    # ------【核心逻辑】first_token_ts：首 token 生成时间戳（单调时钟） ------
    first_token_ts: float = 0.0
    # ------【核心逻辑】last_token_ts：末 token 生成时间戳（单调时钟） ------
    last_token_ts: float = 0.0

    # first token latency
    # ------【核心逻辑】first_token_latency：首 token 延迟（TTFT） ------
    first_token_latency: float = 0.0

    # Track if this request is corrupted (NaNs in logits)
    # ------【核心逻辑】is_corrupted：该请求 logits 是否含 NaN ------
    is_corrupted: bool = False


@dataclass
class FinishedRequestStats:
    # ------【核心逻辑】已完成请求统计 DTO：各阶段延迟与完成原因 ------
    """Stats associated with a finished request."""

    # ------【核心逻辑】finish_reason：请求结束原因（如 stop/length/abort） ------
    finish_reason: "FinishReason"
    # ------【核心逻辑】request_id：请求 ID ------
    request_id: str | None = None
    # ------【核心逻辑】e2e_latency：端到端延迟（到达→完成） ------
    e2e_latency: float = 0.0
    # ------【核心逻辑】num_prompt_tokens：prompt token 数 ------
    num_prompt_tokens: int = 0
    # ------【核心逻辑】num_generation_tokens：生成的 token 数 ------
    num_generation_tokens: int = 0
    # ------【核心逻辑】max_tokens_param：请求的 max_tokens 参数值 ------
    max_tokens_param: int | None = None
    # ------【核心逻辑】queued_time：排队等待时长 ------
    queued_time: float = 0.0
    # ------【chunked prefill】prefill_time：prefill 阶段耗时 ------
    prefill_time: float = 0.0
    # ------【核心逻辑】inference_time：推理总耗时（调度→末 token） ------
    inference_time: float = 0.0
    # ------【核心逻辑】decode_time：decode 阶段耗时 ------
    decode_time: float = 0.0
    # ------【核心逻辑】mean_time_per_output_token：每输出 token 平均耗时 ------
    mean_time_per_output_token: float = 0.0
    # ------【核心逻辑】is_corrupted：该请求是否损坏（logits 含 NaN） ------
    is_corrupted: bool = False
    # ------【前缀缓存】num_cached_tokens：命中前缀缓存的 token 数 ------
    num_cached_tokens: int = 0


@dataclass
class PrefillStats:
    # ------【chunked prefill/前缀缓存】prefill 计算分解：计算/缓存/传输 ------
    """Breakdown of a scheduled prefill computation.

    Fields:
        num_prompt_tokens: Total number of tokens to be prefilled.
        num_computed_tokens: Tokens to be prefilled locally (actual compute work).
        num_cached_tokens: Tokens to be prefilled without actual compute work.
        num_local_cached_tokens: Tokens to be prefilled from local prefix cache.
        num_external_cached_tokens: Tokens to be prefilled from external KV transfer.
        num_cache_creation_tokens: Tokens computed and written to the prefix cache.
    """

    # ------【chunked prefill】num_prompt_tokens：本轮 prefill 的 token 总数 ------
    num_prompt_tokens: int = 0
    # ------【chunked prefill】num_computed_tokens：本地实际计算的 token 数 ------
    num_computed_tokens: int = 0
    # ------【前缀缓存】num_cached_tokens：无需实际计算的缓存 token 数 ------
    num_cached_tokens: int = 0
    # ------【前缀缓存】num_local_cached_tokens：本地前缀缓存命中 token 数 ------
    num_local_cached_tokens: int = 0
    # ------【权重传输/KV 传输】num_external_cached_tokens：外部 KV 命中数 ------
    num_external_cached_tokens: int = 0
    # ------【前缀缓存】num_cache_creation_tokens：写入前缀缓存的 token 数 ------
    num_cache_creation_tokens: int = 0

    def set(
        self,
        num_prompt_tokens: int,
        num_local_cached_tokens: int,
        num_external_cached_tokens: int,
    ):
        # ------【chunked prefill/前缀缓存】set：由缓存 token 推算各分量 ------
        num_cached_tokens = num_local_cached_tokens + num_external_cached_tokens
        assert num_cached_tokens <= num_prompt_tokens

        self.num_prompt_tokens = num_prompt_tokens
        self.num_computed_tokens = num_prompt_tokens - num_cached_tokens
        self.num_cached_tokens = num_cached_tokens
        self.num_local_cached_tokens = num_local_cached_tokens
        self.num_external_cached_tokens = num_external_cached_tokens

    def finalize(self, num_cached_tokens: int) -> None:
        # ------【前缀缓存】finalize：结算实际写入前缀缓存的 token 数 ------
        assert num_cached_tokens >= 0
        self.num_cache_creation_tokens = max(
            0, min(num_cached_tokens, self.num_prompt_tokens) - self.num_cached_tokens
        )


@dataclass
class PromptTokenStats:
    # ------【前缀缓存】prompt token 来源分解：本地计算/缓存/外部传输 ------
    """Breakdown of prompt tokens by source.

    Fields:
        computed: Tokens prefilled locally (actual compute work).
        local_cache_hit: Tokens from local prefix cache.
        external_kv_transfer: Tokens from external KV transfer.
        cached_tokens: Tokens skipped during prefill (from scheduler).
        total: Total prompt tokens.

    Invariants:
        computed + local_cache_hit + external_kv_transfer = total
        local_cache_hit + external_kv_transfer = cached_tokens
    """

    ALL_SOURCES: tuple[str, ...] = (
        "local_compute",
        "local_cache_hit",
        "external_kv_transfer",
    )

    # ------【chunked prefill】computed：本地实际计算的 token 数 ------
    computed: int = 0
    # ------【前缀缓存】local_cache_hit：本地前缀缓存命中 token 数 ------
    local_cache_hit: int = 0
    # ------【权重传输/KV 传输】external_kv_transfer：外部 KV 传输命中数 ------
    external_kv_transfer: int = 0
    # ------【前缀缓存】cached_tokens：prefill 被跳过的缓存 token 数 ------
    cached_tokens: int = 0
    # ------【chunked prefill】total：prompt token 总数 ------
    total: int = 0

    def update_from_output(self, prefill_stats: PrefillStats) -> None:
        # ------【前缀缓存】update_from_output：按输出累加各来源计数 ------
        """Update stats from a prefill output."""
        self.computed += prefill_stats.num_computed_tokens
        self.cached_tokens += prefill_stats.num_cached_tokens
        self.total += prefill_stats.num_prompt_tokens

        self.local_cache_hit += prefill_stats.num_local_cached_tokens
        self.external_kv_transfer += prefill_stats.num_external_cached_tokens

    def get_by_source(self, source: str) -> int:
        # ------【前缀缓存】get_by_source：按来源标签查 token 计数 ------
        """Get token count by source label."""
        source_map = {
            "local_compute": self.computed,
            "local_cache_hit": self.local_cache_hit,
            "external_kv_transfer": self.external_kv_transfer,
        }
        if source not in source_map:
            raise ValueError(f"Unknown source: {source}")
        return source_map[source]


class IterationStats:
    # ------【核心逻辑】单轮输出统计容器：聚合 token/延迟/完成请求 ------
    """Stats associated with a single set of EngineCoreOutputs."""

    def __init__(self):
        # ------【核心逻辑】iteration_timestamp：本轮迭代的墙钟时间戳 ------
        self.iteration_timestamp = time.time()
        # ------【核心逻辑】num_generation_tokens：本轮生成的 token 总数 ------
        self.num_generation_tokens = 0
        # ------【前缀缓存】prompt_token_stats：按来源拆分的 prompt token 统计 ------
        self.prompt_token_stats = PromptTokenStats()
        # ------【核心逻辑】num_preempted_reqs：本轮被抢占的请求数 ------
        self.num_preempted_reqs = 0
        # ------【核心逻辑】finished_requests：本轮完成请求的统计列表 ------
        self.finished_requests: list[FinishedRequestStats] = []
        # ------【核心逻辑】max_num_generation_tokens_iter：每轮最大生成数 ------
        self.max_num_generation_tokens_iter: list[int] = []
        # ------【核心逻辑】n_params_iter：每轮请求参数量序列 ------
        self.n_params_iter: list[int] = []
        # ------【核心逻辑】time_to_first_tokens_iter：每轮 TTFT 序列 ------
        self.time_to_first_tokens_iter: list[float] = []
        # ------【核心逻辑】inter_token_latencies_iter：每轮 ITL 序列 ------
        self.inter_token_latencies_iter: list[float] = []
        # ------【核心逻辑】num_corrupted_reqs：本轮损坏请求数 ------
        self.num_corrupted_reqs: int = 0

    def __repr__(self) -> str:
        # ------【核心逻辑】__repr__：打印所有字段便于调试 ------
        field_to_value_str = ", ".join(f"{k}={v}" for k, v in vars(self).items())
        return f"{self.__class__.__name__}({field_to_value_str})"

    @property
    def num_prompt_tokens(self) -> int:
        # ------【核心逻辑】num_prompt_tokens：prompt token 总数（兼容属性） ------
        """Total prompt tokens (for backward compatibility)."""
        return self.prompt_token_stats.total

    def _time_since(self, start: float) -> float:
        # ------【核心逻辑】_time_since：计算相对本轮迭代时间戳的间隔 ------
        """Calculate an interval relative to this iteration's timestamp."""
        return self.iteration_timestamp - start

    def update_from_output(
        self,
        output: "EngineCoreOutput",
        engine_core_timestamp: float,
        is_prefilling: bool,
        req_stats: RequestStateStats,
        lora_states: "LoRARequestStates",
        lora_name: str | None,
    ):
        # ------【核心逻辑】update_from_output：按单个输出更新本轮统计 ------
        num_new_generation_tokens = len(output.new_token_ids)

        self.num_generation_tokens += num_new_generation_tokens
        if is_prefilling:
            if output.prefill_stats is not None:
                self.prompt_token_stats.update_from_output(output.prefill_stats)

            first_token_latency = self._time_since(req_stats.arrival_time)
            self.time_to_first_tokens_iter.append(first_token_latency)
            req_stats.first_token_latency = first_token_latency

        req_stats.num_generation_tokens += num_new_generation_tokens

        # Track if this request is corrupted (only check once per request)
        # Early exit if already marked as corrupted to avoid redundant checks
        if (
            envs.VLLM_COMPUTE_NANS_IN_LOGITS
            and not req_stats.is_corrupted
            and output.num_nans_in_logits > 0
        ):
            req_stats.is_corrupted = True

        # Process request-level engine core events
        if output.events is not None:
            self.update_from_events(
                output.request_id,
                output.events,
                is_prefilling,
                req_stats,
                lora_states,
                lora_name,
            )

        # Process the batch-level "new tokens" engine core event
        if is_prefilling:
            req_stats.first_token_ts = engine_core_timestamp
        else:
            itl = engine_core_timestamp - req_stats.last_token_ts
            self.inter_token_latencies_iter.append(itl)

        req_stats.last_token_ts = engine_core_timestamp

    def update_from_events(
        self,
        req_id: str,
        events: list["EngineCoreEvent"],
        is_prefilling: bool,
        req_stats: RequestStateStats,
        lora_states: "LoRARequestStates",
        lora_name: str | None,
    ):
        # ------【核心逻辑】update_from_events：按事件更新排队/调度/抢占 ------
        # Avoid circular dependency
        from vllm.v1.engine import EngineCoreEventType

        for event in events:
            if event.type == EngineCoreEventType.QUEUED:
                req_stats.queued_ts = event.timestamp
                lora_states.request_waiting(req_id, lora_name)
            elif event.type == EngineCoreEventType.SCHEDULED:
                if req_stats.scheduled_ts == 0.0:  # ignore preemptions
                    req_stats.scheduled_ts = event.timestamp
                lora_states.request_running(req_id, lora_name)
            elif event.type == EngineCoreEventType.PREEMPTED:
                self.num_preempted_reqs += 1
                lora_states.request_waiting(req_id, lora_name)

    def update_from_finished_request(
        self,
        finish_reason: "FinishReason",
        request_id: str,
        num_prompt_tokens: int,
        max_tokens_param: int | None,
        req_stats: RequestStateStats,
        num_cached_tokens: int = 0,
    ):
        # ------【核心逻辑】update_from_finished_request：结算各阶段延迟 ------
        e2e_latency = self._time_since(req_stats.arrival_time)

        # Queued interval is from first QUEUED event to first SCHEDULED
        queued_time = req_stats.scheduled_ts - req_stats.queued_ts

        # Prefill interval is from first SCHEDULED to first NEW_TOKEN
        # Any preemptions during prefill is included in the interval
        prefill_time = req_stats.first_token_ts - req_stats.scheduled_ts

        # Decode interval is from first NEW_TOKEN to last NEW_TOKEN
        # Any preemptions during decode are included
        decode_time = req_stats.last_token_ts - req_stats.first_token_ts

        # Inference interval is from first SCHEDULED to last NEW_TOKEN
        # Any preemptions during prefill or decode are included
        inference_time = req_stats.last_token_ts - req_stats.scheduled_ts

        # Do not count the token generated by the prefill phase
        mean_time_per_output_token = (
            decode_time / (req_stats.num_generation_tokens - 1)
            if req_stats.num_generation_tokens - 1 > 0
            else 0
        )

        finished_req = FinishedRequestStats(
            finish_reason=finish_reason,
            request_id=request_id,
            e2e_latency=e2e_latency,
            num_prompt_tokens=num_prompt_tokens,
            num_generation_tokens=req_stats.num_generation_tokens,
            max_tokens_param=max_tokens_param,
            queued_time=queued_time,
            prefill_time=prefill_time,
            inference_time=inference_time,
            decode_time=decode_time,
            mean_time_per_output_token=mean_time_per_output_token,
            is_corrupted=req_stats.is_corrupted,
            num_cached_tokens=num_cached_tokens,
        )
        self.finished_requests.append(finished_req)

        # Count corrupted requests when they finish (only once per request)
        if req_stats.is_corrupted:
            self.num_corrupted_reqs += 1


class LoRAStats:
    # ------【LoRA】单个 LoRA 的请求状态：跟踪等待/运行中的请求 ID 集合 ------
    """Tracks waiting and running request IDs for a single LoRA."""

    def __init__(self):
        # ------【LoRA】waiting：等待中的请求 ID 集合 ------
        self.waiting: set[str] = set()
        # ------【LoRA】running：运行中的请求 ID 集合 ------
        self.running: set[str] = set()

    def update(self, req_id: str, waiting: bool, running: bool):
        # ------【LoRA】update：按标志加入/移出对应集合 ------
        assert not (waiting and running)
        if waiting:
            self.waiting.add(req_id)
        else:
            self.waiting.discard(req_id)

        if running:
            self.running.add(req_id)
        else:
            self.running.discard(req_id)

    @property
    def empty(self) -> bool:
        # ------【LoRA】empty：该 LoRA 是否已无任何等待/运行请求 ------
        return not (self.waiting or self.running)


class LoRARequestStates:
    # ------【LoRA】全局 LoRA 请求状态表：维护各 LoRA 等待/运行请求数 ------
    """A per-LoRA count of running and waiting requests."""

    def __init__(self, log_stats: bool = False):
        # ------【LoRA】log_stats：是否启用 LoRA 请求状态统计 ------
        self.log_stats = log_stats
        # ------【LoRA】requests：LoRA 名→LoRAStats 的映射表 ------
        self.requests: defaultdict[str, LoRAStats] = defaultdict(LoRAStats)

    def _request_update(
        self, req_id: str, lora_name: str | None, waiting: bool, running: bool
    ):
        # ------【LoRA】_request_update：更新请求在某 LoRA 下的等待/运行态 ------
        if not self.log_stats or lora_name is None:
            return

        lora_stats = self.requests[lora_name]
        lora_stats.update(req_id, waiting, running)
        if lora_stats.empty:
            del self.requests[lora_name]

    def request_waiting(self, req_id: str, lora_name: str | None):
        # ------【LoRA】request_waiting：标记请求进入等待态（入队/被抢占） ------
        self._request_update(req_id, lora_name, waiting=True, running=False)

    def request_running(self, req_id: str, lora_name: str | None):
        # ------【LoRA】request_running：标记请求进入运行态（被调度） ------
        self._request_update(req_id, lora_name, waiting=False, running=True)

    def request_finished(self, req_id: str, lora_name: str | None):
        # ------【LoRA】request_finished：标记请求完成，移出等待/运行集合 ------
        self._request_update(req_id, lora_name, waiting=False, running=False)

    def update_scheduler_stats(self, scheduler_stats: SchedulerStats | None):
        # ------【LoRA】update_scheduler_stats：把 LoRA 请求数写入调度统计 ------
        if not self.log_stats or scheduler_stats is None:
            return
        for lora_name, stats in self.requests.items():
            scheduler_stats.waiting_lora_adapters[lora_name] = len(stats.waiting)
            scheduler_stats.running_lora_adapters[lora_name] = len(stats.running)
