# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import enum
import time
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from vllm.multimodal.inputs import MultiModalFeatureSpec
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import SamplingParams
from vllm.utils import length_from_prompt_token_ids_or_embeds
from vllm.v1.engine import (
    EngineCoreEvent,
    EngineCoreEventType,
    EngineCoreRequest,
    FinishReason,
)
from vllm.v1.metrics.stats import PrefillStats
from vllm.v1.structured_output.request import StructuredOutputRequest
from vllm.v1.utils import ConstantList

if TYPE_CHECKING:
    from vllm.lora.request import LoRARequest
    from vllm.v1.core.kv_cache_utils import BlockHash


# ------【异步 RPC】StreamingUpdate：EngineCore→调度器(内部 deque 队列)，流式会话续传 DTO，涉及流式输入优化 ------
@dataclass
class StreamingUpdate:
    """Lightweight data for streaming session continuation.

    Contains only the fields needed to update an existing streaming session
    with new input data.
    """

    # ------【核心逻辑】mm_features：本次续传新增的多模态特征（多模态模型输入） ------
    mm_features: list[MultiModalFeatureSpec] | None
    # ------【核心逻辑】prompt_token_ids：本次续传新增的 prompt token id ------
    prompt_token_ids: list[int] | None
    # ------【核心逻辑】max_tokens：最大可生成 token 数（长度上限） ------
    max_tokens: int
    # ------【核心逻辑】arrival_time：本段输入到达时间（用于调度排序） ------
    arrival_time: float
    # ------【核心逻辑】sampling_params：采样参数（可能随续传更新） ------
    sampling_params: SamplingParams | None

    @classmethod
    def from_request(cls, request: "Request") -> "StreamingUpdate | None":
        # ------【异步 RPC】从可恢复 Request 中提取续传字段，构造流式续传消息 ------
        if not request.resumable:
            return None
        return cls(
            mm_features=request.mm_features,
            prompt_token_ids=request.prompt_token_ids,
            max_tokens=request.max_tokens,
            arrival_time=request.arrival_time,
            sampling_params=request.sampling_params,
        )


# ------【核心逻辑】Request：贯穿调度器的核心状态对象，前端→Scheduler/Worker 全程携带，涉及前缀缓存/投机解码/PD分离等优化 ------
class Request:
    def __init__(
        self,
        request_id: str,
        prompt_token_ids: list[int] | None,
        sampling_params: SamplingParams | None,
        pooling_params: PoolingParams | None,
        client_index: int = 0,
        arrival_time: float | None = None,
        prompt_embeds: torch.Tensor | None = None,
        prompt_is_token_ids: list[bool] | None = None,
        mm_features: list[MultiModalFeatureSpec] | None = None,
        lora_request: "LoRARequest | None" = None,
        cache_salt: str | None = None,
        priority: int = 0,
        trace_headers: Mapping[str, str] | None = None,
        block_hasher: Callable[["Request"], list["BlockHash"]] | None = None,
        resumable: bool = False,
        session_id: str | None = None,
        reasoning_ended: bool | None = None,
        reasoning_parser_kwargs: dict[str, Any] | None = None,
        abort_immediately: bool = False,
    ) -> None:
        # ------【核心逻辑】request_id：请求唯一 ID（调度排序与事件回传的关键字） ------
        self.request_id = request_id
        # ------【核心逻辑】client_index：来源客户端索引（多客户端区分请求归属） ------
        self.client_index = client_index
        # ------【核心逻辑】priority：请求优先级（用于优先级调度排序） ------
        self.priority = priority
        # ------【核心逻辑】sampling_params：采样参数（温度/top-p/stop 等） ------
        self.sampling_params = sampling_params
        # ------【核心逻辑】pooling_params：池化参数（embedding/池化模型专用） ------
        self.pooling_params = pooling_params
        # ------【LoRA】lora_request：LoRA 适配器请求，标识挂载哪个低秩适配器 ------
        self.lora_request = lora_request
        # ------【结构化输出/grammar】structured_output_request：约束解码请求（JSON schema/grammar 引导） ------
        self.structured_output_request = StructuredOutputRequest.from_sampling_params(
            sampling_params
        )
        if self.structured_output_request is not None:
            self.structured_output_request.reasoning_ended = reasoning_ended
            self.structured_output_request.reasoning_parser_kwargs = (
                reasoning_parser_kwargs
            )
        # ------【核心逻辑】arrival_time：请求到达时间（默认取当前时间，调度排序依据） ------
        self.arrival_time = arrival_time if arrival_time is not None else time.time()

        # ------【核心逻辑】status：请求当前状态（状态机驱动调度与完成判定） ------
        self.status = RequestStatus.WAITING
        # ------【核心逻辑】events：引擎事件列表（供前端查询 EngineCoreEvent） ------
        self.events: list[EngineCoreEvent] = []
        # ------【核心逻辑】stop_reason：停止原因（int 或 str，用于回传） ------
        self.stop_reason: int | str | None = None

        # ------【PD 分离】kv_transfer_params：连接器 KV 传输参数（P/D 分离跨引擎传 KV） ------
        # P/D: Connector-specific KV transfer parameters.
        self.kv_transfer_params: dict[str, Any] | None = None
        # ------【EP】ec_transfer_params：连接器 encoder-cache 传输参数（跨引擎传编码器缓存） ------
        # E/P/D: Connector-specific encoder-cache transfer parameters.
        self.ec_transfer_params: dict[str, Any] | None = None

        if pooling_params is not None:
            # Pooling models.
            # ------【核心逻辑】max_tokens：池化模型只产出 1 个池化结果 token ------
            self.max_tokens = 1
        elif sampling_params is not None:
            # Generative models.
            assert sampling_params.max_tokens is not None
            # ------【核心逻辑】max_tokens：生成模型的最大输出 token 数（长度上限） ------
            self.max_tokens = sampling_params.max_tokens
            if self.structured_output_request is not None:
                # ------【结构化输出/grammar】等待 grammar 编译完成再调度 ------
                self.status = RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR

            if sampling_params.extra_args is not None:
                self.kv_transfer_params = sampling_params.extra_args.get(
                    "kv_transfer_params"
                )
                self.ec_transfer_params = sampling_params.extra_args.get(
                    "ec_transfer_params"
                )
                # ------【前缀缓存】kv_cache_report_mode：KV cache 上报模式（默认增量上报） ------
                self.kv_cache_report_mode = sampling_params.extra_args.get(
                    "kv_cache_report_mode", "incremental"
                )
            else:
                self.kv_cache_report_mode = "incremental"
        else:
            raise ValueError("sampling_params and pooling_params can't both be unset")

        # ------【核心逻辑】prompt_token_ids：输入 prompt 的 token id 列表 ------
        self.prompt_token_ids = prompt_token_ids
        # ------【核心逻辑】prompt_embeds：输入 prompt 的嵌入（多模态/embeds 混用模式） ------
        self.prompt_embeds = prompt_embeds
        # ------【核心逻辑】prompt_is_token_ids：逐位掩码，标记该位置是 token id 还是 embeds（混合模式） ------
        # Per-position mask used in mixed-mode (chat completion with
        # prompt_embeds). `None` except when both `prompt_token_ids` and
        # `prompt_embeds` are set and their positions are interleaved.
        self.prompt_is_token_ids = prompt_is_token_ids
        # ------【前缀缓存】_prompt_embeds_per_block_hashes：逐块 prompt 嵌入哈希缓存，避免重复哈希 ------
        # Cache per-block prompt-embed hashes to avoid rehashing the same
        # tensor slices when generating extra keys.
        self._prompt_embeds_per_block_hashes: dict[tuple[int, int], bytes] = {} # 多模态相关的，不要看
        # ------【核心逻辑】num_prompt_tokens：prompt 总 token 数（由 token_ids 或 embeds 推导） ------
        self.num_prompt_tokens = length_from_prompt_token_ids_or_embeds(
            prompt_token_ids, prompt_embeds
        )
        # ------【核心逻辑】_output_token_ids：已生成输出 token id 列表（底层可变存储） ------
        self._output_token_ids: list[int] = []
        # ------【核心逻辑】_all_token_ids：prompt+输出全量 token id 列表（KV 定位基础） ------
        self._all_token_ids: list[int] = (
            self.prompt_token_ids.copy()
            if self.prompt_token_ids is not None
            else [0] * self.num_prompt_tokens
        )

        # ------【投机解码/异步 RPC】num_output_placeholders：异步调度中预留的输出占位符数 ------
        # Used in async scheduling.
        self.num_output_placeholders = 0
        # ------【核心逻辑】num_stale_output_tokens：被抢占时在途的 stale 输出 token 数 ------
        # Tokens of output in flight when the request was preempted: delivered
        # on return, but must not mutate the reset counters.
        self.num_stale_output_tokens = 0
        # ------【前缀缓存】drop_stale_output：同一步抢占+恢复时丢弃 stale 输出（reset_prefix_cache） ------
        # Drop the stale output instead, for same-step preempt + resume
        # (reset_prefix_cache).
        self.drop_stale_output = False

        # ------【PP/异步 RPC】num_in_flight_tokens：在途未处理 token 数（异步调度+PP 跑在 GPU 前） ------
        # Tokens of steps whose output is not yet processed (async scheduling
        # and PP run ahead of the GPU); `num_computed_tokens` counts them
        # optimistically.
        self.num_in_flight_tokens = 0

        # ------【PP】next_decode_eligible_step：V2+PP+async 下强制 pp_size 节拍的 decode 步号 ------
        # V2+PP+async: Enforces `pp_size` cadence between same-request decode steps
        # so the worker's broadcast slot ring stays consistent.
        self.next_decode_eligible_step = 0

        # ------【核心逻辑】last_sched_seq：最近一次被调度的步号，围栏延迟释放 block ------
        # Seq of the most recent step this request was scheduled in; fences
        # deferred block freeing (see Scheduler._free_request_blocks).
        self.last_sched_seq = 0

        # ------【投机解码】spec_token_ids：投机解码的候选 token id 列表 ------
        self.spec_token_ids: list[int] = []
        # ------【核心逻辑】num_computed_tokens：已计算（进入 KV cache）的 token 数 ------
        self.num_computed_tokens = 0
        # ------【前缀缓存】cache_salt：缓存盐值（区分不同前缀缓存命名空间） ------
        self.cache_salt: str | None = cache_salt

        # ------【核心逻辑】mm_features：多模态输入特征列表（图像/音频等编码器输入） ------
        # Multi-modal related
        self.mm_features = mm_features or []

        # ------【核心逻辑】output_token_ids/all_token_ids：只读视图，防止直接 append 破坏同步 ------
        # Read-only views
        # Prevent directly appending to these lists since
        # they should also be updated simultaneously.
        self.output_token_ids = ConstantList(self._output_token_ids)
        self.all_token_ids = ConstantList(self._all_token_ids)
        # ------【核心逻辑】trace_headers：追踪 headers（透传观测字段） ------
        # trace_headers
        self.trace_headers = trace_headers
        # ------【核心逻辑】session_id：会话 ID（关联同一流式/会话的多段请求） ------
        self.session_id = session_id

        # ------【chunked prefill】is_prefill_chunk：是否作为非最终 prefill chunk 被调度 ------
        # True if this request is scheduled as a non-final prefill chunk.
        self.is_prefill_chunk = False

        # ------【前缀缓存】shared_prefix_boundary：值得钉入稀疏前缀缓存的共享前缀块对齐边界 ------
        # Block-aligned token position of a proven shared prefix worth pinning
        # in the (sparse) prefix cache; 0 means none. Set at admission for
        # hybrid/Mamba models when a shared prefix is detected (Marconi-style).
        self.shared_prefix_boundary = 0

        # ------【核心逻辑】num_nans_in_logits：logits 中 NaN 计数（>0 表示输出已损坏） ------
        # The number of NaNs in logits. A value greater than 0
        # indicates that the output is corrupted
        self.num_nans_in_logits = 0

        # ------【核心逻辑】num_preemptions：被调度器抢占的次数（抢占统计） ------
        # The number of times this request has been preempted by the scheduler.
        self.num_preemptions = 0

        # ------【核心逻辑】prefill_stats：prefill 阶段统计（供 metrics 上报） ------
        self.prefill_stats: PrefillStats | None = PrefillStats()

        # ------【前缀缓存】block_hashes：已计算的块哈希列表（用于前缀缓存查找/上报） ------
        self.block_hashes: list[BlockHash] = []
        # ------【前缀缓存】_block_hasher：块哈希器（不绑定 self 避免引用循环） ------
        # Store the block hasher without binding self to avoid creating a
        # reference cycle (Request -> partial -> Request) that prevents
        # immediate garbage collection via reference counting.
        self._block_hasher: Callable[[Request], list[BlockHash]] | None = block_hasher
        self.update_block_hashes()

        # ------【前缀缓存】skip_reading_prefix_cache：是否跳过读取前缀缓存 ------
        self.skip_reading_prefix_cache = self.get_skip_reading_prefix_cache()

        # ------【异步 RPC】resumable：请求是否可恢复（流式续传标志） ------
        # Used for streaming
        self.resumable = resumable
        # ------【异步 RPC】streaming_queue：流式续传队列，None 元素表示流结束 ------
        # None entry in the queue means finished.
        self.streaming_queue: deque[StreamingUpdate | None] | None = None

        # ------【核心逻辑】abort_immediately：加入调度器后立即中止（触发连接器 request_finished 钩子） ------
        # If True, request should be aborted immediately after being added to
        # the scheduler so the connector's request_finished hook runs.
        self.abort_immediately = abort_immediately

    @classmethod
    def from_engine_core_request(
        cls,
        request: EngineCoreRequest,
        block_hasher: Callable[["Request"], list["BlockHash"]] | None,
    ) -> "Request":
        # ------【核心逻辑】把前端 EngineCoreRequest DTO 转换为内部 Request 状态对象 ------
        return cls(
            request_id=request.request_id,
            client_index=request.client_index,
            prompt_token_ids=request.prompt_token_ids,
            prompt_embeds=request.prompt_embeds,
            prompt_is_token_ids=request.prompt_is_token_ids,
            mm_features=request.mm_features,
            sampling_params=request.sampling_params,
            pooling_params=request.pooling_params,
            arrival_time=request.arrival_time,
            lora_request=request.lora_request,
            cache_salt=request.cache_salt,
            priority=request.priority,
            trace_headers=request.trace_headers,
            block_hasher=block_hasher,
            resumable=request.resumable,
            session_id=request.session_id,
            reasoning_ended=request.reasoning_ended,
            reasoning_parser_kwargs=request.reasoning_parser_kwargs,
            abort_immediately=request.abort_immediately,
        )

    def append_output_token_ids(
        self,
        token_ids: int | list[int],
    ) -> None:
        # ------【核心逻辑】追加新生成的输出 token，同步维护 _all_token_ids 并更新块哈希 ------
        if isinstance(token_ids, int):
            self._output_token_ids.append(token_ids)
            self._all_token_ids.append(token_ids)
        else:
            self._output_token_ids.extend(token_ids)
            self._all_token_ids.extend(token_ids)

        self.update_block_hashes()

    def update_block_hashes(self) -> None:
        """Compute block hashes for any new full blocks and append them."""
        # ------【前缀缓存】为新写满的 block 计算哈希并追加，供前缀缓存增量上报 ------
        if self._block_hasher is not None:
            self.block_hashes.extend(self._block_hasher(self))

    @property
    def use_structured_output(self) -> bool:
        # ------【结构化输出/grammar】是否启用约束解码 ------
        return self.structured_output_request is not None

    @property
    def num_tokens(self) -> int:
        # ------【核心逻辑】全量 token 数（prompt+输出） ------
        return len(self._all_token_ids)

    @property
    def num_tokens_with_spec(self) -> int:
        # ------【投机解码】含投机候选 token 的总 token 数 ------
        return len(self._all_token_ids) + len(self.spec_token_ids)

    @property
    def num_output_tokens(self) -> int:
        # ------【核心逻辑】已生成的输出 token 数 ------
        return len(self._output_token_ids)

    @property
    def num_encoder_inputs(self) -> int:
        # ------【核心逻辑】编码器输入（多模态特征）数量 ------
        return len(self.mm_features)

    @property
    def has_encoder_inputs(self) -> bool:
        # ------【核心逻辑】是否含编码器输入（多模态） ------
        return self.num_encoder_inputs > 0

    def get_skip_reading_prefix_cache(self) -> bool:
        # ------【前缀缓存】从 sampling/pooling 参数读取是否跳过读取前缀缓存 ------
        if (
            self.sampling_params is not None
            and self.sampling_params.skip_reading_prefix_cache is not None
        ):
            return self.sampling_params.skip_reading_prefix_cache
        elif (
            self.pooling_params is not None
            and self.pooling_params.skip_reading_prefix_cache is not None
        ):
            return self.pooling_params.skip_reading_prefix_cache
        return False

    def is_finished(self) -> bool:
        # ------【核心逻辑】请求是否已进入完成态 ------
        return RequestStatus.is_finished(self.status)

    def get_finished_reason(self) -> FinishReason | None:
        # ------【核心逻辑】将完成态映射为对外 FinishReason ------
        return RequestStatus.get_finished_reason(self.status)

    def get_num_encoder_embeds(self, input_id: int) -> int:
        # ------【核心逻辑】返回第 input_id 个编码器输入的嵌入数 ------
        assert input_id < len(self.mm_features)
        return self.mm_features[input_id].mm_position.get_num_embeds()

    def record_event(
        self,
        event_type: EngineCoreEventType,
        timestamp: float | None = None,
    ) -> None:
        # ------【核心逻辑】记录一条引擎事件（带时间戳） ------
        self.events.append(EngineCoreEvent.new_event(event_type, timestamp))

    def take_events(self) -> list[EngineCoreEvent] | None:
        # ------【核心逻辑】取走并清空事件列表（前端一次性消费） ------
        if not self.events:
            return None
        events, self.events = self.events, []
        return events

    def take_prefill_stats(self) -> PrefillStats | None:
        # ------【核心逻辑】取走 prefill 统计并置空（一次性消费） ------
        if self.prefill_stats is None:
            return None
        prefill_stats = self.prefill_stats
        self.prefill_stats = None
        return prefill_stats

    def __lt__(self, other: "Request") -> bool:
        """
        Compare two requests based on priority, arrival time, and request ID.
        Used in priority scheduling.
        """
        # ------【核心逻辑】按优先级→到达时间→请求 ID 排序，用于优先级调度堆 ------
        if self.priority != other.priority:
            return self.priority < other.priority
        if self.arrival_time != other.arrival_time:
            return self.arrival_time < other.arrival_time
        if self.request_id != other.request_id:
            return self.request_id < other.request_id
        return id(self) < id(other)


# ------【核心逻辑】RequestStatus：请求生命周期状态机，Scheduler/Worker 通过该状态驱动调度与完成判定 ------
class RequestStatus(enum.IntEnum):
    """Status of a request.
    
    | 状态                                      | 含义              | 是否可调度 | 说明                                      |
| --------------------------------------- | --------------- | ----- | --------------------------------------- |
| `RUNNING`                               | 正在执行            | ✅     | 已进入 scheduler 调度，拥有执行状态和 KV cache       |

| `PREEMPTED`                             | 被抢占             | ❌（暂时） | 原来 RUNNING，被 scheduler 踢出，需要后续恢复        |

| `WAITING`                               | 普通等待            | ✅     | 新请求进入后的默认状态，等待 Scheduler 选择             |
| `WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR` | 等待结构化输出 grammar | ❌     | JSON schema / guided decoding 的约束状态未准备好 |
| `WAITING_FOR_REMOTE_KVS`                | 等待远程 KV cache   | ❌     | P/D 分离场景，等待远端传输 KV cache                |
| `WAITING_FOR_STREAMING_REQ`             | 等待流式输入          | ❌     | streaming 输入还没有完成，等待更多输入 token          |


| `FINISHED_STOPPED`                      | 正常停止            | ❌     | EOS 或 stop string 触发结束                  |
| `FINISHED_LENGTH_CAPPED`                | 长度限制结束          | ❌     | 达到 max_tokens 或 max_model_len           |
| `FINISHED_ABORTED`                      | 主动取消            | ❌     | 用户 cancel 或服务端终止                        |
| `FINISHED_IGNORED`                      | 忽略结束            | ❌     | 请求被忽略，不再处理                              |
| `FINISHED_ERROR`                        | 异常结束            | ❌     | 推理过程中发生错误                               |
| `FINISHED_REPETITION`                   | 重复生成结束          | ❌     | repetition stopping 触发                  |
    
    """

    WAITING = enum.auto()
    WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR = enum.auto()
    WAITING_FOR_REMOTE_KVS = enum.auto()
    WAITING_FOR_STREAMING_REQ = enum.auto()
    RUNNING = enum.auto()
    PREEMPTED = enum.auto()
    # ------【核心逻辑】PREEMPTED 之后的状态一律视为完成态（is_finished 判定边界） ------
    # Note: anything after PREEMPTED will be considered
    # as a finished status.
    FINISHED_STOPPED = enum.auto()
    FINISHED_LENGTH_CAPPED = enum.auto()
    FINISHED_ABORTED = enum.auto()
    FINISHED_IGNORED = enum.auto()
    FINISHED_ERROR = enum.auto()
    FINISHED_REPETITION = enum.auto()

    def __str__(self) -> str:
        # ------【核心逻辑】状态名的字符串表示 ------
        return self.name

    @staticmethod
    def is_finished(status: "RequestStatus") -> bool:
        # ------【核心逻辑】PREEMPTED 之后的状态都视为完成态 ------
        return status > RequestStatus.PREEMPTED

    @staticmethod
    def get_finished_reason(status: "RequestStatus") -> FinishReason | None:
        # ------【核心逻辑】完成态 → FinishReason 映射查询 ------
        return _FINISHED_REASON_MAP.get(status)


# Mapping of finished statuses to their finish reasons.
# NOTE: The ignored requests are the requests whose prompt lengths
# are longer than the model's length cap. Therefore, the stop
# reason should also be "length" as in OpenAI API.
_FINISHED_REASON_MAP = {
    RequestStatus.FINISHED_STOPPED: FinishReason.STOP,
    RequestStatus.FINISHED_LENGTH_CAPPED: FinishReason.LENGTH,
    RequestStatus.FINISHED_ABORTED: FinishReason.ABORT,
    RequestStatus.FINISHED_IGNORED: FinishReason.LENGTH,
    RequestStatus.FINISHED_ERROR: FinishReason.ERROR,
    RequestStatus.WAITING_FOR_STREAMING_REQ: FinishReason.STOP,
    RequestStatus.FINISHED_REPETITION: FinishReason.REPETITION,
}
