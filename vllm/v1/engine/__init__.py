# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import enum
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

import msgspec
import numpy as np
import torch

from vllm.config.kv_events import KVEventsConfig
from vllm.lora.request import LoRARequest
from vllm.multimodal.inputs import MultiModalFeatureSpec
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import SamplingParams
from vllm.v1.metrics.stats import PrefillStats, SchedulerStats
from vllm.v1.outputs import LogprobsLists, LogprobsTensors
from vllm.v1.serial_utils import UtilityResult

# Type for pause_generation mode parameter.
# - "abort": Abort all in-flight requests immediately (default).
# - "wait": Wait for in-flight requests to complete before pausing.
# - "keep": Freeze requests in queue; they resume on resume_generation().
PauseMode = Literal["abort", "wait", "keep"]

# These are possible values of RequestOutput.finish_reason,
# so form part of the external API.
FINISH_REASON_STRINGS = ("stop", "length", "abort", "error", "repetition")

EEP_NOTIFICATION_CALL_ID = -1

FT_STATUS_CALL_ID = -2


# ------【EP/EPLB/ZMQ 通信】EEPNotificationType：EngineCore→前端 的 EEP(弹性专家并行)事件通知枚举，经消息通道回传 ------
class EEPNotificationType(enum.Enum):
    # ------【EP/EPLB】RECONFIGURE_FINISHED：专家并行动态扩缩容完成通知 ------
    RECONFIGURE_FINISHED = "RECONFIGURE_FINISHED"
    # ------【EP/EPLB】SHUTDOWN_COMPLETE：专家并行实例关闭完成通知 ------
    SHUTDOWN_COMPLETE = "SHUTDOWN_COMPLETE"


# ------【核心逻辑】FinishReason：EngineCore→前端 的请求结束原因枚举，随 EngineCoreOutput 回传；用 Int 更紧凑 ------
class FinishReason(enum.IntEnum):
    """
    Reason a request finished - stop, length, abort, error, or repetition.

    Int rather than Str for more compact serialization.

    stop - a stop string was emitted
    length - max_tokens was consumed, or max_model_len was reached
    abort - aborted by client
    error - retryable request-level internal error (e.g., KV load failure).
            Invariant: always converted to 500 Internal Server Error.
    repetition - repetitive token pattern detected (hallucination)

    """

    STOP = 0
    LENGTH = 1
    ABORT = 2
    ERROR = 3
    REPETITION = 4

    def __str__(self):
        return FINISH_REASON_STRINGS[self.value]


# ------【显存 profiling/DP/TP/PP】EngineCoreReadyResponse：EngineCore→前端 启动就绪回包，经消息通道回传初始化后的真实配置 ------
@dataclass
class EngineCoreReadyResponse:
    """Sent from EngineCore to each frontend at the end of engine startup.

    Contains post-initialization config that may differ from the original
    values (e.g. max_model_len after KV cache auto-fitting).
    """

    # ------【显存 profiling】max_model_len：KV cache 自动适配后的实际最大序列长度 ------
    max_model_len: int
    # ------【内存池/CuMem】num_gpu_blocks：GPU KV cache 的 block 数量（显存切分结果） ------
    num_gpu_blocks: int
    # ------【内存池/CuMem】block_size：单个 KV block 覆盖的 token 数（前缀缓存分块单位） ------
    block_size: int
    # ------【DP】dp_stats_address：DP 状态统计服务地址（前端拉取统计用） ------
    dp_stats_address: str | None
    # ------【核心逻辑】dtype：模型权重 dtype ------
    dtype: str
    # ------【核心逻辑】vllm_version：当前 vLLM 版本号 ------
    vllm_version: str
    # ------【进程管理】world_size：参与推理的进程总数（含 TP/PP/DP） ------
    world_size: int
    # ------【DP】data_parallel_size：DP 数据并行规模 ------
    data_parallel_size: int
    # ------【TP】tensor_parallel_size：TP 张量并行规模 ------
    tensor_parallel_size: int
    # ------【PP】pipeline_parallel_size：PP 流水线并行规模 ------
    pipeline_parallel_size: int
    # ------【PD 分离】decode_context_parallel_size：decode 阶段上下文并行规模 ------
    decode_context_parallel_size: int
    # ------【DP】data_parallel_rank：本实例在 DP 维度上的 rank ------
    data_parallel_rank: int
    # ------【chunked prefill】max_num_seqs：单 step 最多调度的并发序列数 ------
    max_num_seqs: int
    # ------【chunked prefill】max_num_batched_tokens：单 step 最多批处理的 token 数 ------
    max_num_batched_tokens: int
    # ------【进程管理】instance_id：引擎实例唯一标识 ------
    instance_id: str
    # KV cache capacity (None for encoder-only/attention-free models).
    # ------【显存 profiling/内存池】kv_cache_size_tokens：KV cache 容量（token 数） ------
    kv_cache_size_tokens: int | None = None
    # ------【显存 profiling】kv_cache_max_concurrency：KV cache 允许的最大并发 ------
    kv_cache_max_concurrency: float | None = None
    # ------【显存 profiling】kv_events_config：KV 事件配置（sleep/wake 显存规划） ------
    kv_events_config: KVEventsConfig | None = None


# ------【ZMQ 通信/DP/PD 分离/前缀缓存/LoRA/结构化输出】EngineCoreRequest：前端→EngineCore 的请求消息，经 ZMQ 发送 ------
class EngineCoreRequest(
    msgspec.Struct,
    array_like=True,  # type: ignore[call-arg]
    omit_defaults=True,  # type: ignore[call-arg]
    gc=False,
):  # type: ignore[call-arg]
    # ------【核心逻辑】request_id：内部请求唯一 ID（由 InputProcessor 分配） ------
    request_id: str
    # ------【核心逻辑】prompt_token_ids：纯 token 输入的 prompt token id 列表 ------
    prompt_token_ids: list[int] | None
    # ------【核心逻辑】mm_features：多模态输入特征（图片/音频等），与纯文本互斥 ------
    mm_features: list[MultiModalFeatureSpec] | None
    # ------【结构化输出/grammar】sampling_params：采样参数（温度/top-k/top-p 及结构化输出约束） ------
    sampling_params: SamplingParams | None
    # ------【核心逻辑】pooling_params：pooling 参数（embedding/打分任务，与采样互斥） ------
    pooling_params: PoolingParams | None
    # ------【核心逻辑】arrival_time：请求到达时间戳（用于排队等待时长统计） ------
    arrival_time: float
    # ------【LoRA】lora_request：LoRA 适配器请求，指定本次推理加载的低秩适配器 ------
    lora_request: LoRARequest | None
    # ------【前缀缓存】cache_salt：前缀缓存盐值，改变命中 key 以隔离缓存命名空间 ------
    cache_salt: str | None
    # ------【DP】data_parallel_rank：显式指定请求路由到的 DP rank ------
    data_parallel_rank: int | None
    # ------【核心逻辑】prompt_embeds：预计算 prompt embedding（免 tokenizer 编码路径） ------
    prompt_embeds: torch.Tensor | None = None

    # Per-position mask for mixed-mode inputs (e.g chat completion with
    # prompt_embeds content parts). `True` means the position is a real
    # token ID; `False` means the position uses a pre-computed entry from
    # `prompt_embeds`. `None` for pure-tokens and pure-embeds requests.
    # ------【核心逻辑】prompt_is_token_ids：混合输入逐位掩码（token 与 embed 混合场景） ------
    prompt_is_token_ids: list[bool] | None = None

    # Index of the client, used to ensure outputs are sent back to the same
    # client for this request when scaling out the front-end.
    # ------【异步 RPC】client_index：前端客户端索引，保证输出路由回原客户端 ------
    client_index: int = 0

    # Used in DP case to indicate which wave of requests this is expected to
    # belong to, to cover a race condition where the request is sent before
    # a wave finished notification is received.
    # ------【DP】current_wave：DP 波次标记，用于请求先于波完成通知到达的竞态处理 ------
    current_wave: int = 0
    # ------【核心逻辑】priority：请求调度优先级 ------
    priority: int = 0

    # ------【核心逻辑】trace_headers：追踪 headers（透传做链路追踪） ------
    trace_headers: Mapping[str, str] | None = None
    # ------【PD 分离】resumable：是否支持断点续传（KV 传输失败后恢复） ------
    resumable: bool = False

    # The user-provided request ID. This field is set internally,
    # copied from the provided request_id that's originally assigned
    # to the request_id field, see InputProcessor.assign_request_id().
    # Used in outputs and to support abort(req_id, internal=False).
    # ------【核心逻辑】external_req_id：用户侧原始请求 ID（支持 abort 时按外部 ID 定位） ------
    external_req_id: str | None = None

    # ------【核心逻辑】reasoning_ended：reasoning 模型是否已结束推理阶段的标记 ------
    reasoning_ended: bool | None = None
    # ------【核心逻辑】reasoning_parser_kwargs：reasoning 内容解析器参数 ------
    reasoning_parser_kwargs: dict[str, Any] | None = None

    # If True, the request should be added to the scheduler's waiting queue
    # and immediately aborted, so connector-side cleanup runs via the standard
    # request_finished hook. Used to free P-side prefill blocks when a
    # KV-transfer request is rejected on the D node before engine admission.
    # ------【PD 分离】abort_immediately：立即 abort 标记，用于释放 PD 分离中 P 侧 prefill block ------
    abort_immediately: bool = False

    # ------【核心逻辑】session_id：多轮会话 ID（用于会话级状态管理） ------
    session_id: str | None = None

    # 统一取采样或 pooling 参数（二者互斥，sampling 优先）。
    @property
    def params(self) -> SamplingParams | PoolingParams:
        """Return the processed params (sampling or pooling)."""
        if self.sampling_params is not None:
            return self.sampling_params
        assert self.pooling_params is not None
        return self.pooling_params


# ------【核心逻辑】EngineCoreEventType：EngineCore 请求事件类型枚举，随输出回传前端计算耗时 ------
class EngineCoreEventType(enum.IntEnum):
    """The type of engine core request event."""

    # ------【核心逻辑】QUEUED：请求进入等待队列事件 ------
    QUEUED = 1
    # ------【核心逻辑】SCHEDULED：请求被调度执行事件 ------
    SCHEDULED = 2
    # ------【核心逻辑】PREEMPTED：请求被抢占事件 ------
    PREEMPTED = 3


# ------【核心逻辑】EngineCoreEvent：EngineCore→前端 带单调时间戳的请求事件，用于前端计算阶段耗时 ------
class EngineCoreEvent(msgspec.Struct):
    """A timestamped engine core event associated with a request.

    The timestamp is a monotonic timestamp and is used by the engine
    frontend to calculate intervals between engine core events. These
    timestamps should not be compared with timestamps from other processes.
    """

    # ------【核心逻辑】type：事件类型（排队/调度/抢占） ------
    type: EngineCoreEventType
    # ------【核心逻辑】timestamp：单调时钟时间戳（跨进程不可比） ------
    timestamp: float

    # 构造带时间戳的事件；不传时间戳时用当前单调时钟（仅进程内可比）。
    @classmethod
    def new_event(
        cls, event_type: EngineCoreEventType, timestamp: float | None = None
    ) -> "EngineCoreEvent":
        timestamp = time.monotonic() if timestamp is None else timestamp
        return cls(event_type, timestamp)


# ------【ZMQ 通信/投机解码/LoRA/PD 分离/EP】EngineCoreOutput：EngineCore→前端 单请求输出消息，随批次打包回传 ------
class EngineCoreOutput(
    msgspec.Struct,
    array_like=True,  # type: ignore[call-arg]
    omit_defaults=True,  # type: ignore[call-arg]
    gc=False,
):  # type: ignore[call-arg]
    # ------【核心逻辑】request_id：本次输出对应的请求 ID ------
    request_id: str
    # ------【核心逻辑】new_token_ids：本 step 新生成的 token id 列表 ------
    new_token_ids: list[int]

    # ------【投机解码】new_logprobs：新增 token 的对数概率（list 形式，用于拒绝采样/打分） ------
    new_logprobs: LogprobsLists | None = None
    # ------【核心逻辑】new_prompt_logprobs_tensors：prompt 阶段的 logprobs 张量 ------
    new_prompt_logprobs_tensors: LogprobsTensors | None = None

    # ------【核心逻辑】pooling_output：pooling 任务的输出张量（embedding 场景） ------
    pooling_output: torch.Tensor | None = None

    # ------【核心逻辑】finish_reason：请求结束原因（为 None 表示仍在生成） ------
    finish_reason: FinishReason | None = None
    # ------【结构化输出/grammar】stop_reason：停止原因（命中的停止词/字符串） ------
    stop_reason: int | str | None = None
    # ------【核心逻辑】events：本请求关联的事件时间戳列表（排队/调度/抢占） ------
    events: list[EngineCoreEvent] | None = None
    # ------【PD 分离】kv_transfer_params：KV cache 传输参数（PD 分离中 KV 迁移用） ------
    kv_transfer_params: dict[str, Any] | None = None
    # ------【PD 分离】ec_transfer_params：encoder cache 传输参数（编码器缓存迁移用） ------
    ec_transfer_params: dict[str, Any] | None = None

    # ------【核心逻辑】trace_headers：追踪 headers（透传） ------
    trace_headers: Mapping[str, str] | None = None

    # ------【chunked prefill】prefill_stats：prefill 阶段统计（分块预填充耗时等） ------
    prefill_stats: PrefillStats | None = None

    # ------【EP/EPLB】routed_experts：各 token 路由到的专家 id（专家并行/负载均衡统计） ------
    routed_experts: np.ndarray | None = None
    # The number of NaNs in logits.
    # A value greater than 0 indicates that the output is corrupted.
    # ------【核心逻辑】num_nans_in_logits：logits 中的 NaN 数量（>0 表示输出损坏） ------
    num_nans_in_logits: int = 0

    # 请求是否已结束（finish_reason 非空即结束）。
    @property
    def finished(self) -> bool:
        return self.finish_reason is not None


# ------【ZMQ 通信/异步 RPC】UtilityOutput：EngineCore→前端 utility 调用结果回包，经消息通道回传 ------
class UtilityOutput(
    msgspec.Struct,
    array_like=True,  # type: ignore[call-arg]
    gc=False,
):  # type: ignore[call-arg]
    # ------【异步 RPC】call_id：utility 调用 ID（用于关联请求与回包） ------
    call_id: int

    # Non-None implies the call failed, result should be None.
    # ------【异步 RPC】failure_message：调用失败信息（非空表示失败） ------
    failure_message: str | None = None
    # ------【异步 RPC】result：调用结果（失败时为 None） ------
    result: UtilityResult | None = None


# ------【ZMQ 通信/DP/异步 RPC】EngineCoreOutputs：EngineCore→前端 每 step 输出批次消息，经 ZMQ 回传 ------
class EngineCoreOutputs(
    msgspec.Struct,
    array_like=True,  # type: ignore[call-arg]
    omit_defaults=True,  # type: ignore[call-arg]
    gc=False,
):  # type: ignore[call-arg]
    # NOTE(Nick): We could consider ways to make this more compact,
    # e.g. columnwise layout

    # ------【DP/进程管理】engine_index：引擎实例索引（多引擎并发时区分来源） ------
    engine_index: int = 0

    # [num_reqs]
    # ------【ZMQ 通信】outputs：本 step 各请求的输出列表（长度 = 请求数） ------
    outputs: list[EngineCoreOutput] = []
    # ------【chunked prefill/EPLB】scheduler_stats：调度器统计（排队/批处理/负载均衡等） ------
    scheduler_stats: SchedulerStats | None = None
    # ------【核心逻辑】timestamp：本批次时间戳（默认取当前单调时钟） ------
    timestamp: float = 0.0

    # ------【异步 RPC】utility_output：utility 调用回包（与 call_id 对应） ------
    utility_output: UtilityOutput | None = None
    # ------【核心逻辑】finished_requests：本 step 结束的请求集合 ------
    finished_requests: set[str] | None = None

    # In DP case, used to signal that the current wave of requests
    # has finished and the engines are paused.
    # ------【DP】wave_complete：DP 当前波次完成信号（引擎暂停等待下一波） ------
    wave_complete: int | None = None
    # In DP case, used to signal that a request was received for an
    # "old" wave, so the next wave needs to be started in other engines.
    # ------【DP】start_wave：DP 旧波请求到达信号，触发其他引擎启动下一波 ------
    start_wave: int | None = None

    # 若 timestamp 未显式设置，则用当前单调时钟填充。
    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.monotonic()


# ------【ZMQ 通信/进程管理】EngineCoreRequestType：前端→EngineCore 请求类型字节枚举，直接走 socket 免编码 ------
class EngineCoreRequestType(enum.Enum):
    """
    Request types defined as hex byte strings, so it can be sent over sockets
    without separate encoding step.
    """

    # ------【核心逻辑】ADD：新增推理请求 ------
    ADD = b"\x00"
    # ------【核心逻辑】ABORT：中止请求 ------
    ABORT = b"\x01"
    # ------【DP】START_DP_WAVE：启动 DP 新波次 ------
    START_DP_WAVE = b"\x02"
    # ------【异步 RPC】UTILITY：utility 工具调用 ------
    UTILITY = b"\x03"
    # Sentinel used within EngineCoreProc.
    # ------【进程管理】EXECUTOR_FAILED：executor 失败哨兵（EngineCoreProc 内部用） ------
    EXECUTOR_FAILED = b"\x04"
    # Sentinel to wake up input_queue.get() during shutdown.
    # ------【进程管理】WAKEUP：关闭时唤醒 input_queue.get() 的哨兵 ------
    WAKEUP = b"\x05"


# ------【DP/进程管理/EP】ReconfigureDistributedRequest：控制端→EngineCore 的 DP 规模重配置消息（弹性扩缩容） ------
class ReconfigureDistributedRequest(msgspec.Struct):
    # ------【DP】new_data_parallel_size：重配置后的 DP 规模 ------
    new_data_parallel_size: int
    # ------【DP】new_data_parallel_rank：重配置后的全局 DP rank ------
    new_data_parallel_rank: int
    # ------【DP】new_data_parallel_rank_local：重配置后的本机内 DP rank ------
    new_data_parallel_rank_local: int
    # ------【DP/NCCL 通信】new_data_parallel_master_ip：DP master 节点 IP ------
    new_data_parallel_master_ip: str
    # ------【DP/NCCL 通信】new_data_parallel_master_port：DP master 节点端口 ------
    new_data_parallel_master_port: int
    # ------【DP/NCCL 通信】new_data_parallel_master_port_list：各 DP rank master 端口列表 ------
    new_data_parallel_master_port_list: list[int]
    # ------【DP】coord_store_port：协调存储服务端口 ------
    coord_store_port: int


# ------【DP/进程管理】ReconfigureRankType：DP 重配置时 rank 处理方式枚举 ------
class ReconfigureRankType(enum.IntEnum):
    """
    Rank type for reconfiguring distributed request.
    """

    # ------【DP】KEEP_CURRENT_RANK：保留当前 rank 继续服务 ------
    KEEP_CURRENT_RANK = -1
    # ------【DP】SHUTDOWN_CURRENT_RANK：关闭当前 rank ------
    SHUTDOWN_CURRENT_RANK = -2


# ------【进程管理】EngineStatusType：EngineCore 健康状态枚举 ------
class EngineStatusType(enum.IntEnum):
    # ------【进程管理】HEALTHY：健康 ------
    HEALTHY = 0
    # ------【进程管理】DEAD：已退出 ------
    DEAD = 1
    # ------【进程管理】UNHEALTHY：异常/不健康 ------
    UNHEALTHY = 2
