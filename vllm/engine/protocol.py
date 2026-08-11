# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from vllm.config import ModelConfig, VllmConfig
from vllm.distributed.weight_transfer.base import (
    WeightTransferInitRequest,
    WeightTransferUpdateRequest,
)
from vllm.inputs import EngineInput, PromptType
from vllm.lora.request import LoRARequest
from vllm.outputs import PoolingRequestOutput, RequestOutput
from vllm.pooling_params import PoolingParams
from vllm.renderers import BaseRenderer
from vllm.sampling_params import SamplingParams
from vllm.tasks import SupportedTask
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.input_processor import InputProcessor
from vllm.v1.fault_tolerance.utils import FaultToleranceRequest, FaultToleranceResult

if TYPE_CHECKING:
    from vllm.v1.engine import PauseMode


@dataclass
class StreamingInput:
    """Input data for a streaming generation request.

    This is used with generate() to support multi-turn streaming sessions
    where inputs are provided via an async generator.
    """

    prompt: EngineInput
    sampling_params: SamplingParams | None = None


class EngineClient(ABC):
    """
    === 类说明 ===
        继承: ABC
        职责: vLLM 引擎前端的抽象协议基类。定义所有引擎前端（AsyncLLM / LLM）
              必须实现的完整接口契约，不包含任何具体实现。

    === [抽象] 属性 (4个) ===
        is_running            — 引擎输出处理循环是否在运行
        is_stopped            — 引擎是否已停止（出错）
        errored               — 引擎是否处于错误状态
        dead_error            — 返回引擎死亡时应抛出的异常实例

    === @abstractmethod 抽象方法 (20个) ===
        generate()                      — 核心生成接口，返回 RequestOutput 异步生成器
        encode()                        — 编码/pooling 接口，返回 PoolingRequestOutput 异步生成器
        abort()                         — 中止一个或多个请求
        notify_kv_transfer_request_rejected() — 通知 KV 传输请求被拒，触发连接器侧清理
        is_tracing_enabled()            — 检查是否启用了 OTLP 链路追踪
        do_log_stats()                  — 触发一次统计日志记录
        check_health()                  — 健康检查，不健康时抛出异常
        start_profile() / stop_profile() — 启停性能分析
        reset_mm_cache()                — 重置多模态缓存
        reset_encoder_cache()           — 重置编码器缓存
        reset_prefix_cache()            — 重置前缀缓存
        sleep() / wake_up() / is_sleeping() — 引擎休眠/唤醒
        add_lora()                      — 加载 LoRA 适配器
        pause_generation() / resume_generation() / is_paused() — 暂停/恢复生成
        shutdown()                      — 关闭引擎并清理资源

    === 非抽象方法（默认抛 NotImplementedError, 12个） ===
        scale_elastic_ep()              — 弹性专家并行扩缩容
        collective_rpc()                — 对所有引擎进程的集体 RPC 调用
        handle_fault() / get_status()   — 容错指令与状态查询
        get_supported_tasks()           — 获取模型支持的任务类型
        init_weight_transfer_engine()   — 初始化 RL 训练权重传输引擎
        start_weight_update() / start_draft_weight_update() — 开始权重更新
        update_weights() / finish_weight_update() — 批量权重更新（RL 训练用）
        update_weight_version() / get_weight_version() — 权重版本管理

    === [新增] 核心成员属性 ===
        —— 接口声明（子类需赋值） ——
            vllm_config: VllmConfig         — 全局配置
            model_config: ModelConfig       — 模型配置
            renderer: BaseRenderer          — 渲染器（tokenizer + 多模态处理器）
            input_processor: InputProcessor — 输入处理器（EngineInput → EngineCoreRequest）
    """

    vllm_config: VllmConfig
    model_config: ModelConfig
    renderer: BaseRenderer
    input_processor: InputProcessor

    @property
    @abstractmethod
    def is_running(self) -> bool: ...

    @property
    @abstractmethod
    def is_stopped(self) -> bool: ...

    @property
    @abstractmethod
    def errored(self) -> bool: ...

    @property
    @abstractmethod
    def dead_error(self) -> BaseException: ...

    @abstractmethod
    def generate(
        self,
        prompt: EngineCoreRequest
        | PromptType
        | EngineInput
        | AsyncGenerator[StreamingInput, None],
        sampling_params: SamplingParams,
        request_id: str,
        *,
        prompt_text: str | None = None,
        lora_request: LoRARequest | None = None,
        tokenization_kwargs: dict[str, Any] | None = None,
        trace_headers: Mapping[str, str] | None = None,
        priority: int = 0,
        data_parallel_rank: int | None = None,
        session_id: str | None = None,
        reasoning_ended: bool | None = None,
        reasoning_parser_kwargs: dict[str, Any] | None = None,
    ) -> AsyncGenerator[RequestOutput, None]:
        """Generate outputs for a request."""
        ...

    @abstractmethod
    def encode(
        self,
        prompt: PromptType | EngineInput,
        pooling_params: PoolingParams,
        request_id: str,
        lora_request: LoRARequest | None = None,
        trace_headers: Mapping[str, str] | None = None,
        priority: int = 0,
        tokenization_kwargs: dict[str, Any] | None = None,
        reasoning_ended: bool | None = None,
    ) -> AsyncGenerator[PoolingRequestOutput, None]:
        """Generate outputs for a request from a pooling model."""
        ...

    @abstractmethod
    async def abort(self, request_id: str | Iterable[str]) -> None:
        """Abort a request.

        Args:
            request_id: The unique id of the request,
                        or an iterable of such ids.
        """
        ...

    @abstractmethod
    async def notify_kv_transfer_request_rejected(
        self,
        request_id: str,
        kv_transfer_params: dict[str, Any],
        *,
        data_parallel_rank: int | None = None,
    ) -> None:
        """Notify the engine that a KV-transfer request was rejected before
        engine admission, so connector-side cleanup can run (e.g. free
        prefill blocks pinned on the P node).
        """
        ...

    @abstractmethod
    async def is_tracing_enabled(self) -> bool: ...

    @abstractmethod
    async def do_log_stats(self) -> None: ...

    @abstractmethod
    async def check_health(self) -> None:
        """Raise if unhealthy"""
        ...

    @abstractmethod
    async def start_profile(self) -> None:
        """Start profiling the engine"""
        ...

    @abstractmethod
    async def stop_profile(self) -> None:
        """Stop profiling the engine"""
        ...

    @abstractmethod
    async def reset_mm_cache(self) -> None:
        """Reset the multi-modal cache"""
        ...

    @abstractmethod
    async def reset_encoder_cache(self) -> None:
        """Reset the encoder cache"""
        ...

    @abstractmethod
    async def reset_prefix_cache(
        self, reset_running_requests: bool = False, reset_connector: bool = False
    ) -> bool:
        """Reset the prefix cache and optionally any configured connector cache"""
        ...

    @abstractmethod
    async def sleep(self, level: int = 1, mode: "PauseMode" = "abort") -> None:
        """Sleep the engine"""
        ...

    @abstractmethod
    async def wake_up(self, tags: list[str] | None = None) -> None:
        """Wake up the engine"""
        ...

    @abstractmethod
    async def is_sleeping(self) -> bool:
        """Check whether the engine is sleeping"""
        ...

    @abstractmethod
    async def add_lora(self, lora_request: LoRARequest) -> bool:
        """Load a new LoRA adapter into the engine for future requests."""
        ...

    @abstractmethod
    async def pause_generation(
        self,
        *,
        mode: "PauseMode" = "abort",
        wait_for_inflight_requests: bool = False,
        clear_cache: bool = True,
    ) -> None:
        """Pause new generation/encoding requests.

        Args:
            mode: How to handle in-flight requests:
                - ``"abort"``: Abort all in-flight requests immediately
                  and return partial results with "abort" reason (default).
                - ``"wait"``: Wait for in-flight requests to complete.
                - ``"keep"``: Freeze requests in queue; they resume on
                  :meth:`resume_generation`.
            wait_for_inflight_requests: DEPRECATED. Use ``mode="wait"`` instead.
            clear_cache: DEPRECATED. Whether to clear KV and prefix caches
                after draining.
        """
        ...

    @abstractmethod
    async def resume_generation(self) -> None:
        """Resume accepting generation/encoding requests."""
        ...

    @abstractmethod
    async def is_paused(self) -> bool:
        """Return whether the engine is currently paused."""
        ...

    @abstractmethod
    def shutdown(self, timeout: float | None = None) -> None:
        """Shutdown the engine with optional timeout."""
        ...

    async def scale_elastic_ep(
        self, new_data_parallel_size: int, drain_timeout: int = 300
    ) -> None:
        """Scale the engine"""
        raise NotImplementedError

    async def collective_rpc(
        self,
        method: str,
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict | None = None,
    ):
        """Perform a collective RPC call to the given path."""
        raise NotImplementedError

    async def handle_fault(
        self, fault_tolerance_request: FaultToleranceRequest
    ) -> FaultToleranceResult:
        """send fault tolerance instruction to the engine"""
        raise NotImplementedError

    async def get_status(self):
        """Get fault tolerance status of all engines."""
        raise NotImplementedError

    async def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        """Get supported tasks"""
        raise NotImplementedError

    async def init_weight_transfer_engine(
        self, init_request: WeightTransferInitRequest
    ) -> None:
        """Initialize weight transfer for RL training."""
        raise NotImplementedError

    async def start_weight_update(self) -> None:
        """Start a new weight update."""
        raise NotImplementedError

    async def start_draft_weight_update(self) -> None:
        """Start a new weight update targeting the speculative draft model."""
        raise NotImplementedError

    async def update_weights(self, request: WeightTransferUpdateRequest) -> None:
        """Batched weight update for RL training."""
        raise NotImplementedError

    async def finish_weight_update(self, weight_version: str | None = None) -> None:
        """Finish the weight update and set its version if provided."""
        raise NotImplementedError

    async def update_weight_version(self, new_version: str) -> None:
        """Set the weight version without updating weights."""
        raise NotImplementedError

    async def get_weight_version(self) -> str:
        """Return the latest committed weight version."""
        raise NotImplementedError
