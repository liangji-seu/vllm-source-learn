# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from concurrent.futures import Future
from functools import cached_property
from typing import TYPE_CHECKING, Literal, TypeVar, overload

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.utils import KVOutputAggregator
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorHandshakeMetadata,
)
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.tasks import SupportedTask
from vllm.tracing import instrument
from vllm.utils.import_utils import resolve_obj_by_qualname
from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
from vllm.v1.engine import ReconfigureDistributedRequest
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec
from vllm.v1.outputs import DraftTokenIds, ModelRunnerOutput
from vllm.v1.worker.worker_base import CompilationTimes, WorkerBase

if TYPE_CHECKING:
    from vllm.distributed.kv_transfer.kv_connector.base import KVConnectorBase

logger = init_logger(__name__)

_R = TypeVar("_R")

FailureCallback = Callable[[], None]


class Executor(ABC):
    """
    === 类说明 ===
        继承: ABC
        职责: 执行器抽象基类。负责将 SchedulerOutput (调度任务) 分发到
              Worker 执行模型 forward，并返回 ModelRunnerOutput。
              封装了所有 Worker RPC 调用的统一入口 (collective_rpc)。

    === 子类 ===
        MultiprocExecutor      — 单机多卡, 每 GPU 一个 Worker 进程, MessageQueue 通信
        UniProcExecutor        — 单机单卡, 无 IPC, 直接调用
        RayDistributedExecutor — 多机多卡, Ray 编排
        ExecutorWithExternalLauncher — 外部启动器 (torchrun) 管理 Worker

    === 工厂方法 ===
        get_class(vllm_config) — @staticmethod, 根据配置选择合适的 Executor 子类

    === 核心方法 (对外接口) ===
        —— Worker 动作封装 (均通过 collective_rpc 广播) ——
            execute_model(scheduler_output) → ModelRunnerOutput  — 执行模型 forward
            sample_tokens(grammar_output)   → ModelRunnerOutput  — 采样 token
            execute_dummy_batch()                                — 空 batch 预热
            take_draft_token_ids()          → DraftTokenIds      — 获取 draft token
            initialize_from_config(kv_cache_configs)              — 初始化 KV cache
            compile_or_warm_up_model()                            — 编译/预热模型 + CUDA Graph capture
        —— RPC 基座 ——
            collective_rpc(method, ...) → list[Result] | Future   — (抽象) 向所有 Worker 广播调用
        —— 生命周期 ——
            shutdown()                                            — 关闭执行器
            check_health()                                        — (抽象) 健康检查

    === 核心成员属性 (由 __init__ 设置) ===
        vllm_config: VllmConfig               — 总配置 (包含下面所有子 config)
        model_config / cache_config / lora_config / load_config
        parallel_config / scheduler_config / device_config
        speculative_config / observability_config
        is_sleeping: bool                     — 是否处于休眠状态 (sleep mode)
        kv_output_aggregator                  — KV connector 的输出聚合器 (P/D 分离)
    """

    uses_ray: bool = False  # whether the executor uses Ray for orchestration.
    supports_pp: bool = False  # whether the executor supports PP

    # 这是一个静态的工厂方法
    @staticmethod
    def get_class(vllm_config: VllmConfig) -> type["Executor"]:
        # ------【进程管理】从并行配置读取执行器后端，决定用哪种进程/编排模型 ------
        executor_class: type[Executor]
        parallel_config = vllm_config.parallel_config
        distributed_executor_backend = parallel_config.distributed_executor_backend
        # distributed_executor_backend must be set in VllmConfig.__post_init__
        # ------【进程管理】直接传入 Executor 子类类型：显式指定执行器实现 ------
        if isinstance(distributed_executor_backend, type):
            if not issubclass(distributed_executor_backend, Executor):
                raise TypeError(
                    "distributed_executor_backend must be a subclass of "
                    f"Executor. Got {distributed_executor_backend}."
                )
            executor_class = distributed_executor_backend
        # ------【进程管理】Ray 后端：多机多卡由 Ray 编排 Worker 进程 ------
        elif distributed_executor_backend == "ray":
            if envs.VLLM_USE_RAY_V2_EXECUTOR_BACKEND:
                from vllm.v1.executor.ray_executor_v2 import RayExecutorV2

                executor_class = RayExecutorV2
            else:
                from vllm.v1.executor.ray_executor import RayDistributedExecutor

                executor_class = RayDistributedExecutor # 多机多卡的执行器类
        # ------【进程管理 + 异步 RPC】单机多卡：每卡一个 Worker 进程，MessageQueue 通信 ------
        elif distributed_executor_backend == "mp":
            from vllm.v1.executor.multiproc_executor import MultiprocExecutor

            executor_class = MultiprocExecutor # 单机多卡的执行器类，负责：创建管理多个worker, collective_rpc广播通信
        # ------【进程管理】单机单卡：无多进程、无 IPC，直接调用 Worker ------
        elif distributed_executor_backend == "uni":
            from vllm.v1.executor.uniproc_executor import UniProcExecutor

            executor_class = UniProcExecutor # 单机单卡的执行器类，无需多进程，无需IPC，最简单
        # ------【进程管理】外部启动器：由 torchrun/Slurm 等提前拉起 Worker，vLLM 只负责连接 ------
        elif distributed_executor_backend == "external_launcher":
            # TODO: make v1 scheduling deterministic
            # to support external launcher                # ray是自己管理，ray负责创建worker,跨节点管理
                                                          # external_launcher是 torchrun， 这个是用来连接已经存在的worker的，由外部系统创建好worker了
            executor_class = ExecutorWithExternalLauncher # vLLM 不负责启动 Worker 进程，而是交给外部启动器（例如 torchrun、Slurm）提前启动好
        # ------【进程管理】按字符串路径动态解析自定义执行器类（如插件扩展）──
        elif isinstance(distributed_executor_backend, str):
            executor_class = resolve_obj_by_qualname(distributed_executor_backend)
            if not issubclass(executor_class, Executor):
                raise TypeError(
                    "distributed_executor_backend must be a subclass of "
                    f"Executor. Got {executor_class}."
                )
        # ------【进程管理】无法识别的后端字符串：抛错提示配置错误 ------
        else:
            raise ValueError(
                f"Unknown distributed executor backend: {distributed_executor_backend}"
            )
        return executor_class

    @instrument(span_name="Executor init")
    def __init__(
        self,
        vllm_config: VllmConfig,
    ) -> None:
        # ------ 解包总配置：拆出模型/缓存/并行(TP/PP/DP)/投机/LoRA 等子配置供后续取用 ------
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.lora_config = vllm_config.lora_config
        self.load_config = vllm_config.load_config
        self.parallel_config = vllm_config.parallel_config
        self.scheduler_config = vllm_config.scheduler_config
        self.device_config = vllm_config.device_config
        self.speculative_config = vllm_config.speculative_config
        self.observability_config = vllm_config.observability_config
        # ------【进程管理】交由子类创建/启动 Worker 进程与通信通道 ------
        self._init_executor()
        # ------ sleep 状态位：记录是否休眠及已卸载模块（权重/KV cache）──
        self.is_sleeping = False
        self.sleeping_tags: set[str] = set()
        # ------【PD 分离】KV connector 输出聚合器：汇总 prefill/decode 分离传输的 KV ------
        self.kv_output_aggregator: KVOutputAggregator | None = None

    @abstractmethod
    def _init_executor(self) -> None:
        # ------【进程管理】抽象方法：由子类创建 Worker 进程并建立 RPC 通信通道 ------
        raise NotImplementedError

    def initialize_from_config(self, kv_cache_configs: list[KVCacheConfig]) -> None:
        """Initialize the KV caches on the underlying workers."""
        # ------【异步 RPC + 显存 profiling】广播 KV cache 配置，各 Worker 按显存规划初始化 KV cache ------
        self.collective_rpc("initialize_from_config", args=(kv_cache_configs,))

    def compile_or_warm_up_model(self) -> None:
        """Compile/warm up the model and capture cudagraphs on workers."""
        # ------【CUDA Graph】广播编译/预热命令，各 Worker 编译模型并捕获 CUDA Graph 回传耗时 ------
        compilation_times: list[CompilationTimes] = self.collective_rpc(
            "compile_or_warm_up_model"
        )
        # Propagate compilation time from workers back to the main process.
        # With TP>1, compilation happens in worker processes, so the main
        # process config is never updated. Use max across workers since they
        # compile in parallel.
        # ------【TP】取各 Worker 编译耗时最大值回写主进程配置，TP>1 时编译发生在 Worker ------
        if compilation_times:
            self.vllm_config.compilation_config.compilation_time = max(
                t.language_model for t in compilation_times
            )
            self.vllm_config.compilation_config.encoder_compilation_time = max(
                t.encoder for t in compilation_times
            )

    def register_failure_callback(self, callback: FailureCallback):  # noqa: B027
        """
        Register a function to be called if the executor enters a permanent
        failed state.
        """
        # ------【进程管理】失败回调注册钩子：默认空实现，供子类在 Worker 崩溃时触发 ------
        pass

    def determine_available_memory(self) -> list[int]:  # in bytes
        # ------【显存 profiling】广播探测各 Worker 可用显存，供 scheduler 规划 KV cache ------
        return self.collective_rpc("determine_available_memory")

    def get_kv_cache_specs(self) -> list[dict[str, KVCacheSpec]]:
        # ------【异步 RPC + 显存 profiling】广播到各 Worker 获取 KV cache 规格，用于显存规划 ------
        return self.collective_rpc("get_kv_cache_spec")

    # ------【异步 RPC】重载签名 1：non_block=False，阻塞等待所有 Worker 返回结果 ------
    @overload
    def collective_rpc(
        self,
        method: str | Callable[[WorkerBase], _R],
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict | None = None,
        non_block: Literal[False] = False,
    ) -> list[_R]:
        """
        Execute an RPC call on all workers.

        Args:
            method: Name of the worker method to execute, or a callable that
                is serialized and sent to all workers to execute.

                If the method is a callable, it should accept an additional
                `self` argument, in addition to the arguments passed in `args`
                and `kwargs`. The `self` argument will be the worker object.
            timeout: Maximum time in seconds to wait for execution. Raises a
                [`TimeoutError`][] on timeout. `None` means wait indefinitely.
            args: Positional arguments to pass to the worker method.
            kwargs: Keyword arguments to pass to the worker method.
            non_block: If `True`, returns a list of Futures instead of waiting
                for the results.

        Returns:
            A list containing the results from each worker.

        Note:
            It is recommended to use this API to only pass control messages,
            and set up data-plane communication to pass data.
        """
        pass

    # ------【异步 RPC】重载签名 2：non_block=True，返回 Future 列表，异步不等待 ------
    @overload
    def collective_rpc(
        self,
        method: str | Callable[[WorkerBase], _R],
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict | None = None,
        non_block: Literal[True] = True,
    ) -> Future[list[_R]]:
        pass

    # ------【异步 RPC + 进程管理】抽象实现：子类按各自通信机制把调用广播到所有 Worker ------
    @abstractmethod
    def collective_rpc(
        self, method, timeout=None, args=(), kwargs=None, non_block: bool = False
    ):
        raise NotImplementedError

    def get_kv_connector_handshake_metadata(
        self,
    ) -> list[dict[tuple[int, int], KVConnectorHandshakeMetadata]]:
        # ------【PD 分离】广播获取 KV connector 握手元数据，用于 P/D 实例间建立 KV 传输通道 ------
        return self.collective_rpc("get_kv_connector_handshake_metadata")

    # ------【异步 RPC】重载签名：non_block=False，阻塞式执行模型 forward ------
    @overload
    def execute_model(
        self, scheduler_output: SchedulerOutput, non_block: Literal[False] = False
    ) -> ModelRunnerOutput | None:
        pass

    # ------【异步 RPC】重载签名：non_block=True，异步执行并返回 Future ------
    @overload
    def execute_model(
        self, scheduler_output: SchedulerOutput, non_block: Literal[True] = True
    ) -> Future[ModelRunnerOutput | None]:
        pass

    # ------【异步 RPC】把调度输出广播到所有 Worker 执行模型 forward，返回首个 Worker 结果 ------
    def execute_model(
        self, scheduler_output: SchedulerOutput, non_block: bool = False
    ) -> ModelRunnerOutput | None | Future[ModelRunnerOutput | None]:
        output = self.collective_rpc(  # type: ignore[call-overload]
            "execute_model", args=(scheduler_output,), non_block=non_block
        )
        return output[0]

    # ------【异步 RPC + 结构化输出/grammar】重载签名：阻塞式采样 token（应用 grammar 约束）──
    @overload
    def sample_tokens(
        self, grammar_output: GrammarOutput | None, non_block: Literal[False] = False
    ) -> ModelRunnerOutput:
        pass

    # ------【异步 RPC + 结构化输出/grammar】重载签名：non_block=True 异步采样，返回 Future ------
    @overload
    def sample_tokens(
        self, grammar_output: GrammarOutput | None, non_block: Literal[True] = True
    ) -> Future[ModelRunnerOutput]:
        pass

    # ------【异步 RPC + 结构化输出/grammar】把 grammar 约束广播到所有 Worker 采样 token，返回首个结果 ------
    def sample_tokens(
        self, grammar_output: GrammarOutput | None, non_block: bool = False
    ) -> ModelRunnerOutput | Future[ModelRunnerOutput]:
        output = self.collective_rpc(  # type: ignore[call-overload]
            "sample_tokens", args=(grammar_output,), non_block=non_block
        )
        return output[0]

    def execute_dummy_batch(self) -> None:
        # ------【核心逻辑】空 batch 预热：触发一次 dummy 前向，完成内核与内存池初始化 ------
        self.collective_rpc("execute_dummy_batch")

    def take_draft_token_ids(self) -> DraftTokenIds | None:
        # ------【投机解码】广播取出各 Worker 缓存的 draft token，供验证阶段比对 ------
        output: list[DraftTokenIds] = self.collective_rpc("take_draft_token_ids")
        return output[0]

    def profile(self, is_start: bool = True, profile_prefix: str | None = None):
        # ------【核心逻辑】广播性能剖析起停命令，收集各 Worker 的 profile 数据 ------
        self.collective_rpc("profile", args=(is_start, profile_prefix))

    def save_sharded_state(
        self,
        path: str,
        pattern: str | None = None,
        max_size: int | None = None,
    ) -> None:
        # ------【权重传输】广播按分片保存状态到指定路径，各 Worker 各自落盘自己的分片 ------
        self.collective_rpc(
            "save_sharded_state",
            kwargs=dict(path=path, pattern=pattern, max_size=max_size),
        )

    @abstractmethod
    def check_health(self) -> None:
        """Checks if the executor is healthy. If not, it should raise an
        exception."""
        # ------【进程管理】抽象健康检查：子类检测 Worker 存活，异常时抛出 ------
        raise NotImplementedError

    def shutdown(self) -> None:
        """Shutdown the executor."""
        # ------【进程管理】广播 shutdown 命令，关闭所有 Worker 进程与通信通道 ------
        self.collective_rpc("shutdown")

    def init_kv_output_aggregator(self, connector: "KVConnectorBase") -> None:
        """Init KVOutputAggregator"""
        # ------【PD 分离】按 connector 构造 KV 输出聚合器，汇总 P/D 分离场景的 KV 输出 ------
        self.kv_output_aggregator = KVOutputAggregator.from_connector(
            connector, self.parallel_config.world_size
        )

    @cached_property  # Avoid unnecessary RPC calls
    def supported_tasks(self) -> tuple[SupportedTask, ...]:
        output: list[tuple[SupportedTask, ...]]
        # ------【异步 RPC】广播获取支持的任务类型，并用缓存属性避免重复 RPC ------
        output = self.collective_rpc("get_supported_tasks")
        return output[0]

    def add_lora(self, lora_request: LoRARequest) -> bool:
        assert lora_request.lora_int_id > 0, "lora_id must be greater than 0."
        # ------【LoRA】广播添加 LoRA adapter，all() 保证所有 Worker 成功才返回 True ------
        return all(self.collective_rpc("add_lora", args=(lora_request,)))

    def remove_lora(self, lora_id: int) -> bool:
        assert lora_id > 0, "lora_id must be greater than 0."
        # ------【LoRA】广播移除 LoRA adapter，所有 Worker 确认后才返回 True ------
        return all(self.collective_rpc("remove_lora", args=(lora_id,)))

    def pin_lora(self, lora_id: int) -> bool:
        assert lora_id > 0, "lora_id must be greater than 0."
        # ------【LoRA】广播常驻(pin) LoRA adapter，使其权重常驻显存不被换出 ------
        return all(self.collective_rpc("pin_lora", args=(lora_id,)))

    def list_loras(self) -> set[int]:
        # ------【LoRA + 异步 RPC】广播收集各 Worker 已加载的 LoRA id 集合 ------
        sets: list[set[int]] = self.collective_rpc("list_loras")
        # ------【TP】校验各 Worker 的 LoRA 列表一致，TP 下各卡需加载同一批 LoRA ------
        for s in sets:
            assert s == sets[0], "All workers should have the same LORAs."
        return sets[0]

    def reset_mm_cache(self) -> None:
        """Reset the multi-modal cache in each worker."""
        # ------【核心逻辑】广播重置多模态缓存，清空各 Worker 缓存的多模态输入特征 ------
        self.collective_rpc("reset_mm_cache")

    def reset_encoder_cache(self) -> None:
        """Reset the encoder cache in each worker to clear cached encoder outputs."""
        # ------【核心逻辑】广播重置 encoder 缓存，清空各 Worker 缓存的 encoder 输出 ------
        self.collective_rpc("reset_encoder_cache")

    def sleep(self, level: int = 1):
        # ------【显存 profiling】已休眠则直接返回，避免重复卸载权重与 KV cache ------
        if self.is_sleeping:
            logger.warning("Executor is already sleeping.")
            return
        # ------【显存 profiling】记录开始时刻，用于统计休眠耗时 ------
        time_before_sleep = time.perf_counter()
        # ------【异步 RPC + 显存 profiling】广播 sleep，各 Worker 卸载权重与 KV cache 释放显存 ------
        self.collective_rpc("sleep", kwargs=dict(level=level))
        time_after_sleep = time.perf_counter()
        # ------【显存 profiling】标记已休眠并记录被卸载的模块(权重/KV cache) ------
        self.sleeping_tags = {"weights", "kv_cache"}
        self.is_sleeping = True
        logger.info(
            "It took %.6f seconds to fall asleep.", time_after_sleep - time_before_sleep
        )

    def wake_up(self, tags: list[str] | None = None):
        # ------【显存 profiling】未休眠则直接返回，无需唤醒 ------
        if not self.is_sleeping:
            logger.warning("Executor is not sleeping.")
            return
        # ------【显存 profiling】校验请求唤醒的 tag 是否在休眠列表中，防止唤醒未休眠模块 ------
        if tags:
            for tag in tags:
                if tag not in self.sleeping_tags:
                    logger.warning(
                        "Tag %s is not in sleeping tags %s", tag, self.sleeping_tags
                    )
                    return
        # ------【显存 profiling】记录开始时刻，用于统计唤醒耗时 ------
        time_before_wakeup = time.perf_counter()
        # ------【异步 RPC + 显存 profiling】广播 wake_up，各 Worker 按 tag 重载权重/KV cache ------
        self.collective_rpc("wake_up", kwargs=dict(tags=tags))
        time_after_wakeup = time.perf_counter()
        logger.info(
            "It took %.6f seconds to wake up tags %s.",
            time_after_wakeup - time_before_wakeup,
            tags if tags is not None else self.sleeping_tags,
        )
        # ------【显存 profiling】从休眠列表移除已唤醒的 tag；全部唤醒后清除休眠状态 ------
        if tags:
            for tag in tags:
                self.sleeping_tags.remove(tag)
        else:
            self.sleeping_tags.clear()
        if not self.sleeping_tags:
            self.is_sleeping = False

    def reinitialize_distributed(
        self, reconfig_request: ReconfigureDistributedRequest
    ) -> None:
        # ------【进程管理】分布式拓扑重配置入口：默认不支持，子类按需实现 ------
        raise NotImplementedError

    @classmethod
    def supports_async_scheduling(cls) -> bool:
        """
        Whether the executor supports async scheduling.
        """
        # ------【异步 RPC】查询执行器是否支持异步调度，默认 False ------
        return False


from vllm.v1.executor.uniproc_executor import (  # noqa: E402
    ExecutorWithExternalLauncher as _ExecutorWithExternalLauncher,
)
from vllm.v1.executor.uniproc_executor import (  # noqa: E402
    UniProcExecutor as _UniProcExecutor,
)

# For backwards compatibility.
UniProcExecutor = _UniProcExecutor
ExecutorWithExternalLauncher = _ExecutorWithExternalLauncher
