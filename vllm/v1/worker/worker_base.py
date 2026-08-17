# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, NamedTuple, TypeVar

import torch
import torch.nn as nn

import vllm.ir
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.tracing import instrument
from vllm.utils.import_utils import resolve_obj_by_qualname
from vllm.utils.system_utils import update_environment_variables
from vllm.v1.kv_cache_interface import KVCacheSpec

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
    from vllm.v1.outputs import AsyncModelRunnerOutput, ModelRunnerOutput
else:
    SchedulerOutput = object
    GrammarOutput = object
    AsyncModelRunnerOutput = object
    ModelRunnerOutput = object

logger = init_logger(__name__)

_R = TypeVar("_R")


# ------【核心逻辑】编译耗时记录结构：语言模型与编码器各自的编译/预热耗时（秒） ------
class CompilationTimes(NamedTuple):
    language_model: float
    encoder: float

# 定义一个worker类该有的接口
class WorkerBase:
    """Worker interface that allows vLLM to cleanly separate implementations for
    different hardware. Also abstracts control plane communication, e.g., to
    communicate request metadata to other workers.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = False,
    ) -> None:
        """
        Initialize common worker components.

        Args:
            vllm_config: Complete vLLM configuration
            local_rank: Local device index
            rank: Global rank in distributed setup
            distributed_init_method: Distributed initialization method
            is_driver_worker: Whether this worker handles driver
                responsibilities
        """
        ###########################################################################
        # 1. 先保存配置文件
        ###########################################################################
        # ------【核心逻辑】把 vllm_config 的各子配置拆解保存为快捷引用，方便后续按需访问 ------
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
        self.kv_transfer_config = vllm_config.kv_transfer_config
        self.compilation_config = vllm_config.compilation_config

        # ------【核心逻辑】获取并缓存当前硬件平台对象，后续按设备类型分发到不同实现 ------
        from vllm.platforms import current_platform

        self.current_platform = current_platform

        # ------【TP/DP】记录本地 rank、全局 rank 与分布式初始化方法，供 NCCL 进程组初始化使用 ------
        self.parallel_config.rank = rank
        self.local_rank = local_rank
        self.rank = rank
        self.distributed_init_method = distributed_init_method
        self.is_driver_worker = is_driver_worker

        # ------【核心逻辑】设备与模型运行器占位，延迟到 init_device/load_model 阶段再真正填充 ------
        # Device and model state
        self.device: torch.device | None = None
        self.model_runner: nn.Module | None = None

        # ------【核心逻辑】设置 IR 算子优先级与 torch-wrap 的进程级默认值，worker 生命周期内保持不变 ------
        # IR op priority and torch-wrap state are constant for the worker's
        # lifetime.
        vllm_config.kernel_config.ir_op_priority.set_default()
        vllm.ir.set_default_torch_wrap(
            vllm_config.compilation_config.ir_enable_torch_wrap
        )

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        """Get specifications for KV cache implementation."""
        raise NotImplementedError

    def compile_or_warm_up_model(self) -> CompilationTimes:
        """Prepare model for execution through compilation/warmup.

        Returns:
            Compilation times (language_model, encoder) in seconds.
        """
        raise NotImplementedError

    def check_health(self) -> None:
        """Basic health check (override for device-specific checks)."""
        return

    def init_device(self) -> None:
        """Initialize device state, such as loading the model or other on-device
        memory allocations.
        """
        raise NotImplementedError

    def reset_mm_cache(self) -> None:
        # ------【核心逻辑】若模型运行器实现了 reset_mm_cache，则调用它清空多模态缓存 ------
        reset_fn = getattr(self.model_runner, "reset_mm_cache", None)
        if callable(reset_fn):
            reset_fn()

    def get_model(self) -> nn.Module:
        raise NotImplementedError

    def apply_model(self, fn: Callable[[nn.Module], _R]) -> _R:
        """Apply a function on the model inside this worker."""
        # ------【核心逻辑】把外部函数应用到本 worker 的模型上，供上层统一在模型上执行操作 ------
        return fn(self.get_model())

    def get_model_inspection(self) -> str:
        """Return a transformers-style hierarchical view of the model."""
        # ------【核心逻辑】把模型结构格式化为 transformers 风格的可读视图，便于调试与检查 ------
        from vllm.model_inspection import format_model_inspection

        return format_model_inspection(self.get_model())

    def load_model(self, *, load_dummy_weights: bool = False) -> None:
        """Load model onto target device."""
        raise NotImplementedError

    def execute_model(
        self, scheduler_output: SchedulerOutput
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput | None:
        """If this method returns None, sample_tokens should be called immediately after
        to obtain the ModelRunnerOutput.

        Note that this design may be changed in future if/when structured outputs
        parallelism is re-architected.
        """
        raise NotImplementedError

    def sample_tokens(
        self, grammar_output: GrammarOutput
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput:
        """Should be called immediately after execute_model iff it returned None."""
        raise NotImplementedError

    def get_cache_block_size_bytes(self) -> int:
        """Return the size of a single cache block, in bytes. Used in
        speculative decoding.
        """
        raise NotImplementedError

    def add_lora(self, lora_request: LoRARequest) -> bool:
        raise NotImplementedError

    def remove_lora(self, lora_id: int) -> bool:
        raise NotImplementedError

    def pin_lora(self, lora_id: int) -> bool:
        raise NotImplementedError

    def list_loras(self) -> set[int]:
        raise NotImplementedError

    @property
    def vocab_size(self) -> int:
        """Get vocabulary size from model configuration."""
        # ------【核心逻辑】从模型配置读取词表大小，供采样与输出层等下游使用 ------
        return self.model_config.get_vocab_size()

    def shutdown(self) -> None:
        """Clean up resources held by the worker."""
        return


class WorkerWrapperBase:
    """
    This class represents one process in an executor/engine. It is responsible
    for lazily initializing the worker and handling the worker's lifecycle.
    We first instantiate the WorkerWrapper, which remembers the worker module
    and class name. Then, when we call `update_environment_variables`, and the
    real initialization happens in `init_worker`.
    """

    def __init__(
        self,
        rpc_rank: int = 0,
        global_rank: int | None = None,
    ) -> None:
        """
        Initialize the worker wrapper with the given vllm_config and rpc_rank.
        Note: rpc_rank is the rank of the worker in the executor. In most cases,
        it is also the rank of the worker in the distributed group. However,
        when multiple executors work together, they can be different.
        e.g. in the case of SPMD-style offline inference with TP=2,
        users can launch 2 engines/executors, each with only 1 worker.
        All workers have rpc_rank=0, but they have different ranks in the TP
        group.
        """
        # ------【异步 RPC + TP】rpc_rank 是 worker 在 executor 中的通信 rank，global_rank 是分布式组(TP)中的全局 rank ------
        self.rpc_rank: int = rpc_rank
        self.global_rank: int = self.rpc_rank if global_rank is None else global_rank

        # ------【异步 RPC】worker 与 vllm_config 延迟到 init_worker 才真正初始化，此处仅作类型声明 ------
        # Initialized after init_worker is called
        self.worker: WorkerBase
        self.vllm_config: VllmConfig

    def shutdown(self) -> None:
        # ------【进程管理】惰性初始化下 worker 可能尚未创建，仅在实际存在时才调用其 shutdown 释放资源 ------
        if self.worker is not None:
            self.worker.shutdown()

    def update_environment_variables(
        self,
        envs_list: list[dict[str, str]],
    ) -> None:
        # ------【异步 RPC】按 rpc_rank 取本 worker 的环境变量并写入当前进程，用于多进程差异化配置 ------
        envs = envs_list[self.rpc_rank]
        update_environment_variables(envs)








    @instrument(span_name="Worker init")
    def init_worker(self, all_kwargs: list[dict[str, Any]]) -> None:
        """
        Here we inject some common logic before initializing the worker.
        Arguments are passed to the worker class constructor.
        """
        # ------【异步 RPC】按 rpc_rank 从引擎广播的 all_kwargs 中取出本 worker 的构造参数与 vllm_config ------
        kwargs = all_kwargs[self.rpc_rank]

        vllm_config: VllmConfig | None = kwargs.get("vllm_config")
        assert vllm_config is not None, (
            "vllm_config is required to initialize the worker"
        )
        self.vllm_config = vllm_config

        # ------【通用初始化】开启函数调用追踪并加载通用插件，为构造 worker 做准备（与并行策略无关）──
        vllm_config.enable_trace_function_call_for_thread()

        from vllm.plugins import load_general_plugins

        load_general_plugins()

        # ------【通用初始化】按限定名解析出真正的 Worker 子类（设备特定实现，如 GPU/TPU/CPU）──
        parallel_config = vllm_config.parallel_config
        if isinstance(parallel_config.worker_cls, str):
            worker_class: type[WorkerBase] = resolve_obj_by_qualname( # 解析出真正的Worker子类
                parallel_config.worker_cls
            )
        else:
            raise ValueError(
                "passing worker_cls is no longer supported. "
                "Please pass keep the class in a separate module "
                "and pass the qualified name of the class as a string."
            )

        # ------【异步 RPC】动态继承 worker_extension_cls，扩展 collective_rpc 可调用的方法 ------
        if parallel_config.worker_extension_cls:
            worker_extension_cls = resolve_obj_by_qualname(
                parallel_config.worker_extension_cls
            )
            extended_calls = []
            if worker_extension_cls not in worker_class.__bases__:
                # check any conflicts between worker and worker_extension_cls
                for attr in dir(worker_extension_cls):
                    if attr.startswith("__"):
                        continue
                    assert not hasattr(worker_class, attr), (
                        f"Worker class {worker_class} already has an attribute"
                        f" {attr}, which conflicts with the worker"
                        f" extension class {worker_extension_cls}."
                    )
                    if callable(getattr(worker_extension_cls, attr)):
                        extended_calls.append(attr)
                # dynamically inherit the worker extension class
                worker_class.__bases__ = worker_class.__bases__ + (
                    worker_extension_cls,
                )
                logger.info(
                    "Injected %s into %s for extended collective_rpc calls %s",
                    worker_extension_cls,
                    worker_class,
                    extended_calls,
                )

        # ------【进程管理】取出分配给本 worker 的物理 GPU 编号并覆盖进 parallel_config（多副本/容器绑定显存用）──
        assigned_physical_gpu_ids = kwargs.pop("assigned_physical_gpu_ids", None)
        if assigned_physical_gpu_ids is not None:
            vllm_config.parallel_config.assigned_physical_gpu_ids = (
                assigned_physical_gpu_ids
            )

        # ------【进程管理】取共享内存锁，为多模态处理器 shm 缓存建立跨进程接收缓存 ------
        shared_worker_lock = kwargs.pop("shared_worker_lock", None)
        if shared_worker_lock is None:
            msg = (
                "Missing `shared_worker_lock` argument from executor. "
                "This argument is needed for mm_processor_cache_type='shm'."
            )

            mm_config = vllm_config.model_config.multimodal_config
            if mm_config and mm_config.mm_processor_cache_type == "shm":
                raise ValueError(msg)
            else:
                logger.warning_once(msg)

            self.mm_receiver_cache = None
        else:
            self.mm_receiver_cache = (
                MULTIMODAL_REGISTRY.worker_receiver_cache_from_config(
                    vllm_config,
                    shared_worker_lock,
                )
            )

        # ------【异步 RPC】在当前进程注入 vllm_config 后实例化真正的 Worker 子类，完成延迟初始化 ------
        with set_current_vllm_config(self.vllm_config):
            # To make vLLM config available during worker initialization
            self.worker = worker_class(**kwargs) # 在这里构建的Worker子类的实例


















    def initialize_from_config(self, kv_cache_configs: list[Any]) -> None:
        # ------【TP】按 global_rank 取本 worker 对应的 KV cache 配置（张量并行各 rank 分片不同） ------
        kv_cache_config = kv_cache_configs[self.global_rank]
        assert self.vllm_config is not None
        # ------【核心逻辑】在进程级 vllm_config 上下文中把 KV cache 配置下发到 worker 完成初始化 ------
        with set_current_vllm_config(self.vllm_config):
            self.worker.initialize_from_config(kv_cache_config)  # type: ignore

    def init_device(self):
        assert self.vllm_config is not None
        # ------【显存 profiling】在进程级 vllm_config 上下文中触发 worker 的设备初始化与显存分配 ------
        with set_current_vllm_config(self.vllm_config):
            # To make vLLM config available during device initialization
            self.worker.init_device()  # type: ignore

    def __getattr__(self, attr: str):
        # ------【异步 RPC】把本 wrapper 未定义的属性透明转发给真正的 Worker 实现，实现 RPC 方法的动态代理 ------
        return getattr(self.worker, attr)

    def _apply_mm_cache(self, scheduler_output: SchedulerOutput) -> None:
        # ------【进程管理】未配置 shm 缓存时 receiver_cache 为 None，直接跳过不处理多模态特征 ------
        mm_cache = self.mm_receiver_cache
        if mm_cache is None:
            return

        # ------【进程管理】对新调度的请求用接收缓存替换/更新其多模态特征，避免重复编码 ------
        for req_data in scheduler_output.scheduled_new_reqs:
            req_data.mm_features = mm_cache.get_and_update_features(
                req_data.mm_features
            )

    def execute_model(
        self, scheduler_output: SchedulerOutput
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput | None:
        # ------【进程管理】执行前先应用多模态缓存，把特征填充到调度输出中 ------
        self._apply_mm_cache(scheduler_output)

        # ------【核心逻辑】把调度输出转发给真正的 worker 执行模型前向计算 ------
        return self.worker.execute_model(scheduler_output)

    def reset_mm_cache(self) -> None:
        # ------【进程管理】清空本进程的 shm 接收缓存，使下一轮重新获取多模态特征 ------
        mm_receiver_cache = self.mm_receiver_cache
        if mm_receiver_cache is not None:
            mm_receiver_cache.clear_cache()

        # ------【核心逻辑】同时触发 worker 内部的多模态缓存清理，保证两端缓存一致 ------
        self.worker.reset_mm_cache()
