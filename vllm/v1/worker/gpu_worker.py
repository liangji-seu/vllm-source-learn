# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A GPU worker class."""

import gc
import os
import time
from collections.abc import Callable
from contextlib import AbstractContextManager, contextmanager, nullcontext
from datetime import timedelta
from types import NoneType
from typing import TYPE_CHECKING, Any

import numpy as np
import regex as re
import torch
import torch.nn as nn

import vllm.envs as envs
from vllm.config import CUDAGraphMode, VllmConfig, set_current_vllm_config
from vllm.config.compilation import CompilationMode
from vllm.device_allocator import get_mem_allocator_instance
from vllm.distributed import (
    ensure_model_parallel_initialized,
    init_distributed_environment,
    set_custom_all_reduce,
)
from vllm.distributed.ec_transfer import (
    ensure_ec_transfer_initialized,
    ensure_ec_transfer_shutdown,
)
from vllm.distributed.eplb.eplb_utils import override_envs_for_eplb
from vllm.distributed.kv_transfer import (
    ensure_kv_transfer_initialized,
    ensure_kv_transfer_shutdown,
    get_kv_transfer_group,
    has_kv_transfer_group,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorHandshakeMetadata,
)
from vllm.distributed.parallel_state import (
    Handle,
    checkpoint_prepare_distributed_state,
    checkpoint_restore_distributed_state,
    get_pp_group,
    get_tp_group,
)
from vllm.distributed.weight_transfer import (
    WeightTransferEngine,
    WeightTransferEngineFactory,
)
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.model_executor.warmup.kernel_warmup import kernel_warmup
from vllm.multimodal.gpu_ipc_memory import reserve_mm_ipc_gpu_memory
from vllm.platforms import current_platform
from vllm.profiler.wrapper import CudaProfilerWrapper, TorchProfilerWrapper
from vllm.sequence import IntermediateTensors
from vllm.tasks import SupportedTask
from vllm.tracing import instrument
from vllm.utils.gc_utils import freeze_gc_heap, maybe_attach_gc_debug_callback
from vllm.utils.gpu_sync_debug import enable_gpu_sync_check, with_gpu_sync_check
from vllm.utils.mem_constants import GiB_bytes
from vllm.utils.mem_utils import MemorySnapshot, format_gib, memory_profiling
from vllm.utils.torch_utils import set_random_seed, set_torch_threads_for_runtime
from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec
from vllm.v1.outputs import (
    AsyncModelRunnerOutput,
    DraftTokenIds,
    ModelRunnerOutput,
)
from vllm.v1.utils import compute_iteration_details, report_usage_stats
from vllm.v1.worker.sentinel.gpu_worker_sentinel import WorkerSentinel
from vllm.v1.worker.startup_plan import (
    maybe_apply_startup_plan,
    maybe_save_startup_plan,
)
from vllm.v1.worker.utils import is_residual_scattered_for_sp
from vllm.v1.worker.worker_base import CompilationTimes, WorkerBase
from vllm.v1.worker.workspace import init_workspace_manager

from ...model_executor.model_loader import TensorizerLoader
from .gpu.warmup import warmup_kernels
from .utils import request_memory

logger = init_logger(__name__)

if TYPE_CHECKING:
    from vllm.device_allocator.sleep_mode_backend import SleepModeBackend
    from vllm.model_executor.model_loader.tensorizer import TensorizerConfig
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner


class AsyncIntermediateTensors(IntermediateTensors):
    """IntermediateTensors with lazy comm synchronization"""

    def __init__(
        self,
        tensors: dict[str, torch.Tensor],
        comm_handles: list[Handle] | None = None,
        comm_postprocess: list[Callable[[], None]] | None = None,
    ) -> None:
        # ------【PP + 异步 RPC】暂存上游传来的中间张量与未完成的通信句柄，先不触发同步 ------
        super().__init__(tensors)
        self._comm_handles = comm_handles
        self._comm_postprocess = comm_postprocess
        self._comm_waited = False

    def wait_for_comm(self) -> None:
        # ------【PP + 异步 RPC】惰性同步：等 recv/all-gather 句柄完成并跑后处理，把通信重叠进计算 ------
        if self._comm_waited:
            return
        if self._comm_handles:
            for handle in self._comm_handles:
                handle.wait()
        if self._comm_postprocess:
            for fn in self._comm_postprocess:
                fn()
        self._comm_waited = True

    def __getattribute__(self, name: str):
        # ------【PP】访问 .tensors 前强制等待通信完成，保证下游拿到已就绪的中间张量 ------
        # ensure `.tensors` is ready before use
        if name == "tensors" and not object.__getattribute__(self, "_comm_waited"):
            object.__getattribute__(self, "wait_for_comm")()
        return object.__getattribute__(self, name)

# 一个真正的GPU的worker类的实现
class Worker(WorkerBase):
    # 构造真正的Worker类，专门用来负责干活，和Executor是WorkerProc来负责的
    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = False,
    ):
        # ------【进程管理】先走父类 WorkerBase 构造，把 rank/local_rank/init_method 等分布式身份保存好 ------
        ##########################################################################################
        # 1. 先构建Worker功能实例
        ##########################################################################################
        super().__init__(
            vllm_config=vllm_config,
            local_rank=local_rank,
            rank=rank,
            distributed_init_method=distributed_init_method,
            is_driver_worker=is_driver_worker,
        )

        # ------【CUDA Graph】按环境变量设定 float32 matmul 精度，影响图捕获/回放时算子精度 ------
        # configure float32 matmul precision according to vLLM env.
        precision = envs.VLLM_FLOAT32_MATMUL_PRECISION
        torch.set_float32_matmul_precision(precision)

        # ------【EP/EPLB】创建弹性 EP 执行器，用于运行时动态扩缩专家并行规模 ------
        from vllm.distributed.elastic_ep.elastic_execute import ElasticEPScalingExecutor

        self.elastic_ep_executor = ElasticEPScalingExecutor(self)
        # ------【进程管理】容错模式下挂一个哨兵，接收外部控制命令（如离线/热切换）──
        self.worker_sentinel: WorkerSentinel | None = None
        if self.parallel_config.enable_fault_tolerance:
            self.worker_sentinel = WorkerSentinel(worker=self)
        # ------【显存 profiling】睡眠/唤醒时把权重与 draft 权重缓冲区卸载到 CPU 再恢复，用于省显存 ------
        # Buffers saved before sleep
        self._sleep_saved_buffers: dict[str, torch.Tensor] = {}
        self._sleep_saved_draft_buffers: dict[str, torch.Tensor] = {}

        # ------【异步 RPC】权重传输引擎延迟到 load_model 再建（需模型引用），用于训练侧热更新权重 ------
        # Weight transfer engine is created in `load_model` once the model
        # is available, since the engine needs a reference to the model.
        self.weight_transfer_engine: WeightTransferEngine | None = None
        self._weight_update_active = False
        self._weight_update_is_draft = False

        # ------【核心逻辑】预置 torch/cuda profiler 状态，实际包装器在 profile() 首次 start 时才惰性创建 ------
        # Torch/CUDA profiler. Enabled and configured through profiler_config.
        # Profiler wrapper is created lazily in profile() when start is called,
        # so we have all the information needed for proper trace naming.
        self.profiler: Any | None = None
        self.profiler_config = vllm_config.profiler_config

        # ------【核心逻辑】只校验 profiler 类型是否合法，暂不实例化包装器 ------
        # Only validate profiler config is valid, don't instantiate yet
        if self.profiler_config.profiler not in ("torch", "cuda", None):
            raise ValueError(f"Unknown profiler type: {self.profiler_config.profiler}")

        # ------【核心逻辑】记录用 V1 还是 V2 的 model runner 实现，后续按此分支构造 ------
        # 创建GPUModelRunner
        self.use_v2_model_runner = vllm_config.use_v2_model_runner # 使用v2的modelrunner
        # ------【PP + 异步 RPC】记录上一轮未完成的非阻塞 PP send 句柄，下轮执行前先等它完成 ------
        # pending non-blocking PP send work from the previous iteration
        self._pp_send_work: list[Handle] = []

        # ------【显存 profiling】睡眠模式后端懒加载，首次 sleep/wake 时才解析并缓存进程级状态 ------
        # Resolved lazily on first sleep/wake; persists worker-process state.
        self._sleep_mode_backend: SleepModeBackend | None = None

    def _get_sleep_mode_backend(self) -> "SleepModeBackend":
        # ------【显存 profiling】首次调用才创建睡眠后端并缓存，避免每次都要重新解析工厂 ------
        if self._sleep_mode_backend is None:
            from vllm.device_allocator.sleep_mode_backend import (
                SleepModeBackendFactory,
            )

            self._sleep_mode_backend = SleepModeBackendFactory.create_backend(
                self.vllm_config.model_config
            )
        return self._sleep_mode_backend

    def sleep(self, level: int = 1) -> None:
        # ------【显存 profiling】先同步设备并记录入睡前空闲显存，作为「释放了多少」的基准 ------
        torch.accelerator.synchronize()
        free_bytes_before_sleep = torch.accelerator.get_memory_info()[0]

        # ------【显存 profiling】level 2 深度睡眠前，把主模型与 draft 模型的缓冲区搬到 CPU 暂存 ------
        # Save the buffers before level 2 sleep
        if level == 2:
            model = self.model_runner.model
            self._sleep_saved_buffers = {
                name: buffer.cpu().clone() for name, buffer in model.named_buffers()
            }
            draft = self.get_draft_model()
            if draft is not None:
                self._sleep_saved_draft_buffers = {
                    name: buffer.cpu().clone() for name, buffer in draft.named_buffers()
                }

        # ------【显存 profiling】按 level 挂起后端，真正释放权重/缓存等显存 ------
        self._get_sleep_mode_backend().suspend(level)

        # ------【显存 profiling】轮询等待显存真正被释放（ROCm 额外留 5s 容差），避免误判 ------
        torch.accelerator.synchronize()
        deadline = time.monotonic() + (5.0 if current_platform.is_rocm() else 0)
        while True:
            free_bytes_after_sleep, total = torch.accelerator.get_memory_info()
            freed_bytes = free_bytes_after_sleep - free_bytes_before_sleep
            if freed_bytes >= 0 or time.monotonic() >= deadline:
                break
            time.sleep(0.1)

        # ------【显存 profiling】校验睡眠确实释放了显存，并打印释放/仍在用的大小日志 ------
        used_bytes = total - free_bytes_after_sleep
        assert freed_bytes >= 0, "Memory usage increased after sleeping."
        logger.info(
            "Sleep mode freed %s GiB memory, %s GiB memory is still in use.",
            format_gib(freed_bytes),
            format_gib(used_bytes),
        )

    def wake_up(self, tags: list[str] | None = None) -> None:
        # ------【显存 profiling】按 tags 恢复后端，还原被挂起的权重/缓存等显存 ------
        self._get_sleep_mode_backend().resume(tags)

        # ------【显存 profiling】唤醒时把暂存的主模型缓冲区拷回 GPU，并清空暂存字典 ------
        # Restore the buffers after level 2 sleep
        wake_weights = tags is None or "weights" in tags
        if wake_weights and len(self._sleep_saved_buffers):
            model = self.model_runner.model
            for name, buffer in model.named_buffers():
                if name in self._sleep_saved_buffers:
                    buffer.data.copy_(self._sleep_saved_buffers[name].data)
            self._sleep_saved_buffers = {}

        # ------【显存 profiling】同样恢复投机 draft 模型的缓冲区 ------
        if wake_weights and len(self._sleep_saved_draft_buffers):
            draft = self.get_draft_model()
            if draft is not None:
                for name, buffer in draft.named_buffers():
                    if name in self._sleep_saved_draft_buffers:
                        buffer.data.copy_(self._sleep_saved_draft_buffers[name].data)
            self._sleep_saved_draft_buffers = {}

        # ------【显存 profiling】KV cache 醒来后做收尾（重建零化元数据等）──
        if tags is None or "kv_cache" in tags:
            self.model_runner.post_kv_cache_wake_up()

    def checkpoint_prepare(self) -> None:
        # ------【进程管理】准备分布式检查点：同步各 rank 的通信/随机状态，供容错恢复使用 ------
        checkpoint_prepare_distributed_state()

    def checkpoint_restore(self) -> None:
        # ------【进程管理】从检查点恢复分布式状态（通信句柄、随机数等）──
        checkpoint_restore_distributed_state()

    def _maybe_get_memory_pool_context(self, tag: str) -> AbstractContextManager:
        # ------【内存池/CuMem】CUDA 类设备未开 CuMem 分配器时无需内存池上下文，返回空上下文 ------
        if (
            current_platform.is_cuda_alike()
            and not self.vllm_config.model_config.enable_cumem_allocator
        ):
            return nullcontext()

        # ------【内存池/CuMem】XPU 未开睡眠模式同样不需要内存池上下文 ------
        if (
            current_platform.is_xpu()
            and not self.vllm_config.model_config.enable_sleep_mode
        ):
            return nullcontext()

        # ------【内存池/CuMem】纯 CPU 后端无统一内存池，返回空上下文 ------
        if current_platform.is_cpu():
            return nullcontext()

        # ------【内存池/CuMem】拿到统一分配器单例，权重池要求单实例，最后返回对应 tag 的内存池上下文 ------
        allocator = get_mem_allocator_instance()
        if tag == "weights":
            assert allocator.get_current_usage() == 0, (
                "CuMem allocator can only be used for one instance per process."
            )
        return allocator.use_memory_pool(tag=tag)

    @contextmanager
    def _scoped_allocator_max_split(self, max_split_size_mb: int):
        """Temporarily set max_split_size_mb to reduce allocator fragmentation at the
        cost of more cudaMalloc calls (negligible in practice). Restores the original
        value on exit."""
        # ------【内存池/CuMem】非 CUDA 后端无需调整分配器切分参数，直接放行 ------
        if not current_platform.is_cuda():
            yield
            return

        # ------【内存池/CuMem】解析当前 allocator 配置里已有的 max_split_size_mb，记住原值以便恢复 ------
        conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
        match = re.search(r"max_split_size_mb:(\d+)", conf)
        original_value = match.group(1) if match else None

        # ------【内存池/CuMem】临时把切分上限设为目标值，减少大块权重加载带来的碎片 ------
        torch._C._accelerator_setAllocatorSettings(
            f"max_split_size_mb:{max_split_size_mb}"
        )
        # ------【内存池/CuMem】退出上下文后恢复原值（默认无上限），保证不影响后续正常分配 ------
        try:
            yield
        finally:
            # PyTorch defaults to SIZE_MAX (no limit).
            _SIZE_MAX_MB = (2**64 - 1) // (1024 * 1024)
            restore = original_value if original_value else str(_SIZE_MAX_MB)
            torch._C._accelerator_setAllocatorSettings(f"max_split_size_mb:{restore}")



















    @instrument(span_name="Init device")
    def init_device(self):
        # 【Worker 初始化 · 阶段 1/3】Init Device
        #   初始化设备 + 分布式上下文(DP/TP/PP/EP 通信组) + 构造 model_runner(内含 InputBatch)
        # 如果我们指定是用GPU来运行模型
        if self.device_config.device_type == "cuda":
            # ------【CUDA Graph】Ray 注入的这个环境变量会干扰 CUDA graph 构建，先移除 ------
            # This env var set by Ray causes exceptions with graph building.
            os.environ.pop("NCCL_ASYNC_ERROR_HANDLING", None)
            parallel_config = self.parallel_config # 获取我们的并行化配置，TP，PP，PCP，EP,DP 的相关参数
            # ------【DP】非 Ray 后端且单机 DP 时，把本进程 local_rank 平移到对应 DP 副本的 GPU 区间 ------
            if (
                parallel_config.distributed_executor_backend
                not in ("ray", "external_launcher")
                and parallel_config.data_parallel_backend != "ray"
                and parallel_config.nnodes_within_dp == 1
            ):
                # Use local DP rank if available, otherwise use global DP rank.
                dp_local_rank = self.parallel_config.data_parallel_rank_local
                if dp_local_rank is None:
                    dp_local_rank = self.parallel_config.data_parallel_index

                # ------【TP + PP】每个 DP 副本占用 tp_size×pp_size 张卡，据此平移 local_rank ------
                # 按照配置计算所需GPU个数
                tp_pp_world_size = (
                    self.parallel_config.pipeline_parallel_size
                    * self.parallel_config.tensor_parallel_size
                )

                # DP_LOCAL_RANK * TP_PP_WORLD_SIZE + TP_LOCAL_RANK
                self.local_rank += dp_local_rank * tp_pp_world_size

            # ------【NCCL 通信】发布「逻辑卡→物理卡」映射，供 NIC 亲和 / P2P 拓扑查询使用 ------
            # Publish the logical-to-physical mapping for topology queries
            # such as NIC affinity and P2P checks.
            # assigned_physical_gpu_ids 是一张「逻辑 id → 物理卡」的映射表, 是本机节点运行用的物理GPU列表，按照下标和local_rank对应
            # 逻辑id是self.local_rank

            assigned_physical_gpu_ids = parallel_config.assigned_physical_gpu_ids
            if assigned_physical_gpu_ids is not None:
                from vllm.platforms.interface import set_assigned_physical_gpu_ids

                set_assigned_physical_gpu_ids(assigned_physical_gpu_ids) # 
                assert self.local_rank < len(assigned_physical_gpu_ids), (
                    f"local_rank {self.local_rank} is out of bounds for "
                    f"assigned_physical_gpu_ids {assigned_physical_gpu_ids}"
                )
                # NOTE(patch pr45026): local_world_size is derived from
                # parallel_config.nnodes, which is only set for the "mp"
                # multi-node backend. With the "ray"/"external_launcher"
                # backends nnodes stays 1, so local_world_size collapses to
                # the full world_size and this check wrongly fires on
                # cross-node deployments. assigned_physical_gpu_ids is already
                # per-node and the local_rank bound above fully validates the
                # mapping for these backends, so skip the check for them.
                if parallel_config.distributed_executor_backend not in (
                    "ray",
                    "external_launcher",
                ):
                    assert self.parallel_config.local_world_size <= len(
                        assigned_physical_gpu_ids
                    ), (
                        f"local_world_size ({self.parallel_config.local_world_size})"
                        " exceeds assigned_physical_gpu_ids count "
                        f"({len(assigned_physical_gpu_ids)})"
                    )
            # ------【进程管理】未配置物理卡映射时，兜底校验 DP 调整后的 local_rank 不越界 ------
            else:
                assert self.local_rank < torch.accelerator.device_count(), (
                    f"DP adjusted local rank {self.local_rank} is out of "
                    f"bounds for {torch.accelerator.device_count()} devices."
                )


            # ------【进程管理】把逻辑 local_rank 换算成真正写给 PyTorch 的物理卡号，并设为本进程当前设备 ------
            # visible_device_index 是「最终真正写给 PyTorch 的物理卡号」，作用就一个：决定 self.device 到底是哪张卡，并让 torch 把这张卡设成当前设备
            ###########################################################################
            # 1. 绑定物理GPU卡
            ###########################################################################
            visible_device_index = (
                current_platform.logical_device_id_to_visible_device_id(self.local_rank)
            )
            self.device = torch.device(f"cuda:{visible_device_index}") # 记录下本进程的gpu设备
            '''
            torch.accelerator 是 PyTorch 新出的设备无关 API，等价于老式的 torch.cuda.set_device()，
            但能自动适配 CUDA / ROCm(AMD) / XPU(Intel) / HPU 等不同后端。

            vLLM 要支持多种硬件，所以不用 torch.cuda 这种绑死 NVIDIA 的写法，统一走 torch.accelerator 这个抽象层。
            你在前面 else 分支里也看到了 torch.accelerator.device_count()（366 行），同理——它不是「又一个 accelerator」，而是同一套抽象 API 在干不同的事  
            '''
            torch.accelerator.set_device_index(self.device) # 开始把本进程的torch的cuda设备绑定好

            # ------【进程管理】校验当前设备是否支持模型要用的 dtype，不支持则提前失败 ------
            current_platform.check_if_supports_dtype(self.model_config.dtype)








            # Initialize the distributed environment BEFORE taking
            # memory snapshot
            # This ensures NCCL buffers are allocated before we measure
            # available memory
            # ------【NCCL 通信】先初始化分布式环境确保 NCCL 缓冲已分配，再切分 TP/PP/CP 子组 ------
            # 拉起NCCL通信网络，这是一个包装器，做点切分TP/PP/CP分组这些

            ###########################################################################
            # 2. 拉起NCCL通信网络
            ###########################################################################
            init_worker_distributed_environment(
                self.vllm_config,
                self.rank,
                self.distributed_init_method,
                self.local_rank,
                current_platform.dist_backend,
            )





            # ------【核心逻辑】一次性打印使用的 runner 版本，便于日志排查 ------
            if self.use_v2_model_runner:
                logger.info_once("Using V2 Model Runner")

            # ------【进程管理】统一设置 Python/numpy/torch/CUDA 随机种子，保证整条推理链确定性 ------
            # Set random seed.
            # 设置随机种子，因为 vLLM 的代码里会用到好几套随机数来源（Python 的 random、numpy、torch、以及 GPU 上的 CUDA 随机），
            # 所以要把它们全部设成同一个 seed，才能保证「整条推理链是确定性的」
            set_random_seed(self.model_config.seed)

            # ------【显存 profiling】清掉 Python 垃圾与 PyTorch 缓存后，拍一张初始显存快照作为后续预算基线 ------
            # Now take memory snapshot after NCCL is initialized
                    # 清理python层面的垃圾内存
            gc.collect()

                    # 清理pytorch的显存缓存
            torch.accelerator.empty_cache()

            # take current memory snapshot
            # 这个MemorySnapshot类，就是一个内存快照类，封装了一些torch的方法，来快速获得当前显存的情况
            

            ############################################################
            # 3. 创建内存拍照器，计算可用显存目标值
            ############################################################
            self.init_snapshot = init_snapshot = MemorySnapshot(device=self.device) # 测量还有多少显存
            # ------【显存 profiling】按 gpu_memory_utilization 算出「打算用多少显存」的预算目标值 ------
            # 这个就是用户设置显存使用比例的地方
            self.requested_memory = request_memory(init_snapshot, self.cache_config)# 计算显存预算，总显存 × 利用率，算出一个「我打算用多少显存」的目标值（预算）
            logger.debug("worker init memory snapshot: %r", self.init_snapshot)
            logger.debug(
                "worker requested memory: %sGiB", format_gib(self.requested_memory)
            )

        # ------【进程管理】非 GPU 设备直接报错（GPU worker 只支持 CUDA 类设备）──
        # 不是GPU，直接报错
        else:
            raise RuntimeError(f"Unsupported device type: {self.device_config.device}")

        # ------【进程管理】初始化 workspace 草稿纸缓冲池，给自定义 CUDA kernel 存放中间临时变量 ------
        # Initialize workspace manager
        '''
        这是工作区（workspace）缓冲区的管理器初始化——就是给各种自定义 CUDA kernel 准备「草稿纸」用的临时显存池

        这个workspace就是自定义算子所需要的 存放中间临时变量 的显存空间 的管理器，类似kvcachemanager的作用，不够比较简单
        1. 管理临时scratch的缓冲
        2. 没有语义，纯粹是草稿纸
        3. 生命周期端，一次kernel调用内借，用完还
        4. 没有分配表
        5. 复用逻辑，不够就继续扩容
        6. 并发隔离，靠每个ubatch槽位一份，避免并发互踩
        '''
        num_ubatches = 2 if self.vllm_config.parallel_config.enable_dbo else 1
        init_workspace_manager(self.device, num_ubatches)




        # ------核心逻辑：构造 model_runner（V1/V2 两种实现），后续前向/采样都委托给它 ------
        # Construct the model runner
        # Worker 已经准备好了环境，开始第二步，构造model_runner，开始准备着手处理模型了 ！！！！
        if self.use_v2_model_runner:
            from vllm.v1.worker.gpu.model_runner import (
                GPUModelRunner as GPUModelRunnerV2,
            )

            # HACK(woosuk): This is a temporary fix to avoid type errors.
            self.model_runner: GPUModelRunner = GPUModelRunnerV2(  # type: ignore
                self.vllm_config, self.device
            )
        else:
            from vllm.v1.worker.gpu_model_runner import (
                GPUModelRunner as GPUModelRunnerV1,
            )

            # Worker构造modelrunner v1实例


            ############################################################
            # 4. 构建model runner
            ############################################################
            self.model_runner = GPUModelRunnerV1(self.vllm_config, self.device)

        # ------【核心逻辑】rank0 负责收集并上报使用统计信息（可开关）──
        if self.rank == 0:
            # If usage stat is enabled, collect relevant info.
            report_usage_stats(self.vllm_config)
























    def handle_ft_command(self, ft_request):
        # ------【进程管理】容错模式下的控制命令统一转发给哨兵处理 ------
        assert self.worker_sentinel is not None
        return self.worker_sentinel.handle_command(ft_request)

    # FIXME(youkaichao & ywang96): Use TorchDispatchMode instead of memory pool
    # to hijack tensor allocation.
    def load_model(self, *, load_dummy_weights: bool = False) -> None:
        # 【Worker 初始化 · 阶段 2/3】Load Model
        #   构造模型结构 + 加载权重(按 TP/PP 切分与设备放置) + model.eval()
        #   + 可选 torch.compile / CUDA graph 包装
        '''
        真正把模型权重加载到GPU上的方法

        核心就一句 self.model_runner.load_model(...)，外面包了三层 with 上下文管理器做「加载环境准备」，后面再补一段 weight transfer（多机权重搬运）的可选逻辑
        '''
        # ------【显存 profiling】三层环境准备后加载权重：CuMem 内存池 + thread-local 配置 + 降低分配器碎片 ------
        with (
            self._maybe_get_memory_pool_context(tag="weights"), # 内存池,仅CuMem分配器，普通cuda用户为空

            # 把 vllm_config 设成「当前线程的 config」（thread-local 上下文）。
            # 因为 load_model 内部很深的地方（比如各 model 的 load_weights 实现）
            # 可能通过 get_current_vllm_config() 来取配置，这层保证取得到
            set_current_vllm_config(self.vllm_config),          # 把config设置成当前的config


            # 这是三者里唯一「对普通用户也真的做点事」的。它临时把 PyTorch CUDA 分配器的
            # max_split_size_mb 调成 20 MiB：
            # 为什么：加载权重时会申请很多大块、大小不一的显存，默认分配器容易产生碎片（内存碎成小块浪费掉）
            # 调成 20MiB = 让分配器更细地切分，减少碎片，代价是多几次 cudaMalloc（权重加载是一次性的，无所谓）
            # 退出 with 后 finally 里恢复原值（302-304 行）
            # 20 MiB is the minimum PyTorch allows for max_split_size_mb.
            self._scoped_allocator_max_split(max_split_size_mb=20),# ③ 临时调 allocator 参数
        ):
            self.model_runner.load_model(load_dummy_weights=load_dummy_weights) # 构造模型实例 + 加载权重

        # ------【异步 RPC】可选的多机权重传输：为跨机热更新/加载权重建立传输引擎 ------
        # 多机权重转移的逻辑
        if self.vllm_config.weight_transfer_config is not None:
            self.weight_transfer_engine = WeightTransferEngineFactory.create_engine(
                self.vllm_config.weight_transfer_config,
                self.vllm_config,
                self.device,
                self.model_runner.get_model(), 
            )










    def update_config(self, overrides: dict[str, Any]) -> None:
        # ------【核心逻辑】把配置覆盖项转发给 model_runner 热更新 ------
        self.model_runner.update_config(overrides)

    def reload_weights(self, *args, **kwargs) -> None:
        # ------【权重传输】在 thread-local config 下重载权重，保证深层 load_weights 能取到配置 ------
        with set_current_vllm_config(self.vllm_config):
            self.model_runner.reload_weights(*args, **kwargs)

    ########################################################
    # profiling的方法实现
    ########################################################
    @torch.inference_mode()
    def determine_available_memory(self) -> int:
        """Profiles the peak memory usage of the model to determine how much
        memory can be used for KV cache without OOMs.

        The engine will first conduct a profiling of the existing memory usage. 先测试已存在的显存使用
        Then, it calculates the free memory that can be used for KV cache in 计算空闲的可以用于kvcache的显存空间
        bytes.

        Tip:
            You may limit the usage of GPU memory
            by adjusting the `gpu_memory_utilization` parameter.
        """
        # ------【显存 profiling】应用上次启动保存的启动计划（复用上轮的 KV cache 大小决策）──
        maybe_apply_startup_plan(self)

        # ------【显存 profiling】用户手动指定 kv_cache_memory_bytes：跳过自动测量，直接按该值预留 ------
        # 跳过
        if kv_cache_memory_bytes := self.cache_config.kv_cache_memory_bytes:
            # still need a profile run which compiles the model for
            # max_num_batched_tokens
            self.model_runner.profile_run() 

            msg = (
                f"Initial free memory {format_gib(self.init_snapshot.free_memory)} "
                f"GiB, reserved {format_gib(kv_cache_memory_bytes)} GiB memory for "
                "KV Cache as specified by kv_cache_memory_bytes config and "
                "skipped memory profiling. This does not respect the "
                "gpu_memory_utilization config. Only use kv_cache_memory_bytes "
                "config when you want manual control of KV cache memory "
                "size. If OOM'ed, check the difference of initial free "
                "memory between the current run and the previous run "
                "where kv_cache_memory_bytes is suggested and update it "
                "correspondingly."
            )
            logger.info(msg)
            return reserve_mm_ipc_gpu_memory(
                kv_cache_memory_bytes,
                self.model_config.multimodal_config,
                getattr(self.parallel_config, "_api_process_count", 1),
            )

        # ------【显存 profiling】用假输入跑一次前向，实测模型权重+激活峰值，作为 KV cache 预算的依据 ------
        # Execute a forward pass with dummy inputs to profile the memory usage
        # of the model.
        # 之前没有指定过线程，我们自己profiling
        # 这里是上下文管理器，用来记录这个期间的profiling的输出结果
        ################################################################################################################
        # 1. worker开始指挥 model_runner 进行 profile 测试
        ################################################################################################################
        with memory_profiling(
            self.init_snapshot, # 内存拍照实例
            # 在此之前model_runner已经加载完了model
            weights_memory=int(self.model_runner.model_memory_usage), 
        ) as profile_result:
            self.model_runner.profile_run() # 开始测试运行

        # Profile CUDA graph memory if graphs will be captured.
        # ROCm is included: #44825 moved the profiler to
        # torch.accelerator.get_memory_info (reliable on ROCm, as used by
        # the AMD-CI mem tests), and graph_pool_handle resolves to the same
        # torch.cuda handle the live capture path already uses on ROCm.
        # XPU stays excluded (see #39977).
        # ------【CUDA Graph】若要捕获 CUDA graph，再单独估出图池（graph pool）占用的显存 ------
        cudagraph_memory_estimate = 0
        if (
            current_platform.is_cuda_alike()
            and self.vllm_config.compilation_config.cudagraph_mode != CUDAGraphMode.NONE
        ):
            cudagraph_memory_estimate = self.model_runner.profile_cudagraph_memory()

        # ------【CUDA Graph】按环境变量开关决定是否把图池估算计入峰值激活显存 ------
        # Respect the opt-in flag as originally designed.
        cudagraph_memory_estimate_applied = (
            cudagraph_memory_estimate
            if envs.VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS
            else 0
        )

        # ------【显存 profiling】记录总消耗、峰值激活显存与 CUDA graph 估算，供预算与建议使用 ------
        self.total_consumed = profile_result.total_consumed
        self.peak_activation_memory = (
            profile_result.transient_peak_headroom + cudagraph_memory_estimate_applied
        )
        self.cudagraph_memory_estimate = cudagraph_memory_estimate

        # ------【显存 profiling】校验 profiling 期间无其它进程释放显存，否则测量结果失真 ------
        free_gpu_memory = profile_result.after_profile.free_memory
        # NOTE(woosuk): Here we assume that the other processes using the same
        # GPU did not change their memory usage during the profiling.
        assert self.init_snapshot.free_memory >= free_gpu_memory, (
            "Error in memory profiling. "
            f"Initial free memory {format_gib(self.init_snapshot.free_memory)} GiB, "
            f"current free memory {format_gib(free_gpu_memory)} GiB. "
            "This happens when other processes sharing the same container "
            "release GPU memory while vLLM is profiling during initialization. "
            "To fix this, ensure consistent GPU memory allocation or "
            "isolate vLLM in its own container."
        )
        # ------【显存 profiling】可给 KV cache 的显存 = 预算 - 非 KV 占用 - CUDA graph 池估算 ------
        ################################################################################################################
        # 2. 计算kvccache可用显存
        ################################################################################################################
        self.available_kv_cache_memory_bytes = (
            self.requested_memory
            - profile_result.non_kv_cache_memory
            - cudagraph_memory_estimate_applied
        )

        # ------【显存 profiling】打印初始/剩余显存与 profiling 明细，便于用户调参定位 ------
        unrequested_memory = self.init_snapshot.free_memory - self.requested_memory
        logger.debug(
            "Initial free memory: %s GiB; Requested memory: %f (util), %s GiB",
            format_gib(self.init_snapshot.free_memory),
            self.cache_config.gpu_memory_utilization,
            format_gib(self.requested_memory),
        )
        logger.debug(
            "Free memory after profiling: %s GiB (total), %s GiB (within requested)",
            format_gib(free_gpu_memory),
            format_gib(free_gpu_memory - unrequested_memory),
        )
        logger.debug(profile_result)
        logger.info_once(
            "Available KV cache memory: %s GiB",
            format_gib(self.available_kv_cache_memory_bytes),
        )

        # ------【CUDA Graph】图池内存参与预算时，提示用户调整 gpu_memory_utilization 保持等效 KV cache ------
        if cudagraph_memory_estimate > 0:
            total_mem = self.init_snapshot.total_memory
            current_util = self.cache_config.gpu_memory_utilization
            cg_util_delta = cudagraph_memory_estimate / total_mem
            if envs.VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS:
                equiv_util = round(current_util - cg_util_delta, 4)
                suggested_util = min(
                    round(current_util + cg_util_delta, 4),
                    1.0,
                )
                logger.info(
                    "CUDA graph memory profiling is enabled (default since "
                    "v0.21.0). The current --gpu-memory-utilization=%.4f is "
                    "equivalent to --gpu-memory-utilization=%.4f without "
                    "CUDA graph memory profiling. To maintain the same "
                    "effective KV cache size as before, increase "
                    "--gpu-memory-utilization to %.4f. To disable, set "
                    "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0.",
                    current_util,
                    equiv_util,
                    suggested_util,
                )
            else:
                suggested_util = min(
                    round(current_util + cg_util_delta, 4),
                    1.0,
                )
                logger.warning(
                    "CUDA graph memory profiling is disabled "
                    "(VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0). "
                    "Without it, CUDA graph memory is not accounted for "
                    "during KV cache allocation, which may require lowering "
                    "--gpu-memory-utilization to avoid OOM. Consider "
                    "re-enabling it (the default as of v0.21.0) and increasing "
                    "--gpu-memory-utilization from %.4f to %.4f.",
                    current_util,
                    suggested_util,
                )

        # ------【显存 profiling】扣除多模态输入所需的 IPC 共享显存后，返回最终可用 KV cache 字节数 ------
        return reserve_mm_ipc_gpu_memory(
            int(self.available_kv_cache_memory_bytes),
            self.model_config.multimodal_config,
            getattr(self.parallel_config, "_api_process_count", 1),
        )





















    def get_kv_connector_handshake_metadata(
        self,
    ) -> dict[tuple[int, int], KVConnectorHandshakeMetadata] | None:
        """Get KV connector metadata from this worker if available.

        Returned dict is keyed by `(pp_rank, tp_rank)`.
        """

        # ------【PD 分离】未配置 KV 传输组时无需握手元数据，直接返回 None ------
        if not has_kv_transfer_group():
            return None

        connector = get_kv_transfer_group()
        # Return None for connectors that don't need to exchange handshake
        # metadata across workers.
        # ------【PD 分离】connector 明确表示不需要跨 worker 交换握手信息时返回 None ------
        if (metadata := connector.get_handshake_metadata()) is None:
            return None

        # ------【PD 分离】用 (pp_rank, tp_rank) 作 key 打包本 worker 的握手元数据供协调器汇总 ------
        pp_rank = get_pp_group().rank_in_group
        tp_rank = get_tp_group().rank_in_group
        return {(pp_rank, tp_rank): metadata}

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        # ------【显存 profiling】把 KV cache 规格查询转发给 model_runner ------
        return self.model_runner.get_kv_cache_spec()

    def update_max_model_len(self, max_model_len: int) -> None:
        """Update max_model_len after auto-fit to GPU memory.
        This is called when max_model_len=-1 is used and the engine
        automatically determines the maximum context length that fits
        in GPU memory. Workers need to update their cached max_model_len
        to match the engine's decision.
        """
        # ------【核心逻辑】自动适配后同步 max_model_len 到本地配置与 model_runner ------
        self.model_config.max_model_len = max_model_len
        if self.model_runner is not None:
            self.model_runner.update_max_model_len(max_model_len)
        logger.debug("Updated max_model_len to %d", max_model_len)









    @instrument(span_name="Allocate KV cache")
    def initialize_from_config(self, kv_cache_config: KVCacheConfig) -> None:
        """Allocate GPU KV cache with the specified kv_cache_config."""

        # ------【显存 profiling】用 profiling 后调整好的块数回填本地配置，供 warmup 阶段使用 ------
        # Update local config with adjusted num blocks after profiling,
        # so that it's available to the warmup stage.
        ############################################################################################
        # 1. 更新num_gpu_blocks, 把单层可用的block数更新上去, 这里仅仅只是根据一个block_size个token的所有层的blocks的大小除出来的
        ############################################################################################
        self.cache_config.num_gpu_blocks = kv_cache_config.num_blocks 

        # ------【PD 分离】初始化 KV connector，让 prefill/decode 实例之间能跨机搬运 KV cache ------
        # Init kv cache connector here, because it requires
        # `kv_cache_config`.
        # NOTE(Kuntai): This need to be done before `initialize_kv_cache`,
        # because `initialize_kv_cache` will inject kv cache groups not
        # related to kv cache connector (e.g. kv cache sharing layers).
        ensure_kv_transfer_initialized(self.vllm_config, kv_cache_config)

        # ------【显存 profiling】在 KV cache 内存池上下文中真正分配 KV cache 张量 ------
        ############################################################################################
        # 2. 开始正式通知model_runner来初始化显存，把他全部占用，申请成tensor
        ############################################################################################
        with self._maybe_get_memory_pool_context(tag="kv_cache"):
            self.model_runner.initialize_kv_cache(kv_cache_config)







        # ------【EP/EPLB】开启返回路由专家时，初始化专家捕获器（用于弹性 EP 动态扩缩容）──
        if self.model_config.enable_return_routed_experts:
            self.model_runner.init_routed_experts_capturer()

        # ------【显存 profiling】在 CuMem 池外构建 KV 置零元数据，避免记账张量在睡眠/唤醒时被丢弃 ------
        # Build KV-zero metadata outside the CuMem pool so the bookkeeping
        # GPU tensors (seg_addrs, block-id buffers) use the standard PyTorch
        # allocator and are not discarded during sleep/wake cycles.
        if kv_cache_config.needs_kv_cache_zeroing and hasattr(
            self.model_runner, "_init_kv_zero_meta"
        ):
            self.model_runner._init_kv_zero_meta()












    @instrument(span_name="Warmup (GPU)")
    def compile_or_warm_up_model(self) -> CompilationTimes:
        # ------【核心逻辑】收集需要预热/编译的 batch 尺寸集合 ------
        warmup_sizes: list[int] = []

        # ------【CUDA Graph】vLLM 编译模式下，把用户指定的 compile_sizes 纳入预热列表 ------
        if self.vllm_config.compilation_config.mode == CompilationMode.VLLM_COMPILE:
            # warm up sizes that are not in cudagraph capture sizes,
            # but users still want to compile for better performance,
            # e.g. for the max-num-batched token size in chunked prefill.
            compile_sizes = self.vllm_config.compilation_config.compile_sizes
            warmup_sizes = compile_sizes.copy() if compile_sizes is not None else []  # type: ignore[assignment]
            cg_capture_sizes: list[int] = []

            # ------【CUDA Graph】图捕获尺寸无需重复预热，从预热列表中剔除 ------
            if self.vllm_config.compilation_config.cudagraph_mode != CUDAGraphMode.NONE:
                cg_sizes = self.vllm_config.compilation_config.cudagraph_capture_sizes
                cg_capture_sizes = [] if cg_sizes is None else cg_sizes
                warmup_sizes = [x for x in warmup_sizes if x not in cg_capture_sizes]

            # ------【CUDA Graph】保证每个编译区间至少有一个尺寸触发编译/预热 ------
            compile_ranges = self.vllm_config.compilation_config.get_compile_ranges()
            # For each compile_range, if none of the batch sizes
            # in warmup_sizes or cudagraph_capture_sizes are in the range,
            # add the end of the range to ensure compilation/warmup.
            all_sizes = set(cg_capture_sizes)
            all_sizes.update([x for x in warmup_sizes if isinstance(x, int)])
            for compile_range in compile_ranges:
                if not any(x in compile_range for x in all_sizes):
                    warmup_sizes.append(compile_range.end)

        # ------【CUDA Graph】按尺寸从大到小跑 dummy run 预热编译路径（跳过 EPLB 避免脏指标）──
        # We skip EPLB here since we don't want to record dummy metrics
        for size in sorted(warmup_sizes, reverse=True):
            logger.info("Compile and warming up model for size %d", size)
            self.model_runner._dummy_run(size, skip_eplb=True, remove_lora=False)
        self.model_runner.maybe_remove_all_loras(self.model_runner.lora_config)

        # ------【核心逻辑】在图捕获前预热并调优模型执行用到的 kernel ------
        # Warmup and tune the kernels used during model execution before
        # cuda graph capture.
        kernel_warmup(self)

        # ------【CUDA Graph】非强制 eager 时捕获 CUDA graph，并记录其实际占用显存 ------
        cuda_graph_memory_bytes = 0
        if not self.model_config.enforce_eager:
            cuda_graph_memory_bytes = self.model_runner.capture_model()

        # ------【CUDA Graph】对比图池实际与估算显存并打印差异，校验 profiling 精度 ------
        # Compare actual vs estimated CUDA graph memory (if we did profiling)
        if (
            hasattr(self, "cudagraph_memory_estimate")
            and self.cudagraph_memory_estimate > 0
        ):
            GiB = lambda b: round(b / GiB_bytes, 2)
            diff = abs(cuda_graph_memory_bytes - self.cudagraph_memory_estimate)
            logger.info(
                "CUDA graph pool memory: %s GiB (actual), %s GiB (estimated), "
                "difference: %s GiB (%.1f%%).",
                GiB(cuda_graph_memory_bytes),
                GiB(self.cudagraph_memory_estimate),
                GiB(diff),
                100 * diff / max(cuda_graph_memory_bytes, 1),
            )

        # ------【显存 profiling】未手动指定 KV cache 时，按实测内存给用户建议最优 kv cache 大小 ------
        if self.cache_config.kv_cache_memory_bytes is None and hasattr(
            self, "peak_activation_memory"
        ):
            # Suggests optimal kv cache memory size if we rely on
            # memory_profiling to guess the kv cache memory size which
            # provides peak_activation_memory and a few other memory
            # consumption. `memory_profiling` does not consider
            # CUDAGraph memory size and may not utilize all gpu memory.
            # Users may want fine-grained control to specify kv cache
            # memory size.

            # empirically observed that the memory profiling may
            # slightly underestimate the memory consumption.
            # So leave a small buffer (=150MiB) to avoid OOM.
            redundancy_buffer_memory = 150 * (1 << 20)

            # ------【显存 profiling】累加权重+激活+CUDA graph 得到非 KV 占用，反推两条 kv cache 上限 ------
            non_kv_cache_memory = (
                self.total_consumed
                + self.peak_activation_memory
                + cuda_graph_memory_bytes
            )
            kv_cache_memory_bytes_to_gpu_limit = (
                self.init_snapshot.free_memory
                - non_kv_cache_memory
                - redundancy_buffer_memory
            )
            kv_cache_memory_bytes_to_requested_limit = (
                int(self.requested_memory)
                - non_kv_cache_memory
                - redundancy_buffer_memory
            )

            msg = (
                f"Free memory on device "
                f"({format_gib(self.init_snapshot.free_memory)}/"
                f"{format_gib(self.init_snapshot.total_memory)} GiB) on startup. "
                f"Desired GPU memory utilization is "
                f"({self.cache_config.gpu_memory_utilization}, "
                f"{format_gib(self.requested_memory)} GiB). "
                f"Actual usage is {format_gib(self.total_consumed)} "
                f"GiB for consumed memory (weights + non-torch), "
                f"{format_gib(self.peak_activation_memory)} GiB "
                f"for peak activation, and {format_gib(cuda_graph_memory_bytes)} "
                f"GiB for CUDAGraph memory. Replace gpu_memory_utilization "
                f"config with `--kv-cache-memory="
                f"{kv_cache_memory_bytes_to_requested_limit}` "
                f"({format_gib(kv_cache_memory_bytes_to_requested_limit)} GiB) to fit "
                f"into requested memory, or `--kv-cache-memory="
                f"{kv_cache_memory_bytes_to_gpu_limit}` "
                f"({format_gib(kv_cache_memory_bytes_to_gpu_limit)} GiB) to fully "
                f"utilize gpu memory. Current kv cache memory in use is "
                f"{format_gib(self.available_kv_cache_memory_bytes)} GiB."
            )

            logger.info(msg)

            # ------【显存 profiling】把建议值存成启动计划，供下次启动直接复用 ------
            maybe_save_startup_plan(self, kv_cache_memory_bytes_to_requested_limit)

        # ------【核心逻辑】V2：跑完整 execute+sample 触发 triton kernel 的 JIT 编译 ------
        if self.use_v2_model_runner:
            # V2: Run full execute_model + sample_tokens to JIT compile triton kernels.
            warmup_kernels(self.model_runner, self.execute_model, self.sample_tokens)
        # ------【核心逻辑】V1 末级 rank：预热 sampler 并预分配采样张量，避免后续碎片化 ------
        elif get_pp_group().is_last_rank:
            # V1: Warm up sampler and preallocate memory buffer for logits and other
            # sampling related tensors of max possible shape to avoid memory
            # fragmentation issue.
            # NOTE: This is called after `capture_model` on purpose to prevent
            # memory buffers from being cleared by `torch.accelerator.empty_cache`.
            max_num_reqs = min(
                self.scheduler_config.max_num_seqs,
                self.scheduler_config.max_num_batched_tokens,
            )

            # We skip EPLB here since we don't want to record dummy metrics
            hidden_states, last_hidden_states = self.model_runner._dummy_run(
                num_tokens=max_num_reqs,
                skip_eplb=True,
                cudagraph_runtime_mode=CUDAGraphMode.NONE,
            )
            if self.model_runner.is_pooling_model:
                self.model_runner._dummy_pooler_run(hidden_states)
            else:
                self.model_runner._dummy_sampler_run(hidden_states=last_hidden_states)

        # ------【核心逻辑】重置随机种子，消除初始化/profiling 对随机状态的影响 ------
        # Reset the seed to ensure that the random state is not affected by
        # the model initialization and profiling.
        set_random_seed(self.model_config.seed)

        # ------【CUDA Graph】预热期主动触发 inductor 一次性惰性初始化，避免运行时编译缓存未命中卡顿 ------
        # Eagerly trigger inductor's once-per-process lazy inits during
        # warmup (rather than on a later compile cache-miss at runtime).
        c_config = self.compilation_config
        if c_config.mode != CompilationMode.NONE and c_config.backend == "inductor":
            from vllm.compilation.compiler_interface import (
                trigger_inductor_lazy_init,
            )

            trigger_inductor_lazy_init(self.device)

        # ------【核心逻辑】预热结束，开启 JIT 监控以捕捉推理期的意外编译延迟尖刺 ------
        # All warmup is done — start monitoring for unexpected JIT
        # compilations that would cause latency spikes during inference.
        from vllm.utils.jit_monitor import activate as activate_jit_monitor

        activate_jit_monitor(
            mode=self.observability_config.jit_monitor_mode,
            verbose=self.observability_config.jit_monitor_verbose,
        )

        # ------【显存 profiling】冻结 GC 堆，避免推理期扫描静态大对象（权重/KV/图）──
        # Freeze the worker heap so the GC won't scan static objects
        # (model weights, KV caches, CUDA graphs) during inference.
        freeze_gc_heap()
        maybe_attach_gc_debug_callback()

        # ------【核心逻辑】预热完成，开启 GPU 同步检查门控，后续调用强制同步校验 ------
        # Warmup / first-compile is done — activate the `VLLM_GPU_SYNC_CHECK`
        # gate so subsequent `execute_model` / `sample_tokens` calls enforce it.
        enable_gpu_sync_check()

        # ------【核心逻辑】稳态服务不再需要 torch intra-op 并行，收窄线程数以省资源 ------
        # Startup is done; steady-state serving gets no benefit from torch
        # intra-op parallelism.
        set_torch_threads_for_runtime()

        return CompilationTimes(
            language_model=self.compilation_config.compilation_time,
            encoder=self.compilation_config.encoder_compilation_time,
        )

    def reset_mm_cache(self) -> None:
        # ------【核心逻辑】清空多模态缓存，转发给 model_runner ------
        self.model_runner.reset_mm_cache()

    def reset_encoder_cache(self) -> None:
        # ------【核心逻辑】清空 encoder 缓存 ------
        self.model_runner.reset_encoder_cache()

    def get_model(self) -> nn.Module:
        # ------【权重传输】暴露主模型引用，供权重传输/保存等场景使用 ------
        return self.model_runner.get_model()

    def get_draft_model(self) -> nn.Module | None:
        # ------【投机解码】暴露投机 draft 模型引用 ------
        return self.model_runner.get_draft_model()

    def _set_draft_weight_update_target(self) -> None:
        assert self.weight_transfer_engine is not None

        # ------【权重传输】校验已配置 draft 模型，否则无法对投机模型做权重更新 ------
        draft_model = self.get_draft_model()
        if draft_model is None:
            raise RuntimeError(
                "Draft model weight update requested, but no draft model is configured."
            )

        # ------【权重传输】确认投机配置与 draft 模型配置齐备 ------
        speculative_config = self.speculative_config
        if speculative_config is None or speculative_config.draft_model_config is None:
            raise RuntimeError(
                "Draft model weight update requested, but no draft model "
                "config is configured."
            )

        # ------【权重传输】把权重更新目标切换到 draft 模型及其配置 ------
        self.weight_transfer_engine.set_weight_update_target(
            draft_model, speculative_config.draft_model_config
        )

    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        # ------【核心逻辑】返回本模型支持的任务类型，转发给 model_runner ------
        return self.model_runner.get_supported_tasks()

    def get_compilation_match_table(self) -> dict[str, int]:
        # ------【CUDA Graph】返回编译匹配表，供编译缓存匹配使用 ------
        from vllm.compilation.passes.vllm_inductor_pass import get_match_table

        return get_match_table()

    def get_encoder_timing_stats(self) -> dict[str, dict[str, float | int]]:
        """Get encoder timing stats from model runner."""
        # ------【核心逻辑】返回 encoder 计时统计 ------
        return self.model_runner.get_encoder_timing_stats()

    def annotate_profile(self, scheduler_output):
        # add trace annotation so that we can easily distinguish
        # context/generation request numbers in each iteration.
        # A context request is a request that has not yet generated any tokens
        # ------【核心逻辑】未启用 profiler 时返回空上下文，避免额外开销 ------
        if not self.profiler:
            return nullcontext()

        self.profiler.step()

        # ------【核心逻辑】从调度输出统计本轮 context/generation 请求与 token 数 ------
        iteration_details = compute_iteration_details(scheduler_output)

        if self.vllm_config.profiler_config.detailed_trace_annotation:
            # Compute roofline-model metrics per request, split by phase
            # (context vs generation). These help estimate compute and
            # memory intensity from the trace.
            #
            # Per-request quantities:
            #   query_len = number of scheduled (new) tokens for this request
            #   seq_len   = total sequence length (computed + scheduled tokens)
            #
            # Aggregated across requests in each phase
            # (ctx_=context, gen_=generation):
            #   seq_len_sum = sum of seq_len   (total KV length)
            #   qq_compute  = sum of query_len*query_len
            #                 (proxy for QK^T compute cost)
            #   qk_compute  = sum of query_len*seq_len
            #                 (proxy for QK^T compute cost for decode and
            #                  chunked prefill)
            #   total_scheduled_tokens = scheduled tokens across all requests
            # ------【核心逻辑】初始化 roofline 各相位累积量（context/gen 分开）──
            ctx_seq_len_sum = 0
            ctx_qq_compute = 0
            ctx_qk_compute = 0
            gen_seq_len_sum = 0
            gen_qq_compute = 0
            gen_qk_compute = 0
            total_scheduled_tokens = 0

            # ------【核心逻辑】构造 req_id→已计算 token 数映射，供后续按相位归类 ------
            # Build a map of req_id -> num_computed_tokens for all requests
            new_req_ids = {
                new_req.req_id for new_req in scheduler_output.scheduled_new_reqs
            }
            num_computed_tokens_ids = {
                new_req.req_id: new_req.num_computed_tokens
                for new_req in scheduler_output.scheduled_new_reqs
            }
            for req_id, num_computed_tokens in zip(
                scheduler_output.scheduled_cached_reqs.req_ids,
                scheduler_output.scheduled_cached_reqs.num_computed_tokens,
            ):
                num_computed_tokens_ids[req_id] = num_computed_tokens

            # ------【核心逻辑】逐请求累计各相位的 seq_len/QQ/QK 计算量代理指标 ------
            # Accumulate per-phase metrics
            for req_id, num_tokens in scheduler_output.num_scheduled_tokens.items():
                query_len = num_tokens
                total_scheduled_tokens += query_len
                seq_len = num_computed_tokens_ids.get(req_id, 0) + query_len
                if (
                    scheduler_output.scheduled_cached_reqs.is_context_phase(req_id)
                    or req_id in new_req_ids
                ):
                    ctx_seq_len_sum += seq_len
                    ctx_qq_compute += query_len * query_len
                    ctx_qk_compute += query_len * seq_len
                else:
                    gen_seq_len_sum += seq_len
                    gen_qq_compute += query_len * query_len
                    gen_qk_compute += query_len * seq_len
            # ------【核心逻辑】按相位拼出 trace 注解字符串，便于在时间线上区分 context/gen ------
            annotation = "".join(
                [
                    "execute_",
                    str(total_scheduled_tokens),
                    "_context_",
                    str(iteration_details.num_ctx_requests),
                    "(sq",
                    str(iteration_details.num_ctx_tokens),
                    "sk",
                    str(ctx_seq_len_sum),
                    "sqsq",
                    str(ctx_qq_compute),
                    "sqsk",
                    str(ctx_qk_compute),
                    ")_generation_",
                    str(iteration_details.num_generation_requests),
                    "(sq",
                    str(iteration_details.num_generation_tokens),
                    "sk",
                    str(gen_seq_len_sum),
                    "sqsq",
                    str(gen_qq_compute),
                    "sqsk",
                    str(gen_qk_compute),
                    ")",
                ]
            )
        # ------【核心逻辑】非详细模式下只拼 request/token 数量的简化注解 ------
        else:
            annotation = "".join(
                [
                    "execute_context_",
                    str(iteration_details.num_ctx_requests),
                    "(",
                    str(iteration_details.num_ctx_tokens),
                    ")",
                    "_generation_",
                    str(iteration_details.num_generation_requests),
                    "(",
                    str(iteration_details.num_generation_tokens),
                    ")",
                ]
            )
        # ------【核心逻辑】把注解字符串包成上下文管理器交给 profiler ------
        return self.profiler.annotate_context_manager(annotation)

    @torch.inference_mode()
    @with_gpu_sync_check
    def sample_tokens(
        self, grammar_output: "GrammarOutput | None"
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput:
        # ------【结构化输出/grammar】采样委托给 model_runner，grammar bitmask 在此约束 token 选择 ------
        return self.model_runner.sample_tokens(grammar_output)








    ################################################################################################
    # 开始执行一次调度任务batch
    ################################################################################################
    @torch.inference_mode()
    @with_gpu_sync_check
    def execute_model(
        self, scheduler_output: "SchedulerOutput"
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput | None:
        # ------【PP + 异步 RPC】等上一轮非阻塞 PP send 完成，避免与新迭代的通信重叠冲突 ------
        # ensure any previous non-blocking PP sends are complete
        if self._pp_send_work:
            for handle in self._pp_send_work:
                handle.wait()
            self._pp_send_work = []





        # ------【核心逻辑】读取本轮是否有待调度 token，并初始化中间张量/通信变量 ------
        ########################################################################
        # 1. 读取一些相关调度任务信息
        ########################################################################
        intermediate_tensors = None
        forward_pass = scheduler_output.total_num_scheduled_tokens > 0
        num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        all_gather_tensors = {}
        compilation_config = self.vllm_config.compilation_config
        parallel_config = self.vllm_config.parallel_config





        # ------【PP + TP】PP>1 且开启序列并行(SP)时，预先算出残差是否需要 all-gather ------
        if (
            parallel_config.pipeline_parallel_size > 1
            and compilation_config.pass_config.enable_sp
            and forward_pass
        ):
            # currently only supported by V1 GPUModelRunner
            assert not self.use_v2_model_runner
            num_scheduled_tokens_np = np.array(
                list(scheduler_output.num_scheduled_tokens.values()),
                dtype=np.int32,
            )
            # TODO(lucas): This is pretty gross; ideally we should only ever call
            # `_determine_batch_execution_and_padding` once (will get called again
            # in `execute_model`) but this requires a larger refactor of PP.
            _, batch_desc, _, _, _ = (
                self.model_runner._determine_batch_execution_and_padding(
                    num_tokens=num_scheduled_tokens,
                    num_reqs=len(num_scheduled_tokens_np),
                    num_scheduled_tokens_np=num_scheduled_tokens_np,
                    max_num_scheduled_tokens=num_scheduled_tokens_np.max(),
                    use_cascade_attn=False,  # TODO(lucas): Handle cascade attention
                )
            )
            all_gather_tensors = {
                "residual": not is_residual_scattered_for_sp(
                    self.vllm_config, batch_desc.num_tokens
                )
            }

        # ------【PP + 异步 RPC】非首个 PP 阶段：从上游非阻塞接收中间张量，包装成惰性同步对象 ------
        if forward_pass and not get_pp_group().is_first_rank:
            tensor_dict, comm_handles, comm_postprocess = (
                get_pp_group().irecv_tensor_dict(
                    all_gather_group=get_tp_group(),
                    all_gather_tensors=all_gather_tensors,
                )
            )
            assert tensor_dict is not None
            intermediate_tensors = AsyncIntermediateTensors(
                tensor_dict,
                comm_handles=comm_handles,
                comm_postprocess=comm_postprocess,
            )

        # ------【CUDA Graph】带 profiling 注解调用 model_runner 前向（内部含 CUDA graph 回放 / eager 两种路径）──
        with self.annotate_profile(scheduler_output):
            ########################################################################
            # 2. 开始让model runner来执行这个batch, worker这里主要负责分布式的一些处理
            ########################################################################
            output = self.model_runner.execute_model(
                scheduler_output, intermediate_tensors
            )





            # ------【核心逻辑】V2 pooling 模型在 output 为空时补跑 pool；得到最终输出就直接返回 ------
            if (
                self.use_v2_model_runner
                and self.model_runner.is_pooling_model
                and output is None
            ):
                output = self.model_runner.pool()  # type: ignore
            if isinstance(
                output, ModelRunnerOutput | AsyncModelRunnerOutput | NoneType
            ):
                return output

        # ------【PP】确认输出是中间张量、且本进程既非 external_launcher 也非末级 PP rank ------
        assert isinstance(output, IntermediateTensors)
        parallel_config = self.vllm_config.parallel_config
        assert (
            parallel_config.distributed_executor_backend != "external_launcher"
            and not get_pp_group().is_last_rank
        )

        # ------【PP + 异步 RPC】非末级 PP 阶段：非阻塞把中间张量发给下游，句柄留待下轮等待 ------
        # launch non-blocking send of intermediate tensors
        self._pp_send_work = get_pp_group().isend_tensor_dict(
            output.tensors,
            all_gather_group=get_tp_group(),
            all_gather_tensors=all_gather_tensors,
        )

        return None


















    def take_draft_token_ids(self) -> DraftTokenIds | None:
        # ------【投机解码】取回 draft 模型的草稿 token ids ------
        return self.model_runner.take_draft_token_ids()

    def profile(self, is_start: bool = True, profile_prefix: str | None = None):
        # ------【核心逻辑】未启用 profiler 却调用时直接报错并提示用法 ------
        # Check if profiling is enabled
        if self.profiler_config is None or self.profiler_config.profiler is None:
            raise RuntimeError(
                "Profiling is not enabled. Please set --profiler-config to enable "
                "profiling. Example: "
                "'--profiler-config.profiler=torch --profiler-config.torch_profiler_dir"
                "=YOUR_DIR_PATH_TO_DUMP_TRACE'"
            )

        if is_start:
            # ------【进程管理】start 分支：用前缀+rank 后缀生成 trace 名，区分各 worker 的 trace ------
            # Generate the trace name by combining prefix with comprehensive rank suffix
            from vllm.distributed.utils import get_worker_rank_suffix

            rank_suffix = get_worker_rank_suffix(global_rank=self.rank)

            # Build the full trace name
            if profile_prefix:
                trace_name = f"{profile_prefix}_{rank_suffix}"
            else:
                trace_name = rank_suffix

            # ------【核心逻辑】首次 start 时才按类型实例化 torch/cuda profiler 包装器 ------
            # Create the profiler wrapper only on the first start call
            if self.profiler is None:
                profiler_type = self.profiler_config.profiler
                if profiler_type == "torch":
                    self.profiler = TorchProfilerWrapper(
                        self.profiler_config,
                        worker_name=trace_name,
                        local_rank=self.local_rank,
                        activities=["CPU", "CUDA"],
                    )
                    logger.debug(
                        "Starting torch profiler with trace name: %s", trace_name
                    )
                elif profiler_type == "cuda":
                    self.profiler = CudaProfilerWrapper(self.profiler_config)
                    logger.debug("Starting CUDA profiler")
                else:
                    # Config validation should prevent this code being reached
                    raise ValueError(
                        f"Invalid profiler value of {self.profiler_config.profiler}"
                    )

            # ------【核心逻辑】启动（或重启）profiling ------
            # If profiler already initialized, restart profiling but keep
            # the original trace name from the first initialization.
            self.profiler.start()
        else:
            # ------【核心逻辑】stop 分支：未启动则告警返回，否则停止采样 ------
            if self.profiler is None:
                logger.warning("Profiler was not started, nothing to stop.")
                return
            self.profiler.stop()

    def execute_dummy_batch(self) -> None:
        # ------【CUDA Graph】跑一个均匀 decode 的 dummy 批次，预热 decode 路径 ------
        num_tokens = getattr(self.model_runner, "uniform_decode_query_len", 1)
        self.model_runner._dummy_run(num_tokens, uniform_decode=True)

    def add_lora(self, lora_request: LoRARequest) -> bool:
        # ------【LoRA】加载 LoRA adapter 并返回是否成功 ------
        return self.model_runner.add_lora(lora_request)

    def remove_lora(self, lora_id: int) -> bool:
        # ------【LoRA】卸载指定 LoRA adapter ------
        return self.model_runner.remove_lora(lora_id)

    def list_loras(self) -> set[int]:
        # ------【LoRA】列出当前已加载的 LoRA id 集合 ------
        return self.model_runner.list_loras()

    def pin_lora(self, lora_id: int) -> bool:
        # ------【LoRA】钉住指定 LoRA，避免被逐出缓存 ------
        return self.model_runner.pin_lora(lora_id)

    def check_health(self) -> None:
        # ------【进程管理】worker 只要还在运行就是健康的 ------
        # worker will always be healthy as long as it's running.
        return

    def save_sharded_state(
        self,
        path: str,
        pattern: str | None = None,
        max_size: int | None = None,
    ) -> None:
        # ------【权重传输】把模型分片保存到磁盘，供后续分片加载 ------
        from vllm.model_executor.model_loader import ShardedStateLoader

        ShardedStateLoader.save_model(
            self.model_runner.model,
            path,
            pattern=pattern,
            max_size=max_size,
        )

    def save_tensorized_model(self, tensorizer_config: "TensorizerConfig") -> None:
        # ------【权重传输】按 tensorizer 配置序列化保存模型 ------
        TensorizerLoader.save_model(
            self.get_model(),
            tensorizer_config=tensorizer_config,
            model_config=self.model_config,
        )

    def _check_weight_transfer_engine(self) -> None:
        # ------【权重传输】未配置权重传输时直接报错，提示设置 weight_transfer_config ------
        if self.weight_transfer_engine is None:
            raise RuntimeError(
                "Weight transfer not configured. "
                "Please set weight_transfer_config to enable weight transfer."
            )

    def init_weight_transfer_engine(self, init_info: dict) -> None:
        """
        Initialize weight transfer mechanism.
        For NCCL backend, this creates a process group with the trainer.

        Args:
            init_info: Dictionary containing backend-specific initialization info
        """
        self._check_weight_transfer_engine()
        assert self.weight_transfer_engine is not None
        # ------【权重传输】把 init_info 解析成后端类型化数据类，再真正初始化传输引擎（如建 NCCL 组）──
        # Parse dict into backend-specific typed dataclass
        typed_init_info = self.weight_transfer_engine.parse_init_info(init_info)
        self.weight_transfer_engine.init_transfer_engine(typed_init_info)

    def start_weight_update(self) -> None:
        """
        Start a new weight update session.

        Delegates engine-specific preparation (e.g. layerwise reload setup) to
        the configured weight transfer engine. The worker only tracks that a
        session is active.
        """
        # ------【权重传输】在 thread-local config 下开启主模型权重更新会话 ------
        with set_current_vllm_config(self.vllm_config):
            self._start_weight_update()

    def start_draft_weight_update(self) -> None:
        """
        Like start_weight_update, but retargets the engine at the speculative
        draft model for this session.
        """
        # ------【权重传输】把权重更新会话指向投机 draft 模型 ------
        with set_current_vllm_config(self.vllm_config):
            self._start_weight_update(is_draft=True)

    def _start_weight_update(self, is_draft: bool = False) -> None:
        self._check_weight_transfer_engine()
        assert self.weight_transfer_engine is not None

        # ------【权重传输】校验引擎支持 draft 权重更新，否则报错 ------
        if is_draft and not self.weight_transfer_engine.supports_draft_weight_update:
            raise RuntimeError(
                f"{type(self.weight_transfer_engine).__name__} does not support "
                "draft model weight updates."
            )

        # ------【权重传输】防止重入：已有更新会话在进行时报错 ------
        if self._weight_update_active:
            raise RuntimeError(
                "start_weight_update called while a weight update is already "
                "active. Call finish_weight_update first."
            )

        # ------【权重传输】启动引擎更新会话，失败时回滚目标并上抛，成功后记录会话状态 ------
        try:
            if is_draft:
                self._set_draft_weight_update_target()
            self.weight_transfer_engine.start_weight_update()
        except BaseException:
            self.weight_transfer_engine.reset_weight_update_target()
            raise
        self._weight_update_active = True
        self._weight_update_is_draft = is_draft

    def update_weights(self, update_info: dict) -> None:
        """
        Receive one weight update chunk from the trainer.

        start_weight_update must be called before update_weights and
        finish_weight_update must be called after all chunks have been sent.
        Every chunk loads into whichever model the session's start_weight_update
        / start_draft_weight_update call selected.

        Args:
            update_info: Dictionary containing backend-specific update info
        """
        self._check_weight_transfer_engine()
        assert self.weight_transfer_engine is not None

        # ------【权重传输】没有活动会话时收到更新块属于非法调用 ------
        if not self._weight_update_active:
            raise RuntimeError(
                "start_weight_update must be called before update_weights."
            )

        # ------【权重传输】把一块更新数据交给引擎加载，失败则清理会话状态 ------
        with set_current_vllm_config(self.vllm_config):
            try:
                self.weight_transfer_engine.update_weights(update_info)
            except BaseException:
                self._weight_update_active = False
                self.weight_transfer_engine.reset_weight_update_target()
                raise

    def finish_weight_update(self) -> None:
        """Finish the current weight update session."""
        self._check_weight_transfer_engine()
        assert self.weight_transfer_engine is not None

        # ------【权重传输】没有匹配的 start 就 finish 属于非法调用 ------
        if not self._weight_update_active:
            raise RuntimeError(
                "finish_weight_update called without a matching start_weight_update."
            )

        # ------【权重传输】结束会话并复位目标与状态 ------
        with set_current_vllm_config(self.vllm_config):
            self.weight_transfer_engine.finish_weight_update()
            self.weight_transfer_engine.reset_weight_update_target()
            self._weight_update_active = False

        # ------【权重传输】主模型权重被更新后重置 LoRA 状态，避免过期缓存 ------
        # Weight transfer bypasses GPUModelRunner.reload_weights().
        if not self._weight_update_is_draft:
            self.model_runner.reset_lora_state()

    def shutdown(self) -> None:
        # ------【进程管理】解冻 GC 堆，允许后续正常回收 ------
        gc.unfreeze()

        # ------【PD 分离】关闭 KV/EC 传输连接与 profiler ------
        # has_kv_transfer_group can be None during interpreter shutdown.
        if ensure_kv_transfer_shutdown is not None:
            ensure_kv_transfer_shutdown()
        if ensure_ec_transfer_shutdown is not None:
            ensure_ec_transfer_shutdown()
        if self.profiler is not None:
            self.profiler.shutdown()

        # ------【权重传输】关闭权重传输引擎 ------
        if weight_transfer_engine := getattr(self, "weight_transfer_engine", None):
            weight_transfer_engine.shutdown()

        # ------【核心逻辑】释放 model_runner 持有的 GPU 资源，进程内运行时可回收显存 ------
        # Release GPU resources held by the model runner so that memory
        # can be reclaimed when running in-process
        if model_runner := getattr(self, "model_runner", None):
            model_runner.shutdown()

        # ------【内存池/CuMem】在分配器包装器仍存活时主动释放 CuMem 池，避免拖到解释器终结 ------
        # Release kept-alive cumem pools while the pluggable allocator wrappers
        # and callbacks are still alive, so MemPool teardown is not deferred to
        # interpreter finalization (pytorch/pytorch#145168).
        if current_platform.is_cuda_alike():
            from vllm.device_allocator.cumem import CuMemAllocator

            if CuMemAllocator.instance is not None:
                CuMemAllocator.instance.release_pools()

    def elastic_ep_execute(self, execute_method: str, *args, **kwargs):
        # ------【EP/EPLB】把执行委托给弹性 EP 执行器，支持专家规模动态扩缩容 ------
        return self.elastic_ep_executor.execute(execute_method, *args, **kwargs)











def init_worker_distributed_environment(
    vllm_config: VllmConfig,
    rank: int,
    distributed_init_method: str | None = None,
    local_rank: int = -1,
    backend: str = "nccl",
) -> None:
    """Initialize the distributed environment.


        它是个「包装层」，在做真正的建群之前先做几个前置动作：

        init_batch_invariance() — 初始化 batch-invariant 机制（和 CUDA graph 重放相关）
        override_envs_for_eplb(...) — MoE 的 EPLB 负载均衡环境变量
        set_custom_all_reduce(not disable_custom_all_reduce) — 决定是否用 vLLM 自研的 all-reduce kernel 替代 NCCL 原生的（省开销）
        init_distributed_environment(...) → 真正调 torch.distributed.init_process_group(backend="nccl") ← 核心
        ensure_model_parallel_initialized(...) → 在 world group 之上再切 TP/PP/CP 子组

    
    """
    parallel_config = vllm_config.parallel_config
    # ------【CUDA Graph】初始化 batch-invariant 机制：让捕获的图能对不同 batch 重放 ------
    from vllm.model_executor.layers.batch_invariant import init_batch_invariance

    init_batch_invariance()
    # ------【EP/EPLB】按 MoE backend 覆写 EPLB 负载均衡相关环境变量 ------
    override_envs_for_eplb(
        parallel_config,
        moe_backend=getattr(vllm_config.kernel_config, "moe_backend", None),
    )
    # ------【TP】决定是否用 vLLM 自研 all-reduce kernel 替代 NCCL 原生（省通信开销）──
    set_custom_all_reduce(not parallel_config.disable_custom_all_reduce)

    # ------【NCCL 通信】构造 init_method（默认 env://）与超时时间，供建群使用 ------
    init_method = distributed_init_method or "env://"

    timeout = None
    if parallel_config.distributed_timeout_seconds is not None:
        timeout = timedelta(seconds=parallel_config.distributed_timeout_seconds)

    # ------【NCCL 通信】真正调 torch.distributed.init_process_group 拉起 NCCL 通信网（核心建群）──
    # 构造卡的通信网络
    init_distributed_environment(
        parallel_config.world_size, # 并行配置
        rank, # 全局id
        init_method,# 就是每个worker的zmq的url
        local_rank,# 机内id
        backend, # 选择比如nccl还是自己的all-reduce kernel
        timeout,
    )

    # ------【TP + PP】在 world group 之上再切 TP/PP/CP 子通信组，供各层算子按需 all-reduce ------
    ensure_model_parallel_initialized(
        parallel_config.tensor_parallel_size,
        parallel_config.pipeline_parallel_size,
        parallel_config.prefill_context_parallel_size,
        parallel_config.decode_context_parallel_size,
    )

    # ------【PD 分离】在 KV cache 初始化前先建 encoder 传输连接（EPD 分离模式下 encoder 实例不建 KV）──
    # Init ec connector here before KV caches init
    # NOTE: We do not init KV caches for Encoder-only instance in EPD disagg mode
    ensure_ec_transfer_initialized(vllm_config)
