# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import gc
import os
import queue
import signal
import threading
import time
from collections import defaultdict, deque
from collections.abc import Callable, Generator, Sequence
from concurrent.futures import Future
from contextlib import ExitStack, contextmanager
from enum import IntEnum
from functools import partial
from inspect import isclass, signature
from logging import DEBUG
from multiprocessing.queues import Queue
from typing import Any, TypeVar, cast

import msgspec
import zmq

import vllm.envs as envs
from vllm.config import ParallelConfig, VllmConfig
from vllm.config.pooler import POOLER_CONFIG_LOG_FIELDS
from vllm.distributed import (
    cleanup_dist_env_and_memory,
    stateless_destroy_torch_distributed_process_group,
)
from vllm.envs import enable_envs_cache
from vllm.logger import init_logger
from vllm.logging_utils.dump_input import dump_engine_exception
from vllm.lora.request import LoRARequest
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.tasks import POOLING_TASKS, SupportedTask
from vllm.tracing import instrument, maybe_init_worker_tracer
from vllm.transformers_utils.config import maybe_register_config_serialize_by_value
from vllm.utils import numa_utils
from vllm.utils.gc_utils import (
    freeze_gc_heap,
    maybe_attach_gc_debug_callback,
)
from vllm.utils.hashing import get_hash_fn_by_name
from vllm.utils.network_utils import make_zmq_socket
from vllm.utils.system_utils import decorate_logs, set_process_title
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    generate_scheduler_kv_cache_config,
    get_kv_cache_configs,
    get_request_block_hasher,
    init_none_hash,
    resolve_kv_cache_block_sizes,
    update_kv_cache_capacity,
)
from vllm.v1.core.sched.interface import PauseState, SchedulerInterface
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
from vllm.v1.engine import (
    EEP_NOTIFICATION_CALL_ID,
    EEPNotificationType,
    EngineCoreOutput,
    EngineCoreOutputs,
    EngineCoreReadyResponse,
    EngineCoreRequest,
    EngineCoreRequestType,
    FinishReason,
    PauseMode,
    ReconfigureDistributedRequest,
    ReconfigureRankType,
    UtilityOutput,
    UtilityResult,
)
from vllm.v1.engine.tensor_ipc import TensorIpcReceiver
from vllm.v1.engine.utils import (
    EngineHandshakeMetadata,
    EngineZmqAddresses,
    SignalCallback,
    get_physical_gpu_ids_for_local_dp_rank,
)
from vllm.v1.executor import Executor
from vllm.v1.fault_tolerance.engine_core_sentinel import (
    FT_UTILITY_METHOD,
    EngineCoreSentinel,
    fault_tolerant_wrapper,
)
from vllm.v1.kv_cache_interface import KVCacheConfig, get_kv_cache_spec_kind
from vllm.v1.metrics.stats import SchedulerIterationDetails, SchedulerStats
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder, bytestr
from vllm.v1.structured_output import StructuredOutputManager
from vllm.v1.utils import compute_iteration_details
from vllm.version import __version__ as VLLM_VERSION

logger = init_logger(__name__)


HANDSHAKE_TIMEOUT_MINS = 5

_R = TypeVar("_R")  # Return type for collective_rpc


class EngineCore:
    """
    === 类说明 ===
        继承: object
        职责: vLLM 引擎核心。封装 scheduler + model_executor + KV cache，
              提供 step() 驱动一次"调度→前向→更新"循环。

    === 公有方法 (26个) ===
        —— 核心循环 ——
            step()                    — 单步执行：schedule → execute_model → update_from_output
            step_with_batch_queue()   — 带批处理队列的多步执行（流水线并行）
            post_step()               — 步后钩子，处理投机解码 draft token 更新
        —— 请求管理 ——
            add_request()             — 将请求加入调度器
            abort_requests()          — 中止指定请求
            preprocess_add_request()  — 预处理：多模态缓存 + 结构化输出语法编译
            get_supported_tasks()     — 返回模型支持的任务类型
        —— 调度控制 ——
            pause_scheduler()         — 暂停调度（abort/keep 模式）
            resume_scheduler()        — 恢复调度
            is_scheduler_paused()     — 查询暂停状态
            sleep() / wake_up() / is_sleeping() — 睡眠/唤醒（支持三级卸载）
        —— 缓存管理 ——
            reset_prefix_cache()      — 重置 KV 前缀缓存（模型热更新时）
            reset_encoder_cache()     — 重置编码器缓存
            reset_mm_cache()          — 重置多模态缓存
            get_kv_cache_group_metadata() — 返回 KV cache 组元数据
        —— LoRA ——
            add_lora() / remove_lora() / list_loras() / pin_lora()
        —— 其他 ——
            profile()                 — 开启/关闭 profiling
            execute_dummy_batch()     — 执行哑批次前向（DP 同步用）
            save_sharded_state()      — 保存分片模型状态
            collective_rpc()          — 对所有 Worker 发起 RPC
            shutdown()                — 拆除调度器+执行器，释放资源
            set_weight_version() / get_weight_version() — 权重版本管理

    === 核心成员属性 ===
        —— 模型执行 ——
            model_executor: Executor      — 模型执行器（管理 GPU Worker）
            step_fn: Callable             — 指向 step() 或 step_with_batch_queue()
            use_spec_decode: bool         — 是否启用投机解码
            async_scheduling: bool        — 是否启用异步调度
        —— 调度 ——
            scheduler: SchedulerInterface — 调度器实例
            batch_queue: deque | None     — 流水线并行批处理队列
            structured_output_manager     — 结构化输出管理器
        —— KV Cache ——
            available_gpu_memory_for_kv_cache: int — KV cache 可用 GPU 显存
            request_block_hasher          — 前缀缓存的请求块哈希函数
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        executor_class: type[Executor],
        log_stats: bool,
        executor_fail_callback: Callable | None = None,
        include_finished_set: bool = False,
    ):
        # ------ 插件加载：注册用户扩展（核心逻辑，非优化） ------
        # plugins need to be loaded at the engine/scheduler level too
        from vllm.plugins import load_general_plugins

        load_general_plugins()

        self.vllm_config = vllm_config
        if not vllm_config.parallel_config.data_parallel_rank_local:
            logger.info(
                "Initializing a V1 LLM engine (v%s) with config: %s",
                VLLM_VERSION,
                vllm_config,
            )

        self.log_stats = log_stats
        # Opaque weight version supplied by the caller.
        self._weight_version = "default"

        # ------【进程管理 + 异步 RPC】构造执行器：本地代理 worker 进程，后续经 Future/RPC 驱动 ------
        # Setup Model.






        # 1. 执行器->worker->设置好NCCL通信组，初始化每个卡的显存空间，然后加载好模型

        ####################################################################################
        # 1. 构造一个执行器，启动worker，并完成模型构建 + 权重加载
        ####################################################################################
        self.model_executor = executor_class(vllm_config) # 1. 构造一个执行器类
        self._pooler_config_logged = False
        if executor_fail_callback is not None:
            self.model_executor.register_failure_callback(executor_fail_callback)

        # ------【显存 profiling】预留 KV cache 显存容量字段，稍后由 profiling 回填 ------
        self.available_gpu_memory_for_kv_cache = -1 # 2. 每个引擎后端来管理他的逻辑KVcache 的 显存容量

        # ------【EP/EPLB】弹性专家并行扩缩容：在 KV 初始化前先完成 EEP scale-up ------
        if envs.VLLM_ELASTIC_EP_SCALE_UP_LAUNCH:
            self._eep_scale_up_before_kv_init()



        # 2. 执行器->worker->基于剩余的显存，初始化好kvcache
        # Setup KV Caches and update CacheConfig after profiling.
        # ------【显存 profiling】初始化 KV cache（内部做显存 profiling + 预热），返回配置 ------

        ####################################################################################
        # 2. 开始做 profiling, 初始化kv cache
        ####################################################################################
        kv_cache_config = self._initialize_kv_caches(vllm_config) # 驱动执行器去初始化kv cache





        # ------【结构化输出/grammar】构造结构化输出管理器，编译/管理 grammar bitmask ------
        self.structured_output_manager = StructuredOutputManager(vllm_config) # 结构化输出管理器

        # ------ 核心：构造调度器（连续批处理、请求队列、KV 块分配都在其中） ------
        # Setup scheduler.
        Scheduler = vllm_config.scheduler_config.get_scheduler_cls()

        # ------【chunked prefill】无 KV cache 的模型不支持 chunked prefill，自动关闭 ------
        if len(kv_cache_config.kv_cache_groups) == 0:  # noqa: SIM102
            # Encoder models without KV cache don't support
            # chunked prefill. But do SSM models?
            if vllm_config.scheduler_config.enable_chunked_prefill:
                logger.warning("Disabling chunked prefill for model without KVCache")
                vllm_config.scheduler_config.enable_chunked_prefill = False

        # ------【前缀缓存】解析调度块大小与哈希块大小：前缀缓存按更粗粒度 hash 复用 KV ------
        scheduler_block_size, hash_block_size = resolve_kv_cache_block_sizes(
            kv_cache_config, vllm_config
        )








        
        ####################################################################################
        # 3. 构造一个调度器实例
        ####################################################################################
        self.scheduler: SchedulerInterface = Scheduler( 
            vllm_config=vllm_config,
            kv_cache_config=kv_cache_config,
            structured_output_manager=self.structured_output_manager,
            include_finished_set=include_finished_set,
            log_stats=self.log_stats,
            block_size=scheduler_block_size,
            hash_block_size=hash_block_size,
        )
        # ------【投机解码】标记是否启用投机解码/diffusion，决定 step 后是否取 draft token ------
        self.use_spec_decode = vllm_config.speculative_config is not None
        self.check_for_draft_tokens = (
            self.use_spec_decode or vllm_config.model_config.is_diffusion
        )
        # ------【PD 分离】调度器带 connector 时，让执行器聚合 KV 输出供远端 decoder 消费 ------
        if self.scheduler.connector is not None:  # type: ignore
            self.model_executor.init_kv_output_aggregator(self.scheduler.connector)  # type: ignore

        # ------【核心逻辑】多模态接收缓存：跨进程复用已编码特征，避免重复编码 ------
        mm_registry = MULTIMODAL_REGISTRY
        self.mm_receiver_cache = mm_registry.engine_receiver_cache_from_config(
            vllm_config
        )

        # If a KV connector is initialized for scheduler, we want to collect
        # handshake metadata from all workers so the connector in the scheduler
        # will have the full context
        # ------【PD 分离】收集各 worker 的 KV 传输握手元数据，供 prefill→decode 跨实例传输 ------
        kv_connector = self.scheduler.get_kv_connector() # 这个是调度器里的 KVcache的跨引擎传输用的，用于PD分离的通信管道
        if kv_connector is not None:
            # Collect and store KV connector xfer metadata from workers
            # (after KV cache registration)
            xfer_handshake_metadata = (
                self.model_executor.get_kv_connector_handshake_metadata()
            )

            if xfer_handshake_metadata:
                # xfer_handshake_metadata is list of dicts from workers
                # Each dict already has structure {(pp_rank, tp_rank): metadata}
                # Merge all worker dicts into a single dict
                content: dict[tuple[int, int], Any] = {}
                for worker_dict in xfer_handshake_metadata:
                    if worker_dict is not None:
                        content.update(worker_dict)
                kv_connector.set_xfer_handshake_metadata_pp_aware(content)

        # ------【PP】流水线并行的批处理队列：异步调度/执行以消除气泡，单卡关闭 ------
        # Setup batch queue for pipeline parallelism.
        # Batch queue for scheduled batches. This enables us to asynchronously
        # schedule and execute batches, and is required by pipeline parallelism
        # to eliminate pipeline bubbles.

        # 调度器的调度队列在Scheduler里面，这个是流水线并行的执行缓存，单卡不开，跳过

        self.batch_queue_size = vllm_config.max_concurrent_batches
        self.batch_queue: ( 
            deque[tuple[Future[ModelRunnerOutput], SchedulerOutput, Future[Any]]] | None
        ) = None
        if self.batch_queue_size > 1:
            logger.debug("Batch queue is enabled with size %d", self.batch_queue_size)
            self.batch_queue = deque(maxlen=self.batch_queue_size)

        # ------【核心逻辑】记录两类模型特征：是否 EC 消费者、是否 pooling 模型，供批队列分支判断 ------
        self.is_ec_consumer = (
            vllm_config.ec_transfer_config is None
            or vllm_config.ec_transfer_config.is_ec_consumer
        )
        self.is_pooling_model = vllm_config.model_config.runner_type == "pooling"

        # ------【前缀缓存】按 block hash 复用已算 KV，命中前缀只算增量 token ------
        # 4. 前缀缓存的哈希函数
        self.request_block_hasher: Callable[[Request], list[BlockHash]] | None = None
        if vllm_config.cache_config.enable_prefix_caching or kv_connector is not None:
            caching_hash_fn = get_hash_fn_by_name(
                vllm_config.cache_config.prefix_caching_hash_algo
            )
            init_none_hash(caching_hash_fn)

            self.request_block_hasher = get_request_block_hasher(
                hash_block_size, caching_hash_fn
            )













        # 4. 定义好引擎后端执行一步的逻辑
        # ------【PP】选择主循环：有批处理队列走 step_with_batch_queue，否则走 step ------
        # 选择引擎的主循环用哪个函数
        self.step_fn = (
            self.step if self.batch_queue is None else self.step_with_batch_queue
        )
        self.async_scheduling = vllm_config.scheduler_config.async_scheduling

        self.aborts_queue = queue.Queue[list[str]]()

        self._idle_state_callbacks: list[Callable] = []

        # ------ GC 优化：冻结启动期堆内存，减少老年代 GC 停顿（非并行优化） ------
        # Mark the startup heap as static so that it's ignored by GC.
        # Reduces pause times of oldest generation collections.
        freeze_gc_heap()
        # If enable, attach GC debugger after static variable freeze.
        maybe_attach_gc_debug_callback()
        # Enable environment variable cache (e.g. assume no more
        # environment variable overrides after this point)
        enable_envs_cache()



    # 初始化kv cache
    @instrument(span_name="Prepare model")
    def _initialize_kv_caches(self, vllm_config: VllmConfig) -> KVCacheConfig:
        # 【Worker 初始化 · 阶段 3/3】Initialize KV Cache
        #   注意: 阶段 1(init_device)/2(load_model) 是 Worker 进程启动时自己直接调的;
        #   阶段 3 由 EngineCore 编排, 通过 collective_rpc 远程驱动 worker:
        #   get_kv_cache_specs → determine_available_memory(profile) → initialize_from_config → compile_or_warm_up_model
        start = time.time()

        # ------ 在 enginecore 进程内注册所有 KV cache spec 类型 ------
        # register all kvcache specs in enginecore process.
        register_all_kvcache_specs(vllm_config)



        
        # Get all kv cache needed by the model
        ############################################################################
        # 1. 获取各个group的规格
        ############################################################################
        kv_cache_specs = self.model_executor.get_kv_cache_specs() # 拿到我们各个group的kvcache的规格





        # Some layers (e.g. Prefix LM attention) run non-causally and tag their
        # KV cache spec with ``non_causal=True``. The specs are collected here in
        # the engine-core process (the same process that builds the scheduler),
        # so this is the multiproc-safe place to translate that layer-level
        # signal into a scheduling policy: chunked prefill and prefix caching
        # both assume causal attention and would corrupt non-causal prefill.
        # ------【chunked prefill + 前缀缓存】非因果注意力层会破坏这两种策略，检测到即自动关闭 ------
        if any(
            getattr(spec, "non_causal", False)
            for worker_specs in kv_cache_specs
            for spec in worker_specs.values()
        ):
            if vllm_config.scheduler_config.enable_chunked_prefill:
                logger.info(
                    "Disabling chunked prefill: model has non-causal attention layers."
                )
                vllm_config.scheduler_config.enable_chunked_prefill = False
            if vllm_config.cache_config.enable_prefix_caching:
                logger.info(
                    "Disabling prefix caching: model has non-causal attention layers."
                )
                vllm_config.cache_config.enable_prefix_caching = False

        # ------【显存 profiling】探测模型峰值显存，算出可分配给 KV cache 的余量 ------
        has_kv_cache = any(kv_cache_spec for kv_cache_spec in kv_cache_specs)
        if has_kv_cache:
            # ------【EP/EPLB】弹性 EP scale-up：KV 显存已在 pre-kv-init 阶段算好，直接复用 ------
            if envs.VLLM_ELASTIC_EP_SCALE_UP_LAUNCH:
                # NOTE(yongji): should already be set
                # during _eep_scale_up_before_kv_init
                assert self.available_gpu_memory_for_kv_cache > 0
                available_gpu_memory = [self.available_gpu_memory_for_kv_cache] * len(
                    kv_cache_specs
                )
            else:
                # Profiles the peak memory usage of the model to determine how
                # much memory can be allocated for kv cache.

                ####################################################################################
                # 2. profiling, 开始测试峰值缓冲区占用下，kvcache能用的显存空间
                ####################################################################################
                available_gpu_memory = self.model_executor.determine_available_memory() # 开始测试当前GPU有多少显存可以供给kvcache使用
                self.available_gpu_memory_for_kv_cache = available_gpu_memory[0]




        else:
            # Attention free models don't need memory for kv cache
            available_gpu_memory = [0] * len(kv_cache_specs)

        assert len(kv_cache_specs) == len(available_gpu_memory)

        # ------【异步 RPC】auto-fit 若压缩 max_model_len，广播新值回各 worker ------
        # Track max_model_len before KV cache config to detect auto-fit changes
        max_model_len_before = vllm_config.model_config.max_model_len

        ################################################################################################################
        # 3. 根据空闲显存，更新kv cache config配置
        ################################################################################################################
        kv_cache_configs = get_kv_cache_configs(
            vllm_config, kv_cache_specs, available_gpu_memory
        )

        # If auto-fit reduced max_model_len, sync the new value to workers.
        # This is needed because workers were spawned before memory profiling
        # and have the original (larger) max_model_len cached.
        max_model_len_after = vllm_config.model_config.max_model_len
        if max_model_len_after != max_model_len_before:
            self.collective_rpc("update_max_model_len", args=(max_model_len_after,))

        # ------ 生成调度器侧 KV cache 配置并回写 num_gpu_blocks ------
        scheduler_kv_cache_config = generate_scheduler_kv_cache_config(kv_cache_configs)
        vllm_config.cache_config.num_gpu_blocks = scheduler_kv_cache_config.num_blocks
        kv_cache_groups = scheduler_kv_cache_config.kv_cache_groups
        if kv_cache_groups:
            vllm_config.cache_config.block_size = min(
                g.kv_cache_spec.block_size for g in kv_cache_groups
            )
            update_kv_cache_capacity(vllm_config, scheduler_kv_cache_config)

        vllm_config.validate_block_size()






        # ------【异步 RPC】下发 KV cache 配置到 worker 分配显存 ------

        ############################################################################################################################################
        # 4. 开始让执行器通知worker，开始初始化kvcache 的显存空间，开始切分block
        ############################################################################################################################################
        self.model_executor.initialize_from_config(kv_cache_configs)



        # ------【CUDA Graph】编译/预热模型：捕获 CUDA Graph 供后续 step 回放 ------
        if not envs.VLLM_ELASTIC_EP_SCALE_UP_LAUNCH:
            self.model_executor.compile_or_warm_up_model()

        # ------【核心逻辑】统计初始化各阶段耗时并打印（profile/create/warmup + 编译时间） ------
        elapsed = time.time() - start
        compile_time = vllm_config.compilation_config.compilation_time
        encoder_compile_time = vllm_config.compilation_config.encoder_compilation_time
        if encoder_compile_time > 0:
            logger.info_once(
                "init engine (profile, create kv cache, warmup model) took "
                "%.2f s (compilation: %.2f s — language_model: %.2f s, "
                "encoder: %.2f s)",
                elapsed,
                compile_time + encoder_compile_time,
                compile_time,
                encoder_compile_time,
            )
        elif compile_time > 0:
            logger.info_once(
                "init engine (profile, create kv cache, warmup model) took "
                "%.2f s (compilation: %.2f s)",
                elapsed,
                compile_time,
            )
        else:
            logger.info_once(
                "init engine (profile, create kv cache, warmup model) took %.2f s",
                elapsed,
            )
        return scheduler_kv_cache_config









    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        supported_tasks = self.model_executor.supported_tasks
        self._log_pooler_config(supported_tasks)
        return supported_tasks

    def _log_pooler_config(self, supported_tasks: tuple[SupportedTask, ...]) -> None:
        # ------【核心逻辑】只打印一次：非 pooling/DP 本地 rank/无 pooler 配置则直接跳过 ------
        if self._pooler_config_logged:
            return

        model_config = self.vllm_config.model_config
        pooler_config = model_config.pooler_config
        if (
            self.vllm_config.parallel_config.data_parallel_rank_local
            or model_config.runner_type != "pooling"
            or pooler_config is None
        ):
            return

        supported_pooling_tasks = tuple(
            sorted(set(supported_tasks) & set(POOLING_TASKS))
        )
        if not supported_pooling_tasks:
            return

        self._pooler_config_logged = True
        task_set = set(supported_pooling_tasks)
        use_activation = pooler_config.use_activation
        if use_activation is None:
            use_activation = True
        sources = getattr(model_config, "_pooler_config_sources", {})
        pooling_type_field = (
            "seq_pooling_type"
            if task_set & {"embed", "classify"}
            else "tok_pooling_type"
        )

        def log_field(name: str, field: str) -> str:
            value = (
                use_activation
                if field == "use_activation"
                else getattr(pooler_config, field)
            )
            source = sources.get(field, "unknown")
            return f"{name}={value}(source={source})"

        log_items = [("pooling_type", pooling_type_field)]
        log_items.extend(
            (field, field)
            for field in POOLER_CONFIG_LOG_FIELDS
            if field != pooling_type_field
        )
        config_fields = ", ".join(log_field(name, field) for name, field in log_items)

        logger.info_once(
            "Resolved pooling config: %s, supported_tasks=%s",
            config_fields,
            supported_pooling_tasks,
        )

    def get_kv_cache_group_metadata(self) -> list[dict[str, int | str | None]]:
        """Return msgspec-serializable metadata for scheduler KV cache groups."""
        kv_cache_config = getattr(self.scheduler, "kv_cache_config", None)
        if kv_cache_config is None:
            return []

        # ------ 遍历每个 KV cache 组，导出可序列化元数据（类型/块大小/滑窗） ------
        metadata: list[dict[str, int | str | None]] = []
        for group_idx, group in enumerate(kv_cache_config.kv_cache_groups):
            spec = group.kv_cache_spec
            metadata.append(
                {
                    "group_idx": group_idx,
                    "kind": get_kv_cache_spec_kind(spec).value,
                    "block_size": spec.block_size,
                    "sliding_window": getattr(spec, "sliding_window", None),
                }
            )
        return metadata


    # 指令是ADD的请求
    def add_request(self, request: Request, request_wave: int = 0):
        """Add request to the scheduler.

        `request_wave`: indicate which wave of requests this is expected to
        belong to in DP case
        """
        # Validate the request_id type.
        if not isinstance(request.request_id, str):
            raise TypeError(
                f"request_id must be a string, got {type(request.request_id)}"
            )

        if pooling_params := request.pooling_params:
            supported_pooling_tasks = [
                task for task in self.get_supported_tasks() if task in POOLING_TASKS
            ]

            if pooling_params.task not in supported_pooling_tasks:
                raise ValueError(
                    f"Unsupported task: {pooling_params.task!r} "
                    f"Supported tasks: {supported_pooling_tasks}"
                )

        # ------【PD 分离】带 KV/EC 传输参数但无对应 connector 时降级为本地推理并告警 ------
        if request.kv_transfer_params is not None and (
            not self.scheduler.get_kv_connector()
        ):
            logger.warning(
                "Got kv_transfer_params, but no KVConnector found. "
                "Disabling KVTransfer for this request."
            )

        if (
            request.ec_transfer_params is not None
            and self.scheduler.get_ec_connector() is None
        ):
            logger.warning(
                "Got ec_transfer_params, but no ECConnector found. "
                "Disabling ECTransfer for this request."
            )

        # 实际把他加入调度器的队列里面
        self.scheduler.add_request(request)
        if request.abort_immediately:
            # Immediately abort so the connector's request_finished hook runs
            # to free any pre-admission KV-transfer resources.
            self.abort_requests([request.request_id])



    # abort请求的增加，就是让调度器结束一个请求
    def abort_requests(self, request_ids: list[str]):
        """Abort requests from the scheduler."""

        # TODO: The scheduler doesn't really need to know the
        # specific finish reason, TBD whether we propagate that
        # (i.e. client-aborted vs stop criteria met).
        self.scheduler.finish_requests(request_ids, RequestStatus.FINISHED_ABORTED)

    @contextmanager
    def log_error_detail(self, scheduler_output: SchedulerOutput):
        """Execute the model and log detailed info on failure."""
        # ------ 失败时 dump 引擎异常详情（调度输出+统计），便于定位后再原样重抛 ------
        try:
            yield
        except Exception as err:
            # We do not want to catch BaseException here since we're only
            # interested in dumping info when the exception is due to an
            # error from execute_model itself.

            # NOTE: This method is exception-free
            dump_engine_exception(
                self.vllm_config, scheduler_output, self.scheduler.make_stats()
            )
            raise err

    @contextmanager
    def capture_iteration_details(
        self, scheduler_output: SchedulerOutput | None
    ) -> Generator[SchedulerIterationDetails | None, None, None]:
        enable_details = (
            self.vllm_config.observability_config.enable_logging_iteration_details
        )
        if not self.log_stats or not enable_details:
            yield None
            return
        # 0-token step: let the dummy_batch wrapper log it (avoids double-log).
        if (
            scheduler_output is not None
            and scheduler_output.total_num_scheduled_tokens == 0
        ):
            yield None
            return

        iteration_index = getattr(self, "_iteration_index", 0)
        # scheduler_output=None marks a DP dummy iteration.
        if scheduler_output is None:
            iteration_details = SchedulerIterationDetails(
                iteration_index=iteration_index,
                num_ctx_requests=0,
                num_ctx_tokens=0,
                num_generation_requests=0,
                num_generation_tokens=0,
                elapsed_ms=0.0,
                is_dummy=True,
            )
        else:
            # ------ 有真实调度输出：按调度结果计算迭代统计字段 ------
            details = compute_iteration_details(scheduler_output)
            iteration_details = SchedulerIterationDetails(
                iteration_index=iteration_index,
                num_ctx_requests=details.num_ctx_requests,
                num_ctx_tokens=details.num_ctx_tokens,
                num_generation_requests=details.num_generation_requests,
                num_generation_tokens=details.num_generation_tokens,
                elapsed_ms=0.0,
                num_encoder_inputs=details.num_encoder_inputs,
                num_encoder_output_tokens=details.num_encoder_output_tokens,
            )

        start_time = time.monotonic()
        yield iteration_details
        iteration_details.elapsed_ms = (time.monotonic() - start_time) * 1000
        self._iteration_index = iteration_index + 1

    def _make_iteration_details_stats(
        self, iteration_details: SchedulerIterationDetails
    ) -> SchedulerStats:
        stats = self.scheduler.make_stats() or SchedulerStats()
        stats.iteration_details = iteration_details
        return stats

    def _attach_iteration_details(
        self,
        outputs: dict[int, EngineCoreOutputs],
        iteration_details: SchedulerIterationDetails | None,
    ) -> None:
        if iteration_details is None:
            return

        if (eco := next(iter(outputs.values()), None)) is None:
            outputs[0] = eco = EngineCoreOutputs()
        if eco.scheduler_stats is None:
            eco.scheduler_stats = self._make_iteration_details_stats(iteration_details)
        else:
            eco.scheduler_stats.iteration_details = iteration_details

    def _should_throttle_prefills(self) -> bool:
        """Whether to defer new prefills this step (DP prefill balancing).
        Overridden by the DP engine core; never throttles otherwise."""
        return False







    # 引擎实际的step 方法
    def step(self) -> tuple[dict[int, EngineCoreOutputs], bool]:
        """Schedule, execute, and make output.

        Returns tuple of outputs and a flag indicating whether the model
        was executed.
        """

        # Check for any requests remaining in the scheduler - unfinished,
        # or finished and not yet removed from the batch.
        # ------ 空转检查：无请求直接返回，不触达 GPU ------
        if not self.scheduler.has_requests():
            return {}, False

        # 这里就已经实现了continuous batching了，
        # 调度器调度一次
        # 执行器执行一个step

        # ------ 连续批处理调度：选取请求 + 分配 KV cache 块（核心逻辑） ------
        # 1. 调度器调度：选取请求 + 分配KVcache
        scheduler_output = self.scheduler.schedule(self._should_throttle_prefills())



        # 2. 执行器执行：GPU前向推理, 返回异步RPC的future
        # ------【异步 RPC】非阻塞提交 GPU 前向，返回 Future，与采样/下一步调度流水重叠 ------
        future = self.model_executor.execute_model(scheduler_output, non_block=True)
        # ------【结构化输出/grammar】取语法 bitmask，约束采样 token 符合 schema ------
        grammar_output = self.scheduler.get_grammar_bitmask(scheduler_output) # 获取语法模版



        with (
            self.capture_iteration_details(scheduler_output) as iteration_details,
            self.log_error_detail(scheduler_output),
        ):
            # ------ 阻塞等待 Future 结果；未采样则在此处采样（logits→token） ------
            model_output = future.result() # 阻塞等待本轮调度的结果
            # 因为model runner返回的是未采样的logits这个分布状态，被存放到self.execute_model_state, 所以返回是空的
            if model_output is None:
                model_output = self.model_executor.sample_tokens(grammar_output)

        # ------ 处理执行期间到达的中止请求 ------
        # Before processing the model output, process any aborts that happened
        # during the model execution.
        self._process_aborts_queue()

        # ------ 用模型输出回填调度器状态（完成/计数/前缀缓存命中） ------
        # 3. 根据执行器结果，更新调度器的计数状态
        engine_core_outputs = self.scheduler.update_from_output(
            scheduler_output, model_output
        )
        self._attach_iteration_details(engine_core_outputs, iteration_details)


        # 4. 返回结果
        return engine_core_outputs, scheduler_output.total_num_scheduled_tokens > 0









    def post_step(self, model_executed: bool) -> None:
        # ------【投机解码】非异步调度且本步有前向时，取回 draft token 回填调度器 ------
        # When using async scheduling we can't get draft token ids in advance,
        # so we update draft token ids in the worker process and don't
        # need to update draft token ids here.
        if self.check_for_draft_tokens and not self.async_scheduling and model_executed:
            draft_token_ids = self.model_executor.take_draft_token_ids()
            if draft_token_ids is not None:
                self.scheduler.update_draft_token_ids(draft_token_ids)

    def step_with_batch_queue(
        self,
    ) -> tuple[dict[int, EngineCoreOutputs] | None, bool]:
        """Schedule and execute batches with the batch queue.
        Note that if nothing to output in this step, None is returned.

        The execution flow is as follows:
        1. Try to schedule a new batch if the batch queue is not full.
        If a new batch is scheduled, directly return an empty engine core
        output. In other words, fulfilling the batch queue has a higher priority
        than getting model outputs.
        2. If there is no new scheduled batch, meaning that the batch queue
        is full or no other requests can be scheduled, we block until the first
        batch in the job queue is finished.
        3. Update the scheduler from the output.
        """

        # ------【PP】批队列主循环：异步调度/执行多批消除气泡；此处校验队列存在且未满 ------
        batch_queue = self.batch_queue
        assert batch_queue is not None

        # Try to schedule a new batch if the batch queue is not full, but
        # the scheduler may return an empty batch if all requests are scheduled.
        # Note that this is not blocking.
        assert len(batch_queue) < self.batch_queue_size

        model_executed = False
        deferred_scheduler_output = None
        # ------ 有请求即调度一批（非阻塞）；否则走下方「等待已有批次结果」分支 ------
        if self.scheduler.has_requests():
            scheduler_output = self.scheduler.schedule(self._should_throttle_prefills())
            # ------【异步 RPC】非阻塞提交 GPU 前向，返回 Future 供后续消费 ------
            with self.log_error_detail(scheduler_output):
                exec_future = self.model_executor.execute_model(
                    scheduler_output, non_block=True
                )
            # ------【核心逻辑】EC 消费者按「是否真调了 token」判定本步是否执行前向 ------
            if self.is_ec_consumer:
                model_executed = scheduler_output.total_num_scheduled_tokens > 0

            # ------【结构化输出/grammar】pooling/空批免采样；否则取 bitmask 立即采样，缺 token 则延迟 ------
            if self.is_pooling_model or not model_executed:
                # No sampling required (no requests scheduled).
                future = cast(Future[ModelRunnerOutput], exec_future)
            else:
                if not scheduler_output.pending_structured_output_tokens:
                    # We aren't waiting for any tokens, get any grammar output
                    # and sample immediately.
                    grammar_output = self.scheduler.get_grammar_bitmask(
                        scheduler_output
                    )
                    future = self.model_executor.sample_tokens(
                        grammar_output, non_block=True
                    )
                # ------ 缺上一轮 token，暂存 scheduler_output，待前向结果回来后再采样 ------
                else:
                    # We need to defer sampling until we have processed the model output
                    # from the prior step.
                    deferred_scheduler_output = scheduler_output

            # ------【PP】未延迟采样则入队；队列未满且仍有活时直接返回，优先填满队列而非取结果 ------
            if not deferred_scheduler_output:
                # Add this step's future to the queue.
                batch_queue.appendleft((future, scheduler_output, exec_future))
                if len(batch_queue) < self.batch_queue_size and (
                    model_executed or self.scheduler.has_requests()
                ):
                    # Don't block on next worker response unless the queue is full
                    # or there are no more requests to schedule.
                    return None, model_executed

        # ------ 调度器无请求且队列空：无活可干，返回空结果 ------
        elif not batch_queue:
            # Queue is empty. We should not reach here since this method should
            # only be called when the scheduler contains requests or the queue
            # is non-empty.
            return None, False

        # Block until the next result is available.
        # ------ 阻塞等待队首批次完成，取出其 Future 结果与对应调度输出 ------
        future, scheduler_output, exec_model_fut = batch_queue.pop()
        with (
            self.capture_iteration_details(scheduler_output) as iteration_details,
            self.log_error_detail(scheduler_output),
        ):
            model_output = future.result()
            if model_output is None:
                # None from sample_tokens() implies that the original execute_model()
                # call failed - raise that exception.
                exec_model_fut.result()
                raise RuntimeError("unexpected error")

        # Before processing the model output, process any aborts that happened
        # during the model execution.
        # ------ 先处理执行期中止请求，再用模型输出回填调度器并附迭代统计 ------
        self._process_aborts_queue()
        engine_core_outputs = self.scheduler.update_from_output(
            scheduler_output, model_output
        )
        self._attach_iteration_details(engine_core_outputs, iteration_details)

        # NOTE(nick): We can either handle the deferred tasks here or save
        # in a field and do it immediately once step_with_batch_queue is
        # re-called. The latter slightly favors TTFT over TPOT/throughput.
        # ------【投机解码 + 结构化输出/grammar】补做延迟采样：先校验 draft token 再取 bitmask 采样入队 ------
        if deferred_scheduler_output:
            # When draft tokens are used with structured output, validate them
            # before computing the grammar bitmask for the deferred request.
            if self.check_for_draft_tokens:
                draft_token_ids = self.model_executor.take_draft_token_ids()
                if draft_token_ids is not None:
                    # Update the draft token ids in the scheduler output to
                    # filter out the invalid spec tokens, which will be padded
                    # with -1 and skipped by the grammar bitmask computation.
                    self.scheduler.update_draft_token_ids_in_output(
                        draft_token_ids, deferred_scheduler_output
                    )
            # We now have the tokens needed to compute the bitmask for the
            # deferred request. Get the bitmask and call sample tokens.
            grammar_output = self.scheduler.get_grammar_bitmask(
                deferred_scheduler_output
            )
            future = self.model_executor.sample_tokens(grammar_output, non_block=True)
            batch_queue.appendleft((future, deferred_scheduler_output, exec_future))

        return engine_core_outputs, model_executed

    def _process_aborts_queue(self):
        # ------【核心逻辑】把执行期间积压的中止请求一次性批量 abort，摊薄中止开销 ------
        if not self.aborts_queue.empty():
            request_ids = []
            while not self.aborts_queue.empty():
                ids = self.aborts_queue.get_nowait()
                # Should be a list here, but also handle string just in case.
                request_ids.extend((ids,) if isinstance(ids, str) else ids)
            # More efficient to abort all as a single batch.
            self.abort_requests(request_ids)

    def shutdown(self):
        logger.debug_once("[shutdown] EngineCore: tearing down local resources")
        # ------【核心逻辑】逆序拆除：先清 grammar 后端，再关执行器与调度器，释放 GPU/线程资源 ------
        self.structured_output_manager.clear_backend()
        if self.model_executor:
            self.model_executor.shutdown()
        if self.scheduler:
            self.scheduler.shutdown()

        # ------ GC 优化：解除启动期 gc.freeze，让模型权重/KV 缓存重新可回收，防显存泄漏 ------
        # Undo the gc.freeze() from __init__ so that the objects allocated
        # during engine startup (model weights, KV caches, etc.) become
        # visible to the garbage collector again. Without this, deleting
        # the engine in-process (e.g. unit tests) leaks GPU memory.
        gc.unfreeze()
        # Tear down distributed state initialized in this EngineCore process
        # before it exits and release cached memory.
        cleanup_dist_env_and_memory()
        logger.debug_once("[shutdown] EngineCore: local resource teardown complete")

    def profile(self, is_start: bool = True, profile_prefix: str | None = None):
        self.model_executor.profile(is_start, profile_prefix)

    def reset_mm_cache(self):
        # NOTE: Since this is mainly for debugging, we don't attempt to
        # re-sync the internal caches (P0 sender, P1 receiver)
        if self.scheduler.has_unfinished_requests():
            logger.warning(
                "Resetting the multi-modal cache when requests are "
                "in progress may lead to desynced internal caches."
            )

        # The cache either exists in EngineCore or WorkerWrapperBase
        if self.mm_receiver_cache is not None:
            self.mm_receiver_cache.clear_cache()

        self.model_executor.reset_mm_cache()

    def reset_prefix_cache(
        self, reset_running_requests: bool = False, reset_connector: bool = False
    ) -> bool:
        return self.scheduler.reset_prefix_cache(
            reset_running_requests, reset_connector
        )

    def reset_encoder_cache(self) -> None:
        """Reset the encoder cache to invalidate all cached encoder outputs.

        This should be called when model weights are updated to ensure
        stale vision embeddings computed with old weights are not reused.
        Clears both the scheduler's cache manager and the GPU model runner's cache.
        """
        # NOTE: Since this is mainly for debugging, we don't attempt to
        # re-sync the internal caches (P0 sender, P1 receiver)
        if self.scheduler.has_unfinished_requests():
            logger.warning(
                "Resetting the encoder cache when requests are "
                "in progress may lead to desynced internal caches."
            )

        # Reset the scheduler's encoder cache manager (logical state)
        self.scheduler.reset_encoder_cache()
        # Reset the GPU model runner's encoder cache (physical storage)
        self.model_executor.reset_encoder_cache()

    def _reset_caches(
        self,
        reset_running_requests: bool = True,
        reset_connector: bool = True,
    ) -> None:
        # reset_connector=True so external connectors clear alongside
        # local caches, matching the pause_generation(clear_cache=True)
        # contract. No-op when no connector is configured.
        self.reset_prefix_cache(
            reset_running_requests=reset_running_requests,
            reset_connector=reset_connector,
        )
        self.reset_mm_cache()
        self.reset_encoder_cache()

    def pause_scheduler(
        self, mode: PauseMode = "abort", clear_cache: bool = True
    ) -> Future | None:
        """Pause generation; behavior depends on mode.

        All pause modes queue new adds -- "abort" and "keep" skip step();
        "wait" allows step() so in-flight requests can drain.

        - ``abort``: Set PAUSED_NEW, abort all requests, wait for abort
          outputs to be sent (when running with output_queue), optionally
          clear caches, then complete the returned Future.
        - ``wait``: Set PAUSED_NEW (queue adds, keep stepping); when drained,
          optionally clear caches, then complete the returned Future.
        - ``keep``: Set PAUSED_ALL; return a Future that completes when the
          output queue is empty.
        """
        if mode not in ("keep", "abort", "wait"):
            raise ValueError(f"Invalid pause mode: {mode}")
        if mode == "wait":
            raise ValueError("'wait' mode can't be used in inproc-engine mode")

        # ------ abort 模式：立即中止所有在途请求，再进入暂停态 ------
        if mode == "abort":
            self.scheduler.finish_requests(None, RequestStatus.FINISHED_ABORTED)

        # ------ 按模式设置暂停状态；需要时清空各类缓存，保证休眠前不留脏数据 ------
        pause_state = PauseState.PAUSED_ALL if mode == "keep" else PauseState.PAUSED_NEW
        self.scheduler.set_pause_state(pause_state)
        if clear_cache:
            self._reset_caches()

        return None

    def resume_scheduler(self) -> None:
        """Resume the scheduler and flush any requests queued while paused."""
        # ------ 恢复调度：切回 UNPAUSED，释放暂停期间积压的请求 ------
        self.scheduler.set_pause_state(PauseState.UNPAUSED)

    def is_scheduler_paused(self) -> bool:
        """Return whether the scheduler is in any pause state."""
        return self.scheduler.pause_state != PauseState.UNPAUSED

    def sleep(self, level: int = 1, mode: PauseMode = "abort") -> None | Future:
        """Put the engine to sleep at the specified level.

        Args:
            level: Sleep level.
                - Level 0: Pause scheduling only. Requests are still accepted
                           but not processed. No GPU memory changes.
                - Level 1: Offload model weights to CPU, discard KV cache.
                - Level 2: Discard all GPU memory.
            mode: Pause mode - how to deal with any existing requests, see
                documentation of pause_scheduler method.
        """

        # Pause scheduler before sleeping.
        # ------【显存 profiling】level>=1 会卸载 KV，故先清前缀缓存；level 0 仅暂停调度不动显存 ------
        clear_prefix_cache = level >= 1
        pause_future = self.pause_scheduler(mode=mode, clear_cache=clear_prefix_cache)
        # ------【显存 profiling】level 0：只暂停调度、保留显存，直接返回暂停 Future ------
        if level < 1:
            return pause_future

        # Level 1+: Delegate to executor for GPU memory management
        model_executor = self.model_executor
        # ------【显存 profiling】暂停已同步完成：立即让执行器按 level 卸载权重/KV 到 CPU ------
        if pause_future is None:
            model_executor.sleep(level)
            return None

        # ------【异步 RPC】暂停尚未完成（有在途请求）：挂回调，等暂停完成后再执行卸载 ------
        future = Future[Any]()

        def pause_complete(f: Future):
            try:
                f.result()  # propagate any exception
                future.set_result(model_executor.sleep(level))
            except Exception as e:
                future.set_exception(e)

        logger.info("Waiting for in-flight requests to complete before sleeping...")
        pause_future.add_done_callback(pause_complete)
        return future

    def wake_up(self, tags: list[str] | None = None):
        """Wake up the engine from sleep.

        Args:
            tags: Tags to wake up. Use ["scheduling"] for level 0 wake up.
        """
        # ------【显存 profiling】剥离 scheduling 标签：它只恢复调度、不涉及显存回驻 ------
        if tags is not None and "scheduling" in tags:
            # Remove "scheduling" from tags if there are other tags to process.
            tags = [t for t in tags if t != "scheduling"]

        # ------【显存 profiling】有具体标签时按标签唤醒对应显存；无标签则整体唤醒 ------
        if tags is None or tags:
            self.model_executor.wake_up(tags)

        # Partial wakes intentionally keep the remaining allocations asleep.
        # Resume scheduling only once all executor memory is resident again.
        # ------ 全部显存回驻后才恢复调度，避免部分唤醒状态下调度出错 ------
        if not self.model_executor.is_sleeping:
            self.resume_scheduler()

    def is_sleeping(self) -> bool:
        """Check if engine is sleeping at any level."""
        return self.is_scheduler_paused() or self.model_executor.is_sleeping

    def execute_dummy_batch(self):
        self.model_executor.execute_dummy_batch()

    def add_lora(self, lora_request: LoRARequest) -> bool:
        return self.model_executor.add_lora(lora_request)

    def remove_lora(self, lora_id: int) -> bool:
        return self.model_executor.remove_lora(lora_id)

    def list_loras(self) -> set[int]:
        return self.model_executor.list_loras()

    def pin_lora(self, lora_id: int) -> bool:
        return self.model_executor.pin_lora(lora_id)

    def save_sharded_state(
        self,
        path: str,
        pattern: str | None = None,
        max_size: int | None = None,
    ) -> None:
        self.model_executor.save_sharded_state(
            path=path, pattern=pattern, max_size=max_size
        )

    def collective_rpc(
        self,
        method: str | Callable[..., _R],
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict[str, Any] | None = None,
    ) -> list[_R]:
        return self.model_executor.collective_rpc(method, timeout, args, kwargs)

    def set_weight_version(self, weight_version: str) -> None:
        self._weight_version = weight_version

    def get_weight_version(self) -> str:
        """Return the latest committed weight version."""
        return self._weight_version

    def preprocess_add_request(self, request: EngineCoreRequest) -> tuple[Request, int]:
        """Preprocess the request.

        This function could be directly used in input processing thread to allow
        request initialization running in parallel with Model forward
        """
        # Note on thread safety: no race condition.
        # `mm_receiver_cache` is reset at the end of LLMEngine init,
        # and will only be accessed in the input processing thread afterwards.
        # ------【核心逻辑】复用多模态接收缓存：命中则跳过重复的特征编码 ------
        if self.mm_receiver_cache is not None and request.mm_features:
            request.mm_features = self.mm_receiver_cache.get_and_update_features(
                request.mm_features
            )

        req = Request.from_engine_core_request(request, self.request_block_hasher)
        # ------【结构化输出/grammar】结构化输出请求：异步编译 grammar，调度前检查编译状态 ------
        if req.use_structured_output:
            # Note on thread safety: no race condition.
            # `grammar_init` is only invoked in input processing thread. For
            # `structured_output_manager`, each request is independent and
            # grammar compilation is async. Scheduler always checks grammar
            # compilation status before scheduling request.
            self.structured_output_manager.grammar_init(req)
        return req, request.current_wave

    # ------【EP/EPLB】钩子：弹性 EP 在 KV 初始化前完成 scale-up（子类覆写） ------
    def _eep_scale_up_before_kv_init(self):
        raise NotImplementedError

    # ------【EP/EPLB】钩子：向 EngineCoreClient 发送弹性 EP 扩缩通知（子类覆写） ------
    def _eep_send_engine_core_notification(
        self, notification_type: EEPNotificationType
    ):
        raise NotImplementedError


# ------【核心逻辑】关闭状态机：RUNNING→REQUESTED(收到信号)→SHUTTING_DOWN(排空/中止)→退出 ------
class EngineShutdownState(IntEnum):
    RUNNING = 0
    REQUESTED = 1
    SHUTTING_DOWN = 2


class EngineCoreProc(EngineCore): # 引擎后端进程类是引擎后端类的子类
    """
    === 类说明 ===
        继承: EngineCore
        职责: 引擎后端子进程包装类。在 EngineCore 之上叠加进程间通信层：
              ZMQ socket（DEALER/PUSH）、IO 线程、输入输出队列、握手协议。
              scheduler / model_executor / kv_cache 等核心推理组件均由父类
              EngineCore.__init__ 创建（通过 super().__init__() 调用）。

    === 基类方法实现 (4个) ===
        __init__()               — 覆写父类构造：先创建通信层 + 握手，再 super().__init__ 创建推理组件
        step()                   — 覆写父类 step()：添加容错 sentinel + DP 协调逻辑
        shutdown()               — 覆写父类 shutdown()：先清理 IO 线程/ZMQ socket，再 super().shutdown()
        _maybe_publish_request_counts() — 覆写父类方法：添加 DP coordinator 请求计数发布

    === [新增] 公有方法 (9个) ===
        —— 进程入口 ——
            run_engine_core()        — 静态方法，multiprocessing.Process 的 target
        —— 主循环 ——
            run_busy_loop()          — 引擎后端核心循环：取请求 → 调度 → 前向 → 返回输出
        —— IO 线程 ——
            process_input_sockets()  — 输入线程：ZMQ DEALER → input_queue
            process_output_sockets() — 输出线程：output_queue → ZMQ PUSH
        —— 请求处理 ——
            _process_input_queue()   — 从 input_queue 取请求，调用 scheduler.add_request()
            _process_finish_requests_queue() — 处理中止/停止请求
            _process_engine_step()   — 单步执行：scheduler.schedule() → executor 前向 → update_from_output
        —— 握手 ——
            _perform_handshake()     — 执行一次 ZMQ 握手，获取通信地址
            _perform_handshakes()    — 执行所有握手（支持 DP 多前端 + 外部 LB）

    === [新增] 核心成员属性 ===
        —— 子进程通信 ——
            input_queue: Queue           — ZMQ 输入线程 → 主循环（传输 EngineCoreRequest）
            output_queue: Queue          — 主循环 → ZMQ 输出线程（传输 EngineCoreOutputs）
            tensor_ipc_receiver          — 多模态张量 IPC 接收器（可选）
        —— 生命周期和状态 ——
            engine_index: int            — 引擎编号（DP rank）
            engines_running: bool        — DP 波次运行状态
            shutdown_state               — 关闭状态机（RUNNING→DRAINING→STOPPING→TERMINATED）
            enable_fault_tolerance       — 是否启用容错
            ft_sentinel                  — 容错 sentinel 实例
        —— IO 线程 ——
            input_thread: Thread         — 输入 IO 线程（ZMQ DEALER → input_queue）
            output_thread: Thread        — 输出 IO 线程（output_queue → ZMQ PUSH）
        —— DP 协调 ——
            has_coordinator: bool        — 是否有 DP coordinator
            addresses: EngineZmqAddresses — ZMQ 地址集合（input/output/handshake/coordinator）
            frontend_stats_publish_address — 前端统计发布地址
            publish_dp_lb_stats: bool    — 是否发布 DP 负载均衡统计
            last_counts: tuple[int, int] — 上次请求计数（用于变化检测）
    """

    ENGINE_CORE_DEAD = b"ENGINE_CORE_DEAD"
    addresses: EngineZmqAddresses






    @instrument(span_name="EngineCoreProc init")
    def __init__(
        self,
        vllm_config: VllmConfig,
        local_client: bool,
        handshake_address: str,
        executor_class: type[Executor],
        log_stats: bool,
        client_handshake_address: str | None = None,
        tensor_queue: Queue | None = None,
        *,
        engine_index: int = 0,
    ):
        # ------【异步 RPC + ZMQ 通信】建 input/output 两个队列，解耦 ZMQ IO 线程与主循环 ------
        # 一个引擎后端进程，拥有一个input_queue, output_queue
        self.input_queue = queue.Queue[tuple[EngineCoreRequestType, Any]]() 
        self.output_queue = queue.Queue[tuple[int, EngineCoreOutputs] | bytes]()
        executor_fail_callback = lambda: self.input_queue.put_nowait(
            (EngineCoreRequestType.EXECUTOR_FAILED, b"")
        )

        # ------【ZMQ 通信 + DP】engine_index 同时充当 ZMQ DEALER 的 socket 身份 id（2 字节） ------
        # 引擎编号
        self.engine_index = engine_index
        identity = self.engine_index.to_bytes(length=2, byteorder="little") # Dealer的套接字的id
        self.engines_running = False
        self.shutdown_state = EngineShutdownState.RUNNING

        # ------【异步 RPC】可选的多模态张量 IPC 接收器，跨进程复用已编码特征免重复编码 ------
        # Receiver for tensor IPC: 多模态
        self.tensor_ipc_receiver: TensorIpcReceiver | None = None
        if tensor_queue is not None:
            self.tensor_ipc_receiver = TensorIpcReceiver(tensor_queue)
            logger.info("Using tensor IPC queue for multimodal tensor sharing")



        # ------【ZMQ 通信 + DP】与前端握手，拿到本引擎要用的 ZMQ 输入/输出/协调器地址 ------
        # 握手连接引擎前端
        with self._perform_handshakes( 
            handshake_address,
            identity,
            local_client,
            vllm_config,
            client_handshake_address,
        ) as addresses:
            # ------【DP】判断是否有 DP 协调器 + 是否走内部负载均衡，决定要不要上报请求计数 ------
            # Set up data parallel environment.
            self.has_coordinator = addresses.coordinator_output is not None
            self.frontend_stats_publish_address = (
                addresses.frontend_stats_publish_address
            )
            logger.debug(
                "Has DP Coordinator: %s, stats publish address: %s",
                self.has_coordinator,
                self.frontend_stats_publish_address,
            )
            internal_dp_balancing = (
                self.has_coordinator
                and not vllm_config.parallel_config.data_parallel_external_lb
            )
            # Only publish request queue stats to coordinator for "internal"
            # and "hybrid" LB modes.
            self.publish_dp_lb_stats = internal_dp_balancing
            self.last_counts = (0, 0)

            # ------【DP】保存地址 + 初始化数据并行环境（基类为 no-op，DP 子类覆写） ------
            self.addresses = addresses
            self.process_input_queue_block = True
            self._init_data_parallel(vllm_config)

            # ------【核心逻辑】调用父类构造：真正创建 scheduler / model_executor / kv_cache ------
            super().__init__(
                vllm_config,
                executor_class,
                log_stats,
                executor_fail_callback,
                internal_dp_balancing,
            )

            # ------【进程管理】按配置初始化容错 sentinel，用于多副本状态同步与故障检测 ------
            # Initialize fault tolerance settings.
            self.enable_fault_tolerance = (
                vllm_config.parallel_config.enable_fault_tolerance
            )
            if self.enable_fault_tolerance:
                self.ft_sentinel = EngineCoreSentinel(
                    engine=self,
                    parallel_config=vllm_config.parallel_config,
                )

            # ------【ZMQ 通信】后台 IO 线程：socket 收发与 GPU 前向重叠，靠释放 GIL 提高吞吐 ------
            # Background Threads and Queues for IO. These enable us to
            # overlap ZMQ socket IO with GPU since they release the GIL,
            # and to overlap some serialization/deserialization with the
            # model forward pass.
            # Threads handle Socket <-> Queues and core_busy_loop uses Queue.
            ready_event = threading.Event()


            # ------【ZMQ 通信】启动输入线程：DEALER 收请求 → 塞入 input_queue 供主循环消费 ------
            # process_input_sockets 输入线程
            input_thread = threading.Thread(
                target=self.process_input_sockets,
                args=(
                    addresses.inputs,
                    addresses.coordinator_input,
                    identity,
                    ready_event,
                ),
                daemon=True,
            )
            input_thread.start()



            # ------【ZMQ 通信】启动输出线程：output_queue 取结果 → PUSH 推回前端 ------
            # process_output_sockets 输出线程
            self.output_thread = threading.Thread(
                target=self.process_output_sockets,
                args=(
                    addresses.outputs,
                    addresses.coordinator_output,
                    self.engine_index,
                ),
                daemon=True,
            )
            self.output_thread.start()



            # ------【ZMQ 通信 + DP】等 DP 协调器发来 READY 才完成握手，保证各 DP rank 对齐后启动 ------
            # Don't complete handshake until DP coordinator ready message is
            # received.
            while not ready_event.wait(timeout=10):
                if not input_thread.is_alive():
                    raise RuntimeError("Input socket thread died during startup")
                assert addresses.coordinator_input is not None
                logger.info("Waiting for READY message from DP Coordinator...")









    @contextmanager
    def _perform_handshakes(
        self,
        handshake_address: str,
        identity: bytes,
        local_client: bool,
        vllm_config: VllmConfig,
        client_handshake_address: str | None,
    ) -> Generator[EngineZmqAddresses, None, None]:
        """
        Perform startup handshakes.

        For DP=1 or offline mode, this is with the colocated front-end process.

        For DP>1 with internal load-balancing this is with the shared front-end
        process which may reside on a different node.

        For DP>1 with external or hybrid load-balancing, two handshakes are
        performed:
            - With the rank 0 front-end process which retrieves the
              DP Coordinator ZMQ addresses and DP process group address.
            - With the colocated front-end process which retrieves the
              client input/output socket addresses.
        with the exception of the rank 0 and colocated engines themselves which
        don't require the second handshake.

        Here, "front-end" process can mean the process containing the engine
        core client (which is the API server process in the case the API
        server is not scaled out), OR the launcher process running the
        run_multi_api_server() function in serve.py.
        """
        # ------【ZMQ 通信】创建 ZMQ 上下文，判定本引擎是本地/无头模式，准备第一次握手 ------
        input_ctx = zmq.Context()
        is_local = local_client and client_handshake_address is None
        headless = not local_client
        handshake = self._perform_handshake(
            input_ctx,
            handshake_address,
            identity,
            is_local,
            headless,
            vllm_config,
            vllm_config.parallel_config,
        )
        # ------【ZMQ 通信】无第二握手地址（DP=1/离线）：只需与一个前端握手，直接 yield 地址 ------
        if client_handshake_address is None:
            # We only need to handshake with one party.
            with handshake as addresses:
                yield addresses
        else:
            # ------【ZMQ 通信 + DP】外部队列场景需两次握手：rank0 前端拿协调器地址 + 本地前端拿 I/O 地址 ------
            # We need to handshake with rank 0 front-end and our colocated frontend.
            assert local_client
            local_handshake = self._perform_handshake(
                input_ctx, client_handshake_address, identity, True, False, vllm_config
            )
            with handshake as addresses, local_handshake as client_addresses:
                # 1. Obtain DP Coordinator zmq address and DP process group address
                #    (addresses).
                # 2. Add front-end input/output addresses from colocated front-end
                #    (client_addresses).
                addresses.inputs = client_addresses.inputs
                addresses.outputs = client_addresses.outputs
                yield addresses

        # ------【核心逻辑】握手可能回写配置，重新触发 __post_init__ 让派生字段刷新 ------
        # Update config which may have changed from the handshake
        vllm_config.__post_init__()

    @contextmanager
    def _perform_handshake(
        self,
        ctx: zmq.Context,
        handshake_address: str,
        identity: bytes,
        local_client: bool,
        headless: bool,
        vllm_config: VllmConfig,
        parallel_config_to_update: ParallelConfig | None = None,
    ) -> Generator[EngineZmqAddresses, None, None]:
        # ------【ZMQ 通信】建 DEALER 握手 socket（不 bind，linger 5s 防丢消息），随后向前端注册 ------
        with make_zmq_socket(
            ctx,
            handshake_address,
            zmq.DEALER,
            identity=identity,
            linger=5000,
            bind=False,
        ) as handshake_socket:
            # Register engine with front-end.
            addresses = self.startup_handshake(
                handshake_socket, local_client, headless, parallel_config_to_update
            )
            yield addresses

            # ------【ZMQ 通信 + DP】发 READY 回执；DP>1 时附上配置 hash 供前端校验各 rank 一致性 ------
            # Send ready message.
            ready_msg = {
                "status": "READY",
                "local": local_client,
                "headless": headless,
            }
            # Include config hash for DP configuration validation
            if vllm_config.parallel_config.data_parallel_size > 1:
                ready_msg["parallel_config_hash"] = (
                    vllm_config.parallel_config.compute_hash()
                )

            handshake_socket.send(msgspec.msgpack.encode(ready_msg))

    @staticmethod
    def startup_handshake(
        handshake_socket: zmq.Socket,
        local_client: bool,
        headless: bool,
        parallel_config: ParallelConfig | None = None,
    ) -> EngineZmqAddresses:
        # ------【ZMQ 通信】先发 HELLO 注册包，告知前端自己是本地/无头引擎 ------
        # Send registration message.
        handshake_socket.send(
            msgspec.msgpack.encode(
                {
                    "status": "HELLO",
                    "local": local_client,
                    "headless": headless,
                }
            )
        )

        # ------【ZMQ 通信】阻塞等待前端的 init 消息（含 ZMQ 地址），超时则报错 ------
        # Receive initialization message.
        logger.debug("Waiting for init message from front-end.")
        if not handshake_socket.poll(timeout=HANDSHAKE_TIMEOUT_MINS * 60_000):
            raise RuntimeError(
                "Did not receive response from front-end "
                f"process within {HANDSHAKE_TIMEOUT_MINS} "
                f"minutes"
            )
        init_bytes = handshake_socket.recv()
        init_message: EngineHandshakeMetadata = msgspec.msgpack.decode(
            init_bytes, type=EngineHandshakeMetadata
        )
        logger.debug("Received init message: %s", init_message)

        # ------【核心逻辑】把前端下发的并行配置覆盖到本地，并返回引擎要用的 ZMQ 地址集合 ------
        if parallel_config is not None:
            for key, value in init_message.parallel_config.items():
                setattr(parallel_config, key, value)

        return init_message.addresses









    # 刚好是一个静态方法，工厂方法，无需实例也可以调用， 引擎后端进程的任务入口
    @staticmethod
    def run_engine_core(*args, dp_rank: int = 0, local_dp_rank: int = 0, **kwargs):
        """Launch EngineCore busy loop in background process."""

        # ------【进程管理】注册序列化规则，保证 fork 后 transformer 配置可安全 pickle 传递 ------
        # Ensure we can serialize transformer config after spawning
        maybe_register_config_serialize_by_value()

        # 声明一个引擎后端变量
        engine_core: EngineCoreProc | None = None 
        signal_callback: SignalCallback | None = None
        try:
            # ------【DP】判定是否数据并行，据此设置进程标题与本地 DP rank ------
            vllm_config: VllmConfig = kwargs["vllm_config"]
            parallel_config: ParallelConfig = vllm_config.parallel_config
            data_parallel = parallel_config.data_parallel_size > 1 or dp_rank > 0
            if data_parallel:
                parallel_config.data_parallel_rank_local = local_dp_rank
                process_title = f"EngineCore_DP{dp_rank}"
            else:
                process_title = "EngineCore"
            set_process_title(process_title)



            # ------【NUMA 亲和】初始化 tracing/日志，开启 NUMA 绑核时打印当前 CPU 亲和状态 ------
            maybe_init_worker_tracer("vllm.engine_core", "engine_core", process_title)
            decorate_logs()
            if parallel_config.numa_bind:
                numa_utils.log_current_affinity_state(process_title)

            # ------【PD 分离 + DP】KV 传输的 engine_id 追加 dp_rank，保证各 DP rank 唯一不冲突 ------
            if data_parallel and vllm_config.kv_transfer_config is not None:
                # modify the engine_id and append the dp_rank to it to ensure
                # that the kv_transfer_config is unique for each DP rank.
                vllm_config.kv_transfer_config.engine_id = (
                    f"{vllm_config.kv_transfer_config.engine_id}_dp{dp_rank}"
                )
                logger.debug(
                    "Setting kv_transfer_config.engine_id to %s",
                    vllm_config.kv_transfer_config.engine_id,
                )

            # ------【DP】MoE 的 DP 走 DPEngineCoreProc（需锁步协调），其余走普通 EngineCoreProc ------
            parallel_config.data_parallel_index = dp_rank
            if data_parallel and vllm_config.model_config.is_moe:
                # Set data parallel rank for this engine process.
                parallel_config.data_parallel_rank = dp_rank
                engine_core = DPEngineCoreProc(*args, **kwargs)
            else:
                # Non-MoE DP ranks are completely independent, so treat like DP=1.
                # Note that parallel_config.data_parallel_index will still reflect
                # the original DP rank.

                # ------【DP】非 MoE 的 DP rank 完全独立，重配置后当 DP=1 处理 ------
                parallel_config.reconfigure_for_independent_dp_rank()



                ###############################################################################################
                # 1. 引擎后端进程开始构造enginecoreProc实例
                ###############################################################################################
                # 1. 构造引擎后端进程
                engine_core = EngineCoreProc(*args, engine_index=dp_rank, **kwargs)
            assert engine_core is not None




            # ------【异步 RPC】通过 input_queue 塞 WAKEUP 唤醒空闲引擎，避免阻塞信号处理 ------
            def wakeup_engine():  # 定义唤醒引擎实例的方法
                # Wakes up idle engine via input_queue when shutdown is requested
                # Not safe in a signal handler - we may interrupt the main thread
                # while it is holding the non-reentrant input_queue.mutex
                engine_core.input_queue.put_nowait((EngineCoreRequestType.WAKEUP, None))

            signal_callback = SignalCallback(wakeup_engine)

            # ------【进程管理】注册 SIGTERM/SIGINT 处理器：置 REQUESTED 状态并唤醒主循环优雅退出 ------
            def signal_handler(signum, frame):
                signal_name = signal.Signals(signum).name
                logger.info(
                    "[shutdown] EngineCore: trigger received signal=%s",
                    signal_name,
                )
                engine_core.shutdown_state = EngineShutdownState.REQUESTED
                signal_callback.trigger()

            signal.signal(signal.SIGTERM, signal_handler)
            signal.signal(signal.SIGINT, signal_handler)

            # ------【核心逻辑】进入主循环，直到 shutdown 或 SystemExit 才返回 ------
            ###############################################################################################
            # 2. 引擎后端进程开始进入busy loop
            ###############################################################################################
            engine_core.run_busy_loop()  # 2. 开始循环运行





        # ------【进程管理】异常兜底：启动失败仅记录；运行中致命错误则先发 ENGINE_CORE_DEAD 再抛 ------
        except SystemExit:
            logger.info_once("[shutdown] EngineCore: exiting busy loop")
            raise
        except Exception as e:
            if engine_core is None:
                logger.exception("EngineCore failed to start.")
            else:
                logger.exception("EngineCore encountered a fatal error.")
                engine_core._send_engine_dead()
            raise e
        finally:
            # ------【进程管理】恢复默认信号处理 + 停掉回调 + 兜底 shutdown，保证资源释放 ------
            signal.signal(signal.SIGTERM, signal.SIG_DFL)
            signal.signal(signal.SIGINT, signal.SIG_DFL)
            if signal_callback is not None:
                signal_callback.stop()
            if engine_core is not None:
                engine_core.shutdown()











    def _init_data_parallel(self, vllm_config: VllmConfig):
        # ------【DP】基类 no-op 钩子：DPEngineCoreProc 会覆写以建 DP 进程组与协调状态 ------
        pass

    def has_work(self) -> bool:
        """Returns true if the engine should be stepped."""
        # ------【核心逻辑】有 DP 波次在跑 / 调度器有请求 / 批队列非空，任一为真就该 step ------
        return (
            self.engines_running
            or self.scheduler.has_requests()
            or bool(self.batch_queue)
        )

    def is_running(self) -> bool:
        """Returns true if shutdown has not been requested."""
        # ------【进程管理】只有 RUNNING 状态才继续处理，收到信号后即停止进请求 ------
        return self.shutdown_state == EngineShutdownState.RUNNING










    # 引擎后端进程类 的内部工作循环，主线程
    @fault_tolerant_wrapper
    def run_busy_loop(self):
        """Core busy loop of the EngineCore."""
        while self._handle_shutdown():
            # ------【核心逻辑】主循环：取请求 → 上报计数 → step → 再上报，直到收到 shutdown ------
            # 1) Poll the input queue until there is work to do.
            # 把ZMQ输入线程收集到的请求，灌入 scheduler的等待队列
            self._process_input_queue()



            # ------【DP】step 前上报一次请求计数，保证协调器拿到的队列长度足够新鲜 ------
            # Publish request counts before and after GPU step to ensure freshness.
            # 发布队列状态给DP协调器
            self._maybe_publish_request_counts()



            # 2) Step the engine core and return the outputs.
            # 执行一步推理
            self._process_engine_step()

            # ------【DP】step 后再上报一次，捕获本步调度引起的计数变化 ------
            # DP负载上报
            self._maybe_publish_request_counts()

        raise SystemExit











    def _maybe_publish_request_counts(self):
        # ------【DP】未开启内部 LB 统计上报则直接跳过 ------
        if not self.publish_dp_lb_stats:
            return

        # ------【DP】请求计数变化时才发布，用 client_index=-1 标记是发给协调器的统计消息 ------
        # Publish our request counts (if they've changed).
        counts = self.scheduler.get_request_counts()
        if counts != self.last_counts:
            self.last_counts = counts
            stats = SchedulerStats(
                *counts, kv_cache_usage=self.scheduler.get_kv_cache_usage()
            )
            self.output_queue.put_nowait((-1, EngineCoreOutputs(scheduler_stats=stats)))




    def _process_input_queue(self):
        """Exits when an engine step needs to be performed."""

        waited = False
        # ------【核心逻辑】没活且未关闭时阻塞等待，同时通知所有等待引擎空闲的暂停回调 ------
        while not self.has_work() and self.is_running():
            # Notify callbacks waiting for engine to become idle.
            self._notify_idle_state_callbacks()


            # ------【核心逻辑】input_queue 空时清空 aborts_queue（中止请求也已进 input_queue，此处去重） ------
            if self.input_queue.empty(): # 看input_queue里面是否有内容
                # Drain aborts queue; all aborts are also processed via input_queue.
                with self.aborts_queue.mutex:
                    self.aborts_queue.queue.clear()
                if logger.isEnabledFor(DEBUG):
                    logger.debug("EngineCore waiting for work.")
                    waited = True
            block = self.process_input_queue_block
            try:
                # ------【异步 RPC】阻塞取出一个请求并分发；非阻塞模式（block=False）遇空则跳出 ------
                req = self.input_queue.get(block=block) # 有req，处理它

                self._handle_client_request(*req) # 分发请求
            except queue.Empty:
                break
            if not block:
                break

        if waited:
            logger.debug("EngineCore loop active.")

        # ------【异步 RPC】有活后顺带把积压的请求全部取空，避免频繁切回等待分支 ------
        # Handle any more client requests.
        while not self.input_queue.empty():
            req = self.input_queue.get_nowait()
            self._handle_client_request(*req)





    # 引擎后端EngineCore实例的推理工作
    def _process_engine_step(self) -> bool:
        """Called only when there are unfinished local requests."""

        # ------【核心逻辑】执行一步调度+前向，返回各 client 的输出与「本步是否真跑了前向」 ------
        # Step the engine core.
        # enginecore 实例执行一步推理
        outputs, model_executed = self.step_fn()


        # ------【异步 RPC】把输出按 (client_index, EngineCoreOutputs) 入队，交给输出线程推送 ------
        # Put EngineCoreOutputs into the output queue.
        # 如果有输出，就放入输出队列
        for output in outputs.items() if outputs else ():
            self.output_queue.put_nowait(output)


        # ------【投机解码】post_step 钩子：非异步调度且本步有前向时取回 draft token 回填调度器 ------
        # Post-step hook.
        # 投机解码
        self.post_step(model_executed)

        # ------【PD 分离】未跑前向但调度器仍有活（等远端 KVS/延迟释放）时让渡 GIL，让传输线程推进 ------
        # If no model execution happened but there is still scheduler work
        # (e.g. WAITING_FOR_REMOTE_KVS or delayed KV connector frees), yield
        # the GIL briefly to allow background transfer threads to make progress.
        # 这是 P/D 分离场景下的一个 GIL 让渡技巧。
        if not model_executed and self.scheduler.has_requests():
            time.sleep(0.001)

        # ------【核心逻辑】返回 model_executed：True=有 token 被调度且跑了前向，False=本步空转 ------
        # 本轮step有没有token被调度 + 执行
        # True表示调度器至少选出了一个Token，Executor跑了一次前向推理
        # False表示没有调度任何token, 要么没有请求，要么 WAITING_FOR_REMOTE_KVS，让调度器暂停了
        return model_executed








    def _notify_idle_state_callbacks(self) -> None:
        # ------【异步 RPC】逐个弹出并执行等待空闲的回调（如 pause 的 Future 完成器） ------
        while self._idle_state_callbacks:
            callback = self._idle_state_callbacks.pop()
            callback(self)

    # 处理结束请求
    def _handle_shutdown(self) -> bool:
        # Check if shutdown was requested and handle it
        # ------【进程管理】RUNNING 状态直接返回 True，继续循环 ------
        if self.shutdown_state == EngineShutdownState.RUNNING:
            return True

        # ------【进程管理】收到 REQUESTED：按 timeout 决定 abort（立即中止）还是 drain（排空后退出） ------
        if self.shutdown_state == EngineShutdownState.REQUESTED:
            shutdown_timeout = self.vllm_config.shutdown_timeout
            mode = "abort" if shutdown_timeout == 0 else "drain"

            logger.info(
                "[shutdown] EngineCore: start mode=%s timeout=%ds",
                mode,
                shutdown_timeout,
            )

            # ------【进程管理】abort 模式：立即中止所有在途请求并把中止结果发回各 client ------
            if shutdown_timeout == 0:
                num_requests = self.scheduler.get_num_unfinished_requests()
                if num_requests > 0:
                    logger.info(
                        "[shutdown] EngineCore: aborting in-flight requests count=%d",
                        num_requests,
                    )
                aborted_reqs = self.scheduler.finish_requests(
                    None, RequestStatus.FINISHED_ABORTED
                )
                self._send_abort_outputs(aborted_reqs)
            else:
                # ------【进程管理】drain 模式：记录在途请求数，继续 step 直到自然排空 ------
                num_requests = self.scheduler.get_num_unfinished_requests()
                if num_requests > 0:
                    logger.info(
                        "[shutdown] EngineCore: draining in-flight requests "
                        "count=%d timeout=%ds",
                        num_requests,
                        shutdown_timeout,
                    )

            self.shutdown_state = EngineShutdownState.SHUTTING_DOWN

        # Exit when no work remaining
        # ------【进程管理】排空后返回 False 退出循环，进入资源拆除 ------
        if not self.has_work():
            logger.info(
                "[shutdown] EngineCore: request processing complete; "
                "starting resource teardown"
            )
            return False

        return True






    # 处理input_queue的请求req
    def _handle_client_request(
        self, request_type: EngineCoreRequestType, request: Any
    ) -> None:
        """Dispatch request from client."""

        # ------【异步 RPC】WAKEUP 是信号处理线程塞进来的唤醒包，直接忽略 ------
        # 唤醒请求，不分发，忽略
        if request_type == EngineCoreRequestType.WAKEUP:
            return

        # ------【核心逻辑】ADD：关停期间拒绝，否则加入调度器等待队列 ------
        # 如果是ADD请求-》加入调度队列
        elif request_type == EngineCoreRequestType.ADD:
            req, request_wave = request
            if self._reject_add_in_shutdown(req):
                return
            self.add_request(req, request_wave) # 加入调度队列

        # ------【核心逻辑】ABORT：立即中止指定请求，转发进 abort 队列 ------
        # 如果是ABORT请求-》加入取消请求队列
        elif request_type == EngineCoreRequestType.ABORT:
            self.abort_requests(request)



        # ------【异步 RPC】UTILITY：远程调用工具方法（如 LoRA 增删/权重版本），结果异步回客户端 ------
        elif request_type == EngineCoreRequestType.UTILITY:
            client_idx, call_id, method_name, args = request
            if self._reject_utility_in_shutdown(client_idx, call_id, method_name):
                return
            output = UtilityOutput(call_id)
            # Lazily look-up utility method so that failure will be handled/returned.
            get_result = lambda: (
                (method := getattr(self, method_name))
                and method(*self._convert_msgspec_args(method, args))
            )
            enqueue_output = lambda out: self.output_queue.put_nowait(
                (client_idx, EngineCoreOutputs(utility_output=out))
            )
            self._invoke_utility_method(method_name, get_result, output, enqueue_output)
        # ------【进程管理】执行器回调上报故障时直接抛错，由外层异常处理发 ENGINE_CORE_DEAD ------
        elif request_type == EngineCoreRequestType.EXECUTOR_FAILED:
            raise RuntimeError("Executor failed.")
        else:
            logger.error(
                "Unrecognized input request type encountered: %s", request_type
            )







    def _reject_add_in_shutdown(self, request: Request) -> bool:
        # ------【进程管理】RUNNING 时不拒绝；关停中则回一个 ABORT 结果给 client 并返回 True ------
        if self.shutdown_state == EngineShutdownState.RUNNING:
            return False

        logger.debug(
            "[shutdown] EngineCore: rejecting new request request_id=%s",
            request.request_id,
        )
        self._send_abort_outputs_to_client([request.request_id], request.client_index)
        return True

    def _reject_utility_in_shutdown(
        self, client_idx: int, call_id: int, method_name: str
    ) -> bool:
        # ------【进程管理】关停中拒绝工具调用，直接回一个失败 UtilityOutput 给客户端 ------
        if self.shutdown_state == EngineShutdownState.RUNNING:
            return False

        logger.warning(
            "[shutdown] EngineCore: rejecting utility call method=%s",
            method_name,
        )
        output = UtilityOutput(call_id, failure_message="Server shutting down")
        self.output_queue.put_nowait(
            (client_idx, EngineCoreOutputs(utility_output=output))
        )
        return True

    @staticmethod
    def _invoke_utility_method(
        name: str, get_result: Callable, output: UtilityOutput, enqueue_output: Callable
    ):
        try:
            result = get_result()
            # ------【异步 RPC】工具方法返回 Future 时挂完成回调，异步把最终结果回给客户端 ------
            if isinstance(result, Future):
                # Defer utility output handling until future completion.
                callback = lambda future: EngineCoreProc._invoke_utility_method(
                    name, future.result, output, enqueue_output
                )
                result.add_done_callback(callback)
                return
            output.result = UtilityResult(result)
        except Exception as e:
            # ------【核心逻辑】调用失败：记录异常并把失败信息塞进 output 返回客户端 ------
            logger.exception("Invocation of %s method failed", name)
            output.failure_message = f"Call to {name} method failed: {str(e)}"
        enqueue_output(output)

    @staticmethod
    def _convert_msgspec_args(method, args):
        """If a provided arg type doesn't match corresponding target method
        arg type, try converting to msgspec object."""
        if not args:
            return args
        arg_types = signature(method).parameters.values()
        assert len(args) <= len(arg_types)
        # ------【异步 RPC】按目标方法签名把 msgspec 参数转成对应 Struct 类型，反序列化传参 ------
        return tuple(
            msgspec.convert(v, type=p.annotation)
            if isclass(p.annotation)
            and issubclass(p.annotation, msgspec.Struct)
            and not isinstance(v, p.annotation)
            else v
            for v, p in zip(args, arg_types)
        )

    def _send_engine_dead(self):
        """Send EngineDead status to the EngineCoreClient."""

        # ------【ZMQ 通信】把哨兵 ENGINE_CORE_DEAD 塞进输出队列，通知前端引擎已死 ------
        # Put ENGINE_CORE_DEAD in the queue.
        self.output_queue.put_nowait(EngineCoreProc.ENGINE_CORE_DEAD)

        # ------【进程管理】等输出线程把死亡消息发出再退出，超时 5s 则记录致命日志 ------
        # Wait until msg sent by the daemon before shutdown.
        self.output_thread.join(timeout=5.0)
        if self.output_thread.is_alive():
            logger.fatal(
                "vLLM shutdown signal from EngineCore failed "
                "to send. Please report this issue."
            )

    def _make_ready_response(self) -> EngineCoreReadyResponse:
        # ------【ZMQ 通信】打包引擎就绪元数据（显存/并行度/调度参数），随首个消息发回前端 ------
        parallel_config = self.vllm_config.parallel_config
        scheduler_config = self.vllm_config.scheduler_config
        return EngineCoreReadyResponse(
            max_model_len=self.vllm_config.model_config.max_model_len,
            num_gpu_blocks=self.vllm_config.cache_config.num_gpu_blocks or 0,
            block_size=self.vllm_config.cache_config.block_size,
            dp_stats_address=self.frontend_stats_publish_address,
            dtype=str(self.vllm_config.model_config.dtype).removeprefix("torch."),
            vllm_version=VLLM_VERSION,
            world_size=self.vllm_config.parallel_config.world_size,
            data_parallel_size=parallel_config.data_parallel_size,
            kv_cache_size_tokens=self.vllm_config.cache_config.kv_cache_size_tokens,
            kv_cache_max_concurrency=(
                self.vllm_config.cache_config.kv_cache_max_concurrency
            ),
            tensor_parallel_size=parallel_config.tensor_parallel_size,
            pipeline_parallel_size=parallel_config.pipeline_parallel_size,
            decode_context_parallel_size=parallel_config.decode_context_parallel_size,
            data_parallel_rank=self.engine_index,
            max_num_seqs=scheduler_config.max_num_seqs,
            max_num_batched_tokens=scheduler_config.max_num_batched_tokens,
            instance_id=self.vllm_config.instance_id,
            kv_events_config=self.scheduler.get_kv_event_publisher_config(),
        )

    def process_input_sockets(
        self,
        input_addresses: list[str],
        coord_input_address: str | None,
        identity: bytes,
        ready_event: threading.Event,
    ):
        """Input socket IO thread."""

        # ------【ZMQ 通信】两类解码器：ADD 请求专用 + 通用（可挂多模态张量 IPC 接收器做零拷贝） ------
        # Msgpack serialization decoding with optional tensor IPC receiver.
        add_request_decoder = MsgpackDecoder(
            EngineCoreRequest, oob_tensor_provider=self.tensor_ipc_receiver
        )
        generic_decoder = MsgpackDecoder(oob_tensor_provider=self.tensor_ipc_receiver)

        # ------【ZMQ 通信】为每个前端输入地址建 DEALER socket，协调器地址建 XSUB 订阅 socket ------
        with ExitStack() as stack, zmq.Context() as ctx:
            input_sockets = [
                stack.enter_context(
                    make_zmq_socket(
                        ctx, input_address, zmq.DEALER, identity=identity, bind=False
                    )
                )
                for input_address in input_addresses
            ]
            if coord_input_address is None:
                coord_socket = None
            else:
                coord_socket = stack.enter_context(
                    make_zmq_socket(
                        ctx,
                        coord_input_address,
                        zmq.XSUB,
                        identity=identity,
                        bind=False,
                    )
                )
                # Send subscription message to coordinator.
                coord_socket.send(b"\x01")

            # Register sockets with poller.
            # ------【ZMQ 通信】先向每个输入 socket 发就绪包，前端 ROUTER 才能回发请求；再注册进 poller ------
            poller = zmq.Poller()
            ready_response = self._make_ready_response()
            ready_payload = msgspec.msgpack.encode(ready_response)
            for input_socket in input_sockets:
                # Send initial message to each input socket - this is required
                # before the front-end ROUTER socket can send input messages
                # back to us.
                input_socket.send(ready_payload)
                poller.register(input_socket, zmq.POLLIN)

            # ------【ZMQ 通信 + DP】有协调器时先阻塞等其 READY 才置 ready_event，保证 DP rank 对齐 ------
            if coord_socket is not None:
                # Wait for ready message from coordinator.
                assert coord_socket.recv() == b"READY"
                poller.register(coord_socket, zmq.POLLIN)

            ready_event.set()
            del ready_event
            # ------【ZMQ 通信】主循环：poller 监听所有输入 socket，逐帧解析请求类型与数据 ------
            while True:
                for input_socket, _ in poller.poll():
                    # (RequestType, RequestData)
                    type_frame, *data_frames = input_socket.recv_multipart(copy=False)
                    # NOTE(yongji): ignore READY message sent by DP coordinator
                    # that is used to notify newly started engines
                    # ------【ZMQ 通信 + DP】忽略协调器发给新引擎的 READY 通知帧 ------
                    if type_frame.buffer == b"READY":
                        assert input_socket == coord_socket
                        continue
                    request_type = EngineCoreRequestType(bytes(type_frame.buffer))

                    # Deserialize the request data.
                    request: Any
                    # ------【ZMQ 通信】ADD：专用解码器 + 预处理；失败则回错误并跳过该请求 ------
                    if request_type == EngineCoreRequestType.ADD:
                        req: EngineCoreRequest = add_request_decoder.decode(data_frames)
                        try:
                            request = self.preprocess_add_request(req)
                        except Exception:
                            self._handle_request_preproc_error(req)
                            continue
                    # ------【异步 RPC】UTILITY：通用解码；容错命令走 ft_sentinel 处理不走主循环 ------
                    elif request_type == EngineCoreRequestType.UTILITY:
                        request = generic_decoder.decode(data_frames)
                        client_idx, call_id, method, args = request
                        if method == FT_UTILITY_METHOD:
                            self.ft_sentinel.handle_command(
                                client_idx, call_id, args[0]
                            )
                            continue
                    else:
                        request = generic_decoder.decode(data_frames)

                        # ------【核心逻辑】ABORT 同时进 aborts 队列（可急切处理）和 input 队列（保序） ------
                        if request_type == EngineCoreRequestType.ABORT:
                            # Aborts are added to *both* queues, allows us to eagerly
                            # process aborts while also ensuring ordering in the input
                            # queue to avoid leaking requests. This is ok because
                            # aborting in the scheduler is idempotent.
                            self.aborts_queue.put_nowait(request)

                    # Push to input queue for core busy loop.
                    # ------【异步 RPC】最终把 (类型, 请求) 塞进 input_queue，交给主循环消费 ------
                    self.input_queue.put_nowait((request_type, request))

    def process_output_sockets(
        self, output_paths: list[str], coord_output_path: str | None, engine_index: int
    ):
        """Output socket IO thread."""

        # ------【ZMQ 通信】编码器 + 可复用 buffer 池 + 待回收 buffer 队列，支持零拷贝多帧发送 ------
        # Msgpack serialization encoding.
        encoder = MsgpackEncoder()
        # Send buffers to reuse.
        reuse_buffers: list[bytearray] = []
        # Payload buffers that can't be reused yet because zmq may still be
        # sending them.
        # Buffers of the zero-copy tensor/ndarray frames don't need tracking
        # here: zmq itself holds a reference to each until it's done with it.
        pending = deque[tuple[zmq.MessageTracker, bytearray]]()

        # ------【ZMQ 通信】linger=4000 保证 ENGINE_CORE_DEAD 在关 socket 前一定发出；输出走 PUSH 推给前端 ------
        # We must set linger to ensure the ENGINE_CORE_DEAD
        # message is sent prior to closing the socket.
        with ExitStack() as stack, zmq.Context() as ctx:
            sockets = [
                stack.enter_context(
                    make_zmq_socket(ctx, output_path, zmq.PUSH, linger=4000)
                )
                for output_path in output_paths
            ]
            # ------【ZMQ 通信 + DP】可选的协调器输出 socket，用于发 DP 统计/波次完成消息 ------
            coord_socket = (
                stack.enter_context(
                    make_zmq_socket(
                        ctx, coord_output_path, zmq.PUSH, bind=False, linger=4000
                    )
                )
                if coord_output_path is not None
                else None
            )
            max_reuse_bufs = len(sockets) + 1

            while True:
                output = self.output_queue.get()
                # ------【ZMQ 通信】收到死亡哨兵则向所有前端广播后退出输出线程 ------
                if output == EngineCoreProc.ENGINE_CORE_DEAD:
                    for socket in sockets:
                        socket.send(output)
                    break
                assert not isinstance(output, bytes)
                client_index, outputs = output
                outputs.engine_index = engine_index

                # ------【ZMQ 通信 + DP】client_index=-1 表示发协调器的小消息，不复用 buffer 直接发 ------
                if client_index == -1:
                    # Don't reuse buffer for coordinator message
                    # which will be very small.
                    assert coord_socket is not None
                    coord_socket.send_multipart(encoder.encode(outputs))
                    continue

                # Reclaim buffers that zmq is finished with.
                # ------【ZMQ 通信】回收 zmq 已发完的 buffer 进复用池，避免反复分配 bytearray ------
                while pending and pending[-1][0].done:
                    reclaimed = pending.pop()[1]
                    if len(reuse_buffers) < max_reuse_bufs:
                        reuse_buffers.append(reclaimed)

                # ------【ZMQ 通信】零拷贝编码进复用 buffer 后发送；未发完的 buffer 进 pending 等待回收 ------
                buffer = reuse_buffers.pop() if reuse_buffers else bytearray()
                buffers = encoder.encode_into(outputs, buffer)
                tracker = self._send_msg_tracking_payload(
                    sockets[client_index], buffers
                )
                if not tracker.done:
                    pending.appendleft((tracker, buffer))
                elif len(reuse_buffers) < max_reuse_bufs:
                    # Limit the number of buffers to reuse.
                    reuse_buffers.append(buffer)

    @staticmethod
    def _send_msg_tracking_payload(
        socket: zmq.Socket, buffers: Sequence[bytestr]
    ) -> zmq.MessageTracker:
        """Send `buffers` as a zero-copy multipart message, returning a tracker
        for the *first* frame.

        Used instead of `Socket.send_multipart()` because we reuse the buffer
        passed to `MsgpackEncoder.encode_into()`: `send_multipart()` returns a
        tracker for the last frame only.
        """
        # ------【ZMQ 通信】零拷贝发首帧并拿到 tracker，多帧时首帧带 SNDMORE 其余随后发送 ------
        more_flag = zmq.SNDMORE if len(buffers) > 1 else 0
        tracker = socket.send(buffers[0], more_flag, copy=False, track=True)
        if more_flag:
            socket.send_multipart(buffers[1:], copy=False)
        return tracker

    def _handle_request_preproc_error(self, request: EngineCoreRequest) -> None:
        """Log and return a request-scoped error response for exceptions raised
        from the add request preprocessing in the input socket processing thread.
        """
        logger.exception(
            "Unexpected error pre-processing request %s", request.request_id
        )
        # ------【核心逻辑】预处理失败：回一个 ERROR 结束结果给对应 client，避免其一直等待 ------
        self._send_error_outputs_to_client([request.request_id], request.client_index)

    def pause_scheduler(
        self, mode: PauseMode = "abort", clear_cache: bool = True
    ) -> Future | None:
        """Pause generation; behavior depends on mode.

        All pause modes queue new adds -- "abort" and "keep" skip step();
        "wait" allows step() so in-flight requests can drain.

        - ``abort``: Set PAUSED_NEW, abort all requests, wait for abort
          outputs to be sent (when running with output_queue), optionally
          clear caches, then complete the returned Future.
        - ``wait``: Set PAUSED_NEW (queue adds, keep stepping); when drained,
          optionally clear caches, then complete the returned Future.
        - ``keep``: Set PAUSED_ALL; return a Future that completes when the
          output queue is empty.
        """
        if mode not in ("keep", "abort", "wait"):
            raise ValueError(f"Invalid pause mode: {mode}")

        # ------【显存 profiling】空闲回调：等引擎排空后可选清缓存，再置 Future 结果完成暂停 ------
        def engine_idle_callback(engine: "EngineCoreProc", future: Future[Any]) -> None:
            if clear_cache:
                engine._reset_caches()
            future.set_result(None)

        # ------【显存 profiling】abort 模式立即中止所有在途请求并回发中止结果 ------
        if mode == "abort":
            aborted_reqs = self.scheduler.finish_requests(
                None, RequestStatus.FINISHED_ABORTED
            )
            self._send_abort_outputs(aborted_reqs)

        # ------【显存 profiling】keep 全停（不 step），其余仅停新请求但保留在途继续 step ------
        pause_state = PauseState.PAUSED_ALL if mode == "keep" else PauseState.PAUSED_NEW
        self.scheduler.set_pause_state(pause_state)

        # ------【显存 profiling】已无活则同步完成：直接清缓存返回 None ------
        if self._pause_complete():
            if clear_cache:
                self._reset_caches()
            return None

        # ------【异步 RPC】仍有在途请求：挂空闲回调返回 Future，等排空后异步完成 ------
        future = Future[Any]()
        self._idle_state_callbacks.append(partial(engine_idle_callback, future=future))
        return future

    def _pause_complete(self) -> bool:
        """Returns True if the pause has fully completed and the caller can
        return ``None`` synchronously; False if the pause is still pending
        and the caller should register an idle-state callback to finish it.
        """
        # ------【显存 profiling】无活即暂停完成；有活说明需等排空 ------
        return not self.has_work()

    def _send_finish_outputs_to_client(
        self, req_ids: list[str], client_index: int, finish_reason: FinishReason
    ) -> None:
        # ------【核心逻辑】给每个请求造一个带 finish_reason 的空输出，标记已结束并推给对应 client ------
        outputs = [
            EngineCoreOutput(req_id, [], finish_reason=finish_reason)
            for req_id in req_ids
        ]
        eco = EngineCoreOutputs(finished_requests=req_ids, outputs=outputs)
        self.output_queue.put_nowait((client_index, eco))

    def _send_abort_outputs_to_client(
        self, req_ids: list[str], client_index: int
    ) -> None:
        # ------【核心逻辑】便捷封装：以 ABORT 结束原因回发指定请求的结束结果 ------
        self._send_finish_outputs_to_client(req_ids, client_index, FinishReason.ABORT)

    def _send_error_outputs_to_client(
        self, req_ids: list[str], client_index: int
    ) -> None:
        # ------【核心逻辑】便捷封装：以 ERROR 结束原因回发指定请求的结束结果 ------
        self._send_finish_outputs_to_client(req_ids, client_index, FinishReason.ERROR)

    def _send_abort_outputs(self, aborted_reqs: list[Request]) -> None:
        # TODO(nick) this will be moved inside the scheduler
        if aborted_reqs:
            # ------【核心逻辑】按 client_index 聚合被中止请求，逐客户端回发 ABORT 结果 ------
            # Map client_index to list of request_ids that belong to that client.
            by_client = defaultdict[int, set[str]](set)
            for request in aborted_reqs:
                by_client[request.client_index].add(request.request_id)
            for client_index, req_ids in by_client.items():
                self._send_abort_outputs_to_client(list(req_ids), client_index)


class DPEngineCoreProc(EngineCoreProc):
    """ZMQ-wrapper for running EngineCore in background process
    in a data parallel context."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_client: bool,
        handshake_address: str,
        executor_class: type[Executor],
        log_stats: bool,
        client_handshake_address: str | None = None,
        tensor_queue: Queue | None = None,
    ):
        assert vllm_config.model_config.is_moe, (
            "DPEngineCoreProc should only be used for MoE models"
        )

        scheduler_config = vllm_config.scheduler_config
        self.prefill_schedule_interval = scheduler_config.prefill_schedule_interval

        # ------【DP】step_counter 每 N 步与 DP peer 同步一次；current_wave 记录当前波次 ------
        # Counts forward-passes of the model so that we can synchronize
        # finished with DP peers every N steps.
        self.step_counter = 0
        self.current_wave = 0

        # Two-phase pause protocol state. When pending_pause is True, the
        # engine keeps stepping (dummy batches) while waiting for all DP
        # ranks to also set pending_pause. Once all ranks agree via
        # all-reduce, ignore_start_dp_wave is set so that stale
        # START_DP_WAVE messages cannot re-wake the engines.
        self.pending_pause = False
        self.ignore_start_dp_wave = False

        # ------【EP/EPLB】弹性 EP 扩缩容状态，默认 None（未在进行扩缩） ------
        from vllm.distributed.elastic_ep.elastic_state import ElasticEPScalingState

        self.eep_scaling_state: ElasticEPScalingState | None = None

        # Initialize the engine.
        # ------【DP】把 DP rank 传给父类作为 engine_index，随后建通信层 + 推理组件 ------
        dp_rank = vllm_config.parallel_config.data_parallel_rank
        super().__init__(
            vllm_config,
            local_client,
            handshake_address,
            executor_class,
            log_stats,
            client_handshake_address,
            engine_index=dp_rank,
            tensor_queue=tensor_queue,
        )

    def _init_data_parallel(self, vllm_config: VllmConfig):
        # Configure GPUs and stateless process group for data parallel.
        # ------【DP】取出 DP rank/size/local rank 并做范围校验 ------
        parallel_config = vllm_config.parallel_config
        dp_rank = parallel_config.data_parallel_rank
        dp_size = parallel_config.data_parallel_size
        local_dp_rank = parallel_config.data_parallel_rank_local

        assert dp_size > 1
        assert local_dp_rank is not None
        assert 0 <= local_dp_rank <= dp_rank < dp_size

        self.dp_rank = dp_rank
        self.dp_size = dp_size
        # ------【DP + NCCL 通信】无状态初始化 DP 进程组与 KV store，供 all-reduce 同步锁步状态 ------
        dp_group, dp_store = parallel_config.stateless_init_dp_group(return_store=True)
        self.dp_group, self.dp_store = dp_group, dp_store

    def shutdown(self):
        # ------【进程管理】先关父类推理组件，再销毁 DP 进程组释放 NCCL 资源 ------
        super().shutdown()
        if dp_group := getattr(self, "dp_group", None):
            stateless_destroy_torch_distributed_process_group(dp_group)

    def _pause_complete(self) -> bool:
        """Two-phase DP-aware pause.

        Phase 1: Set local pause state and ``pending_pause`` flag. If the
        engines are idle, kick-start them by setting ``engines_running`` to
        True so ranks enter the stepping loop and reach the all-reduce
        consensus checkpoint in ``_has_global_unfinished_reqs``.

        Phase 2 (in ``_has_global_unfinished_reqs``): Once the all-reduce
        confirms that **all** ranks have ``pending_pause`` set, collectively
        stop stepping and set ``ignore_start_dp_wave`` so that stale
        ``START_DP_WAVE`` messages cannot re-wake any engine.
        """
        # ------【DP】置 pending_pause 并强制 engines_running，让各 rank 进 step 循环达 all-reduce 共识 ------
        self.pending_pause = True
        self.engines_running = True

        return False

    def add_request(self, request: Request, request_wave: int = 0):
        # ------【核心逻辑】先走父类加入调度器，再做 DP 波次对齐 ------
        super().add_request(request, request_wave)
        # ------【DP】有协调器且波次不同：新波次直接更新，旧波次则唤醒引擎并通知协调器开下一波 ------
        if self.has_coordinator and request_wave != self.current_wave:
            if request_wave > self.current_wave:
                self.current_wave = request_wave
            elif (
                not self.engines_running
                and self.scheduler.pause_state == PauseState.UNPAUSED
            ):
                # Request received for an already-completed wave, notify
                # front-end that we need to start the next one.
                self.engines_running = True
                self.output_queue.put_nowait(
                    (-1, EngineCoreOutputs(start_wave=self.current_wave))
                )

    def resume_scheduler(self):
        # ------【DP】暂停在途或正在忽略 START_DP_WAVE 时禁止恢复，先等暂停 Future 完成 ------
        if self.pending_pause or (self.engines_running and self.ignore_start_dp_wave):
            raise RuntimeError(
                "resume_scheduler called while pause is still in "
                "flight. Wait for the pause future to resolve before "
                "resuming."
            )
        if self.engines_running:
            logger.debug("Resume called while engines are not paused, ignoring.")
            return

        super().resume_scheduler()
        self.ignore_start_dp_wave = False

        # Barrier: wait for all DP ranks to have resumed (and cleared
        # ignore_start_dp_wave) before any rank starts stepping. Uses
        # the existing all-reduce which is safe because engines are
        # stopped.
        # ------【DP + NCCL 通信】all-reduce 做恢复栅栏：所有 rank 都恢复后才允许任一方开始 step ------
        has_global_unfinished = ParallelConfig.has_unfinished_dp(
            self.dp_group, self.scheduler.has_unfinished_requests()
        )

        if has_global_unfinished:
            self.engines_running = True

    def barrier(self):
        """Blocking barrier on the DP process group (test-only utility)."""
        # ------【NCCL 通信】DP 进程组上的阻塞 barrier（测试用，保证各 rank 对齐） ------
        import torch.distributed as dist

        dist.barrier(group=self.dp_group)

    def _handle_client_request(
        self, request_type: EngineCoreRequestType, request: Any
    ) -> None:
        # ------【DP】START_DP_WAVE：协调器通知启动新波次；忽略旧波且跳过自身 rank 后唤醒引擎 ------
        if request_type == EngineCoreRequestType.START_DP_WAVE:
            if self.ignore_start_dp_wave:
                return
            new_wave, exclude_eng_index = request
            if exclude_eng_index != self.engine_index and (
                new_wave >= self.current_wave
            ):
                self.current_wave = new_wave
                if not self.engines_running:
                    logger.debug(
                        "EngineCore starting idle loop for wave %d.",
                        new_wave,
                    )
                    self.engines_running = True
        else:
            super()._handle_client_request(request_type, request)

    def _maybe_publish_request_counts(self):
        if not self.publish_dp_lb_stats:
            return

        # Publish our request counts (if they've changed), stamped with the
        # lockstep-synchronized step counter and wave number.
        # ------【DP】计数变化时上报，附上锁步同步的 step_counter 与 current_wave 供协调器做 LB ------
        counts = self.scheduler.get_request_counts()
        if counts != self.last_counts:
            self.last_counts = counts
            stats = SchedulerStats(
                *counts,
                kv_cache_usage=self.scheduler.get_kv_cache_usage(),
                step_counter=self.step_counter,
                current_wave=self.current_wave,
            )
            self.output_queue.put_nowait((-1, EngineCoreOutputs(scheduler_stats=stats)))

    def _should_throttle_prefills(self) -> bool:
        # Throttle new prefills to cadence-aligned steps for DP balancing.
        # step_counter is identical across DP ranks. On a fresh wave the
        # counter is 0, so prefills are admitted immediately after idle.
        # ------【DP】把新 prefill 节流到对齐 step_counter 的步上，保证各 rank 调度节奏一致 ------
        return (
            self.prefill_schedule_interval > 1
            and self.step_counter % self.prefill_schedule_interval != 0
        )

    @fault_tolerant_wrapper
    def run_busy_loop(self):
        """Core busy loop of the EngineCore for data parallel case."""

        # Loop until process is sent a SIGINT or SIGTERM
        # ------【核心逻辑】DP 主循环：取请求 → 上报计数 → 推进 EEP → step → all-reduce 锁步 ------
        while self._handle_shutdown():
            # 1) Poll the input queue until there is work to do.
            self._process_input_queue()
            # Publish request counts before and after GPU step to ensure freshness.
            self._maybe_publish_request_counts()

            # ------【EP/EPLB】若在弹性 EP 扩缩容中：推进状态机，完成则清状态，ready 则阻塞新请求 ------
            if self.eep_scaling_state is not None:
                state = self.eep_scaling_state
                if state.commit_requested or not state.is_ready_for_switch():
                    state.progress()
                if state.is_complete():
                    if state.worker_type == "removing":
                        raise SystemExit
                    self.process_input_queue_block = True
                    self.eep_scaling_state = None
                elif not state.commit_requested and state.is_ready_for_switch():
                    self.process_input_queue_block = True

            executed = self._process_engine_step()
            self._maybe_publish_request_counts()

            local_unfinished_reqs = self.scheduler.has_unfinished_requests()
            if not executed:
                # ------【DP】所有引擎都空闲则继续等，不 step ------
                if not local_unfinished_reqs and not self.engines_running:
                    # All engines are idle.
                    continue

                # Execute a dummy pass when no ready requests ran, unless the
                # engine is sleeping.
                # ------【DP】没 ready 请求但引擎醒着：跑 dummy batch 保持各 rank step 对齐（除非在睡眠） ------
                elif not self.model_executor.is_sleeping:
                    with self.capture_iteration_details(None) as iteration_details:
                        self.execute_dummy_batch()
                    if iteration_details is not None and not self.has_coordinator:
                        stats = self._make_iteration_details_stats(iteration_details)
                        self.output_queue.put_nowait(
                            (0, EngineCoreOutputs(scheduler_stats=stats))
                        )

            # 3) All-reduce operation to determine global unfinished reqs.
            # ------【DP + NCCL 通信】all-reduce 汇总各 rank 是否还有未完成请求，决定是否继续锁步 step ------
            self.engines_running = self._has_global_unfinished_reqs(
                local_unfinished_reqs
            )

            # ------【DP】全局空闲：rank0（或离线 SPMD 各 rank）通知波次完成，然后递增波次并重置计数 ------
            if not self.engines_running:
                if self.dp_rank == 0 or not self.has_coordinator:
                    # Notify client that we are pausing the loop.
                    logger.debug(
                        "Wave %d finished, pausing engine loop.", self.current_wave
                    )
                    # In the coordinator case, dp rank 0 sends updates to the
                    # coordinator. Otherwise (offline spmd case), each rank
                    # sends the update to its colocated front-end process.
                    client_index = -1 if self.has_coordinator else 0
                    self.output_queue.put_nowait(
                        (
                            client_index,
                            EngineCoreOutputs(wave_complete=self.current_wave),
                        )
                    )
                # Increment wave count and reset step counter.
                self.current_wave += 1
                self.step_counter = 0

        raise SystemExit

    def _has_global_unfinished_reqs(self, local_unfinished: bool) -> bool:
        # Optimization - only perform finish-sync all-reduce every 32 steps.
        self.step_counter += 1
        # ------【DP】每 32 步才做一次 all-reduce 完成同步，中间直接假设仍有活，摊薄通信开销 ------
        if self.step_counter % 32 != 0:
            return True

        # ------【DP + NCCL 通信】同步 has_unfinished 与 pause 共识，一次集合通信同时带两个标志 ------
        has_unfinished, pause_consensus = ParallelConfig.sync_dp_state(
            self.dp_group,
            has_unfinished=local_unfinished,
            pending_pause=self.pending_pause,
        )

        # ------【DP】所有 rank 达成暂停共识后：忽略后续 START_DP_WAVE 并清 pending_pause ------
        if pause_consensus:
            self.ignore_start_dp_wave = True
            self.pending_pause = False
            logger.debug("DP pause consensus reached, ignoring START_DP_WAVE.")

        return has_unfinished

    def reinitialize_distributed(
        self, reconfig_request: ReconfigureDistributedRequest
    ) -> str:
        from copy import deepcopy

        from vllm.distributed.elastic_ep.elastic_state import ElasticEPScalingState

        # ------【EP/EPLB】深拷贝并行配置，写入新的 DP size/rank/master 地址与端口 ------
        new_parallel_config = deepcopy(self.vllm_config.parallel_config)
        old_dp_size = new_parallel_config.data_parallel_size
        new_parallel_config.data_parallel_size = reconfig_request.new_data_parallel_size
        if (
            reconfig_request.new_data_parallel_rank
            != ReconfigureRankType.KEEP_CURRENT_RANK
        ):
            new_parallel_config.data_parallel_rank = (
                reconfig_request.new_data_parallel_rank
            )
        new_parallel_config.data_parallel_master_ip = (
            reconfig_request.new_data_parallel_master_ip
        )
        new_parallel_config.data_parallel_master_port = (
            reconfig_request.new_data_parallel_master_port
        )
        new_parallel_config._data_parallel_master_port_list = (
            reconfig_request.new_data_parallel_master_port_list
        )
        new_parallel_config._coord_store_port = reconfig_request.coord_store_port

        is_scale_down = reconfig_request.new_data_parallel_size < old_dp_size
        is_shutdown = (
            reconfig_request.new_data_parallel_rank
            == ReconfigureRankType.SHUTDOWN_CURRENT_RANK
        )

        # ------【EP/EPLB】已有扩缩在途则拒绝，避免并发重配置状态混乱 ------
        if self.eep_scaling_state is not None:
            raise RuntimeError("Elastic EP reconfiguration is already active")

        # ------【EP/EPLB】构造弹性 EP 状态机（removing/existing × scale_down/up），存起来供主循环推进 ------
        state = ElasticEPScalingState(
            model_executor=self.model_executor,
            engine_core=self,
            vllm_config=self.vllm_config,
            new_parallel_config=new_parallel_config,
            worker_type="removing" if is_shutdown else "existing",
            scale_type="scale_down" if is_scale_down else "scale_up",
            reconfig_request=reconfig_request,
        )
        self.eep_scaling_state = state

        # ------【EP/EPLB】解除输入队列阻塞，让主循环能推进扩缩状态机；返回 ready_key 供客户端轮询 ------
        self.process_input_queue_block = False
        logger.info(
            "[Elastic EP] Received reconfiguration request and starting scaling up/down"
        )
        return state.ready_key

    def commit_prepared_elastic_ep(self) -> None:
        state = self.eep_scaling_state
        # ------【EP/EPLB】无就绪的扩缩准备则报错；否则置 commit 标志并解除输入阻塞，触发实际切换 ------
        if state is None or state.commit_requested or not state.is_ready_for_switch():
            raise RuntimeError("No prepared Elastic EP reconfiguration is ready")
        state.commit_requested = True
        self.process_input_queue_block = False
        logger.info("[Elastic EP] Committing prepared reconfiguration")

    def _eep_send_engine_core_notification(
        self, notification_type: EEPNotificationType
    ):
        """
        Send notifications to EngineCoreClient, which can then forward
        the notifications to other engine core processes. It is used for:
        1) In scale down: removing core engines to notify EngineCoreClient
           so EngineCoreClient can release their ray placement groups;
        2) Both scale up/down: to notify EngineCoreClient that existing
           core engines have already switched to the new parallel setup.
        """
        dp_rank = self.vllm_config.parallel_config.data_parallel_rank
        # ------【EP/EPLB】把通知类型 + dp_rank 打包成 utility 结果，供 EngineCoreClient 转发/释放 PG ------
        notification_data = (notification_type.value, dp_rank)
        outputs = EngineCoreOutputs(
            utility_output=UtilityOutput(
                call_id=EEP_NOTIFICATION_CALL_ID,
                result=UtilityResult(notification_data),
            )
        )
        outputs.engine_index = self.engine_index

        # ------【ZMQ 通信】输出线程活着就走 output_queue；否则（如 KV init 前）直接临时建 PUSH 发送 ------
        if hasattr(self, "output_thread") and self.output_thread.is_alive():
            self.output_queue.put_nowait((0, outputs))
        else:
            encoder = MsgpackEncoder()
            with (
                zmq.Context() as ctx,
                make_zmq_socket(
                    ctx, self.addresses.outputs[0], zmq.PUSH, linger=4000
                ) as socket,
            ):
                socket.send_multipart(encoder.encode(outputs))

    def _eep_scale_up_before_kv_init(self):
        from vllm.distributed.elastic_ep.elastic_state import ElasticEPScalingState

        self.ignore_start_dp_wave = True
        # ------【EP/EPLB】新 rank 在 KV 初始化前构造 scale-up 状态机（worker_type=new） ------
        state = ElasticEPScalingState(
            model_executor=self.model_executor,
            engine_core=self,
            vllm_config=self.vllm_config,
            new_parallel_config=self.vllm_config.parallel_config,
            worker_type="new",
            scale_type="scale_up",
            reconfig_request=None,
        )
        if self.eep_scaling_state is not None:
            raise RuntimeError("Elastic EP reconfiguration is already active")
        self.eep_scaling_state = state
        # ------【EP/EPLB】跑 KV 初始化前的状态推进（预分配显存/建组），再解除输入阻塞 ------
        state.run_pre_kv_init_states()
        self.process_input_queue_block = False


class EngineCoreActorMixin:
    """
    Ray actor for running EngineCore in a data parallel context
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        addresses: EngineZmqAddresses,
        dp_rank: int = 0,
        local_dp_rank: int = 0,
    ):
        # Initialize tracer for distributed tracing if configured.
        # ------【进程管理】初始化分布式 tracing，供 Ray actor 场景观测 ------
        maybe_init_worker_tracer(
            instrumenting_module_name="vllm.engine_core",
            process_kind="engine_core",
            process_name=f"DPEngineCoreActor_DP{dp_rank}",
        )

        # ------【DP】保存地址并设置 DP index/local rank，供后续并行配置与 GPU 分配使用 ------
        self.addresses = addresses
        vllm_config.parallel_config.data_parallel_index = dp_rank
        vllm_config.parallel_config.data_parallel_rank_local = local_dp_rank

        # ------【PD 分离】补上 Ray actor 侧缺失的 NIXL side-channel host 环境变量 ------
        self._set_nixl_side_channel_host()

        # Set CUDA_VISIBLE_DEVICES as early as possible in actor life cycle
        # NOTE: in MP we set CUDA_VISIBLE_DEVICES at process creation time,
        # and this cannot be done in the same way for Ray because:
        # 1) Ray manages life cycle of all ray workers (including
        # DPEngineCoreActor)
        # 2) Ray sets CUDA_VISIBLE_DEVICES based on num_gpus configuration
        # To bypass 2, we need to also set
        # RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES, but vLLM workers created
        # thereafter would have CUDA_VISIBLE_DEVICES set, which is sticky:
        # https://github.com/ray-project/ray/blob/e752fc319ddedd9779a0989b6d3613909bad75c9/python/ray/_private/worker.py#L456 # noqa: E501
        # This is problematic because when the vLLM worker (a Ray actor)
        # executes a task, it indexes into the sticky CUDA_VISIBLE_DEVICES
        # rather than directly using the GPU ID, potentially resulting in
        # index out of bounds error. See:
        # https://github.com/ray-project/ray/pull/40461/files#diff-31e8159767361e4bc259b6d9883d9c0d5e5db780fcea4a52ead4ee3ee4a59a78R1860 # noqa: E501
        # and get_accelerator_ids_for_accelerator_resource() in worker.py
        # of ray.
        self._set_visible_devices(vllm_config, local_dp_rank)

    @staticmethod
    def _set_nixl_side_channel_host():
        import ray

        # The driver-side value is excluded from Ray actor env propagation.
        # Fill in an actor-local default while preserving explicit overrides.
        # ------【PD 分离】Ray 不会把 driver 侧该变量传进 actor，用本节点 IP 兜底且不覆盖显式设置 ------
        os.environ.setdefault(
            "VLLM_NIXL_SIDE_CHANNEL_HOST", ray.util.get_node_ip_address()
        )

    def _set_visible_devices(self, vllm_config: VllmConfig, local_dp_rank: int):
        from vllm.platforms import current_platform

        # ------【NUMA 亲和】XPU 无需处理；CUDA 等平台按其控制变量给本地 DP rank 分配物理 GPU ------
        if current_platform.is_xpu():
            pass
        else:
            device_control_env_var = current_platform.device_control_env_var
            self._set_assigned_physical_gpu_ids(
                vllm_config, local_dp_rank, device_control_env_var
            )

    def _set_assigned_physical_gpu_ids(
        self,
        vllm_config: VllmConfig,
        local_dp_rank: int,
        device_control_env_var: str,
    ):
        world_size = vllm_config.parallel_config.world_size
        try:
            # ------【NUMA 亲和】按 local_dp_rank 从设备控制变量里切片出本 rank 的物理 GPU 集合 ------
            physical_gpu_ids = get_physical_gpu_ids_for_local_dp_rank(
                device_control_env_var,
                local_dp_rank,
                world_size,
                user_assigned_gpu_ids=(
                    vllm_config.parallel_config.assigned_physical_gpu_ids
                ),
            )
            vllm_config.parallel_config.assigned_physical_gpu_ids = physical_gpu_ids
        except IndexError as e:
            raise Exception(
                f"Error computing assigned_physical_gpu_ids: "
                f"local range: [{local_dp_rank * world_size}, "
                f"{(local_dp_rank + 1) * world_size}) "
                f'base value: "{os.getenv(device_control_env_var)}"'
            ) from e

    @contextmanager
    def _perform_handshakes(
        self,
        handshake_address: str,
        identity: bytes,
        local_client: bool,
        vllm_config: VllmConfig,
        client_handshake_address: str | None,
    ):
        """
        For Ray, we don't need to actually perform handshake.
        All addresses information is known before the actor creation.
        Therefore, we simply yield these addresses.
        """
        # ------【ZMQ 通信】Ray 下地址在 actor 创建前已知，无需真正握手，直接 yield 已有地址 ------
        yield self.addresses

    def wait_for_init(self):
        """
        Wait until the engine core is initialized.

        This is just an empty method. When ray.get() on this method
        (or any other method of the actor) returns, it is guaranteed
        that actor creation (i.e., __init__) is complete.
        """
        # ------【进程管理】空方法：ray.get() 返回即证明 actor 的 __init__ 已完成 ------
        pass

    def run(self):
        """
        Run the engine core busy loop.
        """
        try:
            # ------【核心逻辑】跑 busy loop；SystemExit 视为正常退出，其它异常记录后上抛 ------
            self.run_busy_loop()  # type: ignore[attr-defined]
        except SystemExit:
            logger.debug("EngineCore exiting.")
            raise
        except Exception:
            logger.exception("EngineCore encountered a fatal error.")
            raise
        finally:
            # ------【进程管理】无论正常/异常退出都兜底 shutdown 释放资源 ------
            self.shutdown()  # type: ignore[attr-defined]


class DPMoEEngineCoreActor(EngineCoreActorMixin, DPEngineCoreProc):
    """Used for MoE model data parallel cases."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_client: bool,
        addresses: EngineZmqAddresses,
        executor_class: type[Executor],
        log_stats: bool,
        dp_rank: int = 0,
        local_dp_rank: int = 0,
    ):
        # ------【DP】先设 DP rank，再按 MRO 依序调两个父类：Mixin 管 Ray/GPU，DPProc 管锁步协调 ------
        vllm_config.parallel_config.data_parallel_rank = dp_rank

        EngineCoreActorMixin.__init__(
            self, vllm_config, addresses, dp_rank, local_dp_rank
        )
        DPEngineCoreProc.__init__(
            self, vllm_config, local_client, "", executor_class, log_stats
        )


class EngineCoreActor(EngineCoreActorMixin, EngineCoreProc):
    """Used for non-MoE and/or non-DP cases."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_client: bool,
        addresses: EngineZmqAddresses,
        executor_class: type[Executor],
        log_stats: bool,
        dp_rank: int = 0,
        local_dp_rank: int = 0,
    ):
        # ------【DP】非 MoE/非 DP：先把本 rank 重配成独立 DP=1，再按 MRO 依序调 Mixin 与 Proc ------
        vllm_config.parallel_config.reconfigure_for_independent_dp_rank()
        EngineCoreActorMixin.__init__(
            self, vllm_config, addresses, dp_rank, local_dp_rank
        )
        EngineCoreProc.__init__(
            self,
            vllm_config,
            local_client,
            "",
            executor_class,
            log_stats,
            engine_index=dp_rank,
        )
