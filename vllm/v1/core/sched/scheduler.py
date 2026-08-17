# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import itertools
import time
from collections import defaultdict, deque
from collections.abc import Iterable
from dataclasses import replace
from typing import Any

from vllm.compilation.cuda_graph import CUDAGraphStat
from vllm.config import KVEventsConfig, VllmConfig
from vllm.distributed.ec_transfer.ec_connector.base import (
    ECConnectorBase,
    ECConnectorMetadata,
    ECConnectorRole,
)
from vllm.distributed.ec_transfer.ec_connector.factory import ECConnectorFactory
from vllm.distributed.kv_events import EventPublisherFactory, KVEventBatch
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.distributed.kv_transfer.kv_connector.v1 import (
    KVConnectorBase_V1,
    KVConnectorRole,
    SupportsHMA,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import KVConnectorStats
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.routed_experts_capturer import (
    RoutedExpertsManager,
)
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
from vllm.multimodal.encoder_budget import MultiModalBudget
from vllm.multimodal.utils import get_mm_features_in_window
from vllm.v1.core.encoder_cache_manager import (
    EncoderCacheManager,
    EncoderDecoderCacheManager,
)
from vllm.v1.core.kv_cache_manager import KVCacheBlocks, KVCacheManager
from vllm.v1.core.kv_cache_metrics import KVCacheMetricsCollector
from vllm.v1.core.kv_cache_utils import KVCacheBlock
from vllm.v1.core.sched.interface import PauseState, SchedulerInterface
from vllm.v1.core.sched.output import (
    CachedRequestData,
    GrammarOutput,
    NewRequestData,
    ScheduledEncoderInputStats,
    SchedulerOutput,
)
from vllm.v1.core.sched.request_queue import (
    RequestQueue,
    SchedulingPolicy,
    create_request_queue,
)
from vllm.v1.core.sched.utils import check_stop, remove_all
from vllm.v1.engine import EngineCoreEventType, EngineCoreOutput, EngineCoreOutputs
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.metrics.perf import ModelMetrics, PerfStats
from vllm.v1.metrics.stats import PrefixCacheStats, SchedulerStats
from vllm.v1.outputs import DraftTokenIds, KVConnectorOutput, ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus, StreamingUpdate
from vllm.v1.spec_decode.dynamic.utils import build_dynamic_sd_schedule_lookup
from vllm.v1.spec_decode.metrics import SpecDecodingStats
from vllm.v1.structured_output import StructuredOutputGrammar, StructuredOutputManager
from vllm.v1.utils import record_function_or_nullcontext

logger = init_logger(__name__)


# 真正的调度器类
class Scheduler(SchedulerInterface):
    """
    === 类说明 ===
        继承: SchedulerInterface(ABC)
        职责: vLLM V1 核心调度器。管理请求生命周期（等待→运行→完成），
              分配 KV cache block，每步选取 batch 交给模型执行器做前向推理。

    === 基类方法实现 (22个) ===
        * schedule()              — 核心：选取请求 + 分配 KV cache
        update_from_output()    — 根据模型输出更新请求状态，返回 EngineCoreOutputs

        * add_request()           — 新请求加入等待队列
        * finish_requests()       — 中止/停止请求

        get_num_unfinished_requests() — 未完成请求数
        has_unfinished_requests()     — 是否有未完成请求
        has_finished_requests()       — 是否有已完成待清理的请求
        has_requests()          — 是否有任何待处理请求
        get_request_counts()    — 返回 (运行中, 等待中) 数量
        get_kv_cache_usage()    — KV cache 使用率

        update_draft_token_ids()      — 更新草稿 token（投机解码）
        update_draft_token_ids_in_output() — 同上 + 更新 SchedulerOutput

        get_grammar_bitmask()   — 获取结构化输出语法掩码

        pause_state / set_pause_state — 暂停状态
        reset_prefix_cache()    — 重置 KV 前缀缓存
        reset_encoder_cache()   — 重置编码器缓存
        make_stats()            — 生成调度统计
        shutdown()              — 关闭
        get_kv_connector() / get_ec_connector() / get_kv_event_publisher_config()

    === [新增] 公有方法 (2个) ===
        reset_connector_cache()     — 重置 KV connector 缓存 (disaggregated prefill)
        make_spec_decoding_stats()  — 生成投机解码统计

    === [新增] 核心成员属性 ===
        —— 请求容器 ——
            requests: dict[str, Request]    — 所有请求 {req_id → Request}

            waiting: RequestQueue           — 等待队列（按策略排队）
            skipped_waiting: RequestQueue   — 因依赖/约束被跳过的请求
            running: list[Request]          — 正在运行的请求列表

            finished_req_ids: set[str]      — 本步完成的请求 ID
        —— 调度约束 ——
            max_num_running_reqs: int       — 最大并发请求数
            max_num_scheduled_tokens: int   — 每步最大调度 token 数
            max_model_len: int              — 模型最大上下文长度
            block_size: int                 — KV cache block 大小
            current_step: int               — 调度步数计数器
            policy: SchedulingPolicy        — 调度策略 (FCFS/Priority)
        —— KV Cache ——
            kv_cache_manager: KVCacheManager — KV cache 管理器（block 分配/回收）
        —— 投机解码 ——
            use_eagle: bool                 — 是否使用 EAGLE 投机解码
            num_spec_tokens: int            — 投机 token 数
            num_lookahead_tokens: int       — 前瞻 token 数
        —— KV 传输 (disaggregated prefill) ——
            connector: KVConnectorBase_V1   — KV connector（P/D 分离场景）
            ec_connector: ECConnectorBase   — EC connector（弹性扩容）
        —— 多模态 ——
            encoder_cache_manager           — 编码器缓存管理器
            max_num_encoder_input_tokens    — 编码器最大输入 token 数
        —— 状态追踪 ——
            sched_step_seq / processed_step_seq — 调度/处理步序号
            deferred_frees: deque           — 延迟释放的 block 队列
    """

    # 构造
    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
        structured_output_manager: StructuredOutputManager,
        block_size: int,
        hash_block_size: int | None = None,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
        include_finished_set: bool = False,
        log_stats: bool = False,
    ) -> None:
        # ------【核心逻辑】缓存并拆解各子配置对象，供后续调度逻辑按需读取 ------
        self.vllm_config = vllm_config
        self.scheduler_config = vllm_config.scheduler_config
        self.cache_config = vllm_config.cache_config
        self.lora_config = vllm_config.lora_config
        self.kv_cache_config = kv_cache_config
        self.kv_events_config = vllm_config.kv_events_config
        self.parallel_config = vllm_config.parallel_config
        self.log_stats = log_stats
        self.observability_config = vllm_config.observability_config
        # ------【核心逻辑】可选创建 KV cache 指标采集器，用于观测 KV 命中与占用 ------
        self.kv_metrics_collector: KVCacheMetricsCollector | None = None
        if self.observability_config.kv_cache_metrics:
            self.kv_metrics_collector = KVCacheMetricsCollector(
                self.observability_config.kv_cache_metrics_sample,
            )
        self.structured_output_manager = structured_output_manager

        # ------【核心逻辑】记录是否 encoder-decoder / encoder-only 架构，影响后续调度分支 ------
        # 这两个是为其他架构准备的
        self.is_encoder_decoder = vllm_config.model_config.is_encoder_decoder
        self.is_encoder_only = vllm_config.is_encoder_only

        # include_finished_set controls whether a separate set of finished
        # request ids should be included in the EngineCoreOutputs returned
        # by update_from_outputs(). This is currently used in the multi-engine
        # case to track request lifetimes efficiently.
        '''
            include_finished_set 控制是否要在 update_from_outputs() 返回的 
            EngineCoreOutputs 中额外附带一个已完成请求 ID 的集合。
            目前用于多引擎（DP）场景，以便高效追踪请求的生命周期。
        '''

        # ------【DP】按 client_index 分组记录完成请求，供多引擎 DP 场景高效追踪生命周期 ------
        # 这个是按 client_index 分组记录本部完成/中止的请求ID，避免前端额外轮询
        self.finished_req_ids_dict: dict[int, set[str]] | None = (
            defaultdict(set) if include_finished_set else None
        )

        # ------【核心逻辑】记录上一步调度过的请求，供 MRV1 多模态/调度历史回溯使用 ------
        # 多模态
        # Track requests scheduled in prior step (MRV1-only).
        self.prev_step_scheduled_req_ids: set[str] = set()



        # ------【chunked prefill】每轮最大请求数 / 最大调度 token 数，构成调度预算的硬约束 ------
        # Scheduling constraints.
        # 调度器的约束条件
        self.max_num_running_reqs = self.scheduler_config.max_num_seqs # 每轮的最大请求数
        self.max_num_scheduled_tokens = (   # 每轮的最大token数
            self.scheduler_config.max_num_scheduled_tokens
            if self.scheduler_config.max_num_scheduled_tokens is not None
            else self.scheduler_config.max_num_batched_tokens
        )

        # ------【核心逻辑】模型最大上下文长度 + KV cache 事件追踪开关 ------
        # 模型的上下文长度，也就是模型能处理的最大 token 数（prompt + output）
        self.max_model_len = vllm_config.model_config.max_model_len
        self.enable_kv_cache_events = ( # KVcache事件追踪开关
            self.kv_events_config is not None
            and self.kv_events_config.enable_kv_cache_events
        )

        # ------【核心逻辑】每步采样 token 数（扩散模型不采样），用于预留位置计算 ------
        # 每步采样的token数：decode采样1个
        # Diffusion models may not sample any tokens for a denoising step.
        self.num_sampled_tokens_per_step = (
            1 if not vllm_config.model_config.is_diffusion else 0
        )

        # ------【PD 分离】初始化 KV connector 相关状态占位，待下方按配置真正创建 ------
        # Create KVConnector for the Scheduler. Note that each Worker
        # will have a corresponding KVConnector with Role=WORKER.
        # KV Connector pushes/pull of remote KVs for P/D and offloading.
        self.connector = None
        self.connector_prefix_cache_stats: PrefixCacheStats | None = None
        self.recompute_kv_load_failures = True
        self.defer_block_free = False


        # PD分离场景的标志位，被抢占的请求的在途输出是否必须丢弃

        # Whether a preempted request's in-flight output must be dropped; see
        # KVConnectorBase_V1.requires_kv_delivery.
        self.requires_kv_delivery = False
        # ------【PD 分离】按配置创建 KV connector（P/D 分离或 offload），并配置加载失败/延迟释放策略 ------
        kv_transfer_config = self.vllm_config.kv_transfer_config
        if kv_transfer_config is not None:
            assert not self.is_encoder_decoder, (
                "Encoder-decoder models are not currently supported with KV connectors"
            )
            self.connector = KVConnectorFactory.create_connector(
                config=self.vllm_config,
                role=KVConnectorRole.SCHEDULER,
                kv_cache_config=self.kv_cache_config,
            )
            if self.log_stats:
                self.connector_prefix_cache_stats = PrefixCacheStats()
            kv_load_failure_policy = kv_transfer_config.kv_load_failure_policy
            self.recompute_kv_load_failures = kv_load_failure_policy == "recompute"

            # ------【PD 分离+异步 RPC】重叠 batch 下延迟释放 block，避免 consumer 重写与未完成写竞态 ------
            # With overlapping batches (async scheduling or PP), a step may
            # still be writing a freed request's KV blocks. A consumer KV
            # Connector can reallocate and fill those blocks via a load that
            # isn't ordered against that write, so defer freeing them.
            multiple_inflight_batches = self.vllm_config.max_concurrent_batches > 1
            if multiple_inflight_batches and kv_transfer_config.is_kv_consumer:
                self.defer_block_free = True

            self.requires_kv_delivery = self.connector.requires_kv_delivery

        # ------【核心逻辑】创建 KV 事件发布器，向外部发送 KV cache 事件（保存/失效等） ------
        self.kv_event_publisher = EventPublisherFactory.create(
            self.kv_events_config,
            self.parallel_config.data_parallel_index,
        )
        # ------【PD 分离】创建 EC connector（弹性容量传输），用于跨实例多模态/编码器缓存搬运 ------
        self.ec_connector = None
        if self.vllm_config.ec_transfer_config is not None:
            self.ec_connector = ECConnectorFactory.create_connector(
                config=self.vllm_config, role=ECConnectorRole.SCHEDULER
            )

        # ------【核心逻辑】校验 GPU block 池大小有效，保证后续 KV cache 分配可用 ------
        num_gpu_blocks = self.cache_config.num_gpu_blocks
        assert num_gpu_blocks is not None and num_gpu_blocks > 0

        # ------【TP】记录 KV block 大小与 decode/prefill 上下文并行(CP)度数，供 KV 分配对齐 ------
        self.block_size = block_size
        self.dcp_world_size = vllm_config.parallel_config.decode_context_parallel_size
        self.pcp_world_size = vllm_config.parallel_config.prefill_context_parallel_size




        # ------【核心逻辑】保存所有请求的全局映射 req_id -> Request ------
        # 调度器的所有请求的全集
        # req_id -> Request
        self.requests: dict[str, Request] = {}

        # 调度策略枚举
        # FCFS： 先来先服务，请求按到达顺序排队
        # PRIORITY： 高优先级请求被schedule()选中

        # ------【核心逻辑】解析调度策略（FCFS/PRIORITY），决定请求挑选顺序 ------
        # Scheduling policy
        try:
            self.policy = SchedulingPolicy(self.scheduler_config.policy)
        except ValueError as e:
            raise ValueError(
                f"Unknown scheduling policy: {self.scheduler_config.policy}"
            ) from e


        # 任务队列

        # ------【核心逻辑】创建就绪队列、跳过队列与运行列表，构成调度器三态容器 ------
        # Priority queues for requests.
        self.waiting = create_request_queue(self.policy) # 就绪队列

        # requests skipped in waiting flow due async deps or constraints.
        self.skipped_waiting = create_request_queue(self.policy) # 阻塞队列

        self.running: list[Request] = []                    # 运行队列


        # The request IDs that are finished in between the previous and the
        # current steps. This is used to notify the workers about the finished
        # requests so that they can free the cached states for those requests.
        # This is flushed at the end of each scheduling step.
        # ------【核心逻辑】记录跨步完成的请求 ID，用于通知 worker 释放其缓存状态 ------
        self.finished_req_ids: set[str] = set() # 上一步到这一步之间新完成的请求，已完成的请求集合

        # IDs of requests preempted since the last call to schedule().

        # ------【核心逻辑】本轮被抢占请求 ID 集合，通知 worker 重置其 CUDA 状态 ------
        self.reset_preempted_req_ids: set[str] = set()        # 本轮被抢占的请求的IDs，用来通知reset，这些请求的CUDA状态（清kvcache, 清cuda graph 缓存）

        # Counter for requests waiting for streaming input. Used to calculate
        # number of unfinished requests
        # ------【核心逻辑】等待流式输入的请求计数，用于精确统计未完成请求数 ------
        self.num_waiting_for_streaming_input: int = 0 # 流式多轮对话场景的计数器

        # ------【PD 分离】记录异步 KV 传输完成/失败的请求 ID，供传输后结算 ------
        # KV Connector: requests in process of async KV loading or recving
        self.finished_recving_kv_req_ids: set[str] = set() # P/D分离：KV传输完成
        self.failed_recving_kv_req_ids: set[str] = set()   # P/D分离：KV传输失败


        # 结构化输出的 grammar 编译失败
        # ------【结构化输出/grammar】记录 grammar 编译失败的请求，稍后作为请求级错误结束 ------
        # Grammar compilation failures to finish as per-request errors in
        # update_from_output.
        self.grammar_compile_error_reqs: set[str] = set()

        # Encoder-related.
        # Calculate encoder cache size if applicable
        # ------【核心逻辑】探测多模态支持并计算编码器输入预算（encoder token 上限） ------
        supports_mm_inputs = mm_registry.supports_multimodal_inputs(
            vllm_config.model_config
        )
        mm_budget = (
            MultiModalBudget(vllm_config, mm_registry) if supports_mm_inputs else None
        )


        # 多模态、encoder-decoder 的encoder cache
        # NOTE: Text-only encoder-decoder models are implemented as
        # multi-modal models for convenience
        # Example: https://github.com/vllm-project/bart-plugin
        if self.is_encoder_decoder:
            assert mm_budget and len(mm_budget.mm_max_toks_per_item) <= 1, (
                "Encoder-decoder models are expected to implement the "
                "multimodal interface with at most one modality."
            )

        # ------【核心逻辑】按多模态预算创建编码器缓存管理器（encoder cache） ------
        self.max_num_encoder_input_tokens = (
            mm_budget.encoder_compute_budget if mm_budget else 0
        )
        encoder_cache_size = mm_budget.encoder_cache_size if mm_budget else 0
        manager_cls_obj = vllm_config.ec_manager_config.get_encoder_cache_manager_obj()
        if manager_cls_obj is not None:
            self.encoder_cache_manager = manager_cls_obj(cache_size=encoder_cache_size)
        else:
            self.encoder_cache_manager = (
                EncoderDecoderCacheManager(cache_size=encoder_cache_size)
                if self.is_encoder_decoder
                else EncoderCacheManager(cache_size=encoder_cache_size)
            )



        # ------【投机解码】解析投机解码配置，设定 eagle/草稿模型与前瞻 token 数 ------
        # 投机解码配置
        speculative_config = vllm_config.speculative_config
        self.use_eagle = False
        self.num_spec_tokens = vllm_config.num_speculative_tokens
        self.num_lookahead_tokens = 0
        self.dynamic_sd_lookup: list[int] | None = None
        if speculative_config is not None:
            # ------【投机解码】batch_size 相关的动态草稿 token 数查找表，替代固定 K ------
            if speculative_config.num_speculative_tokens_per_batch_size:
                self.dynamic_sd_lookup = build_dynamic_sd_schedule_lookup(
                    speculative_config.num_speculative_tokens_per_batch_size,
                    vllm_max_batch_size=self.scheduler_config.max_num_seqs,
                    vllm_num_speculative_tokens=self.num_spec_tokens,
                )
            if speculative_config.use_eagle():
                self.use_eagle = True
                self.num_lookahead_tokens = self.num_spec_tokens
            if speculative_config.uses_draft_model():
                self.num_lookahead_tokens = self.num_spec_tokens
            if speculative_config.use_dflash():
                # DFlash requires an extra lookahead slot since it uses in-fill-style
                # decoding instead of standard next-token sampling, so it has a query
                # for the last sampled token plus queries for each draft token.
                self.num_lookahead_tokens = self.num_spec_tokens + 1
            if speculative_config.use_dspark():
                # DSpark drafts a block of num_spec_tokens query tokens in which the
                # anchor itself is the first prediction position (no separate bonus
                # query), so it needs exactly num_spec_tokens lookahead slots.
                self.num_lookahead_tokens = self.num_spec_tokens














        # 构造KVcache管理器
        '''
        KVCacheManager 是 Scheduler 和底层 KV cache 之间的抽象层
        '''
        # ------【前缀缓存】创建 KVCacheManager，配置前缀缓存/hash block 大小/水线等核心参数 ------
        # Create the KV cache manager.
        if hash_block_size is None:
            hash_block_size = block_size #

        self.hash_block_size = hash_block_size
        self.kv_cache_manager = KVCacheManager(
            kv_cache_config=kv_cache_config, # cache配置
            max_model_len=self.max_model_len, # 模型最大上下文长度
            max_in_flight_tokens=vllm_config.max_in_flight_tokens, # 最大计算中token
            enable_caching=self.cache_config.enable_prefix_caching, # 使能前缀缓存
            use_eagle=self.use_eagle, 
            log_stats=self.log_stats,
            enable_kv_cache_events=self.enable_kv_cache_events, # 使能时间记录
            dcp_world_size=self.dcp_world_size,
            pcp_world_size=1,
            scheduler_block_size=self.block_size, # block大小
            hash_block_size=hash_block_size, # hashblock大小
            metrics_collector=self.kv_metrics_collector,
            watermark=self.scheduler_config.watermark,
        )


        # ------【PD 分离】把 GPU block 池绑定到 KV connector，供远端 KV 读写直接访问显存 ------
        # PD分离
        # Bind GPU block pool to the KV connector. This must happen after
        # kv_cache_manager is constructed so block_pool is available.
        if self.connector is not None:
            self.connector.bind_gpu_block_pool(self.kv_cache_manager.block_pool)

        # ------【PP】记录是否启用流水线并行与 v2 model runner，影响调度/通信路径 ------
        # 流水线并行
        self.use_pp = self.parallel_config.pipeline_parallel_size > 1
        self.use_v2_model_runner = vllm_config.use_v2_model_runner



        # ------【核心逻辑】调度步数计数器，驱动 PP/异步解码节流节奏 ------
        # 调度步数计数器
        # Scheduler iteration counter. Drives the V2+PP+async decode-throttle
        # cadence (`next_decode_eligible_step`).
        self.current_step = 0


        # DP prefill balancing: Flag to track whether the last cadence-aligned
        # prefill batch fully drained the waiting queue. Prefill throttling
        # is disabled in this case.

        # ------【DP】prefill 容量饱和标记 + 是否要求整序列一次放入显存 ------
        # DP prefill 均衡的 容量饱和 标记
        self.prefill_capacity_bound = False
        self.scheduler_reserve_full_isl = (
            self.scheduler_config.scheduler_reserve_full_isl
        )

        # ------【核心逻辑】记录是否有 Mamba 层及新 KV block 是否需要清零 ------
        self.has_mamba_layers = kv_cache_config.has_mamba_layers

        # 新分配的KVcache block 是否需要先清零才能用
        self.needs_kv_cache_zeroing = kv_cache_config.needs_kv_cache_zeroing 


        # ------【PD 分离】异步加载远端 KV 的 block 跳过清零，避免清零与远端写入竞态 ------
        # Blocks that async KV loads will overwrite this step, skipped from
        # zeroing since the zeroing could race the out-of-band write.

        # PD分离用的，异步加载远程 KV 的 block 需要跳过清零——清零操作和远程 KV 数据写入可能竞态，
        # 而且远程数据马上就会完整覆写这些 block，清零是多余工作。不看 P/D 分离直接忽略
        self._skip_zero_block_ids: set[int] = set() 
        # ------【核心逻辑】Mamba align 模式是否需要 block 对齐切分 + 细粒度 partial tail 命中 ------
        self.need_mamba_block_aligned_split = (
            self.has_mamba_layers and self.cache_config.mamba_cache_mode == "align"
        )
        # A finer prefix_match_unit is configured: a mamba partial tail entry
        # can only be registered by a step ending exactly at the prompt's last
        # hash boundary, so the split adds that stop.
        self.mamba_partial_cache_hit = (
            self.need_mamba_block_aligned_split
            and self.hash_block_size < self.block_size
        )

        # Counts of non-empty steps scheduled / processed. update_from_output
        # is called once per scheduled step in FIFO order, so these stay in sync.

        # ------【异步 RPC】调度/处理步序号，构成延迟释放的 fence 屏障 ------
        # 异步调度场景下的延迟释放机制
        self.sched_step_seq = 0 # 发了多少次调度（CPU 侧）
        self.processed_step_seq = 0 # 多少次前向结果回来了（GPU 侧）

        
        # FIFO of (fence_seq, blocks): blocks become safe to free once
        # processed_step_seq >= fence_seq.
        # ------【异步 RPC】延迟释放队列：等 processed_step_seq 追平 fence 才安全回收 block ------
        self.deferred_frees: deque[tuple[int, list[KVCacheBlock]]] = deque()

        # ------【核心逻辑】可选创建 MFU 性能指标对象，用于吞吐/利用率统计 ------
        self.perf_metrics: ModelMetrics | None = None
        if self.log_stats and vllm_config.observability_config.enable_mfu_metrics:
            self.perf_metrics = ModelMetrics(vllm_config)


        # ------【EP/EPLB】MoE 路由专家导出开关，返回每 token 走了哪个 expert 供负载分析 ------
        # MoE 模型专用，打开后就可以返回每个token走了哪个expert的路由，给外部做负载分析
        self.enable_return_routed_experts = (
            vllm_config.model_config.enable_return_routed_experts
        )

        if self.enable_return_routed_experts:
            assert self.dcp_world_size == 1 and self.pcp_world_size == 1, (
                "enable_return_routed_experts does not support context parallelism "
                "(dcp_world_size > 1 or pcp_world_size > 1)"
            )

            # ------【EP/EPLB】创建专家路由管理器，并预留 block 快照以应对异步调度竞态 ------
            self.routed_experts_mgr = RoutedExpertsManager(
                vllm_config=vllm_config,
                kv_cache_config=kv_cache_config,
            )
            # Block-ID snapshot taken at schedule time (before forward),
            # so update_from_output can read slot data even if a later
            # schedule() frees the blocks (async scheduling race).
            self._re_block_ids: dict[str, list[int]] = {}


        # ------【核心逻辑】暂停状态初始化为未暂停，暂停时调度预算会被清零 ------
        # 调度器的状态 = 未暂停
        self._pause_state: PauseState = PauseState.UNPAUSED

        # In-flight requests still prefilling (prefill chunks + in-progress
        # async KV loads). Their remaining-block reservation gates async loads.

        # 正在prefill中的请求集合
        # ------【核心逻辑】正在 prefill 中的请求集合，其剩余 block 预留用于门控异步加载 ------
        self._inflight_prefills: set[Request] = set()







    # [新增] 将 prefill chunk 对齐到 Mamba 状态缓存边界
    def _mamba_block_aligned_split(
        self,
        request: Request,
        num_new_tokens: int,
        num_new_local_computed_tokens: int = 0,
        num_external_computed_tokens: int = 0,
    ) -> int:
        """Clip a prefill chunk so it ends where Mamba state must be cached.

        In "align" cache mode reusable SSM states are materialized at block
        boundaries, plus mandatory early stops (the prompt's partial-tail hash
        boundary, a detected shared-prefix junction). If a block is larger
        than the configured prefill chunk limit, intermediate chunks keep
        private running state until they reach the next cacheable position.
        """
        # ------【核心逻辑】计算本次切分的起始位置，并只在 prefill 阶段做对齐 ------
        start = (
            request.num_computed_tokens
            + num_new_local_computed_tokens
            + num_external_computed_tokens
        )
        # Split only during prefill: `request.num_tokens - 1` extends this to
        # resumed requests replaying their output tokens.
        if start >= max(request.num_prompt_tokens, request.num_tokens - 1):
            return num_new_tokens

        # ------【核心逻辑+投机解码】计算最后一个可缓存 block 边界，eagle 下回退一块避免 miss ------
        block_size = self.cache_config.block_size
        # The last block-aligned position whose state can be cached. With
        # Eagle, FullAttn prunes the last matching block, so back off one
        # block to avoid a Mamba cache miss.
        last_cache_position = request.num_tokens - request.num_tokens % block_size
        if self.use_eagle:
            last_cache_position = max(last_cache_position - block_size, 0)

        # ------【chunked prefill】让 prefill chunk 尽量停在 block 边界，装不下时允许子块推进 ------
        end = start + num_new_tokens
        # Until `last_cache_position`, prefer chunks ending on block
        # boundaries. When a block cannot fit in any configured prefill chunk,
        # allow sub-block progress and re-align at the next reachable boundary.
        if end < last_cache_position:
            max_prefill_tokens = self.max_num_scheduled_tokens
            long_prefill_threshold = self.scheduler_config.long_prefill_token_threshold
            if long_prefill_threshold > 0:
                max_prefill_tokens = min(max_prefill_tokens, long_prefill_threshold)
            aligned_end = end // block_size * block_size
            if aligned_end > start or block_size <= max_prefill_tokens:
                end = aligned_end

        # ------【前缀缓存】收集所有必须提前停止的位置（块边界/尾边界/共享前缀交界），取最早者 ------
        next_block_boundary = (start // block_size + 1) * block_size
        tail_boundary = (
            request.num_prompt_tokens // self.hash_block_size * self.hash_block_size
            if self.mamba_partial_cache_hit
            else 0
        )
        stops = (
            # Resumed mid-block (fine-grained partial hash hit): re-align to
            # the block grid before running on, so the crossed boundary's
            # state is materialized (unless it is past the cacheable range).
            next_block_boundary
            if start % block_size != 0 and next_block_boundary <= last_cache_position
            else 0,
            # Never run past the last cacheable block boundary mid-chunk.
            last_cache_position,
            # Fine-grained hits: the prompt's partial-tail entry can only be
            # registered by a chunk ending exactly at its last hash boundary.
            tail_boundary
            if last_cache_position < tail_boundary < request.num_prompt_tokens
            else 0,
            # Marconi shared-prefix junction, block-floored (a sub-block
            # junction's state is not separately cacheable): cache its state
            # so sibling requests sharing the prefix can reuse it.
            start + (request.shared_prefix_boundary - start) // block_size * block_size
            if start < request.shared_prefix_boundary < end
            else 0,
        )
        # ------【核心逻辑】取 chunk 内部最早的强制停止位作为切分终点，返回本次可推进 token 数 ------
        # Stop at the earliest mandatory position strictly inside the chunk.
        end = min((s for s in stops if start < s < end), default=end)
        return max(end - start, 0)




    '''
    Request token 计数器速查:
      num_prompt_tokens      — 原始 prompt 长度, 初始化后不变
      num_tokens             — 已确认 token = prompt + output (len(_all_token_ids))

      num_tokens_with_spec   — num_tokens + len(spec_token_ids), num_computed ，总tokens数目标（算上投机解码了）
      num_computed_tokens    — 已计算且有 KV cache 的 token 数 (抢占时清零)
      num_new_tokens = 未计算kvcache的tokens数量

      num_output_tokens      — 已产出 output token 数 (判断 max_tokens 停止)


      num_in_flight_tokens   — 正在计算中的tokens数量
      num_output_placeholders — 异步调度预占位数 (调度前赊账, 输出后核销, 抢占时清零)
      num_stale_output_tokens — 被抢占时继承的 in-flight 僵尸计数 (逐步消化, 不改当前计数)

    '''






    # ══════════════════════════════════════════════════════════════
    # 接口实现: schedule() — 核心调度，每次前向调用一次，发出调度任务
    # ══════════════════════════════════════════════════════════════
    def schedule(self, throttle_prefills: bool = False) -> SchedulerOutput:



        # Phase 0: 初始化变量与预算
        # ------【核心逻辑】初始化本轮步计数、token/encoder 预算与各调度结果容器 ------
        self.current_step += 1
        # NOTE(woosuk) on the scheduling algorithm:
        # There's no "decoding phase" nor "prefill phase" in the scheduler.
        # Each request just has the num_computed_tokens and
        # num_tokens_with_spec. num_tokens_with_spec =
        # len(prompt_token_ids) + len(output_token_ids) + len(spec_token_ids).
        # At each step, the scheduler tries to assign tokens to the requests
        # so that each request's num_computed_tokens can catch up its
        # num_tokens_with_spec. This is general enough to cover
        # chunked prefills, prefix caching, speculative decoding,
        # and the "jump decoding" optimization in the future.

        # 不分什么prefill阶段，decode阶段了，直接每步都更新

        scheduled_new_reqs: list[Request] = [] # 从waiting新拉进来的，本轮首次被调度的请求
        scheduled_resumed_reqs: list[Request] = [] # 从waiting中恢复的（之前被抢占），本轮被抢占重新调度的请求
        scheduled_running_reqs: list[Request] = [] # 已经在running中运行的，本轮延续调度的老请求

        preempted_reqs: list[Request] = [] # 从running中踢到waiting中，本轮被抢占的请求

        # 本轮需要新block的请求的表
        req_to_new_blocks: dict[str, KVCacheBlocks] = {}

        # 本轮每个调度的req，在这一轮需要计算的token数量
        num_scheduled_tokens: dict[str, int] = {}

        # 本轮的token预算
        token_budget = self.max_num_scheduled_tokens

        # 如果调度器被设置成暂停，那么就通过把token预算置零来暂停调度。
        if self._pause_state == PauseState.PAUSED_ALL:
            # Do not schedule any requests when paused.
            token_budget = 0

        # 多模态编码器
        # Encoder-related.
        scheduled_encoder_inputs: dict[str, list[int]] = {}
        encoder_compute_budget = self.max_num_encoder_input_tokens

        # 投机解码
        # Spec decode-related.
        # 本轮被调度执行的 request，对应的投机解码（speculative decoding）草稿 token 列表
        scheduled_spec_decode_tokens: dict[str, list[int]] = {} 


        # Whether the running batch contains any prefill requests.
        # 标志位：本轮调度的req是否包含prefill 的req
        prefill_scheduled = False

        # For logging.
        # 创建一个计时器用来计时
        scheduled_timestamp = time.monotonic() 


        # ------【内存池/CuMem】通知 KV cache 管理器开启新步，清空上轮临时分配状态 ------
        # kvcache manager 开始新的一步
        # 不同的attention架构行为不一样：
        # 基础实现 return None, 绝大多数decodr-only模型走这个路径，空操作
        # MambaManager: 走别的方法
        # 通知kv cachemanager, 新的一轮调度器step开始了，清空上一轮临时状态。准备记录这一轮的kvcache分配变化
        self.kv_cache_manager.new_step_starts() 

        # DP prefill balancing: on a throttled (non-cadence-aligned) step, defer
        # all prefill compute unless saturated.

        # defer: 延后执行， defer_prefills = True, 表示这一轮不执行prefill, 
        # DP 下需要defer prefill, 这个后面学习

        # throttle: 表示限制prefill进入
        # ------【DP】DP prefill 均衡：非对齐步延后 prefill 计算，把算力留给 decode ------
        defer_prefills = (
            throttle_prefills and not self.prefill_capacity_bound
        ) and any(not r.is_prefill_chunk for r in self.running)








        ######################################################
        # Phase 1. RUNNING 遍历 (while 循环)
        ######################################################
        
        # First, schedule the RUNNING requests.
        # ------【核心逻辑】Phase1 主循环：遍历 running 队列，逐个分配 KV 并计算调度 token 数 ------
        req_index = 0

        # 只要 还有token预算 && req_index 有效running索引， 这里的req_index，就是running队列的元素指针的作用，用来和while结合，递增查看
        # 所以综合下来的意思就是，只要有token预算，就继续查看running 队列，从0开始查看

        # 从running队列中最老的开始，只要还有token预算
        while req_index < len(self.running) and token_budget > 0:

            # 先拿到我们当前的请求：requset
            request = self.running[req_index]

            # 判断条件1：异步调度的提前终止判断
            # ------【投机解码】草稿全被拒绝也已达 max_tokens，跳过本次调度避免多余 decode 步 ------
            # 如果投机解码的草稿token全部不算，大模型一次采样token就已经达到max_tokens了，就提前终止
            if (
                request.num_output_placeholders > 0 # 有预占位
                # This is (num_computed_tokens + 1) - (num_output_placeholders - 1).
                # Since output placeholders are also included in the computed tokens
                # count, we subtract (num_output_placeholders - 1) to remove any draft
                # tokens, so that we can be sure no further steps are needed even if
                # they are all rejected.
                and request.num_computed_tokens + 2 - request.num_output_placeholders
                >= request.num_prompt_tokens + request.max_tokens
            ):
                # Async scheduling: Avoid scheduling an extra step when we are sure that
                # the previous step has reached request.max_tokens. We don't schedule
                # partial draft tokens since this prevents uniform decode optimizations.
                req_index += 1 # 直接下一个
                continue


            # PP的decode步间约束
            # ------【PP+异步 RPC】强制同请求两次 decode 间隔 pp_size 步，对齐广播槽位环节奏 ------
            if self.current_step < request.next_decode_eligible_step:
                # V2+PP+async: enforce `pp_size` steps between same-req decodes
                # to match worker-side sampled-tokens broadcast slot ring cadence.
                req_index += 1
                continue


            # DP与填充均衡策略
            # ------【DP】均衡策略下，把进行中的 prefill chunk 延后到对齐步，decode 仍继续填满本步 ------
            if defer_prefills and request.is_prefill_chunk:
                # DP prefill balancing: defer this in-progress prefill chunk to a
                # cadence-aligned step; decodes still run to fill this step.
                req_index += 1
                continue



            # 计算num_new_tokens = 本轮要计算的token数量
            num_new_tokens = (
                request.num_tokens_with_spec # 总tokens数量
                + request.num_output_placeholders # 异步调度器占位的部分， 这个先不管
                - request.num_computed_tokens # 已经计算完的tokens数量
            )

            # ------【chunked prefill】超过长 prefill 阈值则按 chunk 截断，避免单请求独占预算 ------
            # 如果这个新的需要计算的tokens数量太长，超过了chunked切分的阈值
            # 既然都超了，那肯定就是prefill的批量填充阶段
            # 如果num_new_tokens = 1， 肯定不会超的，这个就是decode阶段
            if 0 < self.scheduler_config.long_prefill_token_threshold < num_new_tokens:
                num_new_tokens = self.scheduler_config.long_prefill_token_threshold # 强制变更为chunked长度
            num_new_tokens = min(num_new_tokens, token_budget) # chunked长度和我们的token预算取小

            # ------【投机解码】预留每步采样 token 数，防止输入位置超出模型上下文上限 ------
            # Make sure the input position does not exceed the max model len.
            # This is necessary when using spec decoding.
            # 检查一下会不会超出模型的上下文
            num_new_tokens = min(
                num_new_tokens,
                self.max_model_len # 模型上下文，kvcache的最大长度
                - request.num_computed_tokens # 当前已经计算的kvcache的长度
                - self.num_sampled_tokens_per_step, # 投机解码的草稿token的长度
            )


            # 这块是有encoder架构的
            # ------【核心逻辑】有编码器输入时，尝试调度 encoder token 并扣减 encoder 计算预算 ------
            # Schedule encoder inputs.
            encoder_inputs_to_schedule = None
            external_load_encoder_input: list[int] = []
            new_encoder_compute_budget = encoder_compute_budget
            if request.has_encoder_inputs:
                (
                    encoder_inputs_to_schedule,
                    num_new_tokens,
                    new_encoder_compute_budget,
                    external_load_encoder_input,
                ) = self._try_schedule_encoder_inputs(
                    request,
                    request.num_computed_tokens,
                    num_new_tokens,
                    encoder_compute_budget,
                    shift_computed_tokens=1 if self.use_eagle else 0,
                )

            # 这一块是manba架构
            # ------【核心逻辑】Mamba align 模式：把本轮 token 数按 block 对齐切分，避免状态不连续 ------
            if self.need_mamba_block_aligned_split:
                num_new_tokens = self._mamba_block_aligned_split(
                    request, num_new_tokens
                )



            # ------【核心逻辑】本轮无新增 token 可算（PP 未完成/达上限/预算耗尽），跳过该请求 ------
            # 发现没有需要计算的，跳过这个req
            # 一个 request “本轮没有新增token可计算”
            # 不代表请求结束：情况如下：
            #       1. PP > 1, prompt已经调度完成，但是还没完成
            #       2. 达到max_tokens（请求级别的生成长度限制）（req里面会记录，最多生成多少个token长度） 
            #           / max_model_len 达到最大上下文长度了（模型级别）
            #       3. encoder budget耗尽
            #       4. encoder cache耗尽
            #       5. block对齐chunk不足
            if num_new_tokens == 0:
                # The request cannot be scheduled because one of the following
                # reasons:
                # 1. No new tokens to schedule. This may happen when
                #    (1) PP>1 and we have already scheduled all prompt tokens
                #    but they are not finished yet.
                #    (2) Async scheduling and the request has reached to either
                #    its max_total_tokens or max_model_len.
                # 2. The encoder budget is exhausted.
                # 3. The encoder cache is exhausted.
                # 4. Insufficient budget for a block-aligned chunk in hybrid
                #    models with mamba cache mode \"align\".
                # NOTE(woosuk): Here, by doing `continue` instead of `break`,
                # we do not strictly follow the FCFS scheduling policy and
                # allow the lower-priority requests to be scheduled.
                req_index += 1
                continue


            # ------【内存池/CuMem】为请求分配本轮新增的 KV cache block，失败则进入抢占流程 ------
            # 开始分配 block
            # allocate_slots() 分配KVcache
            # Schedule newly needed KV blocks for the request.
            with record_function_or_nullcontext("schedule: allocate_slots"):# 这是个性能分析的trace工具

                #开始为这个req分配block
                while True:
                    # 根据这个req的num_new_tokens数量分配新的blocks
                    new_blocks = self.kv_cache_manager.allocate_slots( # -> KVCacheBlocks = locks: tuple[Sequence[KVCacheBlock], ...]
                        request, # 该请求
                        num_new_tokens, # 该请求要求的总长度
                        num_lookahead_tokens=self.num_lookahead_tokens, # 投机解码的预留token数量
                    )

                    '''
                    模型：Qwen2.5-7B
                    KV cache block size = 16 tokens
                    当前 request 需要新增 40 tokens

                    40 tokens / 16 tokens_per_block
                                                    ≈ 3 blocks

                    new_blocks 可能的长相：
                    new_blocks = KVCacheBlocks(
                                    blocks=(
                                        [
                                            KVCacheBlock(
                                                block_id=123,
                                                ref_count=1,
                                                block_hash=None
                                            ),
                                            KVCacheBlock(
                                                block_id=124,
                                                ref_count=1,
                                                block_hash=None
                                            ),
                                            KVCacheBlock(
                                                block_id=125,
                                                ref_count=1,
                                                block_hash=None
                                            )
                                        ]
                                    )
                                )
                    '''

                    # 确实申请到显存了，直接退出申请block的循环，显存block申请成功 
                    if new_blocks is not None:
                        # The request can be scheduled.
                        break


                    # 如果执行到这里，说明block申请失败了
                    # ------【核心逻辑】显存不足触发抢占：按策略选出 victim 释放其 KV，腾出 block 给当前请求 ------
                    # 就是显存不足了，需要抢占低优先级的req，让他滚到waiting队列，释放掉他的显存kvcache

                    # 这边的逻辑是在running队列中选出一个victim，可以是本req的前面，后面，自己
                    if self.policy == SchedulingPolicy.PRIORITY: #如果我们的调度策略是优先级调度
                        preempted_req = max(
                            self.running,
                            key=lambda r: (r.priority, r.arrival_time), # 优先级最低、到达时间最晚的 request
                        )
                        self.running.remove(preempted_req) # 在running队列中找出这个victim的请求，因为只有running队列上的req才真正占有block

                        # 如果选中的victim是：
                        # 1. 本req前面的，
                        # 2， 且已经被认定为本轮调度名单里面的req
                        # 
                        # 清理本轮的调度名单
                        if preempted_req in scheduled_running_reqs: # 如果这个victim，恰好是本轮要被调度的req
                            preempted_req_id = preempted_req.request_id # 记录下这个victim的req_id
                            scheduled_running_reqs.remove(preempted_req) # 把他移除出本轮的调度列表
                            token_budget += num_scheduled_tokens.pop(preempted_req_id)# 收回这个victim的token预算
                            req_to_new_blocks.pop(preempted_req_id) # 删除掉这个这个victim的req占用的新的block

                            scheduled_spec_decode_tokens.pop(preempted_req_id, None)# victim如果有投机解码的token要求，也取消掉

                            preempted_encoder_inputs = scheduled_encoder_inputs.pop(# victim的多模态的编码器也被除掉了
                                preempted_req_id, None
                            )

                            if preempted_encoder_inputs:
                                # Restore encoder compute budget if the preempted
                                # request had encoder inputs scheduled in this step.
                                num_embeds_to_restore = sum(
                                    preempted_req.get_num_encoder_embeds(i)
                                    for i in preempted_encoder_inputs
                                )
                                encoder_compute_budget += num_embeds_to_restore

                            
                            req_index -= 1 # 因为排在该req前面的请求被清除掉了，所以index 都统一缩小了1
                    else:
                        preempted_req = self.running.pop()


                    # 针对这个victim，执行抢占操作的后处理
                    # ------【核心逻辑】对被抢占请求收尾：释放 block、标记 PREEMPTED、重新入 waiting ------
                    self._preempt_request(
                        preempted_req,
                        scheduled_timestamp,
                        drop_stale_output=self.requires_kv_delivery,
                    )

                    # 本轮被抢占的请求，记录下这个victim
                    preempted_reqs.append(preempted_req)

                    # 如果碰巧这个victim就是自己，这个req就不申请block了，在外面跳出本轮检查
                    if preempted_req == request:
                        # No more request to preempt. Cannot schedule this request.
                        break


            # 如果显存不够了，这个req不能调度
            if new_blocks is None:
                # Cannot schedule this request.
                break


            ####################################################################
            # 调度这个req， 有num_new_tokens, 也有block
            ####################################################################

            # ------【核心逻辑】确认可调度：登记新 block 页表与 token 数，扣减 token 预算 ------
            # 下面开始真正调度这个req，他有本次的任务token数量 = num_new_tokens， 且已经分配好了KV cache block
            # Schedule the request.
            scheduled_running_reqs.append(request) # 加入本轮调度的名单，是原本就在running队列里面的
            prefill_scheduled |= request.is_prefill_chunk # 标志位：这个请求是否是chunked prefill

            request_id = request.request_id # 该req id记录下
            req_to_new_blocks[request_id] = new_blocks # 记录下这个req的新增的block页表
            num_scheduled_tokens[request_id] = num_new_tokens # 记录下这个req需要计算的tokens数量

            token_budget -= num_new_tokens # 更新token预算剩余
            req_index += 1 # 这个req就算排查结束，下一个



            # Speculative decode related.
            # ------【投机解码】截取本轮可验证的草稿 token 数，登记到草稿验证名单，然后清空待重填 ------
            # 如果这个req，说明本轮的执行器的工作是需要进行草稿tokens的验证
            if request.spec_token_ids:  # 这里的request.spec_token_ids，表示draft model实际生成的草稿token的列表
                num_scheduled_spec_tokens = ( # 计算本轮调度 需要要验证的草稿token数量（一轮调度可能验证不完所有的草稿token列表）
                    num_new_tokens
                    + request.num_computed_tokens
                    - request.num_tokens
                    - request.num_output_placeholders # 异步调度的属性，同步调度的placeholder占位=0
                )

                if num_scheduled_spec_tokens > 0:# 如果本轮调度需要验证草稿token
                    spec_token_ids = request.spec_token_ids # 获取draft model的所有草稿token列表
                    if len(spec_token_ids) > num_scheduled_spec_tokens: # 确实足够
                        spec_token_ids = spec_token_ids[:num_scheduled_spec_tokens] # 切片列表
                    scheduled_spec_decode_tokens[request.request_id] = spec_token_ids # 加入本轮的调度名单：草稿token验证名单

                # New spec tokens will be set in `update_draft_token_ids` before the
                # next step when applicable.

                # draft token 不是一个可以无限排队消费的 token 队列，而是一组基于当前上下文生成的临时预测结果（proposal）。
                # 所以没有被取出的旧 draft token，大概率没有保留价值。
                # vLLM 设计上认为：这一批 draft token 是“一次性验证任务”，没有必要跨 step 保留
                # 清空这个请求的草稿token列表，
                # 新的 draft token 会在下一次 step 开始前，由 update_draft_token_ids() 重新填充
                request.spec_token_ids = []  


            # Encoder-related.
            if encoder_inputs_to_schedule:
                scheduled_encoder_inputs[request_id] = encoder_inputs_to_schedule
                # Allocate the encoder cache.
                for i in encoder_inputs_to_schedule:
                    self.encoder_cache_manager.allocate(request, i)
                    if self.ec_connector is not None:
                        self.ec_connector.update_state_after_alloc(request, i)
                encoder_compute_budget = new_encoder_compute_budget

            if external_load_encoder_input:
                for i in external_load_encoder_input:
                    self.encoder_cache_manager.allocate(request, i)
                    if self.ec_connector is not None:
                        self.ec_connector.update_state_after_alloc(request, i)


        # 就这样，更新检查完所有的running队列的每一个请求


        # Phase2: LoRA 统计
        # Record the LoRAs in scheduled_running_reqs
        # ------【LoRA】统计本轮 running 请求需要的不同 LoRA adapter，供后续校验并发上限 ------
        # 统计本轮调度的running请求里面，一共有多少不同的LoRA adapter 需要加载到GPU上
        scheduled_loras: set[int] = set()
        if self.lora_config:
            scheduled_loras = set(
                req.lora_request.lora_int_id
                for req in scheduled_running_reqs # running队列中在本轮调度名单里面的
                if req.lora_request and req.lora_request.lora_int_id > 0
            )
            assert len(scheduled_loras) <= self.lora_config.max_loras







        ###################################################

        # Phase3: WAITTING 遍历
        ###################################################
        # Next, schedule the WAITING requests.
        # ------【核心逻辑】Phase3：无抢占且未暂停时，从 waiting 队列接纳新/恢复请求 ------
        # waitting中没有上次被抢占的，且调度器正常
        # 本轮被抢占的不再重新调度
        if not preempted_reqs and self._pause_state == PauseState.UNPAUSED:

            # 新创建一个临时请求队列
            # 它不是替代 Scheduler 里面长期维护的 self.skipped_waiting 队列，而是本轮 step 使用的临时队列。
            step_skipped_waiting = create_request_queue(self.policy) 


            # 只要还有备选的= 就绪队列 + 阻塞队列，且token预算还有剩余
            while (self.waiting or self.skipped_waiting) and token_budget > 0:



                # Paused streaming sessions (WAITING_FOR_STREAMING_REQ) are not
                # in `running` but still hold a model-runner request slot.

                # num_runing 表示 当前已经占用模型执行槽位的request数量
                # 现在的running队列里面留下的，都是已经进入scheduled_running_req的本轮调度名单里面的。
                # num_waiting_for_streaming_input统计的requset，不在任何队列，处于独立状态，不用深入
                num_running = len(self.running) + self.num_waiting_for_streaming_input # 流式请求的计数器，正在等待流式输入

                if num_running >= self.max_num_running_reqs:# 当前的调度名单，已经足够多了，不在检查waiting队列了
                    break





                # 根据策略，看看我们是从waiting队列拿，还是在阻塞队列拿，只要循环还在，最终可以检查完两个队列
                request_queue = self._select_waiting_queue_for_scheduling() # 优先返回skip阻塞队列，为空就返回waiting队列
                assert request_queue is not None

                # 只取不删，拿到候选队列里面的第一个看看，因为要先检查
                request = request_queue.peek_request() # 优先级队列的方法，返回第一个
                request_id = request.request_id





                # try to promote blocked statuses while traversing skipped queue.
                # ------【异步 RPC】尝试把阻塞中的请求恢复为可调度，恢复不了则临时跳过 ------
                # 判断这个request是否处于阻塞状态，不是真的查询判断，而是先通过状态判断来过滤一波
                # 如果是，尝试把它恢复成正常 waiting， 如果恢复不了，就跳过它
                if self._is_blocked_waiting_status(
                    request.status
                ) and not self._try_promote_blocked_waiting_request(request): # 尝试恢复/提升, 从特殊阻塞状态恢复成可调度状态

                    # 进到这里，说明这个req是阻塞，且，无法恢复可调度状态
                    if request.status == RequestStatus.WAITING_FOR_REMOTE_KVS: # 这个就是具体的阻塞状态
                        logger.debug(
                            "%s is still in WAITING_FOR_REMOTE_KVS state.",
                            request_id,
                        )
                    request_queue.pop_request()
                    step_skipped_waiting.prepend_request(request) # 把本轮确认无法调度的req，暂时保存起来，避免重复检查
                    continue


                '''
                走到这里，说明这个req的状态已经是可调度的waiting了
                '''

                # 异步调度的判断，不用关心
                if (
                    request.num_stale_output_tokens > 0
                    and not request.drop_stale_output
                ):
                    # Deliverable stale output still in flight: resuming now
                    # could resample a position that output later delivers.
                    # It drains within the pipeline depth.
                    request_queue.pop_request()
                    step_skipped_waiting.prepend_request(request)
                    continue


                # 确认这个request,本轮 batch 如果加入这个 request，会不会超过同时支持的 LoRA adapter 数量限制
                if (
                    self.lora_config
                    and request.lora_request
                    and (
                        len(scheduled_loras) == self.lora_config.max_loras
                        and request.lora_request.lora_int_id not in scheduled_loras
                    )
                ):
                    # Scheduling would exceed max_loras, skip.
                    request_queue.pop_request()
                    step_skipped_waiting.prepend_request(request)
                    continue








                num_external_computed_tokens = 0 # 这个 request 有多少 token 的 KV 已经在外部算好了，不需要本机重新计算。PD分离
                load_kv_async = False # 是否异步加载远端 KV cache。
                connector_prefix_cache_queries, connector_prefix_cache_hits = 0, 0 #KV Connector 场景下 prefix cache 的统计量
                did_prefix_cache_lookup = False # 这个 request 本轮到底有没有做过 prefix cache lookup

                # prefix cache
                # ------【前缀缓存】首次调度时查询可复用的前缀 KV（本地/远端），命中即可跳过已算 token ------
                # Get already-cached tokens. 获得已经缓存的tokens
                if request.num_computed_tokens == 0: # 如果这个请求req, 还没有被计算过kv cache
                    did_prefix_cache_lookup = True # 如果这个 request 从来没有执行过 prefill，那么第一次调度它时，需要尝试寻找可以复用的 prefix KV。
                    hit_diverged = False # prefix cache 命中过程中，是否出现了“前缀匹配分叉（divergence）”
                    # hit表示找到了一部分可以服用的prefix kv， diverged：后面的 token 和缓存前缀不一致，不能继续往后复用


                    # Get locally-cached tokens. 开始获得已有的缓存

                    if self.connector is not None: # 有KV connector时的KV查询（设计远程KV，PD分离）
                        # A KV connector transfers the missing suffix, which needs a
                        # hybrid-aware lookup that can diverge across groups.
                        (
                            new_computed_blocks,
                            num_new_local_computed_tokens,
                            request.shared_prefix_boundary,
                            hit_diverged,
                        ) = self.kv_cache_manager.get_computed_blocks_for_connector( # 这个方法支持本地+外部 混合查询
                            request
                        )
                    else: # 普通本地KV cache查询，就是在vllm实例自己的kv cache里面查找前缀缓存
                        (
                            new_computed_blocks, # 新发现的，prefix 命中的 block列表
                            num_new_local_computed_tokens, # 新发现的，已经命中的kv cache的token数量
                            # Marconi shared-prefix junction to pin; 0 if none.
                            request.shared_prefix_boundary, # 多个request之间共享的前缀边界位置
                        ) = self.kv_cache_manager.get_computed_blocks(request) # 分配kv cache block给这个请求



                    # kv connector下匹配和加载远端的kv cache
                    # ------【PD 分离】对比本地与远端命中，取更长者覆盖 sub-block 尾，决定异步加载的 token 数 ------
                    # 它不是单纯“加载远端 KV cache”，而是在 本地 prefix cache 查询结果的基础上，再询问远端 KV 是否有更长的匹配前缀，
                    # 然后决定采用本地 KV、远端 KV，还是两者组合。
                    # Get externally-cached tokens if using a KVConnector.
                    if self.connector is not None:
                        # Present a block-aligned local hit to the connector so
                        # a strictly longer remote hit can supersede a local
                        # sub-block tail without racing its copy-on-write.
                        partial_tail = num_new_local_computed_tokens % self.block_size
                        block_aligned_local = (
                            num_new_local_computed_tokens - partial_tail
                        )
                        ext_tokens, load_kv_async = (
                            self.connector.get_num_new_matched_tokens(
                                request, block_aligned_local
                            )
                        )

                        if ext_tokens is None:
                            # The request cannot be scheduled because
                            # the KVConnector couldn't determine
                            # the number of matched tokens.
                            request_queue.pop_request()
                            step_skipped_waiting.prepend_request(request)
                            continue

                        if partial_tail and ext_tokens > partial_tail:
                            # Remote strictly exceeds the full local hit: drop the
                            # sub-block tail so no CoW is needed, and let the load
                            # cover it. Trim the partial block out of the local
                            # computed blocks so it is not adopted from the cache.
                            new_computed_blocks = (
                                self.kv_cache_manager.truncate_computed_blocks(
                                    new_computed_blocks, block_aligned_local
                                )
                            )
                            num_new_local_computed_tokens = block_aligned_local
                            num_external_computed_tokens = ext_tokens
                        elif partial_tail:
                            # Remote does not exceed the full local hit: keep the
                            # local sub-block tail and load nothing external.
                            num_external_computed_tokens = 0
                            # Nothing to load remotely -> not an async-load step;
                            # clearing avoids the `load_kv_async` assert below.
                            load_kv_async = False
                        else:
                            num_external_computed_tokens = ext_tokens

                        if hit_diverged and num_external_computed_tokens == 0:
                            # No external tokens back the deeper local hit, so its
                            # resume boundary would have no valid Mamba state.
                            # Reconcile to the boundary every group agrees on.
                            (
                                new_computed_blocks,
                                num_new_local_computed_tokens,
                                request.shared_prefix_boundary,
                            ) = self.kv_cache_manager.get_computed_blocks(request)

                        connector_prefix_cache_queries = (
                            request.num_tokens - num_new_local_computed_tokens
                        )
                        connector_prefix_cache_hits = num_external_computed_tokens


                    '''
                    这边有一个远端和本地命中的block的一致性的问题
                    local 和 remote 命中的是同一个 prefix 空间，它们可以组合，但最终必须形成连续、无冲突的最大可复用 prefix。

                    对于完整 block 可以直接拼接；对于partial block，需要让更长的一侧覆盖，避免 copy-on-write 和一致性问题。

                    这里的一致性问题得说一下：

                    我们prefix cache的匹配逻辑，是token级别的，所以

                        假设block1 里面是EFXY block 1, 里面是E，reqA命中，reqB本地命中E，远端命中EFGH，reqC本地命中EFX。 
                        如果reqB 远端命中 就直接把远端的EFGH直接覆盖block1的EFXY，虽然reqA 不受影响，但是reqC却被篡改了cache

                    '''


                    # Total computed tokens (local + external).
                    # ------【前缀缓存+PD 分离】汇总本地+远端命中的已算 token 数，作为后续调度基准 ------
                    num_computed_tokens = ( # 已经命中的cache 的 token数量
                        num_new_local_computed_tokens + num_external_computed_tokens
                    )
                    assert num_computed_tokens <= request.num_tokens

                    # 多模态部分
                    # Skip request with pending mm encoding prefetches
                    if (
                        self.ec_connector is not None
                        and request.mm_features
                        and not self.ec_connector.ensure_cache_available(
                            request, num_computed_tokens
                        )
                    ):
                        request_queue.pop_request()
                        step_skipped_waiting.prepend_request(request)
                        continue


                    # 记录prefill阶段的统计信息，主要用于性能分析（prefill latency、prefix cache命中情况等）
                    # ------【前缀缓存】首次 prefill 记录 prompt 与本地/远端缓存命中 token 数，供延迟与命中率统计 ------
                    # Track first scheduled prefill, not post-preemption repeat prefills
                    if request.prefill_stats and request.num_preemptions <= 0:
                        assert num_computed_tokens <= request.num_prompt_tokens
                        request.prefill_stats.set(
                            num_prompt_tokens=request.num_prompt_tokens,
                            num_local_cached_tokens=num_new_local_computed_tokens,
                            num_external_cached_tokens=num_external_computed_tokens,
                        )

                # 这个req已经被计算过kv cache， 所以肯定不是第一次被调度
                else:
                    # KVTransfer: WAITING reqs have num_computed_tokens > 0
                    # after async KV recvs are completed.
                    new_computed_blocks = self.kv_cache_manager.empty_kv_cache_blocks # 本地命中的cache block
                    num_new_local_computed_tokens = 0
                    num_computed_tokens = request.num_computed_tokens


                # 查询kv cache block 完毕
                ##################



                # 加载kv cache block
                encoder_inputs_to_schedule = None
                external_load_encoder_input = []
                new_encoder_compute_budget = encoder_compute_budget
                pad_spec_decode = False

                # ------【PD 分离+异步 RPC】异步加载远端 KV：本步不分配新 token，仅等待传输完成 ------
                if load_kv_async:
                    # KVTransfer: loading remote KV, do not allocate for new work.
                    assert num_external_computed_tokens > 0
                    num_new_tokens = 0
                elif defer_prefills and num_computed_tokens < request.num_tokens - 1:
                    # DP prefill balancing: defer this step's local prefill
                    # compute to a cadence-aligned step.
                    break
                else:

                    # 普通，走这里，计算本轮这个req需要计算的token数量
                    # ------【核心逻辑】按已算 token 差计算本步新增 token 数（含恢复请求的输出 token） ------
                    # Number of tokens to be scheduled.
                    # We use `request.num_tokens` instead of
                    # `request.num_prompt_tokens` to consider the resumed
                    # requests, which have output tokens.
                    # 他们使用num_tokens 而不是 num_prompt_tokens， 来考虑包括被恢复的请求，因为他们已经有输出token了
                    num_new_tokens = request.num_tokens - num_computed_tokens # 计算这个request（从未调度过，曾经调度过，被抢占了）的本次调度需要计算的tokens


                    # 【投机解码 + cuda graph shape对齐优化】
                    '''
                    解释一下，不同req的投机解码，导致计算的shape不一样，比如reqA, 草稿tokens=4, reqB,不使用投机解码，num_new_tokens=1
                    这会导致GPU完成模型一次前向推理的输入张量形状不一样

                    对于cuda graph
                    希望固定shape: batch_size固定（请求数量）， 序列长度固定（token数量）

                    所以padding填充，就是把一个请求不足的num_new_tokens长度，填充成一样的长度，这样序列长度就固定了，shape自然固定了

                    如果这个req发现num_new_tokens > token_budget，或者，num_computed_tokens + num_new_tokens > max_model_len
                    我们就宁愿不带上这个req，避免他破坏cuda 
                    

                    另外cuda graph，batch_size, num_new_tokens都要一致，
                    整个 CUDA kernel launch 过程中涉及的 tensor shape、内存地址布局、batch size 都必须固定

                    所以，针对batch_size,也就是一个batch，req数量不定的问题，vllm的解决做法是：多个graph实例

                                                    graph(batch_size=1)
                                                    graph(batch_size=2)
                                                    graph(batch_size=4)
                                                    graph(batch_size=8)
                                                    ...
                    这样启动就能多个情况的capture了
                    '''
                    # Pad new decode requests to uniform spec decoding size to
                    # preserve full cudagraph for this step.
                    # Not for diffusion where draft tokens can't be padded.
                    if (
                        (self.num_spec_tokens > 0 and self.dynamic_sd_lookup is None)
                        and self.num_sampled_tokens_per_step > 0
                        and num_new_tokens == 1
                        and (scheduled_running_reqs and not prefill_scheduled)
                    ):
                        num_new_tokens = 1 + self.num_spec_tokens
                        if (
                            num_new_tokens > token_budget
                            or num_computed_tokens + num_new_tokens > self.max_model_len
                        ):
                            # Prefer to not schedule than schedule un-padded here.
                            break
                        pad_spec_decode = True





                    # 判断这次新计算kvcache的新长度，有没有超出chunked的大小
                    threshold = self.scheduler_config.long_prefill_token_threshold
                    if 0 < threshold < num_new_tokens:
                        num_new_tokens = threshold

                    # 【跳过】分块预填充功能必须明确启用，以允许 
                    # 池化请求进行分块处理
                    if (
                        not self.scheduler_config.enable_chunked_prefill
                        and num_new_tokens > token_budget
                    ):
                        # If chunked_prefill is disabled,
                        # we can stop the scheduling here.
                        break



                    num_new_tokens = min(num_new_tokens, token_budget) # 再看看预算够不够
                    assert num_new_tokens > 0

                    # 多模态
                    # Schedule encoder inputs.
                    if request.has_encoder_inputs:
                        (
                            encoder_inputs_to_schedule,
                            num_new_tokens,
                            new_encoder_compute_budget,
                            external_load_encoder_input,
                        ) = self._try_schedule_encoder_inputs(
                            request,
                            num_computed_tokens,
                            num_new_tokens,
                            encoder_compute_budget,
                            shift_computed_tokens=1 if self.use_eagle else 0,
                        )
                        if num_new_tokens == 0:
                            # The request cannot be scheduled.
                            break




                # 【跳过】Mamba架构-Mamba block alignment
                # Skip block alignment when setting up async receive (no local work).
                if self.need_mamba_block_aligned_split and not load_kv_async:
                    num_new_tokens = self._mamba_block_aligned_split(
                        request,
                        num_new_tokens,
                        num_new_local_computed_tokens,
                        num_external_computed_tokens,
                    )
                    if num_new_tokens == 0:
                        break


                # 【跳过】async KV load 下限制 lookahead，属于KV Connector / P-D 分离 / 异步KV加载
                # During async KV load, no forward pass is run yet.
                # Allocate speculative lookahead slots later to avoid
                # mismatching local and remote block counts.
                limit_lookahead_tokens = load_kv_async and self.num_lookahead_tokens > 0
                effective_lookahead_tokens = (
                    0 if limit_lookahead_tokens else self.num_lookahead_tokens
                )


                # 【跳过】cross-attention blocks， 属于Encoder-Decoder模型
                # Determine if we need to allocate cross-attention blocks.
                num_encoder_tokens = 0
                if (
                    self.is_encoder_decoder
                    and request.has_encoder_inputs
                    and encoder_inputs_to_schedule
                ):
                    num_encoder_tokens = sum(
                        request.get_num_encoder_embeds(i)
                        for i in encoder_inputs_to_schedule
                    )

                # 【跳过】async KV reserve blocks， 远程KV加载期间的显存预留管理。
                reserved_blocks = 0
                if load_kv_async:
                    # An async load holds its blocks for the whole transfer with
                    # no forward progress and isn't preemptible here. Admit it
                    # only if it fits in (free - other in-flight reservations), to
                    # avoid deadlock and predictable preemptions.
                    reserved_blocks = self._inflight_prefill_reserved_blocks()




                # ------【内存池/CuMem】为 waiting 请求分配本轮 KV block，传入前缀命中/远端 token/lookahead 等上下文 ------
                # 开始为num_new_tokens分配blocks
                new_blocks = self.kv_cache_manager.allocate_slots(
                    request,
                    num_new_tokens, # 本轮要执行 forward 的新 token 数量（需要新增 KV）
                    num_new_computed_tokens=num_new_local_computed_tokens, # 本地 prefix cache 已经命中的 token 数量
                    new_computed_blocks=new_computed_blocks, # prefix cache 命中的已有 KV block，需要挂载给 request
                    num_lookahead_tokens=effective_lookahead_tokens, # 投机解码预留的额外 KV slot 数
                    num_external_computed_tokens=num_external_computed_tokens, # 远端 KV cache 已经计算好的 token 数（KVConnector/P-D分离）
                    delay_cache_blocks=load_kv_async, # 是否延迟真正加入 KV cache（异步远端 KV 加载时使用）
                    num_encoder_tokens=num_encoder_tokens, # encoder-decoder / 多模态 cross attention 需要的额外 token 数
                    full_sequence_must_fit=self.scheduler_reserve_full_isl, # 是否要求整个序列一次性放入显存
                    reserved_blocks=reserved_blocks, # 已经被异步 KV transfer 占用/预留的 block 数
                    has_scheduled_reqs=bool(self.running), # 当前是否已经有 running 请求，用于一些调度策略判断
                )

                # 如果显存不够了，waiting队列就不在继续往下，直接break结束
                if new_blocks is None:
                    # The request cannot be scheduled.

                    # NOTE: we need to untouch the request from the encode cache
                    # manager
                    if request.has_encoder_inputs:
                        self.encoder_cache_manager.free(request)
                    break



                # 【跳过】KVConnector / P-D 分离路径
                # KVTransfer: the connector uses this info to determine
                # if a load is needed. Note that
                # This information is used to determine if a load is
                # needed for this request.
                if self.connector is not None:
                    self.connector.update_state_after_alloc(
                        request,
                        self.kv_cache_manager.get_blocks(request_id),
                        num_external_computed_tokens,
                    )
                    if (
                        self.connector_prefix_cache_stats is not None
                        and connector_prefix_cache_queries != 0
                    ):
                        self.connector_prefix_cache_stats.record(
                            num_tokens=connector_prefix_cache_queries,
                            num_hits=connector_prefix_cache_hits,
                            preempted=request.num_preemptions > 0,
                        )

                # 本地 Prefix Cache 命中统计
                # ------【前缀缓存】在成功接纳时记录命中统计，避免把未调度的查询也计入 ------
                # Record at admission so unscheduled lookups are not counted.
                if did_prefix_cache_lookup:
                    self.kv_cache_manager.record_prefix_cache_stats(
                        request, num_new_local_computed_tokens
                    )

                # 显存也够，num_new_tokens也知道了，就把这个req从候选的队列里面弹出来
                request = request_queue.pop_request()


                # 它不是“重新加入候选队列等待重新调度”，而是把这个 request 暂时放入阻塞状态，等待异步 KV transfer 完成后再恢复调度资格。
                '''
                                waiting/skipped_waiting
                                        |
                                        |
                                        v
                                本轮scheduler尝试调度
                                        |
                                        v
                                发现需要异步加载remote KV
                                        |
                                        v
                                进入 WAITING_FOR_REMOTE_KVS
                                        |
                                        v
                                等待KV传输完成
                                        |
                                        v
                                恢复成 WAITING
                                        |
                                        v
                                下一轮schedule重新调度
                '''
                # ------【PD 分离+异步 RPC】异步加载路径：置 WAITING_FOR_REMOTE_KVS 状态，等待远端 KV 传输完成 ------
                if load_kv_async: # 异步加载以前的kv cache
                    # If loading async, allocate memory and put request
                    # into the WAITING_FOR_REMOTE_KV state.
                    request.status = RequestStatus.WAITING_FOR_REMOTE_KVS
                    step_skipped_waiting.prepend_request(request)
                    # Set num_computed_tokens even though KVs are not yet loaded.
                    # request.num_computed_tokens will not be used anywhere until
                    # the request finished the KV transfer.
                    #
                    # If a transfer error is reported by the connector,
                    # request.num_computed_tokens will be re-set accordingly in
                    # _update_requests_with_invalid_blocks.
                    #
                    # When the transfer is finished, either successfully or not,
                    # request.num_computed_tokens will correctly reflect the number
                    # of computed tokens.
                    # _update_waiting_for_remote_kv will then cache
                    # only the successfully loaded tokens.
                    request.num_computed_tokens = num_computed_tokens
                    self._inflight_prefills.add(request)

                    # 这里是kvconnector异步加载远端kv的特殊保护逻辑，不能让本地的kv cache 清零初始化和远端kv写入发生竞争
                    if self.needs_kv_cache_zeroing:
                        # Skip zeroing of the blocks the async load will
                        # overwrite; the zeroing could race the write.
                        self._skip_zero_block_ids.update(
                            self.kv_cache_manager.get_zeroing_block_ids_in_range(
                                request.request_id,
                                num_new_local_computed_tokens,
                                num_computed_tokens,
                            )
                        )
                    continue








                # ------【核心逻辑】请求成功调度：加入 running 队列并按状态分类到新增/恢复名单 ------
                # 把这个req加入RUNNING队列，调度成功！！！！
                self.running.append(request)


                if self.log_stats:
                    request.record_event(
                        EngineCoreEventType.SCHEDULED, scheduled_timestamp
                    )
                if request.status == RequestStatus.WAITING:
                    scheduled_new_reqs.append(request) # 这里保存一下本次的调度内容，方便一起发出
                elif request.status == RequestStatus.PREEMPTED:
                    scheduled_resumed_reqs.append(request) # 这里保存一下本次的调度内容，方便一起发出
                else:
                    raise RuntimeError(f"Invalid request status: {request.status}")


                # 检查lora资源是否就位
                if self.lora_config and request.lora_request:
                    scheduled_loras.add(request.lora_request.lora_int_id)

                # 更新这个req的页表
                req_to_new_blocks[request_id] = self.kv_cache_manager.get_blocks(
                    request_id
                )
                # 更新本地调度的这个req的token数
                num_scheduled_tokens[request_id] = num_new_tokens

                # 扣除预算token
                token_budget -= num_new_tokens

                # 这个req的状态更新成运行态
                request.status = RequestStatus.RUNNING

                # 前面的 num_computed_tokens 是 Scheduler 本轮计算出来的临时变量，不一定已经同步回 request 对象
                # 如果这个req之前调度过，那这两个就是相等的，如果req之前没有运行过，是prefix cache命中得到的，那么这个就是需要重新更新
                request.num_computed_tokens = num_computed_tokens # 更新一下这个req已经计算好的token计数


                # 填充投机解码的固定位置
                # ------【投机解码+CUDA Graph】用 -1 占位补齐草稿长度，保持本步 batch 形状固定 ------
                if pad_spec_decode:
                    scheduled_spec_decode_tokens[request_id] = [
                        -1
                    ] * self.num_spec_tokens

                # 【trace】如果本次有被chunked截断后，仍然处于prefill阶段的req，加入一个prefill监听队列
                # Only track requests that will still be prefilling after this chunk.
                if num_computed_tokens + num_new_tokens < request.num_tokens:
                    self._inflight_prefills.add(request)


                # Encoder-related.
                if encoder_inputs_to_schedule:
                    scheduled_encoder_inputs[request_id] = encoder_inputs_to_schedule
                    # Allocate the encoder cache.
                    for i in encoder_inputs_to_schedule:
                        self.encoder_cache_manager.allocate(request, i)
                        if self.ec_connector is not None:
                            self.ec_connector.update_state_after_alloc(request, i)
                    encoder_compute_budget = new_encoder_compute_budget
                # Allocate for external load encoder cache
                if external_load_encoder_input:
                    for i in external_load_encoder_input:
                        self.encoder_cache_manager.allocate(request, i)
                        if self.ec_connector is not None:
                            self.ec_connector.update_state_after_alloc(request, i)




            # 循环结束，下面是善后工作


            # re-queue requests skipped in this pass ahead of older skipped items.
            # 把之前剔除掉的 阻塞状态无法恢复的req  重新加入回去
            if step_skipped_waiting:
                self.skipped_waiting.prepend_requests(step_skipped_waiting)

            # ------【DP】记录本轮是否因容量饱和而停止接纳 prefill，供下步均衡决策 ------
            # DP prefill balancing: on a step that admitted prefills (release),
            # record whether it was capacity-bound.
            if not defer_prefills:
                self.prefill_capacity_bound = bool(self.waiting)


        ##################### 至此，WAITTING队列也检查完了



        # 下面要根据我们的本轮调度名单来发送了，最后检查一遍



        # 检查调度限制是否都满足
        # ------【核心逻辑】校验总调度 token、预算与 running 数量不超过硬约束 ------
        # Check if the scheduling constraints are satisfied.
        total_num_scheduled_tokens = sum(num_scheduled_tokens.values())
        assert total_num_scheduled_tokens <= self.max_num_scheduled_tokens

        assert token_budget >= 0
        assert len(self.running) <= self.max_num_running_reqs
        # Since some requests in the RUNNING queue may not be scheduled in
        # this step, the total number of scheduled requests can be smaller than
        # len(self.running).
        assert len(scheduled_new_reqs) + len(scheduled_resumed_reqs) + len(
            scheduled_running_reqs
        ) <= len(self.running)



        # Get the longest common prefix among all requests in the running queue.
        # ------【前缀缓存】计算 running 队列最长公共前缀 block 数，供 cascade attention 复用 ------
        # This can be potentially used for cascade attention.
        num_common_prefix_blocks = [0] * len(self.kv_cache_config.kv_cache_groups)
        with record_function_or_nullcontext("schedule: get_num_common_prefix_blocks"):
            if self.running:
                any_request_id = self.running[0].request_id
                num_common_prefix_blocks = (
                    self.kv_cache_manager.get_num_common_prefix_blocks(any_request_id)
                )







        #########################################################
            # Phase4: 构造调度器输出    
        #########################################################
        # Construct the scheduler output.


        # new_reqs_data: 本轮第一次进入ModelRunner执行的req
        # ------【核心逻辑】Phase4：构造新进入 ModelRunner 请求的数据（v2 合并恢复请求） ------
        if self.use_v2_model_runner: #如果是v2的model_runner
            scheduled_new_reqs.extend(scheduled_resumed_reqs) # 合并本轮的恢复队列
            scheduled_resumed_reqs.clear()
            new_reqs_data = [
                NewRequestData.from_request(
                    req,
                    req_to_new_blocks[req.request_id].get_block_ids(),
                    req._all_token_ids,
                )
                for req in scheduled_new_reqs
            ]
        else: # v1的model_runner
            new_reqs_data = [
                NewRequestData.from_request(
                    req, req_to_new_blocks[req.request_id].get_block_ids()
                )
                for req in scheduled_new_reqs
            ]

        # cached_reqs_data：已经在 ModelRunner 中存在，本轮继续执行的 request 信息
        # ------【核心逻辑】构造已在 runner 中、本轮续算请求的输入数据（token/block/草稿） ------
        with record_function_or_nullcontext("schedule: make_cached_request_data"):
            cached_reqs_data = self._make_cached_request_data(
                scheduled_running_reqs,
                scheduled_resumed_reqs,
                num_scheduled_tokens,
                scheduled_spec_decode_tokens,
                req_to_new_blocks,
            )


        # v1 modelrunner 的调度历史记录，记录“上一轮 scheduler 实际调度过哪些 request”，供下一轮 v1 ModelRunner 使用。
        # Record the request ids that were scheduled in this step (MRV1-only).
        if not self.use_v2_model_runner:
            self.prev_step_scheduled_req_ids.clear()
            self.prev_step_scheduled_req_ids.update(num_scheduled_tokens.keys())



        # 【跳过】kvconnector /PD分离 producer端的特殊逻辑
        # Producer partial-tail hand-off for external KV connectors. Drained
        # before the CoW retentions are released below, so the pin lands while
        # the cow block still holds a retention ref. Without a producer-side
        # connector nothing consumes the hand-off, so skip the drain (and its
        # pin); the manager drops stale entries when the request's blocks are
        # popped for free.
        pending_partial_tail_offloads = None
        if (
            self.connector is not None
            and self.vllm_config.kv_transfer_config is not None
            and self.vllm_config.kv_transfer_config.is_kv_producer
        ):
            pending_partial_tail_offloads = (
                self.kv_cache_manager.take_partial_tail_offloads() or None
            )




        # 某些 KV block 需要复制（copy），但是复制完成之前，旧 block 不能释放
        # ------【前缀缓存】取出 copy-on-write 复制任务与被保留旧 block，配合延迟释放保证正确性 ------
        kv_cache_block_copies, cow_retained_blocks = (
            self.kv_cache_manager.take_kv_cache_block_copies()
        )

        if kv_cache_block_copies:
            # The copies run with this step's execution; the first non-empty
            # step at or after it gets seq `sched_step_seq + 1` (0-token steps
            # do not advance the seq), and its completion implies the copies
            # have run.
            self._free_cow_retained_blocks(cow_retained_blocks, self.sched_step_seq + 1)
        pending_kv_cache_block_copies = kv_cache_block_copies or None


        # 动态投机解码，解决的问题是：固定的draft token数量 K 不一定适合所有batch_size
        # ------【投机解码】按 batch_size 查表得到动态草稿 token 数 K，替代固定值 ------
        # Dynamic speculative decoding: compute optimal K
        num_spec_tokens_to_schedule = self.num_spec_tokens
        if self.dynamic_sd_lookup is not None and len(num_scheduled_tokens) > 0:
            num_spec_tokens_to_schedule = self.dynamic_sd_lookup[
                len(num_scheduled_tokens)
            ]





        scheduled_encoder_input_stats = None
        if (
            self.log_stats
            and self.observability_config.enable_logging_iteration_details
        ):
            scheduled_encoder_input_stats = self._make_scheduled_encoder_input_stats(
                scheduled_encoder_inputs
            )


        # 汇总所有的调度器输出
        # ------【核心逻辑】汇总调度结果为新/续算请求数据、token 预算、草稿、抢占与 KV 复制任务 ------
        scheduler_output = SchedulerOutput(
            scheduled_new_reqs=new_reqs_data, # 新增的、恢复的（v2）请求
            scheduled_cached_reqs=cached_reqs_data,# 上次继续的请求
            num_scheduled_tokens=num_scheduled_tokens, # 请求-token数的 表
            total_num_scheduled_tokens=total_num_scheduled_tokens, # 总计算token数
            scheduled_spec_decode_tokens=scheduled_spec_decode_tokens, # 请求-草稿tokenlist 表
            scheduled_encoder_inputs=scheduled_encoder_inputs,# 请求-编码器表
            scheduled_encoder_input_stats=scheduled_encoder_input_stats,
            num_common_prefix_blocks=num_common_prefix_blocks, # 最大前缀的block列表
            preempted_req_ids=self.reset_preempted_req_ids, # 本轮被抢占的victim的列表
            # finished_req_ids is an existing state in the scheduler,
            # instead of being newly scheduled in this step.
            # It contains the request IDs that are finished in between
            # the previous and the current steps.
            finished_req_ids=self.finished_req_ids, # ？ 何时更新
            free_encoder_mm_hashes=self.encoder_cache_manager.get_freed_mm_hashes(), # 【跳过】多模态encoder cache相关
            new_block_ids_to_zero=self._get_new_block_ids_to_zero(),# kv cache block 初始化， 告诉执行器，那些block需要清0，但如果是远端kv异步加载，不能清零，否则会和copy冲突
            kv_cache_block_copies=pending_kv_cache_block_copies,# cow复制任务，告诉执行器，需要备份
            partial_tail_offloads=pending_partial_tail_offloads,# 把partial tail KV block发送到远端。
            num_spec_tokens_to_schedule=num_spec_tokens_to_schedule, # 本轮采用多少draft token（单个，在vllm_config里面配置好的）
            ec_manager_metadata=self.encoder_cache_manager.get_manager_metadata(),# 【跳过】多模态 encoder cache manager信息
        )



        # 【跳过】下面一整段，主要是给scheduleroutput, 添加kv connector 元数据，以及维护scheduler step序号
        # NOTE(Kuntai): this function is designed for multiple purposes:
        # 1. Plan the KV cache store
        # 2. Wrap up all the KV cache load / save ops into an opaque object
        # 3. Clear the internal states of the connector
        if self.connector is not None:
            meta = self._build_kv_connector_meta(self.connector, scheduler_output)
            scheduler_output.kv_connector_metadata = meta

        # Build the connector meta for ECConnector
        if self.ec_connector is not None:
            ec_meta: ECConnectorMetadata = self.ec_connector.build_connector_meta(
                scheduler_output
            )
            scheduler_output.ec_connector_metadata = ec_meta

        # Advance the fence only for non-empty steps (those that actually
        # write KV and have their output processed later in update_from_output).
        # ------【异步 RPC】非空步推进 fence 序号，作为延迟释放 block 的回收屏障 ------
        if self.defer_block_free and total_num_scheduled_tokens > 0:
            self.sched_step_seq += 1








        # Phase5: _update_after_schedule： 调度后更新，上面Phase4, 已经把调度任务发出去，in-flight了，
        # ------【核心逻辑】Phase5：调度后同步内部状态，推进各请求已算 token 计数 ------
        # 这里就是更新好调度后的最新结果。
        with record_function_or_nullcontext("schedule: update_after_schedule"):
            self._update_after_schedule(scheduler_output) # 更新 Scheduler 自己内部认为 已经提交出去的状态，真正执行后的结果更新在update_from_output()


        return scheduler_output
















    # [新增] 构造 KV connector 元数据（disaggregated prefill）
    def _build_kv_connector_meta(
        self, connector: KVConnectorBase_V1, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        # ------【PD 分离】委托 connector 用调度输出构建 KV 元数据，供远端 prefill 传输 ------
        return connector.build_connector_meta(scheduler_output)

    # [新增] 返回需要清零的新分配 block ID 列表
    def _get_new_block_ids_to_zero(self) -> list[int] | None:
        # Drain new attention block ids every step so the manager-side list
        # does not grow unbounded; only kv-cache zeroing consumes them.
        # ------【内存池/CuMem】每步排空新分配的 attention block ID，防止管理器侧列表无界增长 ------
        new_block_ids_to_zero = self.kv_cache_manager.take_new_block_ids()
        # ------【核心逻辑】无需清零 KV cache 时直接返回 None，跳过后续过滤 ------
        if not self.needs_kv_cache_zeroing:
            return None

        # ------【核心逻辑】过滤掉本轮跳过清零的 block ID 后清空 skip 集合 ------
        if self._skip_zero_block_ids:
            skip = self._skip_zero_block_ids
            new_block_ids_to_zero = [b for b in new_block_ids_to_zero if b not in skip]
            skip.clear()

        return new_block_ids_to_zero or None


    # 针对victim，执行抢占操作的后处理
    def _preempt_request(
        self, request: Request, timestamp: float, drop_stale_output: bool = False
    ) -> None:
        """Preempt a request and put it back to the waiting queue.

        NOTE: The request should be popped from the running queue outside of this
        method.

        drop_stale_output: drop (rather than deliver) any in-flight output; used
        by reset_prefix_cache, whose same-step resume would otherwise deliver
        tokens out of order, and for connectors with a pending KV hand-off,
        which the preemption's block free would leave without valid KV.
        """

        # ------【核心逻辑】抢占前校验：只有 RUNNING 状态的请求才能被抢占 ------
        # 再检查一下，这个victim是running队列里面的
        assert request.status == RequestStatus.RUNNING, (
            "Only running requests can be preempted"
        )


        # ------【核心逻辑】释放该请求的 KV block 与多模态编码器占用，并取消在途 prefill ------
        self._free_request_blocks(request)# 释放这个请求的block占用
        self.encoder_cache_manager.free(request) #释放多模态编码器占用
        self._inflight_prefills.discard(request) # 发生出去的prefill，也取消掉

        # ------【核心逻辑】标记为 PREEMPTED 并把已算 token 置零，迫使 KV cache 重算 ------
        # 该victim的状态标记为被抢占
        request.status = RequestStatus.PREEMPTED
        request.num_computed_tokens = 0 # 已经计算的kvcache置零，所以后面需要重新计算kvcache了

        # ------【投机解码】清空待主模型验证的草稿 token 列表 ------
        #如果有投机解码的token列表，待大模型验证，也清空掉
        if request.spec_token_ids:
            request.spec_token_ids = []


        # ------【异步 RPC】异步调度下把在途输出标记为 stale，返回时仍投递但不污染计数 ------
        # Async scheduling: mark all in-flight output as stale. Its tokens are
        # still delivered on return (dropping them would perturb spec-decode
        # acceptance) but must not mutate the reset counters; each step drains
        # its share in update_from_output. num_in_flight_tokens already
        # includes any undrained stale share, so assign rather than accumulate.
        # An undrained drop-mode share stays dropped: its positions have
        # already been resampled.
        request.drop_stale_output = drop_stale_output or (      # 这些stale，placeholders，都是异步调度的概念了，这里不做深究
            request.drop_stale_output and request.num_stale_output_tokens > 0
        )
        request.num_stale_output_tokens = request.num_in_flight_tokens
        request.num_output_placeholders = 0

        # ------【核心逻辑】累计抢占次数并按需记录 PREEMPTED 事件 ------
        # victim的被抢占计数 +1
        request.num_preemptions += 1
        if self.log_stats:
            request.record_event(EngineCoreEventType.PREEMPTED, timestamp)

        # ------【核心逻辑】把 victim 塞回 waiting 队列并记入本轮被抢占集合 ------
        # Put the request back to the waiting queue.
        self.waiting.prepend_request(request) # 把这个victim的请求，加入waiting队列
        self.reset_preempted_req_ids.add(request.request_id) # 本轮被抢占的集合，保存这个victim



    # 调度后更新，更新每个请求在执行器计算后的状态
    def _update_after_schedule(self, scheduler_output: SchedulerOutput) -> None:
        # Advance the number of computed tokens for the request AFTER
        # the request is scheduled.
        # 1. The scheduler_output of the current step has to include the
        #    original number of scheduled tokens to determine input IDs.
        # 2. Advance the number of computed tokens here allowing us to
        #    schedule the prefill request again immediately in the next
        #    scheduling step.
        # 3. If some tokens (e.g. spec tokens) are rejected later, the number of
        #    computed tokens will be adjusted in update_from_output.
        # ------【核心逻辑】预先把本步发出的 token 记到 computed/in_flight，后续再核验 ------
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens
        for req_id, num_scheduled_token in num_scheduled_tokens.items():
            request = self.requests[req_id]
            request.num_computed_tokens += num_scheduled_token # 先把发出的计算token，先算到自己头上，后面再核验
            request.num_in_flight_tokens += num_scheduled_token # 标记在gpu计算的token数量
            # ------【异步 RPC】记录本次调度序号作为围栏，供延迟释放 block 判断在途写是否完成 ------
            if self.defer_block_free:
                # Record the in-flight step, to fence deferred block freeing.
                request.last_sched_seq = self.sched_step_seq
            # ------【chunked prefill + 结构化输出/grammar】更新分块预填充标志并汇总结构化输出请求标志 ------
            request.is_prefill_chunk = request.num_computed_tokens < (
                request.num_tokens + request.num_output_placeholders
            )
            scheduler_output.has_structured_output_requests |= (
                request.use_structured_output and not request.is_prefill_chunk
            )
            # ------【核心逻辑】不再是 prefill chunk 后从在途 prefill 集合移除 ------
            # Drop from the in-flight-prefill set once it's no longer prefilling.
            if not request.is_prefill_chunk:
                self._inflight_prefills.discard(request)

        # Snapshot block IDs for routed experts before forward starts.
        # A concurrent schedule() may preempt requests and free blocks
        # before update_from_output runs; the snapshot survives that.
        # Use update() to preserve entries from the previous step that
        # have not yet been consumed by update_from_output (async
        # scheduling may call _update_after_schedule again before the
        # prior update_from_output runs).
        # ------【EP/EPLB】forward 前快照路由专家的 block ID，抵御异步抢占并发释放 ------
        if self.enable_return_routed_experts:
            gid = self.routed_experts_mgr.attn_gid
            self._re_block_ids.update(
                {
                    rid: self.kv_cache_manager.get_blocks(rid).get_block_ids()[gid]
                    for rid in num_scheduled_tokens
                }
            )

        # ------【核心逻辑】清空已结束/被抢占请求集合，避免影响 scheduler_output ------
        # Clear the finished and preempted request IDs.
        # NOTE: We shouldn't just clear() here because it will also affect
        # the scheduler output.
        self.finished_req_ids = set()
        self.reset_preempted_req_ids = set()

    def _update_request_as_session(
        self, session: Request, update: StreamingUpdate
    ) -> None:
        """
        Updates the waiting session with the next streaming update.

        Discards the last sampled output token from the prior input chunk.
        """

        # Current streaming input behaviour: Keep only computed output tokens
        # (discard final sampled output token).
        # ------【核心逻辑】只保留已算的采样输出 token 作为下一输入块，丢弃末尾采样 token ------
        num_computed_tokens = session.num_computed_tokens
        kept_output_tokens = session._all_token_ids[
            session.num_prompt_tokens : num_computed_tokens
        ]
        del session._all_token_ids[num_computed_tokens:]
        session._output_token_ids.clear()
        assert session.prompt_token_ids is not None
        # Extend prompt with kept output tokens.
        session.prompt_token_ids.extend(kept_output_tokens)

        # ------【核心逻辑】把流式多模态特征的 position 偏移到当前会话 token 基准 ------
        if update.mm_features:
            base = session.num_tokens
            for mm_feature in update.mm_features:
                mm_feature.mm_position = replace(
                    mm_feature.mm_position, offset=mm_feature.mm_position.offset + base
                )
            session.mm_features.extend(update.mm_features)

        # ------【核心逻辑】追加新输入块的 prompt token 并更新块哈希 ------
        session._all_token_ids.extend(update.prompt_token_ids or ())
        session.prompt_token_ids.extend(update.prompt_token_ids or ())
        # Update block hashes for the new tokens.
        session.update_block_hashes()
        session.num_prompt_tokens = len(session.prompt_token_ids)
        # ------【核心逻辑】更新到达时间与采样参数，状态转回 WAITING 并扣减流式等待计数 ------
        session.arrival_time = update.arrival_time
        session.sampling_params = update.sampling_params
        if session.status == RequestStatus.WAITING_FOR_STREAMING_REQ:
            self.num_waiting_for_streaming_input -= 1
        session.status = RequestStatus.WAITING

        if self.log_stats:
            session.record_event(EngineCoreEventType.QUEUED)

    def _make_cached_request_data(
        self,
        running_reqs: list[Request],
        resumed_reqs: list[Request],
        num_scheduled_tokens: dict[str, int],
        spec_decode_tokens: dict[str, list[int]],
        req_to_new_blocks: dict[str, KVCacheBlocks],
    ) -> CachedRequestData:
        req_ids: list[str] = []
        new_token_ids: list[list[int]] = []
        new_block_ids: list[tuple[list[int], ...] | None] = []
        all_token_ids: dict[str, list[int]] = {}
        num_computed_tokens: list[int] = []
        num_output_tokens: list[int] = []
        resumed_req_ids = set()

        # ------【核心逻辑】遍历 running+resumed 请求，组装缓存请求数据（token/block 快照） ------
        num_running_reqs = len(running_reqs)
        for idx, req in enumerate(itertools.chain(running_reqs, resumed_reqs)):
            req_id = req.request_id
            req_ids.append(req_id)
            # NOTE: In PP+async scheduling, we consume token ids via a direct GPU
            # broadcast path (`input_batch.prev_sampled_token_ids`), so we can
            # omit this payload.
            # ------【PP】流水线并行下把采样 token 随调度数据回传，因首尾 stage 无直接通信 ------
            if self.use_pp and not self.scheduler_config.async_scheduling:
                # When using PP, the scheduler sends the sampled tokens back,
                # because there's no direct communication between the first-
                # stage worker and the last-stage worker. Otherwise, we don't
                # need to send the sampled tokens back because the model runner
                # will cache them.
                num_tokens = num_scheduled_tokens[req_id] - len(
                    spec_decode_tokens.get(req_id, ())
                )
                token_ids = req.all_token_ids[
                    req.num_computed_tokens : req.num_computed_tokens + num_tokens
                ]
                new_token_ids.append(token_ids)
            # ------【核心逻辑】区分 running 与 resumed 请求，记录需恢复的请求 ID ------
            if idx >= num_running_reqs:
                resumed_req_ids.add(req_id)
            # ------【核心逻辑】非 v2 model runner 下为新出现的请求复制完整 token 列表 ------
            if not self.use_v2_model_runner:  # noqa: SIM102
                if req_id not in self.prev_step_scheduled_req_ids:
                    all_token_ids[req_id] = req.all_token_ids.copy()
            # ------【核心逻辑】收集 block ID、computed/output token 数，填充 CachedRequestData 字段 ------
            new_block_ids.append(
                req_to_new_blocks[req_id].get_block_ids(allow_none=True)
            )
            num_computed_tokens.append(req.num_computed_tokens)
            num_output_tokens.append(
                req.num_output_tokens + req.num_output_placeholders
            )

        return CachedRequestData(
            req_ids=req_ids,
            resumed_req_ids=resumed_req_ids,
            new_token_ids=new_token_ids,
            all_token_ids=all_token_ids,
            new_block_ids=new_block_ids,
            num_computed_tokens=num_computed_tokens,
            num_output_tokens=num_output_tokens,
        )

    def _try_schedule_encoder_inputs(
        self,
        request: Request,
        num_computed_tokens: int,
        num_new_tokens: int,
        encoder_compute_budget: int,
        shift_computed_tokens: int = 0,
    ) -> tuple[list[int], int, int, list[int]]:
        """
        Determine which encoder inputs need to be scheduled in the current step,
        and update `num_new_tokens` and encoder token budget accordingly.

        An encoder input will be scheduled if:
        - Its output tokens overlap with the range of tokens being computed
        in this step, i.e.,
        [num_computed_tokens, num_computed_tokens + num_new_tokens).
        - It is not already computed and stored in the encoder cache.
        - It is not exist on remote encoder cache (via ECConnector)
        - There is sufficient encoder token budget to process it.
        - The encoder cache has space to store it.

        If an encoder input cannot be scheduled due to cache or budget
        limitations, the method adjusts `num_new_tokens` to schedule only the
        decoder tokens up to just before the unschedulable encoder input.

        Note that num_computed_tokens includes both locally cached
        blocks and externally cached blocks (via KVConnector).
        """
        # ------【核心逻辑】无新 token 或无编码器输入时直接返回空结果 ------
        if num_new_tokens == 0 or not request.has_encoder_inputs:
            return [], num_new_tokens, encoder_compute_budget, []
        encoder_inputs_to_schedule: list[int] = []
        mm_features = request.mm_features
        assert mm_features is not None
        assert len(mm_features) > 0
        external_load_encoder_input = []

        # NOTE: since scheduler operates on the request level (possibly with
        # multiple encoder inputs per request), we need to create temporary
        # trackers for accounting at the encoder input level.
        mm_hashes_to_schedule = set()
        num_embeds_to_schedule = 0

        # ------【核心逻辑】用窗口函数定位本步需处理的编码器输入区间 ------
        lo, hi = get_mm_features_in_window(
            mm_features,
            start=num_computed_tokens,
            end=num_computed_tokens + num_new_tokens + shift_computed_tokens,
        )
        # For encoder-decoder, all inputs sit at start_pos=0, so lo=0 always.
        if self.is_encoder_decoder:
            lo = 0

        # ------【核心逻辑】逐个编码器输入判断是否需要在本步调度 ------
        for i in range(lo, hi):
            mm_feature = mm_features[i]
            start_pos = mm_feature.mm_position.offset
            num_encoder_tokens = mm_feature.mm_position.length
            num_encoder_embeds = mm_feature.mm_position.get_num_embeds()
            item_identifier = mm_feature.identifier

            # ------【核心逻辑】编码器-解码器模型：decoder 已计算则跳过编码器输入 ------
            if self.is_encoder_decoder and num_computed_tokens > 0:
                assert start_pos == 0, (
                    "Encoder input should be processed at the beginning of "
                    "the sequence when encoder-decoder models are used."
                )
                # Encoder input has already been computed
                # The calculation here is a bit different. We don't turn encoder
                # output into tokens that get processed by the decoder and
                # reflected in num_computed_tokens. Instead, start_pos reflects
                # the position where we need to ensure we calculate encoder
                # inputs. This should always be 0 to ensure we calculate encoder
                # inputs before running the decoder.  Once we've calculated some
                # decoder tokens (num_computed_tokens > 0), then we know we
                # already calculated encoder inputs and can skip here.
                continue

            # ------【核心逻辑】去重：同一步已调度或已缓存的编码器输入直接跳过 ------
            if not self.is_encoder_decoder:
                # We are not using the encoder cache for encoder-decoder models,
                # yet.
                if item_identifier in mm_hashes_to_schedule:
                    # The same encoder input has already been scheduled in the
                    # current step.
                    continue

                if self.encoder_cache_manager.check_and_update_cache(request, i):
                    # The encoder input is already computed and cached from a
                    # previous step.
                    continue

            # If no encoder input chunking is allowed, we do not want to
            # partially schedule a multimodal item. If the scheduled range would
            # only cover part of the mm input, roll back to before the mm item.
            # ------【chunked prefill】禁止分块的多模态输入只覆盖部分时回退到该项之前 ------
            if (
                self.scheduler_config.disable_chunked_mm_input
                and num_computed_tokens < start_pos
                and (num_computed_tokens + num_new_tokens)
                < (start_pos + num_encoder_tokens)
            ):
                # Account for EAGLE shift when rolling back to avoid
                # encoder cache miss. This ensures the scheduled range
                # stops before start_pos even with the shift.
                num_new_tokens = max(
                    0, start_pos - (num_computed_tokens + shift_computed_tokens)
                )
                break
            # ------【核心逻辑】编码器缓存满或预算耗尽时只调度到该项之前的 decoder token ------
            if not self.encoder_cache_manager.can_allocate(
                request, i, encoder_compute_budget, num_embeds_to_schedule
            ):
                # The encoder cache is full or the encoder budget is exhausted.
                # NOTE(woosuk): We assume that the encoder input tokens should
                # be processed altogether, as the encoder usually uses
                # bidirectional attention.
                if num_computed_tokens + shift_computed_tokens < start_pos:
                    # We only schedule the decoder tokens just before the
                    # encoder input.
                    num_new_tokens = start_pos - (
                        num_computed_tokens + shift_computed_tokens
                    )
                else:
                    # Because of prefix caching, num_computed_tokens is greater
                    # than start_pos even though its encoder input is not
                    # available. In this case, we can't schedule any token for
                    # the request in this step.
                    num_new_tokens = 0
                break

            # Calculate the number of embeddings to schedule in the current range
            # of scheduled encoder placeholder tokens.
            # ------【核心逻辑】计算当前窗口内要调度的 embedding 范围，无 embedding 则跳过 ------
            start_idx_rel = max(0, num_computed_tokens - start_pos)
            end_idx_rel = min(
                num_encoder_tokens, num_computed_tokens + num_new_tokens - start_pos
            )
            curr_embeds_start, curr_embeds_end = (
                mm_feature.mm_position.get_embeds_indices_in_range(
                    start_idx_rel, end_idx_rel
                )
            )
            # There's no embeddings in the current range of encoder placeholder tokens
            # so we can skip the encoder input.
            if curr_embeds_end - curr_embeds_start == 0:
                continue

            # ------【PD 分离】远端 EC connector 已缓存该输入则走外部加载路径，不占本地预算 ------
            if self.ec_connector is not None and self.ec_connector.has_cache_item(
                item_identifier
            ):
                mm_hashes_to_schedule.add(item_identifier)
                external_load_encoder_input.append(i)
                num_embeds_to_schedule += num_encoder_embeds
                continue

            # ------【核心逻辑】本地调度该编码器输入：累计 embeds 并扣减编码器预算 ------
            num_embeds_to_schedule += num_encoder_embeds
            encoder_compute_budget -= num_encoder_embeds
            mm_hashes_to_schedule.add(item_identifier)
            encoder_inputs_to_schedule.append(i)

        return (
            encoder_inputs_to_schedule,
            num_new_tokens,
            encoder_compute_budget,
            external_load_encoder_input,
        )

    def _make_scheduled_encoder_input_stats(
        self, scheduled_encoder_inputs: dict[str, list[int]]
    ) -> ScheduledEncoderInputStats | None:
        stats = ScheduledEncoderInputStats()
        # ------【核心逻辑】统计本步调度的编码器输入数量与输出 token 数 ------
        for req_id, input_ids in scheduled_encoder_inputs.items():
            request = self.requests.get(req_id)
            if request is None:
                continue

            for input_id in input_ids:
                mm_feature = request.mm_features[input_id]
                stats.num_inputs += 1
                stats.output_tokens += mm_feature.mm_position.get_num_embeds()

        return stats if stats.num_inputs else None

    def get_grammar_bitmask(
        self, scheduler_output: SchedulerOutput
    ) -> GrammarOutput | None:
        # ------【结构化输出/grammar】快速出口：本步无结构化输出请求则跳过位掩码生成 ------
        # Collect list of scheduled request ids that use structured output.
        # The corresponding rows of the bitmask will be in this order.
        if not scheduler_output.has_structured_output_requests:
            return None

        structured_output_request_ids = [
            req_id
            for req_id in scheduler_output.num_scheduled_tokens
            if (req := self.requests.get(req_id))
            and (req.use_structured_output and not req.is_prefill_chunk)
        ]
        if not structured_output_request_ids:
            return None

        # ------【结构化输出/grammar】调 structured_output_manager 生成位掩码并包装为 GrammarOutput ------
        bitmask = self.structured_output_manager.grammar_bitmask(
            self.requests,
            structured_output_request_ids,
            scheduler_output.scheduled_spec_decode_tokens,
        )
        return GrammarOutput(structured_output_request_ids, bitmask)


    # 核销调度器的状态， 把 模型算出来的token收回来，落回到每个req上，然后决定谁该结束的一步
    # 收结果 + 销账 + 判定停止
    # （异步调度 + KV connector + 多模态 + 结构化输出 + 投机解码 + perf metrics + DP）这些是附加的优化
    def update_from_output(
        self,
        scheduler_output: SchedulerOutput,
        model_runner_output: ModelRunnerOutput,
    ) -> dict[int, EngineCoreOutputs]:
        
        # ------【核心逻辑】取出本步模型输出与调度元数据：采样 token/logprobs/pooler，
        #   以及 CUDA Graph 统计、KV connector 输出等附加优化结果 ──
        sampled_token_ids = model_runner_output.sampled_token_ids
        logprobs = model_runner_output.logprobs
        prompt_logprobs_dict = model_runner_output.prompt_logprobs_dict
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens
        pooler_outputs = model_runner_output.pooler_output
        num_nans_in_logits = model_runner_output.num_nans_in_logits
        kv_connector_output = model_runner_output.kv_connector_output
        cudagraph_stats = model_runner_output.cudagraph_stats


        # ------【异步 RPC】defer_block_free：异步调度下本步及之前的 GPU 写已完成，
        #   可安全把延迟释放的 KV 块归还内存池 ──
        # defer=延迟
        # Every GPU write enqueued by this and earlier steps has completed, so it is
        # safe to return deferred-free blocks to the pool.
        if self.defer_block_free and scheduler_output.total_num_scheduled_tokens > 0:
            self.processed_step_seq += 1
            self._drain_deferred_frees()

        # ------【性能指标】按 GPU 采集本步性能指标（perf_metrics），供可观测与调优 ------
        perf_stats: PerfStats | None = None
        if self.perf_metrics and self.perf_metrics.is_enabled():
            perf_stats = self.perf_metrics.get_step_perf_stats_per_gpu(scheduler_output)

        # ------【核心逻辑】按 client 分组收集输出；spec_decoding_stats 用于累加【投机解码】统计 ------
        outputs: dict[int, list[EngineCoreOutput]] = defaultdict(list)
        spec_decoding_stats: SpecDecodingStats | None = None

        # ------【PD 分离】远程 KV cache 加载失败：标记受影响请求并回退其已算 token 数，
        #   触发对无效块的重算 ──
        failed_kv_load_req_ids = None
        if kv_connector_output and kv_connector_output.invalid_block_ids:
            # These blocks contain externally computed tokens that failed to
            # load. Identify affected requests and adjust their computed token
            # count to trigger recomputation of the invalid blocks.
            failed_kv_load_req_ids = self._handle_invalid_blocks(
                kv_connector_output.invalid_block_ids,
                num_scheduled_tokens,
            )

        # ------【EP/EPLB】把本步每 token 路由到的专家写入调度器侧 slot 缓冲，
        #   并按 model runner 的请求顺序构建偏移表，供下面逐请求读取路由 ──
        # Persist per-step routed experts into the scheduler-side slot
        # buffer (CPU->CPU fancy-index assign; ~few MB per step).
        # MUST precede the per-request routing reads below: stopped
        # requests may terminate on tokens generated in this very step,
        # whose routing was just D2H'd into model_runner_output.
        routing_data = None
        routing_offsets: dict[str, int] = {}
        if model_runner_output.routed_experts is not None:
            re = model_runner_output.routed_experts
            self.routed_experts_mgr.store_batch(re.routing_data, re.slot_mapping)
            routing_data = re.routing_data.astype(
                self.routed_experts_mgr.routed_experts_by_slot.dtype,
                copy=False,
            )
            # Build offset map using model runner's request order
            # (input_batch ordering), NOT scheduler dict order.
            offset = 0
            for rid in model_runner_output.req_ids:
                routing_offsets[rid] = offset
                offset += num_scheduled_tokens[rid]

        # ------【核心逻辑】主循环：逐请求核销 in-flight 计数、判定停止并回收 KV cache；
        #   注意循环长度可达 1K+，是性能热点，需避免昂贵操作 ──
        # NOTE(woosuk): As len(num_scheduled_tokens) can be up to 1K or more,
        # the below loop can be a performance bottleneck. We should do our best
        # to avoid expensive operations inside the loop.

        # 核销 in-flight 计数
        stopped_running_reqs: set[Request] = set()
        stopped_preempted_reqs: set[Request] = set()
        for req_id, num_tokens_scheduled in num_scheduled_tokens.items():
            assert num_tokens_scheduled > 0
            request = self.requests.get(req_id)
            output_is_stale = False
            if request is not None:
                request.num_in_flight_tokens -= num_tokens_scheduled
                # Drain any stale share (see _preempt_request) in lockstep.
                if request.num_stale_output_tokens > 0:
                    output_is_stale = True
                    request.num_stale_output_tokens -= num_tokens_scheduled
                    assert request.num_stale_output_tokens >= 0
            if failed_kv_load_req_ids and req_id in failed_kv_load_req_ids:
                # skip failed or rescheduled requests from KV load failure
                continue
            # ------【PP + 异步 RPC】请求可能在流水线并行/异步调度执行中被中止，此处跳过；
            #   delay_free_blocks 下用 is_finished() 判断 ──
            if request is None or request.is_finished():
                # The request is already finished. This can happen if the
                # request is aborted while the model is executing it (e.g.,
                # in pipeline parallelism or in async scheduling).
                # NOTE(Kuntai): When delay_free_blocks=True (for async KV
                # cache transfer in KV connector), the aborted request will not
                # be set to None (in order to finish async KV transfer).
                # In this case, we use is_finished() to check.
                continue

            # ------【异步 RPC】drop 模式的 stale 输出（同一步恢复场景）整体丢弃 ------
            # Drop-mode stale output (same-step resume) is discarded entirely.
            if output_is_stale and request.drop_stale_output:
                continue

            #得到本step新产出的token
            req_index = model_runner_output.req_id_to_index[req_id]
            generated_token_ids = (
                sampled_token_ids[req_index] if sampled_token_ids else []
            )

            # ------【投机解码】统计草稿 token 接受/拒绝，并回滚被拒绝 token 的
            #   num_computed_tokens（异步调度下连同 num_output_placeholders）──
            scheduled_spec_token_ids = (
                scheduler_output.scheduled_spec_decode_tokens.get(req_id)
            )
            if scheduled_spec_token_ids and (
                generated_token_ids or self.num_sampled_tokens_per_step == 0
            ):
                num_draft_tokens = len(scheduled_spec_token_ids)
                num_sampled = self.num_sampled_tokens_per_step
                num_accepted = max(len(generated_token_ids) - num_sampled, 0)
                num_rejected = num_draft_tokens - num_accepted
                # Rejections roll back num_computed_tokens (and, under async
                # scheduling, num_output_placeholders, which covers the spec
                # tokens). A stale rejection count predates the preemption
                # rollback and must not apply.
                if not output_is_stale:
                    if request.num_computed_tokens > 0:
                        request.num_computed_tokens -= num_rejected
                    if request.num_output_placeholders > 0:
                        request.num_output_placeholders -= num_rejected
                spec_decoding_stats = self.make_spec_decoding_stats(
                    spec_decoding_stats,
                    num_draft_tokens=num_draft_tokens,
                    num_accepted_tokens=num_accepted,
                    num_invalid_spec_tokens=scheduler_output.num_invalid_spec_tokens,
                    request_id=req_id,
                )

            # ------【核心逻辑】本步确实执行后才释放编码器输入缓存 ------
            # Free encoder inputs only after the step has actually executed.
            if request.has_encoder_inputs:
                self._free_encoder_inputs(request)

            # ------【核心逻辑】初始化每请求的停止标志/新 token/日志概率等局部变量 ------
            stopped = False
            new_logprobs = None
            new_token_ids = generated_token_ids
            pooler_output = pooler_outputs[req_index] if pooler_outputs else None
            kv_transfer_params = None
            ec_transfer_params = None
            prefill_stats = None
            status_before_stop = request.status
            num_output_tokens_before = len(request._output_token_ids)

            # ------【核心逻辑】追加新 token 并判定停止；pooling/encoder-only 也在此判停 ------
            # Check for stop and update request status.
            # 追加token + 判定停止
            if new_token_ids:
                new_token_ids, stopped = self._update_request_with_output(
                    request, new_token_ids, is_stale=output_is_stale
                )
            elif request.pooling_params and pooler_output is not None:
                # Pooling stops as soon as there is output.
                request.status = RequestStatus.FINISHED_STOPPED
                stopped = True
            elif (
                self.is_encoder_only
                and request.num_computed_tokens >= request.num_prompt_tokens
            ):
                # An encoder instance runs the encoder and publishes the
                # embeddings instead of sampling, so it stops as soon as the
                # whole prompt is consumed. Encoder inputs are never scheduled
                # past a multi-modal item the encoder cache could not admit, so
                # a consumed prompt also means every item in it was encoded.
                request.status = RequestStatus.FINISHED_STOPPED # 判定停止
                stopped = True

            # ------【结构化输出/grammar】推进语法状态机：剔除推理段 token 后喂给 grammar，
            #   被拒绝则把请求判为 FINISHED_ERROR ──
            if new_token_ids and self.structured_output_manager.should_advance(
                request, new_token_ids=new_token_ids
            ):
                struct_output_request = request.structured_output_request
                assert struct_output_request is not None
                grammar = struct_output_request.grammar
                assert isinstance(grammar, StructuredOutputGrammar)
                # new_token_ids can be a mixed block of reasoning content, then
                # the reasoning end marker, then the start of the grammar content.
                # Trim the reasoning content so the grammar only sees grammar content.
                advance_token_ids = (
                    self.structured_output_manager.trim_reasoning_for_advance(
                        request, new_token_ids
                    )
                )
                if advance_token_ids and not grammar.accept_tokens(
                    req_id, advance_token_ids
                ):
                    logger.error(
                        "Unexpected: grammar rejected tokens %s for request %s. "
                        "Terminating request.",
                        advance_token_ids,
                        req_id,
                    )
                    request.status = RequestStatus.FINISHED_ERROR
                    request.resumable = False
                    stopped = True

            # ------【EP/EPLB】取回路由专家：prefill 从 slot 缓冲读完整 prompt，
            #   decode 读末尾 token，投机解码读接受区间 ──
            routed_experts = None
            if (
                self.enable_return_routed_experts
                and routing_data is not None
                and new_token_ids
            ):
                req_offset = routing_offsets[req_id]
                end = req_offset + num_tokens_scheduled
                block_ids = self._re_block_ids.pop(req_id, [])
                if num_output_tokens_before == 0:
                    # Prefill completed: read full prompt routing from
                    # slot buffer using the block-ID snapshot taken at
                    # schedule time (immune to async preemption).
                    if (
                        request.sampling_params is not None
                        and request.sampling_params.routed_experts_prompt_start
                        is not None
                    ):
                        prompt_start = (
                            request.sampling_params.routed_experts_prompt_start
                        )
                        assert prompt_start < request.num_prompt_tokens
                    else:
                        prompt_start = 0
                    routed_experts = self.routed_experts_mgr.get(
                        block_ids,
                        request.num_prompt_tokens,
                        token_start=prompt_start,
                    )
                else:
                    if scheduled_spec_token_ids:
                        # Spec decode: accepted tokens at the START of
                        # the scheduled range, rejected at the end.
                        routed_experts = routing_data[
                            req_offset : req_offset + len(new_token_ids)
                        ]
                    else:
                        # Normal decode / re-prefill: token(s) at the END.
                        routed_experts = routing_data[end - len(new_token_ids) : end]

            # ------【前缀缓存】prefill 统计里用 estimate_cached_tokens 记录命中前缀缓存的 token 数 ------
            should_emit_output = bool(
                new_token_ids or pooler_output is not None or stopped
            )
            if should_emit_output:
                prefill_stats = request.take_prefill_stats()
                if prefill_stats is not None:
                    prefill_stats.finalize(
                        self.kv_cache_manager.estimate_cached_tokens(request)
                    )

            # ------【核心逻辑 + PD 分离】记录停止原因、回收请求并释放 KV cache
            #   （_free_request 返回 KV/EC 传输参数供 connector 使用）──
            finish_reason = None
            if stopped:
                # Capture finish_reason BEFORE _handle_stopped_request, which may
                # reset the status to WAITING for streaming requests that continue.
                finish_reason = request.get_finished_reason()
                finished = self._handle_stopped_request(request)
                if finished:
                    kv_transfer_params, ec_transfer_params = self._free_request(request)

                if status_before_stop == RequestStatus.RUNNING:
                    stopped_running_reqs.add(request)
                else:
                    stopped_preempted_reqs.add(request)

            # ------【核心逻辑】按需抽取采样 token 的 logprobs 与 NaN 计数 ------
            # Extract sample logprobs if needed.
            if (
                request.sampling_params is not None
                and request.sampling_params.num_logprobs is not None
                and logprobs
            ):
                new_logprobs = logprobs.slice_request(req_index, len(new_token_ids))

            if num_nans_in_logits is not None and req_id in num_nans_in_logits:
                request.num_nans_in_logits = num_nans_in_logits[req_id]

            # ------【核心逻辑】构造 EngineCoreOutput 交给上层；无输出时不返回部分 prefill 结果 ------
            # Get prompt logprobs for this request.
            prompt_logprobs_tensors = prompt_logprobs_dict.get(req_id)
            if should_emit_output:
                # Add EngineCoreOutput for this Request. # 构造EngineCoreOutput 交给上层（引擎、客户端）
                outputs[request.client_index].append(
                    EngineCoreOutput(
                        request_id=req_id,
                        new_token_ids=new_token_ids,
                        finish_reason=finish_reason,
                        new_logprobs=new_logprobs,
                        new_prompt_logprobs_tensors=prompt_logprobs_tensors,
                        pooling_output=pooler_output,
                        stop_reason=request.stop_reason,
                        events=request.take_events(),
                        prefill_stats=prefill_stats,
                        kv_transfer_params=kv_transfer_params,
                        ec_transfer_params=ec_transfer_params,
                        trace_headers=request.trace_headers,
                        routed_experts=routed_experts,
                        num_nans_in_logits=request.num_nans_in_logits,
                    )
                )
            else:
                # Invariant: EngineCore returns no partial prefill outputs.
                assert not prompt_logprobs_tensors

        # ------【核心逻辑】把本步停止的请求从 running / waiting 队列移除 ------
        # Remove the stopped requests from the running and waiting queues.
        if stopped_running_reqs:
            self.running = remove_all(self.running, stopped_running_reqs)
        if stopped_preempted_reqs:
            # This is a rare case and unlikely to impact performance.
            self.waiting.remove_requests(stopped_preempted_reqs)
            self.skipped_waiting.remove_requests(stopped_preempted_reqs)

        # ------【结构化输出/grammar + PD 分离】语法编译失败或远程 KV 加载失败的请求按错误结束 ------
        error_req_ids = set(self.grammar_compile_error_reqs)
        self.grammar_compile_error_reqs.clear()
        if failed_kv_load_req_ids and not self.recompute_kv_load_failures:
            error_req_ids.update(failed_kv_load_req_ids)

        if error_req_ids:
            error_reqs = self.finish_requests(
                error_req_ids, RequestStatus.FINISHED_ERROR
            )
            for request in error_reqs:
                outputs[request.client_index].append(
                    EngineCoreOutput(
                        request_id=request.request_id,
                        new_token_ids=[],
                        finish_reason=request.get_finished_reason(),
                        events=request.take_events(),
                        trace_headers=request.trace_headers,
                    )
                )

        # ------【PD 分离】KV Connector：更新已完成的远程 KV 传输状态 ------
        # KV Connector: update state for finished KV Transfers.
        if kv_connector_output:
            self._update_from_kv_xfer_finished(kv_connector_output)

        # ------【PD 分离】汇总 worker 侧与调度器侧的 KV connector 统计 ------
        # Worker-side KV connector stats from the model runner output.
        kv_connector_stats: KVConnectorStats | None = (
            kv_connector_output.kv_connector_stats if kv_connector_output else None
        )
        if self.connector:
            # Scheduler-side KV connector stats collected after connector update.
            scheduler_kv_connector_stats = self.connector.get_kv_connector_stats()
            if (
                scheduler_kv_connector_stats is not None
                and not scheduler_kv_connector_stats.is_empty()
            ):
                kv_connector_stats = (
                    kv_connector_stats.aggregate(scheduler_kv_connector_stats)
                    if kv_connector_stats is not None
                    else scheduler_kv_connector_stats
                )

        # ------【PD 分离】收集 KV cache 管理器与 connector 的事件并发布，供观测 KV 缓存活动 ------
        # collect KV cache events from KV cache manager
        events = self.kv_cache_manager.take_events()

        # collect KV cache events from connector
        if self.connector is not None:
            connector_events = self.connector.take_events()
            if connector_events:
                if events is None:
                    events = list(connector_events)
                else:
                    events.extend(connector_events)

        # publish collected KV cache events
        if events:
            batch = KVEventBatch(ts=time.time(), events=events)
            self.kv_event_publisher.publish(batch)

        # ------【核心逻辑】把各 client 的输出封装为 EngineCoreOutputs ------
        # Create EngineCoreOutputs for all clients that have requests with
        # outputs in this step.
        engine_core_outputs = {
            client_index: EngineCoreOutputs(outputs=outs)
            for client_index, outs in outputs.items()
        }

        # ------【核心逻辑】把自上次发送以来结束的请求 ID 挂到对应 client 的输出上 ------
        finished_req_ids = self.finished_req_ids_dict
        if finished_req_ids:
            # Include ids of requests that finished since last outputs
            # were sent.
            for client_index, finished_set in finished_req_ids.items():
                # Set finished request set in EngineCoreOutputs for this client.
                if (eco := engine_core_outputs.get(client_index)) is not None:
                    eco.finished_requests = finished_set
                else:
                    engine_core_outputs[client_index] = EngineCoreOutputs(
                        finished_requests=finished_set
                    )
            finished_req_ids.clear()

        # ------【投机解码 + CUDA Graph + PD 分离】汇总各优化子系统的统计，仅返回给一个前端 ------
        if (
            stats := self.make_stats(
                spec_decoding_stats,
                kv_connector_stats,
                cudagraph_stats,
                perf_stats,
            )
        ) is not None:
            # Return stats to only one of the front-ends.
            if (eco := next(iter(engine_core_outputs.values()), None)) is None:
                # We must return the stats even if there are no request
                # outputs this step.
                engine_core_outputs[0] = eco = EngineCoreOutputs()
            eco.scheduler_stats = stats

        return engine_core_outputs









    @staticmethod
    def _is_blocked_waiting_status(status: RequestStatus) -> bool:
        # ------【核心逻辑】判断请求是否处于被阻塞的等待状态（grammar/远端 KV/流式输入） ------
        return status in (
            RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR,
            RequestStatus.WAITING_FOR_REMOTE_KVS,
            RequestStatus.WAITING_FOR_STREAMING_REQ,
        )

    def _enqueue_waiting_request(self, request: Request) -> None:
        # ------【核心逻辑】按是否被阻塞，把请求分别放入 skipped_waiting 或 waiting 队列 ------
        if self._is_blocked_waiting_status(request.status):
            self.skipped_waiting.add_request(request)
        else:
            self.waiting.add_request(request)

    def _select_waiting_queue_for_scheduling(self) -> RequestQueue | None:
        # ------【核心逻辑】FCFS 策略下优先取 skipped 阻塞队列，为空再取 waiting 队列 ------
        if self.policy == SchedulingPolicy.FCFS:
            return self.skipped_waiting or self.waiting or None # 默认优先返回skip阻塞队列，为空就返回waiting就绪队列

        # ------【核心逻辑】PRIORITY 策略：两队列都有请求时比较队头优先级取小者 ------
        # PRIORITY mode: compare queue heads when both queues are non-empty.
        if self.waiting and self.skipped_waiting:
            waiting_req = self.waiting.peek_request()
            skipped_req = self.skipped_waiting.peek_request()
            return self.waiting if waiting_req < skipped_req else self.skipped_waiting

        return self.waiting or self.skipped_waiting or None

    def _handle_stopped_request(self, request: Request) -> bool:
        """Return True if finished (can be False for resumable requests)."""
        # ------【核心逻辑】不可恢复请求直接视为结束 ------
        if not request.resumable:
            return True

        # ------【核心逻辑】流式输入会话：弹出下一块更新会话，None 表示会话结束 ------
        if request.streaming_queue:
            update = request.streaming_queue.popleft()
            if update is None:
                # Streaming request finished.
                return True
            self._update_request_as_session(request, update)
        else:
            # ------【核心逻辑】无后续块则转入等待流式输入状态并递增计数，再重新入队 ------
            request.status = RequestStatus.WAITING_FOR_STREAMING_REQ
            self.num_waiting_for_streaming_input += 1

        self._enqueue_waiting_request(request)
        return False

    def _update_request_with_output(
        self, request: Request, new_token_ids: list[int], is_stale: bool = False
    ) -> tuple[list[int], bool]:
        # is_stale is only used by the AsyncScheduler override.
        # Append generated tokens and check for stop. Note that if
        # a request is still being prefilled, we expect the model runner
        # to return empty token ids for the request.
        stopped = False
        # ------【核心逻辑】逐个追加输出 token 并检查停止，命中停止则裁剪多余 token ------
        for num_new, output_token_id in enumerate(new_token_ids, 1):
            request.append_output_token_ids(output_token_id)

            # Check for stop and update request state.
            # This must be called before we make the EngineCoreOutput.
            stopped = check_stop(request, self.max_model_len)
            if stopped:
                del new_token_ids[num_new:]  # Trim new tokens if needed.
                break
        return new_token_ids, stopped

    def _free_encoder_inputs(self, request: Request) -> None:
        # ------【核心逻辑】取本请求已缓存的编码器输入 ID；为空则直接返回 ------
        cached_encoder_input_ids = self.encoder_cache_manager.get_cached_input_ids(
            request
        )
        # OPTIMIZATION: Avoid list(set) if the set is empty.
        if not cached_encoder_input_ids:
            return

        # Defer the free by the drafter's look-ahead so an entry stays
        # referenced until the drafter's +1 read has also passed it, mirroring
        # the shift the encoder scheduling path applies.
        # ------【投机解码】按草稿模型的 +1 前瞻延迟释放，防止 drafter 再引用已释放条目 ------
        spec_lookahead = 1 if self.use_eagle else 0

        # Here, we use list(set) to avoid modifying the set while iterating
        # over it.
        # ------【核心逻辑】逐个释放已被 decoder KV cache 覆盖的编码器输入 ------
        for input_id in list(cached_encoder_input_ids):
            mm_feature = request.mm_features[input_id]
            start_pos = mm_feature.mm_position.offset
            num_tokens = mm_feature.mm_position.length
            if self.is_encoder_decoder and request.num_computed_tokens > 0:
                # With Whisper, as soon as we've generated a single token,
                # we know we're done with the encoder input. Cross Attention
                # KVs have been calculated and cached already.
                self.encoder_cache_manager.free_encoder_input(request, input_id)
            elif (
                start_pos + num_tokens + spec_lookahead
                <= request.num_computed_tokens - request.num_output_placeholders
            ):
                # Processed, stored in the decoder KV cache, and far enough past
                # the placeholder range (plus the drafter's look-ahead) that no
                # rejection or drafter gather can reference it.
                self.encoder_cache_manager.free_encoder_input(request, input_id)

    def update_draft_token_ids(self, draft_token_ids: DraftTokenIds) -> None:
        # ------【投机解码】把草稿 token 落到对应请求，跳过已结束请求 ------
        for req_id, spec_token_ids in zip(
            draft_token_ids.req_ids,
            draft_token_ids.draft_token_ids,
        ):
            request = self.requests.get(req_id)
            if request is None or request.is_finished():
                # The request may have been finished. Skip.
                continue

            # ------【投机解码 + chunked prefill】分块预填充阶段忽略草稿 token ------
            if request.is_prefill_chunk:
                # Ignore draft tokens for prefill chunks.
                if request.spec_token_ids:
                    request.spec_token_ids = []
                continue

            # ------【投机解码 + 结构化输出/grammar】按需用 grammar 校验草稿 token 后写入请求 ------
            # Add newly generated spec token ids to the request.
            if self.structured_output_manager.should_advance(request):
                metadata = request.structured_output_request
                spec_token_ids = metadata.grammar.validate_tokens(spec_token_ids)  # type: ignore[union-attr]
            request.spec_token_ids = spec_token_ids

    def update_draft_token_ids_in_output(
        self, draft_token_ids: DraftTokenIds, scheduler_output: SchedulerOutput
    ) -> None:
        num_invalid_spec_tokens: dict[str, int] = {}

        # ------【投机解码】裁剪并校验草稿 token，统计被 grammar 判无效的数量 ------
        sched_spec_tokens = scheduler_output.scheduled_spec_decode_tokens
        for req_id, spec_token_ids in zip(
            draft_token_ids.req_ids,
            draft_token_ids.draft_token_ids,
        ):
            request = self.requests.get(req_id)
            if request is None or request.is_finished():
                # The request may have been finished. Skip.
                continue

            placeholder_spec_tokens = sched_spec_tokens.get(req_id)
            if not placeholder_spec_tokens:
                continue

            # ------【投机解码 + chunked prefill】把草稿裁剪到已调度的投机 token 数 ------
            orig_num_spec_tokens = len(placeholder_spec_tokens)
            # Trim drafts to scheduled number of spec tokens
            # (needed for chunked prefill case for example).
            del spec_token_ids[orig_num_spec_tokens:]
            # ------【结构化输出/grammar】过滤不符合语法约束的投机 token，再用 -1 填充并记录 ------
            # Filter out spec tokens which do not adhere to the grammar.
            if self.structured_output_manager.should_advance(request):
                metadata = request.structured_output_request
                spec_token_ids = metadata.grammar.validate_tokens(spec_token_ids)  # type: ignore[union-attr]
            # Pad to original number of spec tokens.
            num_invalid_tokens = orig_num_spec_tokens - len(spec_token_ids)
            if num_invalid_tokens:
                spec_token_ids.extend([-1] * num_invalid_tokens)
                num_invalid_spec_tokens[req_id] = num_invalid_tokens

            sched_spec_tokens[req_id] = spec_token_ids

        scheduler_output.num_invalid_spec_tokens = num_invalid_spec_tokens

    def get_request_counts(self) -> tuple[int, int]:
        """Returns (num_running_reqs, num_waiting_reqs)."""
        # ------【核心逻辑】返回 running 与 waiting(含 skipped) 请求数 ------
        return len(self.running), len(self.waiting) + len(self.skipped_waiting)

    def get_kv_cache_usage(self) -> float:
        """Returns the fraction of the KV cache currently in use (0.0-1.0)."""
        # ------【核心逻辑】返回 KV cache 当前占用比例 ------
        return self.kv_cache_manager.usage


    # 新增一个请求
    def add_request(self, request: Request) -> None:
        # ------【核心逻辑】请求已存在则按流式输入处理：追加更新或开启下一块 ------
        existing = self.requests.get(request.request_id)
        if existing is not None:
            update = StreamingUpdate.from_request(request)
            if existing.status != RequestStatus.WAITING_FOR_STREAMING_REQ:
                assert existing.streaming_queue is not None, "duplicate request id"
                # Queue next input chunk (or finished sentinel).
                existing.streaming_queue.append(update)
            elif update is not None:
                # Commence next input chunk.
                self._update_request_as_session(existing, update)
            else:
                # Streaming-input session finished.
                self.finish_requests(request.request_id, RequestStatus.FINISHED_ABORTED)
        else:
            # ------【核心逻辑】新请求：可恢复则建流式队列，入队并登记，通知 connector ------
            if request.resumable:
                request.streaming_queue = deque()
            self._enqueue_waiting_request(request)
            self.requests[request.request_id] = request
            if self.connector is not None:
                self.connector.on_new_request(request)
            if self.log_stats:
                request.record_event(EngineCoreEventType.QUEUED)


    # 结束一个请求，就是ABORT的指令请求
    def finish_requests(
        self, request_ids: str | Iterable[str] | None, finished_status: RequestStatus
    ) -> list[Request]:
        """Handles the finish signal from outside the scheduler.

        For example, the API server can abort a request when the client
        disconnects.

        If request_ids is None, all requests will be finished.

        Returns:
            List of requests that were aborted. Will not include any that were
            already finished.
        """
        assert RequestStatus.is_finished(finished_status)
        # ------【核心逻辑】把请求 ID 规范化为集合/迭代器，None 表示全部请求 ------
        if isinstance(request_ids, str):
            request_ids = (request_ids,)
        elif request_ids is not None:
            request_ids = set(request_ids)
        else:
            request_ids = self.requests.keys()

        running_requests_to_remove = set()
        waiting_requests_to_remove = []
        valid_requests = []

        # First pass: collect requests to remove from queues
        # ------【核心逻辑】第一遍：收集有效请求并按 running/waiting 分桶 ------
        for req_id in request_ids:
            request = self.requests.get(req_id)
            if request is None or request.is_finished():
                # Invalid request ID.
                continue

            valid_requests.append(request)
            if request.status == RequestStatus.RUNNING:
                running_requests_to_remove.add(request)
            else:
                if request.status == RequestStatus.WAITING_FOR_STREAMING_REQ:
                    self.num_waiting_for_streaming_input -= 1
                waiting_requests_to_remove.append(request)

        # ------【核心逻辑】批量从 running/waiting 队列移除待结束请求 ------
        # Remove all requests from queues at once for better efficiency
        if running_requests_to_remove:
            self.running = remove_all(self.running, running_requests_to_remove)
        if waiting_requests_to_remove:
            self.waiting.remove_requests(waiting_requests_to_remove)
            self.skipped_waiting.remove_requests(waiting_requests_to_remove)

        # ------【核心逻辑】第二遍：置终态并释放请求；远端 KV 等待中的请求延迟释放 block ------
        # Second pass: set status and free requests
        for request in valid_requests:
            delay_free_blocks = False
            if request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
                delay_free_blocks = (
                    request.request_id not in self.finished_recving_kv_req_ids
                )
                self.finished_recving_kv_req_ids.discard(request.request_id)
                self.failed_recving_kv_req_ids.discard(request.request_id)

            request.status = finished_status
            self._free_request(request, delay_free_blocks=delay_free_blocks)

        return valid_requests

    def _free_request(
        self, request: Request, delay_free_blocks: bool = False
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        assert request.is_finished()

        self._inflight_prefills.discard(request)
        # ------【PD 分离】先通知 KV connector 请求结束，拿到是否延迟释放与传输参数 ------
        connector_delay_free_blocks, kv_xfer_params = self._connector_finished(request)

        # EC Connector: mirror the KV hook. The contract requires firing
        # before the encoder cache is freed so the connector can inspect
        # per-request state (e.g. which mm_hashes it recorded during
        # save_caches()) and emit ec_transfer_params for the response body.
        # ------【PD 分离】镜像 EC connector 钩子，在编码器缓存释放前收集 ec_transfer_params ------
        ec_xfer_params: dict[str, Any] | None = None
        if self.ec_connector is not None:
            ec_delay_free, ec_xfer_params = self.ec_connector.request_finished(request)
            connector_delay_free_blocks |= ec_delay_free

        self.encoder_cache_manager.free(request)
        request_id = request.request_id
        # ------【核心逻辑】登记 finished_req_ids 供上层感知结束 ------
        self.finished_req_ids.add(request_id)
        if self.finished_req_ids_dict is not None:
            self.finished_req_ids_dict[request.client_index].add(request_id)

        delay_free_blocks |= connector_delay_free_blocks
        # ------【PD 分离 + 异步 RPC】按 connector 意见决定是否延迟归还 KV block ------
        if not delay_free_blocks:
            self._free_blocks(request)

        return kv_xfer_params, ec_xfer_params

    def _free_blocks(self, request: Request):
        assert request.is_finished()
        # ------【核心逻辑】真正释放 KV block 并从 requests 表删除 ------
        self._free_request_blocks(request)
        del self.requests[request.request_id]

    @property
    def pause_state(self) -> PauseState:
        return self._pause_state

    def set_pause_state(self, pause_state: PauseState) -> None:
        # ------【核心逻辑】设置暂停状态（PAUSED_ALL/PAUSED_NEW 控制调度启停） ------
        self._pause_state = pause_state

    def _free_request_blocks(self, request: Request):
        """Free the request's KV blocks, deferring the return to the block
        pool when an in-flight GPU step may still write them.
        """
        # ------【异步 RPC】无在途写或非延迟释放时立即归还 block，否则暂存延迟释放列表 ------
        if not self.defer_block_free or (
            # Last scheduled step already processed: no in-flight write remains
            # (always the case for a normal finish), so free now.
            request.last_sched_seq <= self.processed_step_seq
        ):
            self.kv_cache_manager.free(request)
            return
        blocks = self.kv_cache_manager.pop_blocks_for_free(request)
        if blocks:
            self.deferred_frees.append((self.sched_step_seq, blocks))

    def _free_cow_retained_blocks(
        self, blocks: list[KVCacheBlock], fence_seq: int
    ) -> None:
        """Release CoW copy retentions, deferring their return to the block
        pool while the step that runs the copy may still be in flight.
        """
        # ------【异步 RPC】释放 CoW 保留块：围栏步已处理则立即归还，否则延迟 ------
        if not self.defer_block_free or fence_seq <= self.processed_step_seq:
            self.kv_cache_manager.block_pool.free_blocks(blocks)
            return
        self.deferred_frees.append((fence_seq, blocks[::-1]))

    def _drain_deferred_frees(self):
        """Return deferred blocks whose fence step has completed.

        Fences are appended in near-monotonic order (a CoW retention fence
        can lead request-free fences by one step), so stop at the first
        pending one; any satisfied entry behind it is merely freed later.
        """
        # ------【异步 RPC】归还围栏步已完成的延迟 block，逆序释放以便先淘汰尾部块 ------
        while self.deferred_frees:
            fence, _ = self.deferred_frees[0]
            if fence > self.processed_step_seq:
                break
            _, blocks = self.deferred_frees.popleft()
            # Free in reverse order so that the tail blocks are evicted first.
            self.kv_cache_manager.block_pool.free_blocks(reversed(blocks))

    def get_num_unfinished_requests(self) -> int:
        # ------【核心逻辑】按暂停状态返回未完成请求数（全暂停为 0，仅新请求暂停只算 running） ------
        if self._pause_state == PauseState.PAUSED_ALL:
            return 0
        if self._pause_state == PauseState.PAUSED_NEW:
            return len(self.running)
        num_waiting = (
            len(self.waiting)
            + len(self.skipped_waiting)
            - self.num_waiting_for_streaming_input
        )
        return num_waiting + len(self.running)

    def has_finished_requests(self) -> bool:
        # ------【PD 分离】有结束请求或 connector 延迟清理的残留请求时返回 True ------
        if self.finished_req_ids:
            return True
        if self.connector is None:
            return False
        # Finished requests waiting on delayed connector cleanup remain in
        # self.requests after they have been removed from scheduling queues.
        num_in_queues = (
            len(self.waiting) + len(self.skipped_waiting) + len(self.running)
        )
        return len(self.requests) > num_in_queues

    def has_requests(self) -> bool:
        # Override the interface default to also keep the engine alive while a
        # connector still has pending push work (e.g. push-mode WRITE transfers
        # in flight after all "live" requests have finished). Without this hook
        # the engine would quiesce before the connector can drain completions.
        # TODO: replace with a more general mechanism for connectors to keep
        # the scheduler alive.
        # ------【PD 分离】除未完成请求外，connector 仍有待推送到远端的工作时也保持引擎存活 ------
        return (
            self.has_unfinished_requests()
            or self.has_finished_requests()
            or (self.connector is not None and self.connector.has_pending_push_work())
            or (
                self.ec_connector is not None
                and self.ec_connector.has_pending_push_work()
            )
        )

    def reset_prefix_cache(
        self, reset_running_requests: bool = False, reset_connector: bool = False
    ) -> bool:
        """Reset the KV prefix cache.

        If reset_running_requests is True, all the running requests will be
        preempted and moved to the waiting queue.
        Otherwise, this method will only reset the KV prefix cache when there
        is no running requests taking KV cache.
        """
        # ------【前缀缓存】需重置 running 请求时逆序抢占，把 KV 块引用数降到 0 以保证重置成功 ------
        if reset_running_requests:
            # For logging.
            timestamp = time.monotonic()
            # Invalidate all the current running requests KV's by pushing them to
            # the waiting queue. In this case, we can reduce the ref count of all
            # the kv blocks to 0 and thus we can make sure the reset is successful.
            # Preempt in reverse order so the requests will be added back to the
            # running queue in FIFO order.
            while self.running:
                request = self.running.pop()
                self._preempt_request(request, timestamp, drop_stale_output=True)

            # ------【前缀缓存】强制同一步抢占+恢复，清空上一步已调度 ID，model runner 会刷出这些请求 ------
            # Clear scheduled request ids cache. Since we are forcing preemption
            # + resumption in the same step, we must act as if these requests were
            # not scheduled in the prior step. They will be flushed from the
            # persistent batch in the model runner.
            self.prev_step_scheduled_req_ids.clear()

        # ------【前缀缓存】执行 KV cache 管理器前缀缓存重置 ------
        reset_successful = self.kv_cache_manager.reset_prefix_cache()
        if reset_running_requests and not reset_successful:
            raise RuntimeError(
                "Failed to reset KV cache even when all the running requests are "
                "preempted and moved to the waiting queue. This is likely due to "
                "the presence of running requests waiting for remote KV transfer, "
                "which is not supported yet."
            )

        # ------【前缀缓存 + PD 分离】可选：同时重置远端 connector 的前缀缓存 ------
        if reset_connector:
            reset_successful = self.reset_connector_cache() and reset_successful

        return reset_successful

    # ══════════════════════════════════════════════════════════════
    # [新增] 公有 — 重置 KV connector 缓存（disaggregated prefill）
    # ══════════════════════════════════════════════════════════════
    def reset_connector_cache(self) -> bool:
        # ------【PD 分离】无 connector 时视为成功返回，避免级联清理误报失败 ------
        if self.connector is None:
            # No connector attached -> nothing to reset, treat as success so
            # callers that unconditionally request a connector reset (e.g. as
            # part of a cache-clearing cascade after a weight update) don't
            # see reset_prefix_cache() flip to False purely because they
            # didn't configure a connector.
            logger.debug(
                "reset_connector requested but no KV connector is configured; "
                "treating as no-op success."
            )
            return True

        # ------【PD 分离】调用 connector 重置远端前缀缓存，失败则返回 False ------
        if self.connector.reset_cache() is False:
            return False

        # ------【PD 分离】记录 connector 前缀缓存被重置，供统计上报 ------
        if self.log_stats:
            assert self.connector_prefix_cache_stats is not None
            self.connector_prefix_cache_stats.reset = True

        return True

    def reset_encoder_cache(self) -> None:
        """Reset the encoder cache to invalidate all cached encoder outputs.

        This should be called when model weights are updated to ensure
        stale vision embeddings are not reused.
        """
        # ------【核心逻辑】重置编码器缓存，权重更新后使失效的视觉 embedding 不再复用 ------
        self.encoder_cache_manager.reset()

    def make_stats(
        self,
        spec_decoding_stats: SpecDecodingStats | None = None,
        kv_connector_stats: KVConnectorStats | None = None,
        cudagraph_stats: CUDAGraphStat | None = None,
        perf_stats: PerfStats | None = None,
    ) -> SchedulerStats | None:
        # ------【核心逻辑】未开日志统计则直接返回 None ------
        if not self.log_stats:
            return None
        prefix_cache_stats = self.kv_cache_manager.make_prefix_cache_stats()
        assert prefix_cache_stats is not None
        # ------【前缀缓存】取 connector 侧前缀缓存统计（用后重置） ------
        connector_prefix_cache_stats: PrefixCacheStats | None = None
        if self.connector_prefix_cache_stats is not None:
            connector_prefix_cache_stats = self.connector_prefix_cache_stats
            self.connector_prefix_cache_stats = PrefixCacheStats()
        # ------【核心逻辑】排空 KV 淘汰事件用于观测 ------
        eviction_events = (
            self.kv_metrics_collector.drain_events()
            if self.kv_metrics_collector is not None
            else []
        )
        spec_stats = spec_decoding_stats
        connector_stats_payload = (
            kv_connector_stats.data if kv_connector_stats else None
        )
        # ------【核心逻辑】汇总 running/waiting 数量、缓存占用及各类子统计为 SchedulerStats ------
        return SchedulerStats(
            num_running_reqs=len(self.running),
            num_waiting_reqs=len(self.waiting),
            num_skipped_waiting_reqs=len(self.skipped_waiting),
            kv_cache_usage=self.kv_cache_manager.usage,
            prefix_cache_stats=prefix_cache_stats,
            connector_prefix_cache_stats=connector_prefix_cache_stats,
            kv_cache_eviction_events=eviction_events,
            spec_decoding_stats=spec_stats,
            kv_connector_stats=connector_stats_payload,
            cudagraph_stats=cudagraph_stats,
            perf_stats=perf_stats,
        )

    # ══════════════════════════════════════════════════════════════
    # [新增] 公有 — 生成投机解码统计
    # ══════════════════════════════════════════════════════════════
    def make_spec_decoding_stats(
        self,
        spec_decoding_stats: SpecDecodingStats | None,
        num_draft_tokens: int,
        num_accepted_tokens: int,
        num_invalid_spec_tokens: dict[str, int] | None,
        request_id: str,
    ) -> SpecDecodingStats | None:
        # ------【投机解码】未开统计或无草稿 token 时返回 None ------
        if not self.log_stats or not num_draft_tokens:
            return None
        # ------【投机解码】初始化统计并扣除 grammar 无效草稿后记录草稿/接受数 ------
        if spec_decoding_stats is None:
            spec_decoding_stats = SpecDecodingStats.new(self.num_spec_tokens)
        if num_invalid_spec_tokens:
            num_draft_tokens -= num_invalid_spec_tokens.get(request_id, 0)
        spec_decoding_stats.observe_draft(
            num_draft_tokens=num_draft_tokens, num_accepted_tokens=num_accepted_tokens
        )
        return spec_decoding_stats

    def shutdown(self) -> None:
        # ------【PD 分离】关闭 KV 事件发布器与 KV/EC connector，优雅停机 ------
        logger.debug_once("[shutdown] Scheduler: start")
        if self.kv_event_publisher:
            self.kv_event_publisher.shutdown()
        if self.connector is not None:
            self.connector.shutdown()

        if self.ec_connector is not None:
            self.ec_connector.shutdown()

        logger.debug_once("[shutdown] Scheduler: complete")

    ########################################################################
    # KV Connector Related Methods
    ########################################################################

    def get_kv_connector(self) -> KVConnectorBase_V1 | None:
        # ------【PD 分离】返回 KV connector 实例 ------
        return self.connector

    def get_ec_connector(self) -> ECConnectorBase | None:
        # ------【PD 分离】返回 EC connector 实例 ------
        return self.ec_connector

    def get_kv_event_publisher_config(self) -> KVEventsConfig | None:
        # ------【PD 分离】返回 KV 事件发布器配置 ------
        return self.kv_event_publisher.get_publisher_config()

    def _connector_finished(
        self, request: Request
    ) -> tuple[bool, dict[str, Any] | None]:
        """
        Invoke the KV connector request_finished() method if applicable.

        Returns optional kv transfer parameters to be included with the
        request outputs.
        """
        if self.connector is None:
            return False, None

        # ------【PD 分离】把窗口外的前缀块先移除，再把块表交给 connector 处理 ------
        # Free any out-of-window prefix blocks before we hand the block table to
        # the connector, on the processed-token basis (see `allocate_slots`).
        self.kv_cache_manager.remove_skipped_blocks(
            request_id=request.request_id,
            processed_computed_tokens=max(
                0, request.num_computed_tokens - request.num_in_flight_tokens
            ),
            num_prompt_tokens=request.num_prompt_tokens,
        )

        # ------【PD 分离】按已算 token 数取出块 ID 列表供远端传输 ------
        block_ids = self.kv_cache_manager.get_block_ids_for_computed_tokens(
            request_id=request.request_id,
            num_computed_tokens=request.num_computed_tokens,
        )

        # ------【PD 分离】按是否支持 HMA 选择单组/全组的 connector 结束回调 ------
        if not isinstance(self.connector, SupportsHMA):
            # NOTE(Kuntai): We should deprecate this code path after we enforce
            # all connectors to support HMA.
            # Hybrid memory allocator should be already turned off for this
            # code path, but let's double-check here.
            assert len(self.kv_cache_config.kv_cache_groups) == 1
            return self.connector.request_finished(request, block_ids[0])

        return self.connector.request_finished_all_groups(request, block_ids)

    def _request_remaining_blocks(self, request: Request) -> int:
        """Blocks `request` still needs to allocate to hold its full sequence."""
        # ------【核心逻辑】计算请求完整序列还需分配的 block 数（含准入上限） ------
        full_num_tokens = min(request.num_tokens, self.max_model_len)
        return self.kv_cache_manager.coordinator.get_num_blocks_to_allocate(
            request_id=request.request_id,
            num_tokens=full_num_tokens,
            new_computed_blocks=self.kv_cache_manager.empty_kv_cache_blocks.blocks,
            num_encoder_tokens=0,
            total_computed_tokens=request.num_computed_tokens,
            num_local_computed_tokens=request.num_computed_tokens,
            num_tokens_main_model=full_num_tokens,
            apply_admission_cap=True,
        )

    def _inflight_prefill_reserved_blocks(self) -> int:
        """Num blocks in-flight prefills still need to finish (their reservation)."""
        # ------【核心逻辑】汇总所有在途 prefill 请求仍需预留的 block 数 ------
        return sum(
            self._request_remaining_blocks(req) for req in self._inflight_prefills
        )

    def _update_waiting_for_remote_kv(self, request: Request) -> None:
        """
        KV Connector: update request state after async recv is finished.

        When the kv transfer is ready, we cache the blocks
        and the request state will be moved back to WAITING from
        WAITING_FOR_REMOTE_KV.
        """
        assert self.connector is not None

        # ------【PD 分离】远端 KV 加载失败：缓存有效前缀并记录需清零的失效块，或全释放 ------
        if request.request_id in self.failed_recving_kv_req_ids:
            # Request had KV load failures; num_computed_tokens was already
            # updated in _update_requests_with_invalid_blocks
            if request.num_computed_tokens:
                # Cache any valid computed tokens.
                self.kv_cache_manager.cache_blocks(request, request.num_computed_tokens)
                if self.needs_kv_cache_zeroing:
                    # The failed load left the blocks beyond the valid
                    # prefix unwritten and their zeroing was skipped; zero
                    # them before they are recomputed locally.
                    self.kv_cache_manager.record_blocks_for_zeroing(
                        request.request_id, request.num_computed_tokens
                    )
            else:
                # No valid computed tokens, release allocated blocks.
                # There may be a local cache hit on retry.
                # (Freed blocks are re-recorded for zeroing when
                # reallocated, so the skipped blocks need no handling.)
                self.kv_cache_manager.free(request)

            self.failed_recving_kv_req_ids.remove(request.request_id)
        else:
            # ------【PD 分离】远端 KV 就绪后真正缓存 block；满 prompt 命中时重算最后一个 token ------
            # Now that the blocks are ready, actually cache them.
            # This will cache the blocks iff caching is enabled.
            self.kv_cache_manager.cache_blocks(request, request.num_computed_tokens)

            # on a full prompt hit, we need to re-compute the last token
            # in order to be able to sample the next token
            if request.num_computed_tokens == request.num_tokens:
                request.num_computed_tokens = request.num_tokens - 1

        # ------【PD 分离】无论成败都从“已收完”集合移除该请求 ------
        self.finished_recving_kv_req_ids.remove(request.request_id)

    def _try_promote_blocked_waiting_request(self, request: Request) -> bool:
        """
        Try to promote a blocked waiting request back to schedulable states.
        """
        # ------【PD 分离】远端 KV 收完后把请求从阻塞等待提升回可调度状态 ------
        if request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
            # finished_recving_kv_req_ids is populated during
            # update_from_output(), based on worker-side connector signals
            # in KVConnectorOutput.finished_recving
            if request.request_id not in self.finished_recving_kv_req_ids:
                return False
            self._update_waiting_for_remote_kv(request)
            if request.num_preemptions:
                request.status = RequestStatus.PREEMPTED
            else:
                request.status = RequestStatus.WAITING
            return True

        # ------【结构化输出/grammar】语法编译完成后提升请求；编译异常则登记错误 ------
        if request.status == RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR:
            structured_output_req = request.structured_output_request
            if not structured_output_req or structured_output_req.grammar is None:
                return False
            if isinstance(structured_output_req.grammar, Exception):
                self.grammar_compile_error_reqs.add(request.request_id)
                return False
            request.status = RequestStatus.WAITING
            return True

        # ------【核心逻辑】流式输入等待状态暂不提升，等待后续块到达 ------
        if request.status == RequestStatus.WAITING_FOR_STREAMING_REQ:
            assert not request.streaming_queue
            return False

        raise AssertionError(
            "Unexpected blocked waiting status in promotion: "
            f"{request.status.name} for request {request.request_id}"
        )

    def _update_from_kv_xfer_finished(self, kv_connector_output: KVConnectorOutput):
        """
        KV Connector: update the scheduler state based on the output.

        The Worker side connectors add finished_recving and
        finished_sending reqs to the output.
        * if finished_sending: free the blocks
        # if finished_recving: add to state so we can
            schedule the request during the next step.
        """

        # ------【PD 分离】把 worker 侧 connector 输出同步回调度器侧 connector ------
        if self.connector is not None:
            self.connector.update_connector_output(kv_connector_output)

        # KV Connector:: update recv and send status from last step.
        # ------【PD 分离】处理接收完成：等待远端 KV 的记入集合，已结束的释放 block ------
        for req_id in kv_connector_output.finished_recving or ():
            logger.debug("Finished recving KV transfer for request %s", req_id)
            assert req_id in self.requests
            req = self.requests[req_id]
            if req.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
                self.finished_recving_kv_req_ids.add(req_id)
            else:
                assert RequestStatus.is_finished(req.status)
                self._free_blocks(self.requests[req_id])
        # ------【PD 分离】处理发送完成：释放对应请求的 KV block ------
        for req_id in kv_connector_output.finished_sending or ():
            logger.debug("Finished sending KV transfer for request %s", req_id)
            assert req_id in self.requests
            self._free_blocks(self.requests[req_id])

    def _update_requests_with_invalid_blocks(
        self,
        requests: Iterable[Request],
        invalid_block_ids: set[int],
        num_scheduled_tokens: dict[str, int],
        evict_blocks: bool = True,
    ) -> tuple[set[str], int, set[int]]:
        """
        Identify and update requests affected by invalid KV cache blocks.

        This method scans the given requests, detects those with invalid blocks
        and adjusts their `num_computed_tokens` to the longest valid prefix.
        For observability, it also accumulates the total number of tokens that
        will need to be recomputed across all affected requests.

        Args:
            requests: The set of requests to scan for invalid blocks.
            invalid_block_ids: IDs of invalid blocks.
            num_scheduled_tokens: req_id -> number of scheduled tokens.
            evict_blocks: Whether to collect blocks for eviction (False for
                async requests which aren't cached yet).

        Returns:
            tuple:
                - affected_req_ids (set[str]): IDs of requests impacted by
                invalid blocks.
                - total_affected_tokens (int): Total number of tokens that must
                be recomputed across all affected requests.
                - blocks_to_evict (set[int]): Block IDs to evict from cache,
                including invalid blocks and downstream dependent blocks.
        """
        # ------【PD 分离】初始化受影响请求/受影响 token 数/待淘汰块集合 ------
        affected_req_ids: set[str] = set()
        total_affected_tokens = 0
        blocks_to_evict: set[int] = set()
        # If a block is invalid and shared by multiple requests in the batch,
        # these requests must be rescheduled, but only the first will recompute
        # it. This set tracks blocks already marked for recomputation.
        marked_invalid_block_ids: set[int] = set()
        # ------【PD 分离】逐请求扫描其可能含外部 token 的 block，找出失效块 ------
        for request in requests:
            is_affected = False
            marked_invalid_block = False
            req_id = request.request_id
            # TODO (davidb): add support for hybrid memory allocator
            (req_block_ids,) = self.kv_cache_manager.get_block_ids(req_id)
            # We iterate only over blocks that may contain externally computed
            # tokens
            req_num_computed_tokens = (
                request.num_computed_tokens - num_scheduled_tokens.get(req_id, 0)
            )

            req_num_computed_blocks = (
                req_num_computed_tokens + self.block_size - 1
            ) // self.block_size
            # ------【PD 分离】按块遍历，命中失效块则截断已算 token 到最长有效前缀 ------
            for idx, block_id in zip(range(req_num_computed_blocks), req_block_ids):
                if block_id not in invalid_block_ids:
                    continue

                is_affected = True

                if block_id in marked_invalid_block_ids:
                    # This invalid block is shared with a previous request
                    # and was already marked for recomputation.
                    # This means this request can still consider this block
                    # as computed when rescheduled.
                    # Currently this only applies to sync loading; Async
                    # loading does not yet support block sharing
                    continue

                marked_invalid_block_ids.add(block_id)

                if marked_invalid_block:
                    # This request has already marked an invalid block for
                    # recomputation and updated its num_computed_tokens.
                    continue

                marked_invalid_block = True
                # Truncate the computed tokens at the first failed block
                request.num_computed_tokens = idx * self.block_size
                num_affected_tokens = (
                    req_num_computed_tokens - request.num_computed_tokens
                )
                total_affected_tokens += num_affected_tokens

                # collect invalid block and all downstream dependent blocks
                # ------【PD 分离】收集失效块及其下游依赖块用于淘汰 ------
                if evict_blocks:
                    blocks_to_evict.update(req_block_ids[idx:])

            if is_affected:
                # ------【PD 分离】所有失效块都被前序请求共享重算时，回退到仅缓存 token 视为已算 ------
                if not marked_invalid_block:
                    # All invalid blocks of this request are shared with
                    # previous requests and will be recomputed by them.
                    # Revert to considering only cached tokens as computed.
                    # Currently this only applies to sync loading; Async
                    # loading does not yet support block sharing
                    total_affected_tokens += (
                        request.num_computed_tokens - req_num_computed_tokens
                    )
                    request.num_computed_tokens = req_num_computed_tokens

                affected_req_ids.add(request.request_id)

        return affected_req_ids, total_affected_tokens, blocks_to_evict

    def _handle_invalid_blocks(
        self, invalid_block_ids: set[int], num_scheduled_tokens: dict[str, int]
    ) -> set[str]:
        """
        Handle requests affected by invalid KV cache blocks.

        Returns:
            Set of affected request IDs to skip in update_from_output main loop.
        """
        # ------【PD 分离】按失败策略决定：不重算则失败结束请求 ------
        should_fail = not self.recompute_kv_load_failures

        # handle async KV loads (not cached yet, evict_blocks=False)
        # ------【PD 分离】处理异步 KV 加载失败的请求（尚未缓存，不淘汰块） ------
        async_load_reqs = (
            req
            for req in self.skipped_waiting
            if req.status == RequestStatus.WAITING_FOR_REMOTE_KVS
        )
        async_failed_req_ids, num_failed_tokens, _ = (
            self._update_requests_with_invalid_blocks(
                async_load_reqs,
                invalid_block_ids,
                num_scheduled_tokens,
                evict_blocks=False,
            )
        )

        total_failed_requests = len(async_failed_req_ids)
        total_failed_tokens = num_failed_tokens

        # handle sync loads (may be cached, collect blocks for eviction)
        # ------【PD 分离】处理同步 KV 加载失败的请求（可能已缓存，收集块待淘汰） ------
        sync_failed_req_ids, num_failed_tokens, sync_blocks_to_evict = (
            self._update_requests_with_invalid_blocks(
                self.running, invalid_block_ids, num_scheduled_tokens, evict_blocks=True
            )
        )

        total_failed_requests += len(sync_failed_req_ids)
        total_failed_tokens += num_failed_tokens

        # ------【核心逻辑】无失败请求则直接返回空集合 ------
        if not total_failed_requests:
            return set()

        # evict invalid blocks and downstream dependent blocks from cache
        # only when not using recompute policy (where blocks will be recomputed
        # and reused by other requests sharing them)
        # ------【PD 分离】非重算策略下淘汰失效块及其下游依赖块 ------
        if sync_blocks_to_evict and not self.recompute_kv_load_failures:
            self.kv_cache_manager.evict_blocks(sync_blocks_to_evict)

        # ------【PD 分离】失败策略：汇总异步/同步失败请求并报错返回 ------
        if should_fail:
            all_failed_req_ids = async_failed_req_ids | sync_failed_req_ids
            logger.error(
                "Failing %d request(s) due to KV load failure "
                "(failure_policy=fail, %d tokens affected). Request IDs: %s",
                total_failed_requests,
                total_failed_tokens,
                all_failed_req_ids,
            )
            return all_failed_req_ids

        logger.warning(
            "Recovered from KV load failure: "
            "%d request(s) rescheduled (%d tokens affected).",
            total_failed_requests,
            total_failed_tokens,
        )

        # ------【PD 分离】重算策略：记录异步失败请求待重试，返回同步失败 ID 供跳过 ------
        # Mark async requests with KV load failures for retry once loading completes
        self.failed_recving_kv_req_ids |= async_failed_req_ids
        # Return sync affected IDs to skip in update_from_output
        return sync_failed_req_ids
