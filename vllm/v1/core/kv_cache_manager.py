# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import itertools
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, overload

from vllm.distributed.kv_events import BlockStored, KVCacheEvent
from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv
from vllm.v1.core.kv_cache_coordinator import (
    HybridKVCacheCoordinator,
    get_kv_cache_coordinator,
)
from vllm.v1.core.kv_cache_metrics import KVCacheMetricsCollector
from vllm.v1.core.kv_cache_utils import KVCacheBlock, KVCacheBlockCopy
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    CrossAttentionSpec,
    EncoderOnlyAttentionSpec,
    KVCacheConfig,
    get_kv_cache_spec_kind,
    get_kv_cache_spec_sliding_window,
)
from vllm.v1.metrics.stats import PrefixCacheStats
from vllm.v1.request import Request, RequestStatus

logger = init_logger(__name__)


@dataclass
class KVCacheBlocks:
    """
    The allocation result of KVCacheManager, work as the interface between
    Scheduler and KVCacheManager, to hide KVCacheManager's internal data
    structure from the Scheduler.
    """

    blocks: tuple[Sequence[KVCacheBlock], ...]
    """
    `blocks[i][j]` refers to the i-th kv_cache_group
    and the j-th block of tokens.We don't use block of
    tokens as the outer dimension because it assumes all
    kv_cache_groups have the same number of blocks, which is true for now but
    will be broken if we want to give different block_size to different
    kv_cache_groups in the future.

    Each single type KVCacheBlocks could be represented as:
    - list[KVCacheBlock] for more than one KVCacheBlock
    - an empty tuple for requests without KVCacheBlock
      (a precomputed KVCacheBlocks is in KVCacheManager to avoid GC overhead)
    """

    def __add__(self, other: "KVCacheBlocks") -> "KVCacheBlocks":
        """Adds two KVCacheBlocks instances."""
        # ------【核心逻辑】按组逐 block 拼接两个分配结果，供上层累加不同来源的 block ------
        return KVCacheBlocks(
            tuple(
                list(itertools.chain(blk1, blk2))
                for blk1, blk2 in zip(self.blocks, other.blocks)
            )
        )

    @overload
    def get_block_ids(
        self,
        allow_none: Literal[False] = False,
    ) -> tuple[list[int], ...]: ...

    @overload
    def get_block_ids(
        self,
        allow_none: Literal[True] = True,
    ) -> tuple[list[int], ...] | None: ...

    def get_block_ids(
        self,
        allow_none: bool = False,
    ) -> tuple[list[int], ...] | None:
        """
        Converts the KVCacheBlocks instance to block_ids.

        Returns:
            tuple[list[int], ...]: A tuple of lists where:
                - the outer tuple corresponds to KV cache groups
                - each inner list contains the block_ids of the blocks in that
                  group
        """
        # ------【核心逻辑】把 KVCacheBlock 对象序列转成 block_id，按 kv_cache_group 分组返回 ------
        if allow_none and all(len(group) == 0 for group in self.blocks):
            return None
        return tuple([blk.block_id for blk in group] for group in self.blocks)

    def get_unhashed_block_ids(self) -> list[int]:
        """Get block_ids of unhashed blocks from KVCacheBlocks instance."""
        # ------【前缀缓存】筛出尚未登记哈希的 block，供后续计算哈希并写入 prefix cache ------
        assert len(self.blocks) == 1, "Only one group is supported"
        return [block.block_id for block in self.blocks[0] if block.block_hash is None]

    def get_unhashed_block_ids_all_groups(self) -> list[list[int]]:
        """Get block_ids of unhashed blocks from KVCacheBlocks instance."""
        # ------【前缀缓存】遍历所有 kv_cache_group 收集未哈希且非填充的 block，供批量写缓存 ------
        # Skip padding blocks.
        return [
            [
                block.block_id
                for block in group
                if block.block_hash is None and not block.is_null
            ]
            for group in self.blocks
        ]

    def new_empty(self) -> "KVCacheBlocks":
        """
        Creates a new KVCacheBlocks instance with no blocks.
        """
        # ------【核心逻辑】构造与当前组数一致的空分配结果，避免重复分配新对象 ------
        return KVCacheBlocks(tuple(() for _ in range(len(self.blocks))))


class KVCacheManager:
    """
    === 类说明 ===
        继承: object
        职责: KV Cache 抽象管理器。封装了物理 block 池 (Block Pool) + 前缀缓存匹配
              + block 分配/释放/引用计数。Scheduler 通过三个核心 API 与它交互：
              ① get_computed_blocks() — 查前缀命中
              ② allocate_slots()      — 分配新 block
              ③ free()                — 释放 block
              调度器不直接操作物理 block，所有 KV cache 账本逻辑由本类统一管理。

    === 公有方法 (22个) ===
        —— 核心 API (Scheduler 直接调用) ——
            usage                    — @property: KV cache 使用率 (0.0-1.0)
            get_computed_blocks()    — 查本地 prefix cache 命中，返回可复用的 block
            allocate_slots()         — 为请求分配新的 KV cache block（核心分配逻辑）
            free()                   — 释放请求的所有 block，引用计数 -1
        —— 前缀缓存管理 ——
            prefix_cache_lookup_enabled() — 该请求是否启用 prefix cache 查找
            record_prefix_cache_stats()   — 记录 prefix cache 命中统计
            make_prefix_cache_stats()     — 获取并重置 prefix cache 统计
            reset_prefix_cache()          — 重置整个 prefix cache（模型热更新时）
            cache_blocks()                — 将 block 注册到 prefix cache（按 hash 索引）
            get_num_common_prefix_blocks() — 返回与运行中请求的公共前缀 block 数
        —— Block 生命周期 ——
            get_blocks()              — 获取请求当前的 block 分配情况
            get_block_ids()           — 获取请求的 block ID 列表
            get_block_ids_for_computed_tokens() — 获取已计算 token 对应的 block ID
            remove_skipped_blocks()   — 释放超出滑动窗口范围的旧 block
            pop_blocks_for_free()     — 从 coordinator 取出待释放的 block 列表
            evict_blocks()            — 从 prefix cache 强制驱逐指定 block
            estimate_cached_tokens()  — 估算新请求能命中多少 token 的 prefix cache
        —— 辅助 ——
            create_kv_cache_blocks()  — 构造 KVCacheBlocks 对象（复用空对象减少 GC）
            truncate_computed_blocks() — 截断命中 block（用于 connector 场景）
            get_zeroing_block_ids_in_range() — 获取需要清零的 block ID（异步 KV 加载用）
            take_events()             — 获取 KV cache 事件（监控用）
            get_computed_blocks_for_connector() — P/D 分离场景的 prefix cache 查找

    === 核心成员属性 ===
        —— Block 池 ——
            coordinator: KVCacheCoordinator  — 物理 block 协调器（单 group 或混合架构）
            block_pool: BlockPool            — 物理 block 池（分配/释放/引用计数/COW）
            num_kv_cache_groups: int         — KV cache group 数量（decoder-only=1）
            kv_cache_config: KVCacheConfig   — KV cache 配置
        —— Prefix 缓存 ——
            enable_caching: bool             — 是否启用 prefix caching
            prefix_cache_stats               — prefix cache 命中统计（log_stats=True 时）
            empty_kv_cache_blocks            — 预构造的空 KVCacheBlocks（避免 GC 开销）
        —— 容量控制 ——
            max_model_len: int               — 模型最大上下文长度
            watermark_blocks: int            — 保留的最小空闲 block 数（避免频繁抢占）
        —— 辅助 ——
            use_eagle: bool                  — EAGLE 投机解码模式
            log_stats: bool                  — 是否记录统计日志
            enable_kv_cache_events: bool     — 是否发布 KV cache 事件
            metrics_collector                — KV cache 指标收集器
            kv_cache_event_metadata          — KV cache 事件元数据
    """
    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        max_model_len: int,
        scheduler_block_size: int,
        hash_block_size: int,
        max_in_flight_tokens: int | None = None,
        enable_caching: bool = True,
        use_eagle: bool = False,
        log_stats: bool = False,
        enable_kv_cache_events: bool = False,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
        metrics_collector: KVCacheMetricsCollector | None = None,
        watermark: float = 0.0,
    ) -> None:
        # ------【核心逻辑】记录最大上下文长度，未显式给定在途 token 上限时回退到该值 ------
        self.max_model_len = max_model_len
        # When unset, fall back to `max_model_len` so the recycling-aware cap
        # collapses to the prior (uncapped) admission behavior. The scheduler
        # always supplies the real value at runtime.
        if max_in_flight_tokens is None:
            max_in_flight_tokens = max_model_len

        # ------【核心逻辑】缓存开关/事件/投机解码/统计等运行标志与指标收集器注入 ------
        self.enable_caching = enable_caching
        self.enable_kv_cache_events = enable_kv_cache_events
        self.use_eagle = use_eagle
        self.log_stats = log_stats
        self.metrics_collector = metrics_collector
        # FIXME: make prefix cache stats conditional on log_stats. We still need
        # this comment because when the log stats is enabled there are still
        # potential configs we could expose in the future.
        # ------【前缀缓存】仅在开启统计时构造 prefix cache 命中统计对象，省内存 ------
        self.prefix_cache_stats = PrefixCacheStats() if log_stats else None

        # ------【核心逻辑】工厂方法按配置创建单组/混合 KV 协调器，并取出物理 block 池句柄 ------
        self.coordinator = get_kv_cache_coordinator(
            kv_cache_config=kv_cache_config,
            max_model_len=self.max_model_len,
            max_in_flight_tokens=max_in_flight_tokens,
            use_eagle=self.use_eagle,
            enable_caching=self.enable_caching,
            enable_kv_cache_events=enable_kv_cache_events,
            dcp_world_size=dcp_world_size,
            pcp_world_size=pcp_world_size,
            scheduler_block_size=scheduler_block_size,
            hash_block_size=hash_block_size,
            metrics_collector=self.metrics_collector,
        )
        self.num_kv_cache_groups = len(kv_cache_config.kv_cache_groups)
        self.block_pool = self.coordinator.block_pool 
        self.kv_cache_config = kv_cache_config

        # Watermark: minimum number of KV cache blocks to keep free when
        # admitting waiting/preempted requests, to avoid frequent preemptions.
        # ------【核心逻辑】watermark 按比例折算为保留的最小空闲 block 数，防频繁抢占 ------
        assert watermark >= 0.0, "watermark must be non-negative"
        self.watermark_blocks = int(watermark * kv_cache_config.num_blocks)
        # ------【核心逻辑】为每个 kv_cache_group 预取 (类型, 滑动窗口) 元数据供事件标注 ------
        self.kv_cache_event_metadata = tuple(
            (
                get_kv_cache_spec_kind(group.kv_cache_spec).value,
                get_kv_cache_spec_sliding_window(group.kv_cache_spec),
            )
            for group in kv_cache_config.kv_cache_groups
        )

        # Pre-constructed KVCacheBlocks with no blocks, callers should use this
        # via create_kv_cache_blocks instead of creating new ones to avoid GC
        # overhead.
        #
        # We use nested tuples to ensure the empty KVCacheBlocks is immutable.
        # ------【核心逻辑】预构造不可变的空 KVCacheBlocks，复用对象以减少 GC 开销 ------
        self.empty_kv_cache_blocks = KVCacheBlocks(
            tuple(() for _ in range(self.num_kv_cache_groups))
        )

        # ------【PD 分离】记录已交给 connector 离线搬运的 COW block 的 pin，释放时一并回收 ------
        # Off-table cow blocks handed to a KV connector for partial-tail
        # offload; pinned until the request's blocks are freed.
        self._partial_tail_pins: dict[str, list[KVCacheBlock]] = {}

    @property
    def usage(self) -> float:
        """Get the KV cache usage.

        Returns:
            The KV cache usage (between 0.0 and 1.0).
        """
        # ------【核心逻辑】直接透传底层 block 池的使用率 (0.0~1.0) ------
        return self.block_pool.get_usage()

    def make_prefix_cache_stats(self) -> PrefixCacheStats | None:
        """Get (and reset) the prefix cache stats.

        Returns:
            The current prefix caching stats, or None if logging is disabled.
        """
        # ------【前缀缓存】取出当前命中统计并重置为空对象，供日志侧一次性消费 ------
        if not self.log_stats:
            return None
        stats = self.prefix_cache_stats
        self.prefix_cache_stats = PrefixCacheStats()
        return stats

    def prefix_cache_lookup_enabled(self, request: Request) -> bool:
        """Whether a local prefix cache lookup may be run for this request."""
        # ------【前缀缓存】请求需同时满足全局缓存开启且未被标记跳过读缓存才允许查前缀 ------
        return self.enable_caching and not request.skip_reading_prefix_cache

    def record_prefix_cache_stats(self, request: Request, num_hits: int) -> None:
        # ------【前缀缓存】命中统计按请求记录命中 token 数及是否被抢占 ------
        # Don't count a request that skipped the cache lookup.
        if not self.log_stats or not self.prefix_cache_lookup_enabled(request):
            return
        assert self.prefix_cache_stats is not None
        self.prefix_cache_stats.record(
            num_tokens=request.num_tokens,
            num_hits=num_hits,
            preempted=request.num_preemptions > 0,
        )

    def get_computed_blocks(self, request: Request) -> tuple[KVCacheBlocks, int, int]:
        """Get the computed (cached) blocks for the request.
        Note that the computed blocks must be full.

        Args:
            request: The request to get the computed blocks.

        Returns:
            A tuple containing:
                - A list of blocks that are computed for the request.
                - The number of computed tokens.
                - ``shared_prefix_boundary``: the block-aligned token position of
                  a shared prefix that a sparse-retention group (Mamba / sliding
                  window) has not cached yet (Marconi-style APC), or 0 if none.
                  Pinned so ``VLLM_PREFIX_CACHE_RETENTION_INTERVAL`` does not drop
                  the junction and defeat cross-request reuse.
        """
        # ------【前缀缓存】缓存关闭或请求跳过读缓存时直接返回空命中 ------
        # We skip finding the prefix cache hit when prefix caching is
        # disabled or the request is marked as skipping kv cache read
        # (which happens when the request requires prompt logprobs
        # or calls a pooling model with all pooling).
        if not self.prefix_cache_lookup_enabled(request):
            return self.empty_kv_cache_blocks, 0, 0

        # NOTE: When all tokens hit the cache, we must recompute the last token
        # to obtain logits. Thus, set max_cache_hit_length to prompt_length - 1.
        # This can trigger recomputation of an entire block, rather than just
        # the single last token, because allocate_slots() requires
        # num_computed_tokens to be block-size aligned. Removing this limitation
        # could slightly improve performance in the future.
        # ------【前缀缓存】以 prompt 长度-1 为上限查最长公共前缀，留最后一 token 算 logits ------
        max_cache_hit_length = request.num_tokens - 1
        computed_blocks, num_new_computed_tokens, num_uncached = (
            self.coordinator.find_longest_cache_hit(
                request.block_hashes, max_cache_hit_length
            )
        )

        # ------【异步 RPC+前缀缓存】full 上报模式时把复用块作为 BlockStored 事件推给外部消费方 ------
        # When kv_cache_report_mode is "full", emit BlockStored events
        # for the reused prefix cache blocks so that external consumers
        # (e.g. gateway) can learn about them.
        if (
            num_new_computed_tokens > 0
            and self.enable_kv_cache_events
            and getattr(request, "kv_cache_report_mode", "incremental") == "full"
        ):
            for group_idx, group_blocks in enumerate(computed_blocks):
                num_blocks = len(group_blocks)
                if num_blocks > 0:
                    group = self.kv_cache_config.kv_cache_groups[group_idx]
                    block_size = group.kv_cache_spec.block_size
                    self.block_pool.emit_cached_block_events(
                        request,
                        num_blocks,
                        block_size,
                        group_idx,
                    )

        # The junction to pin is where the lagging sparse-retention group stops
        # (``num_new_computed_tokens``) plus the uncached shared prefix -- i.e.
        # the longest single-group hit. Sub-block gaps are left to the mask,
        # which floors to the alignment boundary (a no-op there).
        # ------【前缀缓存】稀疏保留组的共享前缀边界=已命中 token + 未缓存的公共前缀 ------
        shared_prefix_boundary = (
            num_new_computed_tokens + num_uncached if num_uncached else 0
        )

        # ------【核心逻辑】把命中 block 包成 KVCacheBlocks 与命中 token 数、边界一起返回 ------
        blocks = self.create_kv_cache_blocks(computed_blocks)
        return blocks, num_new_computed_tokens, shared_prefix_boundary

    def get_computed_blocks_for_connector(
        self, request: Request
    ) -> tuple[KVCacheBlocks, int, int, bool]:
        """Local prefix-cache lookup for a request scheduled with a KV connector.

        Hybrid (Mamba + full-attention) models can have per-group prefix hits
        diverge under block pressure: the full-attention tail may be evicted
        while a deeper Mamba state survives, or vice versa. Report the
        full-attention hit as the local prefix - the connector transfers the
        remaining suffix and the Mamba state is transferred unconditionally by
        nixl's ``_apply_prefix_caching`` - and flag when that hit ran deeper
        than a lagging group. Such a hit only has a valid Mamba state at its
        boundary if the connector supplies it, so the caller must fall back to
        ``get_computed_blocks`` to reconcile when no external tokens are found.

        Non-hybrid models and already-convergent hits use ``get_computed_blocks``.

        Returns:
            The ``get_computed_blocks`` triple (blocks, number of local computed
            tokens, shared-prefix boundary) plus ``hit_diverged``.
        """
        # ------【PD 分离】非 Mamba+全注意混合模型直接走标准前缀查找，命中未发散 ------
        coordinator = self.coordinator
        if not (
            self.kv_cache_config.has_mamba_layers
            and isinstance(coordinator, HybridKVCacheCoordinator)
            and coordinator.full_attention_group_id is not None
        ):
            return *self.get_computed_blocks(request), False

        # ------【前缀缓存】缓存关闭或请求跳过时返回空命中 ------
        if not self.prefix_cache_lookup_enabled(request):
            return self.empty_kv_cache_blocks, 0, 0, False

        # ------【PD 分离】按组分别查最长命中，比较全注意组与稀疏组命中深度 ------
        fa_group_id = coordinator.full_attention_group_id
        computed, per_group_hits = coordinator.find_longest_cache_hit_per_group(
            request.block_hashes, request.num_tokens - 1
        )
        # ------【PD 分离】稀疏组命中更深说明全注意块被驱逐，回退到共识边界重新对齐 ------
        if any(hit > per_group_hits[fa_group_id] for hit in per_group_hits):
            # A lagging group hit deeper than full attention means its
            # full-attention blocks were evicted; use the reconciled boundary
            # that every group agrees on.
            return *self.get_computed_blocks(request), False

        # ------【PD 分离】取全注意组命中为本地前缀，边界填 0 并标记命中是否发散 ------
        num_local = per_group_hits[fa_group_id]
        blocks = self.create_kv_cache_blocks(computed)
        # Per-group lookups do not detect an uncached shared prefix (boundary 0).
        return blocks, num_local, 0, min(per_group_hits) < num_local

    def allocate_slots(
        self,
        request: Request,
        num_new_tokens: int,
        num_new_computed_tokens: int = 0,
        new_computed_blocks: KVCacheBlocks | None = None,
        num_lookahead_tokens: int = 0,
        num_external_computed_tokens: int = 0,
        delay_cache_blocks: bool = False,
        num_encoder_tokens: int = 0,
        full_sequence_must_fit: bool = False,
        reserved_blocks: int = 0,
        has_scheduled_reqs: bool = True,
    ) -> KVCacheBlocks | None:
        """Add slots for a request with new tokens to append.

        Args:
            request: The request to allocate slots.
            num_new_tokens: The number of new tokens to be allocated and computed.
            num_new_computed_tokens: The number of new computed tokens just
                hitting the prefix caching, excluding external tokens.
            new_computed_blocks: The cached blocks for the above new computed
                tokens, grouped as a tuple by kv cache groups.
            num_lookahead_tokens: The number of speculative tokens to allocate.
                This is used by spec decode proposers with kv-cache such
                as eagle.
            num_external_computed_tokens: The number of tokens that their
                KV caches are not cached by vLLM but cached by the connector.
            delay_cache_blocks: Whether to skip caching the blocks. This is
                used by P/D when allocating blocks used in a KV transfer
                which will complete in a future step.
            num_encoder_tokens: The number of encoder tokens to allocate for
                cross-attention in encoder-decoder models(e.g., Whisper).
                For decoder-only models, this should be 0.
            full_sequence_must_fit: Only allocate blocks if the KV cache has enough
                free blocks to hold the full sequence, accounting for prefix cache hits
                and sliding window. Used as an admission gate to prevent over-admitting
                requests when chunked prefill would otherwise only check the first chunk
            reserved_blocks: Number of free blocks that must be left available for
                other in-flight sequences to complete. The actual allocation is only
                made if it fits within (free blocks - reserved_blocks). Used to gate
                async KV-connector loads so their initial allocation cannot consume
                blocks an already in-flight (prefilling) sequence is relying on.
            has_scheduled_reqs: Whether any requests are already scheduled to run
                this step, controls whether watermark is applied.

        Blocks layout:
        ```
        ----------------------------------------------------------------------
        | < comp > | < new_comp > | < ext_comp >  | < new >  | < lookahead > |
        ----------------------------------------------------------------------
                                                  |   < to be computed >     |
        ----------------------------------------------------------------------
                                  |            < to be allocated >           |
        ----------------------------------------------------------------------
                                  | < to be cached (roughly, |
                                  | details below)>          |
        ----------------------------------------------------------------------
        | Prefix-cached tokens from either vLLM   |
        | or connector. Can be safely removed if  |
        | they are outside sliding window.        |
        ----------------------------------------------------------------------
        |   < cached by vLLM >    | not cached by |
                                  | vLLM, but     |
        | ref_cnt  | ref_cnt not  | cached by     |
        | increased| increased yet| connector     |
        ----------------------------------------------------------------------
        ```

        Abbrivations:

        ```
        comp      = request.num_computed_tokens
        new_comp  = num_new_computed_tokens
                  = len(new_computed_blocks) * block_size
        ext_comp  = num_external_computed_tokens, cached by the connector
        new       = num_new_tokens, including unverified draft tokens
        lookahead = num_lookahead_tokens
        ```

        NOTE: for new tokens which include both verified and unverified draft
        tokens, we only cache the verified tokens (by capping the number at
        `request.num_tokens`).

        The allocation has three stages:
        - Free unnecessary blocks in `comp` and check
           if we have sufficient free blocks (return None if not).
        - Handle prefix tokens (`comp + new_comp + ext_comp`):
            - Free unnecessary blocks (e.g. outside sliding window)
            - Allocate new blocks for `ext_comp` tokens inside
              sliding window
        - Allocate new blocks for tokens to be computed (`new + lookahead`)

        Returns:
            A list of new allocated blocks.
        """
        # When loading KV data asynchronously, we may have zero new tokens to
        # compute while still allocating slots for externally computed tokens.
        # ------【核心逻辑】无新增 token 也无外部已算 token 时属于非法调用，直接报错 ------
        if num_new_tokens == 0 and num_external_computed_tokens == 0:
            raise ValueError(
                "num_new_tokens must be greater than 0 when there are no "
                "external computed tokens"
            )

        # ------【核心逻辑】归一化新命中 block 序列，未提供时复用空对象避免分配 ------
        if new_computed_blocks is not None:
            new_computed_block_list = new_computed_blocks.blocks
        else:
            new_computed_block_list = self.empty_kv_cache_blocks.blocks

        # The number of computed tokens is the number of computed tokens plus
        # the new prefix caching hits
        # ------【核心逻辑】累加本地与外部已算 token 得到总已算 token，并受最大长度约束 ------
        num_local_computed_tokens = (
            request.num_computed_tokens + num_new_computed_tokens
        )
        total_computed_tokens = min(
            num_local_computed_tokens + num_external_computed_tokens,
            self.max_model_len,
        )

        # ------【核心逻辑】仅对等待/被抢占请求且已有请求调度时启用 watermark 预留 ------
        watermark_blocks = 0
        # The watermark is applied to waiting/preempted requests only, and only
        # when there's at least one request already scheduled.
        if has_scheduled_reqs and request.status in (
            RequestStatus.WAITING,
            RequestStatus.PREEMPTED,
        ):
            watermark_blocks = self.watermark_blocks

        # ------【chunked prefill】准入闸门：整条序列装不下时提前拒绝，避免 chunk 只查首块误放行 ------
        if full_sequence_must_fit:
            # First check and fail if the full request sequence won't fit.
            full_num_tokens = min(request.num_tokens, self.max_model_len)

            num_blocks_to_allocate = self.coordinator.get_num_blocks_to_allocate(
                request_id=request.request_id,
                num_tokens=full_num_tokens,
                new_computed_blocks=new_computed_block_list,
                num_encoder_tokens=num_encoder_tokens,
                total_computed_tokens=total_computed_tokens,
                num_local_computed_tokens=num_local_computed_tokens,
                num_tokens_main_model=full_num_tokens,
                apply_admission_cap=True,
            )
            required_blocks = num_blocks_to_allocate + watermark_blocks
            if required_blocks > self.block_pool.get_num_free_blocks():
                return None

        # ------【投机解码】主模型 token 数加 lookahead 草稿 token 得到需要占槽的总 token 数 ------
        num_tokens_main_model = total_computed_tokens + num_new_tokens
        num_tokens_need_slot = min(
            num_tokens_main_model + num_lookahead_tokens, self.max_model_len
        )

        # Free the blocks that are skipped during the attention computation
        # (e.g., tokens outside the sliding window).
        # We can do this even if we cannot schedule this request due to
        # insufficient free blocks.
        # Should call this function before allocating new blocks to reduce
        # the number of evicted blocks.
        # Free on the processed-token basis: in-flight steps' attention windows
        # still read blocks below the optimistic boundary, and rejected spec
        # tokens can roll it back.
        # ------【核心逻辑】先释放滑动窗口外/已处理过的 block，为新分配腾出空间 ------
        self.coordinator.remove_skipped_blocks(
            request.request_id,
            max(0, total_computed_tokens - request.num_in_flight_tokens),
            num_prompt_tokens=request.num_prompt_tokens,
        )

        # ------【核心逻辑】由协调器按 token 与 block_size 计算出本次需分配的 block 数 ------
        num_blocks_to_allocate = self.coordinator.get_num_blocks_to_allocate(
            request_id=request.request_id,
            num_tokens=num_tokens_need_slot,
            new_computed_blocks=new_computed_block_list,
            num_encoder_tokens=num_encoder_tokens,
            total_computed_tokens=num_local_computed_tokens
            + num_external_computed_tokens,
            num_local_computed_tokens=num_local_computed_tokens,
            num_tokens_main_model=num_tokens_main_model,
        )

        # Keep `reserved_blocks` free for other in-flight sequences, and an
        # additional watermark of headroom for waiting/preempted admissions.
        # ------【核心逻辑】扣除 reserved/watermark 后空闲 block 不足则本次分配失败 ------
        available_blocks = self.block_pool.get_num_free_blocks() - reserved_blocks
        required_blocks = num_blocks_to_allocate + watermark_blocks
        if required_blocks > available_blocks:
            # Cannot allocate new blocks
            return None

        # ------【前缀缓存】把前缀命中的 block 并入请求分配表，避免新块分配失败时丢失 ------
        if (
            new_computed_block_list is not self.empty_kv_cache_blocks.blocks
            or num_external_computed_tokens > 0
        ):
            # Append the new computed blocks to the request blocks until now to
            # avoid the case where the new blocks cannot be allocated.
            self.coordinator.allocate_new_computed_blocks(
                request_id=request.request_id,
                new_computed_blocks=new_computed_block_list,
                num_local_computed_tokens=num_local_computed_tokens,
                num_external_computed_tokens=num_external_computed_tokens,
            )

        # ------【核心逻辑】真正从 block 池为新 token/编码器 token 分配物理 block ------
        new_blocks = self.coordinator.allocate_new_blocks(
            request.request_id,
            num_tokens_need_slot,
            num_tokens_main_model,
            num_encoder_tokens,
        )

        # ------【PD 分离】关闭缓存或延迟缓存时跳过写缓存，仅返回新分配块 ------
        # P/D: delay caching blocks if we have to recv from
        # remote. Update state for locally cached blocks.
        if not self.enable_caching or delay_cache_blocks:
            return self.create_kv_cache_blocks(new_blocks)

        # NOTE(woosuk): We want to commit (cache) up to num_local_computed_tokens
        # + num_external_computed_tokens + num_new_tokens, but must exclude
        # "non-committable" tokens (e.g., draft tokens that could be rejected).
        # Therefore, we cap the number at `request.num_tokens`, ensuring only
        # "finalized" tokens are cached.
        # ------【前缀缓存+投机解码】只把已定稿 token 写入缓存，剔除可能被拒绝的草稿 token ------
        num_tokens_to_cache = min(
            total_computed_tokens + num_new_tokens,
            request.num_tokens,
        )
        self.coordinator.cache_blocks(request, num_tokens_to_cache)

        return self.create_kv_cache_blocks(new_blocks)

    def free(self, request: Request) -> None:
        """Free the blocks allocated for the request.
        We free the blocks in reverse order so that the tail blocks are evicted
        first when caching is enabled.

        Args:
            request: The request to free the blocks.
        """
        # ------【PD 分离】先释放离线搬运 pin 住的 COW block，再释放请求主 block ------
        pins = self._partial_tail_pins.pop(request.request_id, None)
        if pins:
            self.block_pool.free_blocks(pins)
        self.coordinator.free(request.request_id)

    def remove_skipped_blocks(
        self,
        request_id: str,
        processed_computed_tokens: int,
        num_prompt_tokens: int | None = None,
    ) -> None:
        """Remove the blocks that are no longer needed from `blocks` and replace
        the removed blocks with null_block.

        Args:
            request_id: The request ID.
            processed_computed_tokens: Computed-token prefix length covering
                fully processed and committed tokens only (safe to free).
            num_prompt_tokens: Optional prompt length for R-SWA gap eviction.
        """
        # ------【核心逻辑】透传请求 ID 与已处理 token 前缀，让协调器释放越界 block ------
        self.coordinator.remove_skipped_blocks(
            request_id, processed_computed_tokens, num_prompt_tokens
        )

    def pop_blocks_for_free(self, request: Request) -> list[KVCacheBlock]:
        """Pop the request's bookkeeping and return its blocks without
        returning them to the block pool. The caller must eventually free
        them in reverse order (so that tail blocks are evicted first).

        Args:
            request: The request to pop the blocks for.

        Returns:
            The request's blocks in allocation order.
        """
        # ------【核心逻辑】取出请求账本 block 但不归还池，由调用方后续逆序释放 ------
        blocks = self.coordinator.pop_blocks_for_free(request.request_id)
        # Pins ride the same (possibly deferred) free as the request blocks.
        # Preemption may release a pin under a still-queued offload — the same
        # exposure normal saves of table blocks already have.
        # ------【PD 分离】pin 块与请求 block 一并取出，抢占时避免离线搬运悬挂 ------
        pins = self._partial_tail_pins.pop(request.request_id, None)
        if pins:
            blocks = pins + blocks
        return blocks

    def evict_blocks(self, block_ids: set[int]) -> None:
        """evict blocks from the prefix cache by their block IDs.

        Args:
            block_ids: Set of block IDs to evict from cache.
        """
        # ------【前缀缓存】按 block ID 集合强制驱逐前缀缓存条目 ------
        self.block_pool.evict_blocks(block_ids)

    def reset_prefix_cache(self) -> bool:
        """Reset prefix cache. This function may be used in RLHF
        flows to invalidate prefix caching after the weights are updated,
        or used for resetting prefix caching status for benchmarking.

        Returns:
            bool: True if the prefix cache is successfully reset,
            False otherwise.
        """
        # ------【权重传输】权重热更新后重置前缀缓存使旧哈希失效，RLHF 常用 ------
        if not self.block_pool.reset_prefix_cache():
            return False
        # ------【前缀缓存】重置时同步打标统计，供日志上报重置事件 ------
        if self.log_stats:
            assert self.prefix_cache_stats is not None
            self.prefix_cache_stats.reset = True
        return True

    def get_num_common_prefix_blocks(self, running_request_id: str) -> list[int]:
        """Calculate the number of common prefix blocks for each kv cache group.

        The function selects a running request and iterates through its blocks.
        A block is considered a common prefix block if ALL requests with
        allocated KV cache share it (i.e., ref_cnt equals the number of entries
        in req_to_blocks).

        NOTE(woosuk): The number of requests with allocated KV cache is **greater
        than or equal to** the number of requests scheduled in the current step.
        This is because having allocated KV cache only indicates that:
        1. The request has not yet finished, and
        2. The request holds its blocks unfreed.

        While all scheduled requests must have allocated KV cache, the inverse
        is not necessarily true. There may be requests with allocated KV cache
        that are not scheduled in the current step.

        This can result in an edge case where the number of common prefix blocks
        is 0, even though all scheduled requests share a common prefix. This
        occurs because there may be unscheduled requests that do not share the
        common prefix. Currently, this case cannot be easily detected, so the
        function returns 0 in such cases.

        Args:
            running_request_id: The request ID of any running request, used to
                identify the common prefix blocks.

        Returns:
            list[int]: The number of common prefix blocks for each kv cache
            group.
        """
        # ------【核心逻辑】透传计算所有在途请求共享的公共前缀 block 数 ------
        return self.coordinator.get_num_common_prefix_blocks(running_request_id)

    def take_events(self) -> list[KVCacheEvent]:
        """Take the KV cache events from the block pool.

        Returns:
            A list of KV cache events.
        """
        # ------【异步 RPC】取出 block 池事件队列，为 BlockStored 补上 KV 语义元数据 ------
        events = self.block_pool.take_events()
        for event in events:
            if not isinstance(event, BlockStored):
                continue
            if event.group_idx is None:
                continue
            if event.group_idx < 0 or event.group_idx >= len(
                self.kv_cache_event_metadata
            ):
                logger.warning(
                    "Group index `%s` not in KV cache metadata", event.group_idx
                )
                continue
            # Annotate here so BlockPool can keep emitting structural cache
            # events without owning semantic KV cache spec metadata.
            kind, sliding_window = self.kv_cache_event_metadata[event.group_idx]
            event.kv_cache_spec_kind = kind
            event.kv_cache_spec_sliding_window = sliding_window
        return events

    def get_blocks(self, request_id: str) -> KVCacheBlocks:
        """Get the blocks of a request."""
        # ------【核心逻辑】取出请求 block 并包装为 KVCacheBlocks ------
        return self.create_kv_cache_blocks(self.coordinator.get_blocks(request_id))

    def get_block_ids(self, request_id: str) -> tuple[list[int], ...]:
        """Get the block ids of a request."""
        # ------【核心逻辑】取请求 block 后转成 block_id 序列 ------
        return self.get_blocks(request_id).get_block_ids()

    def get_block_ids_for_computed_tokens(
        self,
        request_id: str,
        num_computed_tokens: int,
    ) -> tuple[list[int], ...]:
        """Get block ids covering the request's computed tokens."""
        # ------【核心逻辑】取请求 block id，按已算 token 数裁剪有效 block 前缀 ------
        block_ids = self.get_block_ids(request_id)
        clipped_block_ids: list[list[int]] = []
        for group, ids in zip(self.kv_cache_config.kv_cache_groups, block_ids):
            spec = group.kv_cache_spec
            # ------【核心逻辑】交叉注意力/编码器组不参与裁剪，原样保留 ------
            if not isinstance(spec, AttentionSpec) or isinstance(
                spec, (CrossAttentionSpec, EncoderOnlyAttentionSpec)
            ):
                clipped_block_ids.append(ids)
                continue

            # ------【核心逻辑】按已算 token 与 block_size 算出有效块数并截取前缀 ------
            num_valid_blocks = cdiv(num_computed_tokens, spec.block_size)
            clipped_block_ids.append(ids[:num_valid_blocks])
        return tuple(clipped_block_ids)

    def estimate_cached_tokens(self, request: Request) -> int:
        """Estimate the number of tokens cached by the request."""
        # ------【前缀缓存】逐组统计已缓存 token，跨组取最小值为全请求公共缓存量 ------
        cached_tokens: int | None = None
        for group, blocks in zip(
            self.kv_cache_config.kv_cache_groups,
            self.get_blocks(request.request_id).blocks,
        ):
            if isinstance(
                group.kv_cache_spec,
                (CrossAttentionSpec, EncoderOnlyAttentionSpec),
            ):
                # Cross-attention and encoder-only groups are not prefix cached.
                continue

            # ------【前缀缓存】组内取各 block 已缓存 token 数最大值作为该组缓存深度 ------
            group_cached_tokens = 0
            for block in blocks:
                group_cached_tokens = max(
                    group_cached_tokens,
                    block.block_hash_num_tokens or 0,
                )

            cached_tokens = (
                group_cached_tokens
                if cached_tokens is None
                else min(cached_tokens, group_cached_tokens)
            )

        return cached_tokens or 0

    def cache_blocks(self, request: Request, num_computed_tokens: int) -> None:
        """Cache the blocks for the request, if enabled.

        Args:
            request: The request to cache the blocks.
            num_computed_tokens: The number of computed tokens, including tokens
                that are already cached and tokens to be cached.
        """
        # ------【前缀缓存】缓存开启时把已算 token 对应的 block 写入前缀缓存索引 ------
        if self.enable_caching:
            self.coordinator.cache_blocks(request, num_computed_tokens)

    def create_kv_cache_blocks(
        self, blocks: tuple[list[KVCacheBlock], ...]
    ) -> KVCacheBlocks:
        # ------【核心逻辑】仅非空才新建对象，否则复用预构造空对象省 GC ------
        # Only create new KVCacheBlocks for non-empty blocks
        return KVCacheBlocks(blocks) if any(blocks) else self.empty_kv_cache_blocks

    def truncate_computed_blocks(
        self, blocks: KVCacheBlocks, num_computed_tokens: int
    ) -> KVCacheBlocks:
        """Return a lookup-result view truncated at an aligned token endpoint.

        Pure slicing: refcounts are untouched and ``blocks`` is not mutated.
        """
        # ------【前缀缓存】把查找结果按对齐 token 端点纯切片截断，不动引用计数 ------
        truncated: list[list[KVCacheBlock]] = []
        for group_blocks, manager in zip(
            blocks.blocks,
            self.coordinator.single_type_managers,
            strict=True,
        ):
            assert num_computed_tokens % manager.block_size == 0
            num_blocks = num_computed_tokens // manager.block_size
            assert num_blocks <= len(group_blocks)
            truncated.append(list(group_blocks[:num_blocks]))
        return self.create_kv_cache_blocks(tuple(truncated))

    def take_new_block_ids(self) -> list[int]:
        """Drain and return new attention block IDs for zeroing."""
        # ------【异步 RPC】汇总各 manager 需要清零的新 block id 并清空待处理队列 ------
        ids: list[int] = []
        for mgr in self.coordinator.single_type_managers:
            ids.extend(mgr.take_new_block_ids())
        return ids

    def get_zeroing_block_ids_in_range(
        self, request_id: str, start_token: int, end_token: int
    ) -> list[int]:
        """The request's block ids covering [start_token, end_token), from
        the groups whose new blocks are zeroed by the worker."""
        # ------【异步 RPC】返回 [start,end) token 区间内需清零的 block id ------
        ids: list[int] = []
        for mgr in self.coordinator.single_type_managers:
            if mgr.records_new_block_ids:
                start_idx = start_token // mgr.block_size
                end_idx = cdiv(end_token, mgr.block_size)
                blocks = mgr.req_to_blocks[request_id]
                ids.extend(blk.block_id for blk in blocks[start_idx:end_idx])
        return ids

    def record_blocks_for_zeroing(self, request_id: str, start_token: int) -> None:
        """Re-record the request's blocks from start_token onwards for
        zeroing, e.g. blocks a failed async KV load left unwritten.

        start_token must be block-aligned: zeroing a partially-valid block
        would wipe its valid prefix.
        """
        # ------【异步 RPC】失败的异步 KV 加载留白后，从 start_token 起重新登记清零块 ------
        for mgr in self.coordinator.single_type_managers:
            if mgr.records_new_block_ids:
                assert start_token % mgr.block_size == 0
                start_idx = start_token // mgr.block_size
                blocks = mgr.req_to_blocks[request_id]
                mgr.new_block_ids.extend(blk.block_id for blk in blocks[start_idx:])

    def take_kv_cache_block_copies(
        self,
    ) -> tuple[list[KVCacheBlockCopy], list[KVCacheBlock]]:
        """Drain pending copies and return their retained endpoints."""
        # ------【内存池/CuMem】收集各 manager 待完成的 COW 拷贝，转为可下发结构并保留端点块 ------
        pending_copies: list[tuple[KVCacheBlock, KVCacheBlock]] = []
        for mgr in self.coordinator.single_type_managers:
            pending_copies.extend(mgr.take_pending_cow_copies())
        copies = [
            KVCacheBlockCopy(
                src_block_id=source_block.block_id,
                dst_block_id=cow_block.block_id,
            )
            for source_block, cow_block in pending_copies
        ]
        retained_blocks = [block for pair in pending_copies for block in pair]
        return copies, retained_blocks

    def take_partial_tail_offloads(self) -> dict[str, list[tuple[int, int, int]]]:
        """Drain producer partial-tail offload hand-offs per request.

        Returns ``{request_id: [(group_id, block_id, boundary_tokens), ...]}``
        for the durable boundary blocks of producers' last-prompt-boundary
        partial tails. Only mamba "align" groups contribute; empty otherwise.
        A KV connector reads the referenced blocks and offloads them so a later
        request can hit the sub-block prefix.

        Each handed-off block lives off the request block table, so it is
        pinned here and unpinned when the request's blocks are freed — for a
        producer with saved tokens, after the connector reports sends done.
        """
        # ------【PD 分离】取出 mamba 组部分尾部离线搬运任务，pin 住块并登记边界信息 ------
        offloads: dict[str, list[tuple[int, int, int]]] = {}
        for mgr in self.coordinator.single_type_managers:
            for (
                req_id,
                group_id,
                block,
                boundary_tokens,
            ) in mgr.take_pending_partial_tail_offloads():
                self.block_pool.touch((block,))
                self._partial_tail_pins.setdefault(req_id, []).append(block)
                offloads.setdefault(req_id, []).append(
                    (group_id, block.block_id, boundary_tokens)
                )
        return offloads

    def new_step_starts(self) -> None:
        """Notify the coordinator that a new step is starting."""
        # ------【核心逻辑】通知协调器新一步开始，触发内部步级状态更新 ------
        self.coordinator.new_step_starts()
