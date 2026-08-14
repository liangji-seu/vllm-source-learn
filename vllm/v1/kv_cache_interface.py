# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import copy
from collections import Counter
from dataclasses import dataclass, fields, replace
from enum import Enum, IntEnum
from math import prod
from typing import TYPE_CHECKING

import torch
from typing_extensions import Self

from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv, round_up
from vllm.utils.torch_utils import get_dtype_size, nvfp4_kv_cache_full_dim
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# KV cache quantization mode
# ---------------------------------------------------------------------------


# ------【显存 profiling/量化】KVQuantMode：KV cache 量化模式枚举，供 attention backend/kernel 分发量化逻辑，省显存 ------
class KVQuantMode(IntEnum):
    """KV cache quantization mode.

    Used by attention backends and kernels to dispatch quantization logic
    without string matching on ``kv_cache_dtype``.
    """

    # ------【显存 profiling】NONE：不量化，按原始 dtype 存储 ------
    NONE = 0
    # ------【显存 profiling】FP8_PER_TENSOR：整张量共享一个 fp8 scale（当前 fp8 路径） ------
    FP8_PER_TENSOR = 1  # per-tensor scales (current fp8 path)
    # ------【显存 profiling】INT8_PER_TOKEN_HEAD：int8 每 token-每 head 动态 scale ------
    INT8_PER_TOKEN_HEAD = 2  # per-token-head dynamic scales for int8
    # ------【显存 profiling】FP8_PER_TOKEN_HEAD：fp8 每 token-每 head 动态 scale ------
    FP8_PER_TOKEN_HEAD = 3  # per-token-head dynamic scales for fp8
    # ------【显存 profiling】INT4_PER_TOKEN_HEAD：每字节打包 2×int4，RHT+非对称零点 ------
    INT4_PER_TOKEN_HEAD = 4  # packed 2×int4/byte, RHT + asymmetric zp
    # ------【显存 profiling】NVFP4：打包 fp4 数据 + fp8 块 scale（Blackwell 专用） ------
    NVFP4 = 5  # packed fp4 data + fp8 block scales
    # ------【显存 profiling】TURBOQUANT：Hadamard 旋转 + Lloyd-Max 量化，K/V 每 slot 打包 ------
    TURBOQUANT = 6  # Hadamard-rotated Lloyd-Max quant, packed K+V per slot

    @property
    def is_per_token_head(self) -> bool:
        """True for any per-token-head quantization mode."""
        return self in (
            KVQuantMode.INT8_PER_TOKEN_HEAD,
            KVQuantMode.FP8_PER_TOKEN_HEAD,
            KVQuantMode.INT4_PER_TOKEN_HEAD,
        )

    @property
    def is_nvfp4(self) -> bool:
        """True for NVFP4 packed quantization mode."""
        return self == KVQuantMode.NVFP4

    @property
    def is_turboquant(self) -> bool:
        """True for turboquant quantization mode."""
        return self == KVQuantMode.TURBOQUANT


# ------【显存 profiling/量化】get_kv_quant_mode：把 kv_cache_dtype 字符串映射为 KVQuantMode 枚举 ------
def get_kv_quant_mode(kv_cache_dtype: str) -> KVQuantMode:
    """Map a ``kv_cache_dtype`` string to a :class:`KVQuantMode`."""
    if kv_cache_dtype == "int4_per_token_head":
        return KVQuantMode.INT4_PER_TOKEN_HEAD
    if kv_cache_dtype == "int8_per_token_head":
        return KVQuantMode.INT8_PER_TOKEN_HEAD
    if kv_cache_dtype == "fp8_per_token_head":
        return KVQuantMode.FP8_PER_TOKEN_HEAD
    if kv_cache_dtype == "nvfp4":
        return KVQuantMode.NVFP4
    if isinstance(kv_cache_dtype, str) and kv_cache_dtype.startswith("turboquant_"):
        return KVQuantMode.TURBOQUANT
    if isinstance(kv_cache_dtype, str) and kv_cache_dtype.startswith("fp8"):
        return KVQuantMode.FP8_PER_TENSOR
    return KVQuantMode.NONE


# ------【显存 profiling/量化】is_quantized_kv_cache：判断 kv_cache_dtype 是否为量化模式 ------
def is_quantized_kv_cache(kv_cache_dtype: str) -> bool:
    return get_kv_quant_mode(kv_cache_dtype) != KVQuantMode.NONE


# ------【显存 profiling/量化】kv_cache_uses_per_token_head_scales：判断是否需要 per-token-head scale ------
def kv_cache_uses_per_token_head_scales(kv_cache_dtype: str) -> bool:
    """Return True if *kv_cache_dtype* needs per-token-head scales."""
    return get_kv_quant_mode(kv_cache_dtype).is_per_token_head


# ------【核心逻辑】KVCacheSpecKind：KV cache spec 类型标签枚举，供引擎按注意力类型分发逻辑 ------
class KVCacheSpecKind(str, Enum):
    # ------【核心逻辑】FULL_ATTENTION：全注意力（标准因果注意力） ------
    FULL_ATTENTION = "full_attention"
    # ------【MLA】MLA_ATTENTION：多头潜在注意力 ------
    MLA_ATTENTION = "mla_attention"
    # ------【滑动窗口】SLIDING_WINDOW：滑动窗口注意力 ------
    SLIDING_WINDOW = "sliding_window"
    # ------【滑动窗口/MLA】SLIDING_WINDOW_MLA：滑动窗口 + MLA 组合 ------
    SLIDING_WINDOW_MLA = "sliding_window_mla"
    # ------【核心逻辑】MAMBA：Mamba 状态空间模型 ------
    MAMBA = "mamba"
    # ------【chunked prefill】CHUNKED_LOCAL_ATTENTION：分块局部注意力 ------
    CHUNKED_LOCAL_ATTENTION = "chunked_local_attention"
    # ------【核心逻辑】SINK_FULL_ATTENTION：带 sink token 的全注意力 ------
    SINK_FULL_ATTENTION = "sink_full_attention"
    # ------【核心逻辑】ENCODER_ONLY_ATTENTION：仅编码器注意力 ------
    ENCODER_ONLY_ATTENTION = "encoder_only_attention"
    # ------【核心逻辑】CROSS_ATTENTION：交叉注意力（encoder-decoder） ------
    CROSS_ATTENTION = "cross_attention"
    # ------【核心逻辑】UNKNOWN：未知/混合类型 ------
    UNKNOWN = "unknown"


# ------【显存 profiling/核心逻辑】KVCacheSpec：单层 KV cache 布局基类 DTO，Worker 建模层产出→EngineCore/KV cache manager 显存规划 ------
@dataclass(frozen=True)
class KVCacheSpec:
    """
    A base class for specifying the KV cache format of one layer.
    """

    # ------【显存 profiling】block_size：一个 block 可容纳的 token 数（页内 token 数） ------
    # number of tokens in a block
    block_size: int

    # ------【显存 profiling】page_size_bytes：单页字节数，抽象方法由子类按布局实现 ------
    @property
    def page_size_bytes(self) -> int:
        """
        The size of a page with `block_size` tokens in bytes.

        Returns:
            The page size
        """
        raise NotImplementedError

    # ------【MLA/显存 profiling】storage_block_size：实际存储 token 数，默认=block_size，MLA 压缩时除以 compress_ratio ------
    @property
    def storage_block_size(self) -> int:
        return self.block_size

    # ------【显存 profiling】max_memory_usage_bytes：该层 KV cache 最大占用字节数，抽象方法由子类实现 ------
    def max_memory_usage_bytes(self, vllm_config: VllmConfig) -> int:
        """
        The maximum possible memory usage of this KV cache in bytes.

        Returns:
            The KV cache size in bytes
        """
        raise NotImplementedError

    # ------【核心逻辑】max_num_blocks_per_req：单请求所需块表行长度（该 cache group 块表列数） ------
    def max_num_blocks_per_req(self, vllm_config: VllmConfig, max_len: int) -> int:
        """
        The number of block table entries needed per request, i.e. the row
        length of the worker-side block table for this cache group.

        Args:
            vllm_config: The vllm config.
            max_len: The maximum sequence length to size for, including the
                encoder length for encoder-decoder models.
        """
        return cdiv(max_len, self.block_size)

    # ------【显存 profiling】copy_with_new_block_size：复制自身并替换 block_size（对齐块大小调整） ------
    def copy_with_new_block_size(self, block_size: int) -> Self:
        """
        Create a new KVCacheSpec from self but replacing the block size.
        """
        return replace(self, block_size=block_size)

    # ------【核心逻辑】merge：合并同一 KV cache group 的 spec 列表，要求各层完全一致 ------
    @classmethod
    def merge(cls, specs: list[Self]) -> Self:
        """
        Merge a list of KVCacheSpec objects into a single KVCacheSpec object.
        """
        assert all(spec == specs[0] for spec in specs[1:]), (
            "All layers in the same KV cache group must be the same."
        )
        return copy.deepcopy(specs[0])

    # ------【核心逻辑】is_uniform_with_collection：本 spec 是否与所有层 spec 同构（用于 uniform 合并判定） ------
    def is_uniform_with_collection(
        self, kv_cache_specs: dict[str, KVCacheSpec]
    ) -> bool:
        """
        Whether this KVCacheSpec is uniform with all specs of all layers.
        """
        uniform_type_base_spec = KVCacheSpecRegistry.get_uniform_type_base_spec(self)
        assert uniform_type_base_spec is not None, (
            f"Unsupported KV cache spec type: {type(self)}. "
            "Please register it using @register_kv_cache_spec decorator."
        )
        return all(
            isinstance(spec, uniform_type_base_spec) for spec in kv_cache_specs.values()
        )


# ------【显存 profiling】AttentionSpec：标准注意力 KV cache 布局 DTO，补充 K/V 头数、head 维度、dtype、量化模式等字段 ------
@dataclass(frozen=True, kw_only=True)
class AttentionSpec(KVCacheSpec):
    # ------【TP/GQA】num_kv_heads：KV 头数（GQA/MQA 下小于 Q 头数，节省 KV 显存） ------
    num_kv_heads: int
    # ------【显存 profiling】head_size：每个 KV 头的维度 ------
    head_size: int
    # ------【显存 profiling】dtype：KV cache 存储的 torch dtype ------
    dtype: torch.dtype
    # ------【显存 profiling/量化】kv_quant_mode：量化模式，默认 NONE 不量化 ------
    kv_quant_mode: KVQuantMode = KVQuantMode.NONE
    # ------【显存 profiling/对齐】page_size_padded：页字节对齐填充值，None 表示不填充 ------
    page_size_padded: int | None = None
    # ------【CUDA Graph/打包布局】indexes_kv_by_block_stride：KV 是否按 block 步长索引 ------
    indexes_kv_by_block_stride: bool = False

    # ------【显存 profiling】unpadded_page_size_bytes：未对齐页字节数，per-token-head 量化额外计入 scale 占用 ------
    @property
    def unpadded_page_size_bytes(self) -> int:
        unpadded = self.real_page_size_bytes
        # Per-token-head scales are stored in separate tensors managed
        # by the attention backend, but the memory is carved from the
        # raw KV cache allocation so it must be budgeted here.
        if self.kv_quant_mode.is_per_token_head:
            unpadded += (
                2 * self.block_size * self.num_kv_heads * get_dtype_size(torch.float32)
            )
        return unpadded

    # ------【显存 profiling】page_size_bytes：页字节数，有填充用填充值否则用未对齐值 ------
    @property
    def page_size_bytes(self) -> int:
        if self.page_size_padded is not None:
            assert self.page_size_padded >= self.unpadded_page_size_bytes
            return self.page_size_padded
        return self.unpadded_page_size_bytes

    # ------【显存 profiling/量化】real_page_size_bytes：按量化模式算 K+V 实际字节（nvfp4/int4 改变 head 维度） ------
    @property
    def real_page_size_bytes(self) -> int:
        if self.kv_quant_mode.is_nvfp4:
            # Packed layout: fp4 data + fp8 block scales per head.
            head_dim = nvfp4_kv_cache_full_dim(self.head_size)
        elif self.kv_quant_mode == KVQuantMode.INT4_PER_TOKEN_HEAD:
            head_dim = self.head_size // 2
        else:
            head_dim = self.head_size
        return (
            2
            * self.block_size
            * self.num_kv_heads
            * head_dim
            * get_dtype_size(self.dtype)
        )

    # ------【PD 分离/DCP】max_num_blocks_per_req：按 decode 上下文并行分片数折算块表行长度 ------
    def max_num_blocks_per_req(self, vllm_config: VllmConfig, max_len: int) -> int:
        parallel_config = vllm_config.parallel_config
        kv_shard_count = parallel_config.decode_context_parallel_size
        return cdiv(max_len, self.block_size * kv_shard_count)


# ------【滑动窗口/核心逻辑】FullAttentionSpec：全注意力层 KV cache spec；混合模型关闭 hybrid allocator 时，滑动窗口层也按全注意力分配块 ------
@dataclass(frozen=True, kw_only=True)
class FullAttentionSpec(AttentionSpec):
    """
    When hybrid allocator is disabled and the model contains both full
    attention layers and sliding window attention layers, sliding
    window attention are regarded as full attention in KV cache manager
    (blocks are allocated for all tokens), while computed as sliding window
    attention in model runner.
    In this case, we use FullAttentionSpec and record the sliding window size.
    """

    # ------【MQA/GQA】head_size_v：V 头维度，默认回填为 head_size（MQA 下可不同） ------
    head_size_v: int = None  # type: ignore[assignment]

    # ------【滑动窗口】sliding_window：滑动窗口大小，None 表示不使用滑动窗口 ------
    sliding_window: int | None = None
    """
    Default to None for not using sliding window attention.
    """
    # ------【chunked prefill】attention_chunk_size：注意力分块大小，用于 chunked local attention ------
    attention_chunk_size: int | None = None

    # ------【chunked prefill/前缀缓存】non_causal：是否非因果注意力（如 Prefix LM），影响调度策略 ------
    non_causal: bool = False
    """
    Whether the layer attends non-causally (e.g. Prefix LM). Carried on the
    spec so the engine core, which collects specs from all workers before the
    scheduler is built, can adjust scheduling policy (chunked prefill / prefix
    caching) regardless of tensor-parallel layout. It does not affect the KV
    cache layout itself.
    """

    # ------【核心逻辑】__post_init__：head_size_v 缺省时回填为 head_size ------
    def __post_init__(self):
        if self.head_size_v is None:
            object.__setattr__(self, "head_size_v", self.head_size)

    # ------【显存 profiling/PD 分离】max_memory_usage_bytes：按最大长度折算页数×页字节；DCP>1 时先除以世界大小 ------
    def max_memory_usage_bytes(self, vllm_config: VllmConfig) -> int:
        max_model_len = vllm_config.model_config.max_model_len
        dcp_world_size = vllm_config.parallel_config.decode_context_parallel_size
        if dcp_world_size > 1:
            max_model_len = cdiv(max_model_len, dcp_world_size)
        return cdiv(max_model_len, self.block_size) * self.page_size_bytes

    # ------【核心逻辑】merge_window_sizes：合并窗口大小集合，一致返回该值、空返回 None、冲突抛错 ------
    @classmethod
    def merge_window_sizes(cls, window_sizes: set[int]) -> int | None:
        if len(window_sizes) == 0:
            return None
        elif len(window_sizes) == 1:
            return window_sizes.pop()
        else:
            raise ValueError(
                "All attention layers in the same KV cache group must have the "
                "same window size."
            )

    # ------【核心逻辑】merge：合并 FullAttentionSpec 列表；窗口/分块须一致，任一非因果则整组视为非因果 ------
    @classmethod
    def merge(cls, specs: list[Self]) -> Self:
        """
        Merge a list of FullAttentionSpec objects into a single
        FullAttentionSpec object.
        """
        assert all(isinstance(spec, FullAttentionSpec) for spec in specs), (
            "All attention layers in the same KV cache group must be FullAttentionSpec."
        )

        sliding_window = set(
            spec.sliding_window for spec in specs if spec.sliding_window is not None
        )
        attention_chunk_size = set(
            spec.attention_chunk_size
            for spec in specs
            if spec.attention_chunk_size is not None
        )
        assert not any(isinstance(spec, MLAAttentionSpec) for spec in specs), (
            "MLAAttentionSpec should be merged in MLAAttentionSpec.merge"
        )
        merged_spec = cls(
            block_size=specs[0].block_size,
            num_kv_heads=specs[0].num_kv_heads,
            head_size=specs[0].head_size,
            head_size_v=specs[0].head_size_v,
            dtype=specs[0].dtype,
            kv_quant_mode=specs[0].kv_quant_mode,
            page_size_padded=specs[0].page_size_padded,
            indexes_kv_by_block_stride=specs[0].indexes_kv_by_block_stride,
            sliding_window=cls.merge_window_sizes(sliding_window),
            attention_chunk_size=cls.merge_window_sizes(attention_chunk_size),
            # If any layer in the group is non-causal, treat the group as
            # non-causal so the engine core disables incompatible scheduling.
            non_causal=any(spec.non_causal for spec in specs),
        )
        for spec in specs:
            for f in fields(AttentionSpec):
                assert getattr(spec, f.name) == getattr(merged_spec, f.name), (
                    "All attention layers in the same KV cache group must have "
                    "the same attention spec."
                )
        assert (merged_spec.sliding_window is not None) + (
            merged_spec.attention_chunk_size is not None
        ) <= 1, (
            "Model with both sliding window layers and chunked local attention "
            "layers is not supported."
        )
        return merged_spec

    # ------【显存 profiling/量化】real_page_size_bytes：含 V 头的 K+V 实际字节数，nvfp4/int4 调整维度 ------
    @property
    def real_page_size_bytes(self) -> int:
        if self.kv_quant_mode.is_nvfp4:
            # Packed layout per head: fp4 data + fp8 block scales.
            # fp4 data: head_size//2 bytes (2 fp4 values per byte)
            # fp8 block scale: head_size//16 bytes (1 scale per 16 elements)
            last_dim = nvfp4_kv_cache_full_dim(
                self.head_size
            ) + nvfp4_kv_cache_full_dim(self.head_size_v)
        elif self.kv_quant_mode == KVQuantMode.INT4_PER_TOKEN_HEAD:
            last_dim = self.head_size // 2 + self.head_size_v // 2
        else:
            last_dim = self.head_size + self.head_size_v
        return (
            self.block_size * self.num_kv_heads * last_dim * get_dtype_size(self.dtype)
        )


# ------【显存 profiling/对齐】_apply_alignment_padding：按 alignment 对齐页大小，写入 page_size_padded ------
def _apply_alignment_padding(spec: MLAAttentionSpec | SlidingWindowMLASpec):
    if spec.alignment is None:
        return
    actual_page_size = spec.real_page_size_bytes
    padded_page_size = round_up(actual_page_size, spec.alignment)
    if padded_page_size != actual_page_size:
        object.__setattr__(spec, "page_size_padded", padded_page_size)


# ------【显存 profiling/量化】TQFullAttentionSpec：TurboQuant 感知的全注意力 spec，用 TQ slot 字节算页大小 ------
@dataclass(frozen=True, kw_only=True)
class TQFullAttentionSpec(FullAttentionSpec):
    """FullAttentionSpec with TQ-aware page size.

    Python equivalent of the C++ TQ4FullAttentionSpec. Overrides
    real_page_size_bytes to use TQ slot bytes instead of the raw
    head_size * dtype formula.
    """

    # ------【显存 profiling/量化】tq_slot_size：TurboQuant 每 slot 字节数，>0 时覆盖默认页大小公式 ------
    tq_slot_size: int = 0

    # ------【显存 profiling/量化】real_page_size_bytes：TQ slot 大小>0 按 slot 字节算，否则回退父类公式 ------
    @property
    def real_page_size_bytes(self) -> int:
        if self.tq_slot_size > 0:
            return self.block_size * self.num_kv_heads * self.tq_slot_size
        return super().real_page_size_bytes

    # ------【核心逻辑】merge：合并 TQ spec，校验 tq_slot_size 一致后回填 ------
    @classmethod
    def merge(cls, specs: list[Self]) -> Self:
        merged = super().merge(specs)
        assert all(s.tq_slot_size == specs[0].tq_slot_size for s in specs), (
            "All TQ layers in the same KV cache group must use the same tq_slot_size."
        )
        return replace(merged, tq_slot_size=specs[0].tq_slot_size)


# ------【MLA/显存 profiling】MLAAttentionSpec：MLA 多头潜在注意力 spec，低秩压缩 KV 省显存 ------
@dataclass(frozen=True, kw_only=True)
class MLAAttentionSpec(FullAttentionSpec):
    # TODO(Lucas/Chen): less hacky way to do this
    # ------【显存 profiling/量化】cache_dtype_str：缓存 dtype 字符串（如 fp8_ds_mla），区分自定义 MLA 布局 ------
    cache_dtype_str: str | None = None
    # DeepseekV4 only fields. Non-DeepseekV4 MLA models leave these at defaults.
    # ------【显存 profiling/对齐】alignment：页字节对齐粒度，None 表示不填充（DeepseekV4 专用） ------
    alignment: int | None = None  # Default to None for no padding.
    # ------【MLA/显存 profiling】compress_ratio：KV 压缩比，存储块大小为 block_size//compress_ratio ------
    compress_ratio: int = 1  # Default to 1 for no compression.
    # ------【核心逻辑】model_version：模型版本标记（如 deepseek_v4），区分不同 MLA 布局 ------
    model_version: str | None = None
    # Marks draft groups that flatten a non-causal query block into decode rows.
    # ------【投机解码】non_causal_multi_token_decode：草稿组是否把非因果 query 块展平成 decode 行 ------
    non_causal_multi_token_decode: bool = False

    # ------【核心逻辑/对齐】__post_init__：父类初始化后按 alignment 对齐页大小 ------
    def __post_init__(self):
        super().__post_init__()
        _apply_alignment_padding(self)

    # ------【MLA/显存 profiling】storage_block_size：实际存储 token 数 = block_size // compress_ratio ------
    @property
    def storage_block_size(self) -> int:
        return self.block_size // self.compress_ratio

    # ------【MLA/显存 profiling】real_page_size_bytes：按 cache_dtype/量化模式算页字节，deepseek_v4 fp8 走 584B/token ------
    @property
    def real_page_size_bytes(self) -> int:
        if self.cache_dtype_str == "fp8_ds_mla":
            if self.model_version == "deepseek_v4":
                # DeepseekV4: 448B NoPE + 128B RoPE + 8B fp8 scale = 584B per token.
                # head_size stays semantic (512); bytes are determined here.
                return self.storage_block_size * 584
            # V3.2 main MLA: 656-byte custom layout (kv_lora_rank=512 +
            # qk_rope_head_dim=64, head_size=576). See flashmla_sparse.py.
            return self.block_size * 656
        if self.kv_quant_mode == KVQuantMode.INT4_PER_TOKEN_HEAD:
            head_dim = self.head_size // 2
        else:
            head_dim = self.head_size
        return (
            self.storage_block_size
            * self.num_kv_heads
            * head_dim
            * get_dtype_size(self.dtype)
        )

    # ------【核心逻辑】merge：合并 MLA spec，量化/压缩比/版本/块步长须一致 ------
    @classmethod
    def merge(cls, specs: list[Self]) -> Self:
        assert all(isinstance(spec, MLAAttentionSpec) for spec in specs), (
            "All attention layers in the same KV cache group must be MLAAttentionSpec."
        )
        cache_dtype_str_set = set(spec.cache_dtype_str for spec in specs)
        compress_ratio_set = set(spec.compress_ratio for spec in specs)
        model_version_set = set(spec.model_version for spec in specs)
        block_stride_set = set(spec.indexes_kv_by_block_stride for spec in specs)
        assert (
            len(cache_dtype_str_set) == 1
            and len(compress_ratio_set) == 1
            and len(model_version_set) == 1
            and len(block_stride_set) == 1
        ), (
            "All attention layers in the same KV cache group must use the same "
            "quantization method, compress ratio, model version, and KV block "
            "stride indexing."
        )
        merged_spec = cls(
            block_size=specs[0].block_size,
            num_kv_heads=specs[0].num_kv_heads,
            head_size=specs[0].head_size,
            dtype=specs[0].dtype,
            kv_quant_mode=specs[0].kv_quant_mode,
            page_size_padded=specs[0].page_size_padded,
            indexes_kv_by_block_stride=block_stride_set.pop(),
            cache_dtype_str=cache_dtype_str_set.pop(),
            compress_ratio=compress_ratio_set.pop(),
            model_version=model_version_set.pop(),
            non_causal_multi_token_decode=any(
                spec.non_causal_multi_token_decode for spec in specs
            ),
        )
        for spec in specs:
            for f in fields(AttentionSpec):
                assert getattr(spec, f.name) == getattr(merged_spec, f.name), (
                    "All attention layers in the same KV cache group must have "
                    "the same attention spec."
                )
        return merged_spec


# ------【核心逻辑】HiddenStateCacheSpec：隐藏状态缓存层标记，供 extract_hidden_states 使用 ------
@dataclass(frozen=True, kw_only=True)
class HiddenStateCacheSpec(MLAAttentionSpec):
    """Marker for hidden-state cache layers used by extract_hidden_states."""

    pass


# ------【滑动窗口/显存 profiling】RSWASpec：Reference 滑动窗口 spec，prefill token 全局可见，只保留最近 rswa_window 生成 token ------
@dataclass(frozen=True, kw_only=True)
class RSWASpec(FullAttentionSpec):
    """KV cache spec for Reference Sliding Window Attention (R-SWA).

    Prefill (image + text prompt) tokens are always globally visible.
    Only the last ``rswa_window`` generated tokens are kept in the KV cache;
    gap blocks (between the prefill tail and the current decode window) are
    evicted during each decode step to bound memory at
    O(prefix_blocks + window_blocks).
    """

    # ------【滑动窗口/显存 profiling】rswa_window：保留的生成 token 窗口大小，控制显存 O(prefix+window) ------
    rswa_window: int

    # ------【核心逻辑】merge：合并 RSWA spec，rswa_window 须一致，公共字段委托父类合并后回填 ------
    @classmethod
    def merge(cls, specs: list[RSWASpec]) -> RSWASpec:
        assert all(isinstance(spec, RSWASpec) for spec in specs), (
            "All attention layers in the same KV cache group must be RSWASpec."
        )
        rswa_windows = {spec.rswa_window for spec in specs}
        assert len(rswa_windows) == 1, (
            f"All R-SWA layers must share the same rswa_window, got {rswa_windows}"
        )
        # Delegate common field merging to the parent, then reattach rswa_window.
        base = FullAttentionSpec.merge(specs)  # type: ignore[arg-type]
        return cls(
            block_size=base.block_size,
            num_kv_heads=base.num_kv_heads,
            head_size=base.head_size,
            head_size_v=base.head_size_v,
            dtype=base.dtype,
            kv_quant_mode=base.kv_quant_mode,
            page_size_padded=base.page_size_padded,
            indexes_kv_by_block_stride=base.indexes_kv_by_block_stride,
            sliding_window=base.sliding_window,
            attention_chunk_size=base.attention_chunk_size,
            non_causal=base.non_causal,
            rswa_window=rswa_windows.pop(),
        )


# ------【chunked prefill/显存 profiling】ChunkedLocalAttentionSpec：分块局部注意力 spec，只保留一个 chunk 窗口的 KV 省显存 ------
@dataclass(frozen=True, kw_only=True)
class ChunkedLocalAttentionSpec(AttentionSpec):
    # ------【chunked prefill】attention_chunk_size：局部注意力窗口大小 ------
    attention_chunk_size: int

    # ------【chunked prefill/显存 profiling】max_admission_blocks_per_request：单请求准入块数=chunk 窗口+在途 token ------
    def max_admission_blocks_per_request(
        self, max_in_flight_tokens: int, max_model_len: int
    ) -> int:
        """Per-request admission cap, in blocks.

        Single source of truth for both startup pool sizing
        (`max_memory_usage_bytes`) and the runtime admission gate, so requests
        admitted by startup can also be admitted at runtime.

        `max_in_flight_tokens` is the max tokens scheduled but not yet settled
        (one batch per concurrent step); see `VllmConfig.max_in_flight_tokens`.
        """
        # During chunked prefill, we hold KV for at most one chunk window plus
        # the in-flight tokens, since frees happen on the processed-token basis.
        num_tokens = min(
            self.attention_chunk_size + max_in_flight_tokens, max_model_len
        )
        return cdiv(num_tokens, self.block_size)

    # ------【显存 profiling】max_memory_usage_bytes：按准入块数×页字节算池容量 ------
    def max_memory_usage_bytes(self, vllm_config: VllmConfig) -> int:
        max_blocks = self.max_admission_blocks_per_request(
            max_in_flight_tokens=vllm_config.max_in_flight_tokens,
            max_model_len=vllm_config.model_config.max_model_len,
        )
        return max_blocks * self.page_size_bytes

    # ------【核心逻辑】is_uniform_with_collection：所有层须同为 ChunkedLocalAttentionSpec 且 chunk 大小一致 ------
    def is_uniform_with_collection(
        self, kv_cache_specs: dict[str, KVCacheSpec]
    ) -> bool:
        return all(
            isinstance(spec, ChunkedLocalAttentionSpec)
            and spec.attention_chunk_size == self.attention_chunk_size
            for spec in kv_cache_specs.values()
        )


# ------【滑动窗口/显存 profiling】SlidingWindowSpec：滑动窗口注意力 spec，只保留最近 sliding_window 个 token 的 KV ------
@dataclass(frozen=True, kw_only=True)
class SlidingWindowSpec(AttentionSpec):
    # ------【滑动窗口】sliding_window：滑动窗口大小 ------
    sliding_window: int
    # ------【MQA/GQA】head_size_v：V 头维度，缺省回填 head_size ------
    head_size_v: int = None  # type: ignore[assignment]

    # ------【核心逻辑】__post_init__：head_size_v 缺省时回填为 head_size ------
    def __post_init__(self):
        if self.head_size_v is None:
            object.__setattr__(self, "head_size_v", self.head_size)

    # ------【显存 profiling/量化】real_page_size_bytes：含 V 头的页字节，nvfp4 单独处理维度 ------
    @property
    def real_page_size_bytes(self) -> int:
        # Mirror ``FullAttentionSpec.real_page_size_bytes`` for NVFP4 KV cache.
        if self.kv_quant_mode.is_nvfp4:
            last_dim = nvfp4_kv_cache_full_dim(
                self.head_size
            ) + nvfp4_kv_cache_full_dim(self.head_size_v)
            return (
                self.block_size
                * self.num_kv_heads
                * last_dim
                * get_dtype_size(self.dtype)
            )
        return (
            self.block_size
            * self.num_kv_heads
            * (self.head_size + self.head_size_v)
            * get_dtype_size(self.dtype)
        )

    # ------【滑动窗口/显存 profiling】max_admission_blocks_per_request：准入块数=窗口-1+在途 token，+1 因窗口可不落在块边界 ------
    def max_admission_blocks_per_request(
        self, max_in_flight_tokens: int, max_model_len: int
    ) -> int:
        """Per-request admission cap, in blocks.

        Single source of truth for both startup pool sizing
        (`max_memory_usage_bytes`) and the runtime admission gate. Per-request
        real-held blocks plateau at this bound because
        `SlidingWindowManager.remove_skipped_blocks` runs from `allocate_slots`
        before each chunk's `get_num_blocks_to_allocate`.

        `max_in_flight_tokens` is the max tokens scheduled but not yet settled
        (one batch per concurrent step); see `VllmConfig.max_in_flight_tokens`.
        """
        # During chunked prefill, we hold KV for the last `sliding_window-1`
        # computed tokens plus the in-flight tokens (frees happen on the
        # processed-token basis); never more than `max_model_len`.
        num_tokens = min(self.sliding_window - 1 + max_in_flight_tokens, max_model_len)
        # +1 because the sliding window may not start from the beginning of
        # the block. E.g. block size 4 and num_token 4 needs two blocks
        # [XXCD][EF] to store the 6-token window [CDEF].
        return cdiv(num_tokens, self.block_size) + 1

    # ------【显存 profiling/PD 分离】max_memory_usage_bytes：滑动窗口不支持 DCP，按准入块数×页字节 ------
    def max_memory_usage_bytes(self, vllm_config: VllmConfig) -> int:
        assert vllm_config.parallel_config.decode_context_parallel_size == 1, (
            "DCP not support sliding window."
        )
        max_blocks = self.max_admission_blocks_per_request(
            max_in_flight_tokens=vllm_config.max_in_flight_tokens,
            max_model_len=vllm_config.model_config.max_model_len,
        )
        return max_blocks * self.page_size_bytes

    # ------【核心逻辑】is_uniform_with_collection：所有层须同为 SlidingWindowSpec 且窗口一致 ------
    def is_uniform_with_collection(
        self, kv_cache_specs: dict[str, KVCacheSpec]
    ) -> bool:
        return all(
            isinstance(spec, SlidingWindowSpec)
            and spec.sliding_window == self.sliding_window
            for spec in kv_cache_specs.values()
        )


# ------【滑动窗口/MLA/显存 profiling】SlidingWindowMLASpec：滑动窗口+MLA 组合 spec，低秩压缩同时限制窗口省显存 ------
@dataclass(frozen=True, kw_only=True)
class SlidingWindowMLASpec(SlidingWindowSpec):
    """Sliding window attention with MLA cache format."""

    # ------【显存 profiling/量化】cache_dtype_str：缓存 dtype 字符串（如 fp8_ds_mla） ------
    cache_dtype_str: str | None = None
    # DeepseekV4-only: see MLAAttentionSpec.model_version.
    # ------【显存 profiling/对齐】alignment：页字节对齐粒度（DeepseekV4 专用） ------
    alignment: int | None = None  # Default to None for no padding.
    # ------【MLA/显存 profiling】compress_ratio：KV 压缩比 ------
    compress_ratio: int = 1
    # ------【核心逻辑】model_version：模型版本标记（如 deepseek_v4） ------
    model_version: str | None = None

    # ------【核心逻辑/对齐】__post_init__：按 alignment 对齐页大小 ------
    def __post_init__(self):
        _apply_alignment_padding(self)

    # ------【MLA/显存 profiling】storage_block_size：实际存储 token 数 = block_size // compress_ratio ------
    @property
    def storage_block_size(self) -> int:
        return self.block_size // self.compress_ratio

    # ------【MLA/显存 profiling】real_page_size_bytes：deepseek_v4 fp8 走 584B/token，否则按元素大小公式 ------
    @property
    def real_page_size_bytes(self) -> int:
        if self.model_version == "deepseek_v4" and self.cache_dtype_str == "fp8_ds_mla":
            # DeepseekV4 FlashMLA: 448B NoPE + 128B RoPE + 8B fp8 scale = 584B
            # per token. FlashInfer's contiguous bf16/fp8 cache falls through to
            # the element-size formula below.
            return self.storage_block_size * 584
        assert self.model_version in (None, "deepseek_v4"), (
            f"Unsupported model version: {self.model_version}"
        )
        return (
            self.storage_block_size
            * self.num_kv_heads
            * self.head_size
            * get_dtype_size(self.dtype)
        )

    # ------【核心逻辑】merge：合并 SlidingWindowMLA spec，量化/压缩比/版本/窗口/块步长须一致 ------
    @classmethod
    def merge(cls, specs: list[Self]) -> Self:
        assert all(isinstance(spec, SlidingWindowMLASpec) for spec in specs), (
            "All attention layers in the same KV cache group must be "
            "SlidingWindowMLASpec."
        )
        cache_dtype_str_set = set(spec.cache_dtype_str for spec in specs)
        compress_ratio_set = set(spec.compress_ratio for spec in specs)
        model_version_set = set(spec.model_version for spec in specs)
        sliding_window_set = set(spec.sliding_window for spec in specs)
        block_stride_set = set(spec.indexes_kv_by_block_stride for spec in specs)
        assert (
            len(cache_dtype_str_set) == 1
            and len(compress_ratio_set) == 1
            and len(model_version_set) == 1
            and len(sliding_window_set) == 1
            and len(block_stride_set) == 1
        ), (
            "All attention layers in the same KV cache group must use the same "
            "quantization method, compress ratio, model version, sliding "
            "window size, and KV block stride indexing."
        )
        return cls(
            block_size=specs[0].block_size,
            num_kv_heads=specs[0].num_kv_heads,
            head_size=specs[0].head_size,
            dtype=specs[0].dtype,
            page_size_padded=specs[0].page_size_padded,
            indexes_kv_by_block_stride=block_stride_set.pop(),
            sliding_window=sliding_window_set.pop(),
            cache_dtype_str=cache_dtype_str_set.pop(),
            compress_ratio=compress_ratio_set.pop(),
            model_version=model_version_set.pop(),
        )

    # ------【核心逻辑】is_uniform_with_collection：所有层须同为 SlidingWindowMLASpec 且窗口一致 ------
    def is_uniform_with_collection(
        self, kv_cache_specs: dict[str, KVCacheSpec]
    ) -> bool:
        return all(
            isinstance(spec, SlidingWindowMLASpec)
            and spec.sliding_window == self.sliding_window
            for spec in kv_cache_specs.values()
        )


@dataclass(frozen=True)
class MambaSpec(KVCacheSpec):
    shapes: tuple[tuple[int, ...], ...]
    dtypes: tuple[torch.dtype]
    page_size_padded: int | None = None
    mamba_type: MambaAttentionBackendEnum = MambaAttentionBackendEnum.MAMBA2
    mamba_cache_mode: str = "none"
    num_speculative_blocks: int = 0

    @property
    def page_size_bytes(self) -> int:
        page_size = sum(
            prod(shape) * get_dtype_size(dtype)
            for (shape, dtype) in zip(self.shapes, self.dtypes)
        )
        if self.page_size_padded is not None:
            assert self.page_size_padded >= page_size
            return self.page_size_padded
        return page_size

    def max_memory_usage_bytes(self, vllm_config: VllmConfig) -> int:
        if vllm_config.cache_config.mamba_cache_mode == "all":
            max_model_len = vllm_config.model_config.max_model_len
            return (
                cdiv(max_model_len, self.block_size) + self.num_speculative_blocks
            ) * self.page_size_bytes
        elif vllm_config.cache_config.mamba_cache_mode == "align":
            return self.page_size_bytes * (2 + self.num_speculative_blocks)
        else:
            return self.page_size_bytes * (1 + self.num_speculative_blocks)

    def max_num_blocks_per_req(self, vllm_config: VllmConfig, max_len: int) -> int:
        # Mamba state is replicated across DCP/PCP ranks, never sharded, so
        # no CP scaling applies.
        if vllm_config.cache_config.mamba_cache_mode == "align":
            # Block table rows are position-indexed over the full sequence
            # even though only 2 + num_speculative_blocks state blocks are
            # resident at a time (earlier states are nulled out by
            # remove_skipped_blocks), so the row length must cover max_len
            # rather than max_memory_usage_bytes.
            return cdiv(max_len, self.block_size) + self.num_speculative_blocks
        return cdiv(self.max_memory_usage_bytes(vllm_config), self.page_size_bytes)

    def is_uniform_with_collection(
        self, kv_cache_specs: dict[str, KVCacheSpec]
    ) -> bool:
        return all(
            isinstance(spec, MambaSpec)
            and spec.num_speculative_blocks == self.num_speculative_blocks
            for spec in kv_cache_specs.values()
        )


@dataclass(frozen=True)
class EncoderOnlyAttentionSpec(AttentionSpec):
    def max_memory_usage_bytes(self, vllm_config: VllmConfig) -> int:
        # Encoder-only layers do not need KV cache
        return 0


@dataclass(frozen=True)
class CrossAttentionSpec(AttentionSpec):
    """
    KV cache spec for cross-attention layers in encoder-decoder models.
    """

    def max_memory_usage_bytes(self, vllm_config: VllmConfig) -> int:
        # For cross-attention, we need to cache encoder states
        # Get encoder length (e.g., 1500 for Whisper).
        max_encoder_len = vllm_config.scheduler_config.max_num_encoder_input_tokens
        return cdiv(max_encoder_len, self.block_size) * self.page_size_bytes


@dataclass(frozen=True)
class SinkFullAttentionSpec(FullAttentionSpec):
    sink_len: int | None = None

    @classmethod
    def merge(cls, specs: list[Self]) -> Self:
        """
        Merge a list of FullAttentionSpec objects into a single
        FullAttentionSpec object.
        """
        assert all(isinstance(spec, FullAttentionSpec) for spec in specs), (
            "All attention layers in the same KV cache group must be FullAttentionSpec."
        )

        sliding_window = set(
            spec.sliding_window for spec in specs if spec.sliding_window is not None
        )
        attention_chunk_size = set(
            spec.attention_chunk_size
            for spec in specs
            if spec.attention_chunk_size is not None
        )
        assert not any(isinstance(spec, MLAAttentionSpec) for spec in specs), (
            "MLAAttentionSpec should be merged in MLAAttentionSpec.merge"
        )
        merged_spec = cls(
            block_size=specs[0].block_size,
            num_kv_heads=specs[0].num_kv_heads,
            head_size=specs[0].head_size,
            head_size_v=specs[0].head_size_v,
            sink_len=specs[0].sink_len,
            dtype=specs[0].dtype,
            kv_quant_mode=specs[0].kv_quant_mode,
            page_size_padded=specs[0].page_size_padded,
            indexes_kv_by_block_stride=specs[0].indexes_kv_by_block_stride,
            sliding_window=cls.merge_window_sizes(sliding_window),
            attention_chunk_size=cls.merge_window_sizes(attention_chunk_size),
            non_causal=any(spec.non_causal for spec in specs),
        )
        for spec in specs:
            for f in fields(AttentionSpec):
                assert getattr(spec, f.name) == getattr(merged_spec, f.name), (
                    "All attention layers in the same KV cache group must have "
                    "the same attention spec."
                )
        assert (merged_spec.sliding_window is not None) + (
            merged_spec.attention_chunk_size is not None
        ) <= 1, (
            "Model with both sliding window layers and chunked local attention "
            "layers is not supported."
        )
        return merged_spec


@dataclass(frozen=True)
class UniformTypeKVCacheSpecs(KVCacheSpec):
    """
    A KV cache spec for multiple layers with the same type of attention. Here,
    same types means always need the same number of token slots. For example,
    sliding window attentions with different window sizes are not the same type
    and should not be merged into one UniformTypeKVCacheSpecs.
    """

    kv_cache_specs: dict[str, KVCacheSpec]

    @property
    def page_size_bytes(self) -> int:
        return sum(spec.page_size_bytes for spec in self.kv_cache_specs.values())

    def max_memory_usage_bytes(self, vllm_config: VllmConfig) -> int:
        max_num_pages = max(
            cdiv(spec.max_memory_usage_bytes(vllm_config), spec.page_size_bytes)
            for spec in self.kv_cache_specs.values()
        )
        return max_num_pages * self.page_size_bytes

    def max_num_blocks_per_req(self, vllm_config: VllmConfig, max_len: int) -> int:
        # Metadata builders are constructed from the per-layer spec, so the base
        # cdiv(max_len, block_size) would drop its DCP sharding and size the
        # block table wider than those builders expect.
        widths = {
            spec.max_num_blocks_per_req(vllm_config, max_len)
            for spec in self.kv_cache_specs.values()
        }
        assert len(widths) == 1, (
            "All layers in the same KV cache group must need the same number "
            f"of block table entries, got {sorted(widths)}."
        )
        return next(iter(widths))

    @classmethod
    def is_uniform_type(cls, kv_cache_specs: dict[str, KVCacheSpec]) -> bool:
        """
        Whether all layers have the same type of KV cache spec.

        Uses the registry to determine grouping base classes, so custom specs
        that inherit from FullAttentionSpec are treated as full attention.
        """
        block_sizes = set(spec.block_size for spec in kv_cache_specs.values())
        if len(block_sizes) > 1:
            # Different block sizes, not uniform.
            return False
        first_spec = next(iter(kv_cache_specs.values()))
        return first_spec.is_uniform_with_collection(kv_cache_specs)

    @classmethod
    def from_specs(cls, kv_cache_specs: dict[str, KVCacheSpec]) -> Self | None:
        """
        Return a SameTypeKVCacheSpecs object if all layers have the same type
        of KV cache spec. Return None if not.
        """
        if cls.is_uniform_type(kv_cache_specs):
            block_size = next(iter(kv_cache_specs.values())).block_size
            return cls(block_size=block_size, kv_cache_specs=kv_cache_specs)
        else:
            return None

    # NOTE: below util functions are only used by DeepseekV4 for now.
    def get_page_sizes(self) -> list[int]:
        return list(set(spec.page_size_bytes for spec in self.kv_cache_specs.values()))

    def get_num_layer_tuples(self) -> int:
        return Counter(
            spec.page_size_bytes for spec in self.kv_cache_specs.values()
        ).most_common(1)[0][1]

    def max_memory_usage_pages(self, vllm_config: VllmConfig) -> int:
        return max(
            cdiv(spec.max_memory_usage_bytes(vllm_config), spec.page_size_bytes)
            for spec in self.kv_cache_specs.values()
        )


def get_kv_cache_spec_kind(kv_cache_spec: KVCacheSpec) -> KVCacheSpecKind:
    if isinstance(kv_cache_spec, UniformTypeKVCacheSpecs):
        inner_kinds = {
            get_kv_cache_spec_kind(spec)
            for spec in kv_cache_spec.kv_cache_specs.values()
        }
        if len(inner_kinds) == 1:
            return next(iter(inner_kinds))
        return KVCacheSpecKind.UNKNOWN
    # Keep subclass checks before base classes so specialized specs keep their
    # more precise kind.
    if isinstance(kv_cache_spec, SlidingWindowMLASpec):
        return KVCacheSpecKind.SLIDING_WINDOW_MLA
    if isinstance(kv_cache_spec, MLAAttentionSpec):
        return KVCacheSpecKind.MLA_ATTENTION
    if isinstance(kv_cache_spec, SinkFullAttentionSpec):
        return KVCacheSpecKind.SINK_FULL_ATTENTION
    if isinstance(kv_cache_spec, FullAttentionSpec):
        return KVCacheSpecKind.FULL_ATTENTION
    if isinstance(kv_cache_spec, ChunkedLocalAttentionSpec):
        return KVCacheSpecKind.CHUNKED_LOCAL_ATTENTION
    if isinstance(kv_cache_spec, SlidingWindowSpec):
        return KVCacheSpecKind.SLIDING_WINDOW
    if isinstance(kv_cache_spec, MambaSpec):
        return KVCacheSpecKind.MAMBA
    if isinstance(kv_cache_spec, EncoderOnlyAttentionSpec):
        return KVCacheSpecKind.ENCODER_ONLY_ATTENTION
    if isinstance(kv_cache_spec, CrossAttentionSpec):
        return KVCacheSpecKind.CROSS_ATTENTION
    return KVCacheSpecKind.UNKNOWN


def get_kv_cache_spec_sliding_window(kv_cache_spec: KVCacheSpec) -> int | None:
    if isinstance(kv_cache_spec, UniformTypeKVCacheSpecs):
        inner_windows = {
            get_kv_cache_spec_sliding_window(spec)
            for spec in kv_cache_spec.kv_cache_specs.values()
        }
        return next(iter(inner_windows)) if len(inner_windows) == 1 else None
    if isinstance(kv_cache_spec, SlidingWindowSpec):
        return kv_cache_spec.sliding_window
    return None


@dataclass
class KVCacheTensor:
    """
    A class for specifying how the workers should initialize the KV cache.
    """

    size: int  # size of the KV cache tensor in bytes
    shared_by: list[str]  # layer names that share the same KV cache tensor
    offset: int = 0  # byte offset of this layer within a contiguous block
    block_stride: int = 0  # total bytes per block in a packed layout (0 = not packed)


@dataclass
class KVCacheGroupSpec:
    """
    Represents a group of model layers that share the same KV cache block table.
    These layers are regarded as one layer in the KV cache manager.
    """

    # The names of model layers in this group
    layer_names: list[str]
    # The KV cache spec of this manager layer
    kv_cache_spec: KVCacheSpec
    # Whether this group contains EAGLE/MTP draft attention layers.
    is_eagle_group: bool = False


@dataclass
class KVCacheConfig:
    """
    The KV cache configuration of a model.
    """

    num_blocks: int
    """The number of KV cache blocks"""
    kv_cache_tensors: list[KVCacheTensor]
    """How should model runner initialize the KV cache tensors for each layer"""
    kv_cache_groups: list[KVCacheGroupSpec]
    """
    The kv cache groups of the model.
    For models with only one type of attention, there is only one group that
    contains all layers.
    For models with multiple types of attention, there will be multiple groups,
    see `_get_kv_cache_config_uniform_page_size` for more details.
    """

    @property
    def has_mamba_layers(self) -> bool:
        return any(isinstance(g.kv_cache_spec, MambaSpec) for g in self.kv_cache_groups)

    @property
    def has_mixed_precision_kv_cache(self) -> bool:
        """Whether attention groups store their KV cache at more than one precision."""
        kv_cache_precisions: set[tuple[torch.dtype, KVQuantMode]] = set()
        for group in self.kv_cache_groups:
            group_spec = group.kv_cache_spec
            group_specs = (
                list(group_spec.kv_cache_specs.values())
                if isinstance(group_spec, UniformTypeKVCacheSpecs)
                else [group_spec]
            )
            kv_cache_precisions.update(
                (spec.dtype, spec.kv_quant_mode)
                for spec in group_specs
                if isinstance(spec, AttentionSpec)
            )
        return len(kv_cache_precisions) > 1

    @property
    def needs_kv_cache_zeroing(self) -> bool:
        """Whether newly allocated KV cache blocks must be zeroed before use.

        Required for Mamba layers, whose state is read before it is fully written
        (#35219), and for mixed-precision caches, where a block reused across
        groups can be reinterpreted under a different precision and decode stale
        bytes to NaN/Inf. Uniform-precision caches skip zeroing.
        """
        return self.has_mamba_layers or self.has_mixed_precision_kv_cache
