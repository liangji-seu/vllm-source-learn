# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import product as iprod
from typing import Any

import numpy as np
import torch

from vllm.config import CacheConfig, VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.models.interfaces import MultiModalEmbeddings
from vllm.model_executor.models.utils import extract_layer_index
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import largest_power_of_2_divisor
from vllm.utils.mem_utils import MemorySnapshot, format_gib
from vllm.utils.torch_utils import async_tensor_h2d
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionMetadataBuilder,
    MultipleOf,
)
from vllm.v1.core.kv_cache_utils import KVCacheBlockCopy
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    EncoderOnlyAttentionSpec,
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheSpec,
    MambaSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.worker.block_table import get_block_table_width

logger = init_logger(__name__)


def raise_if_nan_logits(num_nans_in_logits: Mapping[str, int]) -> None:
    # ------【核心逻辑】全为 0 表示无 NaN，直接返回避免构建错误信息 ------
    if not any(num_nans_in_logits.values()):
        return

    # ------【核心逻辑】过滤出存在 NaN 的请求，构造 req_id -> nan 数量的映射 ------
    corrupted_requests = {
        req_id: num_nans
        for req_id, num_nans in num_nans_in_logits.items()
        if num_nans > 0
    }
    # ------【核心逻辑】存在 NaN 时抛出带请求明细的运行时错误 ------
    raise RuntimeError(f"NaNs detected in logits: {corrupted_requests}")


@triton.jit(do_not_specialize=["n_blocks"])
def _zero_kv_blocks_kernel(
    seg_addrs_ptr,
    seg_page_sizes_ptr,
    block_ids_ptr,
    n_blocks,
    N_SEGS: tl.constexpr,
    MAX_CHUNKS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Zero KV cache blocks across all segments in a single launch.

    Each segment is a contiguous region of one block's data.  For backends
    where blocks are outermost (block_dim=0) there is one segment per
    buffer.  For backends where K/V is outermost (block_dim=1) there are
    two segments per buffer (one for K, one for V).

    Segments may have different page sizes (e.g. models with multiple KV
    cache groups like MLA + DSA indexer).  Each segment's page size is
    read from seg_page_sizes_ptr; programs whose chunk_index falls beyond
    their segment's page size early-exit.

    seg_addrs_ptr holds absolute byte addresses (int64) for each segment,
    allowing segments to live in different CUDA allocations.

    Programs are mapped as (block_index, seg_index, chunk_index).
    """
    # ------【内存池/CuMem】把一维 program id 解构为 (block, seg, chunk) 三维工作项索引 ------
    pid = tl.program_id(0)
    work_per_block = N_SEGS * MAX_CHUNKS
    # ------【内存池/CuMem】block 索引越界说明该 pid 无对应 block，提前退出 ------
    block_index = pid // work_per_block
    if block_index >= n_blocks:
        return
    # ------【内存池/CuMem】用余数继续解出 seg 与 chunk 索引 ------
    remainder = pid % work_per_block
    seg_index = remainder // MAX_CHUNKS
    chunk_index = remainder % MAX_CHUNKS
    # ------【内存池/CuMem】读取该 segment 的页大小，chunk 越过页范围则提前退出 ------
    page_size_el = tl.load(seg_page_sizes_ptr + seg_index)
    if chunk_index >= page_size_el // BLOCK_SIZE:
        return
    # ------【内存池/CuMem】读取逻辑 block id 与 segment 的绝对地址 ------
    block_id = tl.load(block_ids_ptr + block_index)
    seg_addr = tl.load(seg_addrs_ptr + seg_index)
    # ------【内存池/CuMem】把 uint64 地址强转为 int32 指针用于写 0 ------
    ptr = tl.cast(seg_addr, tl.pointer_type(tl.int32))
    # ------【内存池/CuMem】按 block 逻辑大小 + chunk 偏移计算目标元素偏移量 ------
    offset = (
        block_id.to(tl.int64) * page_size_el.to(tl.int64)
        + chunk_index.to(tl.int64) * BLOCK_SIZE
    )
    # ------【内存池/CuMem】用向量化列索引一次写入 BLOCK_SIZE 个零 ------
    cols = tl.arange(0, BLOCK_SIZE).to(tl.int64)
    tl.store(ptr + offset + cols, tl.zeros([BLOCK_SIZE], dtype=tl.int32))


class KVBlockZeroer:
    """Manages efficient zeroing of KV cache blocks via a Triton kernel.

    Construct once after KV caches are allocated to precompute segment
    addresses, then call :meth:`zero_block_ids` each step to zero
    newly-allocated blocks.
    """

    def __init__(
        self,
        device: torch.device,
        attn_groups_iter: Iterable["AttentionGroup"],
        kernel_block_sizes: list[int],
        cache_dtype: str,
        static_forward_context: dict[str, Any],
        runner_only_attn_layers: set[str] | None = None,
    ) -> None:
        """Precompute the absolute-address table for the Triton zeroing kernel.

        Each entry is the absolute byte address of a segment start on the
        GPU, so segments in different CUDA allocations work correctly.

        Block IDs from the scheduler reference logical blocks whose size
        may differ from the kernel block size (virtual block splitting).
        Each segment's page_size_el accounts for this ratio so that
        ``block_id * page_size_el`` lands at the correct offset.

        Only AttentionSpec layers are processed; Mamba layers are skipped.
        """
        # ------【内存池/CuMem】保存设备并先置元信息为空，待扫描完再组装 ------
        self.device = device
        self._meta: tuple[torch.Tensor, torch.Tensor, int, int, int] | None = None

        # ------【内存池/CuMem】初始化 runner-only 集合与去重/地址/页大小三张表 ------
        if runner_only_attn_layers is None:
            runner_only_attn_layers = set()
        seen_ptrs: set[int] = set()
        seg_addrs: list[int] = []
        seg_page_sizes: list[int] = []

        # ------【内存池/CuMem】遍历每个 attention group，只为 FullAttentionSpec 建清零段 ------
        for group in attn_groups_iter:
            spec = group.kv_cache_spec
            if not isinstance(spec, FullAttentionSpec):
                continue
            # ------【内存池/CuMem】跳过没有对应 kernel block size 的组 ------
            if group.kv_cache_group_id >= len(kernel_block_sizes):
                continue
            # ------【内存池/CuMem】由 manager 块大小与 kernel 块大小之比得到虚拟分块比例 ------
            kernel_bs = kernel_block_sizes[group.kv_cache_group_id]
            ratio = spec.block_size // kernel_bs
            # ------【内存池/CuMem】询问 backend 块维度（block_dim=0 或 1）以定位块步长 ------
            block_dim = group.backend.get_kv_cache_block_dim(
                kernel_bs,
                spec.num_kv_heads,
                spec.head_size,
                cache_dtype_str=cache_dtype,
            )

            # ------【内存池/CuMem】遍历组内每个注意力层，跳过 runner-only 层 ------
            for layer_name in group.layer_names:
                if layer_name in runner_only_attn_layers:
                    continue
                # ------【内存池/CuMem】取出该层 KV cache，非张量（如 Mamba 状态）则跳过 ------
                kv = static_forward_context[layer_name].kv_cache
                if not isinstance(kv, torch.Tensor):
                    continue
                # ------【内存池/CuMem】按底层 data_ptr 去重，共享同一存储的层只登记一次 ------
                dp = kv.data_ptr()
                if dp in seen_ptrs:
                    continue
                seen_ptrs.add(dp)

                # ------【内存池/CuMem】由元素大小与块维度步长算出每个 kernel block 的元素数 ------
                el = kv.element_size()
                cur_bytes = kv.stride(block_dim) * el
                assert cur_bytes % 4 == 0
                kernel_block_el = cur_bytes // 4
                cur_page_el = kernel_block_el * ratio

                # ------【内存池/CuMem】找出比块步长更大的外层维度，用于枚举所有 segment 起始地址 ------
                block_stride_bytes = cur_bytes
                outer_dims = [
                    d
                    for d in range(block_dim)
                    if kv.stride(d) * el > block_stride_bytes
                ]
                outer_strides = [kv.stride(d) * el for d in outer_dims]
                # ------【内存池/CuMem】笛卡尔枚举外层偏移，登记每个 segment 的绝对地址与页大小 ------
                for outer in iprod(*(range(kv.shape[d]) for d in outer_dims)):
                    off_bytes = sum(i * s for i, s in zip(outer, outer_strides))
                    seg_addrs.append(dp + off_bytes)
                    seg_page_sizes.append(cur_page_el)

        # ------【内存池/CuMem】没有任何 segment 时保持元信息为空并返回 ------
        if not seg_addrs:
            self._meta = None
            return

        # ------【内存池/CuMem】统一 chunk 大小为各页大小公因数（上限 1024），保证一次发射覆盖全部 ------
        max_page_size_el = max(seg_page_sizes)
        blk_size = min(
            min(largest_power_of_2_divisor(ps) for ps in seg_page_sizes),
            1024,
        )
        # ------【内存池/CuMem】把地址/页大小搬上 GPU 张量，连同 chunk 数与段数一起缓存 ------
        self._meta = (
            torch.tensor(seg_addrs, dtype=torch.uint64, device=self.device),
            torch.tensor(seg_page_sizes, dtype=torch.int64, device=self.device),
            max_page_size_el // blk_size,
            blk_size,
            len(seg_addrs),
        )

    def zero_block_ids(self, block_ids: list[int]) -> None:
        """Zero the KV cache memory for the given block IDs."""
        # ------【内存池/CuMem】空列表或未初始化元信息时无事可做，直接返回 ------
        if not block_ids or self._meta is None:
            return
        # ------【内存池/CuMem】解包预计算的地址表、页大小、chunk 数与段数 ------
        seg_addrs, seg_page_sizes, max_chunks, blk_size, n_segs = self._meta
        n_blocks = len(block_ids)
        # ------【异步 RPC】把 block_ids 异步从 host 拷贝到 device 作为内核输入 ------
        idx = async_tensor_h2d(block_ids, device=self.device, dtype=torch.int64)
        # ------【内存池/CuMem】grid 覆盖所有 block x segment x chunk 的组合，一次清零 ------
        grid = (n_blocks * n_segs * max_chunks,)
        _zero_kv_blocks_kernel[grid](
            seg_addrs,
            seg_page_sizes,
            idx,
            n_blocks,
            N_SEGS=n_segs,
            MAX_CHUNKS=max_chunks,
            BLOCK_SIZE=blk_size,
        )

    def warmup(self, num_kv_blocks: int) -> None:
        """JIT-compile the zeroing kernel before the first real request."""
        # ------【核心逻辑】对零块做一次空跑预热，提前触发 Triton JIT 编译避免首个请求延迟 ------
        if num_kv_blocks > 0:
            self.zero_block_ids([0])


# ------【核心逻辑】按 backend 与 KV cache spec 把注意力层归组，统一驱动 metadata 构建与缓存绑定 ------
@dataclass
class AttentionGroup:
    backend: type[AttentionBackend]
    layer_names: list[str]
    kv_cache_spec: KVCacheSpec
    kv_cache_group_id: int
    # ------【CUDA Graph】每个 ubatch 一个 metadata builder，隔离 cudagraph 持久缓冲避免冲突 ------
    # When ubatching is enabled we will have a metadata builder for each ubatch
    # so that if they use internal persistent buffers for cudagraphs, and they
    # won't have to worry about conflicting with the other ubatches.
    metadata_builders: list[AttentionMetadataBuilder] = field(
        default_factory=lambda: []
    )

    def create_metadata_builders(
        self,
        vllm_config,
        device,
        kernel_block_size: int | None = None,
        num_metadata_builders: int = 1,
    ):
        # ------【CUDA Graph】指定 kernel block size 时复制 spec 并替换块大小（虚拟分块） ------
        kv_cache_spec_builder = (
            self.kv_cache_spec.copy_with_new_block_size(kernel_block_size)
            if kernel_block_size is not None
            else self.kv_cache_spec
        )
        # ------【CUDA Graph】取该 backend 对应的 metadata builder 类 ------
        builder_cls = self.backend.get_builder_cls()
        builder_kwargs = {}
        # ------【核心逻辑】builder 需要 block_table 宽度时先算出每请求最大块数 ------
        if builder_cls.requires_block_table_width:
            max_num_blocks = self.kv_cache_spec.max_num_blocks_per_req(
                vllm_config, vllm_config.model_config.max_model_len
            )
            # ------【核心逻辑】由最大块数与块大小计算 block_table 宽度传给 builder ------
            builder_kwargs["block_table_width"] = get_block_table_width(
                max_num_blocks, self.kv_cache_spec.block_size, kernel_block_size
            )
        # ------【CUDA Graph】实例化 num_metadata_builders 个 builder（对应 ubatch 数量） ------
        self.metadata_builders = [
            builder_cls(
                kv_cache_spec_builder,
                self.layer_names,
                vllm_config,
                device,
                **builder_kwargs,
            )
            for _ in range(num_metadata_builders)
        ]

    def get_metadata_builder(self, ubatch_id: int = 0) -> AttentionMetadataBuilder:
        # ------【CUDA Graph】按 ubatch_id 取对应的 metadata builder，越界先断言保护 ------
        assert len(self.metadata_builders) > ubatch_id
        return self.metadata_builders[ubatch_id]


def select_common_block_size(
    kv_manager_block_size: int,
    backends: list[type[AttentionBackend]],
) -> int:
    """
    Select a block size that is supported by all backends and is a factor of
    kv_manager_block_size.

    If kv_manager_block_size is supported by all backends, return it directly.
    Otherwise, return the max supported size.

    Args:
        kv_manager_block_size: Block size of KV cache.
        backends: List of attention backend classes.

    Returns:
        The selected block size.

    Raises:
        ValueError: If no valid block size found.
    """

    def block_size_is_supported(
        backends: list[type[AttentionBackend]], block_size: int
    ) -> bool:
        """Check if the block size is supported by all backends."""
        # ------【核心逻辑】逐个 backend 检查给定块大小是否被支持 ------
        for backend in backends:
            is_supported = False
            # ------【核心逻辑】遍历该 backend 声明的全部支持块大小 ------
            for supported_size in backend.get_supported_kernel_block_sizes():
                # ------【核心逻辑】int 格式要求块大小精确相等 ------
                if isinstance(supported_size, int):
                    if block_size == supported_size:
                        is_supported = True
                # ------【核心逻辑】MultipleOf 格式要求块大小是 base 的整数倍 ------
                elif isinstance(supported_size, MultipleOf):
                    if block_size % supported_size.base == 0:
                        is_supported = True
                else:
                    # ------【核心逻辑】遇到未知格式的支持大小直接报错 ------
                    raise ValueError(f"Unknown supported size: {supported_size}")
            # ------【核心逻辑】任一 backend 不支持即整体不支持 ------
            if not is_supported:
                return False
        return True

    # Case 1: if the block_size of kv cache manager is supported by all backends,
    # return it directly.
    # ------【核心逻辑】manager 块大小被所有 backend 支持时直接采用，避免额外分块 ------
    if block_size_is_supported(backends, kv_manager_block_size):
        return kv_manager_block_size

    # Case 2: otherwise, the block_size must be an `int`-format supported size of
    # at least one backend. Iterate over all `int`-format supported sizes in
    # descending order and return the first one that is supported by all backends.
    # Simple proof:
    # If the supported size b is in MultipleOf(x_i) format for all attention
    # backends i, and b a factor of kv_manager_block_size, then
    # kv_manager_block_size also satisfies MultipleOf(x_i) for all i. We will
    # return kv_manager_block_size in case 1.
    # ------【核心逻辑】收集所有 backend 的 int 格式支持大小用于候选搜索 ------
    all_int_supported_sizes = set(
        supported_size
        for backend in backends
        for supported_size in backend.get_supported_kernel_block_sizes()
        if isinstance(supported_size, int)
    )

    # ------【核心逻辑】降序找第一个能整除 manager 块且被全体 backend 支持的公共大小 ------
    for supported_size in sorted(all_int_supported_sizes, reverse=True):
        if kv_manager_block_size % supported_size != 0:
            continue
        if block_size_is_supported(backends, supported_size):
            return supported_size
    # ------【核心逻辑】找不到任何公共块大小时报错 ------
    raise ValueError(f"No common block size for {kv_manager_block_size}. ")


def prepare_kernel_block_sizes(
    kv_cache_config: KVCacheConfig, attn_groups: list[list[AttentionGroup]]
) -> list[int]:
    """
    Generate kernel_block_sizes that matches each block_size.

    For attention backends that support virtual block splitting,
    use the supported block sizes from the backend.
    For other backends (like Mamba), use the same block size (no splitting).

    Args:
        kv_cache_config: The KV cache configuration.
        attn_groups: Attention groups indexed by KV cache group id.

    Returns:
        List of kernel block sizes for each cache group.
    """
    # ------【核心逻辑】为每个 cache group 逐个算出对应的 kernel 块大小列表 ------
    kernel_block_sizes = []
    for kv_cache_gid, kv_cache_group in enumerate(kv_cache_config.kv_cache_groups):
        kv_cache_spec = kv_cache_group.kv_cache_spec
        # ------【核心逻辑】UniformType spec 内各层类型一致，取任意一个用于分发 ------
        if isinstance(kv_cache_spec, UniformTypeKVCacheSpecs):
            # All layers in the UniformTypeKVCacheSpecs have the same type,
            # pick an arbitrary one to dispatch.
            kv_cache_spec = next(iter(kv_cache_spec.kv_cache_specs.values()))
        # ------【核心逻辑】encoder-only spec 无 KV cache，跳过 ------
        if isinstance(kv_cache_spec, EncoderOnlyAttentionSpec):
            continue
        # ------【核心逻辑】attention spec 支持虚拟分块，按 backend 求公共 kernel 块大小 ------
        if isinstance(kv_cache_spec, AttentionSpec):
            # This is an attention backend that supports virtual block splitting.
            kv_manager_block_size = kv_cache_group.kv_cache_spec.block_size
            group_backends = [g.backend for g in attn_groups[kv_cache_gid]]
            selected_kernel_size = select_common_block_size(
                kv_manager_block_size, group_backends
            )
            kernel_block_sizes.append(selected_kernel_size)
        # ------【核心逻辑】Mamba 等非注意力缓存不参与分块，直接用自身块大小 ------
        elif isinstance(kv_cache_spec, MambaSpec):
            # This is likely Mamba or other non-attention cache, no splitting.
            kernel_block_sizes.append(kv_cache_spec.block_size)
        else:
            # ------【核心逻辑】遇到未知 spec 类型直接报错 ------
            raise NotImplementedError(
                f"unknown kv cache spec {kv_cache_group.kv_cache_spec}"
            )
    return kernel_block_sizes


def sanity_check_mm_encoder_outputs(
    mm_embeddings: MultiModalEmbeddings,
    expected_num_items: int,
) -> None:
    """
    Perform sanity checks for the result of
    [`vllm.model_executor.models.SupportsMultiModal.embed_multimodal`][].
    """
    # ------【核心逻辑】校验 embedding 类型必须是 2D tensor 列表/元组或单个 3D tensor ------
    assert isinstance(mm_embeddings, (list, tuple, torch.Tensor)), (
        "Expected multimodal embeddings to be a list/tuple of 2D tensors, "
        f"or a single 3D tensor, but got {type(mm_embeddings)} "
        "instead. This is most likely due to incorrect implementation "
        "of the model's `embed_multimodal` method."
    )

    # ------【核心逻辑】校验 embedding 数量与输入多模态 item 数量一致 ------
    assert len(mm_embeddings) == expected_num_items, (
        "Expected number of multimodal embeddings to match number of "
        f"input items: {expected_num_items}, but got {len(mm_embeddings)=} "
        "instead. This is most likely due to incorrect implementation "
        "of the model's `embed_multimodal` method."
    )

    # ------【核心逻辑】校验每个 embedding 都是 2D 张量 ------
    assert all(e.ndim == 2 for e in mm_embeddings), (
        "Expected multimodal embeddings to be a sequence of 2D tensors, "
        f"but got tensors with shapes {[e.shape for e in mm_embeddings]} "
        "instead. This is most likely due to incorrect implementation "
        "of the model's `embed_multimodal` method."
    )


def request_memory(init_snapshot: MemorySnapshot, cache_config: CacheConfig) -> int:
    """
    Calculate the amount of memory required by vLLM, then validate
    that the current amount of free memory is sufficient for that.

    计算vllm需要的显存大小，然后评估目前的空闲显存是否足够
    """
    # ------【显存 profiling】按 gpu_memory_utilization 占比计算 vllm 需要预留的显存大小 ------
    requested_memory = math.ceil(
        init_snapshot.total_memory * cache_config.gpu_memory_utilization # 这个是用户配置的显存使用比例
    )

    # ------【显存 profiling】校验当前空闲显存是否满足预留需求，不足则抛出显存不足错误 ------
    if init_snapshot.free_memory < requested_memory:
        raise ValueError(
            f"Free memory on device {init_snapshot.device_} "
            f"({format_gib(init_snapshot.free_memory)}/"
            f"{format_gib(init_snapshot.total_memory)} GiB) on startup "
            f"is less than desired GPU memory utilization "
            f"({cache_config.gpu_memory_utilization}, "
            f"{format_gib(requested_memory)} GiB). Decrease GPU memory "
            f"utilization or reduce GPU memory used by other processes."
        )

    return requested_memory


def add_kv_sharing_layers_to_kv_cache_groups(
    shared_kv_cache_layers: dict[str, str],
    kv_cache_groups: list[KVCacheGroupSpec],
    runner_only_attn_layers: set[str] | None = None,
) -> None:
    """
    Sets up KV cache sharing by reusing the allocated KV caches in `kv_caches`
    for layers that do not allocate its own KV cache, based on the mapping in
    `shared_kv_cache_layers`. Adds these layers to the corresponding KV cache
    group, which is needed to ensure that attention metadata is assigned later.

    Args:
        shared_kv_cache_layers: Layer pairings for cross-layer KV sharing.
            If an Attention layer `layer_name` is in the keys of this dict, it
            means this layer will perform attention using the keys and values
            from the KV cache of `shared_kv_cache_layers[layer_name]`.
        kv_cache_groups: The KV cache groups of the model.
    """
    # ------【显存 profiling】没有跨层 KV 共享配置时无需处理，直接返回 ------
    if not shared_kv_cache_layers:
        return

    # ------【显存 profiling】建立 layer_name -> KV cache group 的反查映射 ------
    layer_to_kv_cache_group: dict[str, KVCacheGroupSpec] = {}
    for kv_cache_group in kv_cache_groups:
        for layer_name in kv_cache_group.layer_names:
            layer_to_kv_cache_group[layer_name] = kv_cache_group

    # ------【显存 profiling】把共享层并入目标层所在 group，复用同一份 KV cache 省显存 ------
    for layer_name, target_layer_name in shared_kv_cache_layers.items():
        tgt_kv_cache_group = layer_to_kv_cache_group[target_layer_name]
        tgt_kv_cache_group.layer_names.append(layer_name)

        # ------【显存 profiling】记录为 runner-only 层，后续不为其单独分配/清零 KV cache ------
        if runner_only_attn_layers is not None:
            runner_only_attn_layers.add(layer_name)


def bind_kv_cache(
    kv_caches: dict[str, torch.Tensor],
    forward_context: dict[str, Attention],
    runner_kv_caches: list[torch.Tensor],
    num_attn_module: int = 1,
) -> None:
    """
    Bind the allocated KV cache to both ModelRunner and forward context so
    that the KV cache can be used in the forward pass.

    This function:
      1) Fills the ModelRunner's kv cache list (`runner_kv_caches`) with
         kv_caches.
      2) Associates each attention layer in the `forward_context` with its
         corresponding KV cache in kv_caches.

    Args:
        kv_caches: The allocated kv_caches with layer names as keys.
        forward_context: The global forward context containing all Attention
            layers with layer names as keys.
        runner_kv_caches: The kv_cache declared by ModelRunner.
    """
    # Bind kv_caches to ModelRunner
    # ------【核心逻辑】确认 runner_kv_caches 尚未填充，避免重复绑定 ------
    assert len(runner_kv_caches) == 0

    # Convert kv_caches dict to a list of tensors in the order of layer_index.
    # ------【核心逻辑】按层索引把 layer_name 分组，便于按顺序填充 runner_kv_caches ------
    index2name = defaultdict(list)
    for layer_name in kv_caches:
        index2name[extract_layer_index(layer_name, num_attn_module)].append(layer_name)

    # ------【核心逻辑】按层索引升序遍历，保持 KV cache 与层顺序一致 ------
    for layer_index in sorted(index2name.keys()):
        layer_names = index2name[layer_index]
        # ------【核心逻辑】同一层索引有多个注意力层（如 encoder-decoder）时单独处理 ------
        if len(layer_names) > 1:
            # One typical case is encoder-decoder model, e.g., bart.
            # The cross attention and self attention in the same decoder layer
            # has different layer_name but the same layer_index.

            # TODO - analyze where runner_kv_caches is used and the right
            # way to ensure it properly reflects multiple attention layers
            # in the same decoder block.
            # ------【核心逻辑】CUDA/XPU/CPU 平台上 runner 不受多层同名索引影响，忽略即可 ------
            if (
                current_platform.is_cuda_alike()
                or current_platform.is_xpu()
                or current_platform.is_cpu()
            ):
                # We know that the GPU / CPU runner is not impacted by this
                # case. Some test code depends on runner_kv_caches, but
                # not in a way that's impacted by ignoring this.
                pass
            else:
                # ------【核心逻辑】其它平台不支持该情况，直接抛错 ------
                raise NotImplementedError
        # ------【核心逻辑】按层顺序把 KV cache 张量追加进 runner_kv_caches ------
        for layer_name in layer_names:
            runner_kv_caches.append(kv_caches[layer_name])

    # Bind kv_caches to forward context. Each layer's bind_kv_cache unpacks
    # its raw allocation into the per-layer view(s) it needs (e.g. Mamba
    # splits conv/ssm), so the kv_caches dict can hold a single tensor per
    # layer for the KV connector to register.
    # ------【核心逻辑】逐个把 KV cache 绑定到对应注意力层，供前向计算时使用 ------
    for layer_name, kv_cache in kv_caches.items():
        forward_context[layer_name].bind_kv_cache(kv_cache)


def copy_kv_cache_blocks_inplace(
    kv_caches: Iterable[torch.Tensor | list[torch.Tensor]],
    num_blocks: int,
    kv_cache_block_copies: Sequence[KVCacheBlockCopy],
) -> None:
    # ------【前缀缓存】没有需要复制的块对时直接返回 ------
    if not kv_cache_block_copies:
        return

    # ------【前缀缓存】按底层 storage 地址去重，收集所有唯一共享存储张量 ------
    storage_tensors: list[torch.Tensor] = []
    seen_storage: set[int] = set()
    for entry in kv_caches:
        # Mamba layers hold a list of state tensors; attention layers a single
        # tensor. Both alias the shared block-major backing storage.
        # ------【前缀缓存】把单张量或张量列表统一成可迭代形式 ------
        tensors = entry if isinstance(entry, (list, tuple)) else (entry,)
        for tensor in tensors:
            # ------【前缀缓存】多个层别名同一底层存储时只登记一次 ------
            ptr = tensor.untyped_storage().data_ptr()
            if ptr in seen_storage:
                continue
            seen_storage.add(ptr)
            storage_tensors.append(tensor)

    # ------【前缀缓存】没有任何存储张量时无需复制，直接返回 ------
    if not storage_tensors:
        return
    device = storage_tensors[0].device
    # ------【前缀缓存】把拷贝对转成 int64 张量并异步上传到 device ------
    indices_np = np.array(kv_cache_block_copies, dtype=np.int64)
    indices = async_tensor_h2d(indices_np, device=device)
    # ------【前缀缓存】拆出 src 与 dst 两个索引列 ------
    src_indices, dst_indices = indices.unbind(dim=1)

    # ------【前缀缓存】对每个唯一存储张量执行块级原地拷贝 ------
    for tensor in storage_tensors:
        assert tensor.device == device
        # ------【前缀缓存】用空 uint8 张量直接别名底层 storage，构造字节级视图 ------
        blocks = torch.empty(0, dtype=torch.uint8, device=device)
        blocks.set_(tensor.untyped_storage())
        # Block-major backing storage: block i owns the contiguous byte range
        # [i * page_size, (i + 1) * page_size).
        # ------【前缀缓存】校验字节总数能被块数整除，保证可按块均匀切分 ------
        assert blocks.numel() % num_blocks == 0
        # ------【前缀缓存】重排为 (num_blocks, -1) 后按索引高级赋值完成块拷贝 ------
        blocks = blocks.view(num_blocks, -1)
        blocks[dst_indices] = blocks[src_indices]


def is_residual_scattered_for_sp(
    vllm_config: VllmConfig, num_input_tokens: int
) -> bool:
    """Check if the residual tensor is scattered for sequence parallelism.

    The residual tensor is scattered across tensor parallel ranks when sequence
    parallelism and tensor parallelism is enabled. SP is only supported in
    full-graph compilation mode.
    """
    # ------【TP】未启用序列并行（SP）时残差不做切分，返回 False ------
    if not vllm_config.compilation_config.pass_config.enable_sp:
        return False

    tp = vllm_config.parallel_config.tensor_parallel_size

    # ------【TP】TP=1 时没有跨 rank 切分，残差不会散开 ------
    if tp == 1:
        return False

    # ------【TP】SP 依赖全图编译，校验编译配置满足要求 ------
    assert (
        vllm_config.compilation_config.use_inductor_graph_partition
        or not vllm_config.compilation_config.splitting_ops
    ), "Sequence parallelism requires full-graph compilation"

    # When sequence parallelism is enabled, we always pad num_input_tokens
    # to be a multiple of tensor_parallel_size (tp) earlier.
    # ------【TP】校验 token 数已被提前 padding 成 tp 的整数倍 ------
    assert num_input_tokens % tp == 0

    return True
