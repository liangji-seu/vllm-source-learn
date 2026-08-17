# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV-Cache Utilities."""

import copy
import hashlib
import math
import os
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, replace
from functools import partial
from typing import Any, NamedTuple, NewType, TypeAlias, cast, overload

from vllm import envs
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.utils.hashing import sha256_cbor, xxhash_cbor
from vllm.utils.math_utils import cdiv, round_up
from vllm.utils.mem_utils import format_gib
from vllm.utils.torch_utils import get_dtype_size
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    ChunkedLocalAttentionSpec,
    FullAttentionSpec,
    HiddenStateCacheSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheSpec,
    KVCacheTensor,
    MambaSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry
from vllm.v1.request import Request
from vllm.v1.utils import tensor_data

# BlockHash represents the hash of a single KV-cache block used for
# prefix caching.  Treating it as a distinct type from `bytes` helps
# catch accidental misuse when passing around raw byte strings.
BlockHash = NewType("BlockHash", bytes)

# `BlockHashWithGroupId` combines a `BlockHash` with its KV cache group ID.
# It is represented as raw bytes for compactness and efficiency. The helper
# functions below pack/unpack the `BlockHash` and group id into/from the key.
BlockHashWithGroupId = NewType("BlockHashWithGroupId", bytes)

# ExternalBlockHash is used for reproducible prefix-cache block hashing.
# It's a union of `bytes` and `int` to keep backward compatibility
# after we default block hashing to use sha256 bytes.
ExternalBlockHash: TypeAlias = bytes | int


def make_block_hash_with_group_id(
    block_hash: BlockHash, group_id: int
) -> BlockHashWithGroupId:
    """Pack a `BlockHash` and group id into a `BlockHashWithGroupId`.

    The group id is encoded using 4 bytes in big-endian order and appended to
    the block hash bytes.  This representation avoids creating tuples while
    still allowing us to recover both components when needed.
    """
    # ------【前缀缓存】把 4 字节大端序 group_id 追加到 block_hash 后，拼成紧凑字节串作字典键 ------
    return BlockHashWithGroupId(block_hash + group_id.to_bytes(4, "big", signed=False))


def get_block_hash(key: BlockHashWithGroupId) -> BlockHash:
    """Extract the `BlockHash` from a `BlockHashWithGroupId`."""
    # ------【前缀缓存】去掉末尾 4 字节 group_id，还原出原始 block_hash ------
    return BlockHash(key[:-4])


def get_group_id(key: BlockHashWithGroupId) -> int:
    """Extract the group id from a `BlockHashWithGroupId`."""
    # ------【前缀缓存】从末尾 4 字节大端序解码回 group_id ------
    return int.from_bytes(key[-4:], "big", signed=False)


def maybe_convert_block_hash(hash_bytes: BlockHash) -> ExternalBlockHash:
    # ------【前缀缓存】事件上报需 int 哈希时，把字节哈希截断成 64 位整数返回 ------
    if not envs.VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES:
        return hash_bytes
    return int.from_bytes(hash_bytes, byteorder="big") & ((1 << 64) - 1)


logger = init_logger(__name__)

# The hash seed for the first block of any prefix block sequence.
#
# We use a random value to avoid hash collisions or PYTHONHASHSEED environment
# variable if set such that processes can share the seed if needed. This aligns
# with the behavior of Python's hash() function, which also uses a random seed
# if PYTHONHASHSEED is not set.
#
# The function `init_none_hash` initializes this variable globally.
NONE_HASH: BlockHash
_CBOR_HASH_FUNCTIONS = frozenset({sha256_cbor, xxhash_cbor})


def init_none_hash(hash_fn: Callable[[Any], bytes]):
    global NONE_HASH

    # ------【前缀缓存】读取 PYTHONHASHSEED，未设置且用 CBOR 哈希则告警（哈希不可复现） ------
    hash_seed = os.getenv("PYTHONHASHSEED")
    if hash_seed is None and hash_fn in _CBOR_HASH_FUNCTIONS:
        logger.warning(
            "PYTHONHASHSEED is not set. This will lead to non-reproducible "
            "block-hashes when using CBOR-based hash functions such as "
            "sha256_cbor or xxhash_cbor. Consider setting PYTHONHASHSEED to a "
            "fixed value for reproducibility."
        )

    # ------【前缀缓存】无种子则随机 32 字节、有种子则按哈希函数生成，保证跨进程可复现 ------
    if hash_seed is None:
        NONE_HASH = BlockHash(os.urandom(32))
    else:
        NONE_HASH = BlockHash(hash_fn(hash_seed))


@dataclass(slots=True)
class KVCacheBlock:
    """KV-cache block metadata."""

    # ------【核心逻辑】block_id 唯一编号 + ref_cnt 引用计数，供调度器分配/回收记账 ------
    # Block ID, ranging from 0 to num_gpu_blocks - 1.
    block_id: int
    # Reference count.
    ref_cnt: int = 0
    # ------【前缀缓存】块满载后被缓存时记录其哈希键与覆盖 token 数，供前缀匹配使用 ------
    # The hash key (block hash + group id) of the block, only available
    # when the block is full and cached.
    _block_hash: BlockHashWithGroupId | None = None # 链式hash + groupid
    # Number of prefix tokens covered by _block_hash. For full blocks this is
    # the full block boundary; partial entries can end inside a cache block.
    _block_hash_num_tokens: int | None = None

    # ------【内存池/CuMem】前后指针构成空闲块双向链表，仅由 FreeKVCacheBlockQueue 操作 ------
    # Used to construct a doubly linked list for free blocks.
    # These two attributes should only be manipulated by FreeKVCacheBlockQueue.
    prev_free_block: "KVCacheBlock | None" = None
    next_free_block: "KVCacheBlock | None" = None

    # ------【核心逻辑】空块标记：这类块永远不参与缓存，作为占位/填充使用 ------
    # Whether the block is a null block that should never be cached.
    is_null: bool = False

    # ------【前缀缓存】只读暴露块哈希与哈希覆盖 token 数 ------
    @property
    def block_hash(self) -> BlockHashWithGroupId | None:
        return self._block_hash

    @property
    def block_hash_num_tokens(self) -> int | None:
        return self._block_hash_num_tokens

    def set_block_hash(
        self,
        block_hash: BlockHashWithGroupId,
        num_tokens: int | None = None,
    ) -> None:
        # ------【前缀缓存】断言块尚未有哈希，防止重复写入破坏缓存一致性 ------
        assert self.block_hash is None and self._block_hash_num_tokens is None, (
            "The block already has a hash. This should not happen."
        )
        # ------【前缀缓存】写入块哈希与覆盖 token 数 ------
        self._block_hash = block_hash
        self._block_hash_num_tokens = num_tokens

    def reset_hash(self):
        """Reset the block hash when the block is evicted."""
        # ------【前缀缓存】块被淘汰时清空哈希，避免复用旧哈希造成误命中 ------
        self._block_hash = None
        self._block_hash_num_tokens = None

    def __repr__(self) -> str:
        # Use block_id instead of KVCacheBlock object to avoid calling __repr__
        # on KVCacheBlock object recursively.
        # ------【核心逻辑】取前后块的 block_id 而非对象本身，避免 __repr__ 递归 ------
        prev_block_id = self.prev_free_block.block_id if self.prev_free_block else None
        next_block_id = self.next_free_block.block_id if self.next_free_block else None
        return (
            f"KVCacheBlock(block_id={self.block_id}, "
            f"ref_cnt={self.ref_cnt}, "
            f"_block_hash={self._block_hash!r}, "
            f"_block_hash_num_tokens={self._block_hash_num_tokens}, "
            f"prev_free_block={prev_block_id}, "
            f"next_free_block={next_block_id})"
        )


# ------【前缀缓存】记录一次块复制的源/目标块 id，供 copy-on-write 场景使用 ------
class KVCacheBlockCopy(NamedTuple):
    src_block_id: int
    dst_block_id: int


class FreeKVCacheBlockQueue:
    """This class organizes a list of KVCacheBlock objects to a doubly linked
    list of free blocks. We implement this class instead of using Python
    builtin deque to support removing a block in the middle of the queue
    in O(1) time. To close the performance gap to the builtin deque which is
    implemented in C++, this class does not allocate any Python objects when
    manipulating the linked list. Instead, this class manipulates the
    prev_free_block and next_free_block attributes of the given blocks.

    The queue is ordered by block ID in the beginning. When a block is allocated
    and then freed, it will be appended back with the eviction order:
    1. The least recent used block is at the front (LRU).
    2. If two blocks have the same last accessed time (allocated by the
       same sequence), the one with more hash tokens (the tail of a block
       chain) is at the front.
    Note that we maintain this order by reversing the block order when free
    blocks of a request. This operation is outside of this class.

    Args:
        blocks: A list of KVCacheBlock objects.
    """

    def __init__(self, blocks: list[KVCacheBlock]) -> None:
        # ------【内存池/CuMem】记录空闲块数量，作为 O(1) 分配/回收的计数基准 ------
        self.num_free_blocks = len(blocks)

        # ------【内存池/CuMem】把相邻空闲块首尾相连，建立初始双向链表 ------
        # Initialize doubly links of consecutive blocks
        for i in range(self.num_free_blocks):
            if i > 0:
                blocks[i].prev_free_block = blocks[i - 1]
            if i < self.num_free_blocks - 1:
                blocks[i].next_free_block = blocks[i + 1]

        # ------【内存池/CuMem】创建哑头/尾哨兵，省去边界判空分支，保证每个真实块都有前后指针 ------
        # Create a fake head and a tail block for the doubly linked list to
        # reduce branching in the code
        #
        # The implementation guaranteed that the fake head and tail
        # are NEVER got popped, so we could safely assume each real blocks
        # in the queue has prev and next blocks.
        self.fake_free_list_head = KVCacheBlock(block_id=-1)
        self.fake_free_list_tail = KVCacheBlock(block_id=-1)
        if self.num_free_blocks > 0:
            # ------【内存池/CuMem】非空时把哨兵头/尾分别接到首/尾真实块 ------
            # Connect fake_head and fake_tail to the first and last block
            # respectively.
            self.fake_free_list_head.next_free_block = blocks[0]
            blocks[0].prev_free_block = self.fake_free_list_head
            self.fake_free_list_tail.prev_free_block = blocks[-1]
            blocks[-1].next_free_block = self.fake_free_list_tail
        else:
            # ------【内存池/CuMem】空链表时哨兵头尾直接互连 ------
            # For empty list, simply connect the fake head and tail.
            self.fake_free_list_head.next_free_block = self.fake_free_list_tail
            self.fake_free_list_tail.prev_free_block = self.fake_free_list_head

    def popleft(self) -> KVCacheBlock:
        """Pop the first free block and reduce num_free_blocks by 1.

        Returns:
            The first free block.
        """
        # ------【内存池/CuMem】空队列检测：头哨兵已指向尾哨兵则无可用块，抛错并断言计数一致 ------
        if (
            self.fake_free_list_head.next_free_block is self.fake_free_list_tail
            or self.fake_free_list_head.next_free_block is None
        ):
            assert self.num_free_blocks == 0, (
                f"num_free_blocks ({self.num_free_blocks}) is out of sync "
                "with the free list."
            )
            raise ValueError("No free blocks available")

        # ------【内存池/CuMem】取出队首（头哨兵后继）作为待分配块 ------
        first_block: KVCacheBlock = self.fake_free_list_head.next_free_block

        # ------【内存池/CuMem】防御性校验：队首块必须有有效后继，否则说明调用方逻辑有 bug ------
        if first_block.next_free_block is None:
            # This should not happen if the block is from the free list.
            # It indicates a bug in the caller's logic.
            raise RuntimeError(
                "Invalid block found in popleft() "
                "which doesn't have a valid next_free_block"
            )

        # ------【内存池/CuMem】把头哨兵改接队首的下一个块（第二块或尾哨兵） ------
        # Connect fake_head and the next block of first_block (i.e. second block
        # or fake tail).
        self.fake_free_list_head.next_free_block = first_block.next_free_block
        first_block.next_free_block.prev_free_block = self.fake_free_list_head

        # ------【内存池/CuMem】摘除队首块的前后指针，使其脱离链表 ------
        # Remove the block from the linked list.
        first_block.prev_free_block = first_block.next_free_block = None

        # ------【内存池/CuMem】空闲计数减一并返回该块 ------
        self.num_free_blocks -= 1
        return first_block

    def popleft_n(self, n: int) -> list[KVCacheBlock]:
        """Pop the first n free blocks and reduce num_free_blocks by n.

        Args:
            n: The number of blocks to pop.

        Returns:
            A list of n free blocks.
        """
        # ------【内存池/CuMem】n=0 直接返回空表；否则断言空闲充足并一次性扣减计数 ------
        if n == 0:
            return []
        assert self.num_free_blocks >= n
        self.num_free_blocks -= n

        # ------【内存池/CuMem】从队首连续弹出 n 个块并清空其前后指针 ------
        curr_block = self.fake_free_list_head.next_free_block
        # Pop n blocks from the head of the list
        ret = []
        for _ in range(n):
            assert curr_block is not None
            ret.append(curr_block)
            last_block = curr_block
            curr_block = curr_block.next_free_block
            # Reset prev_free_block and next_free_block of all popped blocks
            last_block.prev_free_block = None
            last_block.next_free_block = None

        # ------【内存池/CuMem】若还剩块，把头哨兵接到新的队首 ------
        if curr_block is not None:
            # The queue is not empty, connect the fake head to
            # the new first block.
            self.fake_free_list_head.next_free_block = curr_block
            curr_block.prev_free_block = self.fake_free_list_head
        return ret

    def remove(self, block: KVCacheBlock) -> None:
        """Remove a block in the free list and reduce num_free_blocks by 1.

        Args:
            block: The block to remove.
        """
        # ------【内存池/CuMem】校验块确实在空闲链表中（前后指针都非空） ------
        if block.prev_free_block is None or block.next_free_block is None:
            # This should not happen if the block is from the free list.
            # It indicates a bug in the caller's logic.
            raise RuntimeError(f"remove() called on an invalid block: {block}")

        # ------【内存池/CuMem】前驱后继直接互连，从链表中绕过该块（O(1) 删除） ------
        # Link the previous block to the next block.
        block.prev_free_block.next_free_block = block.next_free_block
        # Link the next block to the previous block.
        block.next_free_block.prev_free_block = block.prev_free_block

        # ------【内存池/CuMem】清空该块前后指针并扣减计数 ------
        # Remove the block from the linked list.
        block.prev_free_block = block.next_free_block = None
        self.num_free_blocks -= 1

    def append(self, block: KVCacheBlock) -> None:
        """Put a block back into the free list and increase
        num_free_blocks by 1.

        Args:
            block: The block to append.
        """
        # ------【内存池/CuMem】校验尾哨兵有前驱，并取出当前最后一个真实块 ------
        if self.fake_free_list_tail.prev_free_block is None:
            raise RuntimeError(
                "prev_free_block of fake_free_list_tail should always exist"
            )
        last_block: KVCacheBlock = self.fake_free_list_tail.prev_free_block

        # ------【内存池/CuMem】把块插到最后一个真实块之后（队尾） ------
        # Connect the new block after the last block.
        last_block.next_free_block = block
        block.prev_free_block = last_block

        # ------【内存池/CuMem】把尾哨兵改接到新块之后 ------
        # Connect the fake tail after the new block.
        block.next_free_block = self.fake_free_list_tail
        self.fake_free_list_tail.prev_free_block = block

        # ------【内存池/CuMem】空闲计数加一 ------
        self.num_free_blocks += 1

    def prepend_n(self, blocks: list[KVCacheBlock]) -> None:
        """Put a list of blocks at the front of the free list."""
        # ------【内存池/CuMem】空表直接返回，避免后续越界 ------
        if len(blocks) == 0:
            return

        # ------【内存池/CuMem】取出当前队首（头哨兵后继）并断言其存在 ------
        first_block = self.fake_free_list_head.next_free_block
        assert first_block is not None, (
            "next_free_block of fake_free_list_head should always exist"
        )

        # ------【内存池/CuMem】把新块链依次接在头哨兵之后 ------
        prev_block = self.fake_free_list_head
        for block in blocks:
            block.prev_free_block = prev_block
            prev_block.next_free_block = block
            prev_block = block

        # ------【内存池/CuMem】把新链末尾接到原队首，整体插入队首 ------
        prev_block.next_free_block = first_block
        first_block.prev_free_block = prev_block

        # ------【内存池/CuMem】空闲计数加上插入块数 ------
        self.num_free_blocks += len(blocks)

    def append_n(self, blocks: list[KVCacheBlock]) -> None:
        """Put a list of blocks back into the free list

        Args:
            blocks: The blocks to append.
        """
        # ------【内存池/CuMem】空表直接返回 ------
        if len(blocks) == 0:
            return

        # ------【内存池/CuMem】取出当前最后一个真实块并断言其存在 ------
        last_block = self.fake_free_list_tail.prev_free_block
        assert last_block is not None, (
            "prev_free_block of fake_free_list_tail should always exist"
        )
        # ------【内存池/CuMem】把新块依次接在当前队尾之后，建立相邻互连 ------
        # Add inter-connections between consecutive blocks
        for block in blocks:
            block.prev_free_block = last_block
            last_block.next_free_block = block
            last_block = block

        # ------【内存池/CuMem】把新链末尾接到尾哨兵，完成队尾追加 ------
        # Connect the last block of <blocks> to the fake tail
        last_block.next_free_block = self.fake_free_list_tail
        self.fake_free_list_tail.prev_free_block = last_block

        # ------【内存池/CuMem】空闲计数加上追加块数 ------
        self.num_free_blocks += len(blocks)

    def get_all_free_blocks(self) -> list[KVCacheBlock]:
        """Get all free blocks in the free list. Mainly used for testing.

        Returns:
            A list of free blocks.
        """
        ret = []
        # ------【内存池/CuMem】防御性校验头哨兵后继存在 ------
        if self.fake_free_list_head.next_free_block is None:
            raise RuntimeError(
                "next_free_block of fake_free_list_head should always exist"
            )
        # ------【内存池/CuMem】从头哨兵后继遍历到尾哨兵，收集所有空闲块（用于测试） ------
        # Start from the first block
        curr_block: KVCacheBlock = self.fake_free_list_head.next_free_block
        # As long as next_free_block is available, we haven't reached to
        # the fake tail yet.
        while curr_block.next_free_block is not None:
            ret.append(curr_block)
            curr_block = curr_block.next_free_block
        return ret

    def iter_blocks_after(
        self,
        cursor: KVCacheBlock | None,
    ) -> Iterator[KVCacheBlock]:
        """Iterate free blocks in eviction order after the cursor."""
        # ------【内存池/CuMem】从游标后继（或队首）开始，按淘汰顺序定位起点 ------
        if cursor is None:
            curr_block = self.fake_free_list_head.next_free_block
        else:
            curr_block = cursor.next_free_block

        # ------【内存池/CuMem】沿 next 指针逐个产出空闲块，直到尾哨兵 ------
        while curr_block is not None and curr_block is not self.fake_free_list_tail:
            yield curr_block
            curr_block = curr_block.next_free_block


def _gen_mm_extra_hash_keys(
    request: Request, start_token_idx: int, end_token_idx: int, start_mm_idx: int
) -> tuple[list[Any], int]:
    """Generate extra keys related to MultiModal request for block hash
    computation. For multi-modal inputs, the extra keys are
    (mm_hash, start_offset) that indicate a mm input contained in the
    block and its starting offset in the block tokens.

    Args:
        request: The request object.
        start_token_idx: The start token index of the block.
        end_token_idx: The end token index of the block.
        start_mm_idx: The start multi-modal index of the block.

    Returns:
        A tuple of extra keys and the next multi-modal index.
    """
    # ------【前缀缓存】初始化额外键；无多模态特征时直接返回空键 ------
    extra_keys: list[Any] = []

    mm_features = request.mm_features
    if not mm_features:
        return extra_keys, start_mm_idx

    # Note that we assume mm_features are sorted by mm_position.offset.
    # We do not need to check all mm inputs if the start token index is out of
    # range. This usually happens in the late prefill phase and decoding phase.
    # ------【前缀缓存】最后一个 mm 输入已完全落在块起点之前，无需再检查（后期 prefill/解码） ------
    last_pos = mm_features[-1].mm_position
    if last_pos.offset + last_pos.length <= start_token_idx:
        return extra_keys, start_mm_idx

    # ------【前缀缓存】支持 -1 表示最后一个 mm 输入，转换为正向索引 ------
    # Support start_mm_idx == -1 to indicate the last mm input.
    if start_mm_idx < 0:
        assert -start_mm_idx <= len(mm_features)
        start_mm_idx = len(mm_features) + start_mm_idx

    # ------【前缀缓存】遍历块可能覆盖的 mm 输入，逐个判断是否落入本块 token 区间 ------
    curr_mm_idx = start_mm_idx
    while mm_features and curr_mm_idx < len(mm_features):
        mm_feature = mm_features[curr_mm_idx]
        assert mm_feature.identifier is not None
        offset = mm_feature.mm_position.offset
        length = mm_feature.mm_position.length
        if end_token_idx > offset:
            # ------【前缀缓存】块已越过当前 mm 输入：前进到下一个 mm 输入 ------
            if start_token_idx >= offset + length:
                # This block has passed the current mm input.
                curr_mm_idx += 1
                continue

            # The block contains the current mm input. Include its offset
            # relative to the start of the block so prefix-cache keys stay
            # distinct when the same MM item appears at different positions
            # within otherwise-identical placeholder blocks.
            # ------【前缀缓存】块含当前 mm 输入：记录其相对块起点的偏移作键，区分同内容不同位置 ------
            extra_keys.append((mm_feature.identifier, offset - start_token_idx))

            # ------【前缀缓存】块也含 mm 输入末尾则继续检查下一个；否则本块 mm 键收集完毕 ------
            if end_token_idx >= offset + length:
                # If this block contains the end of the current mm input,
                # move to the next mm input as this block may also contain
                # the next mm input.
                curr_mm_idx += 1
            else:
                # Otherwise this block is done with mm inputs.
                break
        # ------【前缀缓存】块尚未到达当前 mm 输入，提前结束 ------
        else:
            # This block has not reached the current mm input.
            break
    return extra_keys, curr_mm_idx


def _gen_lora_extra_hash_keys(request: Request) -> list[str]:
    """Generate extra keys related to LoRA for block hash computation.

    Args:
        request: The request object.

    Returns:
        Return LoRA name of the request if it is a LoRA request. Return empty
        list otherwise.
    """
    # ------【前缀缓存+LoRA】LoRA 请求把 lora_name 纳入块哈希键，区分不同 adapter 的缓存 ------
    if not request.lora_request:
        return []
    return [request.lora_request.lora_name]


def _gen_prompt_embeds_extra_hash_keys(
    request: Request, start_token_idx: int, end_token_idx: int
) -> list[bytes]:
    """Generate extra keys related to prompt embeds for block hash computation.

    Args:
        request: The request object.
        start_token_idx: The start token index of the block.
        end_token_idx: The end token index of the block.

    Returns:
        Return a stable hash of the block prompt embeddings if prompt embeds
        are present. Return empty list otherwise.
    """
    # ------【前缀缓存】无 prompt embeds 时直接返回空键 ------
    if request.prompt_embeds is None:
        return []
    # ------【前缀缓存】以块 token 区间为键查询已缓存的 embedding 哈希，避免重复计算 ------
    block_range = (start_token_idx, end_token_idx)
    embeds_hash = request._prompt_embeds_per_block_hashes.get(block_range)
    # ------【前缀缓存】首次计算块内 embedding 的 sha256 并缓存到 request，供后续块复用 ------
    if embeds_hash is None:
        block_prompt_embeds = request.prompt_embeds[start_token_idx:end_token_idx]
        # Hash prompt embeds once per block and cache on request
        embeds_hash = hashlib.sha256(tensor_data(block_prompt_embeds)).digest()
        request._prompt_embeds_per_block_hashes[block_range] = embeds_hash
    return [embeds_hash]


def generate_block_hash_extra_keys(
    request: Request, start_token_idx: int, end_token_idx: int, start_mm_idx: int
) -> tuple[tuple[Any, ...] | None, int]:
    """Generate extra keys for the block hash. The extra keys can come from
    the multi-modal inputs, request specific metadata (e.g., LoRA names), and
    hashed data from prompt embeddings.

    Args:
        request: The request object.
        start_token_idx: The start token index of the block.
        end_token_idx: The end token index of the block.
        start_mm_idx: The start multi-modal index of the block.

    Returns:
        A tuple of extra keys and the next multi-modal index.
    """
    # ------【前缀缓存】生成多模态额外键并推进 mm 游标 ------
    mm_extra_keys: list[Any]
    mm_extra_keys, new_start_mm_idx = _gen_mm_extra_hash_keys(
        request, start_token_idx, end_token_idx, start_mm_idx
    )
    # ------【前缀缓存+LoRA】生成 LoRA 额外键 ------
    lora_extra_keys: list[str] = _gen_lora_extra_hash_keys(request)
    # ------【前缀缓存】仅首块纳入 cache_salt，用于隔离不同缓存命名空间 ------
    cache_salt_keys: list[str] = (
        [request.cache_salt] if (start_token_idx == 0 and request.cache_salt) else []
    )
    # ------【前缀缓存】生成 prompt embedding 额外键 ------
    prompt_embeds_keys = _gen_prompt_embeds_extra_hash_keys(
        request, start_token_idx, end_token_idx
    )

    # ------【前缀缓存】把所有维度额外键拼接成一个列表 ------
    extra_keys: list[Any] = (
        lora_extra_keys + mm_extra_keys + cache_salt_keys + prompt_embeds_keys
    )

    # ------【前缀缓存】无任何额外键则返回 None，表示用纯 token 哈希即可 ------
    if not extra_keys:
        return None, new_start_mm_idx

    # ------【前缀缓存】转成不可变 tuple 作为哈希输入的一部分返回 ------
    return tuple(extra_keys), new_start_mm_idx


def hash_block_tokens(
    hash_function: Callable[[Any], bytes],
    parent_block_hash: BlockHash | None,
    curr_block_token_ids: Sequence[int],
    extra_keys: tuple[Any, ...] | None = None,
) -> BlockHash:
    """Computes a hash value corresponding to the contents of a block and
    the contents of the preceding block(s). The hash value is used for
    prefix caching. We use LRU cache for this function to avoid recomputing
    hash values for the same block contents.
    Args:
        hash_function: The hash function used to compute block hash.
        parent_block_hash: The hash of the parent block. None
            if this is the first block.
        curr_block_token_ids: A list of token ids in the current
            block. The current block is assumed to be full.
        extra_keys: Extra keys for the block.
    Returns:
        The hash value of the block and the token ids in the block.
        The entire tuple is used as the hash key of the block.
    """
    # ------【前缀缓存】首块无父哈希时用全局 NONE_HASH 作链式哈希起点 ------
    if not parent_block_hash:
        parent_block_hash = NONE_HASH

    # ------【前缀缓存】把 token id 序列转 tuple，保证哈希输入稳定可序列化 ------
    curr_block_token_ids_tuple = tuple(curr_block_token_ids)
    # ------【前缀缓存】对(父哈希, 本块 token, 额外键)整体哈希，实现跨块前缀链式指纹 ------
    return BlockHash(
        hash_function((parent_block_hash, curr_block_token_ids_tuple, extra_keys))
    )


def resolve_kv_cache_block_sizes(
    kv_cache_config: KVCacheConfig,
    vllm_config: VllmConfig,
) -> tuple[int, int]:
    """Resolve (scheduler_block_size, hash_block_size).

    - ``scheduler_block_size`` is the token-alignment invariant used by the
      scheduler (e.g. for ``num_computed_tokens`` rounding). Single group:
      ``cache_config.block_size * dcp``. Multiple groups: LCM of every
      group's effective block size. Attention groups are scaled by DCP;
      Mamba groups keep their full per-rank state and are not scaled.
    - ``hash_block_size`` is the granularity at which ``Request.block_hashes``
      is computed. Single group: equals scheduler block size. Multiple groups:
      ``cache_config.prefix_match_unit`` override if set, else the GCD of
      group block sizes; every group's block size must be divisible by it.
      Returns the scheduler block size (i.e. disables finer hashing) if block
      hashing is inactive or a mamba group's block size diverges from the
      cache block size (mamba_cache_mode != "align").
    """
    # ------【核心逻辑】取出 cache 配置、解码上下文并行度(dcp)与 KV cache 分组 ------
    cache_config = vllm_config.cache_config
    dcp = vllm_config.parallel_config.decode_context_parallel_size
    groups = kv_cache_config.kv_cache_groups

    # ------【TP】单组时块大小按解码上下文并行度(dcp)放大，作为对齐基准直接返回 ------
    if len(groups) <= 1:
        bs = cache_config.block_size * dcp
        return bs, bs

    # ------【TP】注意力组块大小按 dcp 缩放、Mamba 组保持不变，逐组求有效块大小 ------
    group_block_sizes = [
        g.kv_cache_spec.block_size * dcp
        if isinstance(g.kv_cache_spec, AttentionSpec)
        else g.kv_cache_spec.block_size
        for g in groups
    ]
    # ------【核心逻辑】调度器块大小取各组有效块大小的最小公倍数，统一 token 对齐粒度 ------
    scheduler_block_size = math.lcm(*group_block_sizes)

    # Block hashes are only consumed by prefix caching and KV connectors
    # (P/D, offloading); when neither is active, keep hash_block_size equal
    # to the scheduler block size.
    # ------【PD 分离+前缀缓存】前缀缓存与 KV 连接器(P/D、卸载)都不用哈希时，哈希块大小退化为调度块大小 ------
    connector_enabled = vllm_config.kv_transfer_config is not None
    if not (cache_config.enable_prefix_caching or connector_enabled):
        return scheduler_block_size, scheduler_block_size

    # Mamba groups outside align mode break divisibility; back off to the
    # scheduler block size. Read the mode from the resolved group spec because
    # its block size may have been updated independently of cache_config.
    # ------【核心逻辑】非 align 模式 Mamba 组破坏整除性，退化为调度块大小 ------
    if any(
        isinstance(g.kv_cache_spec, MambaSpec)
        and g.kv_cache_spec.mamba_cache_mode != "align"
        for g in groups
    ):
        return scheduler_block_size, scheduler_block_size

    # ------【前缀缓存】哈希块大小取用户指定 prefix_match_unit，否则取各组块大小最大公约数 ------
    requested = cache_config.prefix_match_unit
    hash_block_size = (
        requested if requested is not None else math.gcd(*group_block_sizes)
    )
    # ------【前缀缓存】校验哈希块大小能整除所有组块大小，否则无法对齐直接报错 ------
    if any(bs % hash_block_size != 0 for bs in group_block_sizes):
        raise ValueError(
            f"Invalid prefix_match_unit={hash_block_size}; all KV cache group "
            f"block sizes must be divisible by prefix_match_unit. "
            f"Got group block sizes={group_block_sizes}."
        )
    return scheduler_block_size, hash_block_size


def get_request_block_hasher(
    hash_block_size: int,
    caching_hash_fn: Callable[[Any], bytes],
) -> Callable[[Request], list[BlockHash]]:
    """
    Returns a function which computes the list of un-computed block hashes
    of a request.

    Hashes are computed at ``hash_block_size`` granularity and chained over the
    full prefix, so each hash uniquely fingerprints the prefix ending at its
    boundary. Coarser group block sizes and partial-cache boundaries reuse
    these hashes directly (see ``BlockHashListWithBlockSize``).
    """

    # ------【前缀缓存】闭包捕获哈希块大小与哈希函数，返回按需增量计算块哈希的函数 ------
    def request_block_hasher(request: Request) -> list[BlockHash]:
        # ------【前缀缓存】已算哈希块数×哈希块大小得到下次要算的起始 token 下标 ------
        start_token_idx = len(request.block_hashes) * hash_block_size
        num_tokens = request.num_tokens

        # ------【前缀缓存】剩余 token 不足一个完整哈希块则提前返回空，只算满块 ------
        if start_token_idx + hash_block_size > num_tokens:
            # Early stop when there no new full blocks created.
            return []

        # ------【前缀缓存】非首块时把 mm 游标置 -1，表示只需考虑最后一个 mm 输入 ------
        curr_mm_idx = 0
        if start_token_idx > 0:
            # Set curr_mm_idx = -1 to indicate the last mm input.
            # Note that since we reach to this branch only when the block is
            # completed with generated tokens, we only need to consider the
            # last mm input.
            curr_mm_idx = -1

        # ------【前缀缓存】取已算最后一个块哈希作为链式哈希的父哈希起点 ------
        prev_block_hash_value = (
            request.block_hashes[-1] if request.block_hashes else None
        )
        # ------【前缀缓存】收集新算出块哈希的结果列表 ------
        new_block_hashes: list[BlockHash] = []
        # ------【前缀缓存】循环推进块区间，只对满块计算哈希，不足满块即停 ------
        while True:
            end_token_idx = start_token_idx + hash_block_size
            if end_token_idx > num_tokens:
                # We only hash full blocks
                break

            # ------【前缀缓存】为 MM/LoRA 请求生成块级额外哈希键并推进 mm 游标 ------
            # MM and LoRA requests need extra keys for block-hash computation.
            extra_keys, curr_mm_idx = generate_block_hash_extra_keys(
                request, start_token_idx, end_token_idx, curr_mm_idx
            )

            # ------【前缀缓存】切出本块 token 并链式哈希，得到覆盖完整前缀的块指纹 ------
            # Compute the hash of the current block
            block_tokens = request.all_token_ids[start_token_idx:end_token_idx]
            block_hash = hash_block_tokens(
                caching_hash_fn, prev_block_hash_value, block_tokens, extra_keys
            )

            # ------【前缀缓存】记录哈希并推进块区间与父哈希，进入下一块 ------
            new_block_hashes.append(block_hash)
            start_token_idx += hash_block_size
            prev_block_hash_value = block_hash

        return new_block_hashes

    return request_block_hasher


def _check_enough_kv_cache_memory(
    available_memory: int,
    get_needed_memory: Callable[[], int],
    max_model_len: int,
    estimate_max_model_len: Callable[[int], int],
):
    # ------【显存 profiling】可用显存非正则无 KV cache 空间，报错并提示调大 gpu_memory_utilization ------
    if available_memory <= 0:
        raise ValueError(
            "No available memory for the cache blocks. "
            "Try increasing `gpu_memory_utilization` when initializing the engine "
            "(this flag also controls CPU memory reservation on the CPU "
            "backend, despite its name). "
            "See https://docs.vllm.ai/en/latest/configuration/conserving_memory/ "
            "for more details."
        )

    # ------【显存 profiling】计算支撑一条 max_model_len 请求所需的 KV cache 字节数 ------
    needed_memory = get_needed_memory()

    # ------【显存 profiling+auto-fit】所需显存超可用时，估算能容纳的最大长度并抛错 ------
    if needed_memory > available_memory:
        estimated_max_len = estimate_max_model_len(available_memory)
        estimated_msg = ""
        if estimated_max_len > 0:
            estimated_msg = (
                "Based on the available memory, "
                f"the estimated maximum model length is {estimated_max_len}. "
            )

        raise ValueError(
            f"To serve at least one request with the model's max seq len "
            f"({max_model_len}), ({format_gib(needed_memory)} GiB KV "
            f"cache is needed, which is larger than the available KV cache "
            f"memory ({format_gib(available_memory)} GiB). {estimated_msg}"
            f"Try increasing `gpu_memory_utilization` (which also controls "
            f"CPU memory on the CPU backend) or decreasing `max_model_len` "
            f"when initializing the engine. "
            f"See https://docs.vllm.ai/en/latest/configuration/conserving_memory/ "
            f"for more details."
        )


def max_memory_usage_bytes(
    vllm_config: VllmConfig, kv_cache_specs: Iterable[KVCacheSpec]
) -> int:
    """
    Get the maximum memory usage in bytes for the given KV cache specs.
    """
    # ------【显存 profiling】把所有层的峰值 KV cache 占用字节数求和，得到整体上限 ------
    return sum(spec.max_memory_usage_bytes(vllm_config) for spec in kv_cache_specs)


def estimate_max_model_len(
    vllm_config: VllmConfig,
    kv_cache_spec: dict[str, KVCacheSpec],
    available_memory: int,
) -> int:
    """
    Estimates the maximum model length that can fit in the available memory
    using binary search.

    This function temporarily modifies max_model_len during estimation but
    restores the original value before returning, ensuring no side effects.

    Args:
        vllm_config: The global VllmConfig
        kv_cache_spec: The kv cache spec of each attention layer in the model
        available_memory: Memory available for KV cache in bytes.

    Returns:
        The estimated maximum model length that can fit in the available memory.
    """
    # Save the original max_model_len to restore after estimation
    # ------【显存 profiling】暂存原 max_model_len，估算完再恢复以免污染全局配置 ------
    original_max_model_len = vllm_config.model_config.max_model_len

    # Define a function to check if a given model length fits in memory
    # ------【显存 profiling】闭包：临时改 max_model_len 后比较所需与可用显存，判断能否放下 ------
    def fits_in_memory(model_len: int) -> bool:
        # Temporarily modify the max_model_len for this calculation
        vllm_config.model_config.max_model_len = model_len
        # Calculate memory needed for the given model length
        memory_needed = max_memory_usage_bytes(vllm_config, kv_cache_spec.values())
        return memory_needed <= available_memory

    try:
        # Binary search for the maximum model length
        # ------【显存 profiling】二分搜索边界：[1, 原始 max_model_len] ------
        left, right = 1, original_max_model_len

        # If even the smallest model length doesn't fit, return 0
        # ------【显存 profiling】最小长度都放不下则返回 0，表示无法服务任何请求 ------
        if not fits_in_memory(left):
            return 0

        # Binary search for the maximum model length that fits
        # ------【显存 profiling】在单调可放下的区间上二分，逼近能容纳的最大长度 ------
        result = 1
        while left <= right:
            mid = (left + right) // 2
            if fits_in_memory(mid):
                result = mid
                left = mid + 1
            else:
                right = mid - 1
        return result
    finally:
        # Always restore the original max_model_len to avoid side effects
        # ------【显存 profiling】无论成功与否都恢复原 max_model_len，保证无副作用 ------
        vllm_config.model_config.max_model_len = original_max_model_len


def check_enough_kv_cache_memory(
    vllm_config: VllmConfig,
    kv_cache_spec: dict[str, KVCacheSpec],
    available_memory: int,
):
    """
    Checks whether `available_memory` is enough for the KV cache to hold at
    least one request with the model's max_model_len.

    Args:
        vllm_config: The global VllmConfig
        kv_cache_spec: The kv cache spec of each attention layer in the model
        available_memory: Memory available for KV cache in bytes.

    Raises:
        ValueError: If there is not enough memory available for the KV cache.
    """

    # No need to check for available memory if the kv_cache_spec is empty
    # ------【显存 profiling】空 spec 跳过校验；否则用闭包算所需显存并检查 ------
    if kv_cache_spec:
        _check_enough_kv_cache_memory(
            available_memory,
            lambda: max_memory_usage_bytes(vllm_config, kv_cache_spec.values()),
            vllm_config.model_config.max_model_len,
            lambda am: estimate_max_model_len(vllm_config, kv_cache_spec, am),
        )


def create_kv_cache_group_specs(
    kv_cache_spec: dict[str, KVCacheSpec], grouped_layer_names: list[list[str]]
) -> list[KVCacheGroupSpec]:
    """
    Create KVCacheGroupSpec object for each kv cache group layer.
    The layers in the same group should share the same
    KVCacheSpec.

    Args:
        kv_cache_spec:
            A mapping from each layer name to its corresponding KVCacheSpec.
        grouped_layer_names:
            A list of kv cache groups, where each element is a list of layer
            names that belong to the same group and should share the same
            KVCacheSpec.
    Returns:
        A list of KVCacheGroupSpec objects, one for each group.
    """
    # ------【核心逻辑】初始化分组结果容器，逐组处理预分好的层名列表 ------
    kv_cache_groups = []
    for layer_names_one_group in grouped_layer_names:
        # ------【核心逻辑】按层名取出本组内各层对应的 KVCacheSpec ------
        layer_specs = [
            kv_cache_spec[layer_name] for layer_name in layer_names_one_group
        ]
        # ------【核心逻辑】用首层 spec 合并整组 spec，校验并统一组内配置 ------
        merged_layer_spec = layer_specs[0].merge(layer_specs)
        # ------【核心逻辑】用合并后的 spec 与层名构造一个 KVCacheGroupSpec ------
        kv_cache_groups.append(
            KVCacheGroupSpec(layer_names_one_group, merged_layer_spec)
        )
    return kv_cache_groups


def is_kv_cache_spec_uniform(kv_cache_spec: dict[str, KVCacheSpec]) -> bool:
    """
    Whether all layers in the given KVCacheSpec have the same KV cache spec.
    Note that we regard FullAttentionSpec with and without sliding window as
    the same type.

    Args:
        kv_cache_spec: The kv cache spec of each attention layer in the model

    Returns:
        True if all layers have the same type, False otherwise.
    """

    # ------【核心逻辑】空 spec（如 encoder-only）视为统一，直接返回 True ------
    if not kv_cache_spec:
        # Encoder-only models do not have KV cache, kv_cache_type can be
        # regarded as uniform.
        return True
    # ------【核心逻辑】尝试把所有层 spec 合并成一个，能合并即说明类型一致 ------
    try:
        kv_cache_spec_values = list(kv_cache_spec.values())
        _ = kv_cache_spec_values[0].merge(kv_cache_spec_values)
    except AssertionError:
        # ------【核心逻辑】合并抛 AssertionError 说明存在不兼容类型，判定非统一 ------
        return False
    return True


def get_max_concurrency_for_kv_cache_config(
    vllm_config: VllmConfig, kv_cache_config: KVCacheConfig
) -> float:
    """
    Get the maximum concurrency for the given KV cache configuration.

    A request at max_model_len consumes whole blocks from each group's block
    table — cdiv(per-request bytes, page bytes) of the group's spec — and all
    groups draw those block ids from one shared pool, so the per-request
    total is the sum over groups. The memory/page ratio is identical whether
    a group carries an aggregated UniformTypeKVCacheSpecs (worker config) or
    a representative per-layer spec (scheduler config), so both capacity
    call sites agree.
    """
    # ------【核心逻辑】一条 max_model_len 请求在各组 block table 占用的块数求和 ------
    num_blocks_per_request = sum(
        cdiv(
            group.kv_cache_spec.max_memory_usage_bytes(vllm_config),
            group.kv_cache_spec.page_size_bytes,
        )
        for group in kv_cache_config.kv_cache_groups
    )
    # ------【核心逻辑】总块数 ÷ 单请求块数 = 最大并发请求数 ------
    max_concurrency = kv_cache_config.num_blocks / num_blocks_per_request
    return max_concurrency


def may_override_num_blocks(vllm_config: VllmConfig, num_blocks: int) -> int:
    """
    Override the number of kv cache blocks if `num_gpu_blocks_override` is set.
    The override is logged once, at the call site in `get_kv_cache_configs`.
    """
    # ------【显存 profiling】有 num_gpu_blocks_override 时用它覆盖 profiling 估算的块数 ------
    if vllm_config.cache_config.num_gpu_blocks_override is not None:
        num_blocks = vllm_config.cache_config.num_gpu_blocks_override
    return num_blocks


def _pool_bytes_per_block(
    vllm_config: VllmConfig, kv_cache_groups: list[KVCacheGroupSpec]
) -> int:
    """
    Bytes consumed by one block in the worker's shared KV cache pool, mirroring
    the divisor used by `get_kv_cache_config_from_groups` to convert
    `available_memory` into `num_blocks`. Used to compute the effective KV cache
    capacity once `num_gpu_blocks_override` is applied.
    """
    # ------【内存池/CuMem】单组 UniformType 布局时，一个块的字节数就是该组 page_size ------
    if len(kv_cache_groups) == 1 and isinstance(
        kv_cache_groups[0].kv_cache_spec, UniformTypeKVCacheSpecs
    ):
        return kv_cache_groups[0].kv_cache_spec.page_size_bytes
    # ------【内存池/CuMem】packed 布局下块跨度(block_stride)即每个块的物理字节数 ------
    if _use_packed_kv_cache_config(vllm_config, kv_cache_groups):
        block_stride, _ = _get_packed_kv_cache_layout(kv_cache_groups)
        return block_stride
    # ------【内存池/CuMem】通用布局：每块 = 统一 page_size × 组内层数 ------
    group_size = max(len(g.layer_names) for g in kv_cache_groups)
    page_size = get_uniform_page_size([g.kv_cache_spec for g in kv_cache_groups])
    return page_size * group_size


def get_num_blocks(
    vllm_config: VllmConfig,
    num_layers: int,
    available_memory: int,
    page_size: int,
) -> int:
    """
    Get the number of kv cache blocks.

    Args:
        vllm_config: The global VllmConfig
        num_layers: The number of layers
        available_memory: Memory available for KV cache in bytes.
        page_size: The page size of the KV cache.
    """
    # ------【显存 profiling】可用显存 ÷ 单页大小 ÷ 层数 = 每层可分配的块数 ------
    num_blocks = int(available_memory // page_size // num_layers)
    # ------【显存 profiling】防御性下限到 0，避免显存不足时出现负块数 ------
    num_blocks = max(num_blocks, 0)
    # ------【显存 profiling】支持用户 num_gpu_blocks_override 覆盖后再返回 ------
    return may_override_num_blocks(vllm_config, num_blocks)


def get_uniform_page_size(kv_cache_specs: Iterable[KVCacheSpec]) -> int:
    """
    Get the page size of the KV cache.
    """
    # ------【核心逻辑】收集所有层 page_size 集合，断言唯一后取出统一页大小 ------
    page_sizes = {layer.page_size_bytes for layer in kv_cache_specs}
    assert len(page_sizes) == 1
    return page_sizes.pop()


def _get_kv_cache_groups_uniform_spec(
    kv_cache_specs: dict[str, KVCacheSpec],
) -> list[KVCacheGroupSpec]:
    """
    Generates the KV cache configuration for a model with the same KV cache
    spec for all layers.

    Args:
        kv_cache_specs: The kv cache spec of each attention layer in the model

    Returns:
        The generated KVCacheGroupSpecs
    """

    # ------【核心逻辑】所有层同 spec，直接打包成单一分组复用同一份 block table ------
    return create_kv_cache_group_specs(kv_cache_specs, [list(kv_cache_specs.keys())])


def _get_kv_cache_groups_uniform_type(
    spec: UniformTypeKVCacheSpecs,
) -> list[KVCacheGroupSpec]:
    """
    Generates the KV cache configuration for a model with one type of KV cache
    but different hidden sizes. All layers are merged into one group.

    Args:
        spec: The UniformTypeKVCacheSpecs of the model

    Returns:
        The generated KVCacheGroupSpecs
    """

    # ------【核心逻辑】同型不同 hidden size，全部层合并为一个 UniformType 组 ------
    return [KVCacheGroupSpec(list(spec.kv_cache_specs.keys()), spec)]


def unify_kv_cache_spec_page_size(
    kv_cache_spec: dict[str, KVCacheSpec],
) -> dict[str, KVCacheSpec]:
    """
    Unify the page size of the given KVCacheSpec. If the page size of all layers
    are the same, return the original KVCacheSpec. If not same, unify the page
    size by increasing the block size of layers with smaller page size. Two
    cases cannot be unified by block size alone and pad their physical page to
    the maximum instead: Mamba layers, whose page size comes from state shapes
    and is independent of block size; and attention layers whose page does not
    evenly divide the maximum and whose backend opts in via
    ``AttentionSpec.indexes_kv_by_block_stride`` (the padded page is read through
    a strided view, which not every backend handles). Raise NotImplementedError
    if failed to unify the page size.

    Args:
        kv_cache_spec: The KVCacheSpec of each attention layer in the model

    Returns:
        The updated KVCacheSpec with the same page_size_bytes.
    """
    # ------【核心逻辑】收集各层 page_size 去重，判断是否需要统一 ------
    page_sizes = {layer.page_size_bytes for layer in kv_cache_spec.values()}
    if len(page_sizes) <= 1:
        # All layers have the same page size, no need to unify.
        return kv_cache_spec

    # ------【核心逻辑】以最大 page_size 为统一目标，准备存放改写后 spec 的新字典 ------
    max_page_size = max(page_sizes)
    new_kv_cache_spec = {}
    for layer_name, layer_spec in kv_cache_spec.items():
        # ------【核心逻辑】page_size 已等于目标值的层原样保留 ------
        if layer_spec.page_size_bytes == max_page_size:
            new_kv_cache_spec[layer_name] = layer_spec
        elif isinstance(layer_spec, MambaSpec):
            # MambaSpec's page size is determined by its state shapes and does
            # not scale with block_size, so pad the page instead. This is the
            # same padding mechanism the platform uses to align Mamba pages
            # with the main model's attention page size; it is needed here
            # when another layer (e.g. from a draft model) has a larger page
            # than the already-aligned Mamba page.
            # ------【内存池/CuMem】Mamba 页大小不随 block_size 缩放，只能物理补齐到目标页大小 ------
            new_spec: KVCacheSpec = replace(layer_spec, page_size_padded=max_page_size)
            assert new_spec.page_size_bytes == max_page_size
            new_kv_cache_spec[layer_name] = new_spec
        else:
            layer_page_size = layer_spec.page_size_bytes
            # ------【核心逻辑】目标页大小能整除本层页大小时，按比例放大 block_size 达到统一 ------
            if max_page_size % layer_page_size == 0:
                ratio = max_page_size // layer_page_size
                new_block_size = layer_spec.block_size * ratio
                new_spec = replace(layer_spec, block_size=new_block_size)
            elif (
                isinstance(layer_spec, AttentionSpec)
                and layer_spec.indexes_kv_by_block_stride
            ):
                # ------【内存池/CuMem】不能整除但后端按块步长索引，则物理补齐页大小（strided 视图） ------
                new_spec = replace(layer_spec, page_size_padded=max_page_size)
            else:
                # ------【核心逻辑】既不能整除也不支持补齐，直接报错拒绝统一 ------
                raise NotImplementedError(
                    f"Layer {layer_name}: page size is not divisible by the "
                    "maximum page size and cannot be padded. Padding is only "
                    "supported for attention layers whose backend indexes KV "
                    "pages by the block stride (indexes_kv_by_block_stride is "
                    "True)."
                )
            # ------【核心逻辑】断言改写后页大小已对齐目标，再写入新字典 ------
            assert new_spec.page_size_bytes == max_page_size
            new_kv_cache_spec[layer_name] = new_spec
    return new_kv_cache_spec


def is_kv_cache_type_attention_free(kv_cache_spec: dict[str, KVCacheSpec]) -> bool:
    # kv_cache_spec is an empty dict for attention free models
    # ------【核心逻辑】attention-free 模型 spec 为空字典，据此判断无需分配 KV cache ------
    return not kv_cache_spec


def _get_kv_cache_groups_uniform_page_size(
    kv_cache_spec: dict[str, KVCacheSpec],
) -> list[KVCacheGroupSpec]:
    """
    Generates the KV cache groups for hybrid models with multiple
    attention types but still with a uniform page size (physical memory per
    block per layer) for all layers.

    Detailed explanation about kv cache management of hybrid models:
    The layers in the models are repeated with some patterns, e.g., a model
    with 10 full attention layers and 20 sliding window attention layers can be
    regarded as repeating the pattern (1 * full, 2 * sw) 10 times.
    The KVCacheManager allocates different block tables for each of the 3 layers
    in the pattern, and repeats each of them 10 times to generate the
    block_table for the 30 layers in the model.
    Therefore, we can group the layers in the model into 3 kv_cache_groups, each
    of which contains 10 layers in the model.
    The KVCacheManager allocates the block_table for each group based on its
    kv_cache spec, and the model runner applies the block table to each layer
    in the group.
    For example:
    1. A model only uses full attention. The pattern is
    (num_hidden_layers * full), so there is only one group and the block table
    is shared by all layers. It is already handled by
    `_get_kv_cache_config_uniform_type`.
    2. A model with 10 full attention layers and 20 sliding window
    attention layers. There are 3 layers in the pattern (1 * full, 2 * sw), so
    there are 3 kv_cache_groups, each of which represents 10 layers.

    To simplify the implementation, we make the following assumptions:
    1. Physical memory per block: Must be the same across all KV cache groups.
    Breaking this assumption is non-trivial due to memory fragmentation concerns
    when allocating blocks of different sizes.
    2. Tokens per block (block_size): Currently, we directly use
    `CacheConfig.block_size` for all layers. It can be extended to vary by KV
    cache group, but within each KV cache group, all layers must share the same
    block size.
    3. Physical memory per token per layer: This property is decided by model
    config. Currently we only support models that have the same physical memory
    per token per layer for all layers. Can be relaxed with a simple extension,
    but still need to keep physical memory per block the same for all groups.
    4. Number of layers per group: Currently assumed the same for all layers.
    Can be relaxed with a simple extension, but still need to keep physical
    memory per block the same for all groups.
    5. Attention type within groups: All layers in a group must share the same
    attention type. One exception is that, when
    `--disable-hybrid-kv-cache-manager` is true, the single group for full
    attention layers may also include attention layers using sliding window or
    LLaMA 4 local attention. See `unify_hybrid_kv_cache_specs` for more details.
    6. Support for multiple attention types: The design for most components is
    general to an arbitrary number of attention types. But
    `find_longest_cache_hit` only supports one attention type or two
    types of full-attention plus exactly one another type. The general
    implementation of this function is feasible but we don't know how to
    implement it cleanly yet.

    As we assume tokens per block, physical memory per token per layer, and
    number of layers per group are the same now, we can ensure that physical
    memory per block is the same for all groups.

    Args:
        kv_cache_spec: The KVCacheSpec of each attention layer in the model
    Returns:
        The generated KVCacheGroupSpecs
    """
    # Group all layers by kv_cache_spec.
    # E.g., 2 full attention layers and 3 sliding window attention layers,
    # -> (full.0, full.1), (sw.0, sw.1, sw.2).
    # ------【核心逻辑】先按 KVCacheSpec 对象把各层归到同型桶，方便后续合并与分组 ------
    same_type_layers: dict[KVCacheSpec, list[str]] = defaultdict(list)
    for layer_name, layer_spec in kv_cache_spec.items():
        same_type_layers[layer_spec].append(layer_name)

    # Attempt to further merge same-type layers based on whether their KV
    # cache specs can be merged, to minimize the group count. This benefits
    # situations where specs share a block layout and differ only in a
    # property it can reconcile (e.g. full attention layers differing only in
    # sliding window / attention chunk size).
    # ------【核心逻辑】同型桶若可进一步 merge（仅差异可调和）则并入已有桶，减少组数 ------
    layer_buckets: list[list[str]] = []
    spec_buckets: list[list[KVCacheSpec]] = []
    for layer_spec, layer_names in same_type_layers.items():
        # ------【核心逻辑】尝试把当前层并入某个已能兼容的桶，能合并就退出 ------
        for names, specs in zip(layer_buckets, spec_buckets):
            try:
                # A raise means that the specs are incompatible.
                type(specs[0]).merge([*specs, layer_spec])
            except (AssertionError, ValueError):
                continue
            names.extend(layer_names)
            specs.append(layer_spec)
            break
        else:
            # ------【核心逻辑】没有兼容桶则新建一个桶，保存层名与代表 spec ------
            layer_buckets.append(list(layer_names))
            spec_buckets.append([layer_spec])

    # Split each group into smaller groups, to make the number of layers in each
    # group identical. Add padding to the last group of each type if necessary.
    # E.g., (full.0, full.1), (sw.0, sw.1, sw.2)
    # split to 3 groups with 2 layers each:
    # (full.0, full.1), (sw.0, sw.2), (sw.1, padding).
    # FIXME(Chen): At the moment of writing this code (2025-06-02), all
    # open-source hybrid model follows a n:1 pattern between different attention
    # types (e.g., Gemma3 5:1 between sw and full, LLaMA4 3:1 between local and
    # full), so we can use the "1" in the n:1 pattern as the group size, which
    # is the minimum number of layers among all attention types. Need a better
    # strategy if we want to support more complex patterns (e.g., 20 full + 30
    # sw, where the group size should be 10).
    # ------【核心逻辑】组大小取各型最小层数，保证每组都能凑齐对应层 ------
    min_num_layers = min([len(layers) for layers in layer_buckets])
    group_size = min_num_layers
    max_num_layers = max([len(layers) for layers in layer_buckets])
    if max_num_layers < min_num_layers * 1.5:
        # If the number of layers is not much larger than the minimum number of
        # layers, use the maximum number of layers as the group size to avoid
        # too many padding layers. A typical example is gpt-oss-20b + eagle,
        # with 12 sw + 13 full. We pad it to (13 sw, 13 full) instead of
        # (12 sw, 24 full). 1.5 is a heuristic to avoid too many padding
        # layers while accommodating speculative decoding drafters that add
        # extra layers to one attention type.
        # ------【投机解码+核心逻辑】层数差距不大时用最大值作组大小，避免 draft 引入过多 padding ------
        group_size = max_num_layers
    grouped_layers = []
    for layers in layer_buckets:
        # ------【核心逻辑】算本桶需要补的 padding 层数，非零则告警预估内存浪费比例 ------
        num_padding_layers = group_size - len(layers) % group_size
        if num_padding_layers != group_size:
            logger.warning(
                "Add %d padding layers, may waste at most %.2f%% KV cache memory",  # noqa
                num_padding_layers,
                num_padding_layers / len(layers) * 100,
            )
        num_groups = cdiv(len(layers), group_size)
        # In PP case, say if we have
        # - stage 0: full.0, sw.0, sw.1
        # - stage 1: full.1, sw.2, sw.3
        # We should have 3 groups: (full.0, full.1), (sw.0, sw.2), (sw.1, sw.3)
        # It can't be (full.0, full.1), (sw.0, sw.1), (sw.2, sw.3) because
        # the 3 groups in stage 0 will be (full.0), (sw.0, sw.1), (empty group)
        # and it will be padded to (full.0, padding), (sw.0, sw.1),
        # (padding, padding) to ensure the number of layers in each group is
        # the same and will cause memory waste.
        # To avoid this, we assign layers[i::num_groups] to the i-th group
        # instead of layers[i * group_size: (i + 1) * group_size]
        # ------【PP】按 i::num_groups 交错分配层，避免同 PP stage 出现空组浪费 padding ------
        for i in range(num_groups):
            grouped_layers.append(layers[i::num_groups])
    # ------【核心逻辑】用切好的组名列表生成最终的 KVCacheGroupSpec ------
    return create_kv_cache_group_specs(kv_cache_spec, grouped_layers)


def _get_packed_kv_cache_layout(
    kv_cache_groups: list[KVCacheGroupSpec],
) -> tuple[int, dict[int, list[str]]]:
    """Lay out each cache group densely in one shared block slab.

    A block ID is owned by one cache group at a time, so layouts from different
    groups may overlap. Layers within a group remain disjoint.
    """
    # ------【内存池/CuMem】初始化「偏移→层名列表」映射与块跨度累加器，紧凑铺放各组层 ------
    layers_by_offset: dict[int, list[str]] = defaultdict(list)
    block_stride = 0
    for group in kv_cache_groups:
        spec = group.kv_cache_spec
        byte_offset = 0
        # ------【内存池/CuMem】组内逐层把层铺到当前偏移处，同偏移的层共享同一物理块区间 ------
        for layer_name in group.layer_names:
            # ------【内存池/CuMem】UniformType 组按每层 page_size，其余按组统一 page_size 取值 ------
            if isinstance(spec, UniformTypeKVCacheSpecs):
                page_size = spec.kv_cache_specs[layer_name].page_size_bytes
            else:
                page_size = spec.page_size_bytes
            layers_by_offset[byte_offset].append(layer_name)
            byte_offset += page_size
        # ------【内存池/CuMem】块跨度取所有组所需最大字节数，作为整块 slab 的步长 ------
        block_stride = max(block_stride, byte_offset)
    # ------【内存池/CuMem】断言确实有层被铺放（块跨度非零），返回步长与偏移布局 ------
    assert block_stride > 0
    return block_stride, layers_by_offset


def _use_packed_kv_cache_config(
    vllm_config: VllmConfig,
    kv_cache_groups: list[KVCacheGroupSpec],
) -> bool:
    # ------【内存池/CuMem】所有组都是 UniformTypeKVCacheSpecs 即视为 DeepSeek V4 布局，默认启用 packed ------
    is_dsv4 = all(
        isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs)
        for group in kv_cache_groups
    )
    # ------【PD 分离】从 KV 连接器配置取 extra_config，读取跨层块开关（实验性） ------
    kv_transfer_config = vllm_config.kv_transfer_config
    extra_config = (
        kv_transfer_config.kv_connector_extra_config
        if kv_transfer_config is not None
        else {}
    )
    # NOTE: enable_cross_layers_blocks is an experimental API and subject to change with
    # https://github.com/vllm-project/vllm/issues/42082
    # ------【内存池/CuMem】解析字符串布尔开关，多组时允许跨层共享块 ------
    enable_cross_layers = (
        str(extra_config.get("enable_cross_layers_blocks", "False")).lower() == "true"
    )
    # ------【内存池/CuMem】DSV4 或（开启跨层且多于一组）时采用 packed 配置 ------
    return is_dsv4 or (enable_cross_layers and len(kv_cache_groups) > 1)


def _get_kv_cache_config_packed(
    vllm_config: VllmConfig,
    kv_cache_groups: list[KVCacheGroupSpec],
    available_memory: int,
) -> tuple[int, list[KVCacheTensor]]:
    """Plan a packed per-block KV cache tensor layout.

    Cache groups use dense, overlapping layouts within one block slab. Each
    emitted tensor aliases the same physical backing allocation.
    """
    # ------【内存池/CuMem】先算出紧凑布局的块跨度与各偏移共享层映射 ------
    block_stride, layers_by_offset = _get_packed_kv_cache_layout(kv_cache_groups)

    # ------【显存 profiling】可用显存 ÷ 块跨度得块数，并支持 override 覆盖 ------
    num_blocks = available_memory // block_stride
    num_blocks = may_override_num_blocks(vllm_config, num_blocks)

    # ------【内存池/CuMem】整块 slab 总字节数 = 步长 × 块数 ------
    total_size = block_stride * num_blocks

    # ------【内存池/CuMem】按偏移升序为每个共享槽位生成一个指向同一物理 slab 的 KVCacheTensor ------
    kv_cache_tensors: list[KVCacheTensor] = []
    for byte_offset in sorted(layers_by_offset):
        kv_cache_tensors.append(
            KVCacheTensor(
                size=total_size,
                shared_by=layers_by_offset[byte_offset],
                offset=byte_offset,
                block_stride=block_stride,
            )
        )

    return num_blocks, kv_cache_tensors


def get_kv_cache_config_from_groups(
    vllm_config: VllmConfig,
    kv_cache_groups: list[KVCacheGroupSpec],
    available_memory: int,
) -> KVCacheConfig:
    """
    Generate the KV cache configuration from the KV cache groups and spec
    of each layer.

    Args:
        vllm_config: The global VllmConfig
        kv_cache_groups: The KV cache groups
        available_memory: Memory available for KV cache in bytes
    Returns:
        The generated KVCacheConfig
    """
    # ------【核心逻辑】无组即 attention-free，返回 num_blocks=1 满足 BlockPool 的 null_block ------
    if len(kv_cache_groups) == 0:
        # Attention free models do not have KV cache.
        # Return num_blocks=1 as BlockPool always needs a null_block.
        return KVCacheConfig(
            num_blocks=1,
            kv_cache_tensors=[],
            kv_cache_groups=kv_cache_groups,
        )

    # Determine how model runners should initialize the KV cache tensors.
    # ------【内存池/CuMem】单组 UniformType 特例：按各层 hidden size 分配不同大小的张量 ------
    if len(kv_cache_groups) == 1 and isinstance(
        kv_cache_groups[0].kv_cache_spec, UniformTypeKVCacheSpecs
    ):
        # Special case: all layers have the same type of KV cache but with
        # different hidden sizes. Allocate different amount of memory for each
        # layer based on its hidden size.
        # ------【显存 profiling】可用显存 ÷ 该组 page_size 得块数，并支持 override ------
        num_blocks = (
            available_memory // kv_cache_groups[0].kv_cache_spec.page_size_bytes
        )
        num_blocks = may_override_num_blocks(vllm_config, num_blocks)
        per_layer_specs = kv_cache_groups[0].kv_cache_spec.kv_cache_specs
        # ------【内存池/CuMem】每层单独一个张量，大小按该层 page_size × 块数分配 ------
        kv_cache_tensors = [
            KVCacheTensor(
                size=per_layer_specs[layer_name].page_size_bytes * num_blocks,
                shared_by=[layer_name],
            )
            for layer_name in kv_cache_groups[0].layer_names
        ]
    elif _use_packed_kv_cache_config(vllm_config, kv_cache_groups):
        # DeepSeek V4 uses the packed layout by default. Other multi-group
        # layouts can opt in with --enable-cross-layers.
        # ------【内存池/CuMem】packed 布局：委托专用函数生成紧凑重叠的块 slab 与张量 ------
        num_blocks, kv_cache_tensors = _get_kv_cache_config_packed(
            vllm_config, kv_cache_groups, available_memory
        )
    else:
        # General case:
        # We will have group_size memory pools, each is shared by one layer from
        # each group. As layers of different groups have different block table,
        # they will use different parts of the shared Tensor.
        # The memory layout for 3 groups (full.0, full.1), (sw.0, sw.2),
        # (sw.1, padding) will be: (group_size = 2)
        # full.0, sw.0, sw.1: share a Tensor with size=available_memory//2
        # full.1, sw.2: share another Tensor with size=available_memory//2
        # ------【内存池/CuMem】组大小取各组层数最大值，统一 page_size 作共享池物理页大小 ------
        group_size = max(len(group.layer_names) for group in kv_cache_groups)

        page_size = get_uniform_page_size(
            [group.kv_cache_spec for group in kv_cache_groups]
        )
        assert group_size > 0, "group_size must be greater than 0"
        # ------【显存 profiling】通用路径按 group_size 个池平分显存计算块数 ------
        num_blocks = get_num_blocks(
            vllm_config, group_size, available_memory, page_size
        )
        kv_cache_tensors = []
        # ------【内存池/CuMem】第 i 个池收集各组第 i 层，让不同组的层共享同一张量 ------
        for i in range(group_size):
            shared_by = []
            for j in range(len(kv_cache_groups)):
                if i < len(kv_cache_groups[j].layer_names):
                    shared_by.append(kv_cache_groups[j].layer_names[i])
            kv_cache_tensors.append(
                KVCacheTensor(size=page_size * num_blocks, shared_by=shared_by)
            )

    # ------【核心逻辑】汇总块数、张量列表与分组，构造最终 KVCacheConfig ------
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=kv_cache_tensors,
        kv_cache_groups=kv_cache_groups,
    )


def _promote_local_kv_cache_specs(
    kv_cache_spec: dict[str, KVCacheSpec],
) -> dict[str, KVCacheSpec]:
    """Use full-attention allocation for local-attention cache specs.

    The returned specs affect KV cache management only. Attention modules keep
    their original sliding-window or chunked-local compute behavior.
    """
    # ------【核心逻辑】浅拷贝原 spec 字典，只在副本上改写避免污染输入 ------
    promoted_specs = kv_cache_spec.copy()

    # ------【核心逻辑】原本已统一（同型或同 spec）则无需提升，直接返回 ------
    if is_kv_cache_spec_uniform(
        promoted_specs
    ) or UniformTypeKVCacheSpecs.is_uniform_type(promoted_specs):
        return promoted_specs

    # ------【核心逻辑】检测模型中是否存在 full / sliding-window / chunked-local 三类注意力层 ------
    has_full_attention = any(
        isinstance(spec, FullAttentionSpec) for spec in promoted_specs.values()
    )
    has_sliding_window = any(
        isinstance(spec, SlidingWindowSpec) for spec in promoted_specs.values()
    )
    has_chunked_local_attention = any(
        isinstance(spec, ChunkedLocalAttentionSpec) for spec in promoted_specs.values()
    )
    # ------【核心逻辑】收集所有 full attention 层的 block_size，仅当唯一时作为统一基准 ------
    full_block_sizes = {
        spec.block_size
        for spec in promoted_specs.values()
        if isinstance(spec, FullAttentionSpec)
    }
    full_attention_block_size = (
        next(iter(full_block_sizes)) if len(full_block_sizes) == 1 else None
    )

    # ------【内存池/CuMem】提升后按新 block_size 重算补齐页大小，保证与统一布局对齐 ------
    def promoted_page_size_padded(spec: AttentionSpec, block_size: int) -> int | None:
        if spec.page_size_padded is None:
            return None
        unpadded_page_size = (
            spec.unpadded_page_size_bytes * block_size // spec.block_size
        )
        return max(spec.page_size_padded, unpadded_page_size)

    # ------【核心逻辑】full 与局部注意力共存时，把局部层提升为 full 分配 ------
    if has_full_attention and (has_sliding_window or has_chunked_local_attention):
        for layer_name, spec in kv_cache_spec.items():
            # ------【核心逻辑】SlidingWindowMLASpec 提升为 MLAAttentionSpec，仅改缓存分配不变计算 ------
            if isinstance(spec, SlidingWindowMLASpec):
                block_size = full_attention_block_size or spec.block_size
                promoted_specs[layer_name] = MLAAttentionSpec(
                    block_size=block_size,
                    num_kv_heads=spec.num_kv_heads,
                    head_size=spec.head_size,
                    dtype=spec.dtype,
                    page_size_padded=promoted_page_size_padded(spec, block_size),
                    cache_dtype_str=spec.cache_dtype_str,
                    alignment=spec.alignment,
                    compress_ratio=spec.compress_ratio,
                    model_version=spec.model_version,
                )
            # ------【核心逻辑】SlidingWindowSpec 提升为 FullAttentionSpec，保留滑窗参数供计算 ------
            elif isinstance(spec, SlidingWindowSpec):
                block_size = full_attention_block_size or spec.block_size
                promoted_specs[layer_name] = FullAttentionSpec(
                    block_size=block_size,
                    num_kv_heads=spec.num_kv_heads,
                    head_size=spec.head_size,
                    head_size_v=spec.head_size_v,
                    dtype=spec.dtype,
                    kv_quant_mode=spec.kv_quant_mode,
                    sliding_window=spec.sliding_window,
                    page_size_padded=promoted_page_size_padded(spec, block_size),
                )
            # ------【核心逻辑】ChunkedLocalAttentionSpec 提升为 FullAttentionSpec，保留 chunk 大小 ------
            elif isinstance(spec, ChunkedLocalAttentionSpec):
                block_size = full_attention_block_size or spec.block_size
                promoted_specs[layer_name] = FullAttentionSpec(
                    block_size=block_size,
                    num_kv_heads=spec.num_kv_heads,
                    head_size=spec.head_size,
                    dtype=spec.dtype,
                    attention_chunk_size=spec.attention_chunk_size,
                    page_size_padded=promoted_page_size_padded(spec, block_size),
                )

    # ------【核心逻辑】提升后仍不统一则报错，说明该混合结构无法用 full 分配兜底 ------
    if not (
        is_kv_cache_spec_uniform(promoted_specs)
        or UniformTypeKVCacheSpecs.is_uniform_type(promoted_specs)
    ):
        raise ValueError("Failed to promote local KV cache specs to one unified type.")

    return promoted_specs


def _try_get_full_allocation_fallback_groups(
    kv_cache_spec: dict[str, KVCacheSpec],
) -> list[KVCacheGroupSpec] | None:
    """Try a supported full-allocation fallback for local-attention layers."""
    # ------【核心逻辑】含 hidden-state 缓存层时不走 full 分配兜底，直接放弃 ------
    if any(isinstance(spec, HiddenStateCacheSpec) for spec in kv_cache_spec.values()):
        return None
    # ------【核心逻辑】含 SlidingWindowMLA 或 ChunkedLocal 层时也不适用该兜底路径 ------
    if any(
        isinstance(spec, (SlidingWindowMLASpec, ChunkedLocalAttentionSpec))
        for spec in kv_cache_spec.values()
    ):
        return None

    # ------【核心逻辑】兜底仅针对「MLA + 普通 sliding-window」这一组合，其他组合直接放弃 ------
    has_mla = any(isinstance(spec, MLAAttentionSpec) for spec in kv_cache_spec.values())
    has_regular_swa = any(
        isinstance(spec, SlidingWindowSpec) for spec in kv_cache_spec.values()
    )
    if not (has_mla and has_regular_swa):
        return None

    # ------【核心逻辑】尝试把 sliding-window 层提升为 full attention 分配，失败则返回 None ------
    try:
        promoted_specs = _promote_local_kv_cache_specs(kv_cache_spec)
    except ValueError:
        return None
    # ------【核心逻辑】把提升后的 spec 收敛成单一 UniformType，收敛不了则放弃 ------
    uniform_spec = UniformTypeKVCacheSpecs.from_specs(promoted_specs)
    if uniform_spec is None:
        return None
    # ------【核心逻辑】告警说明仅缓存分配退化为 full，滑窗计算语义不变，然后返回统一组 ------
    logger.warning(
        "KV cache page sizes cannot be unified; treating sliding-window "
        "layers as full attention for cache allocation. Sliding-window "
        "attention compute is unchanged."
    )
    return _get_kv_cache_groups_uniform_type(uniform_spec)


def unify_hybrid_kv_cache_specs(kv_cache_spec: dict[str, KVCacheSpec]):
    """
    This function tries to convert the KV cache specs to one type if the model
    is a hybrid model with multiple type of KV cache. It will convert all
    SlidingWindowSpec to FullAttentionSpec if both types are present.

    Args:
        kv_cache_spec: The kv cache spec of each attention layer in the model
    """

    # ------【核心逻辑】已统一（同型或同 spec）则无需处理，直接返回 ------
    if is_kv_cache_spec_uniform(
        kv_cache_spec
    ) or UniformTypeKVCacheSpecs.is_uniform_type(kv_cache_spec):
        return

    # ------【核心逻辑】告警：禁用混合 KV cache 管理器后滑窗外显存不裁剪，仅保留计算节省 ------
    logger.warning(
        "Hybrid KV cache manager is disabled for this hybrid model, "
        "This means we do not enable any optimizations for saving KV cache "
        "memory (e.g., dropping the KV cache outside the sliding window). "
        "The compute of layers like sliding window is still saved."
    )
    # ------【核心逻辑】把局部注意力层原地提升为 full attention 分配，覆盖原 spec ------
    kv_cache_spec.update(_promote_local_kv_cache_specs(kv_cache_spec))


def group_and_unify_kv_cache_specs(
    kv_cache_spec: dict[str, KVCacheSpec],
) -> list[UniformTypeKVCacheSpecs] | None:
    """
    Group the KV cache specs and unify each group into one UniformTypeKVCacheSpecs.
    Currently, this is only used for DeepseekV4.
    """
    # ------【核心逻辑】没有 SlidingWindowMLASpec 层则不是 DeepSeek V4 场景，返回 None 走通用路径 ------
    if not any(
        isinstance(spec, SlidingWindowMLASpec) for spec in kv_cache_spec.values()
    ):
        return None

    # SlidingWindowMLASpec models with uniform page sizes don't need tuple packing.
    # ------【核心逻辑】所有层页大小一致时无需 tuple packing，直接返回 None ------
    page_sizes = {spec.page_size_bytes for spec in kv_cache_spec.values()}
    if len(page_sizes) <= 1:
        return None

    # ------【核心逻辑】准备 MLA 字典与「(block_size,sliding_window)→层字典」的分桶容器 ------
    mla_specs: dict[str, KVCacheSpec] = {}
    grouped_swa_mla_specs: dict[tuple[int, int], dict[str, KVCacheSpec]] = defaultdict(
        dict
    )
    # NOTE: Here we group SWA layers by (block_size, sliding_window), which separates
    # SWA layers, C4I+C4A layers, and C128A layers into three different groups. It can
    # be fragile with only block_size and sliding_window as keys, but fine for now.
    # ------【核心逻辑】按类型分拣：SWA-MLA 按 (block_size, sliding_window) 分桶，MLA 单独收集 ------
    for name, spec in kv_cache_spec.items():
        if isinstance(spec, SlidingWindowMLASpec):
            grouped_swa_mla_specs[(spec.block_size, spec.sliding_window)][name] = spec
        elif isinstance(spec, MLAAttentionSpec):
            mla_specs[name] = spec

    # ------【核心逻辑】断言存在全注意力 MLA 层，并收敛成单一 UniformType spec ------
    assert len(mla_specs) > 0
    mla_uniform_spec = UniformTypeKVCacheSpecs.from_specs(mla_specs)
    assert mla_uniform_spec is not None

    # ------【核心逻辑】每个 SWA-MLA 桶各自收敛成 UniformType spec，追加到结果列表 ------
    swa_uniform_specs: list[UniformTypeKVCacheSpecs] = []
    for spec_dict in grouped_swa_mla_specs.values():
        uniform_spec = UniformTypeKVCacheSpecs.from_specs(spec_dict)
        assert uniform_spec is not None
        swa_uniform_specs.append(uniform_spec)

    # ------【核心逻辑】返回「全注意力 MLA + 各 SWA-MLA 组」的 spec 列表 ------
    return [mla_uniform_spec, *swa_uniform_specs]


def _approximate_gcd(values: Sequence[int], *, lower_bound: int | None = None) -> int:
    """Pick a chunk size that minimizes total upward padding.

    Each x is rounded up to a multiple of d:

      x -> ceil(x / d) * d

    Total padding is:

      pad(d) = sum_i (ceil(x_i / d) * d - x_i)

    We brute-force d in [lower_bound, max(values)] (fine for small lists / small
    maxima) and return the d with minimum padding. Ties prefer larger d.
    """
    # ------【核心逻辑】入参校验：空列表或含非正值直接报错 ------
    if not values:
        raise ValueError("values must be non-empty")
    if any(x <= 0 for x in values):
        raise ValueError(f"values must be positive, got: {list(values)!r}")

    # ------【核心逻辑】候选除数下界取 lower_bound，上界取最大值；下界超上界则直接返回下界 ------
    min_d = max(1, lower_bound if lower_bound is not None else 1)
    max_d = max(values)
    if min_d > max_d:
        return min_d

    # ------【核心逻辑】暴力枚举每个候选 d，选让总向上补齐量最小的（并列取更大 d） ------
    best_d = min_d
    best_pad: int | None = None
    for d in range(min_d, max_d + 1):
        pad = sum((d - (x % d)) % d for x in values)
        if best_pad is None or pad < best_pad or (pad == best_pad and d > best_d):
            best_pad = pad
            best_d = d

    return best_d


def _get_kv_cache_groups_uniform_groups(
    grouped_specs: list[UniformTypeKVCacheSpecs],
) -> list[KVCacheGroupSpec]:
    """
    Generate the KV cache groups from the grouped specs.
    """
    # ------【核心逻辑】校验所有分组 spec 都是 UniformType，且首个组只含 MLAAttentionSpec ------
    assert len(grouped_specs) > 0 and all(
        isinstance(spec, UniformTypeKVCacheSpecs) for spec in grouped_specs
    )
    # For now, we restrict the first grouped_spec to be UniformTypeKVCacheSpecs
    # containing only MLAAttentionSpec.
    full_mla_spec = grouped_specs[0]
    assert all(
        isinstance(spec, MLAAttentionSpec)
        for spec in full_mla_spec.kv_cache_specs.values()
    )
    # ------【核心逻辑】首个组单独构造为 full MLA 组，作为 packed 布局的基准组 ------
    full_mla_group = KVCacheGroupSpec(
        layer_names=list(full_mla_spec.kv_cache_specs.keys()),
        kv_cache_spec=full_mla_spec,
    )

    # We define a layer tuple as a group of layers with different page sizes, and
    # one UniformTypeKVCacheSpecs contains a list of layer tuples.
    # For example, if we have 11 C4 layers and 10 C128 layers, we can define a layer
    # tuple as [C4I, C4A, C128], and the full_mla_group will contain "11" layer tuples.
    # The other uniform KV cache specs will be similarly partitioned into layer tuples.
    # Say we have 21 SWA layers, all with the same page size, then we will have "21"
    # layer tuples.
    # ------【核心逻辑】统计各组已有的 layer tuple 数量，作为对齐基准 ------
    num_layer_tuples_per_group: list[int] = [
        g_spec.get_num_layer_tuples() for g_spec in grouped_specs
    ]
    # Choose `num_layer_tuples` to minimize total padding across groups.
    # ------【核心逻辑】用近似 GCD 选出最小化跨组 padding 的 tuple 数量（下界取首个组） ------
    num_layer_tuples = _approximate_gcd(
        num_layer_tuples_per_group, lower_bound=num_layer_tuples_per_group[0]
    )
    # Round up to the nearest multiple of `num_layer_tuples` (i.e., padding)
    # ------【核心逻辑】把各组 tuple 数向上取整到该数量倍数，完成 padding 对齐 ------
    num_layer_tuples_per_group = [
        round_up(x, num_layer_tuples) for x in num_layer_tuples_per_group
    ]

    # ------【核心逻辑】取出剩余的 SWA-MLA 组并断言其层都是 SlidingWindowMLASpec ------
    swa_mla_specs = grouped_specs[1:]
    assert all(
        isinstance(spec, SlidingWindowMLASpec)
        for group in swa_mla_specs
        for spec in group.kv_cache_specs.values()
    )

    # Split each SWA UniformKV group into smaller groups to align their
    # numbers of layer tuples. The packed block planner overlays groups, so
    # their page sizes do not need to match.
    # ------【内存池/CuMem】按页大小把每个 SWA 组内层再分桶，构造与 packed 块布局对齐的子组 ------
    swa_mla_groups = []
    for sm_spec in swa_mla_specs:
        layers_per_size: dict[int, list[str]] = defaultdict(list)

        # ------【核心逻辑】同一 UniformKV 组内按 page_size 把层名归类，得到各页大小的层列表 ------
        for layer_name, layer_spec in sm_spec.kv_cache_specs.items():
            layers_per_size[layer_spec.page_size_bytes].append(layer_name)
        # NOTE(yifan): for now, inside a UniformKV group, each page_size should
        # have the same number of layers. This also means we don't need to pad layers
        # inside a partial-full layer tuple.
        # ------【核心逻辑】断言每个页大小对应层数一致，并取单页大小的层数 ------
        assert len(set(len(layers) for layers in layers_per_size.values())) == 1
        num_layers_per_size = len(next(iter(layers_per_size.values())))

        # Split layers inside each UniformKV group for aligned #(layers).
        # See `_get_kv_cache_groups_uniform_page_size` for more details.
        # ------【核心逻辑】按 tuple 数切子组，zip 把同位置各页大小层拼成一个 layer tuple ------
        num_tuple_groups = cdiv(num_layers_per_size, num_layer_tuples)
        layer_tuples = list(zip(*layers_per_size.values()))
        for i in range(num_tuple_groups):
            # ------【核心逻辑】交错取 layer tuple 到第 i 个子组，摊平层名并重建子组 spec ------
            group_layer_tuples = layer_tuples[i::num_tuple_groups]
            # Flatten tuples and build dict for from_specs
            group_layer_names = [
                name for layer_tuple in group_layer_tuples for name in layer_tuple
            ]
            group_layer_specs = {
                name: sm_spec.kv_cache_specs[name] for name in group_layer_names
            }
            sub_sm_spec = UniformTypeKVCacheSpecs.from_specs(group_layer_specs)
            assert sub_sm_spec is not None
            # ------【内存池/CuMem】用摊平后的层名与子组 spec 构造 KVCacheGroupSpec 追加到结果 ------
            swa_mla_groups.append(
                KVCacheGroupSpec(
                    layer_names=group_layer_names,
                    kv_cache_spec=sub_sm_spec,
                )
            )

    # ------【核心逻辑】返回「full MLA 组 + 各 SWA-MLA 子组」的完整分组列表 ------
    return [full_mla_group, *swa_mla_groups]


def _annotate_eagle_groups_deepseek_v4(
    vllm_config: VllmConfig,
    kv_cache_spec: dict[str, KVCacheSpec],
    kv_cache_groups: list[KVCacheGroupSpec],
) -> None:
    # ------【投机解码】未启用 EAGLE 投机解码则无需标记，直接返回 ------
    spec_config = vllm_config.speculative_config
    if spec_config is None or not spec_config.use_eagle():
        return
    # Detection uses the merged MLA spec's model_version.
    # ------【投机解码】按合并后 MLA spec 的 model_version 判断是否 DeepSeek V4，非 V4 直接返回 ------
    if not any(
        getattr(spec, "model_version", None) == "deepseek_v4"
        for spec in kv_cache_spec.values()
    ):
        return
    # DeepseekV4's MTP attention layer is always the last layer, and we flag whichever
    # group contains it.
    # FIXME(yifan): avoid/generalize this hacky check.
    # ------【投机解码】取最后一层（MTP 层），把包含它的组标记为 eagle 组供草稿模型使用 ------
    last_layer = next(reversed(kv_cache_spec))
    for group in kv_cache_groups:
        if last_layer in group.layer_names:
            group.is_eagle_group = True
            break


def _largest_divisor_at_most(value: int, limit: int) -> int:
    # ------【核心逻辑】从 min(value,limit) 向下找 value 的最大约数，找不到返回 1 ------
    for candidate in range(min(value, limit), 0, -1):
        if value % candidate == 0:
            return candidate
    return 1


def get_kv_cache_groups(
    vllm_config: VllmConfig, kv_cache_spec: dict[str, KVCacheSpec]
) -> list[KVCacheGroupSpec]:
    """
    Split the layers in the model into groups with the same KV cache spec.

    Args:
        vllm_config: The global VllmConfig
        kv_cache_spec: The kv cache spec of each attention layer in the model

    Returns:
        The generated KVCacheGroups
    """
    # ------【核心逻辑】禁用混合 KV cache 管理器时，先把混合 spec 统一成单一类型 ------
    if vllm_config.scheduler_config.disable_hybrid_kv_cache_manager:
        unify_hybrid_kv_cache_specs(kv_cache_spec)

    # ------【核心逻辑】attention-free 模型返回空列表，交由 KVCacheManager 特殊处理 ------
    if is_kv_cache_type_attention_free(kv_cache_spec):
        # This returns an empty list to allow for the KVCacheManager to handle
        # attention free models.
        return []

    # ------【核心逻辑】所有层 spec 完全相同（多数模型），打包成单组共享 block table ------
    if is_kv_cache_spec_uniform(kv_cache_spec):
        # KV cache of all layers are the same, which is true for
        # most models. Allocate the same amount of memory for
        # each layer.
        return _get_kv_cache_groups_uniform_spec(kv_cache_spec)
    elif uniform_spec := UniformTypeKVCacheSpecs.from_specs(kv_cache_spec):
        # All layers need the same number of token slots (e.g., all layers are
        # full attention, or all layers are sliding window attention with the
        # same window size). Put all layers into one group.
        # ------【核心逻辑】同型不同 hidden size 时，全部层合并为一个 UniformType 组 ------
        return _get_kv_cache_groups_uniform_type(uniform_spec)
    elif grouped_specs := group_and_unify_kv_cache_specs(kv_cache_spec):
        # DeepseekV4 case: All layers need the same number of token slots,
        # yet some layers are full attention while others are sliding window
        # attention in different sizes. Need to group layers into multiple
        # UniformTypeKVCacheSpecs.
        # ------【核心逻辑】DeepSeek V4 场景：按 full/SWA 分组统一，并标注 eagle 组后返回 ------
        kv_cache_groups = _get_kv_cache_groups_uniform_groups(grouped_specs)
        _annotate_eagle_groups_deepseek_v4(vllm_config, kv_cache_spec, kv_cache_groups)
        return kv_cache_groups

    # Pull HiddenStateCacheSpec layers out before the general multi-group
    # path so they don't affect page-size unification or grouping.
    # ------【核心逻辑】先把 HiddenStateCacheSpec 层单独抽出，避免干扰后续页大小统一与分组 ------
    hidden_specs = {
        k: v for k, v in kv_cache_spec.items() if isinstance(v, HiddenStateCacheSpec)
    }
    filtered_spec = {
        k: v
        for k, v in kv_cache_spec.items()
        if not isinstance(v, HiddenStateCacheSpec)
    }

    # Prefer preserving each layer's cache semantics. If physical pages cannot
    # be unified, try a supported allocation-only fallback before failing.
    # ------【核心逻辑】优先统一页大小；失败时尝试 full 分配兜底，仍不行才抛错 ------
    try:
        filtered_spec = unify_kv_cache_spec_page_size(filtered_spec)
    except NotImplementedError:
        fallback_groups = _try_get_full_allocation_fallback_groups(kv_cache_spec)
        if fallback_groups is None:
            raise
        return fallback_groups
    # ------【核心逻辑】统一页大小后，按 uniform page size 路径生成分组 ------
    groups = _get_kv_cache_groups_uniform_page_size(filtered_spec)

    # Add hidden-state layers back with page aligned to the common page.
    # ------【核心逻辑】把 hidden-state 层加回分组，页对齐公共页，block_size 取最大可行约数 ------
    if hidden_specs:
        common_page = get_uniform_page_size([g.kv_cache_spec for g in groups])
        group_block_size = math.gcd(*(g.kv_cache_spec.block_size for g in groups))
        for name, spec in hidden_specs.items():
            # ------【核心逻辑】由每 token 字节数算最大 block_size，再取 group block_size 的最大约数 ------
            per_token = spec.num_kv_heads * spec.head_size * get_dtype_size(spec.dtype)
            max_block_size = max(common_page // per_token, 1)
            new_bs = _largest_divisor_at_most(group_block_size, max_block_size)
            # ------【内存池/CuMem】统计对齐浪费的字节，打印日志后构造对齐后的 spec 并追加组 ------
            wasted_bytes = common_page - new_bs * per_token
            logger.info(
                "Using block size %d for hidden-state cache layer %s; "
                "page alignment wastes %d bytes (%.2f%%) per block",
                new_bs,
                name,
                wasted_bytes,
                wasted_bytes / common_page * 100,
            )
            aligned = replace(spec, block_size=new_bs, page_size_padded=common_page)
            groups.append(KVCacheGroupSpec([name], aligned))

    return groups


def generate_scheduler_kv_cache_config(
    kv_cache_configs: list[KVCacheConfig],
) -> KVCacheConfig:
    """
    Generate the KV cache configuration for the scheduler.
    """
    # ------【核心逻辑】断言所有 worker 的 num_blocks 一致，保证集中式调度器视图统一 ------
    assert all(
        [cfg.num_blocks == kv_cache_configs[0].num_blocks for cfg in kv_cache_configs]
    )
    # All workers have the same kv_cache_config except layer names, so use
    # an arbitrary one to initialize the scheduler.
    # ------【核心逻辑】深拷贝任一份配置作为调度器基础，避免改动共享对象 ------
    cfg = copy.deepcopy(kv_cache_configs[0])
    for group in cfg.kv_cache_groups:
        # ------【核心逻辑】UniformType 组的 spec 内各层同型，任取一个代表 spec 供调度器使用 ------
        if isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs):
            # All layers in the UniformTypeKVCacheSpecs have the same type,
            # so use an arbitrary one to initialize the scheduler.
            group.kv_cache_spec = next(
                iter(group.kv_cache_spec.kv_cache_specs.values())
            )
    return cfg


def get_kv_cache_capacity(
    vllm_config: VllmConfig, kv_cache_config: KVCacheConfig
) -> tuple[int, float]:
    """
    Get the group-aware KV cache token capacity and max concurrency.
    """
    # ------【核心逻辑】取 max_model_len 与最大并发，二者相乘即为可服务的 token 容量 ------
    max_model_len = vllm_config.model_config.max_model_len
    max_concurrency = get_max_concurrency_for_kv_cache_config(
        vllm_config, kv_cache_config
    )
    return int(max_concurrency * max_model_len), max_concurrency


def update_kv_cache_capacity(
    vllm_config: VllmConfig, kv_cache_config: KVCacheConfig
) -> None:
    """Store and log the resolved KV cache capacity."""
    # ------【核心逻辑】计算 token 容量与最大并发 ------
    num_tokens, max_concurrency = get_kv_cache_capacity(vllm_config, kv_cache_config)
    # ------【核心逻辑】把解析出的容量写回 cache_config，供后续统计与调度使用 ------
    vllm_config.cache_config.kv_cache_size_tokens = num_tokens
    vllm_config.cache_config.kv_cache_max_concurrency = max_concurrency
    max_model_len = vllm_config.model_config.max_model_len
    # ------【核心逻辑】一次性打印 GPU KV cache 大小与按 max_model_len 计的并发倍数 ------
    logger.info_once(
        "GPU KV cache size: %s tokens, "
        "Maximum concurrency for %s tokens per request: %.2fx",
        f"{num_tokens:,}",
        f"{max_model_len:,}",
        max_concurrency,
    )


def _max_memory_usage_bytes_from_groups(
    vllm_config: VllmConfig,
    kv_cache_groups: list[KVCacheGroupSpec],
) -> int:
    """
    Calculate maximum memory usage in bytes from KV cache groups.

    This correctly accounts for padding in hybrid models. For example, if a
    model has 8 full attention layers and 9 sliding window layers, they will
    be padded to 9 full + 9 sliding window for uniform group sizes.
    """
    # ------【显存 profiling】无组即无 KV cache，占用为 0 ------
    if not kv_cache_groups:
        return 0

    # ------【显存 profiling】单组 UniformType 特例：逐层求和各自最大占用 ------
    if len(kv_cache_groups) == 1 and isinstance(
        kv_cache_groups[0].kv_cache_spec, UniformTypeKVCacheSpecs
    ):
        # UniformTypeKVCacheSpecs special case (single group, per-layer specs)
        per_layer_specs = kv_cache_groups[0].kv_cache_spec.kv_cache_specs
        return sum(
            spec.max_memory_usage_bytes(vllm_config)
            for spec in per_layer_specs.values()
        )
    elif all(
        isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs)
        for group in kv_cache_groups
    ):
        # Special case (only DeepseekV4 for now): all groups are
        # UniformTypeKVCacheSpecs.
        # They must already be page_size aligned and share a common padded
        # layer-tuple layout. Even groups with fewer actual tuples still reserve
        # the global number of tuple slots in the shared tensor layout.
        # ------【显存 profiling】DSV4 全 UniformType 路径：按 layer-tuple 布局算共享 slab 的峰值占用 ------
        full_mla_spec = cast(UniformTypeKVCacheSpecs, kv_cache_groups[0].kv_cache_spec)
        # ------【显存 profiling】单个 layer tuple 字节数 = 该组各页大小之和；tuple 数取各组最大 ------
        layer_tuple_bytes = sum(full_mla_spec.get_page_sizes())
        num_layer_tuples = max(
            cast(UniformTypeKVCacheSpecs, group.kv_cache_spec).get_num_layer_tuples()
            for group in kv_cache_groups
        )

        # ------【显存 profiling】各组按「tuple 数 × 所需页数 × tuple 字节数」累计最大占用 ------
        total_max_mem_usage_bytes = 0
        for group in kv_cache_groups:
            group_spec = cast(UniformTypeKVCacheSpecs, group.kv_cache_spec)
            g_max_mem_usage_pages = group_spec.max_memory_usage_pages(vllm_config)
            g_max_mem_usage_page_bytes = (
                num_layer_tuples * g_max_mem_usage_pages * layer_tuple_bytes
            )
            total_max_mem_usage_bytes += g_max_mem_usage_page_bytes
        return total_max_mem_usage_bytes

    # General case: group_size pools, each shared by one layer per group
    # Memory = group_size * page_size * blocks_for_max_len
    # ------【显存 profiling】通用路径：组大小与统一页大小确定后，先算支撑 max_len 的总块数 ------
    group_size = max(len(group.layer_names) for group in kv_cache_groups)
    page_size = get_uniform_page_size(
        [group.kv_cache_spec for group in kv_cache_groups]
    )
    blocks_needed = sum(
        cdiv(group.kv_cache_spec.max_memory_usage_bytes(vllm_config), page_size)
        for group in kv_cache_groups
    )

    # ------【显存 profiling】峰值占用 = 组大小 × 页大小 × 所需块数 ------
    return group_size * page_size * blocks_needed


def _estimate_max_model_len_from_groups(
    vllm_config: VllmConfig,
    kv_cache_groups: list[KVCacheGroupSpec],
    available_memory: int,
) -> int:
    """
    Binary search for the maximum model length that fits in available memory.
    Returns 0 if even 1 token doesn't fit.
    """
    # ------【显存 profiling】暂存原 max_model_len，估算完恢复 ------
    original_max = vllm_config.model_config.max_model_len

    # ------【显存 profiling】闭包：临时改 max_model_len 按分组算峰值占用，判断能否放入显存 ------
    def fits(model_len: int) -> bool:
        vllm_config.model_config.max_model_len = model_len
        return (
            _max_memory_usage_bytes_from_groups(vllm_config, kv_cache_groups)
            <= available_memory
        )

    try:
        # ------【显存 profiling】二分搜索边界 [1, original_max] ------
        left, right = 1, original_max
        # ------【显存 profiling】最小长度都放不下则返回 0，表示无法服务任何请求 ------
        if not fits(left):
            return 0
        result = 1
        # ------【显存 profiling】单调区间上二分逼近能容纳的最大长度 ------
        while left <= right:
            mid = (left + right) // 2
            if fits(mid):
                result = mid
                left = mid + 1
            else:
                right = mid - 1
        return result
    finally:
        # ------【显存 profiling】无论成败恢复原 max_model_len，保证无副作用 ------
        vllm_config.model_config.max_model_len = original_max


def _auto_fit_max_model_len(
    vllm_config: VllmConfig,
    projected_groups_per_worker: list[list[KVCacheGroupSpec]],
    available_memory: list[int],
) -> None:
    """
    When max_model_len is set to -1, this function estimates the largest
    context length that can be supported with the available GPU memory.
    It uses binary search to find the maximum length that fits across all
    workers.

    Args:
        vllm_config: The global VllmConfig (will be modified in-place)
        projected_groups_per_worker: KV cache groups projected to each worker.
        available_memory: Memory available for KV cache in bytes for each
            worker.
    """
    # ------【显存 profiling】暂存原 max_model_len（可能是 -1），作为二分上界 ------
    original_max = vllm_config.model_config.max_model_len

    # ------【显存 profiling】所有 worker 无 KV cache（attention-free）则直接用推导值返回 ------
    if all(not groups for groups in projected_groups_per_worker):
        # All workers have empty specs (attention-free model)
        logger.info_once(
            "Auto-fit max_model_len: attention-free model, "
            "using derived max_model_len=%d",
            original_max,
        )
        return

    # Find the max_model_len that fits across all workers.
    # ------【PP+显存 profiling】逐 worker 估算能容纳的最大长度，取所有 worker 的最小值（受最紧内存 worker 约束） ------
    auto_fit_max = original_max
    limiting_worker_mem = available_memory[0]
    for groups, avail_mem in zip(projected_groups_per_worker, available_memory):
        if not groups:
            continue
        worker_max = _estimate_max_model_len_from_groups(vllm_config, groups, avail_mem)
        if worker_max < auto_fit_max:
            auto_fit_max = worker_max
            limiting_worker_mem = avail_mem

    # ------【显存 profiling】连单 token 都放不下则报错，提示调大 gpu_memory_utilization ------
    if auto_fit_max <= 0:
        raise ValueError(
            "Cannot auto-fit max_model_len: not enough GPU memory available "
            "to serve even a single token. Try increasing `gpu_memory_utilization`."
        )

    # ------【显存 profiling】完整上下文能放下时保持原值并打印确认 ------
    if auto_fit_max >= original_max:
        # The model's full context length fits in memory
        logger.info_once(
            "Auto-fit max_model_len: full model context length %d fits in "
            "available GPU memory",
            original_max,
        )
    else:
        # Need to reduce max_model_len to fit in memory
        # ------【显存 profiling】需缩容时把 max_model_len 回写为估算值并打印缩容信息 ------
        vllm_config.model_config.max_model_len = auto_fit_max
        logger.info_once(
            "Auto-fit max_model_len: reduced from %d to %d to fit in "
            "available GPU memory (%s GiB available for KV cache)",
            original_max,
            auto_fit_max,
            format_gib(limiting_worker_mem),
        )


def _project_kv_cache_groups_to_worker(
    global_kv_cache_groups: list[KVCacheGroupSpec],
    worker_spec: dict[str, KVCacheSpec],
) -> list[KVCacheGroupSpec]:
    """
    Projects global KV cache groups onto a single worker's assigned layers.

    In pipeline parallelism, each worker only owns a subset of layers. This
    function filters the global groups to include only layers present on the
    given worker, adjusting UniformTypeKVCacheSpecs accordingly.

    Args:
        global_kv_cache_groups: The global KV cache groups for the whole model.
        worker_spec: The KV cache spec of each layer on this worker.

    Returns:
        The projected KV cache groups containing only this worker's layers.
    """
    # ------【PP】初始化投影结果，逐全局组过滤出当前 worker 拥有的层 ------
    projected_groups: list[KVCacheGroupSpec] = []
    for group in global_kv_cache_groups:
        # ------【PP】仅保留出现在本 worker spec 中的层名，得到该 worker 的层子集 ------
        worker_layer_names = [
            layer_name for layer_name in group.layer_names if layer_name in worker_spec
        ]
        group_spec = group.kv_cache_spec
        # ------【PP】UniformType 组需按子集重建，剔除本 worker 没有的层 ------
        if worker_layer_names and isinstance(group_spec, UniformTypeKVCacheSpecs):
            group_spec = UniformTypeKVCacheSpecs(
                block_size=group_spec.block_size,
                kv_cache_specs={
                    layer_name: group_spec.kv_cache_specs[layer_name]
                    for layer_name in worker_layer_names
                },
            )
        # ------【PP】构造投影后的组；eagle 标记仅在本 worker 确实含该层时保留 ------
        projected_groups.append(
            KVCacheGroupSpec(
                worker_layer_names,
                group_spec,
                is_eagle_group=group.is_eagle_group and bool(worker_layer_names),
            )
        )
    return projected_groups


def get_kv_cache_configs(
    vllm_config: VllmConfig,
    kv_cache_specs: list[dict[str, KVCacheSpec]],
    available_memory: list[int],
) -> list[KVCacheConfig]:
    """
    Generates the KV cache configurations for a model.
    Since we use a shared centralized controller for all workers, we need the
    `kv_cache_config` to be consistent across all workers to make sure
    the KV cache allocation can be applied to all workers. However, different
    workers may have different memory available, and different type of layers
    (when pipeline parallel is enabled). To handle the difference between
    workers, the current implementation is:
    1. Merge the KV cache specs of all workers to get the KVCacheSpecs for
       the whole model.
    2. Generate the KV cache groups based on the layer ratio of the whole model.
       This also handles spec unification for hybrid models.
    3. Handle auto-fit max_model_len and memory checks using per-worker
       projected groups to account for PP sharding.
    4. Generate the KV cache configs for each worker based on the KV cache
       grouping strategy. (This is reasonable because the layer ratio of
       different PP stages are similar.)
    5. Change the num_blocks of each worker to the smallest among all workers
       and shrink tensor sizes proportionally to avoid allocating unused memory.

    Args:
        vllm_config: The global VllmConfig
        kv_cache_specs: List of dict[layer_name, KVCacheSpec] for each worker.
        available_memory: Memory available for KV cache in bytes for each
            worker.

    Returns:
        The generated KVCacheConfigs for each worker.
    """

    # Merge the KV cache specs of all workers. Different PP stages may have
    # different layer names, and different TP ranks of the same PP stage should
    # have the same KV cache spec.
    # ------【PP+核心逻辑】合并所有 worker 的 KV cache spec 成全局视图，并校验同层跨 worker 一致 ------
    merged_kv_cache_specs: dict[str, KVCacheSpec] = {}
    for kv_cache_spec_one_worker in kv_cache_specs:
        for layer_name, layer_spec in kv_cache_spec_one_worker.items():
            if layer_name not in merged_kv_cache_specs:
                merged_kv_cache_specs[layer_name] = layer_spec
            else:
                assert merged_kv_cache_specs[layer_name] == layer_spec, (
                    "The KV cache specs for the same layer are different "
                    "across workers. This is not supported yet."
                )

    # Check if the KV cache specs are registered correctly.
    # This is to prevent that some layers are initialized with unregistered specs.
    # ------【核心逻辑】校验合并后的 spec 已正确注册，防止出现未注册 spec 的层 ------
    KVCacheSpecRegistry.check_kv_cache_spec_registry(merged_kv_cache_specs)
    # Get global KV cache groups. This also handles spec unification for
    # hybrid models when disable_hybrid_kv_cache_manager is enabled.
    # After this call, merged_kv_cache_specs may be modified in-place.
    # ------【核心逻辑】基于全局 spec 生成 KV cache 分组（含混合模型统一处理） ------
    global_kv_cache_groups = get_kv_cache_groups(vllm_config, merged_kv_cache_specs)

    # If original_max_model_len was -1, automatically
    # determine the maximum model length that fits in available GPU memory.
    # We use per-worker projected groups to account for PP sharding.
    # ------【PP】把全局分组投影到各 worker，得到每个 worker 实际拥有的层分组 ------
    projected_groups_per_worker = [
        _project_kv_cache_groups_to_worker(global_kv_cache_groups, worker_spec)
        for worker_spec in kv_cache_specs
    ]

    # If `num_gpu_blocks_override` is set, the cache size that will actually
    # be allocated is decoupled from the profiled `available_memory`:
    # `may_override_num_blocks` in `get_kv_cache_config_from_groups` clamps
    # `num_blocks` to the override. Reflect that in `available_memory` here so
    # auto-fit, the admission check, and the per-worker config builder all
    # plan against the same effective capacity.
    # ------【显存 profiling】若设置了 num_gpu_blocks_override，把各 worker 可用显存折算成覆盖后的有效容量 ------
    override = vllm_config.cache_config.num_gpu_blocks_override
    if override is not None:
        adjusted_memory: list[int] = []
        for groups, avail_mem in zip(projected_groups_per_worker, available_memory):
            if not groups:
                adjusted_memory.append(avail_mem)
                continue
            # ------【显存 profiling】按该 worker 每块字节数把 override 块数折算回等价显存 ------
            bytes_per_block = _pool_bytes_per_block(vllm_config, groups)
            logger.info(
                "Overriding num_gpu_blocks=%d with num_gpu_blocks_override=%d",
                avail_mem // bytes_per_block,
                override,
            )
            adjusted_memory.append(override * bytes_per_block)
        available_memory = adjusted_memory

    # ------【显存 profiling】max_model_len 为 -1 时自动拟合能容纳的最大上下文长度 ------
    if vllm_config.model_config.original_max_model_len == -1:
        _auto_fit_max_model_len(
            vllm_config, projected_groups_per_worker, available_memory
        )

    # Check if the available memory is enough per worker.
    # ------【显存 profiling】逐 worker 校验可用显存是否足以支撑一条 max_model_len 请求 ------
    for groups, avail_mem in zip(projected_groups_per_worker, available_memory):
        if not groups:
            continue
        _check_enough_kv_cache_memory(
            avail_mem,
            partial(_max_memory_usage_bytes_from_groups, vllm_config, groups),
            vllm_config.model_config.max_model_len,
            partial(_estimate_max_model_len_from_groups, vllm_config, groups),
        )

    # ------【核心逻辑】为每个 worker 生成独立的 KVCacheConfig，先断言所有层都已被分配到组 ------
    kv_cache_configs: list[KVCacheConfig] = []
    for projected_groups, kv_cache_spec_one_worker, available_memory_one_worker in zip(
        projected_groups_per_worker, kv_cache_specs, available_memory
    ):
        assert sum(len(group.layer_names) for group in projected_groups) == len(
            kv_cache_spec_one_worker
        ), "Some layers are not assigned to any group."
        kv_cache_configs.append(
            get_kv_cache_config_from_groups(
                vllm_config, projected_groups, available_memory_one_worker
            )
        )

    # Change the num_blocks of each rank to the smallest among all ranks.
    # We also need to shrink the tensor size proportionally to avoid
    # allocating unused memory.
    # ------【内存池/CuMem】取所有 worker 中最小的块数作为统一块数，保证跨 worker 分配一致 ------
    min_num_blocks = min(
        kv_cache_config.num_blocks for kv_cache_config in kv_cache_configs
    )
    for kv_cache_config in kv_cache_configs:
        num_blocks_old = kv_cache_config.num_blocks
        kv_cache_config.num_blocks = min_num_blocks

        # Shrink tensor size proportionally
        # ------【内存池/CuMem】按新块数等比例收缩每个张量，避免为多余块分配无用显存 ------
        for tensor in kv_cache_config.kv_cache_tensors:
            assert tensor.size % num_blocks_old == 0
            tensor.size = tensor.size // num_blocks_old * min_num_blocks

    return kv_cache_configs


class BlockHashListWithBlockSize:
    """
    Convert block-hash granularity from `hash_block_size` to `target_block_size`.
    Used when KV cache groups have different block sizes: `hash_block_size`
    is the size used to compute the original `block_hashes`; `target_block_size`
    is the group's actual block size.

    Currently, only scaling up by an integer factor is supported (i.e.,
    `target_block_size` is a multiple of `hash_block_size`). Conversion is
    performed lazily on access for efficiency. Each `hash_block_size` hash is
    already chained over its entire prefix, so the hash at the last
    `hash_block_size` boundary of a `target_block_size` block uniquely
    fingerprints that block's prefix; we use it directly.

    Example (`hash_block_size` = 16, `target_block_size` = 32):
    the second 16-size hash already covers tokens 0-31, so it is the 32-size
    hash:

    Block hashes with block_size 16:
    | Token Range | 0-15 | 16-31 | 32-47 | 48-63 |
    |-------------|------|-------|-------|-------|
    | Hash        | A    | B     | C     | D     |

    Block hashes with block_size 32:
    | Token Range | 0-31 | 32-63 |
    |-------------|------|-------|
    | Hash        | B    | D     |

    Args:
        block_hashes: Block hashes to convert, computed at `hash_block_size`.
        hash_block_size: Block size at which `block_hashes` were computed.
        target_block_size: Desired block size; must be a multiple of `hash_block_size`.
    """

    def __init__(
        self,
        block_hashes: list[BlockHash],
        hash_block_size: int,
        target_block_size: int,
    ):
        # ------【前缀缓存】保存原哈希并断言 target 是 hash 块大小的整数倍，算出放大因子 ------
        self.block_hashes = block_hashes
        assert target_block_size % hash_block_size == 0
        self.scale_factor = target_block_size // hash_block_size

    def __len__(self) -> int:
        # ------【前缀缓存】视图长度 = 原哈希数 ÷ 放大因子（目标块更粗，块数更少） ------
        return len(self.block_hashes) // self.scale_factor

    @overload
    def __getitem__(self, idx: int) -> BlockHash: ...

    @overload
    def __getitem__(self, idx: slice) -> list[BlockHash]: ...

    def __getitem__(self, idx):
        # ------【前缀缓存】整数下标直接取对应目标块的哈希 ------
        if isinstance(idx, int):
            return self._get_value_at(idx)

        # ------【前缀缓存】切片下标先归一化 start/stop/step，再逐个取目标块哈希 ------
        if isinstance(idx, slice):
            start, stop, step = idx.indices(len(self))
            return [self._get_value_at(i) for i in range(start, stop, step)]

        # ------【前缀缓存】其他类型下标不支持，抛 TypeError ------
        raise TypeError(f"Invalid index type: {type(idx)!r}")

    def __iter__(self) -> Iterator[BlockHash]:
        # ------【前缀缓存】惰性迭代，逐个产出目标块粒度的哈希 ------
        for i in range(len(self)):
            yield self._get_value_at(i)

    def _get_value_at(self, idx: int) -> BlockHash:
        # The last hash_block_size hash within the target block already chains
        # over the whole prefix, so it is the target block's hash.
        # ------【前缀缓存】目标块内最后一个 hash 块哈希已覆盖整段前缀，直接作目标块哈希 ------
        return self.block_hashes[(idx + 1) * self.scale_factor - 1]


# ------【前缀缓存】块哈希列表联合类型：原始哈希列表或按块大小缩放的懒视图 ------
BlockHashList = list[BlockHash] | BlockHashListWithBlockSize


def resolve_block_hashes(
    block_hashes: BlockHashList,
    hash_block_size: int,
    block_size: int,
    *,
    supports_fine_grained_hash_lookup: bool = False,
    alignment_tokens: int | None = None,
) -> BlockHashList:
    """Resolve the block-hash view at ``block_size``.

    When ``block_size`` equals ``hash_block_size``, reuse the precomputed block
    hashes directly; otherwise view them at ``block_size`` granularity.
    Fine-grained lookup keeps the original hashes for partial cache hits.
    """
    # ------【前缀缓存】目标块大小等于哈希块大小时，直接复用预计算哈希 ------
    if block_size == hash_block_size:
        return block_hashes
    if isinstance(block_hashes, BlockHashListWithBlockSize):
        # Already a block-size view
        # ------【前缀缓存】已是块大小视图时校验放大因子一致，直接返回该视图 ------
        assert block_hashes.scale_factor == block_size // hash_block_size
        return block_hashes
    # Fine-grained partial hits keep the raw hashes. The caller passes
    # alignment_tokens = hash_block_size to enable them, else >= block_size.
    # ------【前缀缓存】细粒度查找且对齐粒度小于块大小时，保留原始哈希以支持部分命中 ------
    if (
        supports_fine_grained_hash_lookup
        and alignment_tokens is not None
        and alignment_tokens < block_size
        and block_size % alignment_tokens == 0
    ):
        return block_hashes
    # ------【前缀缓存】断言块大小是哈希块大小的整数倍后，包装成粗粒度视图返回 ------
    assert block_size % hash_block_size == 0
    return BlockHashListWithBlockSize(block_hashes, hash_block_size, block_size)
