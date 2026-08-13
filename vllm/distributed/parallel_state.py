# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright 2023 The vLLM team.
# Adapted from
# https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/parallel_state.py
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
"""vLLM distributed state.
It takes over the control of the distributed environment from PyTorch.
The typical workflow is:

- call `init_distributed_environment` to initialize the distributed environment.
- call `initialize_model_parallel` or `ensure_model_parallel_initialized` to
 initialize the model parallel groups.

- any code dealing with the distributed stuff

- call `destroy_model_parallel` to destroy the model parallel groups.
- call `destroy_distributed_environment` to destroy the distributed environment.

If you only need to use the distributed environment without model/pipeline
 parallelism, you can skip the model parallel initialization and destruction
 steps.
"""

import contextlib
import gc
import pickle
import weakref
from collections import namedtuple
from collections.abc import Callable
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from datetime import timedelta
from multiprocessing import shared_memory
from typing import TYPE_CHECKING, Any, Protocol
from unittest.mock import patch

import torch
import torch.distributed
import torch.distributed._functional_collectives as funcol
import torch.distributed._symmetric_memory
from torch.distributed import Backend, ProcessGroup, Store

import vllm.envs as envs
from vllm.distributed.device_communicators.base_device_communicator import (
    DeviceCommunicatorBase,
)
from vllm.distributed.utils import (
    StatelessProcessGroup,
    get_cached_tcp_store_client,
)
from vllm.logger import init_logger
from vllm.utils.import_utils import resolve_obj_by_qualname
from vllm.utils.network_utils import get_distributed_init_method
from vllm.utils.system_utils import suppress_stdout
from vllm.utils.torch_utils import (
    direct_register_custom_op,
)

if TYPE_CHECKING:
    from vllm.distributed.stateless_coordinator import StatelessGroupCoordinator


@dataclass
class GraphCaptureContext:
    stream: torch.cuda.Stream


TensorMetadata = namedtuple("TensorMetadata", ["device", "dtype", "size"])


class Handle(Protocol):
    """Minimal async work handle used by P2P send/recv methods."""

    def is_completed(self) -> bool: ...

    def wait(self) -> None: ...


def _split_tensor_dict(
    tensor_dict: dict[str, torch.Tensor | Any],
) -> tuple[list[tuple[str, Any]], list[torch.Tensor]]:
    """Split the tensor dictionary into two parts:
    1. A list of (key, value) pairs. If the value is a tensor, it is replaced
         by its metadata.
    2. A list of tensors.
    """
    metadata_list: list[tuple[str, Any]] = []
    tensor_list: list[torch.Tensor] = []
    # ------【异步 RPC】遍历字典，把张量拆到单独列表、其余存元数据，便于后续分别批量收发 ------
    for key, value in tensor_dict.items():
        if isinstance(value, torch.Tensor):
            # Note: we cannot use `value.device` here,
            # because it contains not only the device type but also the device
            # index (e.g. "cuda:0"). We only need the device type.
            # receiving side will set the device index.
            device = value.device.type
            metadata_list.append(
                (key, TensorMetadata(device, value.dtype, value.size()))
            )
            tensor_list.append(value)
        else:
            metadata_list.append((key, value))
    return metadata_list, tensor_list


_group_name_counter: dict[str, int] = {}


def _get_unique_name(name: str) -> str:
    """Get a unique name for the group.
    Example:
    _get_unique_name("tp") -> "tp:0"
    _get_unique_name("tp") -> "tp:1"
    """
    # ------【进程管理】首次出现时初始化计数器，用 name:index 保证同名组各自唯一 ------
    if name not in _group_name_counter:
        _group_name_counter[name] = 0
    # ------【进程管理】拼出递增编号的唯一名并累加计数器，避免同名组在全局表中冲突 ------
    newname = f"{name}:{_group_name_counter[name]}"
    _group_name_counter[name] += 1
    return newname


_groups: dict[str, Callable[[], "GroupCoordinator | None"]] = {}


def _register_group(group: "GroupCoordinator") -> None:
    # ------【进程管理】以弱引用登记组，避免循环引用导致组对象无法被 GC 回收 ------
    _groups[group.unique_name] = weakref.ref(group)


def _apply_to_device_comms(
    action: Callable[[DeviceCommunicatorBase], None],
) -> None:
    """Apply ``action`` to every group's device communicator.

    Walks the registered parallel groups and skips those without a device
    communicator (absent at ``world_size == 1``).
    """
    comms = []
    # ------【异步 RPC】先收集所有仍存活且带设备通信器的组，避免在遍历中引用已销毁组 ------
    for group_ref in _groups.values():
        group = group_ref()
        if group is None:
            continue
        dc = group.device_communicator
        if dc is None:
            continue
        comms.append(dc)

    # ------【异步 RPC】统一对每个设备通信器执行 action（如 dump/sleep/wake 等批量状态切换） ------
    for dc in comms:
        action(dc)


def all_reduce(tensor: torch.Tensor, group_name: str) -> torch.Tensor:
    # ------【DP/TP】Dynamo 只能传字符串组名，这里反查组对象并分发到 out-of-place all-reduce ------
    assert group_name in _groups, f"Group {group_name} is not found."
    group = _groups[group_name]()
    if group is None:
        raise ValueError(f"Group {group_name} is destroyed.")
    return group._all_reduce_out_place(tensor)


def all_reduce_fake(tensor: torch.Tensor, group_name: str) -> torch.Tensor:
    # ------【CUDA Graph】自定义算子的 fake 实现：只返回同形状空张量，供元编程/形状推导使用 ------
    return torch.empty_like(tensor)


def reduce_scatter(
    tensor: torch.Tensor, dim: int, world_size: int, group_name: str
) -> torch.Tensor:
    # ------【DP/TP】按组名反查组并分发到 out-of-place reduce-scatter（沿维度归约切分） ------
    assert group_name in _groups, f"Group {group_name} is not found."
    group = _groups[group_name]()
    if group is None:
        raise ValueError(f"Group {group_name} is destroyed.")
    return group._reduce_scatter_out_place(tensor, dim)


def reduce_scatter_fake(
    tensor: torch.Tensor, dim: int, world_size: int, group_name: str
) -> torch.Tensor:
    # ------【CUDA Graph】fake 实现：按 world_size 收缩指定维度，模拟 reduce-scatter 的输出形状 ------
    new_shape = list(tensor.shape)
    new_shape[dim] = tensor.shape[dim] // world_size
    return torch.empty(new_shape, dtype=tensor.dtype, device=tensor.device)


def all_gather(
    tensor: torch.Tensor, dim: int, world_size: int, group_name: str
) -> torch.Tensor:
    # ------【DP/TP】按组名反查组并分发到 out-of-place all-gather（沿维度拼接） ------
    assert group_name in _groups, f"Group {group_name} is not found."
    group = _groups[group_name]()
    if group is None:
        raise ValueError(f"Group {group_name} is destroyed.")
    return group._all_gather_out_place(tensor, dim)


def all_gather_fake(
    tensor: torch.Tensor, dim: int, world_size: int, group_name: str
) -> torch.Tensor:
    # ------【CUDA Graph】fake 实现：按 world_size 扩增指定维度，模拟 all-gather 的输出形状 ------
    new_shape = list(tensor.shape)
    new_shape[dim] = tensor.shape[dim] * world_size
    return torch.empty(new_shape, dtype=tensor.dtype, device=tensor.device)


def patched_fused_scaled_matmul_reduce_scatter_fake(
    A: torch.Tensor,
    B: torch.Tensor,
    A_scale: torch.Tensor,
    B_scale: torch.Tensor,
    reduce_op: str,
    orig_scatter_dim: int,
    scatter_dim_after_maybe_reshape: int,
    group_name: str,
    output_shape: list[int],
    bias: torch.Tensor | None = None,
    result_scale: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
    use_fast_accum: bool = False,
) -> torch.Tensor:
    # Copied from
    # https://github.com/pytorch/pytorch/blob/50c338c2da905062449e4d9ac807832d1b5cd90e/torch/distributed/_symmetric_memory/__init__.py#L1189
    # ------【DP/TP】校验 A_scale 是行级还是标量缩放，行级需压平前导维以适配 scaled_mm ------
    if A_scale.numel() > 1:
        if A_scale.shape[:-1] != A.shape[:-1]:
            raise ValueError(
                "For row-wise scaling, the leading dims of A_scale "
                "must match the leading dims of A "
                f"(A shape: {A.shape}, A_scale shape: {A_scale.shape})"
            )
        A_scale = A_scale.flatten(0, -2).contiguous()
    elif A_scale.numel() != 1:
        raise ValueError(
            "Invalid A_scale shape "
            f"(A shape: {A.shape}, A_scale shape: {A_scale.shape})"
        )

    # ------【DP/TP】用 FP8 scaled_mm 做融合矩阵乘，得到尚未归约的局部输出 ------
    C = torch._scaled_mm(
        A.flatten(0, -2).contiguous(),
        B,
        A_scale,
        B_scale,
        bias,
        result_scale,
        out_dtype,
        use_fast_accum,
    )
    C = C.view(*output_shape[:-1], B.shape[1])
    # ------【DP/TP】还原形状后做 reduce-scatter，把按 TP 切分的输出归约并分散到各 rank ------
    res = funcol.reduce_scatter_tensor(
        C,
        reduce_op,
        orig_scatter_dim,  # need original scatter dim for 3D+ output tensor here
        group_name,
    )
    # ------【DP/TP】等待 functional collective 完成，返回最终归约后的结果张量 ------
    res = funcol.wait_tensor(res)
    return res


def _platform_device_type() -> str:
    """Return the device-type string (e.g. ``"cuda"``, ``"xpu"``, ``"cpu"``)
    for the current platform, in the form expected by
    ``torch.distributed.init_process_group(backend=...)``.
    """
    from vllm.platforms import current_platform

    # ------【NCCL 通信】把平台映射成 torch.distributed 认识的设备类型字符串，供 backend 拼接使用 ------
    if current_platform.is_cuda_alike():
        return "cuda"
    elif current_platform.is_xpu():
        return "xpu"
    elif current_platform.is_out_of_tree():
        return current_platform.device_name
    else:
        return "cpu"


def _device_backend_str(torch_distributed_backend: str | Backend) -> str:
    """Normalize ``torch_distributed_backend`` to the ``"<device>:<backend>"``
    format required by ``split_group``'s ``backend`` argument.

    Accepts either a bare backend name (e.g. ``"nccl"``) or an already-prefixed
    string (e.g. ``"cuda:nccl"``).
    """
    backend_str = str(torch_distributed_backend)
    # ------【NCCL 通信】已带 "device:backend" 前缀则直接返回，否则补设备前缀成 "cuda:nccl" ------
    if ":" in backend_str:
        return backend_str
    return f"{_platform_device_type()}:{backend_str}"


def _create_subgroups_split_group(
    group_ranks: list[list[int]],
    group_name: str,
    torch_distributed_backend: str | Backend,
) -> tuple[ProcessGroup, ProcessGroup]:
    """Create the device + CPU subgroups for ``GroupCoordinator`` via
    ``torch.distributed.split_group``.

    ``split_group`` is collective on the parent group, so every parent rank
    must enter with the same ``split_ranks`` definition. Each rank receives
    the subgroup it belongs to.
    """
    from vllm.distributed.utils import (
        get_cpu_distributed_timeout_or_none,
        get_distributed_timeout_or_none,
    )

    device_backend_str = _device_backend_str(torch_distributed_backend)
    # ------【NCCL 通信】用 split_group 切出设备子组（如 NCCL），各父 rank 必须用相同 split_ranks ------
    self_device_group = torch.distributed.split_group(
        split_ranks=group_ranks,
        group_desc=f"{group_name}:device",
        backend=device_backend_str,
        timeout=get_distributed_timeout_or_none(),
    )
    # CPU subgroup: split_group requires the requested backend filter to
    # include the parent's default device type (= the device the parent PG
    # was bound to via ``device_id``), so a cpu-only filter is rejected.
    # Include the device backend in the filter; only the gloo backend is
    # actually used for CPU collectives on this group.
    # ------【NCCL 通信】再切一个 gloo CPU 子组，用于 CPU 上的协调通信（filter 需含设备后端） ------
    self_cpu_group = torch.distributed.split_group(
        split_ranks=group_ranks,
        group_desc=f"{group_name}:cpu",
        backend=f"cpu:gloo,{device_backend_str}",
        timeout=get_cpu_distributed_timeout_or_none(),
    )
    return self_device_group, self_cpu_group


def patched_fused_scaled_matmul_reduce_scatter(
    A: torch.Tensor,
    B: torch.Tensor,
    A_scale: torch.Tensor,
    B_scale: torch.Tensor,
    reduce_op: str,
    orig_scatter_dim: int,
    scatter_dim_after_maybe_reshape: int,
    group_name: str,
    output_shape: list[int],
    bias: torch.Tensor | None = None,
    result_scale: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
    use_fast_accum: bool = False,
) -> torch.Tensor:
    # ------【DP/TP】调用 symmetric-memory 融合算子：一次完成 FP8 matmul + reduce-scatter ------
    return torch.ops.symm_mem.fused_scaled_matmul_reduce_scatter(
        A,
        B,
        A_scale,
        B_scale,
        reduce_op,
        orig_scatter_dim,
        scatter_dim_after_maybe_reshape,
        group_name,
        output_shape,
        bias,
        result_scale,
        out_dtype,
        use_fast_accum,
    )


# ------【CUDA Graph】把集合通信注册为自定义算子，使 Dynamo/CUDA Graph 能将其捕获为图节点 ------
direct_register_custom_op(
    op_name="all_reduce",
    op_func=all_reduce,
    fake_impl=all_reduce_fake,
)

direct_register_custom_op(
    op_name="reduce_scatter",
    op_func=reduce_scatter,
    fake_impl=reduce_scatter_fake,
)

direct_register_custom_op(
    op_name="all_gather",
    op_func=all_gather,
    fake_impl=all_gather_fake,
)

# TODO: Remove this once the pytorch fix
# (https://github.com/pytorch/pytorch/pull/165086) gets released,
# in either 2.9.1 or 2.10
direct_register_custom_op(
    op_name="patched_fused_scaled_matmul_reduce_scatter",
    op_func=patched_fused_scaled_matmul_reduce_scatter,
    fake_impl=patched_fused_scaled_matmul_reduce_scatter_fake,
)


class GroupCoordinator:
    """
    PyTorch ProcessGroup wrapper for a group of processes.
    PyTorch ProcessGroup is bound to one specific communication backend,
        e.g. NCCL, Gloo, MPI, etc.
    GroupCoordinator takes charge of all the communication operations among
        the processes in the group. It manages both CPU and device
        communication.
    """

    # available attributes:
    rank: int  # global rank
    ranks: list[int]  # global ranks in the group
    world_size: int  # size of the group
    # difference between `local_rank` and `rank_in_group`:
    # if we have a group of size 4 across two nodes:
    # Process | Node | Rank | Local Rank | Rank in Group
    #   0     |   0  |  0   |     0      |       0
    #   1     |   0  |  1   |     1      |       1
    #   2     |   1  |  2   |     0      |       2
    #   3     |   1  |  3   |     1      |       3
    local_rank: int  # local rank used to assign devices
    rank_in_group: int  # rank inside the group
    cpu_group: ProcessGroup  # group for CPU communication
    device_group: ProcessGroup  # group for device communication
    # device communicator (if use_device_communicator=True)
    device_communicator: DeviceCommunicatorBase | None
    mq_broadcaster: Any | None  # shared memory broadcaster

    def __init__(
        self,
        group_ranks: list[list[int]],
        local_rank: int,
        torch_distributed_backend: str | Backend,
        use_device_communicator: bool,  # whether to use device communicator
        use_message_queue_broadcaster: bool = False,
        group_name: str | None = None,
        use_all2all: bool = False,
    ):
        # ------【进程管理】生成唯一组名并注册到全局表，便于后续按名查找该组 ------
        group_name = group_name or "anonymous"
        self.unique_name = _get_unique_name(group_name)
        _register_group(self)

        self.rank = torch.distributed.get_rank()
        self.local_rank = local_rank
        self.device_index: int
        assert local_rank >= 0, (
            "local_rank must be provided when creating the world group"
        )
        # ------【进程管理】记录全局 rank/local_rank，并把 device_index 绑定到 local_rank 用于选卡 ------
        self.device_index = local_rank

        self_device_group = None
        self_cpu_group = None

        # ------【NCCL 通信】按环境变量选建组路径：split_group 新路径或 legacy new_group 路径 ------
        # VLLM_DISTRIBUTED_USE_SPLIT_GROUP gates the new ``split_group``
        # codepath. Default (False) preserves the legacy ``new_group`` path.
        if envs.VLLM_DISTRIBUTED_USE_SPLIT_GROUP:
            self_device_group, self_cpu_group = _create_subgroups_split_group(
                group_ranks, group_name, torch_distributed_backend
            )
            # ------【进程管理】在所有子组里定位包含当前 rank 的那个，记下 ranks/world_size/rank_in_group ------
            for ranks in group_ranks:
                if self.rank in ranks:
                    self.ranks = ranks
                    self.world_size = len(ranks)
                    self.rank_in_group = ranks.index(self.rank)
                    break
        else:
            from vllm.distributed.utils import (
                get_cpu_distributed_timeout_or_none,
                get_distributed_timeout_or_none,
            )

            timeout = get_cpu_distributed_timeout_or_none()
            device_timeout = get_distributed_timeout_or_none()

            # ------【NCCL 通信】对每个子组建 NCCL 设备组 + gloo CPU 组，并找到包含当前 rank 的组 ------
            for ranks in group_ranks:
                device_group = torch.distributed.new_group(
                    ranks,
                    backend=torch_distributed_backend,
                    timeout=device_timeout,
                )
                # a group with `gloo` backend, to allow direct coordination between
                # processes through the CPU.
                with suppress_stdout():
                    cpu_group = torch.distributed.new_group(
                        ranks, backend="gloo", timeout=timeout
                    )
                if self.rank in ranks:
                    self.ranks = ranks
                    self.world_size = len(ranks)
                    self.rank_in_group = ranks.index(self.rank)
                    self_device_group = device_group
                    self_cpu_group = cpu_group

        assert self_cpu_group is not None
        assert self_device_group is not None

        # ------【NCCL 通信】保存建组参数并正式绑定 cpu/device 两个进程组供后续通信使用 ------
        self.group_ranks = group_ranks
        self.torch_distributed_backend = torch_distributed_backend

        self.cpu_group = self_cpu_group
        self.device_group = self_device_group

        from vllm.platforms import current_platform

        # ------【NUMA 亲和】按平台把逻辑设备号映射成真实可见设备，构造本组绑定的 torch.device ------
        if current_platform.is_cuda_alike():
            visible_device_index = (
                current_platform.logical_device_id_to_visible_device_id(
                    self.device_index
                )
            )
            self.device = torch.device(f"cuda:{visible_device_index}")
        elif current_platform.is_xpu():
            self.device = torch.device(f"xpu:{self.device_index}")
        elif current_platform.is_out_of_tree():
            self.device = torch.device(
                f"{current_platform.device_name}:{self.device_index}"
            )
        else:
            self.device = torch.device("cpu")

        self.use_device_communicator = use_device_communicator
        self.device_communicator = None
        # ------【NCCL 通信】多卡时按平台解析并创建自定义设备通信器（如 CudaCommunicator）加速集合通信 ------
        if use_device_communicator and self.world_size > 1:
            device_comm_cls = resolve_obj_by_qualname(
                current_platform.get_device_communicator_cls()
            )
            self.device_communicator = device_comm_cls(
                cpu_group=self.cpu_group,
                device=self.device,
                device_group=self.device_group,
                unique_name=self.unique_name,
                use_all2all=use_all2all,
            )

        from vllm.distributed.device_communicators.shm_broadcast import MessageQueue

        self.mq_broadcaster: MessageQueue | None = None
        # ------【异步 RPC】多卡时创建共享内存消息队列，用于对象/元数据的低延迟非阻塞广播 ------
        if use_message_queue_broadcaster and self.world_size > 1:
            self.mq_broadcaster = MessageQueue.create_from_process_group(
                self.cpu_group, 1 << 22, 6
            )

        # TODO(#35915): Remove is_tpu() check once tpu_inference
        # overrides use_custom_op_collectives() to return True.
        # ------【CUDA Graph】决定集合通信走自定义算子路径（可被图捕获）还是直接方法调用 ------
        self.use_custom_op_call = (
            current_platform.is_tpu() or current_platform.use_custom_op_collectives()
        )

        # ------【异步 RPC】CPU 平台且通信器支持 tensor_dict 时，走自定义同步收发路径 ------
        self.use_cpu_custom_send_recv = (
            current_platform.is_cpu()
            and self.device_communicator
            and getattr(self.device_communicator, "supports_tensor_dict", False)
        )

    def make_sibling_device_group(self, group_desc: str | None = None) -> ProcessGroup:
        """Create a new device-side ProcessGroup with the same per-rank membership
        as this coordinator's `device_group`, but backed by a distinct communicator.
        This is a collective call: every world rank must invoke it. Used where we
        want to issue ops that can run concurrently with ops on `device_group`.
        """
        from vllm.distributed.utils import get_distributed_timeout_or_none

        device_timeout = get_distributed_timeout_or_none()
        sibling: ProcessGroup | None = None
        # ------【NCCL 通信】用相同成员建独立 communicator 的兄弟组，便于并行发起互不阻塞的通信 ------
        for ranks in self.group_ranks:
            pg = torch.distributed.new_group(
                ranks,
                backend=self.torch_distributed_backend,
                group_desc=group_desc,
                timeout=device_timeout,
            )
            if self.rank in ranks:
                sibling = pg
        assert sibling is not None
        return sibling

    def create_mq_broadcaster(
        self, writer_rank=0, external_writer_handle=None, blocking=True
    ):
        from vllm.distributed.device_communicators.shm_broadcast import MessageQueue

        # ------【异步 RPC】基于 CPU 组创建共享内存广播队列，支持指定 writer_rank/外部句柄/阻塞模式 ------
        return MessageQueue.create_from_process_group(
            self.cpu_group,
            1 << 22,
            6,
            writer_rank=writer_rank,
            external_writer_handle=external_writer_handle,
            blocking=blocking,
        )

    def create_single_reader_mq_broadcasters(
        self, reader_rank_in_group=0, blocking=False
    ):
        from vllm.distributed.device_communicators.shm_broadcast import MessageQueue

        # ------【异步 RPC】创建单读者消息队列（仅指定 reader_rank 消费），减少多读者竞争开销 ------
        return MessageQueue.create_from_process_group_single_reader(
            self.cpu_group,
            1 << 22,
            6,
            reader_rank=self.ranks[reader_rank_in_group],
            blocking=blocking,
        )

    @property
    def first_rank(self):
        """Return the global rank of the first process in the group"""
        return self.ranks[0]

    @property
    def last_rank(self):
        """Return the global rank of the last process in the group"""
        return self.ranks[-1]

    @property
    def is_first_rank(self):
        """Return whether the caller is the first process in the group"""
        return self.rank == self.first_rank

    @property
    def is_last_rank(self):
        """Return whether the caller is the last process in the group"""
        return self.rank == self.last_rank

    @property
    def next_rank(self):
        """Return the global rank of the process that follows the caller"""
        # ------【PP】环状相邻 rank 计算：next 取 (rank+1)%world_size，供流水线 P2P 通信 ------
        rank_in_group = self.rank_in_group
        world_size = self.world_size
        return self.ranks[(rank_in_group + 1) % world_size]

    @property
    def prev_rank(self):
        """Return the global rank of the process that precedes the caller"""
        # ------【PP】prev 取 (rank-1)%world_size，与 next 配对构成双向环状流水线 ------
        rank_in_group = self.rank_in_group
        world_size = self.world_size
        return self.ranks[(rank_in_group - 1) % world_size]

    @contextmanager
    def graph_capture(self, graph_capture_context: GraphCaptureContext | None = None):
        # ------【CUDA Graph】未显式传上下文时新建专用流，否则复用传入流的 stream 以隔离捕获 ------
        if graph_capture_context is None:
            stream = torch.cuda.Stream()
            graph_capture_context = GraphCaptureContext(stream)
        else:
            stream = graph_capture_context.stream

        # only cuda uses this function,
        # so we don't abstract it into the base class
        # ------【CUDA Graph】初始化空上下文占位，后续按设备通信器类型替换为真正的捕获上下文 ------
        maybe_ca_context = nullcontext()
        maybe_aiter_context = nullcontext()
        from vllm.distributed.device_communicators.cuda_communicator import (
            CudaCommunicator,
        )
        from vllm.distributed.device_communicators.xpu_communicator import (
            XpuCommunicator,
        )

        # ------【CUDA Graph】若通信器带 custom-allreduce comm，进入其捕获上下文让集合通信可被图化 ------
        if self.device_communicator is not None:
            assert isinstance(
                self.device_communicator,
                (CudaCommunicator, XpuCommunicator),
            )
            ca_comm = self.device_communicator.ca_comm
            if ca_comm is not None:
                maybe_ca_context = ca_comm.capture()  # type: ignore

            from vllm._aiter_ops import rocm_aiter_ops

            # ------【CUDA Graph】ROCm 上启用 aiter 优化时，进入 aiter all-reduce 的图捕获上下文 ------
            if rocm_aiter_ops.is_enabled():
                aiter_ar = rocm_aiter_ops.get_aiter_allreduce()
                if aiter_ar is not None:
                    maybe_aiter_context = aiter_ar.capture()  # type: ignore

        # ------【CUDA Graph】捕获流先等当前流跑完，避免把后台初始化算子一并捕获进图 ------
        # ensure all initialization operations complete before attempting to
        # capture the graph on another stream
        curr_stream = torch.cuda.current_stream()
        if curr_stream != stream:
            stream.wait_stream(curr_stream)

        # ------【CUDA Graph】切换到捕获流并叠加各捕获上下文，yield 让调用方在此执行待捕获前向 ------
        with torch.cuda.stream(stream), maybe_ca_context, maybe_aiter_context:
            yield graph_capture_context

    def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
        """
        User-facing all-reduce function before we actually call the
        all-reduce operation.

        We need this because Dynamo does not support passing an arbitrary
        object (`self` in this case) to a custom op. We need to pass the
         group name as a string, and then look up the group coordinator from
         the group name, dispatch the all-reduce operation to the group
         coordinator.

        In addition, PyTorch custom ops do not support mutation or returning
        a new tensor in the same op. So we always make the all-reduce operation
        out-of-place.
        """
        # Bypass the function if we are using only 1 GPU.
        if self.world_size == 1:
            return input_

        # ------【CUDA Graph】走自定义算子路径（可被 Dynamo/图捕获），否则退回直接调用设备通信器 ------
        if self.use_custom_op_call:
            return torch.ops.vllm.all_reduce(input_, group_name=self.unique_name)
        else:
            return self._all_reduce_out_place(input_)

    def _all_reduce_out_place(self, input_: torch.Tensor) -> torch.Tensor:
        if self.device_communicator is None:
            raise ValueError("No device communicator found")
        # ------【NCCL 通信】委托设备通信器执行真正的 all-reduce（out-of-place 满足自定义算子约束） ------
        return self.device_communicator.all_reduce(input_)

    def all_gather(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        world_size = self.world_size
        # Bypass the function if we are using only 1 GPU.
        if world_size == 1:
            return input_
        assert -input_.dim() <= dim < input_.dim(), (
            f"Invalid dim ({dim}) for input tensor with shape {input_.size()}"
        )

        # ------【CUDA Graph】自定义算子路径可被图捕获；否则直接调用设备通信器的 all_gather ------
        if self.use_custom_op_call:
            return torch.ops.vllm.all_gather(
                input_, dim, world_size, group_name=self.unique_name
            )
        else:
            return self._all_gather_out_place(input_, dim)

    def _all_gather_out_place(self, input_: torch.Tensor, dim: int) -> torch.Tensor:
        if self.device_communicator is None:
            raise ValueError("No device communicator found")
        # ------【NCCL 通信】委托设备通信器沿指定维度做 all-gather 拼接 ------
        return self.device_communicator.all_gather(input_, dim)

    def all_gatherv(
        self,
        input_: torch.Tensor | list[torch.Tensor],
        dim: int = 0,
        sizes: list[int] | None = None,
    ):
        if self.device_communicator is None:
            raise ValueError("No device communicator found")
        # ------【NCCL 通信】变长 all-gatherv：各 rank 贡献不等长数据按 sizes 拼接（MoE/序列并行用） ------
        return self.device_communicator.all_gatherv(input_, dim, sizes)

    def reduce_scatter(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        world_size = self.world_size
        # Bypass the function if we are using only 1 GPU.
        if world_size == 1:
            return input_
        assert -input_.dim() <= dim < input_.dim(), (
            f"Invalid dim ({dim}) for input tensor with shape {input_.size()}"
        )

        # ------【CUDA Graph】自定义算子路径可被图捕获；否则直接调用设备通信器做 reduce-scatter ------
        if self.use_custom_op_call:
            return torch.ops.vllm.reduce_scatter(
                input_, dim, world_size, group_name=self.unique_name
            )
        else:
            return self._reduce_scatter_out_place(input_, dim)

    def reduce_scatterv(
        self, input_: torch.Tensor, dim: int = -1, sizes: list[int] | None = None
    ) -> torch.Tensor:
        if self.device_communicator is None:
            raise ValueError("No device communicator found")
        # ------【NCCL 通信】变长 reduce-scatterv：各 rank 贡献不等长数据按 sizes 切分归约 ------
        return self.device_communicator.reduce_scatterv(input_, dim, sizes)

    def _reduce_scatter_out_place(self, input_: torch.Tensor, dim: int) -> torch.Tensor:
        if self.device_communicator is None:
            raise ValueError("No device communicator found")
        # ------【NCCL 通信】委托设备通信器沿指定维度做 reduce-scatter 归约切分 ------
        return self.device_communicator.reduce_scatter(input_, dim)

    def gather(
        self, input_: torch.Tensor, dst: int = 0, dim: int = -1
    ) -> torch.Tensor | None:
        """
        NOTE: We assume that the input tensor is on the same device across
        all the ranks.
        NOTE: `dst` is the local rank of the destination rank.
        """
        world_size = self.world_size
        # Bypass the function if we are using only 1 GPU.
        if world_size == 1:
            return input_
        if self.device_communicator is None:
            raise ValueError("No device communicator found")
        # ------【NCCL 通信】gather：把所有 rank 数据收集到目标 rank（dst），其余 rank 返回 None ------
        return self.device_communicator.gather(input_, dst, dim)

    def broadcast(self, input_: torch.Tensor, src: int = 0):
        """Broadcast the input tensor.
        NOTE: `src` is the local rank of the source rank.
        """
        assert src < self.world_size, f"Invalid src rank ({src})"

        # Bypass the function if we are using only 1 GPU.
        if self.world_size == 1:
            return input_
        # Broadcast.
        # ------【NCCL 通信】在设备组内从 src rank 广播张量，同步覆盖所有 rank 的 input_ ------
        torch.distributed.broadcast(
            input_, src=self.ranks[src], group=self.device_group
        )
        return input_

    def broadcast_object(self, obj: Any | None = None, src: int = 0):
        """Broadcast the input object.
        NOTE: `src` is the local rank of the source rank.
        """
        assert src < self.world_size, f"Invalid src rank ({src})"

        # Bypass the function if we are using only 1 GPU.
        if self.world_size == 1:
            return obj
        # ------【异步 RPC】优先走共享内存消息队列广播（低延迟、非阻塞），仅支持 src=0 ------
        if self.mq_broadcaster is not None:
            assert src == 0, "Message queue broadcaster only supports src=0"
            return self.mq_broadcaster.broadcast_object(obj)
        # ------【异步 RPC】无消息队列时退回 gloo CPU 组的 broadcast_object_list 做对象广播 ------
        if self.rank_in_group == src:
            torch.distributed.broadcast_object_list(
                [obj], src=self.ranks[src], group=self.cpu_group
            )
            return obj
        else:
            recv = [None]
            torch.distributed.broadcast_object_list(
                recv, src=self.ranks[src], group=self.cpu_group
            )
            return recv[0]

    def broadcast_object_list(
        self, obj_list: list[Any], src: int = 0, group: ProcessGroup | None = None
    ):
        """Broadcast the input object list.
        NOTE: `src` is the local rank of the source rank.
        """
        assert src < self.world_size, f"Invalid src rank ({src})"

        # Bypass the function if we are using only 1 GPU.
        if self.world_size == 1:
            return obj_list
        # Broadcast.
        # ------【NCCL 通信】在设备组内广播对象列表（内部先序列化，走 gloo/NCCL 底层实现） ------
        torch.distributed.broadcast_object_list(
            obj_list, src=self.ranks[src], group=self.device_group
        )
        return obj_list

    def send_object(self, obj: Any, dst: int) -> None:
        """Send the input object list to the destination rank."""
        """NOTE: `dst` is the local rank of the destination rank."""

        assert dst < self.world_size, f"Invalid dst rank ({dst})"

        assert dst != self.rank_in_group, (
            "Invalid destination rank. Destination rank is the same "
            "as the current rank."
        )

        # Serialize object to tensor and get the size as well
        # ------【异步 RPC】把对象 pickle 序列化成 uint8 张量，先发长度让接收方据此分配缓冲区 ------
        object_tensor = torch.frombuffer(pickle.dumps(obj), dtype=torch.uint8)

        size_tensor = torch.tensor(
            [object_tensor.numel()], dtype=torch.long, device="cpu"
        )

        # Send object size

        torch.distributed.send(size_tensor, dst=self.ranks[dst], group=self.cpu_group)

        # Send object
        # ------【异步 RPC】再发送序列化后的对象数据本身（CPU 组上的阻塞 send） ------
        torch.distributed.send(object_tensor, dst=self.ranks[dst], group=self.cpu_group)

        return None

    def recv_object(self, src: int) -> Any:
        """Receive the input object list from the source rank."""
        """NOTE: `src` is the local rank of the source rank."""

        assert src < self.world_size, f"Invalid src rank ({src})"

        assert src != self.rank_in_group, (
            "Invalid source rank. Source rank is the same as the current rank."
        )

        # ------【异步 RPC】先接收长度，据此动态分配接收缓冲区 ------
        size_tensor = torch.empty(1, dtype=torch.long, device="cpu")

        # Receive object size
        rank_size = torch.distributed.recv(
            size_tensor, src=self.ranks[src], group=self.cpu_group
        )

        # ------【异步 RPC】按长度分配 uint8 缓冲区，接收序列化后的对象数据 ------
        # Tensor to receive serialized objects into.
        object_tensor = torch.empty(  # type: ignore[call-overload]
            size_tensor.item(),  # type: ignore[arg-type]
            dtype=torch.uint8,
            device="cpu",
        )

        # ------【异步 RPC】接收对象数据并校验长度与数据的发送源一致，最后反序列化还原对象 ------
        rank_object = torch.distributed.recv(
            object_tensor, src=self.ranks[src], group=self.cpu_group
        )

        assert rank_object == rank_size, (
            "Received object sender rank does not match the size sender rank."
        )

        obj = pickle.loads(object_tensor.numpy().tobytes())

        return obj

    def broadcast_tensor_dict(
        self,
        tensor_dict: dict[str, torch.Tensor | Any] | None = None,
        src: int = 0,
        group: ProcessGroup | None = None,
        metadata_group: ProcessGroup | None = None,
    ) -> dict[str, torch.Tensor | Any] | None:
        """Broadcast the input tensor dictionary.
        NOTE: `src` is the local rank of the source rank.
        """
        # Bypass the function if we are using only 1 GPU.
        if not torch.distributed.is_initialized() or self.world_size == 1:
            return tensor_dict

        # ------【异步 RPC】张量走设备组、元数据走 CPU 组，分两条通道广播以降低序列化开销 ------
        group = self.device_group
        metadata_group = self.cpu_group
        assert src < self.world_size, f"Invalid src rank ({src})"

        rank_in_group = self.rank_in_group
        # ------【异步 RPC】src rank 负责拆分字典、广播元数据，再逐个异步广播张量 ------
        if rank_in_group == src:
            metadata_list: list[tuple[Any, Any]] = []
            assert isinstance(tensor_dict, dict), (
                f"Expecting a dictionary, got {type(tensor_dict)}"
            )
            metadata_list, tensor_list = _split_tensor_dict(tensor_dict)
            # `metadata_list` lives in CPU memory.
            # `broadcast_object_list` has serialization & deserialization,
            # all happening on CPU. Therefore, we can use the CPU group.
            self.broadcast_object(metadata_list, src=src)
            # ------【异步 RPC】先广播元数据让接收方预知张量形状，再对每个张量发起异步广播 ------
            async_handles = []
            for tensor in tensor_list:
                if tensor.numel() == 0:
                    # Skip broadcasting empty tensors.
                    continue
                if tensor.is_cpu:
                    # use metadata_group for CPU tensors
                    handle = torch.distributed.broadcast(
                        tensor, src=self.ranks[src], group=metadata_group, async_op=True
                    )
                else:
                    # use group for GPU tensors
                    handle = torch.distributed.broadcast(
                        tensor, src=self.ranks[src], group=group, async_op=True
                    )
                async_handles.append(handle)
            # ------【异步 RPC】等待所有异步广播完成，确保源端张量已全部发出 ------
            for async_handle in async_handles:
                async_handle.wait()

        else:
            # ------【异步 RPC】非源 rank 先收元数据获知每个张量的 device/dtype/size ------
            metadata_list = self.broadcast_object(None, src=src)
            tensor_dict = {}
            async_handles = []
            # ------【异步 RPC】按元数据分配空张量，区分 CPU/GPU 用对应组异步广播接收 ------
            for key, value in metadata_list:
                if isinstance(value, TensorMetadata):
                    tensor = torch.empty(
                        value.size, dtype=value.dtype, device=value.device
                    )
                    if tensor.numel() == 0:
                        # Skip broadcasting empty tensors.
                        tensor_dict[key] = tensor
                        continue
                    if tensor.is_cpu:
                        # use metadata_group for CPU tensors
                        handle = torch.distributed.broadcast(
                            tensor,
                            src=self.ranks[src],
                            group=metadata_group,
                            async_op=True,
                        )
                    else:
                        # use group for GPU tensors
                        handle = torch.distributed.broadcast(
                            tensor, src=self.ranks[src], group=group, async_op=True
                        )
                    async_handles.append(handle)
                    tensor_dict[key] = tensor
                else:
                    tensor_dict[key] = value
            # ------【异步 RPC】等待接收端所有异步广播完成，张量数据全部就位后再返回字典 ------
            for async_handle in async_handles:
                async_handle.wait()
        return tensor_dict

    def _should_use_all_gather(
        self,
        key: str,
        numel: int,
        all_gather_group: "GroupCoordinator | None",
        all_gather_tensors: dict[str, bool] | None,
    ) -> bool:
        if all_gather_group is None:
            return False
        # ------【TP】numel 能被 world_size 整除才可用 all-gather 优化，否则各 rank 切片不均匀 ------
        use_all_gather = numel % all_gather_group.world_size == 0
        if all_gather_tensors is not None:
            use_all_gather = all_gather_tensors.get(key, use_all_gather)
        return use_all_gather

    def send_tensor_dict(
        self,
        tensor_dict: dict[str, torch.Tensor | Any],
        dst: int | None = None,
        all_gather_group: "GroupCoordinator | None" = None,
        all_gather_tensors: dict[str, bool] | None = None,
    ) -> dict[str, torch.Tensor | Any] | None:
        """Send the input tensor dictionary.
        NOTE: `dst` is the local rank of the source rank.

        all_gather_group: The group for the all-gather operation. If provided,
            an optimization is enabled where each rank in the group sends a
            slice of a tensor and the receiver reconstructs it using an
            all-gather, which can improve performance. This is typically the
            tensor-parallel group.
        all_gather_tensors: A dictionary to specify which tensors should use
            the all-gather optimization, which is only effective when
            `all_gather_group` is provided. By default, this optimization is
            on for any tensor whose size is divisible by the
            `all_gather_group`'s world size. However, it should be disabled
            for tensors that are not fully replicated across the group (e.g.,
            the residual tensor when sequence parallelism is enabled). This
            dictionary allows overriding the default behavior on a per-tensor
            basis.
        """
        # Bypass the function if we are using only 1 GPU.
        if not torch.distributed.is_initialized() or self.world_size == 1:
            return tensor_dict
        # ------【异步 RPC】发起非阻塞发送后统一等待所有 handle 完成，实现同步语义 ------
        handles = self.isend_tensor_dict(
            tensor_dict,
            dst=dst,
            all_gather_group=all_gather_group,
            all_gather_tensors=all_gather_tensors,
        )
        for handle in handles:
            handle.wait()
        return None

    def isend_tensor_dict(
        self,
        tensor_dict: dict[str, torch.Tensor | Any],
        dst: int | None = None,
        all_gather_group: "GroupCoordinator | None" = None,
        all_gather_tensors: dict[str, bool] | None = None,
    ) -> list[Handle]:
        if self.world_size <= 1:
            return []

        # ------【PP】默认目标为环上下一 rank，形成流水线相邻层间的数据传递 ------
        if dst is None:
            dst = (self.rank_in_group + 1) % self.world_size
        assert dst < self.world_size, f"Invalid dst rank ({dst})"

        # ------【异步 RPC】CPU 平台走自定义同步 send_tensor_dict 路径，无异步 handle 直接返回 ------
        if self.use_cpu_custom_send_recv:
            if self.device_communicator is None:
                raise ValueError("No device communicator found")
            # custom device communicator path is synchronous
            self.device_communicator.send_tensor_dict(  # type: ignore
                tensor_dict, dst
            )
            return []

        # ------【TP】计算 all-gather 优化的分片大小与 rank，用于把张量切成 1/world_size 再发送 ------
        all_gather_size = 1 if all_gather_group is None else all_gather_group.world_size
        all_gather_rank = (
            0 if all_gather_group is None else all_gather_group.rank_in_group
        )

        group = self.device_group
        metadata_group = self.cpu_group

        # ------【异步 RPC】先发元数据，再按 key 顺序逐个异步发送张量数据 ------
        metadata_list, tensor_list = _split_tensor_dict(tensor_dict)
        self.send_object(metadata_list, dst=dst)

        tensor_keys = [k for k, v in tensor_dict.items() if isinstance(v, torch.Tensor)]
        assert len(tensor_keys) == len(tensor_list)

        handles: list[Handle] = []
        for key, tensor in zip(tensor_keys, tensor_list):
            if tensor.numel() == 0:
                continue

            # ------【TP】启用 all-gather 优化时只发本 rank 负责的 1/world_size 切片，接收端再拼接 ------
            if self._should_use_all_gather(
                key, tensor.numel(), all_gather_group, all_gather_tensors
            ):
                tensor = tensor.reshape(all_gather_size, -1)[all_gather_rank]

            # ------【异步 RPC】按张量设备选 CPU/GPU 组发起异步 isend，CUDA 张量记录流避免提前释放 ------
            comm_group = metadata_group if tensor.is_cpu else group
            handle = torch.distributed.isend(
                tensor, dst=self.ranks[dst], group=comm_group
            )
            if tensor.is_cuda:
                tensor.record_stream(torch.cuda.current_stream(tensor.device))
            handles.append(handle)

        return handles

    def recv_tensor_dict(
        self,
        src: int | None = None,
        all_gather_group: "GroupCoordinator | None" = None,
        all_gather_tensors: dict[str, bool] | None = None,
    ) -> dict[str, torch.Tensor | Any] | None:
        """Recv the input tensor dictionary.
        NOTE: `src` is the local rank of the source rank.

        all_gather_group: The group for the all-gather operation. If provided,
            an optimization is enabled where each rank in the group sends a
            slice of a tensor and the receiver reconstructs it using an
            all-gather, which can improve performance. This is typically the
            tensor-parallel group.
        all_gather_tensors: A dictionary to specify which tensors should use
            the all-gather optimization, which is only effective when
            `all_gather_group` is provided. By default, this optimization is
            on for any tensor whose size is divisible by the
            `all_gather_group`'s world size. However, it should be disabled
            for tensors that are not fully replicated across the group (e.g.,
            the residual tensor when sequence parallelism is enabled). This
            dictionary allows overriding the default behavior on a per-tensor
            basis.
        """
        # Bypass the function if we are using only 1 GPU.
        if not torch.distributed.is_initialized() or self.world_size == 1:
            return None
        # ------【异步 RPC】发起非阻塞接收并等待完成，再执行后处理（如 all-gather 拼接） ------
        tensor_dict, handles, postprocess = self.irecv_tensor_dict(
            src=src,
            all_gather_group=all_gather_group,
            all_gather_tensors=all_gather_tensors,
        )
        for handle in handles:
            handle.wait()
        for fn in postprocess:
            fn()
        return tensor_dict

    def irecv_tensor_dict(
        self,
        src: int | None = None,
        all_gather_group: "GroupCoordinator | None" = None,
        all_gather_tensors: dict[str, bool] | None = None,
    ) -> tuple[
        dict[str, torch.Tensor | Any] | None,
        list[Handle],
        list[Callable[[], None]],
    ]:
        if not torch.distributed.is_initialized() or self.world_size == 1:
            return None, [], []

        # ------【PP】默认源为环上一 rank，与 send 的 dst 规则配对构成流水线数据流 ------
        if src is None:
            src = (self.rank_in_group - 1) % self.world_size
        assert src < self.world_size, f"Invalid src rank ({src})"

        # ------【异步 RPC】CPU 平台走自定义同步 recv_tensor_dict 路径，无 handle 与后处理直接返回 ------
        if self.use_cpu_custom_send_recv:
            if self.device_communicator is None:
                raise ValueError("No device communicator found")
            # custom device communicator path is synchronous
            sync_tensor_dict = self.device_communicator.recv_tensor_dict(  # type: ignore
                src
            )
            return sync_tensor_dict, [], []

        all_gather_size = 1 if all_gather_group is None else all_gather_group.world_size
        all_gather_rank = (
            0 if all_gather_group is None else all_gather_group.rank_in_group
        )

        group = self.device_group
        metadata_group = self.cpu_group

        # ------【异步 RPC】先收元数据，再逐个异步接收张量；需 all-gather 拼接的登记到 postprocess ------
        recv_metadata_list = self.recv_object(src=src)
        tensor_dict: dict[str, Any] = {}
        handles: list[Handle] = []
        postprocess: list[Callable[[], None]] = []

        # ------【异步 RPC】按元数据分配完整张量，区分 all-gather 切片接收与整张量接收两种路径 ------
        for key, value in recv_metadata_list:
            if isinstance(value, TensorMetadata):
                full_tensor = torch.empty(
                    value.size, dtype=value.dtype, device=value.device
                )
                if full_tensor.numel() == 0:
                    tensor_dict[key] = full_tensor
                    continue

                # ------【TP】all-gather 优化：先收本 rank 切片，注册后处理用 all_gather 拼回完整张量 ------
                if self._should_use_all_gather(
                    key, full_tensor.numel(), all_gather_group, all_gather_tensors
                ):
                    orig_shape = full_tensor.shape
                    slice_tensor = full_tensor.reshape(all_gather_size, -1)[
                        all_gather_rank
                    ]
                    comm_group = metadata_group if slice_tensor.is_cpu else group
                    handle = torch.distributed.irecv(
                        slice_tensor, src=self.ranks[src], group=comm_group
                    )
                    handles.append(handle)

                    # ------【TP】闭包捕获参数，等待后再执行 all_gather 还原原始形状 ------
                    def _postprocess(
                        key: str = key,
                        slice_tensor: torch.Tensor = slice_tensor,
                        orig_shape: tuple[int, ...] = tuple(orig_shape),
                        all_gather_group=all_gather_group,
                    ) -> None:
                        assert all_gather_group is not None
                        tensor_dict[key] = all_gather_group.all_gather(
                            slice_tensor, dim=0
                        ).reshape(orig_shape)

                    postprocess.append(_postprocess)
                    tensor_dict[key] = slice_tensor
                # ------【异步 RPC】普通路径：按设备选组异步接收整个张量 ------
                else:
                    comm_group = metadata_group if full_tensor.is_cpu else group
                    handle = torch.distributed.irecv(
                        full_tensor, src=self.ranks[src], group=comm_group
                    )
                    handles.append(handle)
                    tensor_dict[key] = full_tensor
            # ------【异步 RPC】非张量值直接拷贝进字典，无需通信 ------
            else:
                tensor_dict[key] = value

        return tensor_dict, handles, postprocess

    def barrier(self):
        """Barrier synchronization among the group.
        NOTE: don't use `device_group` here! `barrier` in NCCL is
        terrible because it is internally a broadcast operation with
        secretly created GPU tensors. It is easy to mess up the current
        device. Use the CPU group instead.
        """
        # ------【NCCL 通信】用 CPU 组做 barrier：NCCL barrier 内部会隐式造 GPU 张量易弄乱当前设备 ------
        torch.distributed.barrier(group=self.cpu_group)

    def send(self, tensor: torch.Tensor, dst: int | None = None) -> None:
        """Sends a tensor to the destination rank in a blocking way"""
        """NOTE: `dst` is the local rank of the destination rank."""
        if self.device_communicator is None:
            raise ValueError("No device communicator found")
        # ------【异步 RPC】委托设备通信器做阻塞式 send（PP 层间传激活） ------
        self.device_communicator.send(tensor, dst)

    def recv(
        self, size: torch.Size, dtype: torch.dtype, src: int | None = None
    ) -> torch.Tensor:
        """Receives a tensor from the source rank."""
        """NOTE: `src` is the local rank of the source rank."""
        if self.device_communicator is None:
            raise ValueError("No device communicator found")
        # ------【异步 RPC】委托设备通信器按 size/dtype 阻塞式接收张量 ------
        return self.device_communicator.recv(size, dtype, src)

    def destroy(self):
        # ------【进程管理】销毁设备/CPU 进程组并置空，释放底层 NCCL/gloo 通信资源 ------
        if hasattr(self, "device_group"):
            torch.distributed.destroy_process_group(self.device_group)
            del self.device_group
        if hasattr(self, "cpu_group"):
            torch.distributed.destroy_process_group(self.cpu_group)
            del self.cpu_group
        # ------【异步 RPC】销毁设备通信器与消息队列广播器，回收共享内存/自定义通信资源 ------
        if self.device_communicator is not None:
            self.device_communicator.destroy()
        if self.mq_broadcaster is not None:
            self.mq_broadcaster = None

    def dispatch_router_logits(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        is_sequence_parallel: bool = False,
        extra_tensors: list[torch.Tensor] | None = None,
    ) -> (
        tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]
    ):
        # ------【EP/EPLB】MoE 路由分派：按专家并行把 hidden_states/router_logits 发到对应专家 rank ------
        if self.device_communicator is not None:
            return self.device_communicator.dispatch_router_logits(
                hidden_states,
                router_logits,
                is_sequence_parallel,
                extra_tensors,
            )
        else:
            return hidden_states, router_logits

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        is_sequence_parallel: bool = False,
        extra_tensors: list[torch.Tensor] | None = None,
    ) -> (
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor]]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ):
        # ------【EP/EPLB】MoE token 分派：按 topk_ids 把各 token 的权重/隐状态发往持有对应专家的 rank ------
        if self.device_communicator is not None:
            return self.device_communicator.dispatch(
                hidden_states,
                topk_weights,
                topk_ids,
                is_sequence_parallel,
                extra_tensors,
            )
        else:
            return hidden_states, topk_weights, topk_ids

    def combine(
        self, hidden_states, is_sequence_parallel: bool = False
    ) -> torch.Tensor:
        # ------【EP/EPLB】MoE 结果回收：把各专家 rank 输出聚合回原 token 顺序（支持序列并行） ------
        if self.device_communicator is not None:
            return self.device_communicator.combine(hidden_states, is_sequence_parallel)
        else:
            return hidden_states


_WORLD: GroupCoordinator | None = None
_INNER_DP_WORLD: GroupCoordinator | None = None
_NODE_COUNT: int | None = None


def get_world_group() -> GroupCoordinator:
    assert _WORLD is not None, "world group is not initialized"
    return _WORLD


def get_inner_dp_world_group() -> GroupCoordinator:
    assert _INNER_DP_WORLD is not None, "inner dp world group is not initialized"
    return _INNER_DP_WORLD


def init_world_group(
    ranks: list[int], local_rank: int, backend: str
) -> GroupCoordinator:
    # ------【NCCL 通信】用 GroupCoordinator 包装 WORLD 组，提供统一高层通信接口 ------
    return GroupCoordinator(
        group_ranks=[ranks],
        local_rank=local_rank,
        torch_distributed_backend=backend,
        use_device_communicator=False,
        group_name="world",
    )


def init_model_parallel_group(
    group_ranks: list[list[int]],
    local_rank: int,
    backend: str,
    use_message_queue_broadcaster: bool = False,
    group_name: str | None = None,
    use_device_communicator: bool = True,
    use_all2all: bool = False,
) -> GroupCoordinator:
    # ------【TP/PP/DP/EP】按给定 rank 分组构造 GroupCoordinator，作为各并行维度的通信组 ------
    return GroupCoordinator(
        group_ranks=group_ranks,
        local_rank=local_rank,
        torch_distributed_backend=backend,
        use_device_communicator=use_device_communicator,
        use_message_queue_broadcaster=use_message_queue_broadcaster,
        group_name=group_name,
        use_all2all=use_all2all,
    )


def _init_stateless_group(
    group_ranks: list[list[int]],
    group_name: str,
    host: str,
    backend: str,
    coord_store: Store,
    use_device_communicator: bool = True,
    use_all2all: bool = False,
) -> "StatelessGroupCoordinator":
    """Create a StatelessGroupCoordinator with the given parameters."""
    from vllm.distributed.stateless_coordinator import StatelessGroupCoordinator

    # ------【EP/EPLB】构建无状态协调器：从 world 组取 local_rank/rank 并托管远程协调存储 ------
    world = get_world_group()
    return StatelessGroupCoordinator(
        group_ranks=group_ranks,
        local_rank=world.local_rank,
        torch_distributed_backend=backend,
        use_device_communicator=use_device_communicator,
        group_name=group_name,
        host=host,
        coord_store=coord_store,
        global_rank=world.rank,
        global_world_size=world.world_size,
        use_all2all=use_all2all,
    )


def _replace_active_groups(
    *,
    world: GroupCoordinator | None,
    dp: GroupCoordinator | None,
    ep: GroupCoordinator | None,
    eplb: GroupCoordinator | None,
    node_count: int | None,
) -> None:
    """Destroy the current DP/EP/WORLD/EPLB groups and replace them.

    Destruction is collective — all ranks in the old groups must call this
    function together.  Pass all-``None`` to tear down without replacement.
    """
    # ------【DP/EP/EPLB】集体销毁旧组：所有旧组成员须同时调用，保证 destroy 是集合操作 ------
    global _WORLD, _DP, _EP, _EPLB, _NODE_COUNT
    for group in (_DP, _EP, _WORLD, _EPLB):
        if group is not None:
            group.destroy()
    # ------【DP/EP/EPLB】用新组替换全局引用，实现 DP/EP/WORLD/EPLB 热切换 ------
    _WORLD = world
    _DP = dp
    _EP = ep
    _EPLB = eplb
    _NODE_COUNT = node_count


_TP: GroupCoordinator | None = None


def get_tp_group() -> GroupCoordinator:
    assert _TP is not None, "tensor model parallel group is not initialized"
    return _TP


_DCP: GroupCoordinator | None = None


def get_dcp_group() -> GroupCoordinator:
    assert _DCP is not None, "decode context model parallel group is not initialized"
    return _DCP


_PP: GroupCoordinator | None = None


def get_pp_group() -> GroupCoordinator:
    assert _PP is not None, "pipeline model parallel group is not initialized"
    return _PP


_DP: GroupCoordinator | None = None


def get_dp_group() -> GroupCoordinator:
    assert _DP is not None, "data parallel group is not initialized"
    return _DP


_EP: GroupCoordinator | None = None


def get_ep_group() -> GroupCoordinator:
    assert _EP is not None, (
        "expert parallel group is not initialized. "
        "EP group is only created for MoE models with num_experts > 0. "
        "This function should only be called for MoE models."
    )
    return _EP


_EPLB: GroupCoordinator | None = None


def get_eplb_group() -> GroupCoordinator:
    assert _EPLB is not None, (
        "EPLB group is not initialized. "
        "EPLB group is only created for MoE models when EPLB is enabled. "
        "Ensure parallel_config.enable_eplb is True."
    )
    return _EPLB


_PCP: GroupCoordinator | None = None


def get_pcp_group() -> GroupCoordinator:
    assert _PCP is not None, "prefill context parallel group is not initialized"
    return _PCP


@contextmanager
def graph_capture(
    device: torch.device,
    graph_capture_context: GraphCaptureContext | None = None,
):
    """
    `graph_capture` is a context manager which should surround the code that
    is capturing the CUDA graph. Its main purpose is to ensure that some
    operations will be run after the graph is captured, before the graph
    is replayed. It returns a `GraphCaptureContext` object which contains the
    necessary data for the graph capture. Currently, it only contains the
    stream that the graph capture is running on. This stream is set to the
    current CUDA stream when the context manager is entered and reset to the
    default stream when the context manager is exited. This is to ensure that
    the graph capture is running on a separate stream from the default stream,
    in order to explicitly distinguish the kernels to capture
    from other kernels possibly launched on background in the default stream.

    A caller may pass an explicit ``graph_capture_context`` to control the
    stream used (e.g. to capture on the default stream).
    """
    # ------【CUDA Graph】默认在独立 CUDA 流上建捕获上下文，与默认流后台 kernel 隔离 ------
    context = graph_capture_context or GraphCaptureContext(
        torch.cuda.Stream(device=device)
    )
    # ------【CUDA Graph】同时进入 TP 与 PP 组的图捕获上下文，统一管理捕获期间状态 ------
    with get_tp_group().graph_capture(context), get_pp_group().graph_capture(context):
        yield context


logger = init_logger(__name__)

_ENABLE_CUSTOM_ALL_REDUCE = True


def set_custom_all_reduce(enable: bool):
    global _ENABLE_CUSTOM_ALL_REDUCE
    _ENABLE_CUSTOM_ALL_REDUCE = enable


def _init_process_group_for_split_group(
    *,
    backend: str,
    distributed_init_method: str,
    world_size: int,
    rank: int,
    local_rank: int,
    timeout: timedelta | None,
) -> None:
    """Initialize the default PG with both CPU (gloo) and device (e.g. nccl)
    backends and an eager ``device_id`` binding so that subgroups can be
    created via ``split_group`` (which requires the parent communicator to
    be eagerly initialized). Falls back to ``gloo`` on CPU-only systems.
    """
    # ------【进程管理】有 GPU 时建 cpu:gloo + cuda:nccl 双后端并绑定 device_id，供 split_group 切分 ------
    if torch.accelerator.is_available() and backend != "gloo":
        init_backend = "cpu:gloo,cuda:nccl"
        from vllm.platforms import current_platform

        visible_device_index = current_platform.logical_device_id_to_visible_device_id(
            local_rank
        )
        device_id: torch.device | None = torch.device(f"cuda:{visible_device_index}")
    else:
        # ------【进程管理】无 GPU 回退纯 gloo，保证 CPU-only 环境也能初始化 ------
        init_backend = "gloo"
        device_id = None
    # ------【NCCL 通信】eager 初始化默认进程组，使 split_group 能基于父组切分子组 ------
    torch.distributed.init_process_group(
        backend=init_backend,
        init_method=distributed_init_method,
        world_size=world_size,
        rank=rank,
        timeout=timeout,
        device_id=device_id,
    )


def _validate_default_pg_for_split_group() -> None:
    """When an external launcher (e.g. ``torchrun``) initialized the default
    PG, ``GroupCoordinator`` cannot patch in additional backends or change
    the eager-init behavior — ``split_group`` only selects subsets of an
    existing parent. Validate that the parent has both ``device_id`` and a
    CPU (gloo) backend, and emit a descriptive error pointing at the exact
    init call to update otherwise.
    """
    # ------【进程管理】取默认进程组，校验外部 launcher 是否满足 split_group 的初始化要求 ------
    default_pg = torch.distributed.distributed_c10d._get_default_group()
    # ------【进程管理】要求父组绑定 device_id，否则 split_group 无法按设备切分子组 ------
    assert default_pg.bound_device_id is not None, (
        "External launcher initialized the default process group "
        "without device_id. vLLM requires the default PG to be device-"
        "bound for split_group. Pass device_id=torch.device(f'cuda:"
        "{local_rank}') to torch.distributed.init_process_group()."
    )
    # ------【进程管理】要求父组具备 gloo CPU 后端，缺失时抛出含修复提示的错误 ------
    try:
        default_pg._get_backend(torch.device("cpu"))
    except RuntimeError as e:
        raise RuntimeError(
            "External launcher initialized the default process group "
            "without a CPU (gloo) backend. vLLM requires both CPU and "
            "device backends. Pass backend='cpu:gloo,cuda:nccl' to "
            "torch.distributed.init_process_group()."
        ) from e


def _init_elastic_ep_world(
    config, local_rank: int, backend: str, rank: int, world_size: int
) -> None:
    from vllm.distributed.stateless_coordinator import StatelessGroupCoordinator

    # ------【EP/EPLB】按 data_parallel_rank 偏移计算全局 rank/world_size，进入跨 DP 全局空间 ------
    global _WORLD, _NODE_COUNT
    assert _WORLD is None, "world group already initialized"
    parallel_config = config.parallel_config
    global_rank = parallel_config.data_parallel_rank * world_size + rank
    global_world_size = parallel_config.world_size_across_dp
    all_ranks = list(range(global_world_size))
    # ------【EP/EPLB】把全部 rank 编入单一世界组，供无状态协调器统一管理全局通信 ------
    group_ranks = [all_ranks[i : i + 1] for i in range(global_world_size)]
    if global_rank in all_ranks:
        group_ranks = [all_ranks]
    # ------【EP/EPLB+异步 RPC】获取 TCP 协调存储客户端，无状态协调器据此交换组元数据 ------
    coord_store = get_cached_tcp_store_client(
        parallel_config.data_parallel_master_ip, parallel_config._coord_store_port
    )
    # ------【EP/EPLB】构建无状态 WORLD 协调器，不依赖 NCCL，可跨节点弹性伸缩 ------
    world = StatelessGroupCoordinator(
        group_ranks=group_ranks,
        local_rank=local_rank,
        torch_distributed_backend=backend,
        use_device_communicator=False,
        group_name="world",
        host=parallel_config.data_parallel_master_ip,
        coord_store=coord_store,
        global_rank=global_rank,
        global_world_size=global_world_size,
    )
    # ------【EP/EPLB】校验 TP/PP 必须在单节点内，记录节点数并落库全局 WORLD ------
    assert parallel_config.nnodes_within_dp == 1, (
        "Elastic EP is not supported with multi-node TP/PP"
    )
    _NODE_COUNT = _node_count(world.tcp_store_group)
    _WORLD = world


def init_distributed_environment(
    world_size: int = -1,
    rank: int = -1,
    distributed_init_method: str = "env://",
    local_rank: int = -1,
    backend: str = "nccl",
    timeout: timedelta | None = None,
):
    '''
    这就是你说的「构建 NCCL 通信网络」的确切位置。它做两件事：

    rendezvous（握手）：所有 worker 通过 distributed_init_method（那个 tcp://ip:port 地址）互相找到对方，确认「咱们是同一伙人」
    建 WORLD process group：登记好 world_size 和每个进程的 rank，形成一个全局通信组

        一个容易误解的点：NCCL communicator 是「懒」的
        init_process_group 只是「建群、登记名册、握手」。
        真正的 NCCL communicator（GPU 之间那条物理通信链路）是在第一次发生 collective 通信（比如第一次 all-reduce）时才 lazy 创建的。
    '''
    # ------【EP/EPLB】打印初始化参数并读取配置，先判断是否启用弹性 EP（影响后续建组方式）──
    logger.debug(
        "world_size=%d rank=%d local_rank=%d distributed_init_method=%s backend=%s",
        world_size,
        rank,
        local_rank,
        distributed_init_method,
        backend,
    )
    from vllm.config import get_current_vllm_config_or_none

    config = get_current_vllm_config_or_none()
    enable_elastic_ep = config is not None and config.parallel_config.enable_elastic_ep
    # ------【DP】多节点或跨 DP 副本时，按数据并行维度重排 rank/world_size，使各副本进入同一全局组 ------
    if (
        config is not None
        and config.parallel_config.distributed_executor_backend != "external_launcher"
        and (
            config.parallel_config.nnodes > 1
            or config.parallel_config.data_parallel_size > 1
        )
        and not enable_elastic_ep
    ):
        parallel_config = config.parallel_config
        # adjust to take into account data parallelism
        # offset the rank by the data parallel rank
        rank = parallel_config.data_parallel_rank * world_size + rank
        # adjust the world size to take into account data parallelism
        world_size = parallel_config.world_size_across_dp

        # Use appropriate IP and port based on configuration
        # ------【DP】多节点走 master 地址；单节点多 DP 副本各用独立端口做 rendezvous ------
        if parallel_config.nnodes > 1:
            ip = parallel_config.master_addr
            port = parallel_config.master_port
            distributed_init_method = get_distributed_init_method(ip, port)
        else:
            ip = parallel_config.data_parallel_master_ip
            port = parallel_config.get_next_dp_init_port()
            distributed_init_method = get_distributed_init_method(ip, port)
            logger.debug(
                "Adjusting world_size=%d rank=%d distributed_init_method=%s for DP",
                world_size,
                rank,
                distributed_init_method,
            )
    # ------【NCCL 通信】进程组尚未初始化时，进入建组流程（先校验 init_method 与 backend）──
    if not torch.distributed.is_initialized():
        logger.info(
            "world_size=%d rank=%d local_rank=%d distributed_init_method=%s backend=%s",
            world_size,
            rank,
            local_rank,
            distributed_init_method,
            backend,
        )
        assert distributed_init_method is not None, (
            "distributed_init_method must be provided when initializing "
            "distributed environment"
        )
        # ------【NCCL 通信】请求的 backend 不可用时回退到 gloo，保证分布式初始化不失败 ------
        if not torch.distributed.is_backend_available(backend):
            logger.warning(
                "Distributed backend %s is not available; falling back to gloo.",
                backend,
            )
            assert torch.distributed.is_gloo_available(), (
                "Fallback Gloo backend is not available."
            )
            backend = "gloo"
        if envs.VLLM_DISTRIBUTED_USE_SPLIT_GROUP:
            # ------【进程管理】split_group 需提前拿到 local_rank 以计算 device_id（eager 初始化）──
            # split_group needs local_rank early to compute device_id for
            # the eager init. local_rank is not available in torch
            # ProcessGroup, see https://github.com/pytorch/pytorch/issues/122816
            if local_rank == -1:
                local_rank = (
                    int(envs.LOCAL_RANK)
                    if distributed_init_method == "env://"
                    else rank
                )
            _init_process_group_for_split_group( 
                backend=backend,
                distributed_init_method=distributed_init_method,
                world_size=world_size,
                rank=rank,
                local_rank=local_rank,
                timeout=timeout,
            )
        else:
            # ------【NCCL 通信】常规模式：直接建 WORLD 进程组（NCCL communicator 惰性创建）──
            # this backend is used for WORLD
            # 作用：所有 worker 通过 TCP rendezvous 握手，建出 WORLD process group
            torch.distributed.init_process_group( # 真正建群的动作
                backend=backend, # nccl
                init_method=distributed_init_method, # 本worker的zmq的url
                world_size=world_size,
                rank=rank,
                timeout=timeout,
            )
        # ------【EP/EPLB】弹性 EP：额外建一个 gloo CPU 组用于协调 TP/PP 组初始化 ------
        if enable_elastic_ep:
            tp_pp_cpu_group = torch.distributed.new_group(
                backend="gloo", timeout=timeout
            )
            if _node_count(tp_pp_cpu_group) > 1:
                # NOTE(yongji): StatelessGroupCoordinator uses data_parallel_master_ip
                # to initialize all DP/EP groups, hence all ranks within TP/PP group
                # must reside on the same node
                raise RuntimeError(
                    "Elastic EP is not yet supported with multi-node TP/PP"
                )

    # ------【NCCL 通信】split_group 模式下校验默认进程组的切分一致性 ------
    if envs.VLLM_DISTRIBUTED_USE_SPLIT_GROUP and torch.accelerator.is_available():
        _validate_default_pg_for_split_group()

    # ------【进程管理】补全 local_rank：单机场景直接以 rank 作为 local_rank ------
    # set the local rank
    # local_rank is not available in torch ProcessGroup,
    # see https://github.com/pytorch/pytorch/issues/122816
    if local_rank == -1:
        # local rank not set, this usually happens in single-node
        # setting, where we can use rank as local rank
        local_rank = envs.LOCAL_RANK if distributed_init_method == "env://" else rank

    # ------【EP/EPLB】弹性 EP 走独立初始化路径（StatelessGroupCoordinator），完成后直接返回 ------
    global _WORLD, _NODE_COUNT, _INNER_DP_WORLD
    if enable_elastic_ep:
        _init_elastic_ep_world(config, local_rank, backend, rank, world_size)
        return
    # ------【NCCL 通信】用 vLLM 的 GroupCoordinator 包装 WORLD 组，提供高层通信接口并探测节点数 ------
    if _WORLD is None:
        ranks = list(range(torch.distributed.get_world_size()))

        '''
                作用：把刚才那个 WORLD group 用 vLLM 自己的 GroupCoordinator 包一层
│                 （提供 rank_in_group、device_group、广播等高层接口）
        '''
        _WORLD = init_world_group(ranks, local_rank, backend)
        if config is not None and config.parallel_config.nnodes > 1:
            _NODE_COUNT = config.parallel_config.nnodes
        else:
            _NODE_COUNT = _node_count(_WORLD.cpu_group)
        logger.debug("Detected %d nodes in the distributed environment", _NODE_COUNT)
    else:
        assert _WORLD.world_size == torch.distributed.get_world_size(), (
            "world group already initialized with a different world size"
        )
    # ------【DP】跨节点 DP（nnodes_within_dp>1）时，为每个 DP 副本建内部 world 组用于消息广播 ------
    if config is not None and config.parallel_config.nnodes_within_dp > 1:
        if parallel_config.data_parallel_size > 1:
            world_size_inner_dp = parallel_config.world_size
            group_ranks = [
                [dp_rank * world_size_inner_dp + i for i in range(world_size_inner_dp)]
                for dp_rank in range(parallel_config.data_parallel_size)
            ]
            _INNER_DP_WORLD = init_model_parallel_group(
                group_ranks,
                get_world_group().local_rank,
                backend,
                use_message_queue_broadcaster=True,
                group_name="inner_dp_world",
                use_device_communicator=False,
            )
        else:
            _INNER_DP_WORLD = _WORLD


def initialize_model_parallel(
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    prefill_context_model_parallel_size: int = 1,
    decode_context_model_parallel_size: int | None = 1,
    backend: str | None = None,
) -> None:
    """
    Initialize model parallel groups.

    Arguments:
        tensor_model_parallel_size: number of GPUs used for tensor model
            parallelism.
        pipeline_model_parallel_size: number of GPUs used for pipeline model
            parallelism.
        backend: name of torch distributed communication backend.

    Let's say we have a total of 8 GPUs denoted by g0 ... g7 and we
    use 2 GPUs to parallelize the model tensor, and 4 GPUs to parallelize
    the model pipeline. The present function will
    create 4 tensor model-parallel groups and 2 pipeline model-parallel groups:
        4 tensor model-parallel groups:
            [g0, g1], [g2, g3], [g4, g5], [g6, g7]
        2 pipeline model-parallel groups:
            [g0, g2, g4, g6], [g1, g3, g5, g7]
    Note that for efficiency, the caller should make sure adjacent ranks
    are on the same DGX box. For example if we are using 2 DGX-1 boxes
    with a total of 16 GPUs, rank 0 to 7 belong to the first box and
    ranks 8 to 15 belong to the second box.
    """
    # ------【核心逻辑】确认分布式已初始化，并读取并行配置（DP/EPLB 开关、后端等） ------
    # Get world size and rank. Ensure some consistencies.
    assert torch.distributed.is_initialized()

    from vllm.config import get_current_vllm_config

    config = get_current_vllm_config()
    data_parallel_size = config.parallel_config.data_parallel_size
    enable_elastic_ep = config.parallel_config.enable_elastic_ep
    parallel_config = config.parallel_config
    coord_store: Store | None = None
    # ------【EP/EPLB】弹性 EP 分支：从无状态 world 组取全局信息并构建本地 TP/PP/PCP 排名张量 ------
    if enable_elastic_ep:
        coord_store = get_cached_tcp_store_client(
            parallel_config.data_parallel_master_ip,
            parallel_config._coord_store_port,
        )
        # Use stateless world group for global information
        world_size = get_world_group().world_size
        rank = get_world_group().rank
        backend = backend or "nccl"
        tp_pp_pcp_size = (
            tensor_model_parallel_size
            * pipeline_model_parallel_size
            * prefill_context_model_parallel_size
        )
        local_all_ranks = torch.arange(tp_pp_pcp_size).reshape(
            pipeline_model_parallel_size,
            prefill_context_model_parallel_size,
            tensor_model_parallel_size,
        )
    else:
        # ------【核心逻辑】常规分支：从 torch.distributed 取 world_size/rank 与后端 ------
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
        backend = backend or torch.distributed.get_backend(
            get_world_group().device_group
        )

    # the layout order is: ExternalDP x DP x PP x PCP x TP
    # ExternalDP is the data parallel group that is not part of the model,
    # every dp rank can generate independently (in verl integration).
    # DP is the data parallel group that is part of the model,
    # all the ranks in the same DP group should generate simultaneously,
    # i.e. the `generate` call in the same DP group should be called together,
    # otherwise it will cause deadlock.
    # to get group_ranks for each dimension, transpose that dimension to the
    # last dimension, then reshape to 2D, then unbind the last dimension
    # ------【TP/PP/DP】把 rank 重塑为 (ExternalDP,DP,PP,PCP,TP) 张量，便于按维转置切出各并行组 ------
    all_ranks = torch.arange(world_size).reshape(
        -1,
        data_parallel_size,
        pipeline_model_parallel_size,
        prefill_context_model_parallel_size,
        tensor_model_parallel_size,
    )  # noqa

    # ------【TP】切出 TP 组：TP 维已落在最后一维，view+unbind 得到每组的 rank 列表 ------
    # Build the tensor model-parallel groups.
    global _TP
    assert _TP is None, "tensor model parallel group is already initialized"
    group_ranks = all_ranks.view(-1, tensor_model_parallel_size).unbind(0)
    group_ranks = [x.tolist() for x in group_ranks]
    if enable_elastic_ep:
        group_ranks = local_all_ranks.view(-1, tensor_model_parallel_size).unbind(0)
        group_ranks = [x.tolist() for x in group_ranks]
    # message queue broadcaster is only used in tensor model parallel group
    _TP = init_model_parallel_group(
        group_ranks,
        get_world_group().local_rank,
        backend,
        use_message_queue_broadcaster=True,
        group_name="tp",
    )

    # ------【TP/PP】切出 DCP 组：decode 上下文并行，可跨 PCP+TP 组成完整 TP×PCP 组 ------
    # Build the DCP model-parallel groups.
    global _DCP
    assert _DCP is None, "decode context model parallel group is already initialized"
    dcp_size = decode_context_model_parallel_size or 1
    dcp_ranks = local_all_ranks if enable_elastic_ep else all_ranks
    if dcp_size > 1:
        # DCP spans PCP first, then TP for full TP x PCP groups.
        dcp_ranks = dcp_ranks.transpose(-1, -2)
    group_ranks = dcp_ranks.reshape(-1, dcp_size).unbind(0)
    group_ranks = [x.tolist() for x in group_ranks]
    _DCP = init_model_parallel_group(
        group_ranks,
        get_world_group().local_rank,
        backend,
        use_message_queue_broadcaster=True,
        group_name="dcp",
    )

    # ------【PD 分离】切出 PCP（prefill 上下文并行）组，transpose 使 PCP 维落到最后一维 ------
    global _PCP
    assert _PCP is None, "prefill context parallel group is already initialized"
    group_ranks = (
        all_ranks.transpose(3, 4)
        .reshape(-1, prefill_context_model_parallel_size)
        .unbind(0)
    )
    group_ranks = [x.tolist() for x in group_ranks]
    if enable_elastic_ep:
        group_ranks = (
            local_all_ranks.transpose(1, 2)
            .reshape(-1, prefill_context_model_parallel_size)
            .unbind(0)
        )
        group_ranks = [x.tolist() for x in group_ranks]
    _PCP = init_model_parallel_group(
        group_ranks, get_world_group().local_rank, backend, group_name="pcp"
    )

    # ------【PP】切出 PP 组：transpose 使 PP 维落到最后一维再 reshape+unbind ------
    # Build the pipeline model-parallel groups.
    global _PP
    assert _PP is None, "pipeline model parallel group is already initialized"
    group_ranks = (
        all_ranks.transpose(2, 4).reshape(-1, pipeline_model_parallel_size).unbind(0)
    )
    group_ranks = [x.tolist() for x in group_ranks]
    if enable_elastic_ep:
        group_ranks = (
            local_all_ranks.transpose(0, 2)
            .reshape(-1, pipeline_model_parallel_size)
            .unbind(0)
        )
        group_ranks = [x.tolist() for x in group_ranks]
    _PP = init_model_parallel_group(
        group_ranks, get_world_group().local_rank, backend, group_name="pp"
    )

    # ------【DP】切出 DP 组；弹性 EP 走无状态协调器，常规走 GroupCoordinator ------
    global _DP
    assert _DP is None, "data parallel group is already initialized"
    group_ranks = all_ranks.transpose(1, 4).reshape(-1, data_parallel_size).unbind(0)
    group_ranks = [x.tolist() for x in group_ranks]
    if enable_elastic_ep:
        _DP = _init_stateless_group(
            group_ranks,
            "dp",
            parallel_config.data_parallel_master_ip,
            backend,
            coord_store=coord_store,
        )
    else:
        _DP = init_model_parallel_group(
            group_ranks, get_world_group().local_rank, backend, group_name="dp"
        )

    # ------【EP/EPLB】初始化专家并行组，把 DP×PCP×TP 融合为 EP 组供 MoE all2all 通信 ------
    global _EP
    assert _EP is None, "expert parallel group is already initialized"
    # Don't create EP group for dense models.
    # ------【EP/EPLB】仅为 MoE 模型建 EP 组，dense 模型跳过 ------
    if config.model_config is None or config.model_config.is_moe:
        group_ranks = (
            all_ranks.transpose(1, 2)
            .reshape(
                -1,
                data_parallel_size
                * prefill_context_model_parallel_size
                * tensor_model_parallel_size,
            )
            .unbind(0)
        )
        group_ranks = [x.tolist() for x in group_ranks]
        use_all2all = parallel_config.use_all2all
        if enable_elastic_ep:
            _EP = _init_stateless_group(
                group_ranks,
                "ep",
                parallel_config.data_parallel_master_ip,
                backend,
                coord_store=coord_store,
                use_all2all=use_all2all,
            )
        else:
            _EP = init_model_parallel_group(
                group_ranks,
                get_world_group().local_rank,
                backend,
                group_name="ep",
                use_all2all=use_all2all,
            )

        # ------【EP/EPLB】独立建 EPLB 组（与 EP 同 ranks），隔离 MoE 前向与负载均衡集合通信避免死锁 ------
        # Create EPLB group with the same ranks as EP if EPLB is enabled.
        # This is a separate process group to isolate EPLB communications
        # from MoE forward pass collectives and prevent deadlocks when
        # using torch.distributed in execution with torch.distributed in EPLB.
        global _EPLB
        assert _EPLB is None, "EPLB group is already initialized"
        if config.parallel_config.enable_eplb:
            if enable_elastic_ep:
                _EPLB = _init_stateless_group(
                    group_ranks,
                    "eplb",
                    parallel_config.data_parallel_master_ip,
                    backend,
                    coord_store=coord_store,
                )
            else:
                _EPLB = init_model_parallel_group(
                    group_ranks,
                    get_world_group().local_rank,
                    backend,
                    group_name="eplb",
                )
    # If no EP group needed, _EP remains None
    # If no EPLB group needed, _EPLB remains None

    # ------【核心逻辑】打印各并行维度的局部 rank，便于定位本进程在各组中的身份 ------
    logger.info_once(
        "rank %s in world size %s is assigned as "
        "DP rank %s, PP rank %s, PCP rank %s, "
        "TP rank %s, EP rank %s, EPLB rank %s",
        rank,
        world_size,
        _DP.rank_in_group,
        _PP.rank_in_group,
        _PCP.rank_in_group,
        _TP.rank_in_group,
        _EP.rank_in_group if _EP is not None else "N/A",
        _EPLB.rank_in_group if _EPLB is not None else "N/A",
    )


def ensure_model_parallel_initialized(
    tensor_model_parallel_size: int,
    pipeline_model_parallel_size: int,
    prefill_context_model_parallel_size: int = 1,
    decode_context_model_parallel_size: int | None = 1,
    backend: str | None = None,
) -> None:
    """Helper to initialize model parallel groups if they are not initialized,
    or ensure tensor-parallel and pipeline-parallel sizes are equal to expected
    values if the model parallel groups are initialized.
    """
    # ------【核心逻辑】从 world 组取 backend，兼容有无 .backend 属性的不同组类型 ------
    world_group = get_world_group()
    if hasattr(world_group, "backend"):
        backend = backend or world_group.backend
    else:
        backend = backend or torch.distributed.get_backend(world_group.device_group)
    # ------【核心逻辑】未初始化时直接建组并返回，否则进入下方的大小校验 ------
    if not model_parallel_is_initialized():
        initialize_model_parallel(
            tensor_model_parallel_size,
            pipeline_model_parallel_size,
            prefill_context_model_parallel_size,
            decode_context_model_parallel_size,
            backend,
        )
        return

    # ------【TP/PP】逐一校验 TP/PP/PCP/DCP 组大小与期望一致，防止复用不匹配的组 ------
    assert get_tensor_model_parallel_world_size() == tensor_model_parallel_size, (
        "tensor parallel group already initialized, but of unexpected size. "
        f"got: {get_tensor_model_parallel_world_size()=} vs. "
        f"wanted: {tensor_model_parallel_size=}"
    )
    pp_world_size = get_pp_group().world_size
    assert pp_world_size == pipeline_model_parallel_size, (
        "pipeline parallel group already initialized, but of unexpected size. "
        f"got: {pp_world_size=} vs. "
        f"wanted: {pipeline_model_parallel_size=}"
    )
    pcp_world_size = get_pcp_group().world_size
    assert pcp_world_size == prefill_context_model_parallel_size, (
        "prefill context parallel group already initialized, but of unexpected size: "
        f"{pcp_world_size=} vs. "
        f"{prefill_context_model_parallel_size=}"
    )
    dcp_world_size = get_dcp_group().world_size
    dcp_model_parallel_size = decode_context_model_parallel_size or 1
    assert dcp_world_size == dcp_model_parallel_size, (
        "decode context parallel group already initialized, but of unexpected size: "
        f"{dcp_world_size=} vs. "
        f"{dcp_model_parallel_size=}"
    )


def checkpoint_prepare_distributed_state() -> None:
    """Prepare every device communicator for a process checkpoint."""
    # ------【CUDA Graph】同步所有流后让每个设备通信器进入 checkpoint 就绪状态 ------
    torch.accelerator.synchronize()
    _apply_to_device_comms(lambda comm: comm.checkpoint_prepare())
    torch.accelerator.synchronize()


def checkpoint_restore_distributed_state() -> None:
    """Restore every device communicator after a process checkpoint."""
    # ------【CUDA Graph】同步后恢复每个设备通信器，配合进程 checkpoint 快照恢复 ------
    torch.accelerator.synchronize()
    _apply_to_device_comms(lambda comm: comm.checkpoint_restore())
    torch.accelerator.synchronize()


def model_parallel_is_initialized():
    """Check if tensor and pipeline parallel groups are initialized."""
    return _TP is not None and _PP is not None


def get_tensor_model_parallel_world_size() -> int:
    """Return world size for the tensor model parallel group."""
    return get_tp_group().world_size


def get_tensor_model_parallel_rank() -> int:
    """Return my rank for the tensor model parallel group."""
    return get_tp_group().rank_in_group


def get_node_count() -> int:
    """Return the total number of nodes in the distributed environment."""
    assert _NODE_COUNT is not None, "distributed environment is not initialized"
    return _NODE_COUNT


def destroy_model_parallel():
    """Set the groups to none and destroy them."""
    # ------【TP】销毁 TP 通信组并置空，释放底层 NCCL communicator ------
    global _TP

    if _TP:
        _TP.destroy()
    _TP = None

    # ------【TP/PP】销毁 DCP 组并置空，回收 decode 上下文并行通信器 ------
    global _DCP
    if _DCP:
        _DCP.destroy()
    _DCP = None

    # ------【PD 分离】销毁 PCP 组并置空，回收 prefill 上下文并行通信器 ------
    global _PCP
    if _PCP:
        _PCP.destroy()
    _PCP = None

    # ------【PP】销毁 PP 组并置空，回收流水线并行通信器 ------
    global _PP
    if _PP:
        _PP.destroy()
    _PP = None

    # ------【DP】销毁 DP 组并置空，回收数据并行通信器 ------
    global _DP
    if _DP:
        _DP.destroy()
    _DP = None

    # ------【EP/EPLB】销毁 EP 组并置空，回收专家并行通信器 ------
    global _EP
    if _EP:
        _EP.destroy()
    _EP = None

    # ------【EP/EPLB】销毁 EPLB 组并置空，回收负载均衡通信器 ------
    global _EPLB
    if _EPLB:
        _EPLB.destroy()
    _EPLB = None


def destroy_distributed_environment():
    # ------【NCCL 通信】销毁 WORLD 协调器并清空节点数，回收世界组资源 ------
    global _WORLD, _NODE_COUNT
    if _WORLD:
        _WORLD.destroy()
    _WORLD = None
    _NODE_COUNT = None
    # ------【NCCL 通信】销毁底层 torch 进程组，释放全局 NCCL/gloo 通信器 ------
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def cleanup_dist_env_and_memory(shutdown_ray: bool = False):
    logger.debug(
        "[shutdown] Distributed: cleanup start shutdown_ray=%s",
        shutdown_ray,
    )
    # ------【核心逻辑】重置环境变量缓存，确保后续读取 os.environ 的最新值 ------
    # Reset environment variable cache
    envs.disable_envs_cache()

    # Reset rocm_aiter_ops class variables to match current os.environ.
    # These are class-level attributes that persist across tests and are
    # NOT restored by monkeypatch (which only restores os.environ).
    # ------【CUDA Graph】ROCm 下刷新 aiter_ops 类变量，与当前 os.environ 保持一致 ------
    from vllm.platforms import current_platform

    if current_platform.is_rocm():
        from vllm._aiter_ops import rocm_aiter_ops

        rocm_aiter_ops.refresh_env_variables()

    # ------【进程管理】解冻 GC 保护的分配器对象，允许后续回收显存 ------
    # Ensure all objects are not frozen before cleanup
    gc.unfreeze()

    # ------【NCCL 通信】销毁全部并行组与世界组，释放 NCCL 通信器 ------
    destroy_model_parallel()
    destroy_distributed_environment()
    # ------【进程管理】可选关闭 Ray 运行时，回收 Ray 侧资源 ------
    if shutdown_ray:
        import ray  # Lazy import Ray

        ray.shutdown()
    # ------【内存池/CuMem】触发 GC 并清空加速器显存与主机缓存，归还内存池 ------
    gc.collect()
    from vllm.platforms import current_platform

    if not current_platform.is_cpu():
        torch.accelerator.empty_cache()
        try:
            torch.accelerator.empty_host_cache()
        except AttributeError:
            logger.warning(
                "torch.accelerator.empty_host_cache() only available in Pytorch >=2.9"
            )

    logger.debug_once("[shutdown] Distributed: cleanup complete")


def in_the_same_node_as(
    pg: ProcessGroup | StatelessProcessGroup, source_rank: int = 0
) -> list[bool]:
    """
    This is a collective operation that returns if each rank is in the same node
    as the source rank. It tests if processes are attached to the same
    memory system (shared access to shared memory).
    """
    # ------【进程管理】区分 ProcessGroup 与无状态组，取组内 rank/world_size 与全局 ranks ------
    if isinstance(pg, ProcessGroup):
        assert torch.distributed.get_backend(pg) != torch.distributed.Backend.NCCL, (
            "in_the_same_node_as should be tested with a non-NCCL group."
        )
        # local rank inside the group
        rank = torch.distributed.get_rank(group=pg)
        world_size = torch.distributed.get_world_size(group=pg)

        # global ranks of the processes in the group
        ranks = torch.distributed.get_process_group_ranks(pg)
    else:
        rank = pg.rank
        world_size = pg.world_size
        ranks = list(range(world_size))

    # ------【进程管理】每进程本地张量记录是否与 source_rank 同节点，后续集合汇总 ------
    # local tensor in each process to store the result
    is_in_the_same_node = torch.tensor(
        [0] * world_size, dtype=torch.int32, device="cpu"
    )

    magic_message = b"magic_message"
    shm = None

    # ------【进程管理】source_rank 建共享内存段并广播名称，其余进程尝试打开以探测同节点 ------
    try:
        with contextlib.suppress(OSError):
            if rank == source_rank:
                # create a shared memory segment
                shm = shared_memory.SharedMemory(create=True, size=128)
                assert shm.buf is not None, "Buffer was not created"
                shm.buf[: len(magic_message)] = magic_message
                if isinstance(pg, ProcessGroup):
                    torch.distributed.broadcast_object_list(
                        [shm.name], src=ranks[source_rank], group=pg
                    )
                else:
                    pg.broadcast_obj(shm.name, src=source_rank)
                is_in_the_same_node[rank] = 1
            else:
                # try to open the shared memory segment
                if isinstance(pg, ProcessGroup):
                    recv = [None]
                    torch.distributed.broadcast_object_list(
                        recv, src=ranks[source_rank], group=pg
                    )
                    name = recv[0]
                else:
                    name = pg.broadcast_obj(None, src=source_rank)
                # fix to https://stackoverflow.com/q/62748654/9191338
                # Python incorrectly tracks shared memory even if it is not
                # created by the process. The following patch is a workaround.
                with patch(
                    "multiprocessing.resource_tracker.register",
                    lambda *args, **kwargs: None,
                ):
                    shm = shared_memory.SharedMemory(name=name)
                assert shm.buf is not None, "Buffer was not opened"
                if shm.buf[: len(magic_message)] == magic_message:
                    is_in_the_same_node[rank] = 1
    except Exception as e:
        logger.error("Error ignored in is_in_the_same_node: %s", e)
    finally:
        if shm:
            shm.close()

    # ------【进程管理】屏障同步，确保所有进程完成探测后再汇总 ------
    if isinstance(pg, ProcessGroup):
        torch.distributed.barrier(group=pg)
    else:
        pg.barrier()

    # ------【进程管理】source_rank 清理共享内存段，避免资源泄漏 ------
    # clean up the shared memory segment
    with contextlib.suppress(OSError):
        if rank == source_rank and shm:
            shm.unlink()

    # ------【进程管理】集合汇总各进程探测结果，得到每个 rank 是否与 source 同节点的布尔列表 ------
    if isinstance(pg, ProcessGroup):
        torch.distributed.all_reduce(is_in_the_same_node, group=pg)
        aggregated_data = is_in_the_same_node
    else:
        aggregated_data = torch.zeros_like(is_in_the_same_node)
        for i in range(world_size):
            rank_data = pg.broadcast_obj(is_in_the_same_node, src=i)
            aggregated_data += rank_data

    return [x == 1 for x in aggregated_data.tolist()]


def is_global_first_rank() -> bool:
    """
    Check if the current process is the first rank globally across all
    parallelism strategies (PP, TP, DP, EP, etc.).

    Unlike group-specific checks like `get_tensor_model_parallel_rank() == 0`
    or `get_pp_group().is_first_rank`, this function checks the global rank
    across all parallelism dimensions.

    Returns:
        bool: True if this is the global first rank (rank 0), False otherwise.
              Returns True if distributed is not initialized (single process).
    """
    # ------【核心逻辑】优先用 world 组判断全局首 rank，这是最准确的判断方式 ------
    try:
        # If world group is available, use it for the most accurate check
        global _WORLD
        if _WORLD is not None:
            return _WORLD.is_first_rank

        # ------【核心逻辑】未初始化分布式时视作单进程，直接返回 True ------
        # If torch distributed is not initialized, assume single process
        if not torch.distributed.is_initialized():
            return True

        # ------【核心逻辑】回退到 torch 全局 rank 判断，异常时兜底按首 rank 处理 ------
        # Fallback to torch's global rank
        return torch.distributed.get_rank() == 0

    except Exception:
        # If anything goes wrong, assume this is the first rank
        return True


def is_local_first_rank() -> bool:
    """
    Check if the current process is the first local rank (rank 0 on its node).
    """
    # ------【核心逻辑】优先用 world 组的 local_rank 判断是否本节点首 rank ------
    try:
        # prefer the initialized world group if available
        global _WORLD
        if _WORLD is not None:
            return _WORLD.local_rank == 0

        # ------【核心逻辑】未初始化时视作单进程返回 True ------
        if not torch.distributed.is_initialized():
            return True

        # ------【核心逻辑】回退读 LOCAL_RANK 环境变量（env:// launcher 会设置），再兜底全局 rank ------
        # fallback to environment-provided local rank if available
        # note: envs.LOCAL_RANK is set when using env:// launchers (e.g., torchrun)
        try:
            return int(envs.LOCAL_RANK) == 0  # type: ignore[arg-type]
        except Exception:
            return torch.distributed.get_rank() == 0
    except Exception:
        return True


def _node_count(pg: ProcessGroup | StatelessProcessGroup) -> int:
    """
    Returns the total number of nodes in the process group.

    Args:
        pg: The process group to analyze

    Returns:
        int: The total number of nodes
    """
    # ------【进程管理】取组内 world_size，单进程直接返回 1 ------
    if isinstance(pg, ProcessGroup):
        world_size = torch.distributed.get_world_size(group=pg)
    else:
        world_size = pg.world_size

    if world_size == 1:
        return 1

    # ------【进程管理】初始化 rank->node_id 映射，逐个 rank 用连通分量聚类统计节点数 ------
    # Build node assignment map
    node_assignment = [0] * world_size  # rank -> node_id
    next_node_id = 0

    for current_rank in range(world_size):
        if node_assignment[current_rank] != 0:
            continue  # Already assigned to a node

        # ------【进程管理】对未归类 rank 开新节点，并用 in_the_same_node_as 合并同节点 rank ------
        # Assign current rank to a new node
        next_node_id += 1
        node_assignment[current_rank] = next_node_id

        # Find all ranks on the same node as current_rank
        same_node_flags = in_the_same_node_as(pg, current_rank)
        for other_rank, is_same_node in enumerate(same_node_flags):
            if is_same_node and node_assignment[other_rank] == 0:
                node_assignment[other_rank] = next_node_id

    return next_node_id
