# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import inspect
from collections.abc import Callable

import torch
from torch.nn.parameter import UninitializedParameter
from torch.utils._python_dispatch import TorchDispatchMode

from .sanitize import restore_layer_refs, sanitize_layer_refs
from .types import LayerReloadingInfo, LayerTensors
from .utils import get_layer_params_buffers, get_layer_tensors

__all__ = [
    "to_meta_tensor",
    "materialize_meta_tensor",
    "capture_layer_to_meta",
    "restore_layer_on_meta",
    "materialize_layer",
    "get_numel_loaded",
]

# Modules whose tensors are never moved to, or materialized from, the meta device.
SKIP_MODULES: set[str] = {"HadamardTransform"}

# Tensors never loaded by a weight loader, so the layerwise trigger ignores them.
SKIP_LOAD_TENSORS: set[str] = {
    "_expert_map",
    "expert_mask",
    "expert_global_to_physical",
    "expert_physical_to_global",
    "expert_local_to_global",
    "e_score_correction_bias",
}

# Tensors which are never moved to, or materialized from, the meta device.
# `bias` is built after create_weights(), so it is never on meta to begin with.
SKIP_TENSORS: set[str] = SKIP_LOAD_TENSORS | {"bias"}


def to_meta_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Convert a tensor to a meta tensor while preserving class and attributes."""
    # ------【meta 设备】把真实张量转成 meta 张量并保留原类与属性，实现零显存快照 ------
    meta_tensor = tensor.data.to("meta")
    meta_tensor.__class__ = tensor.__class__
    meta_tensor.__dict__ = tensor.__dict__.copy()
    return meta_tensor


def materialize_meta_tensor(meta_tensor: torch.Tensor) -> torch.Tensor:
    """
    Materialize a meta tensor into an actual tensor on the current device.
    Should be called within the torch device context for the given rank.
    """
    # ------【meta 设备】按 meta 张量的 shape/stride/dtype 分配真实显存，恢复类与属性 ------
    tensor = torch.empty_strided(
        size=tuple(meta_tensor.size()),
        stride=tuple(meta_tensor.stride()),
        dtype=meta_tensor.dtype,
        requires_grad=False,
    )
    tensor.__class__ = meta_tensor.__class__
    tensor.__dict__ = meta_tensor.__dict__.copy()
    return tensor


def _is_non_persistent_parameter_alias_buffer(
    layer: torch.nn.Module,
    name: str,
    buffer: torch.Tensor,
    parameter_storage_ptrs: set[int],
) -> bool:
    # ------【层式加载】非持久 buffer 集合之外的直接排除，只关心别名 buffer ------
    if name not in layer._non_persistent_buffers_set:
        return False

    # ------【权重切分】比对底层存储地址判断 buffer 是否与参数共享存储（别名） ------
    buffer_storage_ptr = _tensor_storage_ptr(buffer)
    return (
        buffer_storage_ptr is not None and buffer_storage_ptr in parameter_storage_ptrs
    )


def _tensor_storage_ptr(tensor: torch.Tensor) -> int | None:
    # ------【核心逻辑】未初始化参数无存储，直接返回 None ------
    if isinstance(tensor, UninitializedParameter):
        return None

    # ------【核心逻辑】取张量底层存储地址，meta 张量无存储时捕获异常返回 None ------
    try:
        return tensor.untyped_storage().data_ptr()
    except (RuntimeError, ValueError):
        return None


def _parameter_storage_ptrs(layer: torch.nn.Module) -> set[int]:
    # ------【权重切分】收集层内所有参数的存储地址集合，用于识别别名 buffer ------
    return {
        storage_ptr
        for param in layer.parameters(recurse=True)
        if (storage_ptr := _tensor_storage_ptr(param)) is not None
    }


def capture_layer_to_meta(layer: torch.nn.Module) -> LayerTensors:
    # ------【meta 设备】跳过 HadamardTransform 等从不搬上 meta 的模块 ------
    if layer.__class__.__name__ in SKIP_MODULES:
        return ({}, {})

    # ------【权重切分】取层的参数与 buffer，并统计参数存储地址用于剔除别名 ------
    params, buffers = get_layer_params_buffers(layer)
    parameter_storage_ptrs = _parameter_storage_ptrs(layer)
    return (
        # ------【meta 设备】把每个参数转 meta 并清洗层引用，形成可恢复的零显存快照 ------
        {
            name: sanitize_layer_refs(to_meta_tensor(param), layer)
            for name, param in params.items()
            if name not in SKIP_TENSORS
        },
        # ------【meta 设备】把 buffer 转 meta，排除跳过项与参数别名，避免重复存储 ------
        {
            name: sanitize_layer_refs(to_meta_tensor(buffer), layer)
            for name, buffer in buffers.items()
            if name not in SKIP_TENSORS
            and not _is_non_persistent_parameter_alias_buffer(
                layer, name, buffer, parameter_storage_ptrs
            )
        },
    )


def restore_layer_on_meta(layer: torch.nn.Module, info: LayerReloadingInfo):
    """Restore a layer to model format with tensors on the meta device"""
    # ------【meta 设备】跳过 HadamardTransform 等从不搬上 meta 的模块 ------
    if layer.__class__.__name__ in SKIP_MODULES:
        return

    # ------【层式加载】记录非持久 buffer 名，恢复时据此决定持久属性 ------
    non_persistent = set(layer._non_persistent_buffers_set)
    # ------【meta 设备】删除层上现有真实张量，为 meta 张量腾出位置 ------
    for name in get_layer_tensors(layer):
        if name not in SKIP_TENSORS:
            delattr(layer, name)

    restore_params, restore_buffers = info.restore_metadata
    # ------【meta 设备】恢复 meta 参数并重新绑定层引用，供后续物化加载 ------
    for name, param in restore_params.items():
        if name not in SKIP_TENSORS:
            param = restore_layer_refs(param, layer)
            layer.register_parameter(name, param)

    # ------【meta 设备】恢复 meta buffer 并按持久属性注册，还原层结构 ------
    for name, buffer in restore_buffers.items():
        if name not in SKIP_TENSORS:
            buffer = restore_layer_refs(buffer, layer)
            layer.register_buffer(name, buffer, persistent=name not in non_persistent)


def materialize_layer(layer: torch.nn.Module, info: LayerReloadingInfo):
    """Materialize all meta tensors in a layer to actual tensors."""
    # ------【meta 设备】跳过 HadamardTransform 等从不物化的模块 ------
    if layer.__class__.__name__ in SKIP_MODULES:
        return

    # ------【meta 设备】在目标设备上下文内把层的 meta 张量逐个物化为真实显存 ------
    with info.restore_device:
        for name, tensor in get_layer_tensors(layer).items():
            if name not in SKIP_TENSORS and tensor.is_meta:
                setattr(layer, name, materialize_meta_tensor(tensor))


class CopyCounter(TorchDispatchMode):
    """
    Tracks total number of elements modified with `copy_`.

    Useful for keeping track of weight loading where underlying weights can be
    arbitrarily transformed (such as with `narrow`) before calling copy.

    Note: Assumes that copy kwargs are not used.
    """

    def __init__(self):
        super().__init__()
        # ------【核心逻辑】初始化累计计数器，记录 copy_ 修改的元素总数 ------
        self.copied_numel = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        # ------【权重加载】拦截 copy_ 算子，累计拷贝元素数以推断加载进度 ------
        if func is torch.ops.aten.copy_.default:
            assert args[0].numel() == args[1].numel()
            self.copied_numel += args[0].numel()

        return func(*args, **kwargs)


def get_numel_loaded(
    weight_loader: Callable, args: inspect.BoundArguments
) -> tuple[int, object]:
    """
    Determine how many elements would be loaded by a weight loader call.

    Args:
        weight_loader: used to load weights
        args: bound arguments to weight loader

    Returns:
        number of elements loaded by the weight loader, the return value of the
        weight loader
    """
    # ------【权重加载】在 CopyCounter 上下文中执行 loader，统计实际写入的元素数 ------
    with CopyCounter() as counter:
        return_value = weight_loader(*args.args, **args.kwargs)

    # A weight loader fills a single destination parameter, so the number of
    # loaded elements is at most that parameter's size. Some loaders copy into
    # the parameter more than once -- e.g. ``composed_weight_loader`` runs an
    # in-place post-load transform (``param.copy_(fn(param))``) on top of the
    # initial copy -- which would make CopyCounter report twice the parameter
    # size. Over-counting inflates the layer's loaded-element total and can
    # finalize the layer before every parameter is loaded, silently dropping
    # the trailing parameter(s) (e.g. Mamba ``mixer.D``). Cap the count at the
    # destination size to keep the per-layer accounting correct.
    numel = counter.copied_numel
    param = args.arguments.get("param", None)
    # ------【权重加载】用目标参数大小封顶计数，避免多次拷贝的 loader 被高估导致提前触发 ------
    if isinstance(param, torch.Tensor):
        numel = min(numel, param.numel())
    return numel, return_value
