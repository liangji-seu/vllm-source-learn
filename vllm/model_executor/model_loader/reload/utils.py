# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from inspect import BoundArguments

import torch

from .types import LayerReloadingInfo, LayerTensors

__all__ = [
    "get_layer_tensors",
    "get_layer_params_buffers",
    "get_layer_size",
    "has_device_tensors",
    "get_info_size",
]


def get_layer_tensors(layer: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Get all parameters and buffers from a module as a dict."""
    params, buffers = get_layer_params_buffers(layer)
    # ------【层式加载】合并参数与 buffer 为单一字典，便于统一遍历层的全部张量 ------
    return params | buffers


def get_layer_params_buffers(layer: torch.nn.Module) -> LayerTensors:
    """Get all parameters and buffers of a module as a tuple of dicts."""
    # ------【层式加载】分别收集层内非空的参数与 buffer，供元信息捕获与恢复使用 ------
    return (
        {name: param for name, param in layer._parameters.items() if param is not None},
        {name: buffer for name, buffer in layer._buffers.items() if buffer is not None},
    )


def get_layer_size(layer: torch.nn.Module) -> int:
    """Calculate total number of elements across loadable tensors in a layer.

    Excludes SKIP_LOAD_TENSORS (e.g. _expert_map) which are never loaded via
    weight_loader during layerwise reload.
    """
    from .meta import SKIP_LOAD_TENSORS

    # ------【层式加载】累加可加载张量的元素总数，作为触发层处理的阈值 ------
    return sum(
        tensor.numel()
        for name, tensor in get_layer_tensors(layer).items()
        if name not in SKIP_LOAD_TENSORS
    )


def has_device_tensors(bound_args: BoundArguments) -> bool:
    """
    Return True if the loaded weights exist on an accelerator device

    Args:
        bound_args: args to load weights

    Returns:
        True if weights are on accelerator device
    """
    # ------【显存 profiling】检测加载参数里是否存在显存张量，用于告警多缓冲占用 ------
    return any(
        isinstance(value, torch.Tensor) and value.device.type not in ("meta", "cpu")
        for value in bound_args.arguments.values()
    )


def get_info_size(info: LayerReloadingInfo) -> int:
    """
    Calculate the number of bytes used by loaded weights for a given layer

    Args:
        info: layerwise info to get size of

    Returns:
        number of bytes used by loaded weights
    """
    # ------【显存 profiling】统计缓存权重中显存张量的字节数，用于估算额外显存占用 ------
    return sum(
        value.nbytes
        for _, args in info.loaded_weights
        for value in args.arguments.values()
        if isinstance(value, torch.Tensor) and value.device.type not in ("meta", "cpu")
    )
