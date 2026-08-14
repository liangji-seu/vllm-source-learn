# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable
from functools import wraps
from types import FunctionType
from typing import TYPE_CHECKING

import torch

from vllm.config import ModelConfig

from .layerwise import (
    finalize_layerwise_reload,
    initialize_layerwise_reload,
)

if TYPE_CHECKING:
    from vllm.model_executor.models.utils import AutoWeightsLoader

__all__ = ["set_torchao_reload_attrs", "support_quantized_model_reload_from_hp_weights"]


def set_torchao_reload_attrs(model: torch.nn.Module, model_config: ModelConfig):
    # ------【量化】在模型上标记启用 torchao 重载并保存配置，供装饰器后续读取 ------
    model._do_torchao_reload = True
    model._model_config = model_config


def support_quantized_model_reload_from_hp_weights(original_load_weights: FunctionType):
    """
    Decorator for `load_weights` method for AutoWeightsLoader.load_weights to support
    reloading high precision (bfloat16/float16/float32) weight for an already quantized
    model, this involves restoring the weights to a high precision weights and
    then online quantize the weights.

    Only applies to torchao quantized models. Assumes that all model weights are
    loaded within a single weights iterator (cannot perform batched updates)
    """

    @wraps(original_load_weights)
    def patched_model_load_weights(
        self: "AutoWeightsLoader",
        weights: Iterable[tuple[str, torch.Tensor]],
        *args,
        **kwargs,
    ):
        # ------【核心逻辑】从 loader 取出被加载的模型 ------
        model = self.module

        # ------【量化】未启用 torchao 重载时走原逻辑，保持普通加载路径不变 ------
        if not getattr(model, "_do_torchao_reload", False):
            return original_load_weights(self, weights, *args, **kwargs)

        # ------【量化】进入层式加载：把模型恢复到 meta 并包装 loader 为在线量化 ------
        initialize_layerwise_reload(model)
        # ------【权重加载】执行原始高精度权重加载，权重被缓存而非直接写盘 ------
        loaded_weights = original_load_weights(self, weights, *args, **kwargs)
        # ------【量化】整批加载完后统一物化、在线量化并拷回 kernel 张量 ------
        finalize_layerwise_reload(model, model._model_config)

        return loaded_weights

    return patched_model_load_weights
