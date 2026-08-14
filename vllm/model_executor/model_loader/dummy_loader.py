# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch.nn as nn

from vllm.config import ModelConfig
from vllm.config.load import LoadConfig
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.model_loader.base_loader import BaseModelLoader
from vllm.model_executor.model_loader.reload.layerwise import (
    _get_original_loader,
    get_layerwise_info,
)
from vllm.model_executor.model_loader.reload.meta import materialize_layer
from vllm.model_executor.model_loader.reload.types import LayerReloadingInfo
from vllm.model_executor.model_loader.reload.utils import get_layer_tensors
from vllm.model_executor.model_loader.weight_utils import (
    initialize_dummy_weights,
    initialize_single_dummy_weight,
)


class DummyModelLoader(BaseModelLoader):
    """Model loader that will set model weights to random values."""

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)
        # ------【核心逻辑】dummy loader 不接受额外配置，传入即报错 ------
        if load_config.model_loader_extra_config:
            raise ValueError(
                f"Model loader extra config is not supported for "
                f"load format {load_config.load_format}"
            )

    def download_model(self, model_config: ModelConfig) -> None:
        pass  # Nothing to download

    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None:
        # ------【层式加载】遍历模型每个子模块，按层决定 dummy 权重处理方式 ------
        for layer in model.modules():
            info = get_layerwise_info(layer)
            # ------【层式加载】可逐层重载的量化层走在线量化流程 ------
            if info.can_load():
                self._process_online_quant_layer(layer, info)
            else:
                # NOTE(woosuk): For accurate performance evaluation, we assign
                # random values to the weights.
                # ------【核心逻辑】普通层用随机值填充权重，用于纯性能评测 ------
                initialize_dummy_weights(layer, model_config)

    def _process_online_quant_layer(
        self,
        layer: nn.Module,
        info: LayerReloadingInfo,
    ) -> None:
        """Materialize, apply dummy weights, and run quantization processing."""
        # ------【meta 设备】把 meta 态层物化为真实显存 tensor ------
        materialize_layer(layer, info)

        # ------【核心逻辑】为该层每个 tensor 单独填充随机 dummy 权重 ------
        for tensor in get_layer_tensors(layer).values():
            initialize_single_dummy_weight(tensor)

        # ------【权重加载】恢复每个参数原本的 weight_loader，供后续真实加载复用 ------
        for param in get_layer_tensors(layer).values():
            param.weight_loader = _get_original_loader(param)

        # ------【量化】若层带量化方法，物化后再跑量化后处理(如缩放因子打包) ------
        quant_method = getattr(layer, "quant_method", None)
        if isinstance(quant_method, QuantizeMethodBase):
            quant_method.process_weights_after_loading(layer)

        # ------【层式加载】重置该层重载信息，标记本层处理完成 ------
        info.reset()
