# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import importlib

from torch import nn

from vllm.config import ModelConfig, VllmConfig
from vllm.config.load import LoadConfig
from vllm.model_executor.model_loader.base_loader import BaseModelLoader
from vllm.tracing import instrument

_MODELEXPRESS_LOADER_MODULE = "modelexpress.engines.vllm.loader"
_MISSING_MODELEXPRESS_MODULES = frozenset(
    {
        "modelexpress",
        "modelexpress.engines",
        "modelexpress.engines.vllm",
        _MODELEXPRESS_LOADER_MODULE,
    }
)


def _missing_modelexpress_error() -> ImportError:
    return ImportError(
        "The 'modelexpress' load format requires the ModelExpress Python package. "
        "Install it with `pip install modelexpress`."
    )


class ModelExpressModelLoader(BaseModelLoader):
    """Thin vLLM loader wrapper for ModelExpress."""

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)
        # ------【核心逻辑】延迟实例化第三方 ModelExpress loader 并持有委托对象 ------
        self._loader = self._load_modelexpress_loader(load_config)

    @staticmethod
    def _load_modelexpress_loader(load_config: LoadConfig) -> BaseModelLoader:
        # ------【权重加载】尝试动态导入 ModelExpress 的 vLLM loader 模块 ------
        try:
            module = importlib.import_module(_MODELEXPRESS_LOADER_MODULE)
        except ModuleNotFoundError as exc:
            # ------【核心逻辑】仅当缺失的是 modelexpress 自身模块时转成友好错误 ------
            if exc.name not in _MISSING_MODELEXPRESS_MODULES:
                raise
            raise _missing_modelexpress_error() from exc

        # ------【核心逻辑】从模块取出 MxModelLoader 类并实例化真正的加载器 ------
        ModelExpressVllmLoader = module.MxModelLoader
        return ModelExpressVllmLoader(load_config)

    def download_model(self, model_config: ModelConfig) -> None:
        # ------【下载缓存】委托给底层 ModelExpress loader 完成权重下载 ------
        self._loader.download_model(model_config)

    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None:
        # ------【权重加载】委托给底层 ModelExpress loader 完成权重填充 ------
        self._loader.load_weights(model, model_config)

    @instrument(span_name="Load model")
    def load_model(
        self,
        vllm_config: VllmConfig,
        model_config: ModelConfig,
        prefix: str = "",
    ) -> nn.Module:
        # ------【权重加载】委托底层 loader 构建完整模型实例 ------
        model = self._loader.load_model(
            vllm_config=vllm_config,
            model_config=model_config,
            prefix=prefix,
        )
        # ------【核心逻辑】切到 eval 模式再返回，禁用 dropout 等训练行为 ------
        return model.eval()
