# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: SIM117
import copy
from collections.abc import Generator

import torch
from torch import nn

from vllm.config import ModelConfig, ParallelConfig, VllmConfig
from vllm.config.load import LoadConfig
from vllm.logger import init_logger
from vllm.model_executor.model_loader.base_loader import BaseModelLoader
from vllm.model_executor.model_loader.tensorizer import (
    TensorizerConfig,
    deserialize_tensorizer_model,
    init_tensorizer_model,
    is_vllm_tensorized,
    serialize_vllm_model,
    tensorizer_weights_iterator,
)
from vllm.model_executor.model_loader.utils import (
    get_model_architecture,
    initialize_model,
)
from vllm.utils.torch_utils import set_default_torch_dtype

logger = init_logger(__name__)

BLACKLISTED_TENSORIZER_ARGS = {
    "device",  # vLLM decides this
    "dtype",  # vLLM decides this
    "mode",  # Not meant to be configurable by the user
}


def validate_config(config: dict):
    # ------【序列化】逐项检查用户配置，禁止传 device/dtype/mode 等由 vLLM 决定的参数 ------
    for k, v in config.items():
        if v is not None and k in BLACKLISTED_TENSORIZER_ARGS:
            raise ValueError(f"{k} is not an allowed Tensorizer argument.")


class TensorizerLoader(BaseModelLoader):
    """Model loader using CoreWeave's tensorizer library."""

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)
        # ------【序列化】已是 TensorizerConfig 则直接用，否则先校验再从字典构造配置 ------
        if isinstance(load_config.model_loader_extra_config, TensorizerConfig):
            self.tensorizer_config = load_config.model_loader_extra_config
        else:
            validate_config(load_config.model_loader_extra_config)
            self.tensorizer_config = TensorizerConfig(
                **load_config.model_loader_extra_config["tensorizer_config"]
            )

    def _verify_config(
        self, model_config: ModelConfig, parallel_config: ParallelConfig
    ):
        # ------【序列化】校验 tensorizer 配置与模型配置、并行配置是否一致 ------
        self.tensorizer_config.verify_with_model_config(model_config)
        self.tensorizer_config.verify_with_parallel_config(parallel_config)

    def _get_weights_iterator(
        self,
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        # ------【序列化】把配置构造成 tensorizer 参数并返回权重迭代器 ------
        tensorizer_args = self.tensorizer_config._construct_tensorizer_args()
        return tensorizer_weights_iterator(tensorizer_args)

    def _load_model_serialized_cpu(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> nn.Module:
        """Load a serialized model with tensorizer to the CPU.

        This is only necessary when the model isn't vLLM-tensorized (see
        examples/features/tensorize_vllm_model.py) This should still
        be faster than default HuggingFace loading, but will be slower than
        loading a vLLM-tensorized model.
        """
        device_config = vllm_config.device_config
        model_config = vllm_config.model_config
        # ------【核心逻辑】在指定 dtype 与设备上下文下初始化空模型（meta 零初始化） ------
        with set_default_torch_dtype(model_config.dtype):
            with torch.device(device_config.device):
                model = initialize_model(vllm_config=vllm_config, prefix=prefix)

            # ------【序列化】用 tensorizer 权重迭代器逐块灌入权重 ------
            model.load_weights(self._get_weights_iterator())
        # ------【核心逻辑】模型置为 eval 模式后返回 ------
        return model.eval()

    def download_model(self, model_config: ModelConfig) -> None:
        # ------【序列化】下载前校验 tensorizer 配置与模型配置一致 ------
        self.tensorizer_config.verify_with_model_config(model_config)

        # ------【下载缓存】打开 tensorizer 流以触发序列化文件的下载/预取 ------
        with self.tensorizer_config.open_stream():
            pass

    def _patch_tensorizer_config(self, model_config: ModelConfig) -> TensorizerConfig:
        # ------【序列化】复制配置并注入 model_class/hf_config/dtype 等运行时信息 ------
        model_class = get_model_architecture(model_config)[0]
        tensorizer_config = copy.copy(self.tensorizer_config)
        tensorizer_config.model_class = model_class
        tensorizer_config.hf_config = model_config.hf_config
        tensorizer_config.dtype = model_config.dtype
        return tensorizer_config

    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None:
        """Load serialized model weights with tensorizer.

        Expects a vLLM-tensorized model. See the
        examples/features/tensorize_vllm_model.py example script
        for serializing vLLM models."""
        # ------【序列化】vLLM 已序列化的模型走反序列化直灌，否则按普通权重迭代加载 ------
        if is_vllm_tensorized(self.tensorizer_config):
            tensorizer_config = self._patch_tensorizer_config(model_config)
            deserialize_tensorizer_model(model, tensorizer_config)
        else:
            model.load_weights(self._get_weights_iterator())

    def load_model(
        self, vllm_config: VllmConfig, model_config: ModelConfig, prefix: str = ""
    ) -> nn.Module:
        parallel_config = vllm_config.parallel_config
        # ------【序列化】加载前校验 tensorizer 配置与模型/并行配置一致 ------
        self._verify_config(model_config, parallel_config)

        # ------【TP】多卡张量并行时按 rank 改写 URI，使每卡读各自分片序列化文件 ------
        if parallel_config.tensor_parallel_size > 1:
            from vllm.distributed import get_tensor_model_parallel_rank

            assert self.tensorizer_config.tensorizer_uri is not None
            self.tensorizer_config.tensorizer_uri = (
                self.tensorizer_config.tensorizer_uri % get_tensor_model_parallel_rank()
            )

        # ------【序列化】vLLM 已序列化则直接反序列化建模型，否则走 CPU 序列化加载路径 ------
        if is_vllm_tensorized(self.tensorizer_config):
            tensorizer_config = self._patch_tensorizer_config(model_config)
            device_config = vllm_config.device_config
            # ------【核心逻辑】在指定 dtype 与设备上下文下反序列化构建模型 ------
            with set_default_torch_dtype(model_config.dtype):
                with torch.device(device_config.device):
                    model = init_tensorizer_model(
                        tensorizer_config=tensorizer_config, vllm_config=vllm_config
                    )
            # ------【序列化】建好模型后再灌入剩余权重并返回 ------
            self.load_weights(model, model_config)
            return model
        return self._load_model_serialized_cpu(vllm_config=vllm_config, prefix=prefix)

    @staticmethod
    def save_model(
        model: torch.nn.Module,
        tensorizer_config: TensorizerConfig | dict,
        model_config: ModelConfig,
    ) -> None:
        # ------【序列化】字典形式的配置先转成 TensorizerConfig 对象 ------
        if isinstance(tensorizer_config, dict):
            tensorizer_config = TensorizerConfig(**tensorizer_config)
        # ------【序列化】把模型权重序列化写入 tensorizer 文件 ------
        serialize_vllm_model(
            model=model,
            tensorizer_config=tensorizer_config,
            model_config=model_config,
        )
