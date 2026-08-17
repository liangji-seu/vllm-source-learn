# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import ABC, abstractmethod

import torch
import torch.nn as nn

import vllm.envs as envs
from vllm.config import ModelConfig, VllmConfig
from vllm.config.load import LoadConfig
from vllm.logger import init_logger
from vllm.model_executor.model_loader.reload import finalize_layerwise_processing
from vllm.model_executor.model_loader.utils import (
    initialize_model,
    process_weights_after_loading,
)
from vllm.platforms import current_platform
from vllm.tracing import instrument
from vllm.utils.mem_utils import format_gib
from vllm.utils.torch_utils import set_default_torch_dtype

logger = init_logger(__name__)


class BaseModelLoader(ABC):
    """Base class for model loaders."""

    def __init__(self, load_config: LoadConfig):
        self.load_config = load_config

    @abstractmethod
    def download_model(self, model_config: ModelConfig) -> None:
        """Download a model so that it can be immediately loaded."""
        raise NotImplementedError

    @abstractmethod
    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None:
        """Load weights into a model. This standalone API allows
        inplace weights loading for an already-initialized model"""
        raise NotImplementedError














    @instrument(span_name="Load model")
    def load_model(
        self, vllm_config: VllmConfig, model_config: ModelConfig, prefix: str = ""
    ) -> nn.Module:
        """Load a model with the given configurations."""
        device_config = vllm_config.device_config # cuda
        load_config = vllm_config.load_config
        # ------【显存 profiling】权重加载设备优先取 load_config，否则回退到 device_config ------
        load_device = ( # cuda
            device_config.device if load_config.device is None else load_config.device
        )

        ##################################################################################
        # 1. 指定我们的模型所在设备
        ##################################################################################
        target_device = torch.device(load_device)
        # ------【核心逻辑】在指定 dtype 与设备上下文下初始化空模型（meta 设备零初始化） ------
        with set_default_torch_dtype(model_config.dtype):
            with target_device:
                ####################################################################################
                # 2. 开始构造Qwen2ForCausalLM模型实例
                ####################################################################################
                model = initialize_model( 
                    vllm_config=vllm_config,
                    model_config=model_config,
                    prefix=prefix,
                )

            # ------【核心逻辑】按需打印模型结构，便于排查权重加载前的模型布局 ------
            log_model_inspection(model)

            # ------【权重加载】调用子类实现把权重灌入刚初始化的模型 ------
            logger.debug("Loading weights on %s ...", load_device)
            ###############################################################
            # 3. 开始加载权重
            ###############################################################
            self.load_weights(model, model_config)

            # ------【显存 profiling】CUDA/XPU 上记录加载权重后的峰值显存，用于在线量化测试覆盖 ------
            # Log peak GPU memory after loading weights. This is needed
            # to have test coverage on peak memory for online quantization.
            if current_platform.is_cuda_alike() or current_platform.is_xpu():
                peak_memory = torch.accelerator.max_memory_allocated()
                logger.debug_once(
                    "Peak GPU memory after loading weights: %s GiB",
                    format_gib(peak_memory),
                )

            # ------【量化+层式加载】在线量化时收尾逐层处理，把权重转成 kernel 格式 ------
            # Process weights into kernel format. Note that when using online
            # quantization, weights are (typically) quantized as they are loaded.
            if _has_online_quant(model):
                finalize_layerwise_processing(model, model_config)

            # ------【权重加载】权重加载后统一后处理（含 meta→device、量化等） ------
            process_weights_after_loading(model, model_config, target_device)

        # ------【核心逻辑】把模型置为 eval 模式后返回，完成加载主流程 ------
        ###############################################################
        # 把这个模型设置成推理模式，返回
        ###############################################################
        return model.eval()









def log_model_inspection(model: nn.Module) -> None:
    """Log model structure if VLLM_LOG_MODEL_INSPECTION=1."""
    # ------【核心逻辑】未开启模型结构日志开关时直接短路返回 ------
    if not envs.VLLM_LOG_MODEL_INSPECTION:
        return

    from vllm.model_inspection import format_model_inspection

    # ------【核心逻辑】格式化并打印模型结构，供人工排查加载前的层布局 ------
    logger.info("vLLM model structure:\n%s", format_model_inspection(model))


def _has_online_quant(model: nn.Module):
    # ------【量化】遍历所有子模块，探测是否存在使用 meta 设备的在线量化方法 ------
    for module in model.modules():
        quant_method = getattr(module, "quant_method", None)
        if getattr(quant_method, "uses_meta_device", False):
            return True

    return False
