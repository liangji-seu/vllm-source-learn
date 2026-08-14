# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Utilities for selecting and loading models."""

import inspect
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn
from typing_extensions import assert_never

import vllm.envs as envs
from vllm.config import ModelConfig, VllmConfig, set_current_vllm_config
from vllm.logger import init_logger
from vllm.model_executor.layers.attention import (
    MMEncoderAttention,
)
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.hpc import HpcModule
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.model_loader.reload import (
    record_metadata_for_reloading,
    set_torchao_reload_attrs,
)
from vllm.model_executor.models.interfaces import SupportsQuant
from vllm.tracing import instrument
from vllm.utils.mem_utils import release_device_memory_under_pressure
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

logger = init_logger(__name__)


@instrument(span_name="Initialize model")
def initialize_model(
    vllm_config: VllmConfig,
    *,
    prefix: str = "",
    model_class: type[nn.Module] | None = None,
    model_config: ModelConfig | None = None,
) -> nn.Module:
    """Initialize a model with the given configurations."""
    # ------【核心逻辑】未显式传入时回退到 vllm_config，统一模型配置来源 ------
    if model_config is None:
        model_config = vllm_config.model_config
    # ------【核心逻辑】未指定模型类时按架构名解析出 model_class ------
    if model_class is None:
        model_class, _ = get_model_architecture(model_config)

    # ------【量化】先把融合模块映射注入量化配置，供加载时匹配 packed 权重 ------
    if vllm_config.quant_config is not None:
        configure_quant_config(vllm_config.quant_config, model_class)

    # ------【核心逻辑】反射读取 __init__ 参数名，区分 new-style 与 old-style 模型类 ------
    signatures = inspect.signature(model_class.__init__)
    all_params = [param.name for param in signatures.parameters.values()]
    # ------【核心逻辑】new-style：按 vllm_config+prefix 构造并记录 reload 元数据 ------
    if "vllm_config" in all_params and "prefix" in all_params:
        # new-style model class
        with set_current_vllm_config(vllm_config, check_compile=True, prefix=prefix):
            model = model_class(vllm_config=vllm_config, prefix=prefix)
            record_metadata_for_reloading(model)
            return model

    # ------【核心逻辑】缺少新式参数时告警，进入旧式模型兼容路径 ------
    msg = (
        "vLLM model class should accept `vllm_config` and `prefix` as "
        "input arguments. Possibly you have an old-style model class"
        " registered from out of tree and it is used for new vLLM version. "
        "Check https://docs.vllm.ai/en/latest/design/arch_overview.html "
        "for the design and update the model class accordingly."
    )
    warnings.warn(msg, DeprecationWarning, stacklevel=2)

    logger.warning(
        "Trying to guess the arguments for old-style model class %s",
        model_class,
    )
    # try to be compatible with old-style model class
    # ------【核心逻辑】按参数名逐项填 kwargs，兼容旧式模型的多种构造签名 ------
    kwargs: dict[str, Any] = {}
    if "prefix" in all_params:
        kwargs["prefix"] = prefix
    if "config" in all_params:
        kwargs["config"] = model_config.hf_config
    if "cache_config" in all_params:
        kwargs["cache_config"] = vllm_config.cache_config
    if "quant_config" in all_params:
        kwargs["quant_config"] = vllm_config.quant_config
    if "lora_config" in all_params:
        kwargs["lora_config"] = vllm_config.lora_config
    if "scheduler_config" in all_params:
        kwargs["scheduler_config"] = vllm_config.scheduler_config
    # ------【核心逻辑】old-style：用猜测的 kwargs 构造模型并记录 reload 元数据 ------
    with set_current_vllm_config(vllm_config, check_compile=True, prefix=prefix):
        model = model_class(**kwargs)
        record_metadata_for_reloading(model)

    return model


def process_weights_after_loading(
    model: nn.Module, model_config: ModelConfig, target_device: torch.device
) -> None:
    # ------【量化】遍历所有子模块，找出带 quant_method 的量化层做加载后处理 ------
    for _, module in model.named_modules():
        quant_method = getattr(module, "quant_method", None)
        if isinstance(quant_method, QuantizeMethodBase):
            # When quant methods need to process weights after loading
            # (for repacking, quantizing, etc), they expect parameters
            # to be on the global target device. This scope is for the
            # case where cpu offloading is used, where we will move the
            # parameters onto device for processing and back off after.
            # ------【权重加载】临时把 CPU offload 参数搬到目标设备，再执行量化后处理 ------
            with device_loading_context(module, target_device):
                quant_method.process_weights_after_loading(module)
            # process_weights_after_loading may swap in freshly-created
            # Parameters (e.g. FP8 requantization), which are stamped with the
            # global rank in BasevLLMParameter.__init__. Re-reconcile their TP
            # state to the layer so a later weight reload / RL weight-refit
            # narrows replicated (disable_tp) weights at the correct offset.
            # ------【TP 权重切分】重打包可能换新参数，重对齐 TP 状态避免 offset 错位 ------
            if hasattr(module, "update_param_tp_status"):
                module.update_param_tp_status()
            # Repacking transients above can leave large amounts of memory in
            # the caching allocator, which starves the OS on UMA devices.
            # ------【显存 profiling】释放重打包产生的临时显存，防止 UMA 设备内存饥饿 ------
            release_device_memory_under_pressure(target_device)

    # Initialize post-load attention weights for any attention layer and MM
    # encoder. NOTE: Happens after other modules so we can easily decompress
    # weights.
    # ------【核心逻辑】第二轮遍历 attention/MMEncoder 层做加载后权重初始化 ------
    for _, module in model.named_modules():
        if isinstance(module, (AttentionLayerBase, MMEncoderAttention)) and hasattr(
            module, "process_weights_after_loading"
        ):
            # TODO(lucas): see if there is a way to unify the signatures
            # of process_weights_after_loading
            with device_loading_context(module, target_device):
                module.process_weights_after_loading(model_config.dtype)

    # Process HPC modules (HpcRopeNorm, etc.) that rely on
    # process_weights_after_loading being called from the model's
    # load_weights(). When using DummyModelLoader (e.g. profiling or
    # sleep/wake_up reload), the model's load_weights() is not called, so we
    # must handle HPC modules here generically.
    # ------【显存 profiling】处理 HPC 模块，兼容 DummyModelLoader 不调用 load_weights 的场景 ------
    for _, module in model.named_modules():
        if isinstance(module, HpcModule):
            module.process_weights_after_loading(model)

    # Model-level post-load hook, after the per-layer quant finalize.
    # ------【核心逻辑】模型级 post-load 钩子，在逐层量化收尾后统一调用 ------
    if hasattr(model, "process_weights_after_loading"):
        model.process_weights_after_loading()

    # Needed for torchao model reloading via model.reload_weights
    # @kylesayrs @jerryzh168 this can be removed if callers move to `reload_weights`
    # ------【量化】torchao 记录 reload 属性，供 reload_weights 恢复量化状态 ------
    if model_config.quantization == "torchao":
        set_torchao_reload_attrs(model, model_config)


@contextmanager
def device_loading_context(module: torch.nn.Module, target_device: torch.device):
    # ------【权重加载】目标设备是 CPU 则无需搬运，直接让出模块 ------
    if target_device.type == "cpu":
        # If target is CPU, no need to move anything
        yield module
        return

    # ------【权重加载】记录原始设备与 UVA offload 参数名，供 finally 阶段还原 ------
    original_device_states: dict[str, torch.device] = {}
    uva_offloaded_parameters: list[str] = []

    # ------【权重加载】把仍在 CPU 的参数搬到目标设备，供量化/重打包在设备上处理 ------
    # Store original device states and move parameters to GPU if they're on CPU
    for name, p in module.named_parameters():
        if p.device.type == "cpu":
            original_device_states[name] = p.device
            p.data = p.data.to(target_device)
        if getattr(p, "_vllm_is_uva_offloaded", False):
            uva_offloaded_parameters.append(name)
        # Parameters already on target device are not touched

    # ------【权重加载】上下文管理器主体：yield 期间模块参数停留在目标设备 ------
    try:
        yield module

    finally:
        # ------【权重加载】按环境变量决定是否 pin_memory 加速 CPU 侧访问 ------
        use_pin_memory = (
            is_pin_memory_available()
            and not envs.VLLM_WEIGHT_OFFLOADING_DISABLE_PIN_MEMORY
        )
        # ------【权重加载】还原参数到原始设备，忽略期间新创建的参数 ------
        # Restore parameters to their original devices, ignoring new parameters
        for name, p in module.named_parameters():
            if name in original_device_states:
                original_device: torch.device = original_device_states[name]
                p.data = p.data.to(original_device)

            # ------【权重加载】UVA offload 参数被替换后重新 offload 回 CPU 并恢复标志 ------
            # parameter is UVA offloaded, but was replaced with a new device tensor
            # re-offload it to CPU using UVA
            if name in uva_offloaded_parameters and not getattr(
                p, "_vllm_is_uva_offloaded", False
            ):
                cpu_data = p.data.to(device="cpu")
                if use_pin_memory:
                    cpu_data = cpu_data.pin_memory()
                p.data = get_accelerator_view_from_cpu_tensor(cpu_data)
                p._vllm_is_uva_offloaded = True


_MODEL_ARCH_BY_HASH = dict[int, tuple[type[nn.Module], str]]()
"""Caches the outputs of `_get_model_architecture`."""


def _get_model_architecture(model_config: ModelConfig) -> tuple[type[nn.Module], str]:
    from vllm.model_executor.models.adapters import as_embedding_model, as_seq_cls_model

    # ------【核心逻辑】从 HF 配置读取 architectures 列表，供解析模型实现类 ------
    architectures = getattr(model_config.hf_config, "architectures", None) or []

    # ------【核心逻辑】通过 registry 按架构名解析出 vLLM 模型类与架构字符串 ------
    model_cls, arch = model_config.registry.resolve_model_cls(
        architectures,
        model_config=model_config,
    )

    # ------【核心逻辑】无 vLLM 实现时回退到 Transformers 后端并告警性能损失 ------
    if arch == model_config._get_transformers_backend_cls():
        assert model_config.model_impl != "vllm"
        if model_config.model_impl == "auto":
            logger.warning_once(
                "%s has no vLLM implementation, falling back to Transformers "
                "implementation. Some features may not be supported and "
                "performance may not be optimal.",
                arch,
            )

    # ------【核心逻辑】按 convert_type 决定是否包装为 embedding/分类模型 ------
    convert_type = model_config.convert_type
    if convert_type == "none":
        pass
    elif convert_type == "embed":
        logger.debug_once("Converting to embedding model.")
        model_cls = as_embedding_model(model_cls)
    elif convert_type == "classify":
        logger.debug_once("Converting to sequence classification model.")
        model_cls = as_seq_cls_model(model_cls)
    else:
        assert_never(convert_type)

    return model_cls, arch


def get_model_architecture(model_config: ModelConfig) -> tuple[type[nn.Module], str]:
    # ------【核心逻辑】把影响架构解析的配置项哈希成 key，用于进程内缓存去重 ------
    key = hash(
        (
            model_config.model,
            model_config.convert_type,
            model_config.runner_type,
            model_config.trust_remote_code,
            model_config.model_impl,
            tuple(getattr(model_config.hf_config, "architectures", None) or []),
        )
    )
    # ------【核心逻辑】命中缓存直接返回，省去 registry 查找开销 ------
    if key in _MODEL_ARCH_BY_HASH:
        return _MODEL_ARCH_BY_HASH[key]

    # ------【核心逻辑】未命中则解析并写入缓存，供后续调用复用 ------
    model_cls_and_arch = _get_model_architecture(model_config)
    _MODEL_ARCH_BY_HASH[key] = model_cls_and_arch
    return model_cls_and_arch


def get_model_cls(model_config: ModelConfig) -> type[nn.Module]:
    return get_model_architecture(model_config)[0]


def get_architecture_class_name(model_config: ModelConfig) -> str:
    return get_model_architecture(model_config)[1]


@dataclass
class ParamMapping:
    """
    A class to handle parameter mapping for model weight loading.
    It creates a bidirectional mapping between packed parameters and their
    constituent parts.
    """

    packed_mapping: dict[str, list[str]]
    inverse_packed_mapping: dict[str, tuple[str, int]] = field(default_factory=dict)

    def __post_init__(self):
        # ------【权重加载】遍历 packed 映射，构建 packed 参数到子参数的切片索引 ------
        for packed_name, sub_params in self.packed_mapping.items():
            # ------【权重加载】跳过自包含项（W_pack 只映射自身），无需建反向索引 ------
            # Skip self-contained cases (e.g., {"W_pack": ["W_pack"]})
            if len(sub_params) == 1 and sub_params[0] == packed_name:
                continue
            # ------【权重加载】记录每个子参数所属 packed 参数及其切片下标，供反向查找 ------
            for index, param_name in enumerate(sub_params):
                self.inverse_packed_mapping[param_name] = (
                    packed_name,
                    index,
                )

    def get_sub_modules(self, module_name: str) -> tuple[str, list[str]] | None:
        # ------【权重加载】按后缀匹配模块名，返回 packed key 与子参数列表 ------
        for key, value in self.packed_mapping.items():
            if module_name.endswith(key):
                return key, value
        return None


def configure_quant_config(
    quant_config: QuantizationConfig, model_class: type[nn.Module]
):
    """
    Pass packed_modules_mapping by reference to quant_config so that
    quant_config can properly match fused modules

    Note that model attributes are passed by reference to quant_config,
    enabling them to be updated by model_class.__new__ (ex. chatglm, qwen)

    Once the `SupportsQuant` mixin has been added to all models, this
    function can be removed
    """
    # ------【量化】仅对未混入 SupportsQuant 的旧式模型做映射注入 ------
    if not issubclass(model_class, SupportsQuant):
        hf_to_vllm_mapper = getattr(model_class, "hf_to_vllm_mapper", None)
        # ------【量化】按引用取模型映射表，使量化配置与模型共享同一对象 ------
        packed_mapping = getattr(model_class, "packed_modules_mapping", None)

        # pass mappings by reference to quant_config
        # ------【量化】把 hf->vllm 未堆叠映射注入量化配置，用于权重名匹配 ------
        if hf_to_vllm_mapper is not None:
            quant_config.apply_vllm_mapper(hf_to_vllm_mapper.get_unstacked_mapper())
        # ------【量化】把 packed_modules_mapping 按引用赋给量化配置，识别融合模块 ------
        if packed_mapping is not None:
            quant_config.packed_modules_mapping = packed_mapping
