# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import inspect
from collections.abc import Callable
from functools import wraps
from weakref import WeakKeyDictionary, WeakSet

import torch

from vllm.config import ModelConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.attention import Attention, MLAAttention
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from .meta import (
    SKIP_LOAD_TENSORS,
    capture_layer_to_meta,
    get_numel_loaded,
    materialize_layer,
    restore_layer_on_meta,
)
from .types import LayerReloadingInfo
from .utils import (
    get_info_size,
    get_layer_params_buffers,
    get_layer_size,
    get_layer_tensors,
    has_device_tensors,
)

logger = init_logger(__name__)

__all__ = [
    "get_layerwise_info",
    "record_metadata_for_reloading",
    "initialize_layerwise_reload",
    "finalize_layerwise_processing",
    "finalize_layerwise_reload",
]


# Global dict storing information used for layerwise restoring, loading, and processing.
# For more information regarding what info is stored when, see `LayerReloadingInfo`
#
# Use a weak ref dictionary so that modules can be freed when the model is freed.
# Values are sanitized from references to the layer key in order to avoid circular refs
LAYERWISE_INFO: WeakKeyDictionary[torch.nn.Module, LayerReloadingInfo] = (
    WeakKeyDictionary()
)

# Global set used to track loading for logging purposes only
LOADING_LAYERS: WeakSet[torch.nn.Module] = WeakSet()


def get_layerwise_info(layer: torch.nn.Module) -> LayerReloadingInfo:
    """
    Get information related to restoring and layerwise processing. If no previous
    information existed, a new entry is constructed
    """
    # ------【层式加载】懒创建该层的 LayerReloadingInfo，用弱引用避免循环引用导致无法回收 ------
    if layer not in LAYERWISE_INFO:
        LAYERWISE_INFO[layer] = LayerReloadingInfo(
            restore_metadata=({}, {}),
            restore_device=torch.get_default_device(),
        )

    return LAYERWISE_INFO[layer]


def record_metadata_for_reloading(model: torch.nn.Module):
    """
    Record layer metadata needed for later reloading.

    Stores parameter and buffer metadata as meta tensors for restoration.
    Must be called before `initialize_layerwise_reload`.
    """
    # ------【meta 设备】逐层捕获 meta 张量快照与目标设备，作为后续层式恢复的依据 ------
    for layer in model.modules():
        info = get_layerwise_info(layer)
        info.restore_metadata = capture_layer_to_meta(layer)
        info.restore_device = torch.get_default_device()


@torch.no_grad()
def initialize_layerwise_reload(model: torch.nn.Module):
    """
    Set up layerwise weight loading with deferred processing.

    Must be called after `record_metadata_for_reloading`. This function:
    1. Saves current kernel tensors for later copying
    2. Restores layer parameters/buffers from metadata (on meta device)
    3. Wraps weight loaders to defer processing until all weights are loaded

    When all weights for a layer are loaded, the wrapped loaders will:
    1. Materialize the layer onto the target device
    2. Load all cached weights
    3. Run quantization processing if applicable
    4. Copy processed values back to original tensor storage
    """
    # disable torchao reloading to avoid infinite recursion
    # ------【量化】临时关闭 torchao 重载标志，避免层式加载期间递归触发自身 ------
    model._original_do_torchao_reload = getattr(model, "_do_torchao_reload", False)
    model._do_torchao_reload = False

    for layer in model.modules():
        info = get_layerwise_info(layer)

        # Skip if the layer has already been initialized
        # ------【层式加载】已初始化的层跳过，避免重复包装其 weight_loader ------
        if info.can_load():
            continue

        # Save current tensors for later copying
        # ------【层式加载】保存当前 kernel 张量，供加载完成后拷回原始存储 ------
        info.kernel_tensors = get_layer_params_buffers(layer)
        # snapshot now: restore_layer_on_meta drops alias buffers from the live set
        # ------【层式加载】快照非持久 buffer 集合，防止恢复过程丢失别名 buffer ------
        info.kernel_non_persistent_buffers = set(layer._non_persistent_buffers_set)

        # Restore layer parameters/buffers onto meta device
        # ------【meta 设备】把层恢复到 meta 设备，释放显存等待重新加载权重 ------
        restore_layer_on_meta(layer, info)

        # Wrap weight loaders to buffer loading
        # ------【量化】包装 weight_loader 为在线处理，延迟到整层权重加载完再量化 ------
        initialize_online_processing(layer)


def initialize_online_processing(layer: torch.nn.Module):
    """
    Wrap a layer's weight loaders with online processing loaders.
    Called by either `initialize_layerwise_reload` or an online quantization scheme,
    prevents double wrapping in the case of online quantization + reloading

    Args:
        layer: layer whose parameter weight loaders will be wrapped
    """
    info = get_layerwise_info(layer)

    # Track loading progress to determine when to process/copy
    # ------【层式加载】重置加载计数并统计本层应加载元素总数，作为触发处理阈值 ------
    info.load_numel = 0
    info.load_numel_total = get_layer_size(layer)
    # ------【层式加载】包装所有参数的 weight_loader，进入在线缓冲模式 ------
    _wrap_parameters_weight_loader(layer)


def _wrap_parameters_weight_loader(layer: torch.nn.Module) -> None:
    """Wrap each parameter's weight loader."""
    # Note that nested wrapping will occur for shared tensors
    for name, tensor in get_layer_tensors(layer).items():
        # ------【EP 权重切分】跳过 expert_map 等从不走 weight_loader 的张量，避免误计数 ------
        if name in SKIP_LOAD_TENSORS:
            continue
        # ------【层式加载】仅包装尚未包装的 loader，避免共享张量被重复嵌套 ------
        if _get_weight_loader(tensor).__name__ != "online_process_loader":
            tensor.weight_loader = make_online_process_loader(layer, name)


def make_online_process_loader(layer: torch.nn.Module, param_name: str) -> Callable:
    """Create a wrapped weight loader that defers processing."""
    info = get_layerwise_info(layer)
    param = getattr(layer, param_name)
    original_loader = _get_original_loader(param)
    # ------【层式加载】取得原始 loader 的签名，用于延迟阶段绑定参数并规范化调用 ------
    loader_signature = inspect.signature(original_loader)

    @wraps(original_loader, assigned=("__doc__", "__annotations__"))
    def online_process_loader(*args, **kwargs):
        # ------【层式加载】层已处理完仍收到加载请求（如 qkv 共享权重），直接丢弃避免重复写 ------
        if not info.can_load():
            # Unfortunately, some qconfigs are set up to load the same weight
            # multiple times. For example, CT_WNA16 loads `weight_shape` for
            # each of the qkv partitions. This results in layers loading extra
            # weights (beyond load_numel_total) after it's already processed.
            #
            # Best solution is to ensure that `load_numel_total` reflects the
            # actual number of weights loaded, either by modifying qconfigs to
            # create as many weights as loaded (see padding issue as well)
            # or maybe capturing how many weights are loaded on first pass
            #
            # For now, `load_numel_total` is still safe to use as long as
            # there's no way to reach `load_numel_total` without loading all
            # necessary weights. `weight_shape` is very small, so this is safe.
            # see Limitations(4)
            logger.debug("%s: Excessive loading", layer.__class__.__name__)
            return

        # Re-run on each load: layers may register parameters later (e.g., `bias`).
        # Wrap late parameters and refresh `load_numel_total` so processing waits
        # until all parameters are loaded.
        # ------【层式加载】刷新总数并补包后注册的参数（如 bias），确保等到全部参数加载完 ------
        info.load_numel_total = get_layer_size(layer)
        _wrap_parameters_weight_loader(layer)

        # Bind and normalize arguments
        # ------【核心逻辑】绑定实参并补全默认值，得到规范化后的加载参数 ------
        bound_args = loader_signature.bind(*args, **kwargs)
        bound_args.apply_defaults()

        # Buffer loaded weights, track loading progress
        # ------【层式加载】缓存本次加载的权重参数，并累计已加载元素数作为触发依据 ------
        info.loaded_weights.append((param_name, bound_args))
        num_loaded, ret = get_numel_loaded(original_loader, bound_args)
        info.load_numel += num_loaded

        logger.debug(
            "%s: %d / %d",
            layer.__class__.__name__,
            info.load_numel,
            info.load_numel_total,
        )

        # Do not online process attention layers, must wait until finalize
        # ------【层式加载】attention 层跳过在线处理，留到 finalize 阶段集中处理 ------
        if isinstance(layer, (Attention, MLAAttention)):
            return ret

        # Log warnings allocating excessive buffers on device
        # ------【显存 profiling】检测同时在显存上缓冲多个层，提示按层排序权重可省显存 ------
        if has_device_tensors(bound_args):
            LOADING_LAYERS.add(layer)
            if len(LOADING_LAYERS) >= 2:
                names = sorted([layer.__class__.__name__ for layer in LOADING_LAYERS])
                mem_used = sum(
                    get_info_size(LAYERWISE_INFO[layer]) for layer in LOADING_LAYERS
                )
                logger.warning_once(
                    "Allocating %.1f MB of device memory to buffers to load %s layers. "
                    "This extra memory usage can be avoided by ordering weights "
                    "by their parent layer when reloading.",
                    mem_used / 1e6,
                    str(list(names)),
                )

        # Process and copy when all weights are loaded
        # ------【层式加载】整层权重加载完成时触发量化处理并拷回 kernel 张量存储 ------
        if info.load_numel >= info.load_numel_total:  # type: ignore[operator]
            _layerwise_process(layer, info)
            LOADING_LAYERS.discard(layer)

        return ret

    return online_process_loader


def finalize_layerwise_processing(model: torch.nn.Module, model_config: ModelConfig):
    """
    Apply processing to any layers which were not layerwise processed during loading.
    This includes attention layers and layers which have weight elements which are not
    loaded (due to padding).

    This function should be applied after `initialize_layerwise_reload` is applied
    unwrap the layerwise weight loaders.

    Args:
        model: model to finalize processing for
        model_config: config needed for applying processing to attention layers
    """
    # ------【量化】恢复 torchao 重载标志，结束层式重载期间的临时关闭 ------
    if hasattr(model, "_original_do_torchao_reload"):
        model._do_torchao_reload = model._original_do_torchao_reload

    deferred_attn: list[tuple[torch.nn.Module, LayerReloadingInfo]] = []

    for layer in model.modules():
        info = get_layerwise_info(layer)
        # ------【层式加载】未参与层式加载的层直接复位，保持状态一致 ------
        if not info.can_load():
            info.reset()
            continue

        # Attention/MLA layers are processed after all other layers
        # ------【层式加载】attention 层延迟到 linear 之后处理，因其依赖前序层输出 ------
        if isinstance(layer, (Attention, MLAAttention)):
            deferred_attn.append((layer, info))
            continue

        # No weights were loaded
        # ------【层式加载】无权重加载时按首载/重载两种情况分别兜底处理 ------
        if info.load_numel <= 0:
            # first load: checkpoint did not contain weights for this layer
            # ------【层式加载】首次加载且 checkpoint 无该层权重，直接完成一次处理 ------
            if info.kernel_tensors is None:
                _layerwise_process(layer, info)
                continue

            # reloading: place kernel tensors back as a fallback. Always place, even
            # when nothing is loadable (load_numel_total == 0), so parameter-alias
            # buffers on such layers are restored rather than left deleted.
            # ------【层式加载】重载失败时回退放置 kernel 张量，恢复别名 buffer 避免被删除 ------
            if info.load_numel_total > 0:  # type: ignore[operator]
                logger.warning("%s: Failed to load weights", layer.__class__.__name__)
            _place_kernel_tensors(layer, info)

        # Process non-attention layers which did not load all elements. This can happen
        # if the created weight has extra padding elements which are not loaded
        # Having too many of these delayed layers can lead to excess memory usage
        # see Limitations(4)
        # ------【量化】部分加载（padding 权重未加载全）的层延迟到此集中处理 ------
        elif info.load_numel > 0 and info.load_numel < info.load_numel_total:  # type: ignore[operator]
            logger.debug("%s: Delayed processing", layer.__class__.__name__)
            _layerwise_process(layer, info)

        info.reset()

    # Process attention layers after all other layers are done
    # ------【层式加载】最后统一处理延迟的 attention 层，完成其权重后处理 ------
    for layer, info in deferred_attn:
        _finalize_attention_layer(layer, info, model_config)
        info.reset()

    # ------【显存 profiling】清空加载日志集合，结束本次重载的显存统计 ------
    LOADING_LAYERS.clear()


def finalize_layerwise_reload(*args, **kwargs):
    # ------【核心逻辑】向后兼容别名，直接转发到统一的层式处理收尾逻辑 ------
    finalize_layerwise_processing(*args, **kwargs)


def _finalize_attention_layer(
    layer: torch.nn.Module, info: LayerReloadingInfo, model_config: ModelConfig
) -> None:
    # ------【量化】重载场景：先放回 kernel 张量再从 checkpoint 重载 attention scale 权重 ------
    if info.load_numel > 0 and info.kernel_tensors is not None:
        # Reload with new scale weights from checkpoint
        _place_kernel_tensors(layer, info)
        _reload_attention_scales(layer, info)
    # ------【层式加载】不支持 attention 层单独层式加载，报错提示须等 linear 之后处理 ------
    elif info.load_numel > 0 or info.kernel_tensors is None:
        raise ValueError(
            "Layerwise loading of attention layers is not supported. "
            "Attention must always process after linears."
        )
    else:
        # ------【层式加载】无新权重时直接放回 kernel 张量保持原状 ------
        _place_kernel_tensors(layer, info)
    # ------【量化】对 attention 层执行权重后处理（量化/repack） ------
    layer.process_weights_after_loading(model_config.dtype)


def _reload_attention_scales(layer: torch.nn.Module, info: LayerReloadingInfo) -> None:
    """Load and process attention scale weights (k_scale, v_scale, etc.)
    during reload.

    Assumes dtype/shapes of attention tensors do not change during
    processing, since we use .data.copy_() to preserve kernel tensor
    references."""
    # ------【量化】无量化方法时无需重载 scale，直接返回 ------
    quant_method = getattr(layer, "quant_method", None)
    if quant_method is None:
        return

    # Re-create scale Parameters with sentinel values so unloaded scales
    # are correctly detected by process_weights_after_loading
    # ------【量化】用哨兵值重建 scale 参数，让后处理能识别未加载的 scale ------
    quant_method.create_weights(layer)

    # ------【量化】用原始 loader 逐个把缓存的 scale 权重载入重建后的参数 ------
    for name, args in info.loaded_weights:
        param = getattr(layer, name)
        args.arguments["param"] = param
        _get_weight_loader(param)(*args.args, **args.kwargs)

    # ------【量化】对 attention 量化权重执行后处理 ------
    quant_method.process_weights_after_loading(layer)

    # ------【CUDA Graph】把处理后的值拷回 kernel 张量存储，保留 cudagraph 引用 ------
    _copy_and_restore_kernel_tensors(layer, info)


def _layerwise_process(layer: torch.nn.Module, info: LayerReloadingInfo):
    """
    Finalize layer loading after all weights have been buffered.

    This function:
    1. Materializes the layer onto the target device
    2. Loads all buffered weights
    3. Runs quantization processing if applicable
    4. Copies processed values back to original tensor storage
    """
    # Materialize layer tensors onto device
    # ------【meta 设备】把层的 meta 张量物化到目标设备，准备接收真实权重 ------
    materialize_layer(layer, info)

    # Reset online quantization flag so process_weights_after_loading
    # will run again during reload
    # ------【量化】清除"已做后处理"标记，让重载时量化后处理能再次执行 ------
    if hasattr(layer, "_already_called_process_weights_after_loading"):
        delattr(layer, "_already_called_process_weights_after_loading")

    # Unwrap layerwise loading wrappers
    # ------【层式加载】解开层式包装恢复原始 weight_loader，以便直接写权重 ------
    for param in get_layer_tensors(layer).values():
        param.weight_loader = _get_original_loader(param)

    # Load all buffered weights into materialized layer (using original loaders)
    # ------【权重加载】用原始 loader 把缓存的所有权重写进物化后的张量 ------
    for name, args in info.loaded_weights:
        param = getattr(layer, name)
        args.arguments["param"] = param
        param.weight_loader(*args.args, **args.kwargs)

    # Process weights (quantization, repacking, etc.)
    # ------【量化】执行量化后处理（pack/repack），并重对齐 TP 参数状态避免破坏复制权重 ------
    quant_method = getattr(layer, "quant_method", None)
    if isinstance(quant_method, QuantizeMethodBase):
        quant_method.process_weights_after_loading(layer)
        # Re-reconcile parameter TP state: process_weights_after_loading may
        # have re-created Parameters (stamped with the global rank), which would
        # otherwise break replicated (disable_tp) weights on a subsequent reload.
        if hasattr(layer, "update_param_tp_status"):
            layer.update_param_tp_status()

    # Copy processed values into original tensor storage (preserves cudagraph refs)
    # this code is a no-op if not reloading (because kernel tensors is empty)
    # ------【CUDA Graph】把处理后的值拷回 kernel 张量存储，保留 cudagraph 引用 ------
    if info.kernel_tensors is not None:
        _copy_and_restore_kernel_tensors(layer, info)

    info.reset()
    logger.debug("%s: Processed", layer.__class__.__name__)


def _get_original_loader(tensor: torch.Tensor) -> Callable:
    """Return the weight loader with any layerwise wrappers removed"""
    loader = _get_weight_loader(tensor)
    # ------【层式加载】层层剥开 online_process_loader 包装，还原最原始的加载函数 ------
    while loader.__name__ == "online_process_loader":
        loader = loader.__wrapped__  # type: ignore[union-attr]

    return loader


def _get_weight_loader(tensor: torch.Tensor):
    # ------【权重加载】取张量上绑定的 weight_loader，缺省时回退到默认加载器 ------
    return getattr(tensor, "weight_loader", default_weight_loader)


def _copy_and_restore_kernel_tensors(layer: torch.nn.Module, info: LayerReloadingInfo):
    """Copy processed values into original kernel tensor storage and restore
    kernel tensor references on the layer. Preserves cudagraph references."""
    assert info.kernel_tensors is not None
    parameters, buffers = info.kernel_tensors
    non_persistent = info.kernel_non_persistent_buffers
    # ------【CUDA Graph】收集已加载张量名，用于区分哪些 buffer 需要拷回 ------
    loaded_tensor_names = {name for name, _ in info.loaded_weights}
    # ------【CUDA Graph】把处理后的参数值就地拷回原始参数存储，保持底层引用不变 ------
    for name, param in parameters.items():
        param.data.copy_(getattr(layer, name))
    for name, buffer in buffers.items():
        if name not in layer._buffers:
            continue
        # ------【CUDA Graph】跳过纯别名且未加载的 buffer，避免覆盖别名或被删除项 ------
        if name in non_persistent and name not in loaded_tensor_names:
            continue
        buffer.data.copy_(getattr(layer, name))

    # ------【CUDA Graph】把 kernel 张量重新挂回层上，恢复模型运行态引用 ------
    _place_kernel_tensors(layer, info)


def _place_kernel_tensors(layer: torch.nn.Module, info: LayerReloadingInfo):
    # ------【层式加载】先删除层上现有张量，腾出位置以重新挂载 kernel 张量 ------
    for name in get_layer_tensors(layer):
        delattr(layer, name)

    assert info.kernel_tensors is not None
    parameters, buffers = info.kernel_tensors
    non_persistent = info.kernel_non_persistent_buffers
    # ------【层式加载】重新注册参数，恢复原 kernel 参数引用 ------
    for name, param in parameters.items():
        layer.register_parameter(name, param)
    # ------【层式加载】按是否持久重新注册 buffer，还原别名 buffer 的持久属性 ------
    for name, buffer in buffers.items():
        layer.register_buffer(name, buffer, persistent=name not in non_persistent)
