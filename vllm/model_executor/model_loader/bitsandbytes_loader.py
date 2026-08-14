# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import fnmatch
import glob
import itertools
import math
import os
from collections.abc import Callable, Generator
from typing import Any

import numpy as np
import torch
from packaging import version
from torch import nn
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME

from vllm.config import ModelConfig
from vllm.config.load import LoadConfig
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.logger import init_logger
from vllm.lora.utils import is_moe_model
from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.linear import (
    LinearBase,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.model_loader.base_loader import BaseModelLoader
from vllm.model_executor.model_loader.utils import ParamMapping
from vllm.model_executor.model_loader.weight_utils import (
    download_safetensors_index_file_from_hf,
    download_weights_from_hf,
    filter_duplicate_safetensors_files,
    filter_files_not_needed_for_inference,
    pt_weights_iterator,
    safetensors_weights_iterator,
)
from vllm.model_executor.models import is_pooling_model
from vllm.model_executor.utils import (
    get_moe_expert_mapping,
    get_packed_modules_mapping,
    set_weight_attrs,
)
from vllm.platforms import current_platform
from vllm.transformers_utils.repo_utils import hf_api
from vllm.utils.torch_utils import set_default_torch_dtype

logger = init_logger(__name__)


class BitsAndBytesModelLoader(BaseModelLoader):
    """Model loader to load model weights with BitsAndBytes quantization."""

    possible_config_file_names = ["adapter_config.json"]

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)

        # ------【TP 权重切分+量化】预分配分片分类与量化状态容器，供后续加载时填充 ------
        # Save the module names without sharding.
        self.unsharded_weights_modules: list[str] = []
        # Save the module names that are sharded by column.
        self.column_sharded_weights_modules: list[str] = []
        # Modules whose weights might have fused on disk
        # we need their output_sizes to make shard in flight correctly with TP
        self.maybe_fused_weights_modules: dict[str, list[int]] = {}
        # Store all module names (from transformers) that support
        # BNB quantization.
        self.target_modules: list[str] = []
        self.tp_disabled_modules: list[str] = []
        # Store the mapping of expert parameters for MoE models.
        self.expert_params_mapping: list[tuple[str, str, int, str]] = []
        # mapping weight names from transformers to vllm.
        self.weight_mapper: Callable = lambda name: name
        self.pre_quant: bool = False
        self.load_8bit: bool = False
        self.is_pool_model: bool = False

    def _get_weight_files(
        self,
        model_name_or_path: str,
        allowed_patterns: list[str],
        revision: str | None = None,
    ) -> tuple[str, list[str], str]:
        """Retrieve weight files. Download the files if necessary.

        Return the weight files and the file pattern."""
        # ------【权重加载】先判断本地/远程路径，决定 glob 直读还是走 HF 下载 ------
        is_local = os.path.isdir(model_name_or_path)

        if is_local:
            for pattern in allowed_patterns:
                weight_files = glob.glob(os.path.join(model_name_or_path, pattern))
                if weight_files:
                    return model_name_or_path, weight_files, pattern
        else:
            # ------【下载缓存】远程模型先列仓库文件，再按 pattern 过滤并下载到缓存目录 ------
            repo_files = hf_api().list_repo_files(repo_id=model_name_or_path)
            for pattern in allowed_patterns:
                matching_files = fnmatch.filter(repo_files, pattern)
                if matching_files:
                    hf_folder = download_weights_from_hf(
                        model_name_or_path,
                        self.load_config.download_dir,
                        [pattern],
                        revision,
                        ignore_patterns=self.load_config.ignore_patterns,
                    )
                    return (
                        hf_folder,
                        glob.glob(os.path.join(hf_folder, pattern)),
                        pattern,
                    )

        raise RuntimeError(f"No model weights found in: `{model_name_or_path}`")

    def _prepare_weights(
        self, model_name_or_path: str, revision: str | None
    ) -> tuple[list[str], bool]:
        """Prepare weight files for the model."""

        allowed_patterns = ["*.safetensors", "*.bin", "*.pt"]

        hf_folder, hf_weights_files, matched_pattern = self._get_weight_files(
            model_name_or_path, allowed_patterns, revision
        )

        # ------【权重加载】根据命中的 pattern 判断权重格式(safetensors 或 pt/bin) ------
        use_safetensors = matched_pattern == "*.safetensors"
        is_local = os.path.isdir(model_name_or_path)
        index_file = SAFE_WEIGHTS_INDEX_NAME
        # ------【权重加载】safetensors 需先下载 index 并去重，避免分片与合并文件混用 ------
        if use_safetensors:
            # For models like Mistral-7B-Instruct-v0.3
            # there are both sharded safetensors files and a consolidated
            # safetensors file. Using both breaks.
            # Here, we download the `model.safetensors.index.json` and filter
            # any files not found in the index.
            if not is_local:
                download_safetensors_index_file_from_hf(
                    model_name_or_path,
                    index_file,
                    cache_dir=self.load_config.download_dir,
                    revision=revision,
                )
            hf_weights_files = filter_duplicate_safetensors_files(
                hf_weights_files, hf_folder, index_file
            )
        else:
            # ------【权重加载】pt/bin 权重过滤掉推理不需要的文件，减少下载量 ------
            hf_weights_files = filter_files_not_needed_for_inference(hf_weights_files)

        # ------【权重加载】空权重集合直接报错，提前暴露缺失问题 ------
        if len(hf_weights_files) == 0:
            raise RuntimeError(
                f"Cannot find any model weights with `{model_name_or_path}`"
            )

        return hf_weights_files, use_safetensors

    def _hf_weight_iter(self, hf_weights_files, use_safetensors: bool):
        # ------【权重加载】pool 模型需给权重名补 model. 前缀以对齐 checkpoint ------
        def _maybe_pool_model(module_name: str):
            # For pool model, we need to add the prefix `model.`
            # for the weight name if possible.
            if (
                self.is_pool_model
                and self.target_modules[0].startswith("model.")
                and not module_name.startswith("model.")
            ):
                return "model." + module_name

            return module_name

        # ------【权重加载】按格式选择 safetensors 或 pt 权重迭代器 ------
        if use_safetensors:
            iterator = safetensors_weights_iterator(
                hf_weights_files,
                self.load_config.use_tqdm_on_load,
            )
        else:
            iterator = pt_weights_iterator(
                hf_weights_files,
                self.load_config.use_tqdm_on_load,
                self.load_config.pt_load_map_location,
            )
        # ------【权重加载】做 transformers->vllm 名称映射并保留原名，逐条产出 ------
        for org_name, param in iterator:
            # mapping weight names from transformers to vllm while preserving
            # original names.
            mapped_name = self.weight_mapper(org_name)
            mapped_name = _maybe_pool_model(mapped_name)

            yield org_name, mapped_name, param

    def _get_quantized_weights_iterator(
        self,
        model_name_or_path: str,
        revision: str | None,
    ) -> tuple[Generator[tuple[str, torch.Tensor], None, None], dict[str, Any]]:
        """Get an iterator to the model weights with bitsandbytes quantization,
        as well as the quantization state dictionary."""

        # ------【量化】惰性导入 bitsandbytes 并校验最低版本，避免非量化路径也加载 ------
        # only load the bitsandbytes module when needed
        try:
            import bitsandbytes

            if version.parse(bitsandbytes.__version__) < version.parse("0.46.1"):
                raise ImportError(
                    "bitsandbytes version is wrong. Please "
                    "install bitsandbytes>=0.46.1."
                )
        except ImportError as err:
            raise ImportError(
                "Please install bitsandbytes>=0.46.1 via "
                "`pip install bitsandbytes>=0.46.1` to use "
                "bitsandbytes quantizer."
            ) from err

        # ------【权重加载】先准备权重文件列表，再按是否预量化分发到不同生成器 ------
        hf_weights_files, use_safetensors = self._prepare_weights(
            model_name_or_path, revision
        )

        quant_state_dict: dict[str, Any] = {}

        # ------【量化】按预量化(8bit/4bit)还是运行时量化分发到对应生成器 ------
        if self.pre_quant:
            if self.load_8bit:
                return self._quantized_8bit_generator(
                    hf_weights_files, use_safetensors, quant_state_dict
                ), quant_state_dict
            else:
                return self._quantized_4bit_generator(
                    hf_weights_files, use_safetensors, quant_state_dict
                ), quant_state_dict

        return self._unquantized_generator(
            hf_weights_files, use_safetensors, quant_state_dict
        ), quant_state_dict

    def _is_8bit_weight_name(self, weight_name: str):
        # ------【量化】通过后缀识别 8bit 量化元数据权重(.scb/.weight_format) ------
        quantized_suffix = {".scb", ".weight_format"}
        return any(weight_name.lower().endswith(suffix) for suffix in quantized_suffix)

    def _is_4bit_weight_name(self, weight_name: str):
        # ------【量化】通过后缀识别 4bit 量化元数据权重(absmax/quant_map 等) ------
        quantized_suffix = {
            "absmax",
            "quant_map",
            "nested_absmax",
            "nested_quant_map",
            "bitsandbytes",
        }
        suffix = weight_name.split(".")[-1]
        return any(q_suffix in suffix for q_suffix in quantized_suffix)

    def _quantized_8bit_generator(
        self, hf_weights_files, use_safetensors, quant_state_dict
    ) -> Generator:
        # ------【量化】第一遍遍历：把 .scb 量化状态按 .weight 键收集到 quant_state_dict ------
        for (
            org_weight_name,
            mapped_weight_name,
            weight_tensor,
        ) in self._hf_weight_iter(hf_weights_files, use_safetensors):
            if not mapped_weight_name.lower().endswith(".scb"):
                continue

            weight_key = mapped_weight_name.lower().replace(".scb", ".weight")
            quant_state_dict[weight_key] = weight_tensor

        # ------【量化】第二遍遍历：跳过量化元数据，为带状态的权重打 load_in_8bit 标记 ------
        for (
            org_weight_name,
            mapped_weight_name,
            weight_tensor,
        ) in self._hf_weight_iter(hf_weights_files, use_safetensors):
            if self._is_8bit_weight_name(mapped_weight_name):
                continue

            if mapped_weight_name in quant_state_dict:
                set_weight_attrs(weight_tensor, {"load_in_8bit": True})
                yield org_weight_name, weight_tensor
            else:
                yield org_weight_name, weight_tensor

    def _quantized_4bit_generator(
        self, hf_weights_files, use_safetensors, quant_state_dict
    ) -> Generator:
        from bitsandbytes.functional import QuantState

        # ------【量化】第一遍遍历：收集 4bit 量化状态权重，bitsandbytes 元数据需留在 CPU ------
        # First iterate over all quant state weights
        weight_iterator = self._hf_weight_iter(hf_weights_files, use_safetensors)
        temp_state_dict = {}
        for (
            org_weight_name,
            mapped_weight_name,
            weight_tensor,
        ) in weight_iterator:
            if not self._is_4bit_weight_name(mapped_weight_name):
                continue
            # bitsandbytes library requires
            # weight.quant_state.bitsandbytes__* in CPU
            if "quant_state.bitsandbytes" in mapped_weight_name:
                temp_state_dict[mapped_weight_name] = weight_tensor.cpu().data
            else:
                temp_state_dict[mapped_weight_name] = weight_tensor

        # ------【量化】闭包解析单个权重的量化状态，把相关元数据组装成 QuantState ------
        # Closure to parse quant_state for each prequant weight
        def _parse_quant_state(param_name: str, temp_state_dict: dict) -> QuantState:
            quant_state = {}
            for k in temp_state_dict:
                if param_name + "." in k:
                    quant_state[k] = temp_state_dict[k]

            return QuantState.from_dict(
                quant_state, device=current_platform.device_type
            )

        # ------【量化】第二遍遍历：为带状态的权重解析 QuantState 并随权重一起产出 ------
        # Second iterate over all prequant and normal weights
        # pre quantized weights would have a quant_state
        for (
            org_weight_name,
            mapped_weight_name,
            weight_tensor,
        ) in self._hf_weight_iter(hf_weights_files, use_safetensors):
            if self._is_4bit_weight_name(mapped_weight_name):
                continue

            if (
                f"{mapped_weight_name}.quant_state.bitsandbytes__nf4" in temp_state_dict
            ) or (
                f"{mapped_weight_name}.quant_state.bitsandbytes__fp4" in temp_state_dict
            ):
                quant_state = _parse_quant_state(mapped_weight_name, temp_state_dict)
                quant_state_dict[mapped_weight_name] = quant_state
                yield org_weight_name, weight_tensor
            else:
                yield org_weight_name, weight_tensor

    def _unquantized_generator(
        self, hf_weights_files, use_safetensors, quant_state_dict
    ) -> Generator:
        from bitsandbytes.functional import quantize_4bit

        # ------【TP 权重切分】读取全局 TP 规模与 rank，用于运行时量化的行/列分片 ------
        global_tp_size = get_tensor_model_parallel_world_size()
        global_tp_rank = get_tensor_model_parallel_rank()
        # ------【TP 权重切分】构造权重名匹配辅助函数，去掉 .weight 后缀与模块名对齐 ------
        check_match = (
            lambda weight_name, module_name: weight_name.removesuffix(".weight")
            == module_name
        )
        for (
            org_weight_name,
            mapped_weight_name,
            weight_tensor,
        ) in self._hf_weight_iter(hf_weights_files, use_safetensors):
            # ------【TP 权重切分】禁用 TP 的模块覆盖为单卡参数，避免被错误切分 ------
            # override tp_size and tp_rank if the module has disabled TP
            if any(
                tp_disabled_module in mapped_weight_name
                for tp_disabled_module in self.tp_disabled_modules
            ):
                tp_size = 1
                tp_rank = 0
            else:
                tp_size = global_tp_size
                tp_rank = global_tp_rank

            # ------【量化+TP 权重切分】仅对目标模块的 .weight 做运行时 4bit 量化与分片 ------
            if any(
                target_module in mapped_weight_name
                for target_module in self.target_modules
            ) and mapped_weight_name.endswith(".weight"):
                # ------【TP 权重切分】无分片模块：整块权重原样加载 ------
                # Without sharding
                if any(
                    check_match(mapped_weight_name, module)
                    for module in self.unsharded_weights_modules
                ):
                    weight_sub_tensor = weight_tensor
                # ------【TP 权重切分】按列(最后一维)切分权重，每卡取自己的区间 ------
                # Shard by column
                elif any(
                    check_match(mapped_weight_name, module)
                    for module in self.column_sharded_weights_modules
                ):
                    total_size = weight_tensor.size(-1)
                    start_index = total_size // tp_size * tp_rank
                    end_index = total_size // tp_size * (tp_rank + 1)
                    weight_sub_tensor = weight_tensor[..., start_index:end_index]
                # ------【TP 权重切分】磁盘已融合权重：按输出尺寸累加区间、切片重排后拼回 ------
                # Weights have fused on disk. In this case, we assume that the
                # weight and module use same name.
                elif any(
                    check_match(mapped_weight_name, module)
                    for module in self.maybe_fused_weights_modules
                ):
                    # special case for fused weights
                    # get the size of each shard weight tensor
                    total_shard_sizes = next(
                        (
                            sizes
                            for module, sizes in self.maybe_fused_weights_modules.items()  # noqa: E501
                            if check_match(mapped_weight_name, module)
                        )
                    )
                    total_size = weight_tensor.size(0)
                    assert total_size == sum(total_shard_sizes)
                    # get the start/end index of each shard weight tensor
                    total_start_index = list(
                        itertools.accumulate([0] + total_shard_sizes)
                    )[:-1]
                    shard_weights_index = [
                        (
                            idx + size // tp_size * tp_rank,
                            idx + size // tp_size * (tp_rank + 1),
                        )
                        for idx, size in zip(total_start_index, total_shard_sizes)
                    ]
                    # slice and reorder the weight tensor
                    weight_tensor = [
                        weight_tensor[start_index:end_index, ...]
                        for start_index, end_index in shard_weights_index
                    ]
                    weight_sub_tensor = torch.cat(weight_tensor, dim=0)
                # ------【TP 权重切分】按行(第一维)切分权重，每卡取自己的区间 ------
                # Shard by row
                else:
                    total_size = weight_tensor.size(0)
                    start_index = total_size // tp_size * tp_rank
                    end_index = total_size // tp_size * (tp_rank + 1)
                    weight_sub_tensor = weight_tensor[start_index:end_index, ...]

                # ------【量化】把分片后的权重搬到 GPU，bitsandbytes 量化要求数据在显存 ------
                # bitsandbytes requires data in GPU
                if weight_sub_tensor.is_cuda:
                    loaded_weight = weight_sub_tensor
                else:
                    loaded_weight = weight_sub_tensor.to(
                        device=current_platform.device_type
                    )

                # ------【量化】保证权重连续，规避 bitsandbytes 非连续张量的已知问题 ------
                # remove the following after the issue is fixed:
                # https://github.com/bitsandbytes-foundation/bitsandbytes/issues/1342
                if loaded_weight.is_contiguous() is False:
                    loaded_weight = loaded_weight.contiguous()

                # ------【量化】以 float32 执行 nf4 运行时量化，得到权重与 quant_state ------
                with set_default_torch_dtype(torch.float32):
                    processed_weight, quant_state = quantize_4bit(
                        loaded_weight,
                        compress_statistics=True,
                        quant_type="nf4",
                    )

                quant_state_dict[mapped_weight_name] = quant_state
            # ------【权重加载】非目标模块权重不做量化，原样透传 ------
            else:
                processed_weight = weight_tensor
            yield org_weight_name, processed_weight

    def _get_bnb_target_modules(self, model: nn.Module) -> None:
        """
        Identify and collect all modules that support BitsAndBytes
        quantization.
        """
        # ------【权重加载】遍历模型模块，收集支持 BNB 量化的目标模块名集合 ------
        for name, module in model.named_modules():
            # ------【量化】LinearBase 且带量化配置：映射到 transformers 名并登记目标/禁用 TP 模块 ------
            if isinstance(module, LinearBase) and hasattr(
                module.quant_method, "quant_config"
            ):
                if modules_info := self.modules_mapping.get_sub_modules(name):
                    # Map vllm's names to transformers's names.
                    rep_name, sub_modules = modules_info
                    for sub_name in sub_modules:
                        new_name = name.replace(rep_name, sub_name)
                        self.target_modules.append(new_name)
                        if module.disable_tp:
                            self.tp_disabled_modules.append(new_name)
                # Add original module name even if the module has stacked map,
                # in case model has a mixture of disk-merged and disk-split
                # weights with same last name.
                self.target_modules.append(name)
                if module.disable_tp:
                    self.tp_disabled_modules.append(name)
            # ------【量化+EP 权重切分】RoutedExperts 按 expert 映射登记各 expert 权重名 ------
            elif isinstance(module, RoutedExperts) and hasattr(
                module.quant_method, "quant_config"
            ):
                # TODO: support RoutedExperts with prequant and 8bit.
                if self.pre_quant and self.load_8bit:
                    raise ValueError(
                        "Prequant BitsAndBytes 8bit models with RoutedExperts "
                        "is not supported yet."
                    )
                # Get the corresponding weight name using module name and
                # expert_params_mapping.

                for exp in self.expert_params_mapping:
                    weight_name = exp[1]
                    rep_name = name.replace("experts", "") + weight_name.removesuffix(
                        "."
                    )
                    self.target_modules.append(rep_name)

        # ------【核心逻辑】未收集到任何目标模块则报错，防止静默走错加载路径 ------
        assert self.target_modules, (
            "vLLM currently does not support BNB quantization for"
        )
        f" {type(model).__name__}"

    def _classify_module_sharding(self, model: nn.Module):
        """
        Categorize modules based on their weight sharding requirements
        for tensor parallelism.
        """
        # ------【TP 权重切分】遍历模块，把权重按 TP 分片需求分成无分片/列分片/融合三类 ------
        for name, module in model.named_modules():
            # Some modules like `ReplicatedLinear` should not have their weights
            # sharded. The reason for implementing it this way is to avoid new
            # static variable in the model implementation.
            # ------【TP 权重切分】复制型线性层不切分，登记为无分片模块 ------
            if isinstance(module, (ReplicatedLinear,)):
                self.unsharded_weights_modules.append(name)
            # `QKVParallelLinear` and `MergedColumnParallelLinear` might have
            # fused weights on disk. We need to use the output sizes of these
            # modules to shard the weights correctly.
            # ------【TP 权重切分】QKV/合并列并行层可能在磁盘融合，记录 output_sizes 供运行时切分 ------
            elif isinstance(module, (QKVParallelLinear, MergedColumnParallelLinear)):
                self.maybe_fused_weights_modules[name] = module.output_sizes
            # ------【TP 权重切分】RowParallelLinear 权重沿最后一维切分，登记为列分片模块 ------
            # In TP, these weights are partitioned along the column
            # dimension (dim=-1)
            elif isinstance(module, (RowParallelLinear,)):
                self.column_sharded_weights_modules.append(name)
            # ------【EP 权重切分】MoE expert 的 w2 权重按列分片，登记到列分片集合 ------
            elif isinstance(module, RoutedExperts):
                expert_mapping = self.expert_params_mapping
                for exp in expert_mapping:
                    if exp[-1] == "w2":
                        weight_name = exp[1]
                        rep_name = name.replace(
                            "experts", ""
                        ) + weight_name.removesuffix(".")
                        self.column_sharded_weights_modules.append(rep_name)

    def _verify_model_compatibility(
        self, model: nn.Module, model_config: ModelConfig
    ) -> None:
        """
        Verify that the model is compatible with BitsAndBytes quantization.
        """
        # ------【核心逻辑】校验 load_weights 存在，否则无法按 BNB 路径加载 ------
        if not hasattr(model, "load_weights"):
            raise AttributeError(
                "The required method 'load_weights' is not defined in class"
                f" {type(model).__name__}."
            )

        # ------【核心逻辑】校验 packed_modules_mapping 存在，判断模型是否支持 BNB 量化 ------
        if not hasattr(model, "packed_modules_mapping"):
            raise AttributeError(
                f"Model {type(model).__name__} does not support BitsAndBytes "
                "quantization yet. No 'packed_modules_mapping' found."
            )

        # ------【量化】读取 hf 量化配置，识别 bitsandbytes 预量化并设置 pre_quant 标志 ------
        quant_config = getattr(model_config.hf_config, "quantization_config", None)
        if quant_config and (quant_method := quant_config.get("quant_method")):
            if quant_method == "bitsandbytes":
                self.pre_quant = True
            else:
                raise ValueError(
                    f"BitsAndBytes loader does not support {quant_method} quantization"
                )

        # The quant_states in pre_quantized models cannot work with a split
        # weight tensor. So TP does not work with pre_quantized bnb models.
        # ------【TP 权重切分】预量化的量化状态无法与切分权重配合，禁止与 TP 同用 ------
        if self.pre_quant and get_tensor_model_parallel_world_size() > 1:
            raise ValueError(
                "Prequant BitsAndBytes models with tensor parallelism is not "
                "supported. Please try with pipeline parallelism."
            )
        # ------【量化】从配置读取是否 8bit，决定后续走 8bit 还是 4bit 生成器 ------
        if quant_config and self.pre_quant:
            self.load_8bit = quant_config.get("load_in_8bit", False)

    def _initialize_loader_state(
        self, model: nn.Module, model_config: ModelConfig
    ) -> None:
        """
        Initialize the loader's internal state based on the model and
        configuration.
        """
        # ------【权重加载】判断是否 pooling 模型，影响后续权重名前缀映射 ------
        self.is_pool_model = is_pooling_model(model)
        self.modules_mapping = ParamMapping(get_packed_modules_mapping(model))

        # ------【EP 权重切分】MoE 模型记录 expert 参数映射，用于量化状态融合与分片 ------
        if is_moe_model(model):
            self.expert_params_mapping = get_moe_expert_mapping(model)
        # For some models like Molmo, we need to use hf_to_vllm_mapper
        # to ensure correct loading of weights.
        # ------【权重加载】有 hf_to_vllm_mapper 时用它做名称映射，保证权重名对齐 ------
        if hf_to_vllm_mapper := getattr(model, "hf_to_vllm_mapper", None):
            unstacked_mapper = hf_to_vllm_mapper.get_unstacked_mapper()
            self.weight_mapper = lambda name, m=unstacked_mapper: m._map_name(name)

        # ------【量化】收集 BNB 目标模块并分类其 TP 分片方式，完成加载器状态初始化 ------
        self._get_bnb_target_modules(model)
        self._classify_module_sharding(model)

    def _dequantize_dq(self, quant_states: Any):
        """
        When BNB employs Double Quantization, we perform the dequantization of
        these constants during weight loading rather than at inference time,
        thereby avoiding this computational overhead during inference. This
        comes at the cost of increased memory usage.
        """
        # ------【量化】导入块级解量化函数，在加载期消除双重量化以省去推理开销 ------
        from bitsandbytes.functional import QuantState, dequantize_blockwise

        # ------【量化】单状态双重量化解量化：把嵌套 absmax 解出并展平成 float32 ------
        def _dequantize_single_state(quant_state):
            """Helper function to dequantize a single QuantState object."""
            if not (isinstance(quant_state, QuantState) and quant_state.nested):
                return

            # Copied from: https://github.com/bitsandbytes-foundation/bitsandbytes/blob/0.45.3/bitsandbytes/functional.py#L1352-#L1356
            absmax = dequantize_blockwise(quant_state.absmax, quant_state.state2)
            absmax += quant_state.offset

            # Ensure float32 dtype
            if absmax.dtype != torch.float32:
                absmax = absmax.float()

            quant_state.absmax = absmax
            quant_state.nested = False
            quant_state.offset = None
            quant_state.state2 = None

        # ------【量化】支持 dict 或单个 QuantState，统一解量化后返回 ------
        if isinstance(quant_states, dict):
            for quant_state in quant_states.values():
                _dequantize_single_state(quant_state)
        else:
            _dequantize_single_state(quant_states)
        return quant_states

    def _fuse_moe_quant_states(self, model: nn.Module, quant_states_dict: dict) -> dict:
        """

        This function consolidates individual expert quantization states into
        fused representations for w13 and w2.
        """
        # ------【量化】导入 QuantState 以在加载期构建融合后的量化状态 ------
        from bitsandbytes.functional import QuantState

        # ------【EP 权重切分】无 expert 映射则直接返回空，跳过 MoE 融合 ------
        if not self.expert_params_mapping:
            return dict()

        # ------【EP 权重切分】遍历 MoE 层，把分散的 expert 量化状态融合成 w13/w2 两份 ------
        expert_mapping = self.expert_params_mapping
        expert_qs_dict = {}
        for name, module in model.named_modules():
            if not isinstance(module, RoutedExperts):
                continue
            # ------【EP 权重切分】按映射逐 expert 取出量化状态并解双重量化，按 w1/w2/w3 分组 ------
            w1_states_lst = []
            w2_states_lst = []
            w3_states_lst = []
            for exp in expert_mapping:
                shard_id = exp[-1]
                if shard_id not in ("w1", "w2", "w3"):
                    raise ValueError(
                        f"shard_id must be ['w1','w2','w3'] but got {shard_id}."
                    )
                layer_prefix = name.split("experts")[0]
                weight_qual_name = layer_prefix + exp[1] + "weight"
                quant_state = self._dequantize_dq(quant_states_dict[weight_qual_name])
                if shard_id == "w1":
                    w1_states_lst.append(quant_state)
                elif shard_id == "w2":
                    w2_states_lst.append(quant_state)
                else:
                    w3_states_lst.append(quant_state)
                del quant_states_dict[weight_qual_name]
            # ------【EP 权重切分】校验三个 shard 数量一致，确保可成对融合 ------
            assert len(w1_states_lst) == len(w2_states_lst) == len(w3_states_lst)
            w13_absmax_lst = []
            w2_absmax_lst = []
            w13_total_dim0 = 0
            w2_total_dim0 = 0
            # ------【EP 权重切分】w1/w3 存储交错排列，交替拼装 absmax 并累加输出维度 ------
            for w1_qs, w2_qs, w3_qs in zip(w1_states_lst, w2_states_lst, w3_states_lst):
                assert w1_qs.shape == w3_qs.shape
                assert w1_qs.blocksize == w2_qs.blocksize == w3_qs.blocksize
                assert w1_qs.dtype == w2_qs.dtype == w3_qs.dtype
                # w1 and w3 are interleaved in storage
                w13_absmax_lst.append(w1_qs.absmax)
                w13_absmax_lst.append(w3_qs.absmax)
                w2_absmax_lst.append(w2_qs.absmax)
                w13_total_dim0 += w1_qs.shape[0] + w3_qs.shape[0]
                w2_total_dim0 += w2_qs.shape[0]

            # ------【EP 权重切分】沿维度 0 拼接出 w13 与 w2 的融合 absmax ------
            w13_absmax = torch.cat(w13_absmax_lst)
            w2_absmax = torch.cat(w2_absmax_lst)
            # ------【EP 权重切分】用 w1 的元信息构造 w13 融合量化状态(shape/code/blocksize) ------
            # Create fused quantization state for w13.
            w13_qs = QuantState(
                absmax=w13_absmax,
                shape=(w13_total_dim0, w1_states_lst[0].shape[1]),
                code=w1_states_lst[0].code,
                blocksize=w1_states_lst[0].blocksize,
                quant_type="nf4",
                dtype=w1_states_lst[0].dtype,
            )
            # ------【EP 权重切分】同理构造 w2 融合量化状态 ------
            # Create fused quantization state for w2.
            w2_qs = QuantState(
                absmax=w2_absmax,
                shape=(w2_total_dim0, w2_states_lst[0].shape[1]),
                code=w2_states_lst[0].code,
                blocksize=w2_states_lst[0].blocksize,
                quant_type="nf4",
                dtype=w2_states_lst[0].dtype,
            )
            # ------【EP 权重切分】以 .w13_weight/.w2_weight 命名，与 BitsAndBytesMoEMethod 对齐 ------
            # The weight suffixes .w13_weight and .w2_weight are consistent
            # with the param in BitsAndBytesMoEMethod.
            w13_weight_name = name + ".w13_weight"
            w2_weight_name = name + ".w2_weight"
            expert_qs_dict[w13_weight_name] = w13_qs
            expert_qs_dict[w2_weight_name] = w2_qs
        return expert_qs_dict

    def _stack_quantization_states(
        self, model: nn.Module, quant_state_dict: dict
    ) -> dict[str, dict[int, Any]]:
        stacked_quant_state_dict: dict[str, dict[int, Any]] = {}
        # ------【PP】导入 PP 缺失参数判断，跳过当前流水线 rank 不持有的参数 ------
        # TODO: Change this lazy import to normal import
        # after the checks are updated to run on a new version
        from vllm.model_executor.models.utils import is_pp_missing_parameter

        # ------【权重加载】缓存命名参数字典，用于判断量化状态是否需重命名或丢弃 ------
        param_dict = dict(model.named_parameters())
        # ------【权重加载】遍历量化状态，按打包模块反查确定 shard 索引并堆叠 ------
        for quant_param_name in quant_state_dict:
            # ------【PP】跳过当前流水线 rank 不持有的参数，其量化状态不处理 ------
            if is_pp_missing_parameter(quant_param_name, model):
                continue

            non_stacked_param_name = quant_param_name

            # ------【TP 权重切分】用 inverse_packed_mapping 反查，把打包名还原并记录 shard 索引 ------
            shard_index = 0
            for shard_name, (
                weight_name,
                index,
            ) in self.modules_mapping.inverse_packed_mapping.items():
                # Some models, such as MiniCPM V2.5/2.6, contain both
                # module names 'kv_proj' and 'qkv_proj'. To prevent 'kv_proj'
                # from being incorrectly identified as being present in
                # 'vpm.encoder.layers.0.self_attn.qkv_proj.weight
                shard_pos = quant_param_name.find(shard_name)
                can_correct_rename = (shard_pos > 0) and (
                    quant_param_name[shard_pos - 1] == "."
                )
                # If the quant_param_name is packed, it won't occur in the
                # param_dict before renaming.
                new_quant_param_name = quant_param_name.replace(shard_name, weight_name)
                need_rename = (quant_param_name not in param_dict) and (
                    new_quant_param_name in param_dict
                )
                if can_correct_rename and need_rename:
                    shard_index = index
                    quant_param_name = new_quant_param_name
                    break

            # Models like Clip/Siglip may skip some layers in initialization,
            # causing unused quant_param_name in state_dict.
            # ------【核心逻辑】初始化时被跳过的层(如 Clip/Siglip)其量化状态无用，丢弃 ------
            if quant_param_name not in param_dict:
                continue

            # ------【TP 权重切分】按参数名把各 shard 量化状态按索引堆叠进字典 ------
            if quant_param_name not in stacked_quant_state_dict:
                stacked_quant_state_dict[quant_param_name] = {}

            stacked_quant_state_dict[quant_param_name][shard_index] = quant_state_dict[
                non_stacked_param_name
            ]

        # ------【核心逻辑】k_eq_v 模型(如 Gemma4)把 k_proj 的量化状态复制给 v_proj ------
        # repeat k_proj for v_proj for k_eq_v models (e.g. Gemma4)
        config = getattr(model, "config", None)
        if config is not None:
            text_config = config.get_text_config()
            if getattr(text_config, "attention_k_eq_v", False):
                shard_packed = {
                    name
                    for name, subs in self.modules_mapping.packed_mapping.items()
                    if len(subs) == 3
                }
                for param_name, shards in stacked_quant_state_dict.items():
                    is_target = (
                        isinstance(shards, dict)
                        and len(shards) == 2
                        and any(
                            param_name.endswith(f"{p}.weight") for p in shard_packed
                        )
                    )
                    if is_target:
                        assert 1 in shards and 2 not in shards
                        shards[2] = shards[1]

        return stacked_quant_state_dict

    def _bind_quant_states_to_params(
        self, model: nn.Module, stacked_quant_state_dict: dict
    ) -> None:
        # ------【量化】把堆叠好的量化状态与 shard 偏移绑定为参数属性，供推理期使用 ------
        # save quant_states and offsets as the attributes of the parameters
        param_dict = dict(model.named_parameters())
        # ------【量化】遍历参数，命中堆叠量化状态则写入 bnb_quant_state 属性 ------
        for param_name, param in param_dict.items():
            if param_name in stacked_quant_state_dict:
                quant_states = stacked_quant_state_dict[param_name]
                # Dequantize double quantized values during weight loading.
                self._dequantize_dq(quant_states)
                set_weight_attrs(param, {"bnb_quant_state": quant_states})
                if not isinstance(quant_states, dict):
                    continue

                # ------【量化】按 pack_factor 计算各 shard 偏移量，供融合推理定位权重区间 ------
                pack_ratio = getattr(param, "pack_factor", -1)
                if pack_ratio == -1:
                    raise ValueError(f"pack_factor not set for parameter {param_name}.")

                num_elements = [0] * len(quant_states)
                for seq, quant_state in quant_states.items():
                    num_elements[seq] = math.prod(quant_state.shape) // pack_ratio

                offsets = np.concatenate(([0], np.cumsum(num_elements)))
                # Make torch infer_schema happy
                offsets = torch.tensor(offsets).cpu()
                set_weight_attrs(param, {"bnb_shard_offsets": offsets})

                # ------【量化】8bit 预量化还需预分配 matmul_state，供推理期缓存矩阵乘状态 ------
                if self.load_8bit:
                    set_weight_attrs(
                        param, {"matmul_state": [None] * len(quant_states)}
                    )

    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None:
        # ------【核心逻辑】先校验兼容性并初始化加载器状态(目标模块/分片分类) ------
        self._verify_model_compatibility(model, model_config)
        self._initialize_loader_state(model, model_config)

        # ------【核心逻辑】提示进入 BNB 量化权重加载主流程 ------
        logger.info(
            "Loading weights with BitsAndBytes quantization. May take a while ..."
        )
        # ------【量化】获取量化权重迭代器与量化状态字典，驱动后续逐层加载 ------
        qweight_iterator, quant_state_dict = self._get_quantized_weights_iterator(
            model_config.model,
            model_config.revision,
        )
        # ------【权重加载】记录模型全部参数名，加载后比对找出未初始化的权重 ------
        weights_to_load = {name for name, _ in model.named_parameters()}
        loaded_weights = model.load_weights(qweight_iterator)
        # Some models may have weights loading tracker unimplemented.
        if loaded_weights is not None:
            weights_not_loaded = weights_to_load - loaded_weights
            if weights_not_loaded:
                raise ValueError(
                    "Following weights were not initialized from "
                    f"checkpoint: {weights_not_loaded}"
                )
        # ------【EP 权重切分】融合 MoE expert 量化状态，并与打包堆叠结果合并 ------
        expert_quant_state_dict = self._fuse_moe_quant_states(model, quant_state_dict)

        stacked_quant_state_dict = self._stack_quantization_states(
            model, quant_state_dict
        )

        stacked_quant_state_dict = {
            **expert_quant_state_dict,
            **stacked_quant_state_dict,
        }
        # ------【量化】把融合+堆叠后的量化状态绑定到参数，完成加载 ------
        self._bind_quant_states_to_params(model, stacked_quant_state_dict)
        # ------【显存 profiling】加载完成后清理缓存，回收中间张量占用的显存 ------
        torch.accelerator.empty_cache()

    def download_model(self, model_config: ModelConfig) -> None:
        # ------【下载缓存】仅下载/准备权重文件，不真正加载到内存 ------
        self._prepare_weights(model_config.model, model_config.revision)
