# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import collections
import glob
import os
import time
from collections.abc import Generator
from copy import copy
from typing import Any

import torch
from torch import nn

from vllm.config import ModelConfig
from vllm.config.load import LoadConfig
from vllm.logger import init_logger
from vllm.model_executor.model_loader.base_loader import BaseModelLoader
from vllm.model_executor.model_loader.weight_utils import (
    download_weights_from_hf,
    runai_safetensors_weights_iterator,
)
from vllm.transformers_utils.s3_utils import glob as s3_glob
from vllm.transformers_utils.utils import is_s3

logger = init_logger(__name__)


class ShardedStateLoader(BaseModelLoader):
    """
    Model loader that directly loads each worker's model state dict, which
    enables a fast load path for large tensor-parallel models where each worker
    only needs to read its own shard rather than the entire checkpoint. See
    `examples/features/sharded_state/save_sharded_state_offline.py` for creating
    a sharded checkpoint.
    """

    DEFAULT_PATTERN = "model-rank-{rank}-part-{part}.safetensors"

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)

        # ------【权重加载】浅拷贝 extra_config，避免后续 pop 污染原始配置 ------
        extra_config = (
            {}
            if load_config.model_loader_extra_config is None
            else copy(load_config.model_loader_extra_config)
        )
        # ------【TP 权重切分】取出分片文件命名 pattern，其余键视为非法配置 ------
        self.pattern = extra_config.pop("pattern", self.DEFAULT_PATTERN)
        # ------【权重加载】校验没有未识别的 extra_config 键，提前报错 ------
        if extra_config:
            raise ValueError(
                f"Unexpected extra config keys for load format "
                f"{load_config.load_format}: "
                f"{load_config.model_loader_extra_config.keys()}"
            )

    @staticmethod
    def _filter_subtensors(
        tensors: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """
        Filter out all tensors that share the same memory or a subset of the
        memory of another tensor.
        """
        # ------【权重加载】按底层 storage 指针分组，识别共享同一块显存/内存的 tensor ------
        same_storage_groups: dict[Any, list[tuple[str, torch.Tensor]]] = (
            collections.defaultdict(list)
        )
        # ------【权重加载】只统计非空 tensor，用 untyped_storage 数据指针作为分组键 ------
        for key, tensor in tensors.items():
            if tensor.numel():
                ptr = tensor.untyped_storage().data_ptr()
                same_storage_groups[tensor.device, ptr].append((key, tensor))

        # ------【权重加载】计算 tensor 实际占用的内存结束地址，用于判断覆盖范围 ------
        def get_end_ptr(tensor: torch.Tensor) -> int:
            return tensor.view(-1)[-1].data_ptr() + tensor.element_size()

        # ------【权重加载】逐组去重：只保留不被其它 tensor 完全覆盖的那个 ------
        result: dict[str, torch.Tensor] = {}
        # ------【权重加载】两两比较内存区间，过滤掉共享内存的子 tensor ------
        for group in same_storage_groups.values():
            for k, t in group:
                a, b = t.data_ptr(), get_end_ptr(t)
                for k2, t2 in group:
                    # ------【权重加载】只比较连续 tensor 的覆盖范围，非连续直接跳过 ------
                    if not t2.is_contiguous():
                        continue
                    a2, b2 = t2.data_ptr(), get_end_ptr(t2)
                    if a < a2 or b2 < b:
                        continue
                    if a2 < a or b < b2 or not t.is_contiguous():
                        break  # t2 covers strictly more memory than t.
                    if k2 < k:
                        # Same tensors, keep the one with the smaller key.
                        break
                else:
                    result[k] = t
        # ------【权重加载】返回去重后的 tensor 字典（tie weight 等共享内存只留一份） ------
        return result

    def _prepare_weights(self, model_name_or_path: str, revision: str | None):
        # ------【下载缓存】本地目录或 S3 路径直接复用，无需下载 ------
        if is_s3(model_name_or_path) or os.path.isdir(model_name_or_path):
            return model_name_or_path
        else:
            # ------【下载缓存】从 HuggingFace 预取全部 safetensors 权重文件 ------
            allow_patterns = ["*.safetensors"]
            return download_weights_from_hf(
                model_name_or_path,
                self.load_config.download_dir,
                allow_patterns,
                revision,
                ignore_patterns=self.load_config.ignore_patterns,
            )

    def download_model(self, model_config: ModelConfig) -> None:
        # ------【下载缓存】download 阶段仅负责把权重文件落到本地/对象存储 ------
        self._prepare_weights(model_config.model, model_config.revision)

    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None:
        # ------【TP】延迟导入 TP rank 接口，避免非并行场景下引入分布式依赖 ------
        from vllm.distributed import get_tensor_model_parallel_rank

        model_weights = model_config.model
        # ------【核心逻辑】允许用 model_weights 覆盖默认模型路径 ------
        if model_weights_override := model_config.model_weights:
            model_weights = model_weights_override
        local_model_path = model_weights

        # ------【TP 权重切分】按当前 TP rank 拼出本 worker 专属分片文件 pattern ------
        rank = get_tensor_model_parallel_rank()
        pattern = os.path.join(
            local_model_path,
            self.pattern.format(rank=rank, part="*"),
        )

        # ------【TP 权重切分】根据存储类型(S3/本地)枚举本 rank 的所有分片文件 ------
        filepaths = []
        if is_s3(local_model_path):
            file_pattern = f"*{self.pattern.format(rank=rank, part='*')}"
            filepaths = s3_glob(path=local_model_path, allow_pattern=[file_pattern])
        else:
            filepaths = glob.glob(pattern)
        # ------【权重加载】找不到分片文件时报错，提示当前只支持预分片 checkpoint ------
        if not filepaths:
            # TODO: support un-sharded checkpoints too
            raise ValueError(
                f"Could not find checkpoint files '{pattern}', only "
                f"pre-sharded checkpoints are currently supported!"
            )
        # ------【权重加载】先过滤掉共享内存的重复 tensor，得到待填充的 state_dict ------
        state_dict = self._filter_subtensors(model.state_dict())
        # ------【权重加载】记录加载起始时间，用于统计耗时 ------
        counter_before_loading_weights = time.perf_counter()
        for key, tensor in self.iterate_over_files(filepaths):
            # If loading with LoRA enabled, additional padding may
            # be added to certain parameters. We only load into a
            # narrowed view of the parameter data.
            param_data = state_dict[key].data
            param_shape = state_dict[key].shape
            # ------【LoRA】参数可能被额外 padding，只把文件 tensor 拷入前部窄视图 ------
            for dim, size in enumerate(tensor.shape):
                if size < param_shape[dim]:
                    param_data = param_data.narrow(dim, 0, size)
            if tensor.shape != param_shape:
                logger.warning(
                    "loading tensor of shape %s into parameter '%s' of shape %s",
                    tensor.shape,
                    key,
                    param_shape,
                )
            # ------【权重加载】把分片文件里的 tensor 拷进对应参数显存 ------
            param_data.copy_(tensor)
            # ------【权重加载】标记该 key 已加载，剩余未弹出的即缺失键 ------
            state_dict.pop(key)
        counter_after_loading_weights = time.perf_counter()
        # ------【权重加载】打印整体加载耗时，便于性能评估 ------
        logger.info_once(
            "Loading weights took %.2f seconds",
            counter_after_loading_weights - counter_before_loading_weights,
        )
        # ------【权重加载】校验所有参数都已加载，否则报缺失键错误 ------
        if state_dict:
            raise ValueError(f"Missing keys {tuple(state_dict)} in loaded state!")

    def iterate_over_files(
        self, paths
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        # ------【权重加载】runai_streamer 分片走流式迭代器，其余走 safetensors 逐文件读取 ------
        if self.load_config.load_format == "runai_streamer_sharded":
            yield from runai_safetensors_weights_iterator(paths, True)
        else:
            # ------【权重加载】懒加载 safetensors，逐个文件流式 yield 权重张量 ------
            from safetensors.torch import safe_open

            for path in paths:
                with safe_open(path, framework="pt") as f:
                    for key in f.keys():  # noqa: SIM118
                        tensor = f.get_tensor(key)
                        yield key, tensor

    @staticmethod
    def save_model(
        model: torch.nn.Module,
        path: str,
        pattern: str | None = None,
        max_size: int | None = None,
    ) -> None:
        # ------【序列化】懒加载 safetensors 保存与 TP rank 接口 ------
        from safetensors.torch import save_file

        from vllm.distributed import get_tensor_model_parallel_rank

        # ------【TP 权重切分】缺省用默认分片命名 pattern ------
        if pattern is None:
            pattern = ShardedStateLoader.DEFAULT_PATTERN
        rank = get_tensor_model_parallel_rank()
        # ------【TP 权重切分】初始化分片编号与累计字节数，用于按大小切分保存 ------
        part_idx = 0
        total_size = 0
        # ------【权重加载】过滤共享内存的重复 tensor，避免重复落盘 ------
        state_dict = ShardedStateLoader._filter_subtensors(model.state_dict())
        state_dict_part: dict[str, torch.Tensor] = {}
        for key, tensor in state_dict.items():
            param_size = tensor.nelement() * tensor.element_size()
            # ------【TP 权重切分】超过 max_size 就先把当前分片落盘再开新分片 ------
            if max_size is not None and total_size + param_size > max_size:
                filename = pattern.format(rank=rank, part=part_idx)
                save_file(
                    state_dict_part,
                    os.path.join(path, filename),
                )
                part_idx += 1
                total_size = 0
                state_dict_part = {}
            # ------【TP 权重切分】把当前 tensor 归入本分片并累计大小 ------
            state_dict_part[key] = tensor
            total_size += param_size
        # ------【TP 权重切分】保存最后一个非空分片，避免漏写尾部参数 ------
        if len(state_dict_part) > 0:
            filename = pattern.format(rank=rank, part=part_idx)
            save_file(
                state_dict_part,
                os.path.join(path, filename),
            )
