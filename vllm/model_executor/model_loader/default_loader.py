# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import dataclasses
import glob
import os
import time
from collections.abc import Generator, Iterable
from typing import cast

import torch
from torch import nn
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME

from vllm.config import ModelConfig
from vllm.config.load import LoadConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.torchao import torchao_version_at_least
from vllm.model_executor.model_loader.base_loader import BaseModelLoader
from vllm.model_executor.model_loader.ep_weight_filter import (
    compute_local_expert_ids,
)
from vllm.model_executor.model_loader.weight_utils import (
    download_safetensors_index_file_from_hf,
    download_weights_from_hf,
    fastsafetensors_weights_iterator,
    filter_duplicate_safetensors_files,
    filter_files_not_needed_for_inference,
    get_quant_config,
    instanttensor_weights_iterator,
    maybe_download_from_modelscope,
    multi_thread_pt_weights_iterator,
    multi_thread_safetensors_weights_iterator,
    np_cache_weights_iterator,
    pt_weights_iterator,
    safetensors_weights_iterator,
)
from vllm.tracing import instrument
from vllm.transformers_utils.repo_utils import list_filtered_repo_files

logger = init_logger(__name__)


class DefaultModelLoader(BaseModelLoader):
    """Model loader that can load different file types from disk."""

    # default number of thread when enable multithread weight loading
    DEFAULT_NUM_THREADS = 8

    @dataclasses.dataclass
    class Source:
        """A source for weights."""

        model_or_path: str
        """The model ID or path."""

        revision: str | None
        """The optional model revision."""

        subfolder: str | None = None
        """The subfolder inside the model repo."""

        prefix: str = ""
        """A prefix to prepend to all weights."""

        fall_back_to_pt: bool = True
        """Whether .pt weights can be used."""

        allow_patterns_overrides: list[str] | None = None
        """If defined, weights will load exclusively using these patterns."""

    counter_before_loading_weights: float = 0.0
    counter_after_loading_weights: float = 0.0

    def __init__(self, load_config: LoadConfig):
        # ------【EP 权重切分】初始化基类并预留本地 expert id 集合，供 EP 过滤使用 ------
        super().__init__(load_config)
        self.local_expert_ids: set[int] | None = None

        # ------【核心逻辑】读取 loader 附加配置并校验其必须为 dict ------
        extra_config = load_config.model_loader_extra_config
        if not isinstance(extra_config, dict):
            raise ValueError(
                f"model_loader_extra_config must be a dict for load format "
                f"{load_config.load_format}, got {type(extra_config).__name__}"
            )
        # ------【核心逻辑】用白名单校验附加配置键，拒绝未知键 ------
        allowed_keys = {
            "enable_multithread_load",
            "num_threads",
            "enable_weights_track",
        }
        unexpected_keys = set(extra_config.keys()) - allowed_keys

        # ------【核心逻辑】发现未知配置键则报错，防止拼写错误被静默忽略 ------
        if unexpected_keys:
            raise ValueError(
                f"Unexpected extra config keys for load format "
                f"{load_config.load_format}: "
                f"{unexpected_keys}"
            )

        # ------【并行加载】读取并校验多线程加载开关必须为 bool ------
        enable_multithread_load = extra_config.get("enable_multithread_load", False)
        if not isinstance(enable_multithread_load, bool):
            raise ValueError(
                f"enable_multithread_load must be a bool, got "
                f"{type(enable_multithread_load).__name__}"
            )
        # ------【并行加载】校验 num_threads 必须为正整数，非法则报错 ------
        num_threads = extra_config.get("num_threads")
        if num_threads is not None and not (
            isinstance(num_threads, int) and num_threads > 0
        ):
            raise ValueError(
                f"num_threads must be a positive integer, got {num_threads!r}"
            )

        # ------【权重加载】读取权重跟踪开关，None 表示后续走默认策略 ------
        self.enable_weights_track: bool | None = extra_config.get(
            "enable_weights_track", None
        )

        # The multi-thread loader ignores safetensors_load_strategy, so reject
        # the combination instead of silently dropping the requested strategy.
        # ------【并行加载】多线程加载器只支持 lazy 策略，冲突时提前报错避免静默降级 ------
        if extra_config.get("enable_multithread_load") and (
            load_config.safetensors_load_strategy not in (None, "lazy")
        ):
            raise ValueError(
                "enable_multithread_load does not support "
                "safetensors_load_strategy="
                f"{load_config.safetensors_load_strategy!r}; the multi-thread "
                "loader only implements the default lazy strategy."
            )

    def _prepare_weights(
        self,
        model_name_or_path: str,
        subfolder: str | None,
        revision: str | None,
        fall_back_to_pt: bool,
        allow_patterns_overrides: list[str] | None,
    ) -> tuple[str, list[str], bool]:
        """Prepare weights for the model.

        If the model is not local, it will be downloaded."""
        # ------【下载缓存】优先尝试 ModelScope 下载，失败则沿用原始路径 ------
        model_name_or_path = (
            maybe_download_from_modelscope(model_name_or_path, revision)
            or model_name_or_path
        )

        # ------【权重加载】判断本地/远程并初始化 safetensors 标志与索引文件名 ------
        is_local = os.path.isdir(model_name_or_path)
        load_format = self.load_config.load_format
        use_safetensors = False
        index_file = SAFE_WEIGHTS_INDEX_NAME

        # First check for 'auto' format that mistral files format are present.
        # This is to load mistral models with official format by default.
        # ------【核心逻辑】auto 格式下探测 mistral 官方分片以决定实际加载格式 ------
        if load_format == "auto":
            load_format = (
                "mistral"
                if len(
                    list_filtered_repo_files(
                        model_name_or_path=model_name_or_path,
                        allow_patterns=["consolidated*.safetensors"],
                        revision=revision,
                    )
                )
                > 0
                else "hf"
            )

        # Some quantized models use .pt files for storing the weights.
        # ------【权重加载】按 load_format 映射出允许加载的权重文件匹配模式 ------
        if load_format == "hf":
            allow_patterns = ["*.safetensors", "*.bin"]
        elif (
            load_format == "safetensors"
            or load_format == "fastsafetensors"
            or load_format == "instanttensor"
        ):
            use_safetensors = True
            allow_patterns = ["*.safetensors"]
        elif load_format == "mistral":
            use_safetensors = True
            allow_patterns = ["consolidated*.safetensors"]
            index_file = "consolidated.safetensors.index.json"
        elif load_format == "pt":
            allow_patterns = ["*.pt"]
        elif load_format == "npcache":
            allow_patterns = ["*.bin"]
        else:
            raise ValueError(f"Unknown load_format: {load_format}")

        # Don't fall back to .pt for explicit safetensors formats; otherwise a
        # .pt file is matched and later opened as safetensors.
        # ------【权重加载】非 safetensors 格式允许回退到 .pt 权重文件 ------
        if fall_back_to_pt and not use_safetensors:
            allow_patterns += ["*.pt"]

        # ------【权重加载】显式给定 allow_patterns_overrides 时覆盖默认匹配模式 ------
        if allow_patterns_overrides is not None:
            allow_patterns = allow_patterns_overrides

        # ------【下载缓存】远程模型下载权重到本地缓存，本地模型直接复用路径 ------
        if not is_local:
            hf_folder = download_weights_from_hf(
                model_name_or_path,
                self.load_config.download_dir,
                allow_patterns,
                revision,
                subfolder=subfolder,
                ignore_patterns=self.load_config.ignore_patterns,
            )
        else:
            hf_folder = model_name_or_path

        # ------【权重加载】拼接 subfolder 得到权重文件实际所在目录 ------
        if subfolder is not None:
            hf_folder = os.path.join(hf_folder, subfolder)

        # ------【权重加载】按模式 glob 匹配磁盘权重文件，命中 safetensors 则置位标志 ------
        hf_weights_files: list[str] = []
        for pattern in allow_patterns:
            hf_weights_files += glob.glob(os.path.join(hf_folder, pattern))
            if len(hf_weights_files) > 0:
                if pattern.endswith(".safetensors"):
                    use_safetensors = True
                break

        # ------【权重加载】safetensors 场景下载 index 并去重过滤多余分片文件 ------
        if use_safetensors:
            # For models like Mistral-7B-Instruct-v0.3
            # there are both sharded safetensors files and a consolidated
            # safetensors file. Using both breaks.
            # Here, we download the `model.safetensors.index.json` and filter
            # any files not found in the index.
            if not is_local and len(hf_weights_files) > 1:
                download_safetensors_index_file_from_hf(
                    model_name_or_path,
                    index_file,
                    cache_dir=self.load_config.download_dir,
                    subfolder=subfolder,
                    revision=revision,
                )
            hf_weights_files = filter_duplicate_safetensors_files(
                hf_weights_files, hf_folder, index_file
            )
        # ------【权重加载】pt 场景过滤推理不需要的文件（如 optimizer 状态） ------
        else:
            hf_weights_files = filter_files_not_needed_for_inference(hf_weights_files)

        # ------【核心逻辑】未找到任何权重文件则报错，避免进入空加载流程 ------
        if len(hf_weights_files) == 0:
            raise RuntimeError(
                f"Cannot find any model weights with `{model_name_or_path}`"
            )

        return hf_folder, hf_weights_files, use_safetensors

    def _get_weights_iterator(
        self, source: "Source"
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        """Get an iterator for the model weights based on the load format."""
        # ------【权重加载】准备权重文件列表与格式标志，据此选择迭代器类型 ------
        extra_config = self.load_config.model_loader_extra_config
        hf_folder, hf_weights_files, use_safetensors = self._prepare_weights(
            source.model_or_path,
            source.subfolder,
            source.revision,
            source.fall_back_to_pt,
            source.allow_patterns_overrides,
        )
        # ------【权重加载】npcache 格式使用 np 缓存迭代器读取 *.bin 权重 ------
        if self.load_config.load_format == "npcache":
            # Currently np_cache only support *.bin checkpoints
            assert use_safetensors is False
            weights_iterator = np_cache_weights_iterator(
                source.model_or_path,
                self.load_config.download_dir,
                hf_folder,
                hf_weights_files,
                self.load_config.use_tqdm_on_load,
            )
        # ------【权重加载】safetensors 家族按具体格式选择对应迭代器 ------
        elif use_safetensors:
            # ------【并行加载】fastsafetensors 使用 C++ 快速迭代器加速读取 ------
            if self.load_config.load_format == "fastsafetensors":
                weights_iterator = fastsafetensors_weights_iterator(
                    hf_weights_files,
                    self.load_config.use_tqdm_on_load,
                )
            # ------【并行加载】instanttensor 使用即时张量迭代器读取权重 ------
            elif self.load_config.load_format == "instanttensor":
                weights_iterator = instanttensor_weights_iterator(
                    hf_weights_files,
                    self.load_config.use_tqdm_on_load,
                )
            # ------【权重加载】默认 safetensors 按是否开启多线程选择迭代器 ------
            else:
                # ------【并行加载】多线程并行读取 safetensors 权重提升吞吐 ------
                if extra_config.get("enable_multithread_load"):
                    weights_iterator = multi_thread_safetensors_weights_iterator(
                        hf_weights_files,
                        self.load_config.use_tqdm_on_load,
                        max_workers=extra_config.get(
                            "num_threads", self.DEFAULT_NUM_THREADS
                        ),
                    )
                # ------【权重加载+并行加载】单线程迭代器并透传 prefetch 线程/块大小做读盘预取 ------
                else:
                    weights_iterator = safetensors_weights_iterator(
                        hf_weights_files,
                        self.load_config.use_tqdm_on_load,
                        self.load_config.safetensors_load_strategy,
                        local_expert_ids=self.local_expert_ids,
                        safetensors_prefetch_num_threads=(
                            self.load_config.safetensors_prefetch_num_threads
                        ),
                        safetensors_prefetch_block_size=(
                            self.load_config.safetensors_prefetch_block_size
                        ),
                    )
        # ------【权重加载】pt 权重按是否开启多线程选择迭代器 ------
        else:
            # ------【并行加载】多线程并行读取 pt 权重文件 ------
            if extra_config.get("enable_multithread_load"):
                weights_iterator = multi_thread_pt_weights_iterator(
                    hf_weights_files,
                    self.load_config.use_tqdm_on_load,
                    self.load_config.pt_load_map_location,
                    max_workers=extra_config.get(
                        "num_threads", self.DEFAULT_NUM_THREADS
                    ),
                )
            # ------【权重加载】单线程 pt 迭代器并透传 map_location 控制加载位置 ------
            else:
                weights_iterator = pt_weights_iterator(
                    hf_weights_files,
                    self.load_config.use_tqdm_on_load,
                    self.load_config.pt_load_map_location,
                )

        # ------【核心逻辑】首次加载记录起始时间，用于统计整体加载耗时 ------
        if self.counter_before_loading_weights == 0.0:
            self.counter_before_loading_weights = time.perf_counter()
        # Apply the prefix.
        # ------【权重加载】为权重名统一加前缀后作为迭代器产出 ------
        return ((source.prefix + name, tensor) for (name, tensor) in weights_iterator)

    def get_all_weights(
        self,
        model_config: ModelConfig,
        model: nn.Module,
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        # ------【核心逻辑】构造主权重 Source（prefix 为空、回退策略取自模型属性） ------
        primary_weights = DefaultModelLoader.Source(
            model_config.model,
            model_config.revision,
            prefix="",
            fall_back_to_pt=getattr(model, "fall_back_to_pt_during_load", True),
            allow_patterns_overrides=getattr(model, "allow_patterns_overrides", None),
        )
        # ------【权重加载】先产出主模型的全部权重 ------
        yield from self._get_weights_iterator(primary_weights)

        # ------【核心逻辑】读取模型可选的 secondary_weights 附加权重源 ------
        secondary_weights = cast(
            Iterable[DefaultModelLoader.Source],
            getattr(model, "secondary_weights", ()),
        )
        # ------【权重加载】逐个产出附加权重源（如多模态编码器权重） ------
        for source in secondary_weights:
            yield from self._get_weights_iterator(source)

    def download_model(self, model_config: ModelConfig) -> None:
        # ------【下载缓存】仅触发权重下载预取，不实际加载到模型 ------
        self._prepare_weights(
            model_name_or_path=model_config.model,
            subfolder=None,
            revision=model_config.revision,
            fall_back_to_pt=True,
            allow_patterns_overrides=None,
        )

    def _init_ep_weight_filter(self, model_config: ModelConfig) -> None:
        """Compute local expert ids for EP weight filtering.

        When expert parallelism is active, each rank only needs a subset of
        expert weights.  By computing the set upfront we can skip non-local
        expert tensors *before* reading them from disk.
        """
        # ------【EP 权重切分】获取全局 vLLM 配置与并行配置供 EP 过滤决策 ------
        from vllm.config import get_current_vllm_config

        vllm_config = get_current_vllm_config()
        parallel_config = vllm_config.parallel_config

        # ------【EP 权重切分】仅 MoE + 专家并行 + 开启过滤时才计算本地专家 ------
        if not (
            model_config.is_moe
            and parallel_config.enable_expert_parallel
            and parallel_config.enable_ep_weight_filter
        ):
            return

        # When EPLB is enabled, redundant physical expert slots may map to
        # logical experts that belong to other ranks in the default partition.
        # The weight loader needs to see ALL logical expert weights so it can
        # populate these redundant slots.  Skip the filter entirely.
        # ------【EP/EPLB】EPLB 冗余物理槽需加载全部逻辑专家，故跳过过滤 ------
        if parallel_config.enable_eplb:
            return

        # ------【EP 权重切分】读取专家总数，非法（≤0）则跳过过滤 ------
        num_experts = model_config.get_num_experts()
        if num_experts <= 0:
            return

        # EP size/rank computation mirrors FusedMoEParallelConfig.make():
        #   ep_size = dp_size * pcp_size * tp_size (flattened)
        #   ep_rank = dp_rank * pcp_size * tp_size + pcp_rank * tp_size + tp_rank
        # ------【EP 权重切分】按 dp×pcp×tp 展平计算 ep_size 与 ep_rank ------
        # ------【NCCL 通信】导入分布式进程组工具以计算 EP 规模与 rank ------
        from vllm.distributed import (
            get_dp_group,
            get_pcp_group,
            get_tensor_model_parallel_rank,
        )

        dp_size = parallel_config.data_parallel_size
        tp_size = parallel_config.tensor_parallel_size
        pcp_size = parallel_config.prefill_context_parallel_size
        dp_rank = get_dp_group().rank_in_group if dp_size > 1 else 0
        tp_rank = get_tensor_model_parallel_rank() if tp_size > 1 else 0
        pcp_rank = get_pcp_group().rank_in_group if pcp_size > 1 else 0
        ep_size = dp_size * pcp_size * tp_size
        ep_rank = dp_rank * pcp_size * tp_size + pcp_rank * tp_size + tp_rank

        # ------【EP 权重切分】按放置策略计算本 rank 应加载的本地专家 id 集合 ------
        self.local_expert_ids = compute_local_expert_ids(
            num_experts,
            ep_size,
            ep_rank,
            placement=parallel_config.expert_placement_strategy,
        )
        # ------【EP 权重切分】记录 EP 过滤规模与本地加载专家数，便于诊断 ------
        if self.local_expert_ids is not None:
            logger.info_once(
                "EP weight filter: ep_size=%d, ep_rank=%d, loading %d/%d experts",
                ep_size,
                ep_rank,
                len(self.local_expert_ids),
                num_experts,
            )

    @instrument(span_name="Load weights")
    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None:
        # ------【量化】torchao 序列化 checkpoint 需切换 safetensors 加载策略 ------
        if model_config.quantization == "torchao":
            quant_config = get_quant_config(model_config, self.load_config)
            if (
                hasattr(quant_config, "is_checkpoint_torchao_serialized")
                and quant_config.is_checkpoint_torchao_serialized
                and torchao_version_at_least("0.15.0")
            ):
                self.load_config.safetensors_load_strategy = "torchao"

        # ------【EP 权重切分】加载前先初始化本地专家 id 过滤集合 ------
        self._init_ep_weight_filter(model_config)

        # ------【权重加载】让模型以迭代器流式消费权重并完成参数初始化 ------
        loaded_weights = model.load_weights(self.get_all_weights(model_config, model))

        # ------【核心逻辑】记录加载结束时间并打印整体加载耗时 ------
        self.counter_after_loading_weights = time.perf_counter()
        logger.info_once(
            "Loading weights took %.2f seconds",
            self.counter_after_loading_weights - self.counter_before_loading_weights,
        )
        # We only enable strict check for non-quantized models
        # that have loaded weights tracking by default.
        # ------【权重加载】默认对非量化且返回已加载权重的模型启用跟踪校验 ------
        default_enable_weights_track = (
            model_config.quantization is None and loaded_weights is not None
        )
        enable_weights_track = (
            self.enable_weights_track
            if self.enable_weights_track is not None
            else default_enable_weights_track
        )
        # ------【权重加载】开启跟踪时校验模型参数是否全部来自 checkpoint ------
        if enable_weights_track:
            self.track_weights_loading(model, loaded_weights)

    def track_weights_loading(
        self, model: nn.Module, loaded_weights: set[str] | None
    ) -> None:
        # ------【权重加载】收集模型全部参数名作为期望从 checkpoint 加载的集合 ------
        weights_to_load = {name for name, _ in model.named_parameters()}
        # ------【权重加载】仅在有已加载权重集合时进行完整性校验 ------
        if loaded_weights is not None:
            # ignore online quantization scales
            # ------【量化】遍历模块识别在线/后处理量化，其 scale 可不在 checkpoint 中 ------
            for name, module in model.named_modules():
                quant_method = getattr(module, "quant_method", None)
                has_online_quant = getattr(quant_method, "uses_meta_device", False)
                has_postprocess_quant = getattr(
                    quant_method, "process_weights_after_loading", None
                )
                # ignore kv_cache scale and online quant scale,
                # which can be missing in checkpoints
                # ------【量化】把在线量化与 KV cache scale 参数加入已加载集合予以豁免 ------
                if has_online_quant or has_postprocess_quant:
                    for param_name, _ in module.named_parameters():
                        full_name = f"{name}.{param_name}" if name else param_name
                        loaded_weights.add(full_name)
            # ------【权重加载】求期望集合与已加载集合的差集，找出未初始化参数 ------
            weights_not_loaded = weights_to_load - loaded_weights
            # ------【核心逻辑】存在未初始化权重则抛错，防止权重静默缺失 ------
            if weights_not_loaded:
                raise ValueError(
                    "Following weights were not initialized from "
                    f"checkpoint: {weights_not_loaded}"
                )
