# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Utilities for downloading and initializing model weights."""

import asyncio
import concurrent.futures
import fnmatch
import glob
import hashlib
import json
import os
import tempfile
import threading
import time
from collections import defaultdict
from collections.abc import Callable, Generator, Iterable
from contextlib import contextmanager
from pathlib import Path
from typing import IO, Any

import filelock
import huggingface_hub.constants
import numpy as np
import regex as re
import torch
from safetensors.torch import load, load_file, safe_open, save_file
from tqdm.auto import tqdm
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME

from vllm import envs
from vllm.config import ModelConfig
from vllm.config.load import (
    DEFAULT_SAFETENSORS_PREFETCH_BLOCK_SIZE,
    DEFAULT_SAFETENSORS_PREFETCH_NUM_THREADS,
    LoadConfig,
)
from vllm.distributed import get_tensor_model_parallel_rank, get_world_group
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization import (
    QuantizationConfig,
    get_quantization_config,
)
from vllm.model_executor.model_loader.ep_weight_filter import (
    should_skip_weight,
)
from vllm.platforms import current_platform
from vllm.tracing import instrument
from vllm.transformers_utils.repo_utils import hf_api, hf_fs
from vllm.utils.import_utils import PlaceholderModule

try:
    from runai_model_streamer import SafetensorsStreamer
except ImportError:
    runai_model_streamer = PlaceholderModule("runai_model_streamer")  # type: ignore[assignment]
    SafetensorsStreamer = runai_model_streamer.placeholder_attr("SafetensorsStreamer")

try:
    from fastsafetensors import SingleGroup
except ImportError:
    fastsafetensors = PlaceholderModule("fastsafetensors")
    SingleGroup = fastsafetensors.placeholder_attr("SingleGroup")

from vllm.model_executor.layers.quantization.torchao import torchao_version_at_least

logger = init_logger(__name__)

# use system-level temp directory for file locks, so that multiple users
# can share the same lock without error.
# lock files in the temp directory will be automatically deleted when the
# system reboots, so users will not complain about annoying lock files
temp_dir = tempfile.gettempdir()


# ------【下载缓存】启用 HF Xet 高性能下载通道，加速权重拉取 ------
def enable_xet_high_performance():
    """automatically activates xet high performance mode"""
    if "HF_XET_HIGH_PERFORMANCE" not in os.environ:
        huggingface_hub.constants.HF_XET_HIGH_PERFORMANCE = True


enable_xet_high_performance()


class DisabledTqdm(tqdm):
    def __init__(self, *args, **kwargs):
        kwargs["disable"] = True
        super().__init__(*args, **kwargs)


def get_lock(model_name_or_path: str | Path, cache_dir: str | None = None):
    # ------【下载缓存】选定锁目录并把模型名哈希化，得到确定性的锁文件路径 ------
    lock_dir = cache_dir or temp_dir
    model_name_or_path = str(model_name_or_path)
    os.makedirs(os.path.dirname(lock_dir), exist_ok=True)
    model_name = model_name_or_path.replace("/", "-")
    hash_name = hashlib.sha256(model_name.encode()).hexdigest()
    # add hash to avoid conflict with old users' lock files
    lock_file_name = hash_name + model_name + ".lock"
    # ------【下载缓存】用哈希+模型名命名锁文件，创建可跨用户共享的 0o666 文件锁 ------
    # mode 0o666 is required for the filelock to be shared across users
    lock = filelock.FileLock(os.path.join(lock_dir, lock_file_name), mode=0o666)
    return lock


@contextmanager
def atomic_writer(
    filepath: str | Path, mode: str = "w", encoding: str | None = None
) -> Generator[IO]:
    """
    Context manager that provides an atomic file writing routine.

    The context manager writes to a temporary file and, if successful,
    atomically replaces the original file.

    Args:
        filepath (str or Path): The path to the file to write.
        mode (str): The file mode for the temporary file (e.g., 'w', 'wb').
        encoding (str): The encoding for text mode.

    Yields:
        file object: A handle to the temporary file.
    """
    # ------【下载缓存】在目标文件同目录创建临时文件，保证同文件系统可原子替换 ------
    # Create a temporary file in the same directory as the target file
    # to ensure it's on the same filesystem for an atomic replace.
    temp_dir = os.path.dirname(filepath)
    temp_fd, temp_path = tempfile.mkstemp(dir=temp_dir)

    try:
        # ------【下载缓存】把临时文件句柄交给调用方写入，写入成功后才考虑替换 ------
        # Open the temporary file for writing
        with os.fdopen(temp_fd, mode=mode, encoding=encoding) as temp_file:
            yield temp_file

        # ------【下载缓存】写入成功后才执行原子替换，避免半成品覆盖原文件 ------
        # If the 'with' block completes successfully,
        # perform the atomic replace.
        os.replace(temp_path, filepath)

    # ------【下载缓存】写入失败时记录异常并保留原文件，不产生损坏文件 ------
    except Exception:
        logger.exception(
            "Error during atomic write. Original file '%s' not modified", filepath
        )
        raise
    finally:
        # ------【下载缓存】无论成败都清理残留临时文件，避免磁盘空间泄漏 ------
        # Clean up the temporary file if it still exists.
        if os.path.exists(temp_path):
            os.remove(temp_path)


def _natural_sort_key(filepath: str) -> list:
    """Natural sort key for filenames with numeric components, such as
    model-00001-of-00005.safetensors -> ['model-', 1, '-of-', 5, '.safetensors']"""
    # ------【权重加载】按数字分量自然排序分片名，保证 2 号分片排在 10 号之前 ------
    return [
        int(s) if s.isdigit() else s
        for s in re.split(r"(\d+)", os.path.basename(filepath))
    ]


def maybe_download_from_modelscope(
    model: str,
    revision: str | None = None,
    download_dir: str | None = None,
    ignore_patterns: str | list[str] | None = None,
    allow_patterns: list[str] | str | None = None,
) -> str | None:
    """Download model from ModelScope hub if VLLM_USE_MODELSCOPE is True.

    Returns the path to the downloaded model, or None if the model is not
    downloaded from ModelScope."""
    # ------【下载缓存】仅当开启 ModelScope 才懒加载其下载模块，避免强依赖 ------
    if envs.VLLM_USE_MODELSCOPE:
        # download model from ModelScope hub,
        # lazy import so that modelscope is not required for normal use.
        # pylint: disable=C.
        from modelscope.hub.snapshot_download import snapshot_download

        # ------【下载缓存】加文件锁串行下载，多进程不会重复拉取同一份权重 ------
        # Use file lock to prevent multiple processes from
        # downloading the same model weights at the same time.
        with get_lock(model, download_dir):
            if not os.path.exists(model):
                model_path = snapshot_download(
                    model_id=model,
                    cache_dir=download_dir,
                    local_files_only=huggingface_hub.constants.HF_HUB_OFFLINE,
                    revision=revision,
                    ignore_file_pattern=ignore_patterns,
                    allow_patterns=allow_patterns,
                )
            else:
                model_path = model
        # ------【下载缓存】返回已下载路径；未开启 ModelScope 时返回 None 走 HF 流程 ------
        return model_path
    return None


def _shared_pointers(tensors):
    # ------【权重加载】按底层内存地址给张量分组，找出被多个 key 共享的权重 ------
    ptrs = defaultdict(list)
    for k, v in tensors.items():
        ptrs[v.data_ptr()].append(k)
    # ------【权重加载】筛选出同一块内存被多个 key 引用的共享权重组 ------
    failing = []
    for _, names in ptrs.items():
        if len(names) > 1:
            failing.append(names)
    return failing


def convert_bin_to_safetensor_file(
    pt_filename: str,
    sf_filename: str,
) -> None:
    # ------【权重加载】从 .bin 读出 state_dict，兼容 checkpoint 嵌套格式 ------
    loaded = torch.load(pt_filename, map_location="cpu", weights_only=True)
    if "state_dict" in loaded:
        loaded = loaded["state_dict"]
    # ------【权重加载】剔除共享内存的重复权重 key，仅保留一份避免重复存盘 ------
    shared = _shared_pointers(loaded)
    for shared_weights in shared:
        for name in shared_weights[1:]:
            loaded.pop(name)

    # ------【权重加载】将张量转为连续内存布局，才能安全写入 safetensors ------
    # For tensors to be contiguous
    loaded = {k: v.contiguous() for k, v in loaded.items()}

    # ------【权重加载】创建目录并序列化保存为 safetensors 文件 ------
    dirname = os.path.dirname(sf_filename)
    os.makedirs(dirname, exist_ok=True)
    save_file(loaded, sf_filename, metadata={"format": "pt"})

    # ------【权重加载】校验文件大小差异不超过 1%，捕获转换异常 ------
    # check file size
    sf_size = os.stat(sf_filename).st_size
    pt_size = os.stat(pt_filename).st_size
    if (sf_size - pt_size) / pt_size > 0.01:
        raise RuntimeError(
            f"""The file size different is more than 1%:
         - {sf_filename}: {sf_size}
         - {pt_filename}: {pt_size}
         """
        )

    # ------【权重加载】逐 key 校验转换前后张量数值完全一致，保证无损 ------
    # check if the tensors are the same
    reloaded = load_file(sf_filename)
    for k in loaded:
        pt_tensor = loaded[k]
        sf_tensor = reloaded[k]
        if not torch.equal(pt_tensor, sf_tensor):
            raise RuntimeError(f"The output tensors do not match for key {k}")


# TODO(woosuk): Move this to other place.
def get_quant_config(
    model_config: ModelConfig, load_config: LoadConfig
) -> QuantizationConfig:
    # ------【量化】未指定量化方法直接报错；量化配置是加载量权重的先决条件 ------
    if model_config.quantization is None:
        raise ValueError("Model quantization method is not specified in the config.")
    quant_cls = get_quantization_config(model_config.quantization)

    # ------【量化】优先从 HF 模型配置读取量化配置，跟随 checkpoint 实际格式 ------
    # Read the quantization config from the HF model config, if available.
    hf_quant_config = getattr(model_config.hf_config, "quantization_config", None)
    # ------【量化】多模态模型可能把量化配置藏在 text_config 子配置里 ------
    # some vision model may keep quantization_config in their text_config
    hf_text_config = getattr(model_config.hf_config, "text_config", None)
    if hf_quant_config is None and hf_text_config is not None:
        hf_quant_config = getattr(hf_text_config, "quantization_config", None)
    if hf_quant_config is None:
        # ------【量化】compressed-tensors 用 compression_config 字段，需单独兜底读取 ------
        # compressed-tensors uses a compressions_config
        hf_quant_config = getattr(model_config.hf_config, "compression_config", None)

    # ------【量化+TP 权重切分】把注意力头数注入压缩张量配置，支撑 TP 感知加载头缩放 ------
    # Pipe information about heads to enable TP-aware loading of attn_head scales
    if (
        hf_quant_config is not None
        and hf_quant_config.get("quant_method") == "compressed-tensors"
        and "config_groups" in hf_quant_config
    ):
        if hf_text_config is not None:
            n_heads = getattr(hf_text_config, "num_attention_heads", None)
            n_kv_heads = getattr(hf_text_config, "num_key_value_heads", None)
        else:
            n_heads = getattr(model_config.hf_config, "num_attention_heads", None)
            n_kv_heads = getattr(model_config.hf_config, "num_key_value_heads", None)

        hf_quant_config["total_num_heads"] = n_heads
        hf_quant_config["total_num_kv_heads"] = (
            n_kv_heads if n_kv_heads is not None else n_heads
        )

    # ------【量化】存在内嵌量化配置时用它实例化量化类，checkpoint 决定具体实现 ------
    if hf_quant_config is not None:
        # `model_config.quantization_config` may be set alongside a checkpoint
        # quant config: the checkpoint determines `quant_cls`, and the user's
        # QuantizationConfigArgs is consulted by individual quant methods
        # (e.g. for activation overrides via the MXFP4 oracle).

        # For modelopt_mixed, config.json's quantization_config may or may
        # not contain the per-layer quantized_layers map.  Newer checkpoints
        # embed it directly; older ones keep it only in hf_quant_config.json.
        # If it is missing, fall through to the file-based loading path.
        # ------【量化】modelopt_mixed 缺 quantized_layers 映射时回退到文件加载路径 ------
        if (
            model_config.quantization == "modelopt_mixed"
            and "quantized_layers" not in hf_quant_config
        ):
            pass  # fall through to file-based loading below
        else:
            return quant_cls.from_config(hf_quant_config)

    # ------【量化】无内嵌量化配置时，从 hf_overrides 读取外部量化配置路径 ------
    # if hf_quant_config is None, we will try to get config from
    # hf_overrides
    hf_overrides = model_config.hf_overrides
    if not isinstance(hf_overrides, dict):
        raise ValueError(
            "hf_overrides must be a dict for get_quant_config "
            "to get the quantization config from it."
        )
    # ------【量化】支持通过 from_config_file 从外部 JSON 文件加载量化配置 ------
    quantization_config_file = hf_overrides.get("quantization_config_file", None)
    if quantization_config_file is not None:
        if hasattr(quant_cls, "from_config_file"):
            return quant_cls.from_config_file(quantization_config_file)
        else:
            raise NotImplementedError(
                "from_config_file is specified in hf_override config, "
                "but quant_cls.from_config_file is not implemented in "
                f"{quant_cls}"
            )
    # ------【量化】支持直接传入 JSON 字典字符串构造量化配置 ------
    quantization_config_json = hf_overrides.get("quantization_config_dict_json", None)
    if quantization_config_json is not None:
        if hasattr(quant_cls, "from_config_dict_json"):
            return quant_cls.from_config_dict_json(quantization_config_json)
        else:
            raise NotImplementedError(
                "from_config_dict_json is specified in hf_override config, "
                "but quant_cls.from_config_dict_json is not implemented in "
                f"{quant_cls}"
            )

    # ------【量化】在线量化不读 checkpoint，加载时把 fp16/bf16 权重即时量化 ------
    # Online quantization doesn't read from checkpoint configs - it quantizes
    # fp16/bf16 weights on the fly during loading.
    if model_config.quantization_config is not None:
        from vllm.config.quantization import QuantizationConfigArgs
        from vllm.model_executor.layers.quantization.online.base import (
            OnlineQuantizationConfig,
        )

        assert isinstance(model_config.quantization_config, QuantizationConfigArgs)
        return OnlineQuantizationConfig(args=model_config.quantization_config)

    # ------【量化】bitsandbytes 走即时量化分支，用空配置构造量化类 ------
    # Inflight BNB quantization
    if model_config.quantization == "bitsandbytes":
        return quant_cls.from_config({})
    # ------【下载缓存】确定模型路径：ModelScope 优先下载 json，否则用原始模型名 ------
    model_name_or_path = (
        maybe_download_from_modelscope(
            model_config.model,
            revision=model_config.revision,
            download_dir=load_config.download_dir,
            allow_patterns=["*.json"],
        )
        or model_config.model
    )
    is_local = os.path.isdir(model_name_or_path)
    if not is_local:
        # ------【下载缓存】远端模型加锁只下载 json 配置文件，本地目录直接复用 ------
        # Download the config files.
        with get_lock(model_config.model, load_config.download_dir):
            hf_folder = hf_api().snapshot_download(
                model_config.model,
                revision=model_config.revision,
                allow_patterns="*.json",
                cache_dir=load_config.download_dir,
                local_files_only=huggingface_hub.constants.HF_HUB_OFFLINE,
                tqdm_class=DisabledTqdm,
            )
    else:
        hf_folder = model_name_or_path

    # ------【量化】获取量化类期望的配置文件名；未声明则直接用默认配置 ------
    possible_config_filenames = quant_cls.get_config_filenames()

    # If the quantization config is not found, use the default config.
    if not possible_config_filenames:
        return quant_cls()

    # ------【量化】在下载目录中按文件名后缀筛选出唯一的量化配置文件 ------
    config_files = glob.glob(os.path.join(hf_folder, "*.json"))

    quant_config_files = [
        f for f in config_files if any(f.endswith(x) for x in possible_config_filenames)
    ]
    if len(quant_config_files) == 0:
        raise ValueError(f"Cannot find the config file for {model_config.quantization}")
    if len(quant_config_files) > 1:
        raise ValueError(
            f"Found multiple config files for {model_config.quantization}: "
            f"{quant_config_files}"
        )

    # ------【量化】读取 JSON 后按量化类型注入额外字段或校验 producer 合法性 ------
    quant_config_file = quant_config_files[0]
    with open(quant_config_file) as f:
        config = json.load(f)

        if model_config.quantization == "bitsandbytes":
            config["adapter_name_or_path"] = model_config.model
        elif model_config.quantization in ("modelopt", "modelopt_mixed"):
            if config.get("producer", {}).get("name") == "modelopt":
                return quant_cls.from_config(config)
            else:
                raise ValueError(
                    f"Unsupported quantization config"
                    f" found for {model_config.quantization} in {f}."
                )

    # ------【量化】最终用解析出的配置字典构造并返回量化配置对象 ------
    return quant_cls.from_config(config)


def get_sparse_attention_config(
    model_config: ModelConfig,
    load_config: LoadConfig,
    sparse_attention_config_filename: str = "sparse_attention_config.json",
) -> dict[str, Any]:
    model_name_or_path = model_config.model
    is_local = os.path.isdir(model_name_or_path)
    if not is_local:
        # ------【下载缓存】远端模型加锁只下载 json 配置文件，本地路径直接复用 ------
        # Download the config files.
        with get_lock(model_name_or_path, load_config.download_dir):
            hf_folder = hf_api().snapshot_download(
                model_name_or_path,
                revision=model_config.revision,
                allow_patterns="*.json",
                cache_dir=load_config.download_dir,
                local_files_only=huggingface_hub.constants.HF_HUB_OFFLINE,
                tqdm_class=DisabledTqdm,
            )
    else:
        hf_folder = model_name_or_path

    # ------【核心逻辑】未找到稀疏注意力配置文件时返回空 dict，视为非稀疏模型 ------
    config_file = os.path.join(hf_folder, sparse_attention_config_filename)
    if not os.path.exists(config_file):
        return {}

    # ------【核心逻辑】读取并记录稀疏注意力配置，供后续推理引擎使用 ------
    # Load the sparse attention config.
    with open(config_file) as f:
        config = json.load(f)
    logger.info("Loaded sparse attention config from %s", config_file)

    return config


@instrument(span_name="Download weights - HF")
def download_weights_from_hf(
    model_name_or_path: str,
    cache_dir: str | None,
    allow_patterns: list[str],
    revision: str | None = None,
    subfolder: str | None = None,
    ignore_patterns: str | list[str] | None = None,
) -> str:
    """Download model weights from Hugging Face Hub.

    Args:
        model_name_or_path (str): The model name or path.
        cache_dir (Optional[str]): The cache directory to store the model
            weights. If None, will use HF defaults.
        allow_patterns (list[str]): The allowed patterns for the
            weight files. Files matched by any of the patterns will be
            downloaded.
        revision (Optional[str]): The revision of the model.
        subfolder (Optional[str]): The subfolder within the model repository
            to download weights from.
        ignore_patterns (Optional[Union[str, list[str]]]): The patterns to
            filter out the weight files. Files matched by any of the patterns
            will be ignored.

    Returns:
        str: The path to the downloaded model weights.
    """
    assert len(allow_patterns) > 0
    local_only = huggingface_hub.constants.HF_HUB_OFFLINE
    # ------【下载缓存】联网时先列仓库文件，把下载模式压缩成一次快照下载 ------
    if not local_only:
        # Attempt to reduce allow_patterns to a single pattern
        # so we only have to call snapshot_download once.
        try:
            # ------【下载缓存】列出仓库文件清单，用于精确裁剪要下载的文件集合 ------
            fs = hf_fs()
            file_list = fs.ls(
                os.path.join(model_name_or_path, subfolder or ""),
                detail=False,
                revision=revision,
            )

            # ------【下载缓存】有 safetensors 索引时按索引逐文件精确下载，跳过冗余子目录 ------
            # If downloading safetensors and an index file exists, use the
            # specific file names from the index to avoid downloading
            # unnecessary files (e.g., from subdirectories like "original/").
            index_file = f"{model_name_or_path}/{SAFE_WEIGHTS_INDEX_NAME}"
            if "*.safetensors" in allow_patterns and index_file in file_list:
                index_path = hf_api().hf_hub_download(
                    repo_id=model_name_or_path,
                    filename=SAFE_WEIGHTS_INDEX_NAME,
                    cache_dir=cache_dir,
                    revision=revision,
                    subfolder=subfolder,
                )
                with open(index_path) as f:
                    weight_map = json.load(f)["weight_map"]
                if weight_map:
                    # Extra [] so that weight_map files are treated as a
                    # single allow_pattern in the loop below
                    allow_patterns = [list(set(weight_map.values()))]  # type: ignore[list-item]
                else:
                    allow_patterns = ["*.safetensors"]
            else:
                # ------【下载缓存】无索引时退回用第一个能匹配到文件的 glob 模式下载 ------
                # Use the first pattern found in the HF repo's files.
                for pattern in allow_patterns:
                    if fnmatch.filter(file_list, pattern):
                        allow_patterns = [pattern]
                        break
        # ------【下载缓存】列文件失败则降级为逐模式尝试，保证下载流程不中断 ------
        except Exception as e:
            logger.warning(
                "Failed to get file list for '%s'. Trying each pattern in "
                "allow_patterns individually until weights have been "
                "downloaded. Error: %s",
                model_name_or_path,
                e,
            )

    logger.debug("Using model weights format %s", allow_patterns)
    # ------【下载缓存】加文件锁串行下载，多进程不重复拉取同一权重 ------
    # Use file lock to prevent multiple processes from
    # downloading the same model weights at the same time.
    with get_lock(model_name_or_path, cache_dir):
        start_time = time.perf_counter()
        # ------【下载缓存】逐个模式调用 snapshot_download 并计时，命中即停止尝试 ------
        for allow_pattern in allow_patterns:
            hf_folder = hf_api().snapshot_download(
                model_name_or_path,
                allow_patterns=allow_pattern,
                ignore_patterns=ignore_patterns,
                cache_dir=cache_dir,
                tqdm_class=DisabledTqdm,
                revision=revision,
                local_files_only=local_only,
            )
            # If we have downloaded weights for this allow_pattern,
            # we don't need to check the rest.
            # allow_pattern can be a list (from weight_map) or str (glob)
            if isinstance(allow_pattern, list):
                break
            if any(Path(hf_folder).glob(allow_pattern)):
                break
        # ------【下载缓存】下载耗时超过阈值时打印日志，便于排查慢网络 ------
        time_taken = time.perf_counter() - start_time
        if time_taken > 0.5:
            logger.info(
                "Time spent downloading weights for %s: %.6f seconds",
                model_name_or_path,
                time_taken,
            )
    return hf_folder


def download_safetensors_index_file_from_hf(
    model_name_or_path: str,
    index_file: str,
    cache_dir: str | None,
    subfolder: str | None = None,
    revision: str | None = None,
) -> None:
    """Download hf safetensors index file from Hugging Face Hub.

    Args:
        model_name_or_path (str): The model name or path.
        index_file (str): The safetensors index file name
        cache_dir (Optional[str]): The cache directory to store the model
            weights. If None, will use HF defaults.
        subfolder (Optional[str]): The subfolder within the model repository
            to download weights from.
        revision (Optional[str]): The revision of the model.
    """
    # ------【下载缓存】加文件锁避免多进程重复下载 safetensors 索引文件 ------
    # Use file lock to prevent multiple processes from
    # downloading the same model weights at the same time.
    with get_lock(model_name_or_path, cache_dir):
        try:
            # Download the safetensors index file.
            hf_api().hf_hub_download(
                repo_id=model_name_or_path,
                filename=index_file,
                cache_dir=cache_dir,
                revision=revision,
                subfolder=subfolder,
                local_files_only=huggingface_hub.constants.HF_HUB_OFFLINE,
            )
        # ------【下载缓存】索引缺失只记录日志不阻断，只有部分模型带索引文件 ------
        # If file not found on remote or locally, we should not fail since
        # only some models will have index_file.
        except huggingface_hub.utils.LocalEntryNotFoundError:
            logger.info("No %s found in local cache.", index_file)
        # ------【下载缓存】本地与远端缺失均分别提示，便于定位索引来源 ------
        except huggingface_hub.utils.EntryNotFoundError:
            logger.info("No %s found in remote.", index_file)


# For models like Mistral-7B-v0.3, there are both sharded
# safetensors files and a consolidated safetensors file.
# Passing both of these to the weight loader functionality breaks.
# So, we use the index_file to
# look up which safetensors files should be used.
def filter_duplicate_safetensors_files(
    hf_weights_files: list[str], hf_folder: str, index_file: str
) -> list[str]:
    # model.safetensors.index.json is a mapping from keys in the
    # torch state_dict to safetensors file holding that weight.
    # ------【权重加载】无索引文件时直接返回原文件列表，不做去重 ------
    index_file_name = os.path.join(hf_folder, index_file)
    if not os.path.isfile(index_file_name):
        return hf_weights_files

    # ------【权重加载】读取索引 weight_map，得到每个权重对应的分片文件 ------
    # Iterate through the weight_map (weight_name: safetensors files)
    # to identify weights that we should use.
    with open(index_file_name) as f:
        weight_map = json.load(f)["weight_map"]
    # ------【权重加载】收集索引实际引用的分片文件集合，作为去重白名单 ------
    weight_files_in_index = set()
    for weight_name in weight_map:
        weight_files_in_index.add(os.path.join(hf_folder, weight_map[weight_name]))
    # ------【权重加载】索引引用但磁盘缺失的文件直接报错，防止漏加载权重 ------
    # Check if files referenced in model.safetensors.index.json actually exist.
    # Raise error if any file is missing.
    hf_weights_files_set = set(hf_weights_files)
    missing_files = weight_files_in_index - hf_weights_files_set
    if missing_files:
        raise FileNotFoundError(
            f"Weight files referenced in index but missing: {missing_files}"
        )
    # ------【权重加载】过滤掉索引未引用的冗余分片，只保留真正需要的文件 ------
    # Filter out any fields that are not found in the index file.
    hf_weights_files = [f for f in hf_weights_files if f in weight_files_in_index]
    return hf_weights_files


def filter_files_not_needed_for_inference(hf_weights_files: list[str]) -> list[str]:
    """
    Exclude files that are not needed for inference.

    See https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/trainer.py#L227-L233
    """
    # ------【权重加载】剔除训练态文件（优化器/调度器等），只保留推理需要的权重 ------
    blacklist = [
        "training_args.bin",
        "optimizer.bin",
        "optimizer.pt",
        "scheduler.pt",
        "scaler.pt",
    ]
    hf_weights_files = [
        f for f in hf_weights_files if not any(f.endswith(x) for x in blacklist)
    ]
    return hf_weights_files


# explicitly use pure text format, with a newline at the end
# this makes it impossible to see the animation in the progress bar
# but will avoid messing up with ray or multiprocessing, which wraps
# each line of output with some prefix.
_BAR_FORMAT = "{desc}: {percentage:3.0f}% Completed | {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]\n"  # noqa: E501


def enable_tqdm(use_tqdm_on_load: bool):
    # ------【并行加载】仅主进程开启进度条，避免多进程输出互相污染 ------
    return use_tqdm_on_load and (
        not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
    )


def np_cache_weights_iterator(
    model_name_or_path: str,
    cache_dir: str | None,
    hf_folder: str,
    hf_weights_files: list[str],
    use_tqdm_on_load: bool,
) -> Generator[tuple[str, torch.Tensor], None, None]:
    """Iterate over the weights in the model np files.

    Will dump the model weights to numpy files if they are not already dumped.
    """
    # Convert the model weights from torch tensors to numpy arrays for
    # faster loading.
    # ------【权重加载】准备 np 缓存目录和权重名清单文件路径 ------
    np_folder = os.path.join(hf_folder, "np")
    os.makedirs(np_folder, exist_ok=True)
    weight_names_file = os.path.join(np_folder, "weight_names.json")
    # ------【下载缓存】加文件锁，避免多进程同时把 torch 权重转存为 numpy ------
    # Use file lock to prevent multiple processes from
    # dumping the same model weights to numpy at the same time.
    with get_lock(model_name_or_path, cache_dir):
        if not os.path.exists(weight_names_file):
            weight_names: list[str] = []
            # ------【权重加载】首次访问时把每个 torch 权重 dump 成 .npy，加速后续加载 ------
            for bin_file in tqdm(
                hf_weights_files,
                desc="Loading np_cache checkpoint shards",
                disable=not enable_tqdm(use_tqdm_on_load),
                bar_format=_BAR_FORMAT,
            ):
                state = torch.load(bin_file, map_location="cpu", weights_only=True)
                for name, param in state.items():
                    param_path = os.path.join(np_folder, name)
                    with open(param_path, "wb") as f:
                        np.save(f, param.cpu().detach().numpy())
                    weight_names.append(name)
            with open(weight_names_file, "w") as f:
                json.dump(weight_names, f)

    # ------【权重加载】读取已缓存的权重名清单，供迭代时逐个取回 ------
    with open(weight_names_file) as f:
        weight_names = json.load(f)

    # ------【权重加载】从 numpy 缓存逐一加载并转回 torch 张量后 yield 给上层 ------
    for name in weight_names:
        param_path = os.path.join(np_folder, name)
        with open(param_path, "rb") as f:
            param = np.load(f)
        yield name, torch.from_numpy(param)


def _get_checkpoints_size_bytes(files: list[str]) -> int:
    """Return the total size of the checkpoint files in bytes."""
    # ------【下载缓存】统计所有 checkpoint 分片总字节数，用于判断能否放入 RAM ------
    if not files:
        return 0
    return sum(os.path.getsize(f) for f in files)


def _get_available_ram_bytes() -> int:
    """Return available RAM, honoring cgroup limits."""
    import psutil

    # ------【下载缓存】读取宿主机可用内存，作为 prefetch 是否可行的上限参考 ------
    host_available = psutil.virtual_memory().available

    from vllm.utils.cpu_resource_utils import get_cgroup_memory_limit

    # ------【下载缓存】读取 cgroup 内存限额，容器环境需以更小配额为准 ------
    cgroup_limit, cgroup_usage = get_cgroup_memory_limit()
    if cgroup_limit is None:
        return host_available
    # ------【下载缓存】计算 cgroup 内剩余可用内存，防止超出容器配额 ------
    cgroup_available = (
        cgroup_limit if cgroup_usage is None else max(0, cgroup_limit - cgroup_usage)
    )
    # ------【下载缓存】取宿主机与 cgroup 两者较小值作为真实可用内存 ------
    return min(host_available, cgroup_available)


def _get_fs_type(files: list[str]) -> str:
    """Get the filesystem type of the first file in *files* (Linux only)."""
    # ------【下载缓存】空文件列表直接返回空串，表示文件系统类型未知 ------
    if not files:
        return ""
    try:
        # Only the first file is checked — all checkpoint shards reside
        # in the same directory and therefore on the same filesystem.
        # ------【下载缓存】解析首个分片真实路径，所有分片同目录即同文件系统 ------
        resolved = os.path.realpath(files[0])
        best_mount = ""
        best_fstype = ""
        # /proc/mounts may contain nested mount points (e.g. "/" -> ext4,
        # "/data" -> nfs4, "/data/local" -> ext4).  We pick the entry with
        # the longest matching mount_point — the same "longest prefix match"
        # rule the kernel uses to decide which filesystem serves a path.
        # ------【下载缓存】遍历 /proc/mounts 用最长前缀匹配找到该路径挂载点 ------
        with open("/proc/mounts") as f:
            for line in f:
                parts = line.split()
                if len(parts) < 3:
                    continue
                mount_point, fstype = parts[1], parts[2]
                if (
                    resolved == mount_point
                    or resolved.startswith(os.path.join(mount_point, ""))
                ) and len(mount_point) > len(best_mount):
                    best_mount = mount_point
                    best_fstype = fstype
        # ------【下载缓存】返回识别出的文件系统类型，供判断是否为网络 FS ------
        return best_fstype
    except Exception:
        # /proc/mounts is Linux-specific; on other OSes (or if the read
        # fails for any reason) we fall back to an empty string.
        return ""


def _prefetch_checkpoint(
    file_path: str,
    block_size: int = DEFAULT_SAFETENSORS_PREFETCH_BLOCK_SIZE,
) -> None:
    """Prefetch a checkpoint file into the OS page cache.

    Reads the file in blocks so the kernel caches its pages before workers load
    the same file.
    """
    # ------【下载缓存】校验 prefetch 块大小合法，避免无效或死循环读取 ------
    if block_size < 1:
        raise ValueError("safetensors prefetch block size must be >= 1")

    # ------【下载缓存】按块顺序读完整文件，让内核提前装入 page cache ------
    with open(file_path, "rb") as f:
        while f.read(block_size):
            pass


def _prefetch_all_checkpoints(
    sorted_files: list[str],
    num_prefetch_threads: int = DEFAULT_SAFETENSORS_PREFETCH_NUM_THREADS,
    block_size: int = DEFAULT_SAFETENSORS_PREFETCH_BLOCK_SIZE,
) -> None:
    """Start prefetching checkpoint files into page cache in a background thread."""
    # ------【并行加载】校验预取线程数与块大小，避免非法参数进入线程池 ------
    if num_prefetch_threads < 1:
        raise ValueError("safetensors prefetch num threads must be >= 1")
    if block_size < 1:
        raise ValueError("safetensors prefetch block size must be >= 1")

    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
    else:
        rank = 0
        world_size = 1
    # ------【并行加载+DP】按 rank 步长切分待预取文件，多卡各预取自己那份 ------
    paths_to_prefetch = sorted_files[rank::world_size]
    total_for_rank = len(paths_to_prefetch)

    # ------【并行加载+异步 RPC】用 asyncio 事件循环调度并发预取任务 ------
    async def _prefetch_all() -> None:
        loop = asyncio.get_running_loop()
        completed = 0
        next_log_pct = 10

        # ------【并行加载+异步 RPC】单个文件在独立线程里预取，避免阻塞事件循环 ------
        async def prefetch_one(
            path: str,
            executor: concurrent.futures.ThreadPoolExecutor,
        ) -> None:
            nonlocal completed, next_log_pct
            try:
                # ------【并行加载】把磁盘预取丢给线程池执行，完成后计数并按进度打日志 ------
                await loop.run_in_executor(
                    executor, _prefetch_checkpoint, path, block_size
                )
                completed += 1
                if total_for_rank > 0 and next_log_pct <= 100:
                    pct = 100 * completed / total_for_rank
                    if pct >= next_log_pct:
                        logger.info(
                            "Prefetching checkpoint files: %d%% (%d/%d)",
                            next_log_pct,
                            completed,
                            total_for_rank,
                        )
                        next_log_pct += 10
            # ------【并行加载】单个文件预取失败只告警不中断，其余文件继续预取 ------
            except Exception:
                logger.warning(
                    "Failed to prefetch checkpoint file %r.", path, exc_info=True
                )

        # ------【并行加载】用线程池并发预取，gather 等待本 rank 全部文件读完 ------
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=num_prefetch_threads
        ) as executor:
            await asyncio.gather(
                *(prefetch_one(p, executor) for p in paths_to_prefetch)
            )

    # ------【并行加载】封装 asyncio.run 启动预取协程并统计总耗时 ------
    def _run_prefetch() -> None:
        start = time.perf_counter()
        asyncio.run(_prefetch_all())
        elapsed = time.perf_counter() - start
        logger.info(
            "Prefetching checkpoint files into page cache finished in %.2fs",
            elapsed,
        )

    # ------【并行加载】记录预取启动参数，便于排查线程数和块大小配置 ------
    logger.info(
        "Prefetching checkpoint files into page cache started "
        "(in background, num_threads=%d, block_size=%d bytes)",
        num_prefetch_threads,
        block_size,
    )
    # ------【并行加载+进程管理】起后台守护线程异步预取，主线程继续加载权重 ------
    threading.Thread(target=_run_prefetch, daemon=True).start()











# .safetensors的单线程权重加载迭代器
def safetensors_weights_iterator(
    hf_weights_files: list[str], # 权重文件
    use_tqdm_on_load: bool,
    safetensors_load_strategy: str | None = None,
    local_expert_ids: set[int] | None = None,
    *,
    safetensors_prefetch_num_threads: int = DEFAULT_SAFETENSORS_PREFETCH_NUM_THREADS,
    safetensors_prefetch_block_size: int = DEFAULT_SAFETENSORS_PREFETCH_BLOCK_SIZE,
) -> Generator[tuple[str, torch.Tensor], None, None]:
    """Iterate over the weights in the model safetensor files.

    When *local_expert_ids* is provided, expert weights not belonging to
    this rank are skipped **before** reading from disk, which drastically
    reduces storage I/O for MoE models under EP.
    """
    # ------【权重加载】拼装进度条描述文案，eager 策略显式标注以区分加载模式 ------
    loading_desc = "Loading safetensors checkpoint shards"
    if safetensors_load_strategy == "eager":
        loading_desc += " (eager)"

    # ------【权重加载】按自然序排分片文件名，保证 shard 0/1/2 而非 0/10/11 的加载顺序 ------
    sorted_files = sorted(hf_weights_files, key=_natural_sort_key) # 排序

    # ------【下载缓存】探测 FS 类型、checkpoint 总量与可用内存，判断能否整体装入内存以定预取策略 ------
    fs_type = _get_fs_type(sorted_files)
    is_net_fs = fs_type in ("nfs", "nfs4", "lustre")
    total_bytes = _get_checkpoints_size_bytes(sorted_files)
    avail_bytes = _get_available_ram_bytes()
    ram_threshold_pct = 90
    fits_in_ram = total_bytes <= (ram_threshold_pct / 100.0) * avail_bytes
    fs_name = fs_type.upper() if fs_type else "unknown"

    # ------【下载缓存】一次性打印 FS 与内存占用信息，便于诊断加载性能与预取决策 ------
    logger.info_once(
        "Filesystem type for checkpoints: %s. Checkpoint size: %.2f GiB. "
        "Available RAM: %.2f GiB.",
        fs_name,
        total_bytes / 1024**3,
        avail_bytes / 1024**3,
    )

    # ------【下载缓存】按显式策略或「网络 FS + 可装入内存」启发式决定是否预取到 page cache ------
    should_prefetch = safetensors_load_strategy == "prefetch"
    if safetensors_load_strategy is None:
        if is_net_fs and fits_in_ram:
            should_prefetch = True
        elif is_net_fs and not fits_in_ram:
            logger.warning_once(
                "Network filesystem (%s) detected but checkpoint total size "
                "(%.2f GiB) exceeds %d%% of available RAM (%.2f GiB). "
                "Skipping auto-prefetch.",
                fs_name,
                total_bytes / 1024**3,
                ram_threshold_pct,
                avail_bytes / 1024**3,
            )
        elif not is_net_fs and fits_in_ram:
            logger.info_once(
                "Auto-prefetch is disabled because the filesystem (%s) is not a "
                "recognized network FS (NFS/Lustre). If you want to force "
                "prefetching, start vLLM with --safetensors-load-strategy=prefetch.",
                fs_name,
            )
        elif not is_net_fs and not fits_in_ram:
            logger.info_once(
                "Auto-prefetch is disabled because the filesystem (%s) is not a "
                "recognized network FS (NFS/Lustre) and the checkpoint size "
                "(%.2f GiB) exceeds %d%% of available RAM (%.2f GiB).",
                fs_name,
                total_bytes / 1024**3,
                ram_threshold_pct,
                avail_bytes / 1024**3,
            )
    elif should_prefetch and not fits_in_ram:
        logger.warning_once(
            "safetensors_load_strategy='prefetch' was explicitly specified, but "
            "checkpoint total size (%.2f GiB) exceeds %d%% of available RAM "
            "(%.2f GiB). This may cause out-of-memory errors.",
            total_bytes / 1024**3,
            ram_threshold_pct,
            avail_bytes / 1024**3,
        )

    # ------【下载缓存+并行加载】多线程异步 IO 把全部 checkpoint 读入 OS page cache，加速后续首次访问 ------
    if should_prefetch:
        _prefetch_all_checkpoints(
            sorted_files,
            num_prefetch_threads=safetensors_prefetch_num_threads,
            block_size=safetensors_prefetch_block_size,
        )

    # ------【量化】记录 torchao 跨分片尚未补齐的 tensor 子类数据，供后续分片继续拼接 ------
    leftover_state_dict: dict[str, torch.Tensor] = {}







    ##################################################################
    # 1. 开始加载权重
    ##################################################################
    # ------【权重加载】主循环逐分片迭代并显示进度条，按加载策略分发到不同分支 ------
    for st_file in tqdm( # 每一个文件
        sorted_files,
        desc=loading_desc,
        disable=not enable_tqdm(use_tqdm_on_load),
        bar_format=_BAR_FORMAT,
    ):
        # ------【权重加载】eager 策略：整文件读入内存后反序列化，跳过按需懒加载以换取更快解压 ------
        if safetensors_load_strategy == "eager":
            with open(st_file, "rb") as f:
                state_dict = load(f.read())
            # ------【EP 权重切分】按 rank 的 local_expert_ids 过滤掉不属于本 rank 的 expert 权重 ------
            for name, param in state_dict.items():
                if not should_skip_weight(name, local_expert_ids):
                    yield name, param
        elif safetensors_load_strategy == "torchao":
            # we can't load flattened torchao tensor subclasses directly into the model
            # instead we reconstruct the subclasses here before returning
            # ------【量化】校验 torchao 版本，flatten/unflatten 子类还原接口为 0.15.0 新增 ------
            if not torchao_version_at_least("0.15.0"):
                raise ValueError(
                    "Please use torchao version >= 0.15.0 "
                    "to load torchao safetensors checkpoint"
                )
            # ------【量化】引入 torchao 的 unflatten 工具，把打平的子类数据还原回 tensor ------
            from torchao.prototype.safetensors.safetensors_support import (
                unflatten_tensor_state_dict,
            )

            # ------【量化+EP 权重切分】safe_open 懒加载逐张量读取并跳过本 rank 无关的 expert 权重 ------
            with safe_open(st_file, framework="pt") as f:
                state_dict = {}
                for name in f.keys():  # noqa: SIM118
                    if should_skip_weight(name, local_expert_ids):
                        continue
                    state_dict[name] = f.get_tensor(name)

                # update with leftover tensor data from previous iteration, if any
                # ------【量化】合并上一分片遗留子类数据并 unflatten 还原，剩余数据留待后续分片拼接 ------
                state_dict.update(leftover_state_dict)
                metadata = f.metadata()
                # due to sharded checkpoints, we are not guaranteed that we have all
                # tensor subclass data on one file
                # state_dict has the leftover data from this step and we wait for
                # missing information to be provided in a future iteration
                unflattened_state_dict, leftover_state_dict = (
                    unflatten_tensor_state_dict(state_dict, metadata)
                )
            # ------【量化】产出还原后的 torchao 子类张量给上层 loader ------
            yield from unflattened_state_dict.items()
        # ------【权重加载+EP 权重切分】默认路径：safe_open 懒加载逐张量读取并过滤无关 expert 权重 ------
        else:

            ##################################################################
            # 默认加载每一个.safetensors
            ##################################################################
            # safe_open是safetensors的库函数，pt表示是pytorch框架的张量类型。所以返回的是 torch.Tensor
            # 除此之外还有np = numpy, tf=tensorflow
            with safe_open(st_file, framework="pt") as f:
                for name in f.keys():  # noqa: SIM118 # 遍历每一个权重名
                    if should_skip_weight(name, local_expert_ids):
                        continue
                    param = f.get_tensor(name) # 懒加载mmap出一个权重
                    yield name, param # 抛出










def multi_thread_safetensors_weights_iterator(
    hf_weights_files: list[str],
    use_tqdm_on_load: bool,
    max_workers: int = 4,
) -> Generator[tuple[str, torch.Tensor], None, None]:
    """Multi-Thread iterate over the weights in the model safetensor files."""

    # ------【并行加载】工作函数：单个分片整读进内存并反序列化，供线程池并发执行 ------
    def _load_file(st_file: str):
        result = load_file(st_file, device="cpu")
        return result

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Note to use generator here so we do not store all the loaded files in memory
        # at the same time, which can cause OOM for large models.
        # ------【并行加载】惰性生成 Future，避免同时把全部已加载分片常驻内存导致 OOM ------
        futures = (executor.submit(_load_file, st_file) for st_file in hf_weights_files)
        # ------【并行加载】as_completed 按完成顺序消费 Future，谁先加载完先产出 ------
        futures_iter = tqdm(
            concurrent.futures.as_completed(futures),
            total=len(hf_weights_files),
            desc="Multi-thread loading shards",
            disable=not enable_tqdm(use_tqdm_on_load),
            bar_format=_BAR_FORMAT,
        )

        # ------【并行加载】逐个取出结果，pop 出每个键随即释放引用以尽早回收内存 ------
        for future in futures_iter:
            state_dict = future.result()
            del future
            for key in list(state_dict):
                yield key, state_dict.pop(key)


def runai_safetensors_weights_iterator(
    hf_weights_files: list[str],
    use_tqdm_on_load: bool,
    is_distributed: bool = False,
) -> Generator[tuple[str, torch.Tensor], None, None]:
    """Iterate over the weights in the model safetensor files."""
    with SafetensorsStreamer() as streamer:
        is_cuda_alike = current_platform.is_cuda_alike()
        # ------【权重加载】分布式 + CUDA 场景下直读进 GPU，否则落到 CPU ------
        device = (
            f"cuda:{current_platform.current_device()}"
            if is_distributed and is_cuda_alike
            else "cpu"
        )

        # ------【并行加载】RunAI Streamer 异步流式读取分片元数据与张量，边读边用 ------
        streamer.stream_files(
            hf_weights_files,
            device=device,
            is_distributed=is_distributed,
        )
        # ------【权重加载】统计总张量数，用于进度条总量显示 ------
        total_tensors = sum(
            len(tensors_meta)
            for tensors_meta in streamer.files_to_tensors_metadata.values()
        )

        # ------【并行加载】以张量为粒度取流并显示进度，控制刷新频率降低打屏开销 ------
        tensor_iter = tqdm(
            streamer.get_tensors(),
            total=total_tensors,
            desc="Loading safetensors using Runai Model Streamer",
            bar_format=_BAR_FORMAT,
            disable=not enable_tqdm(use_tqdm_on_load),
            mininterval=2,
        )

        # ------【权重加载】clone 一份让张量持有独立内存，脱离 streamer 缓冲区后仍有效 ------
        for name, tensor in tensor_iter:
            yield name, tensor.clone()


def fastsafetensors_weights_iterator(
    hf_weights_files: list[str],
    use_tqdm_on_load: bool,
) -> Generator[tuple[str, torch.Tensor], None, None]:
    """Iterate over the weights in the model safetensor files
    using fastsafetensor library.

    Uses ParallelLoader for pipelined loading: the producer thread
    prepares metadata for the next shard while the consumer yields
    tensors from the current shard.
    """
    # ------【并行加载】引入 fastsafetensors 的流水线 loader，生产/消费双线程重叠 IO 与解压 ------
    from fastsafetensors.parallel_loader import ParallelLoader

    # ------【TP】取全局进程组用于判断 TP 规模，未初始化时退化为单进程组 ------
    if torch.distributed.is_initialized():
        pg = torch.distributed.group.WORLD
    else:
        pg = SingleGroup()

    # ------【权重加载】固定当前 CUDA 设备并按自然序排分片，保证加载顺序确定 ------
    device = torch.device(f"cuda:{current_platform.current_device()}")
    hf_weights_files = sorted(hf_weights_files, key=_natural_sort_key)

    # Use nogds=True for TP > 1 to avoid cuFileDriverOpen() which
    # initializes the GDS DMA subsystem for all visible GPUs, creating
    # unwanted CUDA contexts on every device.
    # ------【TP】TP>1 时禁用 GDS，避免 cuFileDriverOpen 给所有可见 GPU 建多余 CUDA context ------
    nogds = pg.size() > 1

    # ------【并行加载】用队列长度控制生产/消费者之间缓冲深度，平衡吞吐与内存 ------
    queue_size = envs.VLLM_FASTSAFETENSORS_QUEUE_SIZE
    tqdm_enabled = enable_tqdm(use_tqdm_on_load)

    # ------【并行加载】构造 ParallelLoader 的工厂函数，便于 GDS 失败时用 nogds=True 重试 ------
    def _make_loader(nogds: bool) -> "ParallelLoader":
        return ParallelLoader(
            pg=pg,
            hf_weights_files=hf_weights_files,
            queue_size=queue_size,
            use_tqdm_on_load=tqdm_enabled,
            device=str(device),
            nogds=nogds,
        )

    # GDS can fail either at construction or lazily inside the producer
    # thread during iteration (e.g. cuFileHandleRegister returning
    # CU_FILE_HANDLE_NOT_REGISTERED on a filesystem without GDS support).
    # Catch both and fall back to nogds, but only before yielding any
    # tensor -- restarting mid-stream would reload earlier shards.
    pl = None
    yielded = False
    try:
        try:
            # ------【并行加载】首次尝试原配置构造 loader 并迭代，记录是否已产出任何张量 ------
            pl = _make_loader(nogds)
            for name, tensor in pl.iterate_weights():
                yielded = True
                yield name, tensor
        except RuntimeError as e:
            # ------【并行加载】GDS 失败时仅当尚未产出张量才回退 nogds 重试，避免重复读分片 ------
            if nogds or yielded or "gds" not in str(e):
                raise
            logger.warning_once(
                "GDS not enabled, setting `nogds=True`.\n"
                "For more information, see: https://github.com/foundation-model-stack/"
                "fastsafetensors?tab=readme-ov-file#basic-api-usages"
            )
            # ------【并行加载】关闭旧 loader 并用 nogds=True 重建，重新迭代全部权重 ------
            if pl is not None:
                pl.close()
            pl = _make_loader(nogds=True)
            yield from pl.iterate_weights()
    finally:
        # ------【并行加载】无论成功还是异常，最后释放 loader 持有的线程与缓冲资源 ------
        if pl is not None:
            pl.close()


def instanttensor_weights_iterator(
    hf_weights_files: list[str],
    use_tqdm_on_load: bool,
) -> Generator[tuple[str, torch.Tensor], None, None]:
    """Iterate over the weights in the model safetensor files
    using instanttensor library."""
    # ------【并行加载】懒引入 instanttensor，缺失时给出可操作的安装提示 ------
    try:
        import instanttensor
    except ImportError as e:
        raise ImportError(
            "Please install instanttensor via `pip install vllm[instanttensor]`"
        ) from e

    # ------【权重加载】InstantTensor 依赖 NVIDIA GPU，非 CUDA 直接拒绝 ------
    if not current_platform.is_cuda():
        raise ValueError("InstantTensor requires NVIDIA GPUs")

    # ------【TP】取 device 进程组用于多卡协同读取，单卡或单测未初始化时置 None ------
    try:
        world_group = get_world_group()
    except AssertionError:
        # Entering here only in unit tests where the world group is not initialized.
        process_group = None
    else:
        process_group = world_group.device_group if world_group.world_size > 1 else None

    # ------【权重加载】取当前设备索引用于 GPU 直读，避免数据先落 CPU 再搬运 ------
    device = current_platform.current_device()

    # copy=True yields tensors that own their memory, staying valid after the
    # context exits or InstantTensor reuses its buffer.
    # ------【并行加载】copy=True 让张量拥有独立内存，脱离上下文或缓冲复用后仍有效 ------
    with instanttensor.safe_open(
        hf_weights_files,
        framework="pt",
        device=device,
        process_group=process_group,
        copy=True,
    ) as f:
        # Track bytes so the bar reports load throughput (GB/s).
        # ------【权重加载】按字节数推进进度条，实时显示 GB/s 加载吞吐 ------
        pbar = tqdm(
            total=f.total_tensor_size,
            desc="Loading safetensors using InstantTensor loader",
            disable=not enable_tqdm(use_tqdm_on_load),
            bar_format=_BAR_FORMAT,
            position=tqdm._get_free_pos(),
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            mininterval=1.0,
        )
        try:
            # ------【并行加载】逐张量产出并更新已读字节数，结束时关闭进度条 ------
            for name, tensor in f.tensors():
                pbar.update(tensor.numel() * tensor.element_size())
                yield name, tensor
        finally:
            pbar.close()


def pt_weights_iterator(
    hf_weights_files: list[str],
    use_tqdm_on_load: bool,
    pt_load_map_location: str | dict[str, str] = "cpu",
) -> Generator[tuple[str, torch.Tensor], None, None]:
    """Iterate over the weights in the model bin/pt files."""
    # ------【权重加载】逐个 bin/pt 分片加载并显示进度条 ------
    for bin_file in tqdm(
        hf_weights_files,
        desc="Loading pt checkpoint shards",
        disable=not enable_tqdm(use_tqdm_on_load),
        bar_format=_BAR_FORMAT,
    ):
        # ------【权重加载】weights_only 反序列化到指定设备，逐项产出后释放 state 引用 ------
        state = torch.load(
            bin_file, map_location=pt_load_map_location, weights_only=True
        )
        yield from state.items()
        del state


def multi_thread_pt_weights_iterator(
    hf_weights_files: list[str],
    use_tqdm_on_load: bool,
    pt_load_map_location: str | dict[str, str] = "cpu",
    max_workers: int = 4,
) -> Generator[tuple[str, torch.Tensor], None, None]:
    """Multi-Thread iterate over the weights in the model bin/pt files."""

    # ------【并行加载】工作函数：单个 pt 分片 weights_only 反序列化，供线程池并发执行 ------
    def _load_file(bin_file: str):
        return torch.load(
            bin_file, map_location=pt_load_map_location, weights_only=True
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # ------【并行加载】一次性提交全部分片加载任务，由线程池并发反序列化 ------
        futures = [
            executor.submit(_load_file, bin_file) for bin_file in hf_weights_files
        ]
        # ------【并行加载】as_completed 按完成顺序消费，先加载完的分片先产出 ------
        futures_iter = tqdm(
            concurrent.futures.as_completed(futures),
            total=len(hf_weights_files),
            desc="Multi-thread loading pt checkpoint shards",
            disable=not enable_tqdm(use_tqdm_on_load),
            bar_format=_BAR_FORMAT,
        )

        # ------【并行加载】逐个取结果产出权重，产出后立即释放引用降低峰值内存 ------
        for future in futures_iter:
            state = future.result()
            yield from state.items()
            del state


def convert_pyslice_to_tensor(x: Any) -> torch.Tensor:
    """convert PySafeSlice object from safetensors to torch.Tensor

    PySafeSlice object supports indexing, which is done before loading the
    actual tensor and can reduce the amount of memory being read into the
    memory. However, it does not support more advanced functionalities
    like `.view()` or `.t()`. Therefore, if we need to modify the loaded
    tensor with these more complicated operators, we need to convert to
    tensor first.
    """
    # ------【权重加载】PySafeSlice 仅按索引懒读，需真正 tensor 操作时切片物化成 torch.Tensor ------
    if not isinstance(x, torch.Tensor):
        x = x[:]
    return x




######################################################################
# 这个是实际的拷贝器
######################################################################
def default_weight_loader(param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
    """Default weight loader."""
    try:
        # ------【权重加载】标量(1 元素)权重先 view 对齐形状再拷贝，避免形状不匹配 ------
        if param.numel() == 1 and loaded_weight.numel() == 1:
            # Sometimes scalar values aren't considered tensors with shapes
            # so if both param and loaded_weight are a scalar,
            # reshape to match before copying
            param.data.copy_(loaded_weight.view(param.shape))
        else:
            # ------【权重加载】校验形状后整块拷贝，形状不符即抛错暴露命名/切分问题 ------
            assert param.size() == loaded_weight.size(), (
                f"Attempted to load weight ({loaded_weight.size()}) "
                f"into parameter ({param.size()})"
            )

            ##############################################
            # param 对应级别的module,去他的实例里面找对应的张量parameter, 这个实例，以及空的张量参数，已经在GPU了
            # loaded_weight这个是(data), 也就是从权重数据里面，把name全部切成prefix后剩下的空的data
            ##############################################
            param.data.copy_(loaded_weight) # 这里拷贝， loaded_weight是mmap出来的CPU张量， param.data是GPU的参数
    except Exception:
        # NOTE: This exception is added for the purpose of setting breakpoint to
        # debug weight loading issues.
        # ------【权重加载】捕获后原样 re-raise，仅为调试时便于在此打断点 ------
        raise


def row_parallel_weight_loader(
    param: torch.Tensor, loaded_weight: torch.Tensor
) -> None:
    """Load weights that are row-parallelized."""
    # ------【TP 权重切分】取本 rank 的 TP 序号并确定切分维（一维偏置不切分） ------
    tp_rank = get_tensor_model_parallel_rank()
    shard_dim = 0 if param.dim() != 1 else None

    if shard_dim is not None:
        # ------【TP 权重切分】按 tp_rank 从完整权重窄取出本 rank 负责的行分片 ------
        shard_size = param.data.shape[shard_dim]
        start_idx = tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(shard_dim, start_idx, shard_size)

    # ------【TP 权重切分】切分后走通用 loader 完成形状校验与拷贝 ------
    return default_weight_loader(param, loaded_weight)


LoaderFunction = Callable[[torch.Tensor, torch.Tensor], None]


def sharded_weight_loader(shard_axis: int) -> LoaderFunction:
    """Create a weight loader that shards the weights along the given axis"""

    def loader(param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
        # ------【TP 权重切分】闭包捕获 shard_axis，按 tp_rank 窄取本 rank 的分片并加载 ------
        tp_rank = get_tensor_model_parallel_rank()

        shard_size = param.data.shape[shard_axis]
        start_idx = tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(shard_axis, start_idx, shard_size)

        return default_weight_loader(param, loaded_weight)

    # ------【TP 权重切分】返回可复用的闭包 loader，调用方指定 shard_axis 即得对应切分器 ------
    return loader


def composed_weight_loader(
    loader: LoaderFunction, fn: Callable[[torch.Tensor], torch.Tensor]
) -> LoaderFunction:
    """Create a weight loader that post-processes the weights after loading"""

    def composed_loader(param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
        # ------【权重加载】先走底层 loader 装载，再对参数施加后处理（如量化反量化）回写 ------
        loader(param, loaded_weight)
        param.data.copy_(fn(param))
        return

    # ------【权重加载】返回组合后的 loader，供模型自定义权重加载逻辑复用 ------
    return composed_loader


def initialize_dummy_weights(
    model: torch.nn.Module,
    model_config: ModelConfig,
    low: float = -1e-3,
    high: float = 1e-3,
    seed: int = 1234,
) -> None:
    """Initialize model weights with random values.

    The model weights must be randomly initialized for accurate performance
    measurements. Additionally, the model weights should not cause NaNs in the
    forward pass. We empirically found that initializing the weights with
    values between -1e-3 and 1e-3 works well for most models.

    We use per-parameter random seed, so that dummy weights are consistent,
    even if the model is partitioned across multiple devices. When the seed
    is fixed, the random values generated by this function only depends on
    the parameter's number of elements and its data type.
    """
    # ------【核心逻辑】遍历全部参数做伪随机初始化，供无权重时的性能基准测试使用 ------
    for param in model.state_dict().values():
        initialize_single_dummy_weight(param, low, high, seed)


@torch.no_grad()
def initialize_single_dummy_weight(
    param: torch.Tensor,
    low: float = -1e-3,
    high: float = 1e-3,
    seed: int = 1234,
) -> None:
    # ------【meta 设备】meta 张量零初始化留待逐层加载阶段处理（如在线量化） ------
    if param.device.type == "meta":
        return  # deferred to finalize_layerwise_processing (e.g. online quant)

    # ------【量化】整型参数（如 GPTQ qweight/qzeros）不随机初始化，ROCm 上置零保证可复现 ------
    if not torch.is_floating_point(param):
        if current_platform.is_rocm():
            # On ROCm, integer params (e.g. GPTQ qweight/qzeros) are left
            # as torch.empty() by default, giving non-deterministic values
            # across processes. Zero them for reproducibility.
            param.zero_()
        return

    if current_platform.is_tpu():
        # ------【核心逻辑】TPU 上先在 CPU 生成随机数再拷贝，规避 HBM 直接 uniform 的显存压力 ------
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        # Note: The param.uniform_ function cannot be used in this
        # context because it demands more TPU HBM than directly copying
        # from a CPU tensor.
        # Note: We avoid using torch.rank_like as it doesn't currently
        # support the generator argument.
        # ------【核心逻辑】CPU 生成均匀分布张量并拷贝进参数，随后显式同步设备 ------
        param.copy_(
            (high - low)
            * torch.rand(
                param.shape,
                generator=generator,
                dtype=param.dtype,
                layout=param.layout,
                requires_grad=param.requires_grad,
                device="cpu",
            )
            + low
        )
        torch._sync(param)
        return

    # ------【核心逻辑】按参数所在设备建生成器并固定种子，保证跨卡分片时伪随机结果一致 ------
    generator = torch.Generator(device=param.data.device)
    generator.manual_seed(seed)
    # ------【量化】<16 位（如 FP8）先转 fp16 再 uniform 后转回，因 uniform_ 不支持低位宽 ------
    if torch.finfo(param.data.dtype).bits < 16:
        # uniform_ doesn't support < 16-bit datatypes (FP8)
        dtype = param.data.dtype
        tmp_param = param.data.to(torch.float16)
        tmp_param = tmp_param.uniform_(low, high, generator=generator).to(dtype)
        param.data.copy_(tmp_param)
    else:
        # ------【核心逻辑】常规浮点类型直接在参数上原地 uniform 初始化 ------
        param.uniform_(low, high, generator=generator)


def maybe_remap_kv_scale_name(name: str, params_dict: dict) -> str | None:
    """Remap the name of FP8 k/v_scale parameters.

    This function handles the remapping of FP8 k/v_scale parameter names.
    It detects if the given name ends with a suffix and attempts to remap
    it to the expected name format in the model. If the remapped name is not
    found in the params_dict, a warning is printed and None is returned.

    Args:
        name (str): The original loaded checkpoint parameter name.
        params_dict (dict): Dictionary containing the model's named parameters.

    Returns:
        str: The remapped parameter name if successful, or the original name
             if no remapping is needed.
        None: If the remapped name is not found in params_dict.
    """
    # Already in vLLM's expected form (e.g. weights pre-renamed by a
    # `WeightsMapper` from the quant config). Skip the regex remap, which
    # would otherwise double-apply the `.attn` prefix and drop the weight.
    # ------【量化】已是 vLLM 期望格式则直接返回，避免正则二次加 .attn 前缀导致权重丢失 ------
    if name in params_dict:
        return name
    # ------【量化】旧式 kv_scale 废弃：映射为 k_scale 并复制到 v_scale，缺失时报 warning ------
    if name.endswith(".kv_scale"):
        logger.warning_once(
            "DEPRECATED. Found kv_scale in the checkpoint. "
            "This format is deprecated in favor of separate k_scale and "
            "v_scale tensors and will be removed in a future release. "
            "Functionally, we will remap kv_scale to k_scale and duplicate "
            "k_scale to v_scale"
        )
        # NOTE: we remap the deprecated kv_scale to k_scale
        # ------【量化】把 .kv_scale 改写成 .attn.k_scale，若模型无该名则放弃加载 ------
        remapped_name = name.replace(".kv_scale", ".attn.k_scale")
        if remapped_name not in params_dict:
            logger.warning_once(
                "Found kv_scale in the checkpoint (e.g. %s), but not found the expected name in the model (e.g. %s). kv_scale is not loaded.",  #  noqa: E501
                name,
                remapped_name,
            )
            return None
        return remapped_name

    # ------【量化】检测是否为 MLA 注意力，据此选定 attn 子模块前缀（mla_attn 或 attn） ------
    if any("mla_attn" in key for key in params_dict):
        attn_str = "mla_attn.mla_attn"
        logger.debug_once(
            f"Found mla_attn with k_scale and v_scale in "
            f"the checkpoint, using {attn_str} as attn_str"
        )
    else:
        attn_str = "attn"
    # Define scale name mapping patterns in order of precedence
    # ------【量化】按优先级列写各厂商 scale/zero_point 命名到 vLLM 规范名的正则映射 ------
    scale_mapping_patterns = [
        # ModelOpt format: .self_attn.{k,v}_proj.{k,v}_scale ->
        # .self_attn.attn.{k,v}_scale
        (
            r"\.self_attn\.([kv])_proj\.([kv])_scale$",
            rf".self_attn.{attn_str}.\2_scale",
        ),
        # QKV proj format: .self_attn.qkv_proj.{k,v}_scale ->
        # .self_attn.attn.{k,v}_scale
        (r"\.self_attn\.qkv_proj\.([kv])_scale$", r".self_attn.attn.\1_scale"),
        # Qwen3 MoE format: .self_attn.qkqkv_proj.{k,v}_scale ->
        # .self_attn.attn.{k,v}_scale
        (r"\.self_attn\.qkqkv_proj\.([kv])_scale$", r".self_attn.attn.\1_scale"),
        # NemotronH format: .mixer.{k,v}_proj.{k,v}_scale ->
        # .mixer.attn.{k,v}_scale
        (r"\.mixer\.[kv]_proj\.([kv])_scale$", r".mixer.attn.\1_scale"),
        # HYV3 format: .self_attn.q.scale -> .self_attn.attn.q_scale
        (r"\.self_attn\.q\.scale$", r".self_attn.attn.q_scale"),
        # HYV3 format: .self_attn.{k,v}_cache.scale ->
        # .self_attn.attn.{k,v}_scale
        (r"\.self_attn\.([kv])_cache\.scale$", r".self_attn.attn.\1_scale"),
        # Default format: .{k,v}_scale -> .attn.{k,v}_scale
        (r"\.([qkv])_scale$", r".attn.\1_scale"),
        (r"\.([qkv])_zero_point$", r".attn.\1_zero_point"),
    ]

    # Check if name ends with k_scale or v_scale
    # ------【量化】仅对 scale/zero_point 类名字尝试重映射，其余直接原样返回 ------
    if name.endswith(
        (
            ".k_scale",
            ".v_scale",
            ".q_scale",
            ".k_zero_point",
            ".v_zero_point",
            ".q_zero_point",
            ".q.scale",
            ".k_cache.scale",
            ".v_cache.scale",
        )
    ):
        # ------【量化】逐个正则匹配并在命中且模型存在目标名时改写，否则警告并放弃 ------
        import regex as re

        for pattern, replacement in scale_mapping_patterns:
            if re.search(pattern, name):
                remapped_name = re.sub(pattern, replacement, name)
                if remapped_name not in params_dict:
                    scale_type = name.split(".")[-1]
                    logger.warning_once(
                        "Found %s in the checkpoint (e.g. %s), but not found the expected name in the model (e.g. %s). %s is not loaded.",  # noqa: E501
                        scale_type,
                        name,
                        remapped_name,
                        scale_type,
                    )
                    return None
                return remapped_name

    # If there were no matches, return the untouched param name
    # ------【量化】无任何匹配则原样返回，由上层 loader 决定如何处理 ------
    return name


def maybe_remap_moe_expert_param_name(
    name: str,
    params_dict: dict[str, torch.nn.Parameter],
) -> str:
    """
    Remap MoE expert parameter names to account for routed_experts hierarchy.

    This handles the transition from the old FusedMoE structure where weights
    were directly in the experts module, to the new MoERunner → RoutedExperts
    structure.

    Checkpoint weights have names like:
        layers.0.mlp.experts.w13_weight
        layers.0.feed_forward.experts.w2_input_scale
    But actual parameters are now:
        layers.0.mlp.experts.routed_experts.w13_weight
        layers.0.feed_forward.experts.routed_experts.w2_input_scale

    This function inserts 'routed_experts.' into the path when needed.

    Args:
        name: Parameter name from checkpoint
        params_dict: Dictionary of model parameters (from named_parameters())

    Returns:
        Remapped parameter name if routed_experts hierarchy exists,
        otherwise the original name
    """
    # Only remap if this looks like an expert parameter
    # ------【EP 权重切分】非 expert 参数直接返回，减少后续无谓的字符串匹配 ------
    if ".experts." not in name:
        return name

    # Skip if already has routed_experts
    # ------【EP 权重切分】已是新层级结构则无需改写，避免重复插入 routed_experts ------
    if ".experts.routed_experts." in name:
        return name

    # Expert parameter patterns to check
    # ------【EP 权重切分】枚举 expert 权重/量化参数的常见后缀，用于判断是否为 expert 张量 ------
    expert_param_suffixes = [
        "w13_weight",
        "w2_weight",
        "w13_weight_scale",
        "w2_weight_scale",
        "w13_input_scale",
        "w2_input_scale",
        "w13_bias",
        "w2_bias",
        "w13_scale",
        "w2_scale",
        "w13_g_idx",
        "w2_g_idx",
        "w13_qweight",
        "w2_qweight",
        "w13_qzeros",
        "w2_qzeros",
        "w13_weight_shape",
        "w2_weight_shape",
    ]

    # Check if this is an expert weight parameter
    # ------【EP 权重切分】命中任一后缀即视为 expert 权重，否则原样返回 ------
    is_expert_param = any(
        f".{suffix}" in name or name.endswith(suffix)
        for suffix in expert_param_suffixes
    )

    if not is_expert_param:
        return name

    # Try inserting routed_experts after .experts.
    # ------【EP 权重切分】在 .experts. 后插入 routed_experts，仅当模型存在该名时才采用 ------
    new_name = name.replace(".experts.", ".experts.routed_experts.", 1)

    # Only use the new name if it exists in the model
    if new_name in params_dict:
        return new_name

    # Otherwise return original name (old checkpoint format or different structure)
    return name


def remap_moe_expert_weights(
    weights: Iterable[tuple[str, torch.Tensor]],
    params_dict: dict[str, torch.nn.Parameter],
) -> Generator[tuple[str, torch.Tensor], None, None]:
    """
    Wrapper generator that remaps MoE expert parameter names for backward compatibility.

    This allows models with custom weight loading to automatically handle both old
    and new checkpoint formats without needing model-specific remapping code.

    Usage:
        params_dict = dict(model.named_parameters())
        for name, weight in remap_moe_expert_weights(weights, params_dict):
            # name is automatically remapped if needed
            param = params_dict[name]
            ...

    Args:
        weights: Iterator of (name, tensor) tuples from checkpoint
        params_dict: Dictionary of model parameters (from named_parameters())

    Yields:
        (remapped_name, tensor) tuples
    """
    # ------【EP 权重切分】逐条对权重名做 expert 兼容重映射，让新旧 checkpoint 格式都能对上参数 ------
    for name, weight in weights:
        remapped_name = maybe_remap_moe_expert_param_name(name, params_dict)
        yield (remapped_name, weight)
