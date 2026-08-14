# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import contextlib
import contextvars
import dataclasses
import json
import os
import tempfile
import threading
import time
from collections.abc import Generator, MutableMapping
from dataclasses import asdict, dataclass, field, fields
from typing import TYPE_CHECKING, Any, ClassVar

import regex as re
import torch
from torch import nn
from torch.utils._python_dispatch import TorchDispatchMode
from transformers import PretrainedConfig

import vllm.envs as envs
from vllm.config import ModelConfig, ParallelConfig, VllmConfig, set_current_vllm_config
from vllm.logger import init_logger
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.platforms import current_platform
from vllm.transformers_utils.repo_utils import hf_api
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.utils.import_utils import PlaceholderModule

if TYPE_CHECKING:
    from vllm.engine.arg_utils import EngineArgs

try:
    from tensorizer import (
        DecryptionParams,
        EncryptionParams,
        TensorDeserializer,
        TensorSerializer,
    )
    from tensorizer.stream_io import open_stream
    from tensorizer.utils import convert_bytes, get_mem_usage, no_init_or_tensor

except ImportError:
    tensorizer = PlaceholderModule("tensorizer")
    DecryptionParams = tensorizer.placeholder_attr("DecryptionParams")
    EncryptionParams = tensorizer.placeholder_attr("EncryptionParams")
    TensorDeserializer = tensorizer.placeholder_attr("TensorDeserializer")
    TensorSerializer = tensorizer.placeholder_attr("TensorSerializer")
    open_stream = tensorizer.placeholder_attr("stream_io.open_stream")
    convert_bytes = tensorizer.placeholder_attr("utils.convert_bytes")
    get_mem_usage = tensorizer.placeholder_attr("utils.get_mem_usage")
    no_init_or_tensor = tensorizer.placeholder_attr("utils.no_init_or_tensor")

__all__ = [
    "EncryptionParams",
    "DecryptionParams",
    "TensorDeserializer",
    "TensorSerializer",
    "open_stream",
    "convert_bytes",
    "get_mem_usage",
    "no_init_or_tensor",
    "TensorizerConfig",
]

logger = init_logger(__name__)
_TENSORIZER_ENGINE_CLEANUP_GRACE_S = 10.0


def is_valid_deserialization_uri(uri: str | None) -> bool:
    # ------【序列化】校验反序列化 URI 是否为合法来源（S3/HTTP/HTTPS 或本地文件），避免后续打开无效路径 ------
    if uri:
        scheme = uri.lower().split("://")[0]
        return scheme in {"s3", "http", "https"} or os.path.exists(uri)
    return False


def tensorizer_kwargs_arg(value):
    # ------【序列化】把 CLI 传入的 JSON 字符串解析成 dict，供序列化/反序列化 kwargs 透传 ------
    loaded = json.loads(value)
    if not isinstance(loaded, dict):
        raise argparse.ArgumentTypeError(
            f"Not deserializable to dict: {value}. serialization_kwargs and "
            f"deserialization_kwargs must be "
            f"deserializable from a JSON string to a dictionary. "
        )
    return loaded


class MetaTensorMode(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}

        # ------【meta 设备】拦截 aten::empty 并把未指定 device 的空张量改放到 meta 设备，实现零显存初始化 ------
        if func._schema.name == "aten::empty" and "device" not in kwargs:
            kwargs["device"] = "meta"

        return func(*args, **kwargs)


def meta_tensor_mode(
    loading_code=None,
):
    if loading_code is None:
        return _NoInitOrTensorImpl.context_manager()
    elif callable(loading_code):
        # ------【meta 设备】传入可调用对象时在 meta 上下文内执行它，加载到 meta 张量后直接返回结果 ------
        with _NoInitOrTensorImpl.context_manager():
            return loading_code()
    else:
        raise TypeError(
            "expected a callable to evaluate,"
            " or None if being used as a context manager;"
            f' got an object of type "{type(loading_code).__name__}" instead.'
        )


class _NoInitOrTensorImpl:
    _MODULES = (torch.nn.Linear, torch.nn.Embedding, torch.nn.LayerNorm)
    _MODULE_ORIGINALS = tuple((m, m.reset_parameters) for m in _MODULES)

    is_active = contextvars.ContextVar("_NoInitOrTensorImpl.is_active", default=False)
    _count_active: int = 0
    _count_active_lock = threading.Lock()

    @classmethod
    @contextlib.contextmanager
    def context_manager(cls):
        # ------【meta 设备】幂等守卫：已激活时直接 yield，避免嵌套调用重复给 reset_parameters 打补丁 ------
        if cls.is_active.get():
            yield
            return

        with cls._count_active_lock:
            cls._count_active += 1
            if cls._count_active == 1:
                # ------【meta 设备】仅在首次激活时把各层的 reset_parameters 替换为禁用的空实现，跳过随机初始化开销 ------
                for mod in cls._MODULES:
                    mod.reset_parameters = cls._disable(mod.reset_parameters)

        reset_token = cls.is_active.set(True)

        try:
            # ------【meta 设备】在 meta 张量模式下执行加载代码，所有空张量分配到 meta 设备实现零显存初始化 ------
            with MetaTensorMode():
                yield
        finally:
            cls.is_active.reset(reset_token)
            with cls._count_active_lock:
                cls._count_active -= 1
                if cls._count_active == 0:
                    # ------【meta 设备】退出时在计数归零后恢复各层原始的 reset_parameters，避免污染后续正常初始化 ------
                    for mod, original in cls._MODULE_ORIGINALS:
                        mod.reset_parameters = original

    @staticmethod
    def _disable(func):
        def wrapper(*args, **kwargs):
            # ------【meta 设备】激活状态下直接短路返回 None，跳过原 reset_parameters 的随机初始化 ------
            if not _NoInitOrTensorImpl.is_active.get():
                return func(*args, **kwargs)

        return wrapper


@dataclass
class TensorizerConfig(MutableMapping):
    tensorizer_uri: str | None = None
    tensorizer_dir: str | None = None
    vllm_tensorized: bool | None = None
    verify_hash: bool | None = None
    num_readers: int | None = None
    encryption_keyfile: str | None = None
    s3_access_key_id: str | None = None
    s3_secret_access_key: str | None = None
    s3_endpoint: str | None = None
    lora_dir: str | None = None
    stream_kwargs: dict[str, Any] | None = None
    serialization_kwargs: dict[str, Any] | None = None
    deserialization_kwargs: dict[str, Any] | None = None
    _extra_serialization_attrs: dict[str, Any] | None = field(init=False, default=None)
    model_class: type[torch.nn.Module] | None = field(init=False, default=None)
    hf_config: PretrainedConfig | None = field(init=False, default=None)
    dtype: str | torch.dtype | None = field(init=False, default=None)
    _is_sharded: bool = field(init=False, default=False)
    _fields: ClassVar[tuple[str, ...]]
    _keys: ClassVar[frozenset[str]]
    """Configuration class for Tensorizer settings.
    
    These settings configure the behavior of model serialization and 
    deserialization using Tensorizer.
    
    Attributes:
        tensorizer_uri: Path to serialized model tensors. Can be a local file 
            path or a S3 URI. This is a required field unless lora_dir is 
            provided and the config is meant to be used for the
            `tensorize_lora_adapter` function. Unless a `tensorizer_dir` or 
            `lora_dir` is passed to this object's initializer, this is 
            a required argument.
        tensorizer_dir: Path to a directory containing serialized model tensors,
            and all other potential model artifacts to load the model, such as 
            configs and tokenizer files. Can be passed instead of 
            `tensorizer_uri` where the `model.tensors` file will be assumed 
            to be in this directory.
        vllm_tensorized: If True, indicates that the serialized model is a 
            vLLM model. This is used to determine the behavior of the 
            TensorDeserializer when loading tensors from a serialized model.
            It is far faster to deserialize a vLLM model as it utilizes
            tensorizer's optimized GPU loading. Note that this is now
            deprecated, as serialized vLLM models are now automatically
            inferred as vLLM models.
        verify_hash: If True, the hashes of each tensor will be verified 
            against the hashes stored in the metadata. A `HashMismatchError` 
            will be raised if any of the hashes do not match.
        num_readers: Controls how many threads are allowed to read concurrently
            from the source file. Default is `None`, which will dynamically set
            the number of readers based on the number of available 
            resources and model size. This greatly increases performance.
        encryption_keyfile: File path to a binary file containing a  
            binary key to use for decryption. `None` (the default) means 
            no decryption. See the example script in 
            examples/features/tensorize_vllm_model.py. 
        s3_access_key_id: The access key for the S3 bucket. Can also be set via
            the S3_ACCESS_KEY_ID environment variable.
        s3_secret_access_key: The secret access key for the S3 bucket. Can also
            be set via the S3_SECRET_ACCESS_KEY environment variable.
        s3_endpoint: The endpoint for the S3 bucket. Can also be set via the
            S3_ENDPOINT_URL environment variable.
        lora_dir: Path to a directory containing LoRA adapter artifacts for 
            serialization or deserialization. When serializing LoRA adapters 
            this is the only necessary parameter to pass to this object's 
            initializer.
    """

    def __post_init__(self):
        # check if the configuration is for a sharded vLLM model
        # ------【TP 权重切分】检测 tensorizer_uri 是否含 %0dd 分片模板，标记为 TP 分片权重模型 ------
        self._is_sharded = (
            isinstance(self.tensorizer_uri, str)
            and re.search(r"%0\dd", self.tensorizer_uri) is not None
        )

        # ------【序列化】校验 tensorizer_dir 与 lora_dir 互斥，LoRA 序列化只能单独指定 lora_dir ------
        if self.tensorizer_dir and self.lora_dir:
            raise ValueError(
                "Only one of tensorizer_dir or lora_dir may be specified. "
                "Use lora_dir exclusively when serializing LoRA adapters, "
                "and tensorizer_dir or tensorizer_uri otherwise."
            )
        # ------【序列化】两者同时给出时以 tensorizer_uri 为准，据此反推 tensorizer_dir ------
        if self.tensorizer_dir and self.tensorizer_uri:
            logger.warning_once(
                "Provided both tensorizer_dir and tensorizer_uri. "
                "Inferring tensorizer_dir from tensorizer_uri as the "
                "latter takes precedence."
            )
            self.tensorizer_dir = os.path.dirname(self.tensorizer_uri)
        # ------【序列化】未显式给 tensorizer_uri 时按 lora_dir/tensorizer_dir 约定推导默认文件路径，缺省则报错 ------
        if not self.tensorizer_uri:
            if self.lora_dir:
                self.tensorizer_uri = f"{self.lora_dir}/adapter_model.tensors"
            elif self.tensorizer_dir:
                self.tensorizer_uri = f"{self.tensorizer_dir}/model.tensors"
            else:
                raise ValueError(
                    "Unable to resolve tensorizer_uri. "
                    "A valid tensorizer_uri or tensorizer_dir "
                    "must be provided for deserialization, and a "
                    "valid tensorizer_uri, tensorizer_uri, or "
                    "lora_dir for serialization."
                )
        else:
            # ------【序列化】从 tensorizer_uri 反推 tensorizer_dir，保证目录信息始终可用 ------
            self.tensorizer_dir = os.path.dirname(self.tensorizer_uri)

        # ------【序列化】为 serialization/deserialization kwargs 兜底空 dict，避免下游解包时抛 None 异常 ------
        if not self.serialization_kwargs:
            self.serialization_kwargs = {}
        if not self.deserialization_kwargs:
            self.deserialization_kwargs = {}

    def to_serializable(self) -> dict[str, Any]:
        # Due to TensorizerConfig needing to be msgpack-serializable, it needs
        # support for morphing back and forth between itself and its dict
        # representation

        # TensorizerConfig's representation as a dictionary is meant to be
        # linked to TensorizerConfig in such a way that the following is
        # technically initializable:
        # TensorizerConfig(**my_tensorizer_cfg.to_serializable())

        # This means the dict must not retain non-initializable parameters
        # and post-init attribute states

        # Also don't want to retain private and unset parameters, so only retain
        # not None values and public attributes

        raw_tc_dict = asdict(self)
        blacklisted = []

        # ------【序列化】uri 与 dir 同时存在时剔除冗余的 tensorizer_dir，保证 dict 可直接再初始化 ------
        if "tensorizer_uri" in raw_tc_dict and "tensorizer_dir" in raw_tc_dict:
            blacklisted.append("tensorizer_dir")

        if "tensorizer_dir" in raw_tc_dict and "lora_dir" in raw_tc_dict:
            blacklisted.append("tensorizer_dir")

        tc_dict = {}
        for k, v in raw_tc_dict.items():
            # ------【序列化】过滤黑名单/私有/None 字段，只保留公开且已设置的参数以支持往返序列化 ------
            if (
                k not in blacklisted
                and k not in tc_dict
                and not k.startswith("_")
                and v is not None
            ):
                tc_dict[k] = v

        return tc_dict

    def _construct_tensorizer_args(self) -> "TensorizerArgs":
        return TensorizerArgs(self)  # type: ignore

    def verify_with_parallel_config(
        self,
        parallel_config: "ParallelConfig",
    ) -> None:
        # ------【TP 权重切分】TP>1 时要求 uri 含分片模板，否则序列化/反序列化无法按 rank 找到对应分片 ------
        if parallel_config.tensor_parallel_size > 1 and not self._is_sharded:
            raise ValueError(
                "For a sharded model, tensorizer_uri should include a"
                " string format template like '%04d' to be formatted"
                " with the rank of the shard"
            )

    def verify_with_model_config(self, model_config: "ModelConfig") -> None:
        # ------【量化】量化模型用 tensorizer 反序列化不稳定，提前打警告提示可能出错 ------
        if model_config.quantization is not None and self.tensorizer_uri is not None:
            logger.warning(
                "Loading a model using Tensorizer with quantization on vLLM"
                " is unstable and may lead to errors."
            )

    def open_stream(self, tensorizer_args: "TensorizerArgs | None" = None):
        if tensorizer_args is None:
            tensorizer_args = self._construct_tensorizer_args()

        # ------【序列化】用配置组装 stream_kwargs 打开目标流，支持本地文件/S3/HTTP(S) 读写 ------
        return open_stream(self.tensorizer_uri, **tensorizer_args.stream_kwargs)

    def keys(self):
        return self._keys

    def __len__(self):
        return len(fields(self))

    def __iter__(self):
        return iter(self._fields)

    def __getitem__(self, item: str) -> Any:
        # ------【序列化】仅允许读取已声明的字段，非法 key 抛 KeyError 保证 MutableMapping 契约 ------
        if item not in self.keys():
            raise KeyError(item)
        return getattr(self, item)

    def __setitem__(self, key: str, value: Any) -> None:
        if key not in self.keys():
            # Disallow modifying invalid keys
            raise KeyError(key)
        setattr(self, key, value)

    def __delitem__(self, key, /):
        # ------【序列化】仅允许删除已声明的字段，维持配置对象的字段集合一致 ------
        if key not in self.keys():
            raise KeyError(key)
        delattr(self, key)


TensorizerConfig._fields = tuple(f.name for f in fields(TensorizerConfig))
TensorizerConfig._keys = frozenset(TensorizerConfig._fields)


@dataclass
class TensorizerArgs:
    tensorizer_uri: str | None = None
    tensorizer_dir: str | None = None
    encryption_keyfile: str | None = None

    def __init__(self, tensorizer_config: TensorizerConfig):
        # ------【序列化】把 TensorizerConfig 的字段批量拷贝到 args 对象，作为序列化/反序列化的输入参数 ------
        for k, v in tensorizer_config.items():
            setattr(self, k, v)
        self.file_obj = tensorizer_config.tensorizer_uri
        # ------【序列化】S3 凭证优先取显式配置，否则回退到环境变量，支持多种部署环境注入 ------
        self.s3_access_key_id = (
            tensorizer_config.s3_access_key_id or envs.S3_ACCESS_KEY_ID
        )
        self.s3_secret_access_key = (
            tensorizer_config.s3_secret_access_key or envs.S3_SECRET_ACCESS_KEY
        )
        self.s3_endpoint = tensorizer_config.s3_endpoint or envs.S3_ENDPOINT_URL

        # ------【序列化】组装打开流的 S3 连接参数，并合并用户自定义 stream_kwargs 覆盖默认值 ------
        self.stream_kwargs = {
            "s3_access_key_id": tensorizer_config.s3_access_key_id,
            "s3_secret_access_key": tensorizer_config.s3_secret_access_key,
            "s3_endpoint": tensorizer_config.s3_endpoint,
            **(tensorizer_config.stream_kwargs or {}),
        }

        # ------【并行加载+序列化】组装反序列化参数：校验哈希/解密密钥/并发读线程数，并合并自定义覆盖 ------
        self.deserialization_kwargs = {
            "verify_hash": tensorizer_config.verify_hash,
            "encryption": tensorizer_config.encryption_keyfile,
            "num_readers": tensorizer_config.num_readers,
            **(tensorizer_config.deserialization_kwargs or {}),
        }

        if self.encryption_keyfile:
            # ------【序列化】若提供密钥文件则读入密钥并构造解密参数，覆盖 deserialization_kwargs 中的 encryption ------
            with open_stream(
                tensorizer_config.encryption_keyfile,
                **self.stream_kwargs,
            ) as stream:
                key = stream.read()
                decryption_params = DecryptionParams.from_key(key)
                self.deserialization_kwargs["encryption"] = decryption_params

    @staticmethod
    def add_cli_args(parser: FlexibleArgumentParser) -> FlexibleArgumentParser:
        """Tensorizer CLI arguments"""

        # Tensorizer options arg group
        # ------【序列化】创建 tensorizer 专用参数组，集中暴露序列化/反序列化相关的命令行开关 ------
        group = parser.add_argument_group(
            "tensorizer options",
            description=(
                "Options for configuring the behavior of the"
                " tensorizer deserializer when "
                "load_format=tensorizer is specified when "
                "initializing an LLMEngine, either via the CLI "
                "when running the vLLM OpenAI inference server "
                "with a JSON string passed to "
                "--model-loader-extra-config or as arguments given "
                "to TensorizerConfig when passed to "
                "model_loader_extra_config in the constructor "
                "for LLMEngine."
            ),
        )

        group.add_argument(
            "--tensorizer-uri",
            type=str,
            help="Path to serialized model tensors. Can be a local file path,"
            " or an HTTP(S) or S3 URI.",
        )
        group.add_argument(
            "--verify-hash",
            action="store_true",
            help="If enabled, the hashes of each tensor will be verified"
            " against the hashes stored in the file metadata. An exception"
            " will be raised if any of the hashes do not match.",
        )
        group.add_argument(
            "--encryption-keyfile",
            type=str,
            default=None,
            help="The file path to a binary file containing a binary key to "
            "use for decryption. Can be a file path or S3 network URI.",
        )
        group.add_argument(
            "--num-readers",
            default=None,
            type=int,
            help="Controls how many threads are allowed to read concurrently "
            "from the source file. Default is `None`, which will dynamically "
            "set the number of readers based on the available resources "
            "and model size. This greatly increases performance.",
        )
        group.add_argument(
            "--s3-access-key-id",
            type=str,
            default=None,
            help="The access key for the S3 bucket. Can also be set via the "
            "S3_ACCESS_KEY_ID environment variable.",
        )
        group.add_argument(
            "--s3-secret-access-key",
            type=str,
            default=None,
            help="The secret access key for the S3 bucket. Can also be set via "
            "the S3_SECRET_ACCESS_KEY environment variable.",
        )
        group.add_argument(
            "--s3-endpoint",
            type=str,
            default=None,
            help="The endpoint for the S3 bucket. Can also be set via the "
            "S3_ENDPOINT_URL environment variable.",
        )

        return parser

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> "TensorizerArgs":
        attrs = [attr.name for attr in dataclasses.fields(cls)]
        # ------【序列化】仅把 CLI 命名空间中存在的 dataclass 字段映射进 args，跳过未提供的参数 ------
        tensorizer_args = cls(
            **{attr: getattr(args, attr) for attr in attrs if hasattr(args, attr)}
        )
        return tensorizer_args


def _check_tensors_on_meta_device(model: nn.Module) -> None:
    for tensor in model.state_dict().values():
        # ------【meta 设备】遍历 state_dict，若仍有 meta 张量说明反序列化漏加载，报错定位参数不匹配 ------
        if tensor.device.type == "meta":
            raise ValueError(
                "The serialized model contains tensors on the meta device,"
                " indicating that some tensors were not loaded properly."
                " Please check that the parameters of the model being"
                " specified match that of the serialized model, such as"
                " its quantization."
            )


def _resize_lora_embeddings(model: nn.Module):
    """Modify LoRA embedding layers to use bigger tensors
    to allow for adapter added tokens."""
    for child in model.modules():
        if (
            isinstance(child, VocabParallelEmbedding)
            and child.weight.shape[0] < child.num_embeddings_per_partition
        ):
            # ------【LoRA】LoRA 新增 token 需更大词表，先分配扩展后的空张量以容纳 adapter 新增 embedding ------
            new_weight = torch.empty(
                child.num_embeddings_per_partition,
                child.embedding_dim,
                dtype=child.weight.dtype,
                device=child.weight.device,
            )
            new_weight[: child.weight.shape[0]].copy_(child.weight.data)
            new_weight[child.weight.shape[0] :].fill_(0)
            child.weight.data = new_weight


def init_tensorizer_model(
    tensorizer_config: TensorizerConfig, vllm_config: VllmConfig
) -> nn.Module:
    assert tensorizer_config.hf_config is not None
    model_args = tensorizer_config.hf_config
    model_args.dtype = tensorizer_config.dtype
    assert tensorizer_config.model_class is not None
    # TODO: Do we need to consider old-style model class?
    # ------【meta 设备】在 meta 模式与 vllm_config 上下文中实例化模型，只建图不分配显存，供后续灌权重 ------
    with meta_tensor_mode(), set_current_vllm_config(vllm_config, check_compile=True):
        return tensorizer_config.model_class(vllm_config=vllm_config)


def deserialize_tensorizer_model(
    model: nn.Module, tensorizer_config: TensorizerConfig
) -> None:
    tensorizer_args = tensorizer_config._construct_tensorizer_args()
    # ------【序列化】反序列化前校验 URI 合法，避免后续打开无效路径时抛出难以定位的错误 ------
    if not is_valid_deserialization_uri(tensorizer_config.tensorizer_uri):
        raise ValueError(
            f"{tensorizer_config.tensorizer_uri} is not a valid "
            f"tensorizer URI. Please check that the URI is correct. "
            f"It must either point to a local existing file, or have a "
            f"S3, HTTP or HTTPS scheme."
        )
    before_mem = get_mem_usage()
    start = time.perf_counter()
    device_index = torch.accelerator.current_device_index()
    device_type = current_platform.device_type
    # ------【权重加载】打开流并构造 TensorDeserializer，直接按 dtype/device 把权重灌入模型（支持 GPU 直载） ------
    with (
        open_stream(
            tensorizer_config.tensorizer_uri, mode="rb", **tensorizer_args.stream_kwargs
        ) as stream,
        TensorDeserializer(
            stream,
            dtype=tensorizer_config.dtype,
            device=f"{device_type}:{device_index}",
            **tensorizer_args.deserialization_kwargs,
        ) as deserializer,
    ):
        deserializer.load_into_module(model)
        end = time.perf_counter()

    # ------【序列化】统计反序列化字节数、耗时与吞吐，用于对比不同配置下的加载性能 ------
    total_bytes_str = convert_bytes(deserializer.total_tensor_bytes)
    duration = end - start
    per_second = convert_bytes(deserializer.total_tensor_bytes / duration)
    after_mem = get_mem_usage()
    deserializer.close()
    logger.info(
        "Deserialized %s in %0.2fs, %s/s", total_bytes_str, end - start, per_second
    )
    logger.info("Memory usage before: %s", before_mem)
    logger.info("Memory usage after: %s", after_mem)

    # ------【meta 设备】反序列化后校验是否残留 meta 张量，并扩容 LoRA embedding 再清除临时 marker ------
    _check_tensors_on_meta_device(model)
    _resize_lora_embeddings(model)
    del model.vllm_tensorized_marker


def tensorizer_weights_iterator(
    tensorizer_args: "TensorizerArgs",
) -> Generator[tuple[str, torch.Tensor], None, None]:
    logger.warning(
        "Deserializing HuggingFace models is not optimized for "
        "loading on vLLM, as tensorizer is forced to load to CPU. "
        "Consider deserializing a vLLM model instead for faster "
        "load times. See the "
        "examples/features/tensorize_vllm_model.py example script "
        "for serializing vLLM models."
    )

    deserializer_args = tensorizer_args.deserialization_kwargs
    stream_kwargs = tensorizer_args.stream_kwargs
    stream = open_stream(tensorizer_args.tensorizer_uri, **stream_kwargs)
    # ------【权重加载】强制 CPU 反序列化并逐个 yield 权重张量，供 HF 模型加载路径迭代使用 ------
    with TensorDeserializer(stream, **deserializer_args, device="cpu") as state:
        yield from state.items()
    del state


def is_vllm_tensorized(tensorizer_config: "TensorizerConfig") -> bool:
    """
    Infer if the model is a vLLM model by checking the weights for
    a vLLM tensorized marker.

    Args:
        tensorizer_config: The TensorizerConfig object containing the
            tensorizer_uri to the serialized model.

    Returns:
        bool: True if the model is a vLLM model, False otherwise.
    """
    tensorizer_args = tensorizer_config._construct_tensorizer_args()
    # ------【序列化】用 lazy_load 打开反序列化器，仅读元数据不加载张量，用于快速判断模型类型 ------
    deserializer = TensorDeserializer(
        open_stream(tensorizer_args.tensorizer_uri, **tensorizer_args.stream_kwargs),
        **tensorizer_args.deserialization_kwargs,
        lazy_load=True,
    )
    if tensorizer_config.vllm_tensorized:
        logger.warning(
            "Please note that newly serialized vLLM models are automatically "
            "inferred as vLLM models, so setting vllm_tensorized=True is "
            "only necessary for models serialized prior to this change."
        )
        return True
    # ------【序列化】通过检查序列化文件中是否存在 vllm 标记张量来推断是否为 vLLM 模型 ------
    return ".vllm_tensorized_marker" in deserializer


def serialize_extra_artifacts(
    tensorizer_args: TensorizerArgs, served_model_name: str | list[str] | None
) -> None:
    if not isinstance(served_model_name, str):
        raise ValueError(
            f"served_model_name must be a str for serialize_extra_artifacts, "
            f"not {type(served_model_name)}."
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        # ------【下载缓存】从 HF 下载除权重外的配置/分词器等工件到临时目录，权重由 tensorizer 单独处理 ------
        hf_api().snapshot_download(
            served_model_name,
            local_dir=tmpdir,
            ignore_patterns=[
                "*.pt",
                "*.safetensors",
                "*.bin",
                "*.cache",
                "*.gitattributes",
                "*.md",
            ],
        )
        for artifact in os.scandir(tmpdir):
            if not artifact.is_file():
                continue
            # ------【序列化】把每个工件逐字节写入 tensorizer 目录，与序列化权重一起构成完整可加载包 ------
            with (
                open(artifact.path, "rb") as f,
                open_stream(
                    f"{tensorizer_args.tensorizer_dir}/{artifact.name}",
                    mode="wb+",
                    **tensorizer_args.stream_kwargs,
                ) as stream,
            ):
                logger.info("Writing artifact %s", artifact.name)
                stream.write(f.read())


def serialize_vllm_model(
    model: nn.Module,
    tensorizer_config: TensorizerConfig,
    model_config: "ModelConfig",
) -> nn.Module:
    # ------【序列化】给模型注册 meta 设备上的 marker 参数，用于反序列化时识别 vLLM 序列化模型 ------
    model.register_parameter(
        "vllm_tensorized_marker",
        nn.Parameter(torch.tensor((1,), device="meta"), requires_grad=False),
    )

    tensorizer_args = tensorizer_config._construct_tensorizer_args()

    encryption_params = None
    # ------【序列化】若配置加密密钥文件则读入并构造 EncryptionParams，序列化时对权重加密 ------
    if (keyfile := tensorizer_config.encryption_keyfile) is not None:
        with open(keyfile, "rb") as f:
            key = f.read()
        encryption_params = EncryptionParams(key=key)

    if (output_file := tensorizer_args.tensorizer_uri) is None:
        raise ValueError("tensorizer_uri must be specified for serialization.")
    # ------【TP 权重切分】分片模型按 TP rank 格式化输出路径，使每个 rank 写出各自的权重分片文件 ------
    if tensorizer_config._is_sharded:
        from vllm.distributed import get_tensor_model_parallel_rank

        output_file = output_file % get_tensor_model_parallel_rank()

    with open_stream(
        output_file, mode="wb+", **tensorizer_args.stream_kwargs
    ) as stream:
        # ------【序列化】构造 TensorSerializer 并整体写出模型参数（含加密），完成权重序列化 ------
        serializer = TensorSerializer(
            stream,
            encryption=encryption_params,
            **(tensorizer_config.serialization_kwargs or {}),
        )
        serializer.write_module(model)
        serializer.close()

    serialize_extra_artifacts(tensorizer_args, model_config.served_model_name)

    logger.info("Successfully serialized model to %s", str(output_file))
    return model


def tensorize_vllm_model(
    engine_args: "EngineArgs",
    tensorizer_config: TensorizerConfig,
    generate_keyfile: bool = True,
):
    """Utility to load a model and then serialize it with Tensorizer

    Intended to be used separately from running a vLLM server since it
    creates its own Engine instance.
    """
    engine_config = engine_args.create_engine_config()
    # ------【核心逻辑】先校验模型/并行配置，确保后续序列化行为与运行配置一致再启动 ------
    tensorizer_config.verify_with_model_config(engine_config.model_config)
    tensorizer_config.verify_with_parallel_config(engine_config.parallel_config)

    # generate the encryption key before creating the engine to support sharding
    # ------【序列化】在创建 engine 前生成并写入随机加密密钥，保证各分片 worker 复用同一把密钥 ------
    if (
        generate_keyfile
        and (keyfile := tensorizer_config.encryption_keyfile) is not None
    ):
        encryption_params = EncryptionParams.random()
        with open_stream(
            keyfile,
            mode="wb+",
            s3_access_key_id=tensorizer_config.s3_access_key_id,
            s3_secret_access_key=tensorizer_config.s3_secret_access_key,
            s3_endpoint=tensorizer_config.s3_endpoint,
        ) as stream:
            stream.write(encryption_params.key)

    from vllm.v1.engine.llm_engine import LLMEngine

    # ------【序列化】从配置创建引擎实例，以便在各 worker 上真正加载模型用于序列化 ------
    engine = LLMEngine.from_vllm_config(engine_config)
    error: BaseException | None = None
    try:
        # ------【异步 RPC】通过 collective_rpc 广播 save_tensorized_model，让每个 worker 并行序列化自己的权重分片 ------
        engine.collective_rpc(
            "save_tensorized_model",
            kwargs={"tensorizer_config": tensorizer_config.to_serializable()},
        )
    except BaseException as operation_error:
        error = operation_error

    def shutdown_engine_core() -> None:
        # ------【进程管理】序列化完成后关闭 engine core，附加额外宽限时间等待清理完成 ------
        engine.engine_core.shutdown(
            timeout=(
                envs.VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS
                + _TENSORIZER_ENGINE_CLEANUP_GRACE_S
            )
        )

    for name, callback in (
        ("renderer", engine.renderer.shutdown),
        ("engine core", shutdown_engine_core),
    ):
        try:
            callback()
        except BaseException as shutdown_error:
            # ------【进程管理】逐个关闭 renderer 与 engine core，并把关闭失败信息汇总到 error 以便最终抛出 ------
            logger.exception("Failed to shut down tensorization %s", name)
            if error is None:
                error = shutdown_error
            elif hasattr(error, "add_note"):
                error.add_note(f"{name} shutdown also failed: {shutdown_error!r}")

    if error is not None:
        raise error


def tensorize_lora_adapter(lora_path: str, tensorizer_config: TensorizerConfig):
    """
    Uses tensorizer to serialize a LoRA adapter. Assumes that the files
    needed to load a LoRA adapter are a safetensors-format file called
    adapter_model.safetensors and a json config file called adapter_config.json.

    Serializes the files in the tensorizer_config.tensorizer_dir
    """
    import safetensors

    from vllm.lora.utils import get_adapter_absolute_path

    lora_dir = get_adapter_absolute_path(lora_path)

    tensor_path = config_path = ""

    # ------【LoRA】扫描 adapter 目录，定位 adapter_model 权重与 adapter_config 配置文件路径 ------
    for file in os.listdir(lora_dir):
        if file.startswith("adapter_model"):
            tensor_path = lora_dir + "/" + file
        if file.startswith("adapter_config"):
            config_path = lora_dir + "/" + file
        if tensor_path and config_path:
            break

    # ------【LoRA+量化】按后缀选择 safetensors 或 torch.bin 加载 LoRA 权重，不支持其他格式则报错 ------
    if tensor_path.endswith(".safetensors"):
        tensors = safetensors.torch.load_file(tensor_path)
    elif tensor_path.endswith(".bin"):
        tensors = torch.load(tensor_path, weights_only=True)
    else:
        raise ValueError(
            f"Unsupported adapter model file: {tensor_path}. "
            f"Must be a .safetensors or .bin file."
        )

    # ------【序列化】读入 adapter 配置 JSON，后续原样写回 tensorizer 目录 ------
    with open(config_path) as f:
        config = json.load(f)

    tensorizer_args = tensorizer_config._construct_tensorizer_args()

    # ------【序列化】把 LoRA 配置 JSON 写入 tensorizer 目录，构成完整 adapter 工件 ------
    with open_stream(
        f"{tensorizer_config.tensorizer_dir}/adapter_config.json",
        mode="wb+",
        **tensorizer_args.stream_kwargs,
    ) as f:
        f.write(json.dumps(config).encode("utf-8"))

    lora_uri = f"{tensorizer_config.tensorizer_dir}/adapter_model.tensors"
    # ------【LoRA+序列化】用 TensorSerializer 将 LoRA 权重 state_dict 序列化写出，完成 adapter 落盘 ------
    with open_stream(lora_uri, mode="wb+", **tensorizer_args.stream_kwargs) as f:
        serializer = TensorSerializer(f)
        serializer.write_state_dict(tensors)
        serializer.close()

    logger.info(
        "Successfully serialized LoRA files to %s",
        str(tensorizer_config.tensorizer_dir),
    )
