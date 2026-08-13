# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib
from collections.abc import Callable
from typing import TYPE_CHECKING, cast

from vllm.config.kv_transfer import KVTransferConfig
from vllm.distributed.kv_transfer.kv_connector.base import (
    KVConnectorBase,
    KVConnectorBaseType,
)
from vllm.distributed.kv_transfer.kv_connector.v1 import (
    KVConnectorRole,
    supports_hma,
)
from vllm.logger import init_logger
from vllm.utils.func_utils import supports_kw

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import KVCacheConfig

logger = init_logger(__name__)


class KVConnectorFactory:
    _registry: dict[str, Callable[[], type[KVConnectorBase]]] = {}

    @classmethod
    def register_connector(cls, name: str, module_path: str, class_name: str) -> None:
        """Register a connector with a lazy-loading module and class name."""
        # ------【PD 分离】去重检查：同名 connector 不允许重复注册，避免覆盖已有条目 ------
        if name in cls._registry:
            raise ValueError(f"Connector '{name}' is already registered.")

        # ------【PD 分离+懒加载】闭包 loader 延迟 import 模块，注册时不加载、实例化时才 import ------
        def loader() -> type[KVConnectorBase]:
            module = importlib.import_module(module_path)
            return getattr(module, class_name)

        # ------【PD 分离】把名字映射到 loader，注册表存储按需加载的工厂函数 ------
        cls._registry[name] = loader

    @classmethod
    def create_connector(
        cls,
        config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig",
    ) -> KVConnectorBase:
        # ------【PD 分离】取出 KV transfer 配置并校验非空，没配置就无法创建 connector ------
        kv_transfer_config = config.kv_transfer_config
        if kv_transfer_config is None:
            raise ValueError("kv_transfer_config must be set to create a connector")
        # ------【PD 分离】解析出具体 connector 类（外部模块路径优先于内置注册表） ------
        connector_cls = cls.get_connector_class(kv_transfer_config)

        # check if the connector supports HMA
        # ------【PD 分离】HMA(混合 KV cache 管理器)能力校验：启用 HMA 但 connector 不支持则报错 ------
        hma_enabled = not config.scheduler_config.disable_hybrid_kv_cache_manager
        if hma_enabled and not cls.supports_hma_config(kv_transfer_config):
            raise ValueError(
                f"Connector {connector_cls.__name__} does not support HMA but "
                f"HMA is enabled. Please set `--disable-hybrid-kv-cache-manager`."
            )

        # ------【PD 分离】打印创建日志，记录 connector 名与 engine_id 便于追踪 ------
        logger.info(
            "Creating v1 connector with name: %s and engine_id: %s",
            connector_cls.__name__,
            kv_transfer_config.engine_id,
        )
        # NOTE(Kuntai): v1 connector is explicitly separated into two roles.
        # Scheduler connector:
        # - Co-locate with scheduler process
        # - Should only be used inside the Scheduler class
        # Worker connector:
        # - Co-locate with worker process
        # - Should only be used inside the forward context & attention layer
        # We build separately to enforce strict separation
        # ------【PD 分离】按 role(scheduler/worker)分别实例化，强制隔离两个进程侧的 connector ------
        return connector_cls(config, role, kv_cache_config)

    @classmethod
    def get_connector_class_by_name(
        cls, connector_name: str
    ) -> type[KVConnectorBaseType]:
        """Get a registered connector class by name.

        Raises ValueError if the connector is not registered.

        Args:
            connector_name: Name of the registered connector.

        Returns:
            The connector class.
        """
        # ------【PD 分离】未注册的 connector 直接报错，快速暴露配置错误 ------
        if connector_name not in cls._registry:
            raise ValueError(f"Connector '{connector_name}' is not registered.")
        # ------【PD 分离】调用 loader 真正 import 并返回类，实现按需懒加载 ------
        return cls._registry[connector_name]()

    @classmethod
    def get_connector_class(
        cls, kv_transfer_config: "KVTransferConfig"
    ) -> type[KVConnectorBaseType]:
        # ------【PD 分离】取出 connector 名并校验非空，名字是后续查找的键 ------
        connector_name = kv_transfer_config.kv_connector
        if connector_name is None:
            raise ValueError("Connector name is not set in KVTransferConfig")
        # ------【PD 分离】校验外部模块路径：不允许为空字符串，避免 import 空路径 ------
        connector_module_path = kv_transfer_config.kv_connector_module_path
        if connector_module_path is not None and not connector_module_path:
            raise ValueError("kv_connector_module_path cannot be an empty string.")
        # ------【PD 分离+懒加载】外部模块路径优先于内置注册表，从外部路径 import 用户自定义 connector ------
        if connector_module_path:
            # External module path takes priority over internal registry.
            connector_module = importlib.import_module(connector_module_path)
            try:
                connector_cls = getattr(connector_module, connector_name)
            except AttributeError as e:
                raise AttributeError(
                    f"Class {connector_name} not found in {connector_module_path}"
                ) from e
            connector_cls = cast(type[KVConnectorBaseType], connector_cls)
            # ------【PD 分离】兼容性校验：外部 connector 必须支持 kv_cache_config 三参构造签名 ------
            if not supports_kw(connector_cls, "kv_cache_config"):
                msg = (
                    f"Connector {connector_cls.__name__} uses deprecated "
                    "2-argument constructor signature. External v1 KV "
                    "connectors must accept kv_cache_config as the third "
                    "constructor argument and pass it to super().__init__()."
                )
                logger.error(msg)
                raise ValueError(msg)
        elif connector_name in cls._registry:
            # ------【PD 分离】内置注册表命中：调用 loader 懒加载类 ------
            connector_cls = cls._registry[connector_name]()
        else:
            # ------【PD 分离】外部路径未给且注册表也未命中，报不支持该类型 ------
            raise ValueError(f"Unsupported connector type: {connector_name}")
        return connector_cls

    @classmethod
    def supports_hma_config(cls, kv_transfer_config: "KVTransferConfig") -> bool:
        """Return whether this KV transfer config supports HMA.

        MultiConnector is a special case: the wrapper class implements
        SupportsHMA, but effective support depends on every configured child.
        """
        # ------【PD 分离】解析 connector 类，再判断其 HMA 支持能力 ------
        connector_cls = cls.get_connector_class(kv_transfer_config)
        # ------【PD 分离】普通 connector：直接用 SupportsHMA 协议判断单类支持 ------
        if kv_transfer_config.kv_connector != "MultiConnector":
            return supports_hma(connector_cls)

        from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import (
            MultiConnector,
        )

        # ------【PD 分离】MultiConnector 特例：需递归检查每个子 connector 是否都支持 HMA ------
        return MultiConnector.all_children_support_hma(kv_transfer_config)


# Register various connectors here.
# The registration should not be done in each individual file, as we want to
# only load the files corresponding to the current connector.

KVConnectorFactory.register_connector(
    "ExampleConnector",
    "vllm.distributed.kv_transfer.kv_connector.v1.example_connector",
    "ExampleConnector",
)

KVConnectorFactory.register_connector(
    "ExampleHiddenStatesConnector",
    "vllm.distributed.kv_transfer.kv_connector.v1.example_hidden_states_connector",
    "ExampleHiddenStatesConnector",
)

KVConnectorFactory.register_connector(
    "LMCacheConnectorV1",
    "vllm.distributed.kv_transfer.kv_connector.v1.lmcache_connector",
    "LMCacheConnectorV1",
)

KVConnectorFactory.register_connector(
    "LMCacheMPConnector",
    "vllm.distributed.kv_transfer.kv_connector.v1.lmcache_mp_connector",
    "LMCacheMPConnector",
)

KVConnectorFactory.register_connector(
    "NixlConnector",
    "vllm.distributed.kv_transfer.kv_connector.v1.nixl",
    "NixlConnector",
)

KVConnectorFactory.register_connector(
    "NixlPullConnector",
    "vllm.distributed.kv_transfer.kv_connector.v1.nixl",
    "NixlPullConnector",
)

KVConnectorFactory.register_connector(
    "NixlPushConnector",
    "vllm.distributed.kv_transfer.kv_connector.v1.nixl",
    "NixlPushConnector",
)

KVConnectorFactory.register_connector(
    "MultiConnector",
    "vllm.distributed.kv_transfer.kv_connector.v1.multi_connector",
    "MultiConnector",
)

KVConnectorFactory.register_connector(
    "MoRIIOConnector",
    "vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_connector",
    "MoRIIOConnector",
)

KVConnectorFactory.register_connector(
    "OffloadingConnector",
    "vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector",
    "OffloadingConnector",
)

KVConnectorFactory.register_connector(
    "DecodeBenchConnector",
    "vllm.distributed.kv_transfer.kv_connector.v1.decode_bench_connector",
    "DecodeBenchConnector",
)

KVConnectorFactory.register_connector(
    "MooncakeConnector",
    "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector",
    "MooncakeConnector",
)
KVConnectorFactory.register_connector(
    "MooncakeStoreConnector",
    "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.connector",
    "MooncakeStoreConnector",
)
KVConnectorFactory.register_connector(
    "FlexKVConnectorV1",
    "vllm.distributed.kv_transfer.kv_connector.v1.flexkv_connector",
    "FlexKVConnectorV1",
)
KVConnectorFactory.register_connector(
    "SimpleCPUOffloadConnector",
    "vllm.distributed.kv_transfer.kv_connector.v1.simple_cpu_offload_connector",
    "SimpleCPUOffloadConnector",
)
KVConnectorFactory.register_connector(
    "HF3FSKVConnector",
    "vllm.distributed.kv_transfer.kv_connector.v1.hf3fs.hf3fs_connector",
    "HF3FSKVConnector",
)
