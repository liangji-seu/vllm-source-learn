# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import msgspec


# ------【LoRA】LoRARequest：打包进 Request/EngineCoreRequest 的 LoRA 适配器标识消息，承载适配器定位与加载选项 ------
class LoRARequest(
    msgspec.Struct,
    omit_defaults=True,  # type: ignore[call-arg]
    array_like=True,
):  # type: ignore[call-arg]
    """
    Request for a LoRA adapter.

    lora_int_id must be globally unique for a given adapter.
    This is currently not enforced in vLLM.

    load_inplace: If True, forces reloading the adapter even if one
        with the same lora_int_id already exists in the cache. This replaces
        the existing adapter in-place. If False (default), only loads if the
        adapter is not already loaded.
    """

    lora_name: str  # ------【LoRA】lora_name：适配器唯一名字，用于跨引擎标识/判等/哈希 ------
    lora_int_id: int  # ------【LoRA】lora_int_id：适配器全局唯一整型 id（须 >0） ------
    lora_path: str = ""  # ------【LoRA】lora_path：适配器权重本地路径，加载时读取 ------
    base_model_name: str | None = msgspec.field(default=None)  # ------【LoRA】base_model_name：绑定的基础模型名，用于缓存 key 区分 ------
    tensorizer_config_dict: dict | None = None  # ------【LoRA/权重传输】tensorizer_config_dict：tensorizer 序列化/反序列化配置 ------
    load_inplace: bool = False  # ------【LoRA】load_inplace：是否强制原地重载同名适配器（替换缓存中的旧实例） ------
    is_3d_lora_weight: bool = False  # ------【LoRA】is_3d_lora_weight：MoE 权重是否为 3D 融合 gate_up/down 布局 ------
    """Whether this adapter's MoE weights are stored in the 3D fused
    `gate_up_proj` / `down_proj` layout (one fused tensor per layer) or the
    2D per-expert split layout (separate `gate_proj` / `up_proj` / `down_proj`
    tensors per expert). Only consulted when the engine is started with
    `enable_mixed_moe_lora_format=True`; otherwise it is ignored and the
    on-disk format is inferred from the base model."""

    # ------【LoRA】__post_init__：构造后校验 id>0 且 path 非空，避免非法适配器进入缓存 ------
    def __post_init__(self):
        if self.lora_int_id < 1:
            raise ValueError(f"id must be > 0, got {self.lora_int_id}")

        # Ensure lora_path is not empty
        assert self.lora_path, "lora_path cannot be empty"

    # ------【LoRA】adapter_id/name/path：便捷访问器，统一暴露适配器 id/名字/路径 ------
    @property
    def adapter_id(self):
        return self.lora_int_id

    @property
    def name(self):
        return self.lora_name

    @property
    def path(self):
        return self.lora_path

    # ------【LoRA】__eq__/__hash__：基于 lora_name 判等与哈希，使同名适配器跨引擎等价、可入集合/dict ------
    def __eq__(self, value: object) -> bool:
        """
        Overrides the equality method to compare LoRARequest
        instances based on lora_name. This allows for identification
        and comparison lora adapter across engines.
        """
        return isinstance(value, self.__class__) and self.lora_name == value.lora_name

    def __hash__(self) -> int:
        """
        Overrides the hash method to hash LoRARequest instances
        based on lora_name. This ensures that LoRARequest instances
        can be used in hash-based collections such as sets and dictionaries,
        identified by their names across engines.
        """
        return hash(self.lora_name)
