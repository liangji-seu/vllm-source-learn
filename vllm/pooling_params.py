# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from copy import deepcopy
from typing import Any

import msgspec

from vllm.config import ModelConfig, PoolerConfig
from vllm.exceptions import VLLMValidationError
from vllm.logger import init_logger
from vllm.sampling_params import RequestOutputKind
from vllm.tasks import PoolingTask, check_removed_pooling_task

logger = init_logger(__name__)


# ------【DP/核心逻辑】LateInteractionParams：晚期交互(ColBERT 类)打分请求元数据，API 层→worker 侧打分流程；query_key 参与 DP 路由与缓存查找 ------
class LateInteractionParams(
    msgspec.Struct,
    omit_defaults=True,  # type: ignore[call-arg]
    array_like=True,
):  # type: ignore[call-arg]
    """Metadata for worker-side late-interaction scoring.

    Attributes:
        mode:
            - "cache_query": cache query token embeddings
            - "score_doc": score a document against a cached query.
        query_key: stable key used for both DP routing and worker cache lookup.
        query_uses: expected number of document requests
    """

    # ------【核心逻辑】mode：打分模式 "cache_query"缓存查询向量 / "score_doc"用缓存查询给文档打分 ------
    mode: str
    # ------【DP】query_key：稳定 key，用于 DP 路由把同 query 请求分发到同 worker 并做缓存查找 ------
    query_key: str
    # ------【核心逻辑】query_uses：该 query 预期的文档请求数，用于判断查询缓存何时可释放 ------
    query_uses: int | None = None


# ------【核心逻辑】PoolingParams：embedding/pooling 类请求的采样参数消息字段，API 层→engine/worker；含前缀缓存跳过、维度裁剪等优化开关 ------
class PoolingParams(
    msgspec.Struct,
    omit_defaults=True,  # type: ignore[call-arg]
    array_like=True,
):  # type: ignore[call-arg]
    """API parameters for pooling models.

    Attributes:
        use_activation: Whether to apply activation function to the pooler outputs.
            `None` uses the pooler's default, which is `True` in most cases.
        dimensions: Reduce the dimensions of embeddings
            if model support matryoshka representation.
    """

    # ------【核心逻辑】use_activation：是否对 pooler 输出应用激活函数；None 用 pooler 默认（多为 True） ------
    # --8<-- [start:common-pooling-params]
    use_activation: bool | None = None
    # --8<-- [end:common-pooling-params]

    ## for embeddings models
    # ------【核心逻辑】dimensions：matryoshka 表示法下裁剪 embedding 输出维度，涉及 embedding 压缩优化 ------
    # --8<-- [start:embed-pooling-params]
    dimensions: int | None = None
    # --8<-- [end:embed-pooling-params]

    ## for step pooling models
    # ------【核心逻辑】step_tag_id：step pooling 模型指定聚合哪个 step 的标签 token id ------
    step_tag_id: int | None = None
    # ------【核心逻辑】returned_token_ids：step pooling 返回哪些 token 的向量 id 列表 ------
    returned_token_ids: list[int] | None = None

    ## Internal use only
    # ------【核心逻辑】task：pooling 任务类型(embed/classify/token_embed/token_classify 等)，决定可用参数与校验分支 ------
    task: PoolingTask | None = None
    # ------【核心逻辑】requires_token_ids：是否需要在输出中返回 token id（内部使用） ------
    requires_token_ids: bool = False
    # ------【前缀缓存】skip_reading_prefix_cache：本请求是否跳过读前缀缓存；token 级 pooling 输出数可能少于 prompt token 数 ------
    skip_reading_prefix_cache: bool | None = None
    # ------【DP/核心逻辑】late_interaction_params：晚期交互打分元数据，随请求透传给 worker ------
    late_interaction_params: LateInteractionParams | None = None
    # ------【核心逻辑】extra_kwargs：透传给 pooler 前向计算的额外 kwarg 字典 ------
    extra_kwargs: dict[str, Any] | None = None
    # ------【核心逻辑】output_kind：输出时机，pooling 场景强制 FINAL_ONLY ------
    output_kind: RequestOutputKind = RequestOutputKind.FINAL_ONLY

    # ------【核心逻辑】all_parameters：本请求可配置的全部参数名，供参数校验遍历 ------
    @property
    def all_parameters(self) -> list[str]:
        return ["dimensions", "use_activation"]

    # ------【核心逻辑】valid_parameters：各 task 类型允许的合法参数集合，用于合并与校验 ------
    @property
    def valid_parameters(self):
        return {
            "embed": ["dimensions", "use_activation"],
            "classify": ["use_activation"],
            "token_embed": ["dimensions", "use_activation"],
            "token_classify": ["use_activation"],
        }

    # ------【核心逻辑】clone：深拷贝一份 PoolingParams，供采样参数隔离复用 ------
    def clone(self) -> "PoolingParams":
        """Returns a deep copy of the PoolingParams instance."""
        return deepcopy(self)

    # ------【核心逻辑】verify：请求级参数校验入口，依次合并默认值→设置默认值→校验合法参数 ------
    def verify(self, model_config: ModelConfig) -> None:
        # plugin task uses io_processor.parse_request to verify inputs,
        # skipping PoolingParams verify
        if self.task == "plugin":
            if self.skip_reading_prefix_cache is None:
                self.skip_reading_prefix_cache = True
            return

        # skipping verify, let plugins configure and validate pooling params
        if self.task not in self.valid_parameters:
            return

        # NOTE: Task validation needs to done against the model instance,
        # which is not available in model config. So, it's not included
        # in this method
        self._merge_default_parameters(model_config)
        self._set_default_parameters(model_config)
        self._verify_valid_parameters()

    # ------【核心逻辑】_merge_default_parameters：用 pooler_config 缺省值回填本请求未显式设置的参数 ------
    def _merge_default_parameters(self, model_config: ModelConfig) -> None:
        pooler_config = model_config.pooler_config
        if pooler_config is None:
            return

        if self.task is None:
            raise ValueError("task must be set before merging parameters")
        valid_parameters = self.valid_parameters[self.task]

        for k in valid_parameters:
            if getattr(pooler_config, k, None) is None:
                continue

            if getattr(self, k, None) is None:
                setattr(self, k, getattr(pooler_config, k))

        if self.skip_reading_prefix_cache is None:
            # If prefix caching is enabled,
            # the output of all pooling may less than n_prompt_tokens,
            # we need to skip reading cache at this request.
            if self.task in ["token_embed", "token_classify"]:
                self.skip_reading_prefix_cache = True
            else:
                self.skip_reading_prefix_cache = False

        self._verify_step_pooling(pooler_config, valid_parameters)

    # ------【核心逻辑】_verify_step_pooling：校验 step pooling 专属参数仅在 STEP 池化类型下可用 ------
    def _verify_step_pooling(
        self,
        pooler_config: PoolerConfig,
        valid_parameters: list[str],
    ):
        step_pooling_parameters = ["step_tag_id", "returned_token_ids"]
        if pooler_config.tok_pooling_type != "STEP":
            invalid_parameters = []
            for k in step_pooling_parameters:
                if getattr(self, k, None) is not None:
                    invalid_parameters.append(k)

            if invalid_parameters:
                raise VLLMValidationError(
                    f"Task {self.task} only supports {valid_parameters} "
                    f"parameters, does not support "
                    f"{invalid_parameters} parameters"
                )
        else:
            for k in step_pooling_parameters:
                if getattr(pooler_config, k, None) is None:
                    continue

                if getattr(self, k, None) is None:
                    setattr(self, k, getattr(pooler_config, k))

    # ------【核心逻辑】_set_default_parameters：按 task 设置 use_activation 默认值，并校验 matryoshka dimensions 合法范围 ------
    def _set_default_parameters(self, model_config: ModelConfig):
        if self.task in ["embed", "token_embed"]:
            if self.use_activation is None:
                self.use_activation = True

            if self.dimensions is not None:
                dimensions = self.dimensions
                model_name = model_config.served_model_name
                embedding_size = model_config.embedding_size
                valid_range = f"[1, {embedding_size}]"
                dimensions_in_range = 1 <= dimensions <= embedding_size
                if not model_config.is_matryoshka:
                    raise VLLMValidationError(
                        f"Model {model_name!r} does not support Matryoshka "
                        f"embeddings; dimensions must be unset "
                        f"(received dimensions={dimensions})."
                    )

                if not dimensions_in_range:
                    raise VLLMValidationError(
                        f"Model {model_name!r} only supports dimensions in "
                        f"range {valid_range}, got {dimensions}."
                    )

                mds = model_config.matryoshka_dimensions
                if mds is not None and dimensions not in mds:
                    raise VLLMValidationError(
                        f"Model {model_name!r} only supports Matryoshka "
                        f"dimensions {str(mds)}, got {dimensions}."
                    )

        elif self.task in ["classify", "token_classify"]:
            if self.use_activation is None:
                self.use_activation = True
        else:
            raise ValueError(f"Unknown pooling task: {self.task!r}")

    # ------【核心逻辑】_verify_valid_parameters：校验请求未携带该 task 不支持的参数 ------
    def _verify_valid_parameters(self):
        if self.task is None:
            raise ValueError("task must be set before verifying parameters")
        valid_parameters = self.valid_parameters[self.task]
        invalid_parameters = []
        for k in self.all_parameters:
            if k in valid_parameters:
                continue

            if getattr(self, k, None) is not None:
                invalid_parameters.append(k)

        if invalid_parameters:
            raise VLLMValidationError(
                f"Task {self.task!r} only supports {valid_parameters} "
                f"parameters, does not support "
                f"{invalid_parameters} parameters"
            )

    # ------【核心逻辑】__repr__：自定义可读字符串，便于日志/调试打印关键字段 ------
    def __repr__(self) -> str:
        return (
            f"PoolingParams("
            f"task={self.task}, "
            f"dimensions={self.dimensions}, "
            f"use_activation={self.use_activation}, "
            f"step_tag_id={self.step_tag_id}, "
            f"returned_token_ids={self.returned_token_ids}, "
            f"requires_token_ids={self.requires_token_ids}, "
            f"skip_reading_prefix_cache={self.skip_reading_prefix_cache}, "
            f"late_interaction_params={self.late_interaction_params}, "
            f"extra_kwargs={self.extra_kwargs})"
        )

    # ------【核心逻辑】__post_init__：构造后自检，pooling 输出时机必须为 FINAL_ONLY ------
    def __post_init__(self) -> None:
        check_removed_pooling_task(self.task)
        if self.output_kind != RequestOutputKind.FINAL_ONLY:
            raise VLLMValidationError(
                "For pooling output_kind has to be FINAL_ONLY, "
                f"got {self.output_kind!r}"
            )
