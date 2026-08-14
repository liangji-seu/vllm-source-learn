# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sampling parameters for text generation."""

import copy
import json as json_mod
import math
from dataclasses import field
from enum import Enum, IntEnum
from functools import cached_property
from typing import Annotated, Any

import msgspec
from pydantic import BeforeValidator
from pydantic.dataclasses import dataclass

import vllm.envs as envs
from vllm.config import ModelConfig, SpeculativeConfig, StructuredOutputsConfig
from vllm.exceptions import VLLMValidationError
from vllm.logger import init_logger
from vllm.tokenizers import TokenizerLike
from vllm.utils.mistral import is_mistral_tokenizer
from vllm.v1.serial_utils import PydanticMsgspecMixin

logger = init_logger(__name__)

_SAMPLING_EPS = 1e-5
_MAX_TEMP = 1e-2

MAX_LOGPROB_TOKEN_IDS = 128
"""Upper bound on `SamplingParams.logprob_token_ids` list length. Must match
the per-request row width allocated by the sampler's `LogprobTokenIdsState`."""


def validate_thinking_token_budget(value: int | float | bool | None) -> int | None:
    """Validate ``thinking_token_budget``; return ``None`` if unset."""
    if value is None:
        return None
    if isinstance(value, (bool, float)) or not isinstance(value, int):
        raise VLLMValidationError(
            "`thinking_token_budget` must be a non-negative integer "
            "or -1 for unlimited.",
            parameter="thinking_token_budget",
            value=value,
        )
    if value == -1:
        return None
    if value < 0:
        raise VLLMValidationError(
            "`thinking_token_budget` must be a non-negative integer "
            "or -1 for unlimited.",
            parameter="thinking_token_budget",
            value=value,
        )
    return value


ThinkingTokenBudget = Annotated[
    int | None,
    BeforeValidator(validate_thinking_token_budget),
]


# ------【核心逻辑】SamplingType：采样策略枚举，决定采样器走贪心/随机/带种子随机路径 ------
class SamplingType(IntEnum):
    # ------【核心逻辑】GREEDY：贪心采样，取概率最高 token（argmax），输出确定 ------
    GREEDY = 0
    # ------【核心逻辑】RANDOM：随机采样，按概率分布抽取，输出多样 ------
    RANDOM = 1
    # ------【核心逻辑】RANDOM_SEED：带种子随机采样，结果可复现，便于对比实验 ------
    RANDOM_SEED = 2


# maybe make msgspec?
# ------【结构化输出/grammar】StructuredOutputsParams：结构化输出约束 DTO，构造约束解码 logit processor ------
@dataclass
class StructuredOutputsParams:
    # One of these fields will be used to build a logit processor.
    # ------【结构化输出/grammar】json：JSON Schema 约束（str/dict），约束解码产出合法 JSON ------
    json: str | dict | None = None
    # ------【结构化输出/grammar】regex：正则表达式约束，生成文本需匹配该正则 ------
    regex: str | None = None
    # ------【结构化输出/grammar】choice：候选字符串列表约束，输出只能取自给定选项 ------
    choice: list[str] | None = None
    # ------【结构化输出/grammar】grammar：GBNF 等文法约束，按文法规则约束解码 ------
    grammar: str | None = None
    # ------【结构化输出/grammar】json_object：仅强制输出 JSON 对象，不校验具体 schema ------
    json_object: bool | None = None
    # These are other options that can be set.
    # ------【结构化输出/grammar】disable_any_whitespace：禁用任意空白符，收紧 JSON 匹配 ------
    disable_any_whitespace: bool = False
    # ------【结构化输出/grammar】disable_additional_properties：禁止 schema 之外的额外属性 ------
    disable_additional_properties: bool = False
    # ------【结构化输出/grammar】whitespace_pattern：自定义空白匹配模式，替代默认空白定义 ------
    whitespace_pattern: str | None = None
    # ------【结构化输出/grammar】structural_tag：结构化标签，约束 JSON 推理类 token ------
    structural_tag: str | None = None

    # ------【结构化输出/grammar】_backend：约束解码后端（xgrammar/outlines），Processor 校验时填入 ------
    _backend: str | None = field(default=None, init=False)
    """CAUTION: Should only be set by Processor._validate_structured_output"""
    # ------【结构化输出/grammar】_backend_was_auto：后端是否自动选择，用于回退/报错 ------
    _backend_was_auto: bool = field(default=False, init=False)
    """CAUTION: Should only be set by Processor._validate_structured_output"""

    def __post_init__(self):
        """Validate that some fields are mutually exclusive."""
        count = sum(
            [
                self.json is not None,
                self.regex is not None,
                self.choice is not None,
                self.grammar is not None,
                self.json_object is not None,
                self.structural_tag is not None,
            ]
        )
        if count > 1:
            raise VLLMValidationError(
                "You can only use one kind of structured outputs constraint "
                f"but multiple are specified: {self.__dict__}"
            )
        if count < 1:
            raise VLLMValidationError(
                "You must use one kind of structured outputs constraint "
                f"but none are specified: {self.__dict__}"
            )

    def all_constraints_none(self) -> bool:
        """
        Returns True if all structured-output constraint fields are None.
        """
        return all(
            getattr(self, field) is None
            for field in (
                "json",
                "regex",
                "choice",
                "grammar",
                "json_object",
                "structural_tag",
            )
        )

    def all_non_structural_tag_constraints_none(self) -> bool:
        """
        Returns True if all structured-output constraint fields are None.
        """
        return all(
            getattr(self, field) is None
            for field in (
                "json",
                "regex",
                "choice",
                "grammar",
                "json_object",
            )
        )


# ------【核心逻辑】RepetitionDetectionParams：重复惩罚/检测参数 DTO，检测输出中重复 N-gram 模式 ------
@dataclass
class RepetitionDetectionParams:
    """Parameters for detecting repetitive N-gram patterns in output tokens."""

    # ------【核心逻辑】max_pattern_size：检测的最大 N-gram 长度，0 表示禁用重复检测 ------
    max_pattern_size: int = 0
    """Maximum size of N-gram pattern to detect for sequence repetition.
    Set to 0 to disable. Must be used together with min_count."""

    # ------【核心逻辑】min_pattern_size：检测的最小 N-gram 长度，0 时默认取 1 ------
    min_pattern_size: int = 0
    """Minimum N-gram pattern size to check for sequence repetition.
    If set to 0, it defaults to 1.
    Must be <= max_pattern_size."""

    # ------【核心逻辑】min_count：N-gram 重复触发检测的最小次数，须 >= 2 ------
    min_count: int = 0
    """Minimum number of times an N-gram pattern must repeat to trigger
    detection. Must be >= 2. Example: 3 for detecting a phrase repeated
    3 times. Must be used together with max_pattern_size."""

    def __post_init__(self):
        if (
            self.max_pattern_size < 0
            or self.min_pattern_size < 0
            or self.min_pattern_size > self.max_pattern_size
        ):
            raise VLLMValidationError(
                "max_pattern_size, min_pattern_size must be >=0, "
                "with min_pattern_size <= max_pattern_size. "
                "Set both to 0 to disable repetitive pattern detection."
            )
        if self.max_pattern_size > 0 and self.min_count < 2:
            raise VLLMValidationError(
                "min_count must be >= 2 to detect repetitive patterns "
                "in engine output. If you do not wish to detect repetitive "
                "patterns, set max_pattern_size to 0."
            )


# ------【核心逻辑】RequestOutputKind：输出返回模式枚举，决定 RequestOutput 返回全量/增量/仅最终结果 ------
class RequestOutputKind(Enum):
    # Return entire output so far in every RequestOutput
    # ------【核心逻辑】CUMULATIVE：每次 RequestOutput 返回完整累计输出 ------
    CUMULATIVE = 0
    # Return only deltas in each RequestOutput
    # ------【核心逻辑】DELTA：每次只返回增量 delta，减少传输量 ------
    DELTA = 1
    # Do not return intermediate RequestOutput
    # ------【核心逻辑】FINAL_ONLY：仅返回最终结果，不返回中间结果 ------
    FINAL_ONLY = 2


def _is_non_tekken_mistral(tokenizer: TokenizerLike) -> bool:
    return is_mistral_tokenizer(tokenizer) and not tokenizer.is_tekken


def _get_llg_tokenizer(tokenizer: TokenizerLike) -> Any:
    return tokenizer.llg_tokenizer if is_mistral_tokenizer(tokenizer) else None


# ------【核心逻辑】SamplingParams：采样参数消息 DTO，API/引擎→调度器与采样器传递采样配置，涉及前缀缓存/投机解码/结构化输出等优化 ------
class SamplingParams(
    PydanticMsgspecMixin,
    msgspec.Struct,
    omit_defaults=True,  # type: ignore[call-arg]
    # required for @cached_property.
    dict=True,
):  # type: ignore[call-arg]
    """Sampling parameters for text generation.

    Overall, we follow the sampling parameters from the OpenAI text completion
    API (https://platform.openai.com/docs/api-reference/completions/create).
    In addition, we support beam search, which is not supported by OpenAI.
    """

    # ------【核心逻辑】n：单请求返回的序列数量，受 VLLM_MAX_N_SEQUENCES 限制，n>1 触发多序列并行生成 ------
    n: int = 1
    """Number of outputs to return for the given prompt request.

    The maximum allowed value is controlled by the ``VLLM_MAX_N_SEQUENCES``
    environment variable (default: 16384).

    NOTE:
        `AsyncLLM` streams outputs by default. When `n > 1`, all `n` outputs
        are generated and streamed cumulatively per request. To see all `n`
        outputs upon completion, use `output_kind=RequestOutputKind.FINAL_ONLY`
        in `SamplingParams`."""
    # ------【核心逻辑】presence_penalty：存在惩罚，>0 鼓励用新 token，作用于 logit 处理器 ------
    presence_penalty: float = 0.0
    """Penalizes new tokens based on whether they appear in the generated text
    so far. Values > 0 encourage the model to use new tokens, while values < 0
    encourage the model to repeat tokens."""
    # ------【核心逻辑】frequency_penalty：频率惩罚，按已生成 token 出现频次调低 logit ------
    frequency_penalty: float = 0.0
    """Penalizes new tokens based on their frequency in the generated text so
    far. Values > 0 encourage the model to use new tokens, while values < 0
    encourage the model to repeat tokens."""
    # ------【核心逻辑】repetition_penalty：重复惩罚，惩罚 prompt+已生成文本中出现的 token ------
    repetition_penalty: float = 1.0
    """Penalizes new tokens based on whether they appear in the prompt and the
    generated text so far. Values > 1 encourage the model to use new tokens,
    while values < 1 encourage the model to repeat tokens."""
    # ------【核心逻辑】temperature：采样温度，越低越确定，0 表示贪心采样 ------
    temperature: float = 1.0
    """Controls the randomness of the sampling. Lower values make the model
    more deterministic, while higher values make the model more random. Zero
    means greedy sampling."""
    # ------【核心逻辑】top_p：核采样累计概率阈值，取累计概率达 top_p 的 token 集合 ------
    top_p: float = 1.0
    """Controls the cumulative probability of the top tokens to consider. Must
    be in (0, 1]. Set to 1 to consider all tokens."""
    # ------【核心逻辑】top_k：只保留概率最高的 k 个 token，0/-1 表示不限制 ------
    top_k: int = 0
    """Controls the number of top tokens to consider. Set to 0 (or -1) to
    consider all tokens."""
    # ------【核心逻辑】min_p：相对最高概率的最小概率门槛，低于该比值的 token 被过滤 ------
    min_p: float = 0.0
    """Represents the minimum probability for a token to be considered,
    relative to the probability of the most likely token. Must be in [0, 1].
    Set to 0 to disable this."""
    # ------【核心逻辑】seed：随机种子，固定后采样结果可复现，-1 转 None ------
    seed: int | None = None
    """Random seed to use for the generation."""
    # ------【核心逻辑】stop：停止字符串列表，命中即终止生成并截断输出 ------
    stop: str | list[str] | None = None
    """String(s) that stop the generation when they are generated. The returned
    output will not contain the stop strings."""
    # ------【核心逻辑】stop_token_ids：停止 token id 列表，命中即终止生成 ------
    stop_token_ids: list[int] | None = None
    """Token IDs that stop the generation when they are generated. The returned
    output will contain the stop tokens unless the stop tokens are special
    tokens."""
    # ------【核心逻辑】ignore_eos：是否忽略 EOS token 继续生成 ------
    ignore_eos: bool = False
    """Whether to ignore the EOS token and continue generating
    tokens after the EOS token is generated."""
    # ------【核心逻辑/显存 profiling】max_tokens：每序列最大生成 token 数，供调度器预估 KV 与输出长度 ------
    max_tokens: int | None = 16
    """Maximum number of tokens to generate per output sequence."""
    # ------【核心逻辑】min_tokens：EOS/stop 前至少生成的 token 数 ------
    min_tokens: int = 0
    """Minimum number of tokens to generate per output sequence before EOS or
    `stop_token_ids` can be generated"""
    # ------【核心逻辑】logprobs：每输出 token 返回的 log 概率数，-1 返回全词表 ------
    logprobs: int | None = None
    """Number of log probabilities to return per output token. When set to
    `None`, no probability is returned. If set to a non-`None` value, the
    result includes the log probabilities of the specified number of most
    likely tokens, as well as the chosen tokens. Note that the implementation
    follows the OpenAI API: The API will always return the log probability of
    the sampled token, so there may be up to `logprobs+1` elements in the
    response. When set to -1, return all `vocab_size` log probabilities."""
    # ------【核心逻辑】prompt_logprobs：每 prompt token 返回的 log 概率数，-1 返回全词表 ------
    prompt_logprobs: int | None = None
    """Number of log probabilities to return per prompt token.
    When set to -1, return all `vocab_size` log probabilities."""
    # ------【核心逻辑】logprob_token_ids：只对指定 token id 返回 logprobs，比 -1 更省显存/计算 ------
    logprob_token_ids: list[int] | None = None
    """Specific token IDs to return logprobs for. More efficient than
    logprobs=-1 when you only need logprobs for a small set of tokens.
    When set, logprobs for exactly these token IDs will be returned,
    in addition to the sampled token. This is useful for scoring tasks
    where you want to compare probabilities of specific label tokens."""
    # ------【核心逻辑/显存 profiling】flat_logprobs：扁平结构返回 logprobs，降低 GC 开销提升性能 ------
    flat_logprobs: bool = False
    """Whether to return logprobs in flatten format (i.e. FlatLogprob)
    for better performance.
    NOTE: GC costs of FlatLogprobs is significantly smaller than
    list[dict[int, Logprob]]. After enabled, PromptLogprobs and
    SampleLogprobs would populated as FlatLogprobs."""
    # NOTE: This parameter is only exposed at the engine level for now.
    # It is not exposed in the OpenAI API server, as the OpenAI API does
    # not support returning only a list of token IDs.
    # ------【核心逻辑】detokenize：是否把 token id 解码回文本，仅引擎层暴露 ------
    detokenize: bool = True
    """Whether to detokenize the output."""
    # ------【核心逻辑】skip_special_tokens：输出时是否跳过特殊 token ------
    skip_special_tokens: bool = True
    """Whether to skip special tokens in the output."""
    # ------【核心逻辑】spaces_between_special_tokens：特殊 token 之间是否加空格 ------
    spaces_between_special_tokens: bool = True
    """Whether to add spaces between special tokens in the output."""
    # ------【核心逻辑】include_stop_str_in_output：输出文本是否保留停止字符串 ------
    include_stop_str_in_output: bool = False
    """Whether to include the stop strings in output text."""
    # ------【核心逻辑】output_kind：输出返回模式（累计/增量/仅最终），决定 RequestOutput 打包粒度 ------
    output_kind: RequestOutputKind = RequestOutputKind.CUMULATIVE
    # ------【异步 RPC】stream_interval：流式输出聚合的 token 间隔，减少 RequestOutput 消息频率 ------
    stream_interval: int | None = None
    """Number of newly generated tokens to batch into each streamed
    `RequestOutput`. Raises the interval above the engine-level
    `--stream-interval`. Values below engine setting are clamped up to it.
    The first and final outputs are always emitted immediately."""
    # ------【核心逻辑】skip_clone：为 True 时 clone 用浅拷贝省深拷贝开销（需独占实例） ------
    skip_clone: bool = False
    """Internal flag indicating that this SamplingParams instance is safe to
    reuse without cloning. When True, clone() will return self without
    performing a deep copy. This should only be set when the params object
    is guaranteed to be dedicated to a single request and won't be modified
    in ways that would affect other uses."""

    # The below fields are not supposed to be used as an input.
    # They are set in post_init.
    # ------【核心逻辑】output_text_buffer_length：为停止字符串评估预留的回退字符数 ------
    output_text_buffer_length: int = 0
    # ------【核心逻辑】_eos_token_id：内部 EOS token id，由引擎填充，供终止判断 ------
    _eos_token_id: int | None = None
    # ------【核心逻辑】_all_stop_token_ids：全部停止 token id 集合，post_init 聚合，供终止判断 ------
    _all_stop_token_ids: set[int] = msgspec.field(default_factory=set)

    # Fields used to construct logits processors
    # ------【结构化输出/grammar】structured_outputs：结构化输出约束，构造约束解码 logit processor ------
    structured_outputs: StructuredOutputsParams | None = None
    """Parameters for configuring structured outputs."""
    # ------【核心逻辑】logit_bias：token→偏置映射，构造 logit 偏置处理器调整分数 ------
    logit_bias: dict[int, float] | None = None
    """If provided, the engine will construct a logits processor that applies
    these logit biases."""
    # ------【核心逻辑】allowed_token_ids：白名单 token id，只保留这些 token 的分数 ------
    allowed_token_ids: list[int] | None = None
    """If provided, the engine will construct a logits processor which only
    retains scores for the given token ids."""
    # ------【核心逻辑】extra_args：透传给自定义采样实现/插件的额外参数 ------
    extra_args: dict[str, Any] | None = None
    """Arbitrary additional args, that can be used by custom sampling
    implementations, plugins, etc. Not used by any in-tree sampling
    implementations."""
    # ------【EP/EPLB】routed_experts_prompt_start：返回路由专家数据时跳过前 N 个 prompt token，避免多轮重复 ------
    routed_experts_prompt_start: int = 0
    """When enable_return_routed_experts is active, skip the first
    routed_experts_prompt_start prompt tokens from the returned routing
    data. In multi-turn agent scenarios, set this to the length of the
    already-returned prefix to avoid duplicating routing for prompt tokens
    covered by earlier turns. Default 0 returns routing for all prompt
    tokens."""

    # Fields used for bad words
    # ------【核心逻辑】bad_words：禁用词列表，其末 token 被禁止补全 ------
    bad_words: list[str] | None = None
    """Words that are not allowed to be generated. More precisely, only the
    last token of a corresponding token sequence is not allowed when the next
    generated token can complete the sequence."""
    # ------【核心逻辑】_bad_words_token_ids：内部禁用词 token 序列，由 update_from_tokenizer 编码填入 ------
    _bad_words_token_ids: list[list[int]] | None = None

    # ------【前缀缓存】skip_reading_prefix_cache：为 True 跳过读前缀缓存，需 prompt_logprobs 时自动置 True ------
    skip_reading_prefix_cache: bool | None = None
    # ------【核心逻辑】thinking_token_budget：思考（推理）阶段最大 token 预算，-1 表示不限 ------
    thinking_token_budget: int | None = None
    """Maximum number of tokens allowed for thinking operations."""

    # ------【核心逻辑】repetition_detection：重复 N-gram 检测参数，命中提前终止省 token ------
    repetition_detection: RepetitionDetectionParams | None = None
    """Parameters for detecting repetitive N-gram patterns in output tokens.
    If such repetition is detected, generation will be ended early. LLMs can
    sometimes generate repetitive, unhelpful token patterns, stopping only
    when they hit the maximum output length (e.g. 'abcdabcdabcd...' or
    '\\emoji \\emoji \\emoji ...'). This feature can detect such behavior
    and terminate early, saving time and tokens."""

    # ------【核心逻辑】from_optional：批量构造 SamplingParams，把 None 归一化为默认值并校验 logit_bias ------
    @staticmethod
    def from_optional(
        n: int | None = 1,
        presence_penalty: float | None = 0.0,
        frequency_penalty: float | None = 0.0,
        repetition_penalty: float | None = 1.0,
        temperature: float | None = 1.0,
        top_p: float | None = 1.0,
        top_k: int = 0,
        min_p: float = 0.0,
        seed: int | None = None,
        stop: str | list[str] | None = None,
        stop_token_ids: list[int] | None = None,
        bad_words: list[str] | None = None,
        thinking_token_budget: int | None = None,
        include_stop_str_in_output: bool = False,
        ignore_eos: bool = False,
        max_tokens: int | None = 16,
        min_tokens: int = 0,
        logprobs: int | None = None,
        prompt_logprobs: int | None = None,
        detokenize: bool = True,
        skip_special_tokens: bool = True,
        spaces_between_special_tokens: bool = True,
        output_kind: RequestOutputKind = RequestOutputKind.CUMULATIVE,
        stream_interval: int | None = None,
        structured_outputs: StructuredOutputsParams | None = None,
        logit_bias: dict[int, float] | dict[str, float] | None = None,
        allowed_token_ids: list[int] | None = None,
        extra_args: dict[str, Any] | None = None,
        skip_clone: bool = False,
        repetition_detection: RepetitionDetectionParams | None = None,
        logprob_token_ids: list[int] | None = None,
    ) -> "SamplingParams":
        if logit_bias is not None:
            # Fast path uses a dict comprehension; on failure we iterate once
            # to identify the exact offending entry for the error message.
            try:
                logit_bias = {
                    int(token): min(100.0, max(-100.0, bias))
                    for token, bias in logit_bias.items()
                }
            except (ValueError, TypeError):
                invalid_keys = []
                converted_logit_bias = {}
                for token, bias in logit_bias.items():
                    try:
                        token_id = int(token)
                    except (ValueError, TypeError):
                        invalid_keys.append(token)
                        continue
                    converted_logit_bias[token_id] = min(100.0, max(-100.0, bias))
                if invalid_keys:
                    raise VLLMValidationError(
                        f"logit_bias contains key(s) that cannot be "
                        f"converted to integer token IDs: {invalid_keys!r}",
                        parameter="logit_bias",
                        value=invalid_keys,
                    ) from None
                logit_bias = converted_logit_bias

        return SamplingParams(
            n=1 if n is None else n,
            presence_penalty=0.0 if presence_penalty is None else presence_penalty,
            frequency_penalty=0.0 if frequency_penalty is None else frequency_penalty,
            repetition_penalty=1.0
            if repetition_penalty is None
            else repetition_penalty,
            temperature=1.0 if temperature is None else temperature,
            top_p=1.0 if top_p is None else top_p,
            top_k=top_k,
            min_p=min_p,
            seed=seed,
            stop=stop,
            stop_token_ids=stop_token_ids,
            bad_words=bad_words,
            thinking_token_budget=thinking_token_budget,
            include_stop_str_in_output=include_stop_str_in_output,
            ignore_eos=ignore_eos,
            max_tokens=max_tokens,
            min_tokens=min_tokens,
            logprobs=logprobs,
            prompt_logprobs=prompt_logprobs,
            logprob_token_ids=logprob_token_ids,
            detokenize=detokenize,
            skip_special_tokens=skip_special_tokens,
            spaces_between_special_tokens=spaces_between_special_tokens,
            output_kind=output_kind,
            stream_interval=stream_interval,
            structured_outputs=structured_outputs,
            logit_bias=logit_bias,
            allowed_token_ids=allowed_token_ids,
            extra_args=extra_args,
            skip_clone=skip_clone,
            repetition_detection=repetition_detection,
        )

    # ------【核心逻辑】__post_init__：归一化参数（温度下限/stop 列表化/logprobs True→1）并触发校验与聚合 ------
    def __post_init__(self) -> None:
        if 0 < self.temperature < _MAX_TEMP:
            logger.warning(
                "temperature %s is less than %s, which may cause numerical "
                "errors nan or inf in tensors. We have maxed it out to %s.",
                self.temperature,
                _MAX_TEMP,
                _MAX_TEMP,
            )
            self.temperature = max(self.temperature, _MAX_TEMP)

        if self.seed == -1:
            self.seed = None

        self.thinking_token_budget = validate_thinking_token_budget(
            self.thinking_token_budget
        )

        if self.stop is None:
            self.stop = []
        elif isinstance(self.stop, str):
            self.stop = [self.stop]

        if self.stop_token_ids is None:
            self.stop_token_ids = []

        if self.bad_words is None:
            self.bad_words = []

        if self.logprobs is True:
            self.logprobs = 1

        if self.prompt_logprobs is True:
            self.prompt_logprobs = 1

        # Number of characters to hold back for stop string evaluation
        # until sequence is finished.
        if self.stop and not self.include_stop_str_in_output:
            self.output_text_buffer_length = max(len(s) for s in self.stop) - 1

        self._verify_args()

        if self.temperature < _SAMPLING_EPS:
            # Zero temperature means greedy sampling.
            self.top_p = 1.0
            self.top_k = 0
            self.min_p = 0.0
            self._verify_greedy_sampling()

        # eos_token_id is added to this by the engine
        self._all_stop_token_ids.update(self.stop_token_ids)

        if self.skip_reading_prefix_cache is None:
            # If prefix caching is enabled,
            # the output of prompt logprobs may less than n_prompt_tokens,
            # we need to skip reading cache at this request.
            self.skip_reading_prefix_cache = self.prompt_logprobs is not None

    # ------【核心逻辑】_verify_args：逐字段合法性校验，非法值抛 VLLMValidationError ------
    def _verify_args(self) -> None:
        if not isinstance(self.n, int):
            raise VLLMValidationError(
                f"n must be an int, but is of type {type(self.n)}"
            )
        if self.n < 1:
            raise VLLMValidationError(f"n must be at least 1, got {self.n}.")
        max_n = envs.VLLM_MAX_N_SEQUENCES
        if self.n > max_n:
            raise VLLMValidationError(
                f"n must be at most {max_n}, got {self.n}. "
                "To increase this limit, set the VLLM_MAX_N_SEQUENCES "
                "environment variable."
            )
        if not -2.0 <= self.presence_penalty <= 2.0:
            raise VLLMValidationError(
                f"presence_penalty must be in [-2, 2], got {self.presence_penalty}."
            )
        if not -2.0 <= self.frequency_penalty <= 2.0:
            raise VLLMValidationError(
                f"frequency_penalty must be in [-2, 2], got {self.frequency_penalty}."
            )
        if not math.isfinite(self.repetition_penalty):
            raise VLLMValidationError(
                "repetition_penalty must be a finite number, "
                f"got {self.repetition_penalty}."
            )
        if self.repetition_penalty <= 0.0:
            raise VLLMValidationError(
                "repetition_penalty must be greater than zero, got "
                f"{self.repetition_penalty}."
            )
        if not math.isfinite(self.temperature):
            raise VLLMValidationError(
                f"temperature must be a finite number, got {self.temperature}.",
                parameter="temperature",
                value=self.temperature,
            )
        if self.temperature < 0.0:
            raise VLLMValidationError(
                f"temperature must be non-negative, got {self.temperature}.",
                parameter="temperature",
                value=self.temperature,
            )
        if self.temperature > 2.0:
            raise VLLMValidationError(
                f"temperature must be in [0, 2], got {self.temperature}.",
                parameter="temperature",
                value=self.temperature,
            )
        if not 0.0 < self.top_p <= 1.0:
            raise VLLMValidationError(
                f"top_p must be in (0, 1], got {self.top_p}.",
                parameter="top_p",
                value=self.top_p,
            )
        # quietly accept -1 as disabled, but prefer 0
        if self.top_k < -1:
            raise VLLMValidationError(
                f"top_k must be 0 (disable), or at least 1, got {self.top_k}."
            )
        if not isinstance(self.top_k, int):
            raise VLLMValidationError(
                f"top_k must be an integer, got {type(self.top_k).__name__}"
            )
        if not 0.0 <= self.min_p <= 1.0:
            raise VLLMValidationError(f"min_p must be in [0, 1], got {self.min_p}.")
        if self.max_tokens is not None and self.max_tokens < 1:
            raise VLLMValidationError(
                f"max_tokens must be at least 1, got {self.max_tokens}.",
                parameter="max_tokens",
                value=self.max_tokens,
            )
        if self.min_tokens < 0:
            raise VLLMValidationError(
                f"min_tokens must be greater than or equal to 0, got {self.min_tokens}."
            )
        if self.max_tokens is not None and self.min_tokens > self.max_tokens:
            raise VLLMValidationError(
                f"min_tokens must be less than or equal to "
                f"max_tokens={self.max_tokens}, got {self.min_tokens}."
            )
        if self.stream_interval is not None and self.stream_interval < 1:
            raise VLLMValidationError(
                f"stream_interval must be at least 1, got {self.stream_interval}.",
                parameter="stream_interval",
                value=self.stream_interval,
            )
        if self.logprobs is not None and self.logprobs != -1 and self.logprobs < 0:
            raise VLLMValidationError(
                f"logprobs must be non-negative or -1, got {self.logprobs}.",
                parameter="logprobs",
                value=self.logprobs,
            )
        if (
            self.prompt_logprobs is not None
            and self.prompt_logprobs != -1
            and self.prompt_logprobs < 0
        ):
            raise VLLMValidationError(
                f"prompt_logprobs must be non-negative or -1, got "
                f"{self.prompt_logprobs}.",
                parameter="prompt_logprobs",
                value=self.prompt_logprobs,
            )
        assert isinstance(self.stop_token_ids, list)
        if not all(isinstance(st_id, int) for st_id in self.stop_token_ids):
            raise VLLMValidationError(
                f"stop_token_ids must contain only integers, got {self.stop_token_ids}."
            )
        assert isinstance(self.stop, list)
        if any(not stop_str for stop_str in self.stop):
            raise VLLMValidationError("stop cannot contain an empty string.")
        if self.stop and not self.detokenize:
            raise VLLMValidationError(
                "stop strings are only supported when detokenize is True. "
                "Set detokenize=True to use stop."
            )
        assert isinstance(self.bad_words, list)
        if any(not bad_word for bad_word in self.bad_words):
            raise VLLMValidationError(
                f"bad_words cannot contain an empty string. "
                f"Got bad_words={self.bad_words}"
            )

    # ------【核心逻辑】_verify_greedy_sampling：贪心采样时校验 n 必须为 1 ------
    def _verify_greedy_sampling(self) -> None:
        if self.n > 1:
            raise VLLMValidationError(
                f"n must be 1 when using greedy sampling, got {self.n}."
            )

    # ------【核心逻辑】update_from_generation_config：用 generation_config 的 eos 等默认值回填参数，聚合停止 token ------
    def update_from_generation_config(
        self,
        generation_config: dict[str, Any],
        eos_token_id: int | None = None,
    ) -> None:
        """Update if there are non-default values from generation_config"""
        if not self.ignore_eos:
            self._eos_token_id = eos_token_id

        if eos_token_id is not None:
            # Add the eos token id into the sampling_params to support
            # min_tokens processing.
            self._all_stop_token_ids.add(eos_token_id)

        # Update eos_token_id for generation
        if (eos_ids := generation_config.get("eos_token_id")) is not None:
            # it can be either int or list of int
            eos_ids = {eos_ids} if isinstance(eos_ids, int) else set(eos_ids)
            if eos_token_id is not None:
                # We don't need to include the primary eos_token_id in
                # stop_token_ids since it's handled separately for stopping
                # purposes.
                eos_ids.discard(eos_token_id)
            if eos_ids:
                self._all_stop_token_ids.update(eos_ids)
                if not self.ignore_eos:
                    assert self.stop_token_ids is not None
                    eos_ids.update(self.stop_token_ids)
                    self.stop_token_ids = list(eos_ids)

    # ------【核心逻辑】update_from_tokenizer：把 bad_words 编码为 token id 序列（含前缀空格两种形式） ------
    def update_from_tokenizer(self, tokenizer: TokenizerLike) -> None:
        if not self.bad_words:
            return
        self._bad_words_token_ids = []
        for bad_word in self.bad_words:
            # To prohibit words both at the beginning
            # and in the middle of text
            # (related to add_prefix_space tokenizer parameter)
            for add_prefix_space in [False, True]:
                prefix = " " if add_prefix_space else ""
                prompt = prefix + bad_word.lstrip()
                prompt_token_ids = tokenizer.encode(
                    text=prompt, add_special_tokens=False
                )

                # If no space at the beginning
                # or if prefix space produces a new word token
                if (not add_prefix_space) or (
                    add_prefix_space
                    and prompt_token_ids[0] != self._bad_words_token_ids[-1][0]
                    and len(prompt_token_ids) == len(self._bad_words_token_ids[-1])
                ):
                    self._bad_words_token_ids.append(prompt_token_ids)

        invalid_token_ids = [
            token_id
            for bad_words_token_ids in self._bad_words_token_ids
            for token_id in bad_words_token_ids
            if token_id < 0 or token_id > tokenizer.max_token_id
        ]
        if len(invalid_token_ids) > 0:
            raise VLLMValidationError(
                f"The model vocabulary size is {tokenizer.max_token_id + 1},"
                f" but the following tokens"
                f" were specified as bad: {invalid_token_ids}."
                f" All token id values should be integers satisfying:"
                f" 0 <= token_id <= {tokenizer.max_token_id}.",
                parameter="bad_words",
                value=self.bad_words,
            )

    # ------【核心逻辑】sampling_type：根据 temperature/seed 推导采样类型，缓存避免重复计算 ------
    @cached_property
    def sampling_type(self) -> SamplingType:
        if self.temperature < _SAMPLING_EPS:
            return SamplingType.GREEDY
        if self.seed is not None:
            return SamplingType.RANDOM_SEED
        return SamplingType.RANDOM

    @property
    def eos_token_id(self) -> int | None:
        return self._eos_token_id

    @property
    def all_stop_token_ids(self) -> set[int]:
        return self._all_stop_token_ids

    @property
    def bad_words_token_ids(self) -> list[list[int]] | None:
        # For internal use only. Backward compatibility not guaranteed
        return self._bad_words_token_ids

    # ------【核心逻辑】num_logprobs：实际返回的 sample logprobs 数，计入 logprob_token_ids ------
    @property
    def num_logprobs(self) -> int | None:
        """Number of sample logprobs to return per output token, or `None` if
        no sample logprobs were requested. Takes `logprob_token_ids` into
        account: when `logprobs` is unset but `logprob_token_ids` is set,
        returns `len(logprob_token_ids)`."""
        if self.logprobs is not None:
            return self.logprobs
        return len(self.logprob_token_ids) if self.logprob_token_ids else None

    # ------【核心逻辑】clone：skip_clone 时浅拷贝否则深拷贝，控制请求级副本开销 ------
    def clone(self) -> "SamplingParams":
        """If skip_clone is True, uses shallow copy instead of deep copy."""
        if self.skip_clone:
            return copy.copy(self)

        return copy.deepcopy(self)

    # ------【核心逻辑】verify：结合 model/投机/结构化输出配置做引擎级参数校验入口 ------
    def verify(
        self,
        model_config: ModelConfig,
        speculative_config: SpeculativeConfig | None,
        structured_outputs_config: StructuredOutputsConfig | None,
        tokenizer: TokenizerLike | None,
    ) -> None:
        self._validate_logprobs(model_config)
        self._validate_logit_bias(model_config)
        self._validate_logits_processors(model_config)
        self._validate_allowed_token_ids(tokenizer)
        self._validate_spec_decode(speculative_config)
        self._validate_diffusion(model_config)
        self._validate_structured_outputs(
            model_config, structured_outputs_config, tokenizer
        )

    # ------【核心逻辑】_validate_logprobs：校验 logprobs 不超模型 max_logprobs 上限 ------
    def _validate_logprobs(self, model_config: ModelConfig) -> None:
        max_logprobs = model_config.max_logprobs
        if max_logprobs == -1:
            max_logprobs = model_config.get_vocab_size()

        # Validate sample logprobs.
        if num_logprobs := self.logprobs:
            if num_logprobs == -1:
                num_logprobs = model_config.get_vocab_size()
            if num_logprobs > max_logprobs:
                raise VLLMValidationError(
                    f"Requested sample logprobs of {num_logprobs}, "
                    f"which is greater than max allowed: {max_logprobs}",
                    parameter="logprobs",
                    value=num_logprobs,
                )

        # Validate logprob_token_ids.
        if self.logprob_token_ids is not None:
            n = len(self.logprob_token_ids)
            if n > MAX_LOGPROB_TOKEN_IDS:
                raise VLLMValidationError(
                    f"Requested logprob_token_ids of length {n}, "
                    f"which is greater than max allowed: {MAX_LOGPROB_TOKEN_IDS}",
                    parameter="logprob_token_ids",
                    value=n,
                )
            vocab_size = model_config.get_vocab_size()
            invalid_token_ids = [
                token_id
                for token_id in self.logprob_token_ids
                if token_id < 0 or token_id >= vocab_size
            ]
            if invalid_token_ids:
                raise VLLMValidationError(
                    f"token_id(s) {invalid_token_ids} in logprob_token_ids "
                    f"contain out-of-vocab token ids. Vocabulary size: "
                    f"{vocab_size}",
                    parameter="logprob_token_ids",
                    value=invalid_token_ids,
                )
            if self.logprobs is not None and self.logprobs != n:
                raise VLLMValidationError(
                    f"When both logprobs and logprob_token_ids are set, "
                    f"logprobs must equal len(logprob_token_ids). Got "
                    f"logprobs={self.logprobs}, len(logprob_token_ids)={n}.",
                    parameter="logprob_token_ids",
                    value=n,
                )

        # Validate prompt logprobs.
        if num_prompt_logprobs := self.prompt_logprobs:
            if num_prompt_logprobs == -1:
                num_prompt_logprobs = model_config.get_vocab_size()
            if num_prompt_logprobs > max_logprobs:
                raise VLLMValidationError(
                    f"Requested prompt logprobs of {num_prompt_logprobs}, "
                    f"which is greater than max allowed: {max_logprobs}",
                    parameter="prompt_logprobs",
                    value=num_prompt_logprobs,
                )

    # ------【核心逻辑】_validate_logit_bias：校验 logit_bias token id 在词表范围内 ------
    def _validate_logit_bias(self, model_config: ModelConfig) -> None:
        """Validate logit_bias token IDs are within vocabulary range."""
        if not self.logit_bias:
            return

        vocab_size = model_config.get_vocab_size()
        invalid_token_ids = [
            token_id
            for token_id in self.logit_bias
            if token_id < 0 or token_id >= vocab_size
        ]

        if invalid_token_ids:
            raise VLLMValidationError(
                f"token_id(s) {invalid_token_ids} in logit_bias contain "
                f"out-of-vocab token ids. Vocabulary size: {vocab_size}",
                parameter="logit_bias",
                value=invalid_token_ids,
            )

    # ------【核心逻辑】_validate_logits_processors：委托校验自定义 logits processor 参数 ------
    def _validate_logits_processors(self, model_config: ModelConfig) -> None:
        from vllm.v1.sample.logits_processor import (
            validate_logits_processors_parameters,
        )

        validate_logits_processors_parameters(model_config.logits_processors, self)

    # ------【核心逻辑】_validate_allowed_token_ids：校验 allowed_token_ids 非空且在词表内 ------
    def _validate_allowed_token_ids(self, tokenizer: TokenizerLike | None) -> None:
        allowed_token_ids = self.allowed_token_ids
        if allowed_token_ids is None:
            return

        if len(allowed_token_ids) == 0:
            raise VLLMValidationError(
                "allowed_token_ids is not None and empty!",
                parameter="allowed_token_ids",
                value=allowed_token_ids,
            )

        if tokenizer is not None:
            vocab_size = len(tokenizer)
            invalid_token_ids = [
                token_id
                for token_id in allowed_token_ids
                if token_id < 0 or token_id >= vocab_size
            ]
            if invalid_token_ids:
                raise VLLMValidationError(
                    "allowed_token_ids contains out-of-vocab token id!",
                    parameter="allowed_token_ids",
                    value=invalid_token_ids,
                )

    # ------【投机解码】_validate_spec_decode：校验参数与投机解码兼容性（min_p/logit_bias 暂不支持） ------
    def _validate_spec_decode(
        self,
        speculative_config: SpeculativeConfig | None,
    ) -> None:
        if speculative_config is None:
            return

        # Some sampling parameters are not yet compatible with spec decoding.
        if self.min_p > _SAMPLING_EPS or self.logit_bias:
            raise VLLMValidationError(
                "The min_p and logit_bias sampling parameters "
                "are not yet supported with speculative decoding."
            )

    # ------【核心逻辑】_validate_diffusion：扩散模型不支持逐请求采样参数时抛错 ------
    def _validate_diffusion(self, model_config: ModelConfig) -> None:
        if not model_config.is_diffusion:
            return

        # Diffusion models denoise a whole canvas per step with a fixed
        # temperature schedule, so per-request sampling parameters are not
        # supported. Penalties are ignored by the sampler with a warning.
        if (
            self.temperature != 1.0
            or self.min_p > _SAMPLING_EPS
            or self.seed is not None
            or self.min_tokens > 0
            or self.logit_bias
            or self.bad_words
            or self.allowed_token_ids
        ):
            raise VLLMValidationError(
                "The temperature, min_p, seed, min_tokens, logit_bias, "
                "bad_words, and allowed_token_ids sampling parameters "
                "are not yet supported with diffusion models."
            )

    # ------【结构化输出/grammar】_validate_structured_outputs：选定/校验约束解码后端并按后端验证请求 ------
    def _validate_structured_outputs(
        self,
        model_config: ModelConfig,
        structured_outputs_config: StructuredOutputsConfig | None,
        tokenizer: TokenizerLike | None,
    ) -> None:
        if structured_outputs_config is None or self.structured_outputs is None:
            return

        if model_config.is_diffusion:
            # Diffusion LLMs denoise a whole canvas of tokens in parallel
            # rather than sampling left-to-right, which the grammar FSM
            # requires. Without this check, requests fail mid-generation
            # with an FSM rejection (HTTP 500). See issue #45436.
            raise VLLMValidationError(
                "Structured outputs are not yet supported for diffusion "
                "language models. Remove the structured output constraint "
                "(e.g. `response_format`, `structured_outputs`) from the "
                "request."
            )

        if tokenizer is None:
            raise VLLMValidationError(
                "Structured outputs requires a tokenizer so it can't be used with 'skip_tokenizer_init'"  # noqa: E501
            )

        backend = structured_outputs_config.backend
        if _backend := self.structured_outputs._backend:
            # Request-level backend selection is not supported.
            # The values may differ if `params` is reused and was set
            # to a specific backend based on `auto` behavior in a previous
            # request. We remember that it was set as a result of `auto`
            # using the `_backend_was_auto` field set in the params.
            if backend != _backend and not (
                backend == "auto" and self.structured_outputs._backend_was_auto
            ):
                raise VLLMValidationError(
                    "Request-level structured output backend selection is not "
                    f"supported. The request specified '{_backend}', but vLLM "
                    f"was initialised with '{backend}'. This error can be "
                    "resolved by removing '_backend' from the request."
                )
        else:
            self.structured_outputs._backend = backend

        # Request content validation
        if (
            isinstance(self.structured_outputs.choice, list)
            and not self.structured_outputs.choice
        ):
            # It is invalid for choice to be an empty list
            raise VLLMValidationError(
                f"Choice '{self.structured_outputs.choice}' cannot be an empty list"  # noqa: E501
            )
        # Reject empty string grammar early to avoid engine-side crashes
        if (
            isinstance(self.structured_outputs.grammar, str)
            and self.structured_outputs.grammar.strip() == ""
        ):
            raise VLLMValidationError(
                "structured_outputs.grammar cannot be an empty string"
            )
        # Reject empty string json schema early to avoid engine-side crashes
        if (
            isinstance(self.structured_outputs.json, str)
            and self.structured_outputs.json.strip() == ""
        ):
            raise VLLMValidationError(
                "structured_outputs.json cannot be an empty string"
            )
        # Reject json_object=False early to avoid engine-side crashes
        if self.structured_outputs.json_object is False:
            raise VLLMValidationError(
                "structured_outputs.json_object must be True if set; omit "
                "structured_outputs to disable structured outputs"
            )

        from vllm.v1.structured_output.backend_guidance import (
            has_guidance_unsupported_json_features,
            validate_guidance_grammar,
        )
        from vllm.v1.structured_output.backend_lm_format_enforcer import (
            validate_structured_output_request_lm_format_enforcer,
        )
        from vllm.v1.structured_output.backend_outlines import (
            validate_structured_output_request_outlines,
        )
        from vllm.v1.structured_output.backend_xgrammar import validate_xgrammar_grammar

        if backend.startswith("xgrammar"):
            # xgrammar with no fallback
            validate_xgrammar_grammar(self)
        elif backend.startswith("guidance"):
            if _is_non_tekken_mistral(tokenizer=tokenizer):
                raise VLLMValidationError(
                    "Non-tekken Mistral tokenizers are not supported for the 'guidance'"
                    " structured output backend. Please either use a more recent "
                    "Mistral model, the ['xgrammar', 'outlines'] "
                    "backends or tokenizer_mode='hf' instead."
                )
            # TODO: ideally we would have the LLTokenizer here as Lark syntax
            # allows <|special_token|> and similar, see
            # https://github.com/guidance-ai/llguidance/blob/main/docs/syntax.md#special-tokens
            # Without tokenizer these are disallowed in grammars.
            validate_guidance_grammar(
                self,
                tokenizer=_get_llg_tokenizer(tokenizer),
            )
        elif backend == "outlines":
            # outlines backend
            validate_structured_output_request_outlines(self)
        elif backend == "lm-format-enforcer":
            # lm format enforcer backend
            if is_mistral_tokenizer(tokenizer):
                raise VLLMValidationError(
                    "Mistral tokenizer is not supported for the 'lm-format-enforcer' "
                    "structured output backend. Please use ['xgrammar', 'outlines'] "
                    "backends or tokenizer_mode='hf' instead."
                )
            validate_structured_output_request_lm_format_enforcer(self)
        else:
            # NOTE: backend must be "auto" here, because we have
            # checked supported_backends above.
            # In this mode, we set opinionated defaults based on what we think
            # will satisfy the most use cases without having to worry about
            # this setting. We include fallback behavior here, but not with any
            # other setting where a specific backend was specified.
            try:
                validate_xgrammar_grammar(self)
                self.structured_outputs._backend = "xgrammar"
            except ValueError:
                # The request either failed validation
                # or includes some jsonschema feature(s) that
                # are not supported in xgrammar.

                skip_guidance = _is_non_tekken_mistral(tokenizer)

                # Check if schema has features unsupported by guidance
                so_params = self.structured_outputs
                if not skip_guidance and so_params.json:
                    if isinstance(so_params.json, str):
                        schema = json_mod.loads(so_params.json)
                    else:
                        schema = so_params.json
                    skip_guidance = has_guidance_unsupported_json_features(schema)

                if skip_guidance:
                    # Fall back to outlines if the tokenizer is non-tekken Mistral or
                    # the schema contains features unsupported by guidance
                    validate_structured_output_request_outlines(self)
                    self.structured_outputs._backend = "outlines"
                else:
                    # Fall back to guidance by default.
                    validate_guidance_grammar(
                        self,
                        tokenizer=_get_llg_tokenizer(tokenizer),
                    )
                    self.structured_outputs._backend = "guidance"
            # Remember that this backend was set automatically
            self.structured_outputs._backend_was_auto = True

        # Run post-init validation. This is also important to ensure subsequent
        # roundtrip serialization/deserialization won't fail.
        self.structured_outputs.__post_init__()

    def __repr__(self) -> str:
        return (
            f"SamplingParams(n={self.n}, "
            f"presence_penalty={self.presence_penalty}, "
            f"frequency_penalty={self.frequency_penalty}, "
            f"repetition_penalty={self.repetition_penalty}, "
            f"temperature={self.temperature}, "
            f"top_p={self.top_p}, "
            f"top_k={self.top_k}, "
            f"min_p={self.min_p}, "
            f"seed={self.seed}, "
            f"stop={self.stop}, "
            f"stop_token_ids={self.stop_token_ids}, "
            f"bad_words={self.bad_words}, "
            f"thinking_token_budget={self.thinking_token_budget}, "
            f"include_stop_str_in_output={self.include_stop_str_in_output}, "
            f"ignore_eos={self.ignore_eos}, "
            f"max_tokens={self.max_tokens}, "
            f"min_tokens={self.min_tokens}, "
            f"logprobs={self.logprobs}, "
            f"prompt_logprobs={self.prompt_logprobs}, "
            f"skip_special_tokens={self.skip_special_tokens}, "
            "spaces_between_special_tokens="
            f"{self.spaces_between_special_tokens}, "
            f"structured_outputs={self.structured_outputs}, "
            f"extra_args={self.extra_args})"
        )

    # ------【CUDA Graph】for_sampler_warmup：构造覆盖全采样分支的参数用于采样器预热/图捕获 ------
    @staticmethod
    def for_sampler_warmup() -> "SamplingParams":
        """Set parameters to exercise all sampler logic."""
        return SamplingParams(
            temperature=0.9,
            top_p=0.9,
            top_k=50,
            min_p=0.1,
            frequency_penalty=0.5,
            presence_penalty=0.5,
            repetition_penalty=1.2,
            min_tokens=2,
            logit_bias={0: -1.0, 1: 0.5},
            _bad_words_token_ids=[[0], [1, 2]],
            logprobs=5,
            prompt_logprobs=1,
        )


# ------【核心逻辑】BeamSearchParams：束搜索参数 DTO，配置 beam search 解码的束宽与长度惩罚 ------
class BeamSearchParams(
    msgspec.Struct,
    omit_defaults=True,  # type: ignore[call-arg]
    # required for @cached_property.
    dict=True,
):  # type: ignore[call-arg]
    """Beam search parameters for text generation."""

    # ------【核心逻辑】beam_width：束宽，同时维护的候选序列数，越大搜索越广耗时越高 ------
    beam_width: int
    # ------【核心逻辑】max_tokens：每个束序列最大生成 token 数 ------
    max_tokens: int
    # ------【核心逻辑】ignore_eos：是否忽略 EOS token 继续生成 ------
    ignore_eos: bool = False
    # ------【核心逻辑】temperature：采样温度，束搜索默认 0 走贪心打分 ------
    temperature: float = 0.0
    # ------【核心逻辑】length_penalty：长度惩罚，>1 偏向长序列，<1 偏向短序列 ------
    length_penalty: float = 1.0
    # ------【核心逻辑】include_stop_str_in_output：输出文本是否保留停止字符串 ------
    include_stop_str_in_output: bool = False
    # ------【结构化输出/grammar】structured_outputs：结构化输出约束，构造约束解码 logit processor ------
    structured_outputs: StructuredOutputsParams | None = None
