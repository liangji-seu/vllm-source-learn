# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING

from vllm.config.ec_manager_config import EncoderCacheManagerMetadata

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt
    import torch

    from vllm.distributed.ec_transfer.ec_connector.base import ECConnectorMetadata
    from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata
    from vllm.lora.request import LoRARequest
    from vllm.multimodal.inputs import MultiModalFeatureSpec
    from vllm.pooling_params import PoolingParams
    from vllm.sampling_params import SamplingParams
    from vllm.v1.core.kv_cache_utils import KVCacheBlockCopy
    from vllm.v1.request import Request
else:
    ECConnectorMetadata = object
    KVConnectorMetadata = object
    KVCacheBlockCopy = object
    LoRARequest = object
    MultiModalFeatureSpec = object
    PoolingParams = object
    SamplingParams = object
    Request = object


# ------【核心逻辑】NewRequestData：Scheduler→Worker 的首次调度整包消息，走 MessageQueue 下发后 worker 缓存复用，避免每步重发完整请求数据 ------
@dataclass
class NewRequestData:
    # ------【核心逻辑】req_id：请求唯一标识，作为 worker 端缓存与状态跟踪的 key ------
    req_id: str
    # ------【核心逻辑】prompt_token_ids：原始 prompt 的 token 序列；embedding 输入时可为 None，改用 prompt_embeds ------
    prompt_token_ids: list[int] | None
    # ------【核心逻辑】mm_features：多模态特征（图像/音频等），供多模态编码器与输入注入使用 ------
    mm_features: list[MultiModalFeatureSpec]
    # ------【核心逻辑】sampling_params：采样参数（温度/top-k/top-p 等），仅文本生成类请求非空 ------
    sampling_params: SamplingParams | None
    # ------【核心逻辑】pooling_params：pooling/embedding 类模型的汇聚参数，与 sampling 互斥使用 ------
    pooling_params: PoolingParams | None
    # ------【内存池/CuMem】block_ids：该请求占用的 KV cache block id（每层一个 list），指向显存池中的块 ------
    block_ids: tuple[list[int], ...]
    # ------【前缀缓存】num_computed_tokens：已计算过的 token 数，作为前缀复用起点跳过重复计算 ------
    num_computed_tokens: int
    # ------【LoRA】lora_request：请求绑定的 LoRA 适配器，worker 据此加载对应低秩权重 ------
    lora_request: LoRARequest | None
    # ------【核心逻辑】prompt_embeds：预计算的 prompt embedding（embedding 类输入），避免重复 encode ------
    prompt_embeds: "torch.Tensor | None" = None
    # ------【核心逻辑】prompt_is_token_ids：逐项标记多模态输入是 token 序列还是原始特征，供输入组装判断 ------
    prompt_is_token_ids: list[bool] | None = None

    # ------【核心逻辑】prefill_token_ids：仅 v2 runner 使用的 prefill token 序列，与 prompt_token_ids 分离传输 ------
    # Only used for v2 model runner.
    prefill_token_ids: list[int] | None = None

    @classmethod
    def from_request(
        cls,
        request: Request,
        block_ids: tuple[list[int], ...],
        prefill_token_ids: list[int] | None = None,
    ) -> "NewRequestData":
        # ------【核心逻辑】把 Request 字段映射成 NewRequestData 首次调度整包下发，worker 缓存后免重复传输 ------
        return cls(
            req_id=request.request_id,
            prompt_token_ids=request.prompt_token_ids,
            mm_features=request.mm_features,
            sampling_params=request.sampling_params,
            pooling_params=request.pooling_params,
            block_ids=block_ids,
            num_computed_tokens=request.num_computed_tokens,
            lora_request=request.lora_request,
            prompt_embeds=request.prompt_embeds,
            prompt_is_token_ids=request.prompt_is_token_ids,
            prefill_token_ids=prefill_token_ids,
        )

    def __repr__(self) -> str:
        # ------【核心逻辑】只取 prompt_embeds 的形状而非张量内容，避免日志打印超长/敏感数据 ------
        prompt_embeds_shape = (
            self.prompt_embeds.shape if self.prompt_embeds is not None else None
        )
        # ------【核心逻辑】拼接关键字段为可读字符串，便于日志/调试定位请求状态 ------
        return (
            f"NewRequestData("
            f"req_id={self.req_id},"
            f"prompt_token_ids={self.prompt_token_ids},"
            f"prefill_token_ids={self.prefill_token_ids},"
            f"mm_features={self.mm_features},"
            f"sampling_params={self.sampling_params},"
            f"block_ids={self.block_ids},"
            f"num_computed_tokens={self.num_computed_tokens},"
            f"lora_request={self.lora_request},"
            f"prompt_embeds_shape={prompt_embeds_shape}"
            ")"
        )

    # Version of __repr__ with the prompt data obfuscated
    def anon_repr(self) -> str:
        # ------【核心逻辑】只统计 prompt 相关字段长度做脱敏，避免日志泄露 prompt 原文内容 ------
        prompt_token_ids_len = (
            len(self.prompt_token_ids) if self.prompt_token_ids is not None else None
        )
        prompt_embeds_shape = (
            self.prompt_embeds.shape if self.prompt_embeds is not None else None
        )
        prefill_token_ids_len = (
            len(self.prefill_token_ids) if self.prefill_token_ids is not None else None
        )
        # ------【核心逻辑】用长度替代原文拼接脱敏后的请求信息，用于安全日志输出 ------
        return (
            f"NewRequestData("
            f"req_id={self.req_id},"
            f"prompt_token_ids_len={prompt_token_ids_len},"
            f"prefill_token_ids_len={prefill_token_ids_len},"
            f"mm_features={self.mm_features},"
            f"sampling_params={self.sampling_params},"
            f"block_ids={self.block_ids},"
            f"num_computed_tokens={self.num_computed_tokens},"
            f"lora_request={self.lora_request},"
            f"prompt_embeds_shape={prompt_embeds_shape}"
            ")"
        )


# ------【核心逻辑】CachedRequestData：Scheduler→Worker 的增量调度消息（diff），worker 已缓存请求数据，只下发本轮新增部分以省通信 ------
@dataclass
class CachedRequestData:
    # ------【核心逻辑】req_ids：本轮被增量调度的已缓存请求 id 列表（与下面各字段按下标对齐） ------
    req_ids: list[str]
    # ------【前缀缓存】resumed_req_ids：从 prefix cache 恢复的请求集合，其 new_block_ids 直接替换而非追加旧 block ------
    # For request ids not in resumed_req_ids, new_block_ids will be appended to
    # the request's block IDs. For those in the set, new_block_ids will be used as the
    # request's block IDs instead of appending to the existing block IDs.
    resumed_req_ids: set[str]
    # ------【PP】new_token_ids：仅流水线并行使用，非 PP 时为空；本轮每个请求新增的 token id ------
    # NOTE(woosuk): new_token_ids is only used for pipeline parallelism.
    # When PP is not used, new_token_ids will be empty.
    new_token_ids: list[list[int]]
    # ------【核心逻辑】all_token_ids：MRV1-only，未在上一轮调度的请求需传给 connector 的 token id ------
    # MRV1-only: For requests not scheduled in the last step, propagate the token ids
    # to the connector. Won't contain requests scheduled in the prior step.
    all_token_ids: dict[str, list[int]]
    # ------【内存池/CuMem】new_block_ids：本轮新分配给各请求的 KV cache block id（resumed 请求为替换语义） ------
    new_block_ids: list[tuple[list[int], ...] | None]
    # ------【前缀缓存】num_computed_tokens：各请求本轮已计算 token 数，决定从哪个位置继续计算 ------
    num_computed_tokens: list[int]
    # ------【核心逻辑】num_output_tokens：各请求累计已生成的输出 token 数，用于判断 prefill/decode 阶段 ------
    num_output_tokens: list[int]

    # Version of dataclass repr with token IDs obfuscated.
    def anon_repr(self) -> str:
        # ------【核心逻辑】把 token 列表转为长度，脱敏后仅保留计数用于日志 ------
        new_token_ids_lens = [len(toks) for toks in self.new_token_ids]
        all_token_ids_lens = {
            req_id: len(toks) for req_id, toks in self.all_token_ids.items()
        }
        # ------【核心逻辑】拼接缓存请求的脱敏信息，便于日志追踪增量调度状态 ------
        return (
            f"CachedRequestData("
            f"req_ids={self.req_ids},"
            f"resumed_req_ids={self.resumed_req_ids},"
            f"new_token_ids_lens={new_token_ids_lens},"
            f"all_token_ids_lens={all_token_ids_lens},"
            f"new_block_ids={self.new_block_ids},"
            f"num_computed_tokens={self.num_computed_tokens},"
            f"num_output_tokens={self.num_output_tokens}"
            f")"
        )

    def __repr__(self) -> str:
        # ------【核心逻辑】统一走 anon_repr，所有日志输出默认对 token 内容脱敏 ------
        return self.anon_repr()

    @property
    def num_reqs(self) -> int:
        # ------【核心逻辑】返回本次缓存的请求数量，供调度器统计增量请求规模 ------
        return len(self.req_ids)

    @cached_property
    def _req_id_to_num_output_tokens(self) -> dict[str, int]:
        """Cache mapping of req_id to num_output_tokens for O(1) lookup.

        This cached property is safe because CachedRequestData instances
        are created fresh each scheduling iteration and not mutated during
        computation of iteration details.
        """
        # ------【核心逻辑】构建 req_id→输出 token 数的哈希映射，让后续查询 O(1) 而非线性扫描 ------
        return dict(zip(self.req_ids, self.num_output_tokens))

    def is_context_phase(self, req_id: str) -> bool:
        # ------【核心逻辑】以输出 token 数是否为 0 判断请求是否仍处于 prefill/上下文阶段 ------
        num_output_tokens = self._req_id_to_num_output_tokens.get(req_id)
        return num_output_tokens is not None and num_output_tokens == 0

    @classmethod
    def make_empty(cls) -> "CachedRequestData":
        # ------【核心逻辑】构造空的 CachedRequestData 占位对象，表示本轮没有已缓存的增量请求 ------
        return cls(
            req_ids=[],
            resumed_req_ids=set(),
            new_token_ids=[],
            all_token_ids={},
            new_block_ids=[],
            num_computed_tokens=[],
            num_output_tokens=[],
        )


# ------【核心逻辑】ScheduledEncoderInputStats：Scheduler 内部对单轮编码器输入的统计计数，用于显存/计算预算评估 ------
@dataclass
class ScheduledEncoderInputStats:
    """Stats for encoder inputs scheduled in one iteration."""

    # ------【核心逻辑】num_inputs：本轮调度需要处理的编码器输入数量（如多模态图片张数） ------
    num_inputs: int = 0
    # ------【核心逻辑】output_tokens：这些编码器输入预计产生的 token 数，用于输入长度预算 ------
    output_tokens: int = 0


# ------【核心逻辑】SchedulerOutput：Scheduler→Worker 每步的核心打包消息（MessageQueue 异步下发），聚合新请求/增量/投机/编码器/前缀缓存等全部调度结果 ------
@dataclass
class SchedulerOutput:
    # list of the requests that are scheduled for the first time.
    # We cache the request's data in each worker process, so that we don't
    # need to re-send it every scheduling step.
    scheduled_new_reqs: list[NewRequestData] # 本轮调度的新req名单（prompt, chunked）
    # list of the requests that have been scheduled before.
    # Since the request's data is already cached in the worker processes,
    # we only send the diff to minimize the communication cost.
    scheduled_cached_reqs: CachedRequestData # 本轮调度的旧req名单(chunked, decode)

    # req_id -> num_scheduled_tokens
    # Number of tokens scheduled for each request.
    num_scheduled_tokens: dict[str, int] # 本轮的req的计算的token数列表
    # Total number of tokens scheduled for all requests.
    # Equal to sum(num_scheduled_tokens.values())
    total_num_scheduled_tokens: int # 本轮需要计算的总的tokens数列表
    # req_id -> spec_token_ids
    # If a request does not have any spec decode tokens, it will not be
    # included in the dictionary.
    scheduled_spec_decode_tokens: dict[str, list[int]] # 本轮的req里面，需要大模型验证的草稿token列表 的 字典
    # req_id -> encoder input indices that need processing.
    # E.g., if a request has [0, 1], it could mean the vision encoder needs
    # to process that the request's 0-th and 1-th images in the current step.
    # ------【核心逻辑】scheduled_encoder_inputs：req_id→本轮需处理的编码器输入索引（如第几张图），供多模态编码器按需计算 ------
    scheduled_encoder_inputs: dict[str, list[int]]
    # Number of common prefix blocks for all requests in each KV cache group.
    # This can be used for cascade attention.
    num_common_prefix_blocks: list[int] # 最常用的前缀的blocks

    # Request IDs that are finished in between the previous and the current
    # steps. This is used to notify the workers about the finished requests
    # so that they can free the cached states for those requests.
    finished_req_ids: set[str] # 上一轮结束的req
    # list of mm_hash strings associated with the encoder outputs to be
    # freed from the encoder cache.
    # ------【核心逻辑】free_encoder_mm_hashes：需从编码器缓存释放的 mm_hash 列表，编码器结果按 hash 复用，此处触发淘汰 ------
    free_encoder_mm_hashes: list[str]

    # ------【核心逻辑】scheduled_encoder_input_stats：本轮编码器输入的统计信息，供编码器侧预算与调度决策 ------
    scheduled_encoder_input_stats: ScheduledEncoderInputStats | None = None

    # Request IDs that are preempted in this step.
    # Only used for v2 model runner.
    preempted_req_ids: set[str] | None = None # 本轮被抢占，赶回waiting的req

    # Whether any of the scheduled requests use structured output.
    # Set only in async scheduling case.
    # ------【结构化输出/grammar】has_structured_output_requests：本轮是否含结构化输出请求，决定是否走 grammar 约束解码 ------
    has_structured_output_requests: bool = False

    # Whether the scheduled requests have all the output tokens they
    # need to perform grammar bitmask computation.
    # ------【结构化输出/grammar】pending_structured_output_tokens：请求是否已有全部输出 token 以进行 grammar bitmask 计算 ------
    pending_structured_output_tokens: bool = False

    # Used for adjusting acceptance rate calculation.
    # ------【投机解码】num_invalid_spec_tokens：req_id→被拒草稿 token 数，用于校准投机接受率统计 ------
    num_invalid_spec_tokens: dict[str, int] | None = None

    # KV Cache Connector metadata.
    # ------【PD 分离/异步 RPC】kv_connector_metadata：外部 KV 传输 connector 元数据，用于 disaggregation 场景跨实例搬运 KV cache ------
    kv_connector_metadata: KVConnectorMetadata | None = None

    # EC Cache Connector metadata
    # ------【异步 RPC】ec_connector_metadata：编码器缓存 connector 元数据，用于分布式编码器结果传输 ------
    ec_connector_metadata: ECConnectorMetadata | None = None
    # EC Cache Manager metadata
    # ------【核心逻辑】ec_manager_metadata：编码器缓存管理器元数据，描述编码器缓存内容/复用状态 ------
    ec_manager_metadata: EncoderCacheManagerMetadata | None = None
    # Block IDs freshly allocated from the pool during this scheduling step.
    # The worker zeros the corresponding GPU memory before the blocks are used,
    # preventing stale NaN/data from corrupting attention or SSM computation.
    # ------【内存池/CuMem】new_block_ids_to_zero：本轮新分配的 block id，worker 使用前先清零显存，防陈旧 NaN/脏数据污染 attention/SSM ------
    new_block_ids_to_zero: list[int] | None = None

    # CoW copies to apply after zeroing new blocks and before forward.
    # ------【前缀缓存】kv_cache_block_copies：清零后、forward 前执行的 KV cache CoW 拷贝，用于前缀复用/块迁移 ------
    kv_cache_block_copies: list[KVCacheBlockCopy] | None = None

    # Producer partial-tail offload hand-off for external KV connectors:
    # {request_id: [(group_id, block_id, boundary_tokens), ...]} pointing at
    # the durable boundary block of a producer's last-prompt-boundary partial
    # tail (mamba "align" CoW target). None unless partial hash hits are active.
    # ------【前缀缓存】partial_tail_offloads：生产者最后 prompt 边界的部分尾块交接信息，指向 mamba "align" CoW 目标块，仅部分 hash 命中时非空 ------
    partial_tail_offloads: dict[str, list[tuple[int, int, int]]] | None = None

    # Dynamic speculative decoding: optimal K chosen by scheduler.
    # Number of spec tokens to schedule for the next step.
    # ------【投机解码】num_spec_tokens_to_schedule：动态投机解码下调度器选定的下一轮最优草稿 token 数 K ------
    num_spec_tokens_to_schedule: int = 0

    @classmethod
    def make_empty(cls) -> "SchedulerOutput":
        # ------【核心逻辑】构造空的 SchedulerOutput，作为无请求可调度时的默认返回值 ------
        return cls(
            scheduled_new_reqs=[],
            scheduled_cached_reqs=CachedRequestData.make_empty(),
            num_scheduled_tokens={},
            total_num_scheduled_tokens=0,
            scheduled_spec_decode_tokens={},
            scheduled_encoder_inputs={},
            num_common_prefix_blocks=[],
            finished_req_ids=set(),
            free_encoder_mm_hashes=[],
        )


# ------【结构化输出/grammar】GrammarOutput：Scheduler/Worker→grammar 模块的结构化输出约束位掩码消息，worker 据此做约束解码 ------
@dataclass
class GrammarOutput:
    # ------【结构化输出/grammar】structured_output_request_ids：需要结构化输出的请求 id 列表 ------
    # ids of structured output requests.
    structured_output_request_ids: list[str]
    # ------【结构化输出/grammar】grammar_bitmask：与上述 id 顺序对齐的 grammar 位掩码，标记各位置合法 token ------
    # Bitmask ordered as structured_output_request_ids.
    grammar_bitmask: "npt.NDArray[np.int32]"
