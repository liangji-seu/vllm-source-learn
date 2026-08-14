# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
from collections import defaultdict, deque
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import torch

from vllm.lora.request import LoRARequest
from vllm.outputs import (
    STREAM_FINISHED,
    CompletionOutput,
    PoolingOutput,
    PoolingRequestOutput,
    RequestOutput,
)
from vllm.sampling_params import RequestOutputKind
from vllm.tokenizers import TokenizerLike
from vllm.tracing import (
    SpanAttributes,
    SpanKind,
    extract_trace_context,
    instrument_manual,
)
from vllm.utils import length_from_prompt_token_ids_or_embeds
from vllm.v1.engine import EngineCoreOutput, EngineCoreRequest, FinishReason
from vllm.v1.engine.detokenizer import IncrementalDetokenizer
from vllm.v1.engine.logprobs import LogprobsProcessor
from vllm.v1.engine.parallel_sampling import ParentRequest
from vllm.v1.metrics.stats import (
    IterationStats,
    LoRARequestStates,
    RequestStateStats,
    SchedulerStats,
)

# shared empty CPU tensor used as a placeholder pooling output
EMPTY_CPU_TENSOR = torch.empty(0, device="cpu")


# ------【异步 RPC】RequestOutputCollector：请求输出收集器，EngineCore 生产者→asyncio generate 消费任务的非阻塞交接缓冲 ------
class RequestOutputCollector:
    """
    Collects streamed RequestOutputs per individual request,
    for hand-off to the consuming asyncio generate task.

    When streaming deltas, RequestOutputs are merged if the
    producer gets ahead of the consumer.
    """

    def __init__(self, output_kind: RequestOutputKind, request_id: str):
        # ------【核心逻辑】aggregate：是否为 DELTA 增量模式，决定多次输出是否合并 ------
        self.aggregate = output_kind == RequestOutputKind.DELTA
        # ------【核心逻辑】request_id：内部请求 ID，标识该收集器所属请求 ------
        self.request_id = request_id
        # ------【异步 RPC】output：当前暂存输出，put/get 单缓冲交接 ------
        self.output: RequestOutput | PoolingRequestOutput | Exception | None = None
        # ------【异步 RPC】ready：asyncio 事件，get() 阻塞等待 put 到来 ------
        self.ready = asyncio.Event()

        # ------【异步 RPC】_input_stream_task：后台输入流任务句柄，close 时取消 ------
        self._input_stream_task: asyncio.Task | None = None

    # ------【异步 RPC】put：非阻塞写入输出；生产者领先时合并 DELTA 输出，异常直接覆盖 ------
    def put(self, output: RequestOutput | PoolingRequestOutput | Exception) -> None:
        """Non-blocking put operation."""
        if self.output is None or isinstance(output, Exception):
            self.output = output
            self.ready.set()
        elif isinstance(self.output, RequestOutput) and isinstance(
            output, RequestOutput
        ):
            # This ensures that request outputs with different request indexes
            # (if n > 1) do not override each other.
            self.output.add(output, aggregate=self.aggregate)
        elif isinstance(self.output, PoolingRequestOutput) and isinstance(
            output, PoolingRequestOutput
        ):
            self.output = output

    # ------【异步 RPC】get：阻塞等待直到有输出；异常输出在此抛出 ------
    async def get(self) -> RequestOutput | PoolingRequestOutput:
        """Get operation blocks on put event."""
        while (output := self.output) is None:
            await self.ready.wait()
        self.output = None
        self.ready.clear()
        if isinstance(output, Exception):
            raise output
        return output

    # ------【异步 RPC】get_nowait：非阻塞取走输出，无则返回 None ------
    def get_nowait(self) -> RequestOutput | PoolingRequestOutput | None:
        """Non-blocking get operation."""
        output = self.output
        if output is not None:
            self.output = None
            self.ready.clear()
        if isinstance(output, Exception):
            raise output
        return output

    # ------【异步 RPC】close：取消后台输入流任务，释放句柄 ------
    def close(self):
        if self._input_stream_task is not None:
            self._input_stream_task.cancel()
        self._input_stream_task = None

    # ------【异步 RPC】__del__：析构时线程安全地取消未完成的后台任务 ------
    def __del__(self):
        if (task := self._input_stream_task) is not None:
            task.get_loop().call_soon_threadsafe(task.cancel)
            self._input_stream_task = None


# ------【核心逻辑】OutputProcessorOutput：OutputProcessor 处理结果容器，EngineCore 输出→前端的批量打包 DTO ------
@dataclass
class OutputProcessorOutput:
    # ------【核心逻辑】request_outputs：本轮要返回前端的请求输出列表(无队列 LLMEngine 时) ------
    request_outputs: list[RequestOutput | PoolingRequestOutput]
    # ------【核心逻辑】reqs_to_abort：detokenizer 检出 stop 串但 EngineCore 未结束时需中止的请求 ID ------
    reqs_to_abort: list[str]


# ------【异步 RPC】StreamingUpdate：流式输入增量 DTO，承载子请求完成时追加到 RequestState 的新 prompt 片段 ------
@dataclass
class StreamingUpdate:
    """Streaming input update data for output processor.

    Contains the incremental prompt data to be applied to a request state
    when the current sub-request completes.
    """

    # ------【核心逻辑】prompt：本轮新增的 prompt 文本片段(可能为 None) ------
    prompt: str | None
    # ------【核心逻辑】prompt_token_ids：本轮新增的 prompt token id 片段 ------
    prompt_token_ids: list[int] | None
    # ------【核心逻辑】arrival_time：增量片段到达时间戳，用于更新请求统计 ------
    arrival_time: float
    # ------【核心逻辑】final：是否流式输入最终片段，置位后停止继续等待 ------
    final: bool = False


# ------【核心逻辑】RequestState：单请求在 OutputProcessor 侧的状态快照，聚合 detokenize/logprobs/LoRA/统计，输出到前端 ------
class RequestState:
    def __init__(
        self,
        request_id: str,
        external_req_id: str,
        parent_req: ParentRequest | None,
        request_index: int,
        lora_request: LoRARequest | None,
        output_kind: RequestOutputKind,
        prompt: str | None,
        prompt_token_ids: list[int] | None,
        prompt_embeds: torch.Tensor | None,
        logprobs_processor: LogprobsProcessor | None,
        detokenizer: IncrementalDetokenizer | None,
        max_tokens_param: int | None,
        arrival_time: float,
        queue: RequestOutputCollector | None,
        log_stats: bool,
        stream_interval: int,
        top_p: float | None = None,
        n: int | None = None,
        temperature: float | None = None,
        stream_input: bool = False,
    ):
        # ------【核心逻辑】request_id：内部随机生成的请求 ID，作为 request_states 字典键 ------
        self.request_id = request_id
        # ------【核心逻辑】external_req_id：用户侧/API 传入的外部请求 ID，用于对外输出 ------
        self.external_req_id = external_req_id
        # ------【投机解码】parent_req：并行采样(n>1)的父请求对象，聚合多个子请求输出 ------
        self.parent_req = parent_req
        # ------【核心逻辑】request_index：本请求在父请求 n 个采样中的索引(0..n-1) ------
        self.request_index = request_index
        # ------【LoRA】lora_request：本请求绑定的 LoRA 适配器请求对象 ------
        self.lora_request = lora_request
        # ------【LoRA】lora_name：LoRA 适配器名称，用于统计与完成后释放引用 ------
        self.lora_name = lora_request.lora_name if lora_request is not None else None
        # ------【核心逻辑】output_kind：输出模式(DELTA/FINAL_ONLY/累积)，决定增量还是全量返回 ------
        self.output_kind = output_kind
        # ------【核心逻辑】prompt：解码后的 prompt 文本，流式输入时逐段拼接 ------
        self.prompt = prompt
        # ------【核心逻辑】prompt_token_ids：prompt 的 token id 序列 ------
        self.prompt_token_ids = prompt_token_ids
        # ------【核心逻辑】prompt_embeds：多模态等场景直接提供的 prompt embedding ------
        self.prompt_embeds = prompt_embeds
        # ------【核心逻辑】prompt_len：prompt 有效长度(token 数或 embed 数)，用于统计 ------
        self.prompt_len = length_from_prompt_token_ids_or_embeds(
            self.prompt_token_ids, self.prompt_embeds
        )
        # ------【核心逻辑】logprobs_processor：对数概率处理，按需计算 sample/prompt logprobs ------
        self.logprobs_processor = logprobs_processor
        # ------【核心逻辑】detokenizer：增量解码器，把 token id 流转文本并做停止词检测 ------
        self.detokenizer = detokenizer
        # ------【核心逻辑】max_tokens_param：用户指定的最大生成 token 数上限 ------
        self.max_tokens_param = max_tokens_param
        # ------【核心逻辑】top_p：核采样参数，仅用于 tracing 上报 ------
        self.top_p = top_p
        # ------【投机解码】n：并行采样数量，>1 表示存在多个子请求 ------
        self.n = n
        # ------【核心逻辑】temperature：采样温度参数，仅用于 tracing 上报 ------
        self.temperature = temperature
        # ------【chunked prefill】is_prefilling：是否处于 prefill 阶段，用于统计 cached token ------
        self.is_prefilling = True
        # ------【异步 RPC】queue：AsyncLLM 模式下挂接的收集器，把输出投递到 generate() 任务 ------
        self.queue = queue
        # ------【前缀缓存】num_cached_tokens：命中前缀缓存的 token 数，对外报告 ------
        self.num_cached_tokens = 0
        # ------【前缀缓存】num_cache_creation_tokens：本次写入前缀缓存的 token 数 ------
        self.num_cache_creation_tokens = 0

        # ------【核心逻辑】stats：请求级统计快照(延迟/吞吐)，log_stats 关闭时为 None ------
        self.stats = RequestStateStats(arrival_time=arrival_time) if log_stats else None

        # Routed experts accumulation (prompt + sample chunks)
        # ------【EP/EPLB】routed_experts_chunks：各 chunk 命中的路由专家 id，结束时拼接上报 ------
        self.routed_experts_chunks: list[np.ndarray] = []

        # Stream Interval
        # ------【核心逻辑】stream_interval：流式输出间隔，每 N 个 token 返回一次 ------
        self.stream_interval = stream_interval
        # ------【核心逻辑】sent_tokens_offset：DELTA 模式下已发送 token 的偏移量 ------
        self.sent_tokens_offset = 0  # Offset of sent tokens

        # Streaming input queue
        # ------【异步 RPC】streaming_input：是否启用流式输入(resumable) ------
        self.streaming_input = stream_input
        # ------【异步 RPC】input_chunk_queue：待应用的流式输入增量队列，子请求完成时逐个出队 ------
        self.input_chunk_queue: deque[StreamingUpdate] | None = (
            deque() if stream_input else None
        )

    # ------【异步 RPC】apply_streaming_update：把流式输入增量应用到状态(拼接 prompt、更新长度与到达时间) ------
    def apply_streaming_update(self, update: StreamingUpdate) -> None:
        # Apply the update to the request state.
        self.streaming_input = not update.final
        # TODO also include relevant output tokens in new prompt here
        #     (match scheduler behavior).
        if update.prompt:
            self.prompt = (
                (self.prompt + update.prompt) if self.prompt else update.prompt
            )
        if self.prompt_token_ids:
            self.prompt_token_ids.extend(update.prompt_token_ids or ())
        else:
            self.prompt_token_ids = update.prompt_token_ids or []
        assert self.prompt_token_ids is not None
        self.prompt_len = len(self.prompt_token_ids)
        if self.stats is not None:
            self.stats.arrival_time = update.arrival_time
        self.is_prefilling = True

    # ------【核心逻辑】from_new_request：由 EngineCoreRequest 构造 RequestState(解析采样参数、建 detokenizer/logprobs) ------
    @classmethod
    def from_new_request(
        cls,
        tokenizer: TokenizerLike | None,
        request: EngineCoreRequest,
        prompt: str | None,
        parent_req: ParentRequest | None,
        request_index: int,
        queue: RequestOutputCollector | None,
        log_stats: bool,
        stream_interval: int,
    ) -> "RequestState":
        if sampling_params := request.sampling_params:
            if not sampling_params.detokenize:
                tokenizer = None
            output_kind = sampling_params.output_kind
            if sampling_params.stream_interval is not None:
                # clamp to the engine-level stream interval.
                stream_interval = max(sampling_params.stream_interval, stream_interval)
            logprobs_processor = LogprobsProcessor.from_new_request(
                tokenizer=tokenizer,
                request=request,
            )
            detokenizer = IncrementalDetokenizer.from_new_request(
                tokenizer=tokenizer,
                request=request,
            )
            max_tokens_param = sampling_params.max_tokens
            top_p = sampling_params.top_p
            n = sampling_params.n
            temperature = sampling_params.temperature
        else:
            logprobs_processor = None
            detokenizer = None
            max_tokens_param = None
            top_p = None
            n = None
            temperature = None
            assert request.pooling_params is not None
            output_kind = request.pooling_params.output_kind

        assert request.external_req_id is not None
        return cls(
            request_id=request.request_id,
            external_req_id=request.external_req_id,
            parent_req=parent_req,
            request_index=request_index,
            lora_request=request.lora_request,
            output_kind=output_kind,
            prompt=prompt,
            prompt_token_ids=request.prompt_token_ids,
            prompt_embeds=request.prompt_embeds,
            logprobs_processor=logprobs_processor,
            detokenizer=detokenizer,
            max_tokens_param=max_tokens_param,
            top_p=top_p,
            n=n,
            temperature=temperature,
            arrival_time=request.arrival_time,
            queue=queue,
            log_stats=log_stats,
            stream_interval=stream_interval,
            stream_input=request.resumable,
        )

    # ------【核心逻辑】make_request_output：组装 RequestOutput，含 stream_interval 节流与父请求聚合 ------
    def make_request_output(
        self,
        new_token_ids: list[int],
        pooling_output: torch.Tensor | None,
        finish_reason: FinishReason | None,
        stop_reason: int | str | None,
        kv_transfer_params: dict[str, Any] | None = None,
        ec_transfer_params: dict[str, Any] | None = None,
    ) -> RequestOutput | PoolingRequestOutput | None:
        finished = finish_reason is not None
        final_only = self.output_kind == RequestOutputKind.FINAL_ONLY

        if not finished and final_only:
            # Only the final output is required in FINAL_ONLY mode.
            return None

        if self.stream_interval > 1:
            assert self.detokenizer is not None

            # Send output request only when
            # 1. It has finished, or
            # 2. It is the first token, or
            # 3. It has reached the stream interval number of tokens
            if not (
                finished
                or self.sent_tokens_offset == 0
                or self.detokenizer.num_output_tokens() - self.sent_tokens_offset
                >= self.stream_interval
            ):
                return None

            if self.output_kind == RequestOutputKind.DELTA:
                # Send tokens from the offset in DELTA mode, otherwise all
                # tokens are sent.
                new_token_ids = self.detokenizer.output_token_ids[
                    self.sent_tokens_offset :
                ]
                self.sent_tokens_offset = self.detokenizer.num_output_tokens()

        external_req_id = self.external_req_id

        if pooling_output is not None:
            return self._new_request_output(
                external_req_id,
                [self._new_pooling_output(pooling_output)],
                finished,
            )

        output = self._new_completion_output(new_token_ids, finish_reason, stop_reason)

        if self.parent_req is None:
            outputs = [output]
        else:
            outputs, finished = self.parent_req.get_outputs(self.request_id, output)
            if not outputs:
                return None
            external_req_id = self.parent_req.external_req_id

        return self._new_request_output(
            external_req_id,
            outputs,
            finished,
            kv_transfer_params,
            ec_transfer_params,
        )

    # ------【核心逻辑】_new_request_output：构造最终 RequestOutput/PoolingRequestOutput，填入缓存统计与 metrics ------
    def _new_request_output(
        self,
        external_req_id: str,
        outputs: list[CompletionOutput] | list[PoolingOutput],
        finished: bool,
        kv_transfer_params: dict[str, Any] | None = None,
        ec_transfer_params: dict[str, Any] | None = None,
    ) -> RequestOutput | PoolingRequestOutput:
        # If prompt embeds were used, put placeholder prompt token ids
        prompt_token_ids = self.prompt_token_ids
        if prompt_token_ids is None and self.prompt_embeds is not None:
            prompt_token_ids = [0] * len(self.prompt_embeds)
        assert prompt_token_ids is not None

        first_output = outputs[0]
        if isinstance(first_output, PoolingOutput):
            assert len(outputs) == 1
            return PoolingRequestOutput(
                request_id=external_req_id,
                outputs=first_output,
                num_cached_tokens=self.num_cached_tokens,
                prompt_token_ids=prompt_token_ids,
                finished=finished,
            )
        assert self.logprobs_processor is not None
        if self.output_kind == RequestOutputKind.DELTA:
            # Side effect: logprobs processor forgets prompt logprobs
            prompt_logprobs = self.logprobs_processor.pop_prompt_logprobs()
        else:
            prompt_logprobs = self.logprobs_processor.prompt_logprobs

        return RequestOutput(
            request_id=external_req_id,  # request_id is what was provided externally
            lora_request=self.lora_request,
            prompt=self.prompt,
            prompt_token_ids=prompt_token_ids,
            prompt_logprobs=prompt_logprobs,
            outputs=cast(list[CompletionOutput], outputs),
            finished=finished,
            kv_transfer_params=kv_transfer_params,
            ec_transfer_params=ec_transfer_params,
            num_cached_tokens=self.num_cached_tokens,
            num_cache_creation_tokens=self.num_cache_creation_tokens,
            metrics=self.stats,
        )

    # ------【核心逻辑】_new_completion_output：构造 CompletionOutput，按 delta 模式裁剪文本/logprobs，结束时拼接路由专家 ------
    def _new_completion_output(
        self,
        token_ids: list[int],
        finish_reason: FinishReason | None,
        stop_reason: int | str | None,
    ) -> CompletionOutput:
        assert self.detokenizer is not None
        assert self.logprobs_processor is not None
        finished = finish_reason is not None
        delta = self.output_kind == RequestOutputKind.DELTA

        # Prepare text and token_ids, based on delta mode
        text = self.detokenizer.get_next_output_text(finished, delta)
        if not delta:
            token_ids = self.detokenizer.output_token_ids

        # Prepare logprobs, based on delta mode
        logprobs = self.logprobs_processor.logprobs
        if delta and logprobs:
            logprobs = logprobs[-len(token_ids) :]

        # Concatenate routed experts on finish
        routed_experts = None
        if finished and self.routed_experts_chunks:
            routed_experts = np.concatenate(self.routed_experts_chunks, axis=0)

        return CompletionOutput(
            index=self.request_index,
            text=text,
            token_ids=token_ids,
            routed_experts=routed_experts,
            logprobs=logprobs,
            cumulative_logprob=self.logprobs_processor.cumulative_logprob,
            finish_reason=str(finish_reason) if finished else None,
            stop_reason=stop_reason if finished else None,
        )

    # ------【核心逻辑】_new_pooling_output：把 pooling 张量包装成 PoolingOutput ------
    def _new_pooling_output(self, pooling_output: torch.Tensor) -> PoolingOutput:
        return PoolingOutput(data=pooling_output)


# ------【核心逻辑】OutputProcessor：输出编排器，把 EngineCoreOutputs 加工成 RequestOutputs 并分发/上报 ------
class OutputProcessor:
    """Process EngineCoreOutputs into RequestOutputs."""

    def __init__(
        self,
        tokenizer: TokenizerLike | None,
        *,
        log_stats: bool,
        stream_interval: int = 1,
        tracing_enabled: bool = False,
    ):
        # ------【核心逻辑】log_stats：是否记录请求级统计，关闭则 stats 为 None ------
        self.log_stats = log_stats
        # ------【核心逻辑】tokenizer：分词器，用于 detokenize/logprobs(可能为 None) ------
        self.tokenizer = tokenizer
        # ------【核心逻辑】stream_interval：引擎级流式输出间隔，请求级可再放大 ------
        self.stream_interval = stream_interval
        # ------【核心逻辑】request_states：内部请求 ID → 请求状态快照 的映射 ------
        self.request_states: dict[str, RequestState] = {}
        # ------【投机解码】parent_requests：父请求 ID → 父请求对象，聚合并行采样输出 ------
        self.parent_requests: dict[str, ParentRequest] = {}
        # ------【核心逻辑】external_req_ids：外部请求 ID → 内部请求 ID 列表，支持一对多(n>1)映射 ------
        self.external_req_ids: defaultdict[str, list[str]] = defaultdict(list)
        # ------【LoRA】lora_states：LoRA 活跃/峰值统计状态 ------
        self.lora_states = LoRARequestStates(log_stats)
        # ------【核心逻辑】tracing_enabled：是否开启 OpenTelemetry 追踪上报 ------
        self.tracing_enabled = tracing_enabled

    # ------【核心逻辑】get_num_unfinished_requests：返回未完成请求数(等于 request_states 长度) ------
    def get_num_unfinished_requests(self):
        return len(self.request_states)

    # ------【核心逻辑】has_unfinished_requests：是否还有未完成请求 ------
    def has_unfinished_requests(self) -> bool:
        return len(self.request_states) > 0

    # ------【异步 RPC】propagate_error：把异常投递到所有请求的 queue，唤醒阻塞中的 generate() 任务 ------
    def propagate_error(self, e: Exception):
        """Propagate error to all generate() tasks."""

        for _, state in self.request_states.items():
            assert state.queue is not None
            state.queue.put(e)

    # ------【核心逻辑】abort_requests：中止一批请求(支持外部/内部 ID、父子采样级联)，返回需 EngineCore 中止的内部 ID ------
    def abort_requests(self, request_ids: Iterable[str], internal: bool) -> list[str]:
        """Abort a list of requests.

        The request_ids may be either external request IDs (those passed to
        InputProcessor.process_inputs()) or internal request IDs (those randomly
        generated when creating the EngineCoreRequest).

        If an external request ID is provided, and that external request ID
        was used for multiple requests, all requests associated with that external
        request ID are aborted.

        In the case of parallel sampling, a request ID may be used to identify
        a parent request, in which case the associated child requests are aborted
        also.
        """
        internal_req_ids = []
        for request_id in request_ids:
            if internal:
                # Internal ID - this may be a parent request
                internal_req_ids.append(request_id)

                # Remove internal ID from the external->internal mapping
                if req_state := self.request_states.get(request_id):
                    external_req_id = req_state.external_req_id
                    internal_ids = self.external_req_ids[external_req_id]
                    internal_ids.remove(request_id)
                    if not internal_ids:
                        del self.external_req_ids[external_req_id]
            elif internal_ids := self.external_req_ids.pop(request_id, []):
                # External ID - abort all requests in the external->internal mapping
                internal_req_ids.extend(internal_ids)

        request_ids_to_abort = []
        for request_id in internal_req_ids:
            req_state = self.request_states.pop(request_id, None)
            if req_state is not None:
                self.lora_states.request_finished(request_id, req_state.lora_name)
                request_ids_to_abort.append(request_id)
                # Produce final abort output.
                if req_state.queue is not None and (
                    request_output := req_state.make_request_output(
                        new_token_ids=[],
                        # Set pooling_output is not None to
                        # correctly enter the abort pooling branch
                        pooling_output=EMPTY_CPU_TENSOR
                        if req_state.detokenizer is None
                        else None,
                        finish_reason=FinishReason.ABORT,
                        stop_reason=None,
                        kv_transfer_params=None,
                        ec_transfer_params=None,
                    )
                ):
                    req_state.queue.put(request_output)
            elif parent := self.parent_requests.get(request_id):
                # Abort children prior to removing the parent.
                if parent.child_requests:
                    child_reqs = list(parent.child_requests)
                    child_reqs = self.abort_requests(child_reqs, internal=True)
                    request_ids_to_abort.extend(child_reqs)
                self.parent_requests.pop(request_id, None)
        return request_ids_to_abort

    # ------【核心逻辑】add_request：注册新请求状态，重复 ID 走流式输入更新路径，维护外部→内部 ID 映射 ------
    def add_request(
        self,
        request: EngineCoreRequest,
        prompt: str | None,
        parent_req: ParentRequest | None = None,
        request_index: int = 0,
        queue: RequestOutputCollector | None = None,
    ) -> None:
        request_id = request.request_id
        req_state = self.request_states.get(request_id)
        if req_state is not None:
            self._update_streaming_request_state(req_state, request, prompt)
            return

        req_state = RequestState.from_new_request(
            tokenizer=self.tokenizer,
            request=request,
            prompt=prompt,
            parent_req=parent_req,
            request_index=request_index,
            queue=queue,
            log_stats=self.log_stats,
            stream_interval=self.stream_interval,
        )
        self.request_states[request_id] = req_state
        if parent_req:
            self.parent_requests[parent_req.request_id] = parent_req

        # Track the external_req_id -> [internal_req_id, ...] mapping
        self.external_req_ids[req_state.external_req_id].append(request_id)

    # ------【异步 RPC】_update_streaming_request_state：队列化流式输入增量，最终片段标记 final 并清理 ------
    def _update_streaming_request_state(
        self, req_state: RequestState, request: EngineCoreRequest, prompt: str | None
    ) -> None:
        """Queue a streaming update instead of immediately applying it."""
        if not request.resumable:
            # Final request - just mark completion, don't add its dummy tokens.
            if req_state.input_chunk_queue is None:
                # Engine already finished - emit final output and clean up.
                self._finish_request(req_state)
                if req_state.queue is not None:
                    # Emit a final output with finished=True
                    # to unblock the generate() loop.
                    req_state.queue.put(STREAM_FINISHED)
            elif req_state.input_chunk_queue:
                req_state.input_chunk_queue[-1].final = True
            else:
                req_state.streaming_input = False
            return

        update = StreamingUpdate(
            prompt=prompt,
            prompt_token_ids=request.prompt_token_ids,
            arrival_time=request.arrival_time,
        )

        # Apply request updates now if the last input already completed.
        if req_state.input_chunk_queue is None:
            req_state.apply_streaming_update(update)
            req_state.input_chunk_queue = deque()
        else:
            # Queue the streaming update otherwise.
            req_state.input_chunk_queue.append(update)

    # ------【核心逻辑】process_outputs：唯一遍历 EngineCoreOutputs 的入口，统计→detokenize→组包→入队/返回 ------
    def process_outputs(
        self,
        engine_core_outputs: list[EngineCoreOutput],
        engine_core_timestamp: float | None = None,
        iteration_stats: IterationStats | None = None,
    ) -> OutputProcessorOutput:
        """
        Process the EngineCoreOutputs:
        1) Compute stats for logging
        2) Detokenize
        3) Create and handle RequestOutput objects:
            * If there is a queue (for usage with AsyncLLM),
              put the RequestOutput objects into the queue for
              handling by the per-request generate() tasks.

            * If there is no queue (for usage with LLMEngine),
              return a list of RequestOutput objects.

        NOTE FOR DEVELOPERS

        vLLM V1 minimizes the number of python loops over the full
        batch to ensure system overheads are minimized. This is the
        only function that should loop over EngineCoreOutputs.

        If you need to touch every element of the batch, do it from
        within the loop below.
        """

        request_outputs: list[RequestOutput | PoolingRequestOutput] = []
        reqs_to_abort: list[str] = []
        for engine_core_output in engine_core_outputs:
            req_id = engine_core_output.request_id
            req_state = self.request_states.get(req_id)
            if req_state is None:
                # Ignore output for already-aborted request.
                continue

            # 1) Compute stats for this iteration.
            self._update_stats_from_output(
                req_state, engine_core_output, engine_core_timestamp, iteration_stats
            )

            new_token_ids = engine_core_output.new_token_ids
            pooling_output = engine_core_output.pooling_output
            finish_reason = engine_core_output.finish_reason
            stop_reason = engine_core_output.stop_reason
            kv_transfer_params = engine_core_output.kv_transfer_params
            ec_transfer_params = engine_core_output.ec_transfer_params
            if engine_core_output.routed_experts is not None:
                req_state.routed_experts_chunks.append(
                    engine_core_output.routed_experts
                )

            if req_state.is_prefilling:
                if engine_core_output.prefill_stats is not None:
                    req_state.num_cached_tokens = (
                        engine_core_output.prefill_stats.num_cached_tokens
                    )
                    req_state.num_cache_creation_tokens = (
                        engine_core_output.prefill_stats.num_cache_creation_tokens
                    )
                req_state.is_prefilling = False

            if pooling_output is None:
                assert req_state.detokenizer is not None
                assert req_state.logprobs_processor is not None
                # 2) Detokenize the token ids into text and perform stop checks.
                stop_string = req_state.detokenizer.update(
                    new_token_ids, finish_reason == FinishReason.STOP
                )
                if stop_string:
                    finish_reason = FinishReason.STOP
                    stop_reason = stop_string

                # 3) Compute sample and prompt logprobs for request,
                # if required.
                req_state.logprobs_processor.update_from_output(engine_core_output)

            # 4) Create and handle RequestOutput objects.
            if request_output := req_state.make_request_output(
                new_token_ids,
                pooling_output,
                finish_reason,
                stop_reason,
                kv_transfer_params,
                ec_transfer_params,
            ):
                if req_state.streaming_input:
                    request_output.finished = False

                if req_state.queue is not None:
                    # AsyncLLM: put into queue for handling by generate().
                    req_state.queue.put(request_output)
                else:
                    # LLMEngine: return list of RequestOutputs.
                    request_outputs.append(request_output)

            # Free completed requests.
            if finish_reason is not None:
                if req_state.streaming_input:
                    if req_state.input_chunk_queue:
                        update = req_state.input_chunk_queue.popleft()
                        req_state.apply_streaming_update(update)
                    else:
                        req_state.input_chunk_queue = None
                else:
                    self._finish_request(req_state)
                    if not engine_core_output.finished:
                        # If req not finished in EngineCore, but Detokenizer
                        # detected stop string, abort needed in EngineCore.
                        reqs_to_abort.append(req_id)

                    # Track per-request stats
                    self._update_stats_from_finished(
                        req_state, finish_reason, iteration_stats
                    )
                    if self.tracing_enabled:
                        self.do_tracing(engine_core_output, req_state, iteration_stats)

        return OutputProcessorOutput(
            request_outputs=request_outputs,
            reqs_to_abort=reqs_to_abort,
        )

    # ------【核心逻辑】_finish_request：从状态表移除完成请求，清理外部 ID 映射与父请求 ------
    def _finish_request(self, req_state: RequestState) -> None:
        req_id = req_state.request_id
        self.request_states.pop(req_id)

        internal_ids = self.external_req_ids[req_state.external_req_id]
        internal_ids.remove(req_id)
        if not internal_ids:
            del self.external_req_ids[req_state.external_req_id]

        # Remove parent request if applicable.
        parent_req = req_state.parent_req
        if parent_req and not parent_req.child_requests:
            self.parent_requests.pop(parent_req.request_id, None)

    # ------【LoRA】update_scheduler_stats：把调度器统计喂给 LoRA 状态，更新活跃/峰值计数 ------
    def update_scheduler_stats(self, scheduler_stats: SchedulerStats | None):
        self.lora_states.update_scheduler_stats(scheduler_stats)

    # ------【核心逻辑】do_tracing：构造延迟/用量属性，上报单请求 OpenTelemetry span ------
    def do_tracing(
        self,
        engine_core_output: EngineCoreOutput,
        req_state: RequestState,
        iteration_stats: IterationStats | None,
    ) -> None:
        assert req_state.stats is not None
        assert iteration_stats is not None

        metrics = req_state.stats
        arrival_time_ns = int(metrics.arrival_time * 1e9)
        trace_context = extract_trace_context(engine_core_output.trace_headers)
        prompt_length = length_from_prompt_token_ids_or_embeds(
            req_state.prompt_token_ids, req_state.prompt_embeds
        )

        # Calculate timing metrics
        e2e_time = iteration_stats.iteration_timestamp - metrics.arrival_time
        queued_time = metrics.scheduled_ts - metrics.queued_ts
        prefill_time = metrics.first_token_ts - metrics.scheduled_ts
        decode_time = metrics.last_token_ts - metrics.first_token_ts
        inference_time = metrics.last_token_ts - metrics.scheduled_ts

        # Build attributes dict
        attributes: dict[str, Any] = {
            SpanAttributes.GEN_AI_LATENCY_TIME_TO_FIRST_TOKEN: (
                metrics.first_token_latency
            ),
            SpanAttributes.GEN_AI_LATENCY_E2E: e2e_time,
            SpanAttributes.GEN_AI_LATENCY_TIME_IN_QUEUE: queued_time,
            SpanAttributes.GEN_AI_USAGE_PROMPT_TOKENS: prompt_length,
            SpanAttributes.GEN_AI_USAGE_COMPLETION_TOKENS: (
                metrics.num_generation_tokens
            ),
            SpanAttributes.GEN_AI_LATENCY_TIME_IN_MODEL_PREFILL: prefill_time,
            SpanAttributes.GEN_AI_LATENCY_TIME_IN_MODEL_DECODE: decode_time,
            SpanAttributes.GEN_AI_LATENCY_TIME_IN_MODEL_INFERENCE: inference_time,
            SpanAttributes.GEN_AI_REQUEST_ID: req_state.external_req_id,
        }

        # Add optional request parameters
        if req_state.top_p:
            attributes[SpanAttributes.GEN_AI_REQUEST_TOP_P] = req_state.top_p
        if req_state.max_tokens_param:
            attributes[SpanAttributes.GEN_AI_REQUEST_MAX_TOKENS] = (
                req_state.max_tokens_param
            )
        if req_state.temperature:
            attributes[SpanAttributes.GEN_AI_REQUEST_TEMPERATURE] = (
                req_state.temperature
            )
        if req_state.n:
            attributes[SpanAttributes.GEN_AI_REQUEST_N] = req_state.n

        instrument_manual(
            span_name="llm_request",
            start_time=arrival_time_ns,
            attributes=attributes,
            context=trace_context,
            kind=SpanKind.SERVER,
        )

    # ------【核心逻辑】_update_stats_from_output：把 EngineCore 输出喂给迭代统计(含 LoRA 状态) ------
    def _update_stats_from_output(
        self,
        req_state: RequestState,
        engine_core_output: EngineCoreOutput,
        engine_core_timestamp: float | None,
        iteration_stats: IterationStats | None,
    ):
        if iteration_stats is None:
            return

        assert engine_core_timestamp is not None
        assert req_state.stats is not None
        iteration_stats.update_from_output(
            engine_core_output,
            engine_core_timestamp,
            req_state.is_prefilling,
            req_state.stats,
            self.lora_states,
            req_state.lora_name,
        )

    # ------【核心逻辑】_update_stats_from_finished：请求结束时汇总最终统计并释放 LoRA 引用 ------
    def _update_stats_from_finished(
        self,
        req_state: RequestState,
        finish_reason: FinishReason | None,
        iteration_stats: IterationStats | None,
    ):
        if iteration_stats is None:
            return

        assert finish_reason is not None
        assert req_state.stats is not None
        iteration_stats.update_from_finished_request(
            finish_reason=finish_reason,
            request_id=req_state.external_req_id,
            num_prompt_tokens=req_state.prompt_len,
            max_tokens_param=req_state.max_tokens_param,
            req_stats=req_state.stats,
            num_cached_tokens=req_state.num_cached_tokens,
        )
        self.lora_states.request_finished(req_state.request_id, req_state.lora_name)

        ParentRequest.observe_finished_request(
            req_state.parent_req, iteration_stats, req_state.stats.num_generation_tokens
        )
