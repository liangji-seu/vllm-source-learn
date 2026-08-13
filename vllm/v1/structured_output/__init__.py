# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import itertools
import multiprocessing
from collections.abc import Iterable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.reasoning import ReasoningParserManager
from vllm.tokenizers import cached_tokenizer_from_config
from vllm.utils.import_utils import LazyLoader
from vllm.v1.structured_output.backend_guidance import GuidanceBackend
from vllm.v1.structured_output.backend_types import (
    StructuredOutputBackend,
    StructuredOutputGrammar,
)
from vllm.v1.structured_output.backend_xgrammar import XgrammarBackend

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt
    import torch

    from vllm.reasoning import ReasoningParser
    from vllm.v1.request import Request
else:
    torch = LazyLoader("torch", globals(), "torch")


logger = init_logger(__name__)


class StructuredOutputManager:
    """Engine-level manager for structured output requests."""

    def __init__(self, vllm_config: VllmConfig):
        # ------【核心逻辑】初始化 manager 的三个核心状态：后端、推理解析器类、vllm_config ------
        self.backend: StructuredOutputBackend | None = None
        # We only store the class of the reasoner in the manager.
        # The parser instance is request-scoped because some reasoning parsers
        # depend on per-request chat-template kwargs.
        self.reasoner_cls: type[ReasoningParser] | None = None
        self.vllm_config = vllm_config

        # ------【TP】external_launcher 每 TP rank 各有一个调度器，禁用异步编译保持确定性 ------
        # When in external_launcher mode, async grammar compilation causes deadlocks
        # due to external_launcher mode having a scheduler for each TP rank.
        # Async grammar compilation causes the
        # WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR → WAITING transition to
        # happen at different times on different TP ranks,
        # breaking the determinism assumption that external_launcher relies on.
        self._use_async_grammar_compilation = (
            vllm_config.parallel_config.distributed_executor_backend
            != "external_launcher"
        )

        # ------【结构化输出/grammar】bitmask 张量与全 1 掩码先置空/占位，按需再分配 ------
        self._grammar_bitmask: torch.Tensor | None = None
        self._full_mask = torch.tensor(-1, dtype=torch.int32)

        # ------【结构化输出/grammar】按 max_num_seqs 决定并行填 mask 阈值并建专用线程池 ------
        max_batch_size = self.vllm_config.scheduler_config.max_num_seqs
        self.fill_bitmask_parallel_threshold = 128
        if self.fill_bitmask_parallel_threshold < max_batch_size:
            self.fill_bitmask_parallel_batch_size = 16
            # Use:
            # - at least 1 CPU
            # - at most half the number of CPUs or 8, whichever is less
            max_workers = max(1, min(multiprocessing.cpu_count() // 2, 8))
            self.executor_for_fillmask = ThreadPoolExecutor(max_workers=max_workers)

        # ------【结构化输出/grammar】需要 tokenizer 时才建编译线程池并解析推理解析器 ------
        if not self.vllm_config.model_config.skip_tokenizer_init:
            # The default max_workers if not specified is the number of
            # CPUs * 5, which is way too high since these tasks are CPU-bound,
            # not I/O bound. We also know we would never dominate CPU usage
            # with just grammar compilation, so we set it to half the number
            # of CPUs.
            # ------【结构化输出/grammar】编译为 CPU 密集，worker 数取 CPU 一半避免过度并发 ------
            max_workers = max(1, (multiprocessing.cpu_count() + 1) // 2)
            self.executor = ThreadPoolExecutor(max_workers=max_workers)
            # ------【结构化输出/grammar】从配置缓存加载 tokenizer 供 grammar 编译使用 ------
            self.tokenizer = cached_tokenizer_from_config(
                model_config=self.vllm_config.model_config
            )
            # ------【结构化输出/grammar】可选加载推理解析器插件（动态导入）------
            reasoning_parser_plugin = (
                self.vllm_config.structured_outputs_config.reasoning_parser_plugin
            )
            if reasoning_parser_plugin and len(reasoning_parser_plugin) > 3:
                ReasoningParserManager.import_reasoning_parser(reasoning_parser_plugin)

            # ------【结构化输出/grammar】按配置名解析出推理解析器类，请求级实例化 ------
            reasoning_parser = (
                self.vllm_config.structured_outputs_config.reasoning_parser
            )
            if reasoning_parser:
                self.reasoner_cls = ReasoningParserManager.get_reasoning_parser(
                    reasoning_parser
                )

        # ------【结构化输出/grammar】是否在推理段也启用约束解码的开关 ------
        self.enable_in_reasoning = (
            self.vllm_config.structured_outputs_config.enable_in_reasoning
        )

    def _get_reasoner(self, request: "Request") -> "ReasoningParser | None":
        structured_req = request.structured_output_request
        # ------【结构化输出/grammar】无结构化请求或未配置解析器时直接返回 None ------
        if structured_req is None or self.reasoner_cls is None:
            return None

        # ------【结构化输出/grammar】惰性构建请求级解析器，复用与前端一致的模板 kwargs ------
        if structured_req.reasoner is None:
            # Lazily build the request-local parser so the structured-output
            # gate observes the same template kwargs used by the frontend.
            parser_kwargs = structured_req.reasoning_parser_kwargs or {}
            structured_req.reasoner = self.reasoner_cls(
                tokenizer=self.tokenizer,
                **parser_kwargs,
            )
        return structured_req.reasoner

    def grammar_init(self, request: "Request") -> None:
        # ------【结构化输出/grammar】非结构化请求直接跳过初始化 ------
        if request.structured_output_request is None:
            return

        # ------【核心逻辑】类型检查期校验 sampling_params 与 structured_outputs 非空 ------
        if TYPE_CHECKING:
            assert (
                request.sampling_params is not None
                and request.sampling_params.structured_outputs is not None
            )

        # ------【结构化输出/grammar】首次需要时惰性初始化后端（V1 仅支持单一后端）------
        # Initialize the backend the first time it is needed.
        #
        # NOTE: We only support a single backend. We do NOT support different
        # backends on a per-request basis in V1 (for now, anyway...).
        # _backend is set in Processor._validate_structured_output
        if self.backend is None:
            assert request.sampling_params is not None
            # ------【结构化输出/grammar】取出请求指定的后端名与词表大小供后端构造 ------
            backend = request.sampling_params.structured_outputs._backend
            vocab_size = self.vllm_config.model_config.get_vocab_size()
            # ------【结构化输出/grammar】按后端名分派：xgrammar 分支 ------
            if backend == "xgrammar":
                self.backend = XgrammarBackend(
                    self.vllm_config,
                    tokenizer=self.tokenizer,
                    vocab_size=vocab_size,
                )
            # ------【结构化输出/grammar】guidance 分支 ------
            elif backend == "guidance":
                self.backend = GuidanceBackend(
                    self.vllm_config,
                    tokenizer=self.tokenizer,
                    vocab_size=vocab_size,
                )
            # ------【结构化输出/grammar】outlines 分支（延迟导入避免无谓开销）------
            elif backend == "outlines":
                from vllm.v1.structured_output.backend_outlines import OutlinesBackend

                self.backend = OutlinesBackend(
                    self.vllm_config,
                    tokenizer=self.tokenizer,
                    vocab_size=vocab_size,
                )
            # ------【结构化输出/grammar】lm-format-enforcer 分支（延迟导入）------
            elif backend == "lm-format-enforcer":
                from vllm.v1.structured_output.backend_lm_format_enforcer import (  # noqa: E501
                    LMFormatEnforcerBackend,
                )

                self.backend = LMFormatEnforcerBackend(
                    self.vllm_config,
                    tokenizer=self.tokenizer,
                    vocab_size=vocab_size,
                )
            else:
                # ------【结构化输出/grammar】未知后端直接报错 ------
                raise ValueError(f"Unsupported structured output backend: {backend}")

        grammar: Future[StructuredOutputGrammar] | StructuredOutputGrammar
        # ------【异步 RPC】异步模式提交线程池，否则同步编译并把异常包装进 Future ------
        if self._use_async_grammar_compilation:
            grammar = self.executor.submit(self._create_grammar, request)
        else:
            try:
                grammar = self._create_grammar(request)
            except Exception as e:
                grammar = Future()
                grammar.set_exception(e)
        # ------【结构化输出/grammar】把编译结果（Future 或 grammar）挂到请求供调度器取用 ------
        request.structured_output_request.grammar = grammar

    def _create_grammar(self, request: "Request") -> StructuredOutputGrammar:
        struct_request = request.structured_output_request
        # ------【结构化输出/grammar】取出结构化请求并断言非空 ------
        assert struct_request is not None
        # Note that the request was validated in the engine core client,
        # so at this point we know it is a supported type of request. Grammar
        # compilation may still fail; the Future carries that error to the
        # scheduler so it can fail only this request.
        try:
            # ------【结构化输出/grammar】解包 key 并调用后端编译 grammar，异常由 Future 携带 ------
            request_type, grammar_spec = struct_request.structured_output_key
            assert self.backend is not None
            return self.backend.compile_grammar(request_type, grammar_spec)
        except Exception:
            # ------【结构化输出/grammar】编译失败打日志后重抛，仅使该请求失败 ------
            logger.exception(
                "Failed to compile grammar for request %s", request.request_id
            )
            raise

    def _fill_bitmasks(
        self, batch: Iterable[tuple[StructuredOutputGrammar, int, bool]]
    ) -> None:
        # ------【结构化输出/grammar】断言全局 bitmask 张量已分配 ------
        assert self._grammar_bitmask is not None
        # ------【结构化输出/grammar】逐条：约束则填充 bitmask，否则整行置全 1 ------
        for grammar, index, apply_bitmask in batch:
            if apply_bitmask and not grammar.is_terminated():
                grammar.fill_bitmask(self._grammar_bitmask, index)
            else:
                # Note that for thinking support, we will need to
                # reset the relevant part of the bitmask for consequent
                # requests here.
                self._grammar_bitmask[index].fill_(self._full_mask)

    def _async_submit_fill_bitmask(
        self, batch: list[tuple[StructuredOutputGrammar, int, bool]]
    ) -> Future:
        # ------【异步 RPC】把一批 bitmask 填充任务提交到专用线程池，返回 Future ------
        return self.executor_for_fillmask.submit(self._fill_bitmasks, batch)

    def grammar_bitmask(
        self,
        requests: dict[str, "Request"],
        structured_output_request_ids: list[str],
        scheduled_spec_decode_tokens: dict[str, list[int]],
    ) -> "npt.NDArray[np.int32] | None":
        # Prepare the structured output bitmask for this batch.
        # ------【结构化输出/grammar】无结构化请求时直接返回 None，跳过 bitmask 准备 ------
        if not structured_output_request_ids:
            return None

        # Covers both speculative decoding and diffusion LLMs (canvas_length).
        # ------【投机解码】取投机 token 数，同时覆盖 diffusion 的 canvas_length ------
        max_num_spec_tokens = self.vllm_config.num_speculative_tokens

        # ------【结构化输出/grammar】bitmask 未分配时按最大 batch 与投机位一次性分配 ------
        if self._grammar_bitmask is None:
            assert self.backend is not None
            max_batch_size = self.vllm_config.scheduler_config.max_num_seqs

            # Allocate a bitmask for each token needing to be checked:
            # one for each speculative position, and one more for the
            # bonus token / non-speculative token.
            # ------【结构化输出/grammar】为每个需校验的 token（含各投机位）预留一行掩码 ------
            self._grammar_bitmask = self.backend.allocate_token_bitmask(
                max_batch_size * (1 + max_num_spec_tokens)
            )

        # Generate a batched bitmask for all structured output requests.
        # When speculative decoding is enabled, we need to include multiple
        # masks for each request, one for each possible bonus token position.
        # These are stored inline in the tensor and unpacked by the gpu runner.
        # ------【结构化输出/grammar】用累计下标把每个请求/投机位的掩码内联写入张量 ------
        cumulative_index = 0

        # Optimized parallel filling of bitmasks for
        # non-spec, large-batch-size cases
        # ------【异步 RPC】大 batch 且非投机时走并行填掩码路径 ------
        if (
            len(structured_output_request_ids) > self.fill_bitmask_parallel_threshold
            and max_num_spec_tokens == 0
        ):
            # ------【异步 RPC】初始化 Future 列表与待提交批次缓冲 ------
            promises = []
            batch = []
            # ------【异步 RPC】遍历请求，收集 grammar/apply_bitmask 并按批提交 ------
            for req_id in structured_output_request_ids:
                request = requests[req_id]
                structured_output_request = request.structured_output_request
                if TYPE_CHECKING:
                    assert structured_output_request is not None
                grammar = structured_output_request.grammar
                if TYPE_CHECKING:
                    assert isinstance(grammar, StructuredOutputGrammar)

                apply_bitmask = self.should_fill_bitmask(request)
                batch.append((grammar, cumulative_index, apply_bitmask))
                # ------【异步 RPC】凑满一个并行批次即提交给线程池并清空缓冲 ------
                if len(batch) == self.fill_bitmask_parallel_batch_size:
                    promises.append(self._async_submit_fill_bitmask(batch))
                    batch = []

                cumulative_index += 1
            # ------【异步 RPC】提交尾部不足一个整批的剩余项 ------
            if batch:
                promises.append(self._async_submit_fill_bitmask(batch))

            # Wait for all bitmask filling tasks to complete.
            # ------【异步 RPC】阻塞等待所有并行填充任务完成后再继续 ------
            for promise in promises:
                promise.result()
        else:
            # Fallback to serial filling of bitmasks for small-batch-size cases
            # ------【结构化输出/grammar】小 batch 回退到串行填充路径 ------
            for req_id in structured_output_request_ids:
                request = requests[req_id]
                structured_output_request = request.structured_output_request

                if TYPE_CHECKING:
                    assert structured_output_request is not None
                grammar = structured_output_request.grammar
                if TYPE_CHECKING:
                    assert isinstance(grammar, StructuredOutputGrammar)
                apply_bitmask = self.should_fill_bitmask(request)

                # ------【结构化输出/grammar】检测推理结束点：无 bitmask 且未在推理段约束时生效 ------
                reasoner = self._get_reasoner(request)
                detect_reasoning_end = (
                    not apply_bitmask
                    and reasoner is not None
                    and not self.enable_in_reasoning
                )
                # ------【结构化输出/grammar】惰性构造的模拟 token 流与历史长度，用于流式判推理结束 ------
                simulated_buf: list[int] | None = None
                history_len = 0

                # ------【结构化输出/grammar】记录本窗口 grammar 推进次数与推理结束标志 ------
                state_advancements = 0
                post_reasoning_end_in_window = False
                req_tokens = scheduled_spec_decode_tokens.get(req_id, ())
                # ------【投机解码】逐投机 token 推进：填掩码、判推理结束、advance grammar ------
                for i, token in enumerate(req_tokens):
                    # ------【结构化输出/grammar】先为当前投机位填充掩码 ------
                    self._fill_bitmasks(((grammar, cumulative_index, apply_bitmask),))
                    advance_grammar = apply_bitmask
                    # ------【投机解码】填充位 token（-1）不参与约束也不推进 grammar ------
                    if token == -1:
                        apply_bitmask = False
                        advance_grammar = False
                    # ------【结构化输出/grammar】推理结束点落在窗口内时切换为约束并跳过该 token 推进 ------
                    elif (
                        detect_reasoning_end
                        and reasoner is not None
                        and not apply_bitmask
                    ):
                        if simulated_buf is None:
                            history = list(request.all_token_ids)
                            history_len = len(history)
                            simulated_buf = history + list(req_tokens)
                        simulated = simulated_buf[: history_len + i + 1]
                        if reasoner.is_reasoning_end_streaming(simulated, [token]):
                            # Reasoning ended mid-window. Constrain the rest
                            # of the window via bitmask. Skip grammar advance
                            # through the marker (it is reasoning content);
                            # try to advance through subsequent drafts so the
                            # next bitmask row reflects the post-advance state,
                            # but tolerate rejection since those drafts predate
                            # the bitmask and are not guaranteed valid.
                            apply_bitmask = True
                            advance_grammar = False
                            post_reasoning_end_in_window = True
                    # ------【结构化输出/grammar】用当前 token 尝试推进 grammar，被拒且非推理边界则报错 ------
                    if advance_grammar and not grammar.is_terminated():
                        accepted = grammar.accept_tokens(req_id, [token])
                        if accepted:
                            state_advancements += 1
                        elif not post_reasoning_end_in_window:
                            raise AssertionError(
                                (token, req_id, scheduled_spec_decode_tokens)
                            )
                    # ------【结构化输出/grammar】每处理一个投机位，掩码行下标前进一位 ------
                    cumulative_index += 1
                # Diffusion LLMs don't sample a bonus token after the
                # scheduled positions, so skip its bitmask in that case.
                # ------【结构化输出/grammar】非 diffusion 时还需填充 bonus token 位的掩码 ------
                if not (self.vllm_config.model_config.is_diffusion and req_tokens):
                    # bonus_apply must be True when the bonus-row position
                    # should be grammar-constrained. Two triggers:
                    # - should_fill_bitmask(request): reasoning was already
                    #   over at step start (or no reasoner /
                    #   enable_in_reasoning).
                    # - apply_bitmask: reasoning ended mid-window in this
                    #   call and was flipped True after the marker;
                    #   should_fill_bitmask still returns False here because
                    #   reasoning_ended is only persisted later by
                    #   should_advance.
                    # ------【结构化输出/grammar】bonus 位是否约束：步起始已结束或本窗口推理中途结束 ------
                    bonus_apply = self.should_fill_bitmask(request) or apply_bitmask
                    self._fill_bitmasks(((grammar, cumulative_index, bonus_apply),))
                    cumulative_index += 1
                # ------【结构化输出/grammar】回滚临时推进的 grammar 状态，保持跨步一致性 ------
                if state_advancements > 0:
                    grammar.rollback(state_advancements)

        # ------【结构化输出/grammar】按实际累计下标截取有效行，丢弃多余预留 ------
        bitmask_tensor = self._grammar_bitmask
        if cumulative_index < bitmask_tensor.shape[0]:
            bitmask_tensor = bitmask_tensor[:cumulative_index]

        # After finishing with the xgrammar operations, we convert to
        # np.ndarray, because that is much more efficient for serialization
        # and deserialization when sending this to the GPU workers.
        # ------【核心逻辑】转成 numpy 数组返回，利于向 GPU worker 序列化传输 ------
        return bitmask_tensor.numpy()

    def should_fill_bitmask(self, request: "Request") -> bool:
        # NOTE (Hanchen) if enable_in_reasoning is True, it means that
        # the model needs to be constrained in reasoning. So we should always
        # enable the bitmask filling.
        # ------【结构化输出/grammar】取请求级解析器判断是否需要按推理段门控约束 ------
        reasoner = self._get_reasoner(request)
        if reasoner is not None:
            # ------【结构化输出/grammar】enable_in_reasoning 时推理段也需约束，恒填充 ------
            if self.enable_in_reasoning:
                return True
            assert request.structured_output_request is not None
            # ------【结构化输出/grammar】首次访问时惰性计算推理结束标志并缓存 ------
            if request.structured_output_request.reasoning_ended is None:
                # This should be removed here, but since `openai_gptoss`
                # is an independent code path, it is kept for now.
                # After unifying the `openai_gptoss` and non-`openai_gptoss` styles,
                # it can be removed.
                request.structured_output_request.reasoning_ended = (
                    reasoner.is_reasoning_end(request.prompt_token_ids or [])
                )
            return request.structured_output_request.reasoning_ended
        # ------【结构化输出/grammar】无解析器（非思维链）默认需要约束 ------
        return True

    def should_advance(
        self,
        request: "Request",
        new_token_ids: list[int] | None = None,
    ) -> bool:
        # ------【结构化输出/grammar】非结构化请求不推进 FSM ------
        if not request.use_structured_output:
            return False

        # To determine whether we can advance the FSM.
        # Supports thinking usage where we skip the reasoning components.
        # ------【核心逻辑】类型检查期断言 grammar 已就绪 ------
        if TYPE_CHECKING:
            assert request.structured_output_request is not None
            assert request.structured_output_request.grammar is not None
        # by default, we should always advance
        # for cases that don't use thinking mode.
        # ------【结构化输出/grammar】无解析器（非思维链）默认始终推进 ------
        reasoner = self._get_reasoner(request)
        if reasoner is None:
            return True

        # if the model needs structured in reasoning, we should advance
        # ------【结构化输出/grammar】推理段也约束时直接推进 ------
        if self.enable_in_reasoning:
            return True

        structured_req = request.structured_output_request
        # ------【结构化输出/grammar】推理已结束则推进；否则继续判定本步是否结束 ------
        if structured_req.reasoning_ended:
            return True

        # Check if reasoning ends in *this* step.
        # When the caller passes new_token_ids (the tokens that were just
        # appended this step), use it directly as the delta window. The
        # placeholder-derived fallback assumes num_output_placeholders ==
        # len(new_token_ids), which breaks under async scheduling + spec
        # decode when some drafts are rejected (#43388): the placeholder
        # count remains > 0 after the step and the computed delta window
        # starts past the reasoning-end marker.
        # ------【结构化输出/grammar】计算本步增量 token 窗口：优先用显式传入的 new_token_ids ------
        all_token_ids = request.all_token_ids
        if new_token_ids:
            # The tokens were already appended this step, so the step window
            # starts exactly len(new_token_ids) from the end.
            # ------【结构化输出/grammar】有显式增量则从末尾回推窗口起点 ------
            start = len(all_token_ids) - len(new_token_ids)
            delta_ids: Iterable[int] = new_token_ids
        else:
            # ------【结构化输出/grammar】回退：用占位符推算增量起点（异步+投机下可能失准）------
            delta_from = request.num_computed_tokens - request.num_output_placeholders
            start = (
                delta_from
                if delta_from >= 0
                else max(len(all_token_ids) + delta_from, 0)
            )
            delta_ids = itertools.islice(all_token_ids, start, None)
        # ------【结构化输出/grammar】流式判定推理在本步结束，记录边界并允许推进 ------
        if reasoner.is_reasoning_end_streaming(all_token_ids, delta_ids):
            structured_req.reasoning_ended = True

            # Record the boundary so the scheduler can exclude reasoning tokens.
            # ------【结构化输出/grammar】定位推理结束 token 的绝对下标供调度器剔除推理段 ------
            end_index = self._find_reasoning_end_index(reasoner, all_token_ids, start)

            structured_req.reasoning_end_token_index = end_index
            return True

        # ------【结构化输出/grammar】推理尚未结束则不推进 grammar ------
        return False

    @staticmethod
    def _find_reasoning_end_index(
        reasoner: "ReasoningParser", all_token_ids: Sequence[int], start: int
    ) -> int:
        """Locates the last reasoning token within ``all_token_ids[start:]``.

        Returns:
            The absolute index of the token at which
            ``is_reasoning_end_streaming`` first fires. Falls back to the
            final index when no single token triggers the detection (e.g.
            a multi-token marker only recognized on the full delta), which
            conservatively treats the whole step as reasoning content.
        """
        # ------【结构化输出/grammar】前缀从窗口起点累积，逐 token 检测推理结束点 ------
        prefix = list(itertools.islice(all_token_ids, start))
        # ------【结构化输出/grammar】逐 token 追加并流式检测，命中即返回该下标 ------
        for idx in range(start, len(all_token_ids)):
            token = all_token_ids[idx]
            prefix.append(token)
            if reasoner.is_reasoning_end_streaming(prefix, [token]):
                return idx
        # ------【结构化输出/grammar】未命中时保守地把整步视为推理内容 ------
        return len(all_token_ids) - 1

    def trim_reasoning_for_advance(
        self, request: "Request", new_token_ids: list[int]
    ) -> list[int]:
        """Drops reasoning content from tokens about to advance the grammar.

        When reasoning ends mid-step (see should_advance), the step's output
        still contains reasoning tokens up to and including the end marker.
        Those are not grammar content: feeding them to accept_tokens makes
        the grammar reject the marker and kills the request (#44006).

        Returns:
            The suffix of ``new_token_ids`` that follows the reasoning-end
            marker. Steps fully after the boundary are returned unchanged.
        """
        structured_req = request.structured_output_request
        # ------【结构化输出/grammar】无结构化请求则原样返回 ------
        if structured_req is None:
            return new_token_ids
        end_idx = structured_req.reasoning_end_token_index
        # ------【结构化输出/grammar】无推理结束边界则无需裁剪 ------
        if end_idx is None:
            return new_token_ids
        first_idx = len(request.all_token_ids) - len(new_token_ids)
        # ------【结构化输出/grammar】计算边界之前落在本步内的推理 token 数量 ------
        num_reasoning = end_idx + 1 - first_idx
        if num_reasoning <= 0:
            return new_token_ids
        # ------【结构化输出/grammar】裁掉推理段，只把边界后的 token 交给 grammar 推进 ------
        return new_token_ids[num_reasoning:]

    def clear_backend(self) -> None:
        # ------【结构化输出/grammar】销毁后端释放其底层资源 ------
        if self.backend is not None:
            self.backend.destroy()
