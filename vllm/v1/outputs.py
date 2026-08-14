# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import ABC, abstractmethod
from collections.abc import Sequence
from copy import copy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, NamedTuple, TypeAlias

import numpy as np
import torch

from vllm.compilation.cuda_graph import CUDAGraphStat
from vllm.v1.core.sched.output import SchedulerOutput

if TYPE_CHECKING:
    from vllm.distributed.kv_events import KVConnectorKVEvents
    from vllm.distributed.kv_transfer.kv_connector.v1.base import (
        KVConnectorWorkerMetadata,
    )
    from vllm.distributed.kv_transfer.kv_connector.v1.metrics import KVConnectorStats
else:
    KVConnectorStats = object
    KVConnectorWorkerMetadata = object
    KVConnectorKVEvents = object


# ------【投机解码/核心逻辑】LogprobsLists：Worker→Scheduler 的 logprobs 载体（CPU/numpy 形式），随 ModelRunnerOutput 序列化回传，供调度器/前端组装最终 output；cu_num_generated_tokens 支持投机解码下各请求生成 token 数不同 ------
class LogprobsLists(NamedTuple):
    # ------【核心逻辑】logprob_token_ids：每个生成位置 top-k 候选 token id ------
    # [num_reqs x num_generated_tokens, max_num_logprobs + 1]
    logprob_token_ids: np.ndarray
    # ------【核心逻辑】logprobs：上述候选 token 对应的对数概率 ------
    # [num_reqs x num_generated_tokens, max_num_logprobs + 1]
    logprobs: np.ndarray
    # ------【核心逻辑】sampled_token_ranks：实际采样 token 在候选列表中的排名 ------
    # [num_reqs x num_generated_tokens]
    sampled_token_ranks: np.ndarray
    # ------【投机解码】cu_num_generated_tokens：各请求累计生成 token 数的前缀和，投机解码下每个请求生成数不同，据此切片 ------
    # [num_reqs]
    # Used for slicing the logprobs in cases like speculative
    # decoding where the number of generated tokens may be
    # different for each request.
    cu_num_generated_tokens: list[int] | None = None

    # ------【投机解码】slice_request：按请求索引+位置数切出单请求的 logprobs 子块 ------
    def slice_request(self, req_idx: int, num_positions: int):
        if self.cu_num_generated_tokens is not None:
            req_idx = self.cu_num_generated_tokens[req_idx]
        end_idx = req_idx + num_positions
        return LogprobsLists(
            self.logprob_token_ids[req_idx:end_idx],
            self.logprobs[req_idx:end_idx],
            self.sampled_token_ranks[req_idx:end_idx],
            None,
        )


# ------【投机解码/核心逻辑】LogprobsTensors：Worker 内部 GPU 侧的 logprobs 载体（torch.Tensor 形式），经 tolists/to_cpu_nonblocking 下放到 CPU 再序列化回传给调度器 ------
class LogprobsTensors(NamedTuple):
    # ------【核心逻辑】logprob_token_ids：每个生成位置 top-k 候选 token id（GPU 张量） ------
    # [num_reqs x num_generated_tokens, max_num_logprobs + 1]
    logprob_token_ids: torch.Tensor
    # ------【核心逻辑】logprobs：上述候选 token 对应的对数概率（GPU 张量） ------
    # [num_reqs x num_generated_tokens, max_num_logprobs + 1]
    logprobs: torch.Tensor
    # ------【核心逻辑】selected_token_ranks：实际采样 token 在候选列表中的排名（GPU 张量） ------
    # [num_reqs x num_generated_tokens]
    selected_token_ranks: torch.Tensor
    # ------【投机解码】cu_num_generated_tokens：各请求累计生成 token 数的前缀和，投机解码下每个请求生成数不同，据此切片 ------
    # [num_reqs]
    cu_num_generated_tokens: list[int] | None = None

    # ------【异步 RPC】tolists：D2H 同步下放到 CPU 并转 numpy，供序列化回传给调度器 ------
    def tolists(self, cu_num_generated_tokens: list[int] | None = None):
        return LogprobsLists(
            self.logprob_token_ids.cpu().numpy(),
            self.logprobs.cpu().numpy(),
            self.selected_token_ranks.cpu().numpy(),
            cu_num_generated_tokens
            if cu_num_generated_tokens is not None
            else self.cu_num_generated_tokens,
        )

    # ------【异步 RPC】to_cpu_nonblocking：非阻塞异步 D2H，与默认流计算重叠，降低回传拷贝开销 ------
    def to_cpu_nonblocking(self) -> "LogprobsTensors":
        if self.logprob_token_ids.device.type == "cpu":
            return self
        return LogprobsTensors(
            self.logprob_token_ids.to("cpu", non_blocking=True),
            self.logprobs.to("cpu", non_blocking=True),
            self.selected_token_ranks.to("cpu", non_blocking=True),
            self.cu_num_generated_tokens,
        )

    # ------【CUDA Graph/核心逻辑】filter：按 bool mask 过滤出子集，用于 CUDA Graph 回放或部分请求选取 ------
    def filter(self, mask: torch.Tensor) -> "LogprobsTensors":
        """Filter the logprobs tensors with the given bool mask."""
        assert self.cu_num_generated_tokens is None, (
            "filter can't be used with cu_num_generated_tokens"
        )
        return LogprobsTensors(
            self.logprob_token_ids[mask],
            self.logprobs[mask],
            self.selected_token_ranks[mask],
        )

    # ------【DP/TP】cat：把多份扁平化 logprobs 张量拼接（如数据并行/张量并行多路结果合并） ------
    @staticmethod
    def cat(
        tensors: Sequence["LogprobsTensors"],
        cu_num_generated_tokens: list[int] | None = None,
    ) -> "LogprobsTensors":
        """Concatenate flattened logprob tensors."""
        assert tensors
        assert cu_num_generated_tokens is not None or all(
            tensor.cu_num_generated_tokens is None for tensor in tensors
        )
        if len(tensors) == 1:
            tensor = tensors[0]
            if cu_num_generated_tokens is None:
                return tensor
            return tensor._replace(cu_num_generated_tokens=cu_num_generated_tokens)
        return LogprobsTensors(
            logprob_token_ids=torch.cat(
                [tensor.logprob_token_ids for tensor in tensors]
            ),
            logprobs=torch.cat([tensor.logprobs for tensor in tensors]),
            selected_token_ranks=torch.cat(
                [tensor.selected_token_ranks for tensor in tensors]
            ),
            cu_num_generated_tokens=cu_num_generated_tokens,
        )

    # ------【核心逻辑】empty_cpu：构造 CPU 空占位张量，用于无 logprobs 场景的占位输出 ------
    @staticmethod
    def empty_cpu(
        num_positions: int, num_tokens_per_position: int
    ) -> "LogprobsTensors":
        """Create empty LogprobsTensors on CPU."""

        logprob_token_ids = torch.empty(
            (num_positions, num_tokens_per_position), dtype=torch.int32, device="cpu"
        )
        logprobs = torch.empty_like(logprob_token_ids, dtype=torch.float32)
        selected_token_ranks = torch.empty(
            num_positions, dtype=torch.int32, device="cpu"
        )
        return LogprobsTensors(
            logprob_token_ids=logprob_token_ids,
            logprobs=logprobs,
            selected_token_ranks=selected_token_ranks,
        )


# ------【EP/EPLB+异步 RPC】RoutedExpertsTensors：Worker 侧路由专家数据的设备端快照，异步 D2H 到 CPU 后随 ModelRunnerOutput 回传给调度器做专家负载均衡统计 ------
class RoutedExpertsTensors(NamedTuple):
    """Device-side snapshot of routed experts data, pending async D2H.

    Produced by :class:`GPUModelRunner` at the end of each async-scheduled
    step. The copy stream waits on the default stream, then issues
    non-blocking D2H via :meth:`to_cpu_nonblocking` into a pinned CPU
    buffer; :class:`AsyncGPUModelRunnerOutput.get_output` synchronizes
    the copy before the scheduler reads it.

    Sliced to ``total_num_scheduled_tokens`` (step-level, across all
    requests — NOT per-request). Both ``routing_data`` and
    ``slot_mapping`` must be private clones when sourced from shared
    capturer / prepare-input buffers, so the next forward pass /
    ``_prepare_inputs`` on the default stream does not race with a
    D2H still pending on the copy stream.
    """

    # ------【EP/EPLB】routing_data：每个 token 在每层命中的专家 id 矩阵（供负载均衡统计） ------
    # (num_scheduled_tokens, num_layers, num_experts_per_tok)
    routing_data: torch.Tensor
    # ------【EP/EPLB】slot_mapping：每行 routing_data 对应的物理 KV cache 槽位 ------
    # (num_scheduled_tokens,)
    slot_mapping: torch.Tensor

    def to_cpu_nonblocking(self) -> "RoutedExpertsTensors":
        """Issue non-blocking D2H on the current stream.

        NOTE: ``non_blocking=True`` only delivers true overlap when the
        CPU target is pinned. The current fallback here allocates a
        new pageable CPU tensor per call, which silently degrades to a
        synchronous copy; acceptable because the sync happens on the
        dedicated copy stream, not the default stream.
        """
        if self.routing_data.device.type == "cpu":
            return self
        return RoutedExpertsTensors(
            self.routing_data.to("cpu", non_blocking=True),
            self.slot_mapping.to("cpu", non_blocking=True),
        )

    def tolists(self) -> "RoutedExpertsLists":
        """Convert to the numpy-backed form consumed by the scheduler.

        ``.cpu()`` is a no-op when the tensor is already on CPU, so this
        is cheap for the post-D2H case; for raw device tensors it will
        synchronously block, which is only reached in tests.
        """
        return RoutedExpertsLists(
            self.routing_data.cpu().numpy(),
            self.slot_mapping.cpu().numpy(),
        )


# ------【EP/EPLB】RoutedExpertsLists：Worker→Scheduler 的 CPU/numpy 形式路由专家数据，供调度器按 slot_mapping 落库做专家负载均衡 ------
class RoutedExpertsLists(NamedTuple):
    """CPU-side routed experts, the form :meth:`RoutedExpertsManager.store_batch`
    consumes.

    Batched per scheduler step: the leading dim is the number of tokens
    scheduled across all requests in this step (``total_num_scheduled_tokens``),
    not per-request tokens. ``slot_mapping[i]`` tells the scheduler which
    physical KV-cache slot row ``i`` of ``routing_data`` belongs to.
    """

    # (num_scheduled_tokens, num_layers, num_experts_per_tok)
    routing_data: np.ndarray
    # (num_scheduled_tokens,)
    slot_mapping: np.ndarray


# ------【核心逻辑】PoolerOutput：池化层（embedding/rerank 模型）产出的每请求向量，形状随所用 pooler 而异，随 ModelRunnerOutput 回传 ------
# [num_reqs, <dynamic>]
# The shape of each element depends on the pooler used
PoolerOutput: TypeAlias = torch.Tensor | list[torch.Tensor] | list[torch.Tensor | None]


# ------【投机解码/核心逻辑】SamplerOutput：Worker 内部（采样器→模型 runner）的 GPU 侧采样结果，函数调用/内存直传，非跨进程消息；变长生成统一 padding 到 max_num_generated_tokens ------
@dataclass
class SamplerOutput:
    # ------【核心逻辑】sampled_token_ids：每请求采样到的 token id 矩阵，不足处用 PLACEHOLDER_TOKEN_ID 填充 ------
    # [num_reqs, max_num_generated_tokens]
    # Different requests can have different number of generated tokens.
    # All requests are padded to max_num_generated_tokens.
    # PLACEHOLDER_TOKEN_ID (-1 by default) is used for padding.
    sampled_token_ids: torch.Tensor
    # ------【核心逻辑】logprobs_tensors：配套的 GPU 侧 logprobs，None 表示未请求 ------
    logprobs_tensors: LogprobsTensors | None


# ------【PD 分离+异步 RPC】KVConnectorOutput：Worker→Scheduler 的 KV 连接器回传（PD 分离下 KV 跨实例传输的完成通知/统计），随 ModelRunnerOutput 走 MessageQueue 回传，驱动 KVOutputAggregator ------
@dataclass
class KVConnectorOutput:
    # ------【PD 分离】finished_sending：已完成 KV 发送（可被远端消费）的请求 id 集合 ------
    # [req_ids]
    finished_sending: set[str] | None = None
    # ------【PD 分离】finished_recving：已完成 KV 接收（本地已就绪）的请求 id 集合 ------
    finished_recving: set[str] | None = None
    # ------【PD 分离】kv_connector_stats：KV 传输统计指标（耗时/字节等） ------
    kv_connector_stats: KVConnectorStats | None = None
    # ------【PD 分离】kv_cache_events：KV 缓存事件（发布/订阅跨实例通知） ------
    kv_cache_events: KVConnectorKVEvents | None = None
    # ------【PD 分离】kv_connector_worker_meta：worker 侧连接器元数据（含拓扑/能力） ------
    kv_connector_worker_meta: KVConnectorWorkerMetadata | None = None
    # ------【PD 分离】invalid_block_ids：外部计算 KV 块加载失败的 block id，引用它的请求需重算 ------
    # IDs of externally computed KV blocks that failed to load.
    # Requests referencing these blocks should be rescheduled to recompute them
    invalid_block_ids: set[int] = field(default_factory=set)
    # ------【PD 分离】expected_finished_count：每请求期望收到的发送/接收完成通知数（Nixl 等握手型连接器用，默认 0 表示不变） ------
    # Configuration describing how many finished sending/receiving
    # notifications should be expected for each request. This allows
    # handshake-based connectors like Nixl to update the KVOutputAggregator.
    # It captures a static setup info and should almost always remain constant
    # for a given connector after discovery. Default value entails no change.
    expected_finished_count: int = 0

    # ------【PD 分离】is_empty：判断无任何 KV 连接器回传，用于决定是否返回共享的空输出 ------
    def is_empty(self):
        return (
            not self.finished_sending
            and not self.finished_recving
            and not self.kv_connector_stats
            and not self.kv_cache_events
            and not self.invalid_block_ids
            and not self.kv_connector_worker_meta
        )


# ------【核心逻辑/异步 RPC】ECConnectorOutput：Worker→Scheduler 的 EC（弹性编码器缓存）连接器回传，按多模态输入 hash（mm_hash）标记已发送/已接收的编码器缓存 ------
@dataclass
class ECConnectorOutput:
    # ------【核心逻辑】finished_sending：已完成编码器缓存发送的 mm_hash 集合 ------
    # [mm_hash]
    finished_sending: set[str] | None = None
    # ------【核心逻辑】finished_recving：已完成编码器缓存接收的 mm_hash 集合 ------
    finished_recving: set[str] | None = None


# ------【异步 RPC/DP+TP】ModelRunnerOutput：Worker→Scheduler 核心回传消息（引擎主数据流），序列化后经 MessageQueue/共享内存发给调度进程；张量字段尽量用 list/numpy 以避免序列化开销 ------
# ModelRunnerOutput is serialized and sent to the scheduler process.
# This is expensive for torch.Tensor so prefer to use list instead.
@dataclass
class ModelRunnerOutput:
    # ------【核心逻辑】req_ids：本次 step 参与计算的所有请求 id ------
    # [num_reqs]
    req_ids: list[str]
    # ------【核心逻辑】req_id_to_index：req_id → 行索引，便于按请求定位采样/池化结果 ------
    # req_id -> index
    req_id_to_index: dict[str, int]

    # ------【投机解码/核心逻辑】sampled_token_ids：每请求本次 step 生成的 token id 序列，投机/跳步解码下各请求长度可不同 ------
    # num_reqs x num_generated_tokens
    # num_generated_tokens is the number of tokens
    # generated in the current step. It can be different for
    # each request due to speculative/jump decoding.
    sampled_token_ids: list[list[int]] = field(default_factory=list)

    # ------【核心逻辑】logprobs：每请求生成阶段的 logprobs（CPU/numpy 形式，含候选 token id、概率、采样排名） ------
    # [num_reqs, max_num_logprobs + 1]
    # [num_reqs, max_num_logprobs + 1]
    # [num_reqs]
    logprobs: LogprobsLists | None = None

    # ------【核心逻辑】prompt_logprobs_dict：预填充（prompt）阶段的 logprobs，按 req_id 索引（GPU 张量形式，前端请求时才填充） ------
    # req_id -> (token_ids, logprobs, ranks)
    # [prompt_len, num_prompt_logprobs]
    # [prompt_len, num_prompt_logprobs]
    # [prompt_len]
    prompt_logprobs_dict: dict[str, LogprobsTensors | None] = field(
        default_factory=dict
    )

    # ------【核心逻辑】pooler_output：每请求的池化向量（embedding/rerank 模型），None 占位表示尚未产出 ------
    # [num_reqs, hidden_size]
    pooler_output: list[torch.Tensor | None] | None = None

    # ------【PD 分离】kv_connector_output：KV 连接器的回传（跨实例 KV 传输完成通知），None 表示未启用 ------
    kv_connector_output: KVConnectorOutput | None = None

    # ------【核心逻辑】ec_connector_output：EC 编码器缓存连接器回传，None 表示未启用 ------
    ec_connector_output: ECConnectorOutput | None = None

    # ------【核心逻辑】num_nans_in_logits：每请求 logits 中的 NaN 数量（健康监控/日志），req_id → 计数 ------
    # req_id -> num_nans_in_logits
    num_nans_in_logits: dict[str, int] | None = None

    # ------【CUDA Graph】cudagraph_stats：本次 step CUDA Graph 执行统计（命中/回放/padding 等），供可观测性 ------
    # information related to cudagraph execution
    cudagraph_stats: CUDAGraphStat | None = None

    # ------【EP/EPLB】routed_experts：worker 捕获的每 step 路由专家数据（含 routing_data 与 slot_mapping），供调度器落库做专家负载均衡 ------
    # Per-step routed experts data captured by the worker.
    # ``routing_data`` shape: (num_scheduled_tokens, num_layers,
    #                         num_experts_per_tok); expert IDs as uint8/uint16.
    # ``slot_mapping`` shape: (num_scheduled_tokens,); physical KV-cache
    #                         slot for each row of routing_data.
    # ``num_scheduled_tokens`` is step-level (total across all requests
    # in this step), not per-request. The scheduler persists this into
    # its slot buffer via ``slot_buffer[slot_mapping] = routing_data``.
    # ``None`` when ``enable_return_routed_experts`` is off.
    routed_experts: RoutedExpertsLists | None = None

    # ------【PD 分离】with_kv_conn_output_only：仅携带 KV 连接器输出的轻量回传，避免为纯 KV 通知构造完整输出 ------
    @staticmethod
    def with_kv_conn_output_only(
        kv_connector_output: KVConnectorOutput | None,
    ) -> "ModelRunnerOutput":
        """Return ModelRunnerOutput containing the provided KVConnectorOutput,
        otherwise empty. Returns None if kv_connector_output is passed as None.
        """
        if kv_connector_output is None or kv_connector_output.is_empty():
            return EMPTY_MODEL_RUNNER_OUTPUT
        output = copy(EMPTY_MODEL_RUNNER_OUTPUT)
        output.kv_connector_output = kv_connector_output
        return output


# ------【异步 RPC】AsyncModelRunnerOutput：异步调度下 Worker→Scheduler 输出的包装抽象，get_output 阻塞等待结果（可能含 D2H 拷贝）后返回 ModelRunnerOutput ------
# ModelRunnerOutput wrapper for async scheduling.
class AsyncModelRunnerOutput(ABC):
    # ------【异步 RPC】get_output：阻塞取回就绪的 ModelRunnerOutput（单次调用），内部可能等待设备→主机拷贝完成 ------
    @abstractmethod
    def get_output(self) -> ModelRunnerOutput:
        """Get the ModelRunnerOutput for this async output.

        This is a blocking call that waits until the results are ready, which
        might involve copying device tensors to the host.
        This method should only be called once per AsyncModelRunnerOutput.
        """
        pass


# ------【投机解码】DraftTokenIds：Worker→Scheduler 的草稿 token 回传（executor.take_draft_token_ids 经 collective_rpc 聚合），投喂给调度器更新草稿状态 ------
@dataclass
class DraftTokenIds:
    # ------【投机解码】req_ids：参与草稿生成的请求 id ------
    # [num_reqs]
    req_ids: list[str]
    # ------【投机解码】draft_token_ids：每请求草稿模型提议的候选 token 序列 ------
    # num_reqs x num_draft_tokens
    draft_token_ids: list[list[int]]


# ------【核心逻辑】make_empty_encoder_model_runner_output：为纯 encoder 实例构造无生成数据的占位回传（仅 req 记账，无采样 token），调度器据此结束预填充 ------
def make_empty_encoder_model_runner_output(
    scheduler_output: "SchedulerOutput",
) -> ModelRunnerOutput:
    """
    Create a ModelRunnerOutput stub that contains the correct
    per-request bookkeeping but no generated data yet.
    """
    if not scheduler_output.num_scheduled_tokens:
        return EMPTY_MODEL_RUNNER_OUTPUT

    # Convert to list so we get a deterministic, indexable sequence
    req_ids: list[str] = list(scheduler_output.num_scheduled_tokens.keys())

    # Give every request its own contiguous index
    req_id_to_index: dict[str, int] = {rid: idx for idx, rid in enumerate(req_ids)}

    # An encoder instance never samples, so it emits no tokens at all. The
    # scheduler finishes these requests once their prompt is fully encoded
    # (see `Scheduler.update_from_output`).
    sampled_token_ids: list[list[int]] = [[] for _ in req_ids]

    # Pooler outputs are not available yet ⇒ use None placeholders
    pooler_output: list[torch.Tensor | None] = [None for _ in req_ids]

    return ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index=req_id_to_index,
        sampled_token_ids=sampled_token_ids,
        pooler_output=pooler_output,
    )


EMPTY_MODEL_RUNNER_OUTPUT = ModelRunnerOutput(req_ids=[], req_id_to_index={})
