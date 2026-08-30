# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import functools
import gc
import itertools
import threading
import time
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import AbstractContextManager, contextmanager, nullcontext
from copy import copy, deepcopy
from dataclasses import dataclass, replace
from functools import reduce
from typing import TYPE_CHECKING, Any, NamedTuple, TypeAlias, cast

import numpy as np
import torch
import torch.distributed
import torch.nn as nn
from tqdm import tqdm

import vllm.envs as envs
from vllm.compilation.breakable_cudagraph import (
    BreakableCUDAGraphWrapper,
    is_breakable_cudagraph_enabled,
)
from vllm.compilation.counter import compilation_counter
from vllm.compilation.cuda_graph import CUDAGraphStat, CUDAGraphWrapper
from vllm.compilation.monitor import set_cudagraph_capturing_enabled
from vllm.config import (
    CompilationMode,
    CUDAGraphMode,
    VllmConfig,
    get_layers_from_vllm_config,
    set_current_vllm_config,
    update_config,
)
from vllm.config.cache import CacheConfig
from vllm.config.ec_manager_config import EncoderCacheManagerMetadata
from vllm.config.model import PROCESSED_LOGPROBS_MODES
from vllm.distributed.ec_transfer import get_ec_transfer, has_ec_transfer
from vllm.distributed.eplb.eplb_state import EplbState
from vllm.distributed.kv_transfer import get_kv_transfer_group, has_kv_transfer_group
from vllm.distributed.kv_transfer.kv_connector.utils import copy_kv_blocks
from vllm.distributed.parallel_state import (
    GraphCaptureContext,
    get_dcp_group,
    get_pp_group,
    get_tp_group,
    graph_capture,
    is_global_first_rank,
)
from vllm.forward_context import (
    BatchDescriptor,
    set_forward_context,
)
from vllm.logger import init_logger
from vllm.lora.layers import BaseLayerWithLoRA, LoRAMapping, LoRAMappingType
from vllm.model_executor.layers.attention import Attention, MLAAttention
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.fused_moe.all2all_utils import get_ep_all2all_manager
from vllm.model_executor.layers.fused_moe.routed_experts_capturer import (
    RoutedExpertsCapturer,
    bind_routed_experts_capturer,
)
from vllm.model_executor.layers.mamba.ops.ssu_dispatch import (
    initialize_mamba_ssu_backend,
)
from vllm.model_executor.layers.rotary_embedding import (
    MRotaryEmbedding,
    XDRotaryEmbedding,
)
from vllm.model_executor.model_loader import get_model_loader
from vllm.model_executor.model_loader.reload import (
    finalize_layerwise_reload,
    initialize_layerwise_reload,
)
from vllm.model_executor.models.interfaces import (
    MixtureOfExperts,
    MultiModalEmbeddings,
    SupportsMRoPE,
    SupportsMultiModal,
    SupportsXDRoPE,
    is_mixture_of_experts,
    supports_eagle3,
    supports_mrope,
    supports_multimodal_pruning,
    supports_realtime,
    supports_transcription,
    supports_xdrope,
)
from vllm.model_executor.models.interfaces_base import (
    VllmModelForPooling,
    is_pooling_model,
    is_text_generation_model,
)
from vllm.model_executor.offloader import (
    create_offloader,
    get_offloader,
    set_offloader,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.encoder_budget import MultiModalBudget
from vllm.multimodal.inputs import (
    BatchedTensorInputs,
    MultiModalKwargsItem,
    PlaceholderRange,
)
from vllm.multimodal.utils import (
    copy_mm_embedding_modality,
    get_mm_features_in_window,
    group_and_batch_mm_kwargs,
    set_mm_embedding_modality,
)
from vllm.platforms import current_platform
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import SamplingType
from vllm.sequence import IntermediateTensors
from vllm.tasks import GenerationTask, PoolingTask, SupportedTask
from vllm.tracing import instrument
from vllm.utils import length_from_prompt_token_ids_or_embeds
from vllm.utils.math_utils import cdiv, round_up
from vllm.utils.mem_utils import DeviceMemoryProfiler, format_gib
from vllm.utils.nvtx_pytorch_hooks import PytHooks
from vllm.utils.platform_utils import num_compute_units
from vllm.utils.torch_utils import (
    PIN_MEMORY,
    async_tensor_h2d,
    current_stream,
    is_quantized_kv_cache,
    kv_cache_dtype_str_to_dtype,
)
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadata,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
)
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadataBuilder
from vllm.v1.attention.backends.linear_attn import (
    BailingLinearAttentionMetadataBuilder,
)
from vllm.v1.attention.backends.mamba2_attn import Mamba2AttentionMetadataBuilder
from vllm.v1.attention.backends.utils import (
    NULL_BLOCK_ID,
    create_fast_prefill_custom_backend,
    get_dcp_local_seq_lens,
    reorder_batch_to_split_decodes_and_prefills,
)
from vllm.v1.core.sched.output import NewRequestData
from vllm.v1.cudagraph_dispatcher import CudagraphDispatcher
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    ChunkedLocalAttentionSpec,
    CrossAttentionSpec,
    EncoderOnlyAttentionSpec,
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheSpec,
    KVCacheSpecKind,
    KVQuantMode,
    MambaSpec,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
    get_kv_cache_spec_kind,
)
from vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry
from vllm.v1.outputs import (
    EMPTY_MODEL_RUNNER_OUTPUT,
    AsyncModelRunnerOutput,
    DraftTokenIds,
    ECConnectorOutput,
    KVConnectorOutput,
    LogprobsLists,
    LogprobsTensors,
    ModelRunnerOutput,
    PoolerOutput,
    RoutedExpertsLists,
    RoutedExpertsTensors,
    SamplerOutput,
    make_empty_encoder_model_runner_output,
)
from vllm.v1.pool.late_interaction_runner import LateInteractionRunner
from vllm.v1.pool.metadata import PoolingMetadata, PoolingStates
from vllm.v1.sample.logits_processor import LogitsProcessors, build_logitsprocs
from vllm.v1.sample.logits_processor.interface import LogitsProcessor
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.rejection_sampler import RejectionSampler
from vllm.v1.sample.sampler import Sampler
from vllm.v1.spec_decode.custom_class_proposer import create_custom_proposer
from vllm.v1.spec_decode.dflash import DFlashProposer
from vllm.v1.spec_decode.draft_model import DraftModelProposer
from vllm.v1.spec_decode.eagle import EagleProposer
from vllm.v1.spec_decode.extract_hidden_states import ExtractHiddenStatesProposer
from vllm.v1.spec_decode.gemma4 import Gemma4Proposer
from vllm.v1.spec_decode.medusa import MedusaProposer
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
from vllm.v1.spec_decode.ngram_proposer_gpu import (
    NgramProposerGPU,
    copy_num_valid_draft_tokens,
    update_ngram_gpu_tensors_incremental,
    update_scheduler_for_invalid_drafts,
)
from vllm.v1.spec_decode.step3p5 import Step3p5MTPProposer
from vllm.v1.spec_decode.suffix_decoding import SuffixDecodingProposer
from vllm.v1.spec_decode.utils import update_num_computed_tokens_for_batch_change
from vllm.v1.structured_output.utils import apply_grammar_bitmask
from vllm.v1.utils import CpuGpuBuffer, record_function_or_nullcontext
from vllm.v1.worker import mamba_utils
from vllm.v1.worker.block_table import SlotMappingMode
from vllm.v1.worker.cp_utils import (
    check_attention_cp_compatibility,
    get_dcp_dummy_context_len,
    prepare_dcp_dummy_context_metadata,
)
from vllm.v1.worker.dp_utils import coordinate_batch_across_dp
from vllm.v1.worker.ec_connector_model_runner_mixin import ECConnectorModelRunnerMixin
from vllm.v1.worker.gpu.attn_utils import _reshape_attention_kv_cache
from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch
from vllm.v1.worker.gpu_ubatch_wrapper import UBatchWrapper
from vllm.v1.worker.kv_connector_model_runner_mixin import KVConnectorModelRunnerMixin
from vllm.v1.worker.lora_model_runner_mixin import LoRAModelRunnerMixin
from vllm.v1.worker.ubatch_utils import (
    UBatchSlices,
    check_ubatch_thresholds,
    maybe_create_ubatch_slices,
    split_attn_metadata,
)
from vllm.v1.worker.utils import is_residual_scattered_for_sp, raise_if_nan_logits
from vllm.v1.worker.workspace import lock_workspace

from .utils import (
    AttentionGroup,
    KVBlockZeroer,
    add_kv_sharing_layers_to_kv_cache_groups,
    bind_kv_cache,
    copy_kv_cache_blocks_inplace,
    prepare_kernel_block_sizes,
    sanity_check_mm_encoder_outputs,
)

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
    from vllm.v1.spec_decode.ngram_proposer import NgramProposer
    from vllm.v1.worker.encoder_cudagraph import EncoderCudaGraphManager

logger = init_logger(__name__)


def _get_parameter_for_reload(model: nn.Module, name: str) -> nn.Parameter:
    """Resolve checkpoint names without changing the model's module tree."""
    # ------【权重传输】按最后一个点拆出模块名与参数名，定位层粒度 reload 要更新的参数 ------
    module_name, _, parameter_name = name.rpartition(".")
    module = model.get_submodule(module_name)
    # ------【LoRA+权重传输】LoRA 包装层需解包到 base_layer，才能拿到真实权重参数 ------
    if isinstance(module, BaseLayerWithLoRA):
        module = module.base_layer
    return module.get_parameter(parameter_name)


AttnMetadataDict: TypeAlias = dict[str, AttentionMetadata]
# list when ubatching is enabled
PerLayerAttnMetadata: TypeAlias = list[AttnMetadataDict] | AttnMetadataDict


# Wrapper for ModelRunnerOutput to support overlapped execution.
class AsyncGPUModelRunnerOutput(AsyncModelRunnerOutput):
    def __init__(
        self,
        model_runner_output: ModelRunnerOutput,
        sampled_token_ids: torch.Tensor,
        logprobs_tensors: LogprobsTensors | None,
        invalid_req_indices: list[int],
        async_output_copy_stream: torch.cuda.Stream,
        vocab_size: int,
        routed_experts: RoutedExpertsTensors | None = None,
        check_ep_fault: bool = False,
    ):
        self._model_runner_output = model_runner_output
        self._invalid_req_indices = invalid_req_indices

        # ------【异步 RPC】用 blocking(sleep) 事件同步异步拷贝，避免忙轮询 CUDA driver lock ------
        # Event on the copy stream so we can synchronize the non-blocking copy.
        # Blocking (sleep) event to avoid busy-polling the CUDA driver lock.
        self.async_copy_ready_event = torch.cuda.Event(blocking=True)

        # ------【内存池/CuMem】持有 device 张量引用，防止拷贝完成前被提前释放 ------
        # Keep a reference to the device tensor to avoid it being
        # deallocated until we finish copying it to the host.
        self._sampled_token_ids = sampled_token_ids
        self.vocab_size = vocab_size
        self._logprobs_tensors = logprobs_tensors
        self._routed_experts = routed_experts
        self._has_fault: torch.Tensor | None = None

        # ------【异步 RPC】在独立 stream 上发起非阻塞 D2H 拷贝，不等待其完成 ------
        # Initiate the copy on a separate stream, but do not synchronize it.
        default_stream = torch.cuda.current_stream()
        with torch.cuda.stream(async_output_copy_stream):
            async_output_copy_stream.wait_stream(default_stream)
            self.sampled_token_ids_cpu = self._sampled_token_ids.to(
                "cpu", non_blocking=True
            )
            self._logprobs_tensors_cpu = (
                self._logprobs_tensors.to_cpu_nonblocking()
                if self._logprobs_tensors
                else None
            )
            self._routed_experts_cpu = (
                self._routed_experts.to_cpu_nonblocking()
                if self._routed_experts is not None
                else None
            )
            # ------【EP/EPLB】异步查询 EP all2all 是否发生 rank 超时故障，结果一并拷回 CPU ------
            if check_ep_fault:
                has_fault = get_ep_all2all_manager().query_fault()
                self._has_fault = has_fault.to("cpu", non_blocking=True)
            self.async_copy_ready_event.record()

    def get_output(self) -> ModelRunnerOutput:
        """Copy the device tensors to the host and return a ModelRunnerOutput.

        This function blocks until the copy is finished.
        """
        max_gen_len = self.sampled_token_ids_cpu.shape[-1]
        # ------【异步 RPC】阻塞等待异步拷贝事件完成，保证读取 CPU 侧结果安全 ------
        self.async_copy_ready_event.synchronize()

        # ------【内存池/CuMem】拷贝完成后立即释放 device 张量，降低显存峰值 ------
        # Release the device tensors once the copy has completed.
        del self._logprobs_tensors
        del self._sampled_token_ids
        # ------【投机解码】单 token 时逐条清理无效请求，多候选时走拒绝采样解析 ------
        if max_gen_len == 1:
            valid_sampled_token_ids = self.sampled_token_ids_cpu.tolist()
            for i in self._invalid_req_indices:
                valid_sampled_token_ids[i].clear()
            logprobs_lists = None
            if self._logprobs_tensors_cpu is not None:
                logprobs_lists = self._logprobs_tensors_cpu.tolists()
        else:
            valid_sampled_token_ids, logprobs_lists = RejectionSampler.parse_output(
                self.sampled_token_ids_cpu,
                self.vocab_size,
                self._invalid_req_indices,
                logprobs_tensors=self._logprobs_tensors_cpu,
            )

        output = self._model_runner_output
        # ------【核心逻辑】把解析出的采样 token 与 logprobs 写回最终输出对象 ------
        output.sampled_token_ids = valid_sampled_token_ids
        output.logprobs = logprobs_lists

        # ------【EP/EPLB+异步 RPC】把已拷回 CPU 的路由专家结果转成 list 写回输出 ------
        if self._routed_experts_cpu is not None:
            output.routed_experts = self._routed_experts_cpu.tolists()
        del self._routed_experts

        # ------【EP/EPLB】EP all2all 检测到 rank 超时故障时抛错并上报活跃掩码 ------
        if self._has_fault is not None and self._has_fault.item():
            mask = get_ep_all2all_manager().query_active_mask()
            raise RuntimeError(
                "Fault detected in EP all2all communication: "
                "one or more ranks timed out during dispatch/combine. "
                f"Mask: {mask.cpu().tolist()}"
            )

        return output


def _copy_pooler_output_to_cpu(
    raw_pooler_output: PoolerOutput, finished_mask: list[bool]
) -> list[torch.Tensor | None]:
    num_reqs = len(finished_mask)

    # ------【核心逻辑】raw_pooler_output 为单个 tensor 时，按完成掩码分三类拷贝 ------
    if isinstance(raw_pooler_output, torch.Tensor):
        if raw_pooler_output.shape[0] != num_reqs:
            raise ValueError(
                "Pooler output batch size does not match finished mask size: "
                f"{raw_pooler_output.shape[0]} != {num_reqs}."
            )

        num_finished = sum(finished_mask)
        # ------【核心逻辑】无完成/全完成两条快捷路径，避免逐条索引开销 ------
        if num_finished == 0:
            return [None] * num_reqs
        if num_finished == num_reqs:
            return list(raw_pooler_output.to("cpu", non_blocking=True))

        # partial finished
        # ------【异步 RPC】部分完成时只 index_select 出完成行做非阻塞拷贝 ------
        finished_indices = [i for i, include in enumerate(finished_mask) if include]
        index_tensor = torch.tensor(
            finished_indices, device=raw_pooler_output.device, dtype=torch.long
        )
        finished_outputs = raw_pooler_output.index_select(0, index_tensor).to(
            "cpu", non_blocking=True
        )
        partial_pooler_output: list[torch.Tensor | None] = [None] * num_reqs
        for i, out in zip(finished_indices, finished_outputs):
            partial_pooler_output[i] = out
        return partial_pooler_output

    # ------【核心逻辑】list 形式时逐条按完成掩码做非阻塞拷贝 ------
    assert isinstance(raw_pooler_output, list)
    if len(raw_pooler_output) != num_reqs:
        raise ValueError(
            "Pooler output batch size does not match finished mask size: "
            f"{len(raw_pooler_output)} != {num_reqs}."
        )

    pooler_output: list[torch.Tensor | None] = [None] * num_reqs
    for i, (out, include) in enumerate(zip(raw_pooler_output, finished_mask)):
        if include and out is not None:
            pooler_output[i] = out.to("cpu", non_blocking=True)
    return pooler_output


class AsyncGPUPoolingModelRunnerOutput(AsyncModelRunnerOutput):
    def __init__(
        self,
        model_runner_output: ModelRunnerOutput,
        raw_pooler_output: PoolerOutput,
        finished_mask: list[bool],
        async_output_copy_stream: torch.cuda.Stream,
    ):
        self._model_runner_output = model_runner_output

        # ------【异步 RPC】blocking(sleep) 事件同步异步拷贝，避免忙轮询 CUDA driver lock ------
        # Event on the copy stream so we can synchronize the non-blocking copy.
        # Blocking (sleep) event to avoid busy-polling the CUDA driver lock.
        self.async_copy_ready_event = torch.cuda.Event(blocking=True)

        # ------【内存池/CuMem】持有 device 张量引用，防止拷贝完成前被提前释放 ------
        # Keep a reference to the device tensors to avoid them being
        # deallocated until we finish copying it to the host.
        self._raw_pooler_output = raw_pooler_output

        # ------【异步 RPC】在独立 stream 上发起非阻塞 pooler 输出拷贝，不等待完成 ------
        # Initiate the copy on a separate stream, but do not synchronize it.
        default_stream = torch.cuda.current_stream()
        with torch.cuda.stream(async_output_copy_stream):
            async_output_copy_stream.wait_stream(default_stream)
            self._model_runner_output.pooler_output = _copy_pooler_output_to_cpu(
                raw_pooler_output=self._raw_pooler_output,
                finished_mask=finished_mask,
            )
            self.async_copy_ready_event.record()

    def get_output(self) -> ModelRunnerOutput:
        """Copy the device tensors to the host and return a ModelRunnerOutput.
        This function blocks until the copy is finished.
        """
        # ------【异步 RPC】阻塞等待异步拷贝完成，保证读取 CPU 侧结果安全 ------
        self.async_copy_ready_event.synchronize()

        # ------【内存池/CuMem】拷贝完成后立即释放 device 张量，降低显存峰值 ------
        # Release the device tensors once the copy has completed.
        del self._raw_pooler_output
        return self._model_runner_output


class ExecuteModelState(NamedTuple):
    """Ephemeral cached state transferred between execute_model() and
    sample_tokens(), after execute_model() returns None."""

    scheduler_output: "SchedulerOutput"
    logits: torch.Tensor
    spec_decode_metadata: SpecDecodeMetadata | None
    spec_decode_common_attn_metadata: CommonAttentionMetadata | None
    hidden_states: torch.Tensor
    sample_hidden_states: torch.Tensor
    aux_hidden_states: list[torch.Tensor] | None
    ec_connector_output: ECConnectorOutput | None
    cudagraph_stats: CUDAGraphStat | None
    slot_mappings: dict[str, torch.Tensor] | list[dict[str, torch.Tensor]] | None












# 这个是modelrunner v1, 我们先看这个
class GPUModelRunner(
    LoRAModelRunnerMixin, KVConnectorModelRunnerMixin, ECConnectorModelRunnerMixin
):
    '''
    这个GPUModelRunner就是专门负责跑模型运行的了，也就是我们kuipa的demo,比如构建采样器啊，各个层算子的实例这些
    '''
    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        ################################################
        # 1. 配置与元信息
        ################################################
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.offload_config = vllm_config.offload_config
        self.compilation_config = vllm_config.compilation_config
        self.lora_config = vllm_config.lora_config
        self.load_config = vllm_config.load_config
        self.parallel_config = vllm_config.parallel_config
        self.scheduler_config = vllm_config.scheduler_config
        self.speculative_config = vllm_config.speculative_config
        self.observability_config = vllm_config.observability_config

        # ------【核心逻辑】建立局部别名，减少后续冗长的 self 前缀访问 ------
        model_config = self.model_config
        cache_config = self.cache_config
        scheduler_config = self.scheduler_config
        parallel_config = self.parallel_config
        self.device = device
        self.dtype = self.model_config.dtype

        # ------【EP/EPLB+DP】MoE 且开启数据并行时，判断 EP all2all 是否支持容错检测 ------
        self.check_ep_fault = False
        if parallel_config.data_parallel_size > 1 and self.model_config.is_moe:
            self.check_ep_fault = get_ep_all2all_manager().support_fault_tolerance





        # ------【内存池/CuMem】把 KV cache 的字符串 dtype 转成 torch dtype ------
        self.kv_cache_dtype = kv_cache_dtype_str_to_dtype(
            cache_config.cache_dtype, self.model_config
        )

        # ------【核心逻辑】记录 pooling/多模态等运行类型标志，供后续分支使用 ------
        self.is_pooling_model = model_config.runner_type == "pooling"
        self.enable_prompt_embeds = model_config.enable_prompt_embeds
        self.is_multimodal_raw_input_only_model = (
            model_config.is_multimodal_raw_input_only_model
        )
        # These will be overridden in load_model()
        # ------【核心逻辑】多模态剪枝/顺序视频编码等标志先置默认，load_model 时再覆盖 ------
        self.is_multimodal_pruning_enabled = False
        self.requires_sequential_video_encoding = False
        # ------【显存 profiling】routed experts 在 profiling/dummy 阶段禁止运行，初始化后才开启 ------
        # Set to True after init_routed_experts_capturer() completes.
        # Prevents routed experts code from running during profiling/dummy run.
        self.routed_experts_initialized = False
        self.max_model_len = model_config.max_model_len

        # ------【DP】decode context parallel 的规模与 rank，决定后续 DCP 相关分支 ------
        # Always set to false after the first forward pass
        self.dcp_world_size = self.parallel_config.decode_context_parallel_size
        self.dcp_rank = 0 if self.dcp_world_size <= 1 else get_dcp_group().rank_in_group








        # ------【核心逻辑】记录单步可调度的最大 token 数与请求数，作为缓冲上限 ------
        self.max_num_tokens = scheduler_config.max_num_batched_tokens
        self.max_num_reqs = scheduler_config.max_num_seqs

        # ------【PP+NCCL 通信】external_launcher 且多 PP rank 时广播输出，保证跨 rank 同步 ------
        # Broadcast PP output for external_launcher (torchrun)
        # to make sure we are synced across pp ranks
        # TODO: Support overlapping micro-batches
        # https://github.com/vllm-project/vllm/issues/18019
        self.broadcast_pp_output = (
            self.parallel_config.distributed_executor_backend == "external_launcher"
            and len(get_pp_group().ranks) > 1
        )

        # ------【核心逻辑】缓存 query 头数与输入 embedding 维度，供 attention/位置编码使用 ------
        # Model-related.
        self.num_query_heads = model_config.get_num_attention_heads(parallel_config)
        self.inputs_embeds_size = model_config.get_inputs_embeds_size()
        # Only relevant for models using ALiBi (e.g, MPT)
        self.use_alibi = model_config.uses_alibi

        # ------【核心逻辑】cascade attention 与多模态前缀 LM 标志 ------
        self.cascade_attn_enabled = not self.model_config.disable_cascade_attn
        self.is_mm_prefix_lm = self.model_config.is_mm_prefix_lm

        # ------【核心逻辑】多模态注册表与 M/XD-RoPE 变体标志 ------
        # Multi-modal data support
        self.mm_registry = MULTIMODAL_REGISTRY
        self.uses_mrope = model_config.uses_mrope
        self.uses_xdrope_dim = model_config.uses_xdrope_dim
        self.supports_mm_inputs = self.mm_registry.supports_multimodal_inputs(
            model_config
        )

        # ------【核心逻辑】encoder-decoder 模型记录 encoder 最大输入长度，其余置 0 ------
        if self.model_config.is_encoder_decoder:
            # Maximum length of the encoder input, only for encoder-decoder
            # models.
            self.max_encoder_len = scheduler_config.max_num_encoder_input_tokens
        else:
            self.max_encoder_len = 0

        # ------【异步 RPC】异步调度标志，决定是否用独立 stream 重叠 D2H 拷贝 ------
        # Async scheduling
        self.use_async_scheduling = self.scheduler_config.async_scheduling








        ################################################
        # 2. 构造采样器
        ################################################
        # Sampler
        self.sampler = Sampler(
            logprobs_mode=self.model_config.logprobs_mode,
            use_fp64_gumbel=self.model_config.use_fp64_gumbel,
        )

        # ------【EP/EPLB】EPLB 状态与 MoE 模型引用延迟到加载模型时初始化 ------
        self.eplb_state: EplbState | None = None
        self._moe_model: MixtureOfExperts | None = None
        # NOTE(yongji): flag to temporarily disable EPLB during scaling up/down
        self.eep_eplb_suppressed = False
        """
        State of the expert parallelism load balancer.

        Will be lazily initialized when the model is loaded.
        """






        ################################################
        # 3. KV cache tensor 列表声明
        ################################################
        # ------【内存池/CuMem】KV cache 张量与注意力分组等先占位，待 initialize_kv_cache 再填 ------
        # Lazy initializations
        # self.model: nn.Module  # Set after load_model
        # Initialize in initialize_kv_cache
        self.kv_caches: list[torch.Tensor] = [] # 每层的kvcache张量
        # Initialize in initialize_kv_cache_tensors
        self.cross_layers_kv_cache: torch.Tensor | None = None
        self.cross_layers_attn_backend: type[AttentionBackend] | None = None
        # indexes: [kv_cache_group_id][attn_group]
        self.attn_groups: list[list[AttentionGroup]] = []
        # self.kv_cache_config: KVCacheConfig

        # ------【前缀缓存】用 mm_hash 作 key 缓存 encoder 输出，供多模态请求复用 ------
        # mm_hash ->  encoder_output
        self.encoder_cache: dict[str, torch.Tensor] = {}
        self.late_interaction_runner = LateInteractionRunner()

        # ------【CUDA Graph】encoder cudagraph 管理器延迟到 load_model 后按需初始化 ------
        # Encoder CUDA graph manager (initialized after model load if enabled)
        self.encoder_cudagraph_manager: EncoderCudaGraphManager | None = None

        # ------【投机解码】辅助隐藏状态输出标志，EAGLE/DFlash 等草稿方式需要额外输出 ------
        self.use_aux_hidden_state_outputs = False
        # Set up speculative decoding.
        # NOTE(Jiayi): currently we put the entire draft model on
        # the last PP rank. This is not ideal if there are many
        # layers in the draft model.
        # ------【投机解码+PP】drafter 只放在最后一个 PP rank，按 method 分派不同的草稿生成器 ------
        if self.speculative_config and get_pp_group().is_last_rank:
            self.drafter: (
                NgramProposer  # noqa: F823
                | NgramProposerGPU
                | SuffixDecodingProposer
                | EagleProposer
                | DFlashProposer
                | DraftModelProposer
                | MedusaProposer
                | ExtractHiddenStatesProposer
                | Gemma4Proposer
                | Step3p5MTPProposer
            )
            # ------【投机解码】custom_class：通过用户自定义 proposer 工厂创建 drafter ------
            if self.speculative_config.method == "custom_class":
                self.drafter = create_custom_proposer(  # type: ignore[assignment]
                    self.vllm_config
                )
            # ------【投机解码】ngram：CPU 侧 n-gram 匹配历史 token 生成草稿 ------
            elif self.speculative_config.method == "ngram":
                from vllm.v1.spec_decode.ngram_proposer import NgramProposer

                self.drafter = NgramProposer(self.vllm_config)
            # ------【投机解码】draft_model：用独立小模型（如 EAGLE 草稿模型）生成候选 ------
            elif self.speculative_config.uses_draft_model():
                self.drafter = DraftModelProposer(
                    vllm_config=self.vllm_config,
                    device=self.device,
                    runner=self,
                )
            # ------【投机解码+异步 RPC】ngram_gpu：GPU 侧 n-gram，预分配 pinned 缓冲做异步 D2H ------
            elif self.speculative_config.use_ngram_gpu():
                self.drafter = NgramProposerGPU(self.vllm_config, self.device, self)
                self.num_tokens_no_spec_gpu = torch.zeros(
                    self.max_num_reqs, dtype=torch.int32, device=device
                )
                self.token_ids_gpu_tensor = torch.zeros(
                    self.max_num_reqs,
                    self.max_model_len,
                    dtype=torch.int32,
                    device=device,
                )
                self._ngram_pinned_idx_buf = torch.zeros(
                    self.max_num_reqs, dtype=torch.long, pin_memory=True
                )
                self._ngram_pinned_val_buf = torch.zeros(
                    self.max_num_reqs, dtype=torch.int32, pin_memory=True
                )
            # ------【投机解码】gemma4_mtp：Gemma4 的多 token 预测头作为草稿 ------
            elif self.speculative_config.use_gemma4_mtp():
                self.drafter = Gemma4Proposer(self.vllm_config, self.device, self)
            # ------【投机解码】step3p5_mtp：Step3.5 多 token 预测作为草稿 ------
            elif self.speculative_config.use_step3p5_mtp():
                self.drafter = Step3p5MTPProposer(self.vllm_config, self.device, self)
            # ------【投机解码】dflash：DFlash proposer，需额外输出辅助隐藏状态 ------
            elif self.speculative_config.use_dflash():
                self.drafter = DFlashProposer(self.vllm_config, self.device, self)
                self.use_aux_hidden_state_outputs = True
            # ------【投机解码】suffix：基于后缀匹配的解码 proposer ------
            elif self.speculative_config.method == "suffix":
                self.drafter = SuffixDecodingProposer(self.vllm_config)
            # ------【投机解码】eagle：EAGLE 隐藏状态预测，eagle3 额外用辅助隐藏状态 ------
            elif self.speculative_config.use_eagle():
                self.drafter = EagleProposer(self.vllm_config, self.device, self)
                if self.speculative_config.method == "eagle3":
                    self.use_aux_hidden_state_outputs = (
                        self.drafter.eagle3_use_aux_hidden_state
                    )
            # ------【投机解码】medusa：多头并行预测多个后续 token ------
            elif self.speculative_config.method == "medusa":
                self.drafter = MedusaProposer(
                    vllm_config=self.vllm_config, device=self.device
                )
            # ------【投机解码】extract_hidden_states：直接抽取隐藏状态作为草稿，需辅助输出 ------
            elif self.speculative_config.method == "extract_hidden_states":
                self.drafter = ExtractHiddenStatesProposer(
                    vllm_config=self.vllm_config, device=self.device
                )
                self.use_aux_hidden_state_outputs = True
            else:
                raise ValueError(
                    "Unknown speculative decoding method: "
                    f"{self.speculative_config.method}"
                )
            # ------【投机解码】拒绝采样器校验草稿 token 是否被主模型接受 ------
            self.rejection_sampler = RejectionSampler(
                self.sampler, self.speculative_config, self.device
            )

        # ------【投机解码】初始化投机 token 数并推导 drafter 最大长度（草稿模型优先） ------
        self.num_spec_tokens = 0
        self.prev_num_spec_tokens = 0
        self.valid_sampled_token_count_gpu: torch.Tensor | None = None
        if self.speculative_config:
            self.num_spec_tokens = self.speculative_config.num_speculative_tokens
            self.prev_num_spec_tokens = self.num_spec_tokens
            draft_config = self.speculative_config.draft_model_config
            if draft_config is not None and draft_config.max_model_len is not None:
                self.effective_drafter_max_model_len = draft_config.max_model_len
            else:
                self.effective_drafter_max_model_len = self.max_model_len
        # ------【投机解码+异步 RPC】异步调度且启用投机时开启异步投机解码路径 ------
        self.use_async_spec_decode = (
            self.use_async_scheduling and self.num_spec_tokens > 0
        )






        ################################################
        # 4. 请求状态 
        ################################################

        '''
        每个model_runner实例对象，维护的内部属性：

        self.requests : dict[req_name, CachedRequestState]
        
        '''
        # 每个req的持久状态缓存
        # Request states.
        self.requests: dict[str, CachedRequestState] = {} ####################### 持久化批处理 = 持久化状态对象的列表


        # NOTE(rob): num_prompt_logprobs only includes reqs
        # that are currently in the prefill phase.
        self.num_prompt_logprobs: dict[str, int] = {}


        
        # Input Batch
        # NOTE(Chen): Ideally, we should initialize the input batch inside
        # `initialize_kv_cache` based on the kv cache config. However, as in
        # https://github.com/vllm-project/vllm/pull/18298, due to some unknown
        # reasons, we have to initialize the input batch before `load_model`,
        # quantization + weight offloading will fail otherwise. As a temporary
        # solution, we initialize the input batch here, and re-initialize it
        # in `initialize_kv_cache` if the block_sizes here is different from
        # the block_sizes in the kv cache config.
        # ------【核心逻辑】构建自定义 logits 处理器序列，供 InputBatch 采样阶段调用 ------
        logits_processors = model_config.logits_processors # 原始打分处理器
        custom_logitsprocs: Sequence[str | type[LogitsProcessor]] = (
            tuple(logits_processors) if logits_processors is not None else ()
        )
        # ------【内存池/CuMem】用占位 block_size 预初始化 input_batch，之后按真实 KV 配置重建 ------
        placeholder_block_size = (
            self.cache_config.block_size or CacheConfig.DEFAULT_BLOCK_SIZE
        )
        placeholder_max_num_blocks = cdiv(
            max(self.max_model_len, self.max_encoder_len), placeholder_block_size
        )
        self._init_block_sizes = [placeholder_block_size]
        self._init_kernel_block_sizes = [placeholder_block_size]
        self._init_max_num_blocks = [placeholder_max_num_blocks]
        self._init_slot_mapping_modes = [SlotMappingMode.TOKEN_TO_KV_SLOT]



        ################################################
        # 5. InputBatch 转换器 
        ################################################

        # batch 级输入管理，绑定了 vocab 大小、block 大小、logitsprocs 等元信息。step 时把 scheduler 给的一批 req 组装进这里
        self.input_batch = InputBatch(
            max_num_reqs=self.max_num_reqs, # batch的最大req数
            # We need to use the encoder length for encoder-decoder
            # because of KV cache for cross-attention.
            max_model_len=max(self.max_model_len, self.max_encoder_len), # 每个req的最大kv长度
            max_num_batched_tokens=self.max_num_tokens, # 这个batch的最大tokens数量
            device=self.device,
            vocab_size=self.model_config.get_vocab_size(), # 词袋大小
            block_sizes=[placeholder_block_size],
            kernel_block_sizes=[placeholder_block_size],
            max_num_blocks_per_req=[placeholder_max_num_blocks],
            num_spec_tokens=self.num_spec_tokens, # 投机解码的草稿token数量
            logitsprocs=build_logitsprocs( # 构建的打分处理器，在采样前，修改这个分数，施加约束
                self.vllm_config,
                self.device,
                PIN_MEMORY,
                self.is_pooling_model,
                custom_logitsprocs,
            ),
            # We currently don't know whether a particular custom logits processor
            # uses output token ids so we set this conservatively. Thinking-budget
            # tracking is requested dynamically when a budgeted request is in the batch.
            logitsprocs_need_output_token_ids=bool(custom_logitsprocs),
            is_pooling_model=self.is_pooling_model,
            cp_kv_cache_interleave_size=self.parallel_config.cp_kv_cache_interleave_size,
            reasoning_config=self.vllm_config.reasoning_config,
            use_replayssm=self.cache_config.use_replayssm,
        )

        # ------【异步 RPC】异步调度时用独立 stream 重叠采样 token 的 GPU→CPU 拷贝 ------
        # Separate cuda stream for overlapping transfer of sampled token ids from
        # GPU to CPU when async scheduling is enabled.
        self.async_output_copy_stream: torch.cuda.Stream | None = None
        # cuda event to synchronize use of reused CPU tensors between steps
        # when async scheduling is enabled.
        self.prepare_inputs_event: torch.Event | None = None
        if self.use_async_scheduling:
            self.async_output_copy_stream = torch.cuda.Stream()
            # Blocking (sleep) event to avoid busy-polling the CUDA driver lock;
            # under TP contention that spin can balloon and make the rank a straggler.
            self.prepare_inputs_event = torch.cuda.Event(blocking=True)

        # ------【CUDA Graph】按配置捕获的 batch size 升序排列，供运行时选择对应 graph ------
        # self.cudagraph_batch_sizes sorts in ascending order.
        ##############################################
        # 看看config里面，是否有指定 图 的key形状，如果有，就排序后，保存下
        ##############################################
        if (
            self.compilation_config.cudagraph_capture_sizes
            and self.compilation_config.cudagraph_mode != CUDAGraphMode.NONE
        ):
            self.cudagraph_batch_sizes = sorted(
                self.compilation_config.cudagraph_capture_sizes
            )
        else:
            self.cudagraph_batch_sizes = []

        # ------【核心逻辑】缓存设备属性（SM 数量等），供 kernel 调优与分派使用 ------
        # Cache the device properties.
        self._init_device_properties()

        # ------【核心逻辑】encoder 耗时统计注册表 + 线程锁，用于可观测性 ------
        # Encoder timing registry for observability
        self.encoder_timing_registry: dict[str, EncoderTimingStats] = {}
        self._encoder_timing_lock = threading.Lock()

        # ------【CUDA Graph+内存池/CuMem】预分配 CUDA graph 回放所需的持久缓冲，避免每步重复分配 ------
        # Persistent buffers for CUDA graphs.




        #################
        # 这边其实就是构造本次batch的快照的输入缓冲区，只不过利用了一个CPU-GPU的双份+numpy视图，cpu侧的buffer负责写入，GPU侧的显存区域负责读取。
        # 这些字段，每个都是单独的一块，大小 = 最大容量， 合起来分角度描述当前的batch
        # 这些buffer的内容不累计，每个step来新batch, 就把活跃的前n个位置复写成新值，GPU来读这一step的输入快照

        # 真正的持续状态是requests: 每个req的元信息，跨step累计

        # 这些 _make_buffer 的 buffer 只是每次的输入暂存区，固定分配是为了两点：

            # CUDA graph 回放要固定地址；
            # 避免每步 malloc/free。

        #####################
        # input_ids, positions, query_start_loc, seq_lens, num_computed_tokens/ req_indices 
        # 这些是 CUDA graph 回放时要复用的固定张量,提前按 max_num_tokens/max_num_reqs 一次性分配，避免每步重复 malloc
        '''
                场景
        请求	        类型	                                本轮 query token
        req0	        prefill，prompt="你好介绍一下vLLM"	        9 个 token（整个 prompt）
        req1	        decode，已生成 30 个 token	                1 个新 token
        req2	        decode，已生成 15 个 token	                1 个新 token
        req3	        prefill，prompt="什么是attention"	        7 个 token
        
        本轮 query token 总数 = 9 + 1 + 1 + 7 = 18。

----------------------------------------------------------------------------------------------------
        query_start_loc = [0, 9, 10, 11, 18]   # 长度 5 = 4 请求 + 1

        input_ids  (扁平一维，长度 18):
        [你,好,介,绍,一,下,v,L,L,M | 新token | 新token | 什,么,是,a,t,t,e]
        └──── req0 (0~8)    ────┘  req1(9)  req2(10)  └── req3 (11~17) ──┘

        positions  (每个 token 在自己序列里的绝对位置):
        [0,1,2,3,4,5,6,7,8              | 30 |  15      | 0,1,2,3,4,5,6]
        └── req0: prompt 从 0 排到 8 ──┘ ↑req1   ↑req2  └─ req3: prompt 从 0 排 ─┘

        seq_lens = [9, 31, 16, 7]
                ↑req0(9)  ↑req1(30+1=31)  ↑req2(15+1=16)  ↑req3(7)


----------------------------------------------------------------------------------------------------                       
        三个数组怎么对上
        query_start_loc 告诉你每个请求的 query 在扁平数组里从哪到哪：

            req0 → input_ids[0:9]
            req1 → input_ids[9:10]
            req2 → input_ids[10:11]
            req3 → input_ids[11:18]
        positions 和 input_ids 一一对应（同一扁平索引），但值不同：

        prefill 的 req0/req3：从 0 递增（因为是 prompt，位置从头排）
        decode 的 req1/req2：只有一个值 30/15（这是该 token 在它自己整条序列里的绝对位置，不是扁平数组下标）
        seq_lens 是每个请求完整序列长度（KV 侧要读多少），和 query 数无关：

        req1 的 seq_lens=31，但本轮 query 只有 1 个（query_start_loc 区间长度 1）。

----------------------------------------------------------------------------------------------------
        query_start_loc:  扁平数组下标 → 每个请求的 query 起止（Q 侧，本轮要算几个 token）
        positions:        每个 query token 在各自序列里的绝对位置（喂给 RoPE 位置编码）
        seq_lens:         每个请求完整长度（KV 侧，attention 要读多少历史）
        input_ids         扁平化的所有req的token ids

        '''

        ################################################################################################
        # 5. 每轮batch输入的整理统计信息 的 缓冲区申请
        ################################################################################################
        self.input_ids = self._make_buffer(self.max_num_tokens, dtype=torch.int32) 
        self.positions = torch.zeros(
            self.max_num_tokens, dtype=torch.int64, device=self.device # gpu侧的张量：长度是本轮调度的token最大个数
        )
        '''
        query就是一个req本轮要计算的token部分，prefill就是prompt, decode就是1token

        不同的请求，query长度不同，无法用规整的[batch, seq_len]二维矩阵装，
        vllm的做法是把所有req的query token 压成一条一维数组，然后用query_start_loc记录每个请求的起点。
        '''
        self.query_start_loc = self._make_buffer( # cpu侧的buffer, 
            self.max_num_reqs + 1, dtype=torch.int32
        )
        self.seq_lens = torch.zeros(
            self.max_num_reqs, dtype=torch.int32, device=self.device
        )

        # ------【异步 RPC】pinned CPU 上界缓冲，供 CPU 侧免同步读取 seq_lens ------
        self.optimistic_seq_lens_cpu = torch.zeros(
            self.max_num_reqs, dtype=torch.int32, pin_memory=PIN_MEMORY
        )


        # ------【核心逻辑】已计算 token 数、草稿 token 数、请求索引与位置映射等状态缓冲 ------
        # 每个req已经被计算的tokens数
        self.num_computed_tokens = torch.zeros(
            self.max_num_reqs, dtype=torch.int32, device=self.device
        )
        # 每个req的草稿token数
        self.prev_num_draft_tokens = self._make_buffer(
            self.max_num_reqs, dtype=torch.int32
        )
        self.req_indices = self._make_buffer(self.max_num_tokens, dtype=torch.int64)
        # Maps current batch position -> previous batch position (-1 for new reqs)
        self.prev_positions = self._make_buffer(self.max_num_reqs, dtype=torch.int64)
        self.num_scheduled_tokens = self._make_buffer(
            self.max_num_reqs, dtype=torch.int32
        )

        # ------【核心逻辑】encoder 序列长度缓冲；DCP 开启时额外分配 local seq_lens ------
        self.encoder_seq_lens = self._make_buffer(self.max_num_reqs, dtype=torch.int32)
        if self.dcp_world_size > 1:
            self.dcp_local_seq_lens = self._make_buffer(
                self.max_num_reqs, dtype=torch.int32
            )
        # ------【CUDA Graph】inputs_embeds 可能为 bf16，禁 numpy 避免 RuntimeError ------
        # Because inputs_embeds may be bfloat16 and we don't need a numpy
        # version of this tensor, avoid a RuntimeError by not creating a
        # numpy buffer.
        self.inputs_embeds = self._make_buffer(
            self.max_num_tokens, self.inputs_embeds_size, dtype=self.dtype, numpy=False
        )
        # ------【投机解码】is_token_ids/丢弃掩码/草稿数与接受 token 数等剩余缓冲 ------
        self.is_token_ids = self._make_buffer(self.max_num_tokens, dtype=torch.bool)
        self.discard_request_mask = self._make_buffer(
            self.max_num_reqs, dtype=torch.bool
        )
        self.num_decode_draft_tokens = self._make_buffer(
            self.max_num_reqs, dtype=torch.int32
        )
        self.num_accepted_tokens = self._make_buffer(
            self.max_num_reqs, dtype=torch.int32
        )

        # ------【核心逻辑】M-RoPE 3D 位置缓冲，多一维 dummy 保证非连续以兼容 torch compile ------
        # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
        if self.uses_mrope:
            # NOTE: `mrope_positions` is implemented with one additional dummy
            # position on purpose to make it non-contiguous so that it can work
            # with torch compile.
            # See detailed explanation in https://github.com/vllm-project/vllm/pull/12128#discussion_r1926431923

            # NOTE: When M-RoPE is enabled, position ids are 3D regardless of
            # the modality of inputs. For text-only inputs, each dimension has
            # identical position IDs, making M-RoPE functionally equivalent to
            # 1D-RoPE.
            # See page 5 of https://arxiv.org/abs/2409.12191
            self.mrope_positions = self._make_buffer(
                (3, self.max_num_tokens + 1), dtype=torch.int64
            )

        # ------【核心逻辑】XD-RoPE 位置缓冲，按分配的维度数扩展 ------
        # Only relevant for models using XD-RoPE (e.g, HunYuan-VL)
        if self.uses_xdrope_dim > 0:
            # Similar to mrope but use assigned dimension number for RoPE, 4 as default.
            self.xdrope_positions = self._make_buffer(
                (self.uses_xdrope_dim, self.max_num_tokens + 1), dtype=torch.int64
            )

        # ------【PP】intermediate_tensors 仅非首个 PP rank 使用，load_model 后填充 ------
        # None in the first PP rank. The rest are set after load_model.
        self.intermediate_tensors: IntermediateTensors | None = None

        # ------【CUDA Graph+核心逻辑】缓存 arange 张量避免每步重复创建，int64 防长上下文溢出 ------
        # OPTIMIZATION: Cache the arange tensors rather than creating them
        # every step. Keep in int64 to avoid overflow with long context.
        # - arange_np: immutable [0, 1, 2, ...] used as source for batched computation
        # - query_pos: CpuGpuBuffer for the computed batched arange result
        arange_size = max(self.max_num_reqs + 1, self.max_num_tokens)
        self.arange_np = np.arange(arange_size, dtype=np.int64)
        self.query_pos = self._make_buffer(arange_size, dtype=torch.int64)
        self._arange_scratch = np.empty(arange_size, dtype=np.int64)

        # ------【内存池/CuMem】跨层 KV 共享映射表；fast prefill 时预分配 logits 索引张量 ------
        # Layer pairings for cross-layer KV sharing.
        # If an Attention layer `layer_name` is in the keys of this dict, it
        # means this layer will perform attention using the keys and values
        # from the KV cache of `shared_kv_cache_layers[layer_name]`.
        self.shared_kv_cache_layers: dict[str, str] = {}
        self.kv_sharing_fast_prefill_eligible_layers: set[str] = set()

        self.kv_sharing_fast_prefill_logits_indices = None
        if self.cache_config.kv_sharing_fast_prefill:
            self.kv_sharing_fast_prefill_logits_indices = torch.zeros(
                self.max_num_tokens, dtype=torch.int32, device=self.device
            )

        # ------【投机解码】decode 阶段统一 query 长度 = 1 + 投机 token 数 ------
        ################################################## 记录下均匀decode下的每个req的token数量应该是多少
        self.uniform_decode_query_len = 1 + self.num_spec_tokens

        # ------【CUDA Graph】运行时 cudagraph 分派器，按 batch size 选择对应 graph ------
        # Cudagraph dispatcher for runtime cudagraph dispatching.
        ######################################################################################
        # 构造一个初始的 图调度器 cudagraphdispatcher
        ######################################################################################
        self.cudagraph_dispatcher = CudagraphDispatcher(self.vllm_config)

        # ------【显存 profiling】多模态预算器，控制 encoder 的显存与计算预算 ------
        self.mm_budget = (
            MultiModalBudget(self.vllm_config, self.mm_registry)
            if self.supports_mm_inputs
            else None
        )

        self.reorder_batch_threshold: int | None = None

        # ------【内存池/CuMem】仅 runner 侧 KV 配置存在的注意力层集合（KV 共享/encoder-only） ------
        # Attention layers that are only in the KVCacheConfig of the runner
        # (e.g., KV sharing, encoder-only attention), but not in the
        # KVCacheConfig of the scheduler.
        self.runner_only_attn_layers: set[str] = set()

        # ------【投机解码】缓存上一步草稿 token/probs，供下一步投机复用 ------
        # Cached outputs.
        self._draft_token_ids: list[list[int]] | torch.Tensor | None = None
        self._draft_probs: torch.Tensor | None = None
        self._draft_prob_req_ids: list[str] | None = None
        # N-gram GPU path: async D2H buffer/event for per-request valid draft counts.
        self._num_valid_draft_tokens: torch.Tensor | None = None
        self._num_valid_draft_tokens_cpu: torch.Tensor | None = None
        self._num_valid_draft_tokens_event: torch.cuda.Event | None = None
        self._num_valid_draft_tokens_copy_stream: torch.cuda.Stream | None = None
        # ------【投机解码+异步 RPC】ngram_gpu 路径预分配 pinned CPU 缓冲与事件做异步 D2H ------
        if (
            self.speculative_config is not None
            and self.speculative_config.use_ngram_gpu()
        ):
            self._num_valid_draft_tokens_cpu = torch.empty(
                self.max_num_reqs, dtype=torch.int32, pin_memory=PIN_MEMORY
            )
            self._num_valid_draft_tokens_event = torch.cuda.Event()
            self._num_valid_draft_tokens_copy_stream = torch.cuda.Stream()

        # ------【异步 RPC】pinned CPU 缓冲 + event 用于采样 token 的异步 D2H 拷贝 ------
        self._draft_token_req_ids: list[str] | None = None
        self.transfer_event = torch.Event()
        self.sampled_token_ids_pinned_cpu = torch.empty(
            (self.max_num_reqs, 1),
            dtype=torch.int64,
            device="cpu",
            pin_memory=PIN_MEMORY,
        )

        # ------【投机解码+异步 RPC】预分配拷贝有效采样数/草稿 token 的 CPU 缓冲与 stream/event ------
        # Pre-allocated tensor for copying valid sampled token counts to CPU,
        # with dedicated stream for overlapping and event for coordination.
        self.valid_sampled_token_count_event: torch.Event | None = None
        self.valid_sampled_token_count_copy_stream: torch.cuda.Stream | None = None
        # We also copy the drafted tokens to the CPU asynchronously,
        # in case we need them for structured outputs.
        self.draft_token_ids_event: torch.Event | None = None
        self.draft_token_ids_copy_stream: torch.cuda.Stream | None = None
        self.valid_sampled_token_count_cpu: torch.Tensor | None = None
        self.draft_token_ids_cpu: torch.Tensor | None = None
        self.num_accepted_tokens_event: torch.Event | None = None
        if self.num_spec_tokens:
            self.draft_token_ids_event = torch.Event()
            self.num_accepted_tokens_event = torch.Event()
            self.draft_token_ids_copy_stream = torch.cuda.Stream()
            self.draft_token_ids_cpu = torch.empty(
                (self.max_num_reqs, self.num_spec_tokens),
                dtype=torch.int64,
                device="cpu",
                pin_memory=PIN_MEMORY,
            )
            if self.use_async_scheduling:
                self.valid_sampled_token_count_event = torch.Event()
                self.valid_sampled_token_count_copy_stream = torch.cuda.Stream()
                self.valid_sampled_token_count_cpu = torch.empty(
                    self.max_num_reqs,
                    dtype=torch.int32,
                    device="cpu",
                    pin_memory=PIN_MEMORY,
                )

        # ------【显存 profiling】创建权重 offloader，需在任何 get_offloader 之前调用 ------
        # Model weight offloader
        # Make sure this is called before any get_offloader call
        set_offloader(create_offloader(self.offload_config))

        # ------【核心逻辑】execute_model 与 sample_tokens 之间传递的临时状态 ------
        # Ephemeral state transferred between execute_model() and sample_tokens().
        self.execute_model_state: ExecuteModelState | None = None
        self.kv_connector_output: KVConnectorOutput | None = None
        # ------【核心逻辑】Mamba 状态索引与缓冲懒初始化占位 ------
        self.mamba_state_idx: dict[str, int] = {}
        self._mamba_bufs: mamba_utils.MambaBuffers | None = None
        self.mamba_prev_last_scheduled_idx: CpuGpuBuffer | None = None
        # ------【投机解码】Mamba + 投机解码时记录上一调度索引，供状态对齐 ------
        if self.cache_config.mamba_cache_mode == "all" and self.num_spec_tokens > 0:
            self.mamba_prev_last_scheduled_idx = self._make_buffer(
                self.max_num_reqs, dtype=torch.int32
            )
        self.layerwise_nvtx_hooks_registered = False
















    def update_max_model_len(self, max_model_len: int) -> None:
        # ------【核心逻辑】更新主模型最大长度 ------
        self.max_model_len = max_model_len
        # ------【投机解码】草稿模型未显式设长度时，跟随主模型最大长度 ------
        if self.speculative_config:
            draft_config = self.speculative_config.draft_model_config
            if draft_config is None or draft_config.max_model_len is None:
                self.effective_drafter_max_model_len = self.max_model_len

    def reset_mm_cache(self) -> None:
        """
        Clear the multi-modal cache that was used during profiling,
        but no longer needed during inference.
        """
        # ------【前缀缓存】清空 profiling 期间使用、推理不再需要的多模态缓存 ------
        if self.mm_budget:
            self.mm_budget.reset_cache()
        self.late_interaction_runner.clear()

    def reset_encoder_cache(self) -> None:
        """Clear the GPU-side encoder cache storing vision embeddings.

        This should be called when model weights are updated to ensure
        stale embeddings computed with old weights are not reused.
        """
        # ------【前缀缓存】权重更新后清空 encoder 缓存，防止复用旧权重的过期嵌入 ------
        self.encoder_cache.clear()
        self.late_interaction_runner.clear()

    def post_kv_cache_wake_up(self) -> None:
        # ------【显存 profiling】sleep 唤醒后重初始化 FP8 KV scale ------
        self.init_fp8_kv_scales()

    @torch.inference_mode()
    def init_fp8_kv_scales(self) -> None:
        """
        Re-initialize the KV cache and FP8 scales after waking from sleep.
        1. Zero out the KV cache tensors to remove garbage data from re-allocation.
        2. Reset Attention layer scaling factors (_k_scale, _v_scale) to 1.0.
          If these are left at 0.0 (default after wake_up), all KV cache values
          become effectively zero, causing gibberish output.
        """
        # ------【内存池/CuMem】非量化 KV cache 无需处理 FP8 scale，直接返回 ------
        if not is_quantized_kv_cache(self.cache_config.cache_dtype):
            return

        # ------【显存 profiling】wake 后把 KV cache 张量清零，去除重分配残留的垃圾数据 ------
        kv_caches = getattr(self, "kv_caches", [])
        for cache_entry in kv_caches:
            if cache_entry is None:
                continue
            # Hybrid models (Mamba, DeltaNet) store per-layer state as a
            # list of tensors rather than a single tensor.
            if isinstance(cache_entry, list):
                for t in cache_entry:
                    t.zero_()
            else:
                cache_entry.zero_()

        # ------【显存 profiling+内存池/CuMem】重置注意力层 K/V scale 为 1.0，避免全零导致乱码 ------
        k_attr_names = ("_k_scale", "k_scale")
        v_attr_names = ("_v_scale", "v_scale")

        attn_layers = self.compilation_config.static_forward_context
        for name, module in attn_layers.items():
            if isinstance(module, (Attention, MLAAttention)):
                # TODO: Generally, scale is 1.0 if user uses on-the-fly fp8
                # kvcache quant. However, to get better accuracy, compression
                # frameworks like llm-compressors allow users to tune the
                # scale. We may need to restore the specific calibrated scales
                # here in the future.
                k_scale_val, v_scale_val = 1.0, 1.0

                # Processing K Scale
                for attr in k_attr_names:
                    if hasattr(module, attr):
                        param = getattr(module, attr)
                        if isinstance(param, torch.Tensor):
                            param.fill_(k_scale_val)

                # Processing V Scale
                for attr in v_attr_names:
                    if hasattr(module, attr):
                        param = getattr(module, attr)
                        if isinstance(param, torch.Tensor):
                            param.fill_(v_scale_val)

    def _get_positions(self, num_tokens: Any):
        # ------【核心逻辑】按输入是标量还是切片索引，从对应位置缓冲取位置张量 ------
        if isinstance(num_tokens, int):
            # ------【核心逻辑】M-RoPE/XD-RoPE 走各自多维缓冲，否则走普通 positions ------
            if self.uses_mrope:
                return self.mrope_positions.gpu[:, :num_tokens]
            if self.uses_xdrope_dim > 0:
                return self.xdrope_positions.gpu[:, :num_tokens]
            return self.positions[:num_tokens]
        else:
            if self.uses_mrope:
                return self.mrope_positions.gpu[:, num_tokens]
            if self.uses_xdrope_dim > 0:
                return self.xdrope_positions.gpu[:, num_tokens]
            return self.positions[num_tokens]

    def _make_buffer(
        self, *size: int | torch.SymInt, dtype: torch.dtype, numpy: bool = True
    ) -> CpuGpuBuffer:
        # ------【内存池/CuMem】统一封装 CPU/GPU 双缓冲分配，供各持久缓冲复用 ------
        return CpuGpuBuffer(
            *size,
            dtype=dtype,
            device=self.device,
            with_numpy=numpy,
        )

    def _get_mamba_bufs(self) -> mamba_utils.MambaBuffers:
        # Only reachable on the ``mamba_cache_mode == "align"`` path.
        # The postprocess sub-object is additionally gated on spec
        # decode + hybrid model.
        assert self.cache_config.mamba_cache_mode == "align"
        # ------【核心逻辑】懒创建 Mamba 状态缓冲（align 模式），避免非必要路径额外开销 ------
        if self._mamba_bufs is None:
            self._mamba_bufs = mamba_utils.MambaBuffers.create(
                max_num_reqs=self.max_num_reqs,
                kv_cache_config=self.kv_cache_config,
                copy_funcs=self.model.get_mamba_state_copy_func(),
                make_buffer=self._make_buffer,
                device=self.device,
                with_postprocess_align=(
                    self.speculative_config is not None and self.model_config.is_hybrid
                ),
            )
        return self._mamba_bufs

    def _init_model_kwargs(self):
        model_kwargs = dict[str, Any]()

        # ------【核心逻辑】非 pooling 模型无需额外 kwargs，直接返回空 dict ------
        if not self.is_pooling_model:
            return model_kwargs

        num_reqs = self.input_batch.num_reqs
        pooling_params = self.input_batch.get_pooling_params()

        # ------【核心逻辑】收集带 compressed_token_type_ids 的请求索引 ------
        token_type_id_requests = dict[int, Any]()
        for i, param in enumerate(pooling_params):
            if (
                param.extra_kwargs is not None
                and (token_types := param.extra_kwargs.get("compressed_token_type_ids"))
                is not None
            ):
                token_type_id_requests[i] = token_types

        # ------【核心逻辑】无 token_type 需求时提前返回，跳过后续构造 ------
        if len(token_type_id_requests) == 0:
            return model_kwargs

        # Build ids on CPU using the CPU-resident upper bound for seq_lens;
        # `torch.arange(seq_lens[i])` with a GPU scalar would force a sync.
        # ------【异步 RPC】用 pinned CPU 上界构造 token_type_ids，避免 GPU 标量触发同步 ------
        seq_lens_cpu = self.optimistic_seq_lens_cpu[:num_reqs].tolist()
        token_type_ids = []

        for i in range(num_reqs):
            seq_len_i = seq_lens_cpu[i]
            pos = token_type_id_requests.get(i, seq_len_i)
            ids = (torch.arange(seq_len_i) >= pos).int()
            token_type_ids.append(ids)

        token_type_ids_cpu = torch.empty(
            sum(seq_lens_cpu), dtype=torch.int32, pin_memory=PIN_MEMORY
        )
        torch.cat(token_type_ids, out=token_type_ids_cpu)
        model_kwargs["token_type_ids"] = token_type_ids_cpu.to(
            device=self.device, non_blocking=True
        )
        return model_kwargs

    def _may_reorder_batch(self, scheduler_output: "SchedulerOutput") -> None:
        """
        Update the order of requests in the batch based on the attention
        backend's needs. For example, some attention backends (namely MLA) may
        want to separate requests based on if the attention computation will be
        compute-bound or memory-bound.

        Args:
            scheduler_output: The scheduler output.
        """
        # Attention free models have zero kv_cache_groups, however models
        # like Mamba are also attention free but use the kv_cache for
        # keeping its internal state. This is why we check the number
        # of kv_cache groups instead of solely checking
        # for self.model_config.is_attention_free.
        # ------【核心逻辑】无 KV 组的 attention-free 模型直接跳过重排 ------
        if len(self.kv_cache_config.kv_cache_groups) == 0:
            return

        # ------【核心逻辑】按阈值把 decode/prefill 分开重排，适配 MLA 等后端访存模式 ------
        if self.reorder_batch_threshold is not None:
            reorder_batch_to_split_decodes_and_prefills(
                self.input_batch,
                scheduler_output,
                decode_threshold=self.reorder_batch_threshold,
            )

    def _init_kv_zero_meta(self) -> None:
        """One-time precomputation for _zero_block_ids.

        Called from gpu_worker.py outside the CuMem pool context.
        """
        # ------【内存池/CuMem】一次性预计算 KV 块清零元数据，供后续按块清零复用 ------
        self._kv_block_zeroer = KVBlockZeroer(
            self.device,
            attn_groups_iter=self._kv_cache_spec_attn_group_iterator(),
            kernel_block_sizes=self._kernel_block_sizes,
            cache_dtype=self.cache_config.cache_dtype,
            runner_only_attn_layers=self.runner_only_attn_layers,
            static_forward_context=self.compilation_config.static_forward_context,
        )

    def _zero_block_ids(self, block_ids: list[int]) -> None:
        """Zero the KV cache memory for the given block IDs."""
        # ------【内存池/CuMem】对新分配 KV 块清零，防脏数据污染 attention/SSM 计算 ------
        if hasattr(self, "_kv_block_zeroer"):
            self._kv_block_zeroer.zero_block_ids(block_ids)

    # Note: used for model runner override.
    def _init_device_properties(self) -> None:
        """Initialize attributes from torch.cuda.get_device_properties"""

        # ------【核心逻辑】缓存 SM 数量，供 kernel 分派与调优使用 ------
        self.num_sms = num_compute_units(self.device.index)

    # Note: used for model runner override.
    def _sync_device(self) -> None:
        # ------【核心逻辑】同步当前加速器设备，保证后续操作有序 ------
        torch.accelerator.synchronize()

    def _get_or_create_async_output_copy_stream(self) -> torch.cuda.Stream:
        # ------【异步 RPC】懒创建异步输出拷贝 stream，已存在则复用 ------
        stream = self.async_output_copy_stream
        if stream is None:
            stream = torch.cuda.Stream()
            self.async_output_copy_stream = stream
        return stream

    def _on_request_state_removed(
        self,
        req_id: str,
        req_state: CachedRequestState | None,
    ) -> None:
        """Hook for platform runners to clean request-scoped side caches."""
        # ------【核心逻辑】请求状态移除钩子，供平台 runner 清理请求级侧缓存 ------
        del req_id, req_state

    def _process_encoder_cache_scheduler_output(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> None:
        """Apply scheduler-side encoder cache lifecycle updates."""
        # ------【前缀缓存】按调度器释放列表逐条删除 encoder 缓存，回收显存 ------
        for mm_hash in scheduler_output.free_encoder_mm_hashes:
            self.encoder_cache.pop(mm_hash, None)






    def _update_states(self, scheduler_output: "SchedulerOutput") -> Callable | None:
        """Update the cached states and the persistent batch with the scheduler
        output.

        The updated states are used by the `_prepare_inputs` function to create
        the input GPU tensors for the model.

        The SamplingMetadata is updated and copied to the GPU if there is a
        new/resumed/paused/finished request in the batch.
        """
        # ------【核心逻辑】遍历本步完成的请求，从缓存状态字典中移除并触发清理钩子 ------
        # Remove finished requests from the cached states.
        ########################################
        # 1. 先把scheduler_output里面的上一轮已经完成的req，从model_runner.requests里面删掉
        ########################################
        for req_id in scheduler_output.finished_req_ids:
            req_state = self.requests.pop(req_id, None) # 直接删掉model_runner.requests里面的这个req
            self._on_request_state_removed(req_id, req_state)
            self.num_prompt_logprobs.pop(req_id, None)


        
        # ------【核心逻辑】通知 late-interaction(如 ColBERT)清理已结束请求的相关状态 ------
        self.late_interaction_runner.on_requests_finished(
            scheduler_output.finished_req_ids
        )
        # ------【核心逻辑】把已结束请求从常驻 batch 移除，处理 abort-再提交的同 ID 重叠边界 ------
        # Remove the finished requests from the persistent batch.
        # NOTE(woosuk): There could be an edge case where finished_req_ids and
        # scheduled_req_ids overlap. This happens when a request is aborted and
        # then resubmitted with the same ID. In this case, we treat them as two
        # distinct requests - clearing the cached states for the first request
        # and handling the second as a new request.

        ########################################
        # 2. 先把scheduler_output里面的上一轮已经完成的req，从input_batch里面删掉
        ########################################
        for req_id in scheduler_output.finished_req_ids:
            self.input_batch.remove_request(req_id)





        # ------【内存池/CuMem】把新分配的 KV 块清零，防止脏数据污染 attention/SSM 计算 ------
        # Zero GPU memory for freshly allocated cache blocks to prevent
        # stale NaN/data from corrupting attention or SSM computation.
        ########################################
        # 3. 处理本调度的batch中，需要清零的新block
        ########################################
        if scheduler_output.new_block_ids_to_zero:
            self._zero_block_ids(scheduler_output.new_block_ids_to_zero)


        ########################################
        # 4. 直接拷贝局部命中block块
        ########################################
        # ------【前缀缓存】就地执行 KV 块拷贝，复用前缀共享块(copy-on-write) ------
        if scheduler_output.kv_cache_block_copies:
            copy_kv_cache_blocks_inplace(
                self.kv_caches,
                self.kv_cache_config.num_blocks,
                scheduler_output.kv_cache_block_copies,
            )

        # ------【前缀缓存】按调度器输出释放/复用多模态 encoder 输出缓存 ------
        # Free the cached encoder outputs.
        self._process_encoder_cache_scheduler_output(scheduler_output)

        # Remove the unscheduled requests from the persistent batch.
        # NOTE(woosuk): The unscheduled requests are either preempted requests
        # or running requests that are not scheduled in this step. We remove
        # them from the persistent batch but keep their cached states since
        # they will be scheduled again sometime in the future.
        # ------【核心逻辑】计算未调度请求集合(被抢占/本步未排上)，仅移出 batch 保留其状态 ------
        ########################################
        # 5. 处理在本轮中被踢掉的req，从inputbatch中移除掉
        ########################################
        scheduled_req_ids = scheduler_output.num_scheduled_tokens.keys() # 本轮被调度过来的req
        cached_req_ids = self.input_batch.req_id_to_index.keys() # inputbatch中缓存的持久化的req ids
        resumed_req_ids = scheduler_output.scheduled_cached_reqs.resumed_req_ids # 本轮中加入的恢复的req
        # NOTE(zhuohan): cached_req_ids and resumed_req_ids are usually disjoint,
        # so `(scheduled_req_ids - resumed_req_ids) == scheduled_req_ids` holds
        # apart from the forced-preemption case in reset_prefix_cache. And in
        # that case we include the resumed_req_ids in the unscheduled set so
        # that they get cleared from the persistent batch before being re-scheduled
        # in the normal resumed request path.
        unscheduled_req_ids = cached_req_ids - (scheduled_req_ids - resumed_req_ids) # 这一批消失的，也就是这一个batch不调度的req的集合
        # NOTE(woosuk): The persistent batch optimization assumes that
        # consecutive batches contain mostly the same requests. If batches
        # have low request overlap (e.g., alternating between two distinct
        # sets of requests), this optimization becomes very inefficient.
        for req_id in unscheduled_req_ids:
            self.input_batch.remove_request(req_id) # 把他们从inputbatch中移除掉






        # ------【投机解码】判断是否走 ngram_gpu 草稿路径，并预分配新请求跟踪列表 ------
        is_ngram_gpu = (
            self.speculative_config is not None
            and self.speculative_config.use_ngram_gpu()
        )
        if is_ngram_gpu:
            ngram_gpu_new_reqs: list[CachedRequestState] = []

        # ------【核心逻辑+投机解码】准备待加入 batch 的请求列表与投机解码的延迟修正队列 ------
        reqs_to_add: list[CachedRequestState] = [] # 新请求，创建持久化状态对象后，先进入这个列表，后面统一加入inputbatch
        deferred_spec_decode_corrections = []




        ########################################
        # 6. 现在开始更新新的req到inputbatch
        ########################################
        # ------【核心逻辑】遍历新调度请求：流式续写更新既有状态，否则新建请求状态 ------
        # Add new requests to the cached states.
        for new_req_data in scheduler_output.scheduled_new_reqs:
            req_id = new_req_data.req_id
            # ------【核心逻辑】流式请求：同 req_id 已在缓存中，走更新路径而非新建 ------
            if req_id in self.requests:
                # For streaming case only.
                req_state = self._update_streaming_request(req_id, new_req_data)
                reqs_to_add.append(req_state)
                continue

            # ------【核心逻辑】提取采样/池化参数，后续据此构造请求状态 ------
            sampling_params = new_req_data.sampling_params
            pooling_params = new_req_data.pooling_params

            # ------【核心逻辑】随机种子采样需为每个请求建独立的随机生成器 ------
            if (
                sampling_params
                and sampling_params.sampling_type == SamplingType.RANDOM_SEED
            ):
                generator = torch.Generator(device=self.device)
                generator.manual_seed(sampling_params.seed)
            else:
                generator = None

            # ------【核心逻辑】pooling 模型：应用任务特定的池化参数更新 ------
            if self.is_pooling_model:
                assert pooling_params is not None
                task = pooling_params.task
                assert task is not None, "You did not set `task` in the API"

                model = cast(VllmModelForPooling, self.get_model())
                to_update = model.pooler.get_pooling_updates(task)
                to_update.apply(pooling_params)



            # ------【核心逻辑】构造请求持久状态对象并登记到缓存字典 ------
            ########################################
            # 7. 为这些新更新的req，创建持久化状态对象，并登记到缓存字典requests
            ########################################
            req_state = CachedRequestState(
                req_id=req_id, # 新的req的id
                prompt_token_ids=new_req_data.prompt_token_ids, #prompt的token列表
                prompt_embeds=new_req_data.prompt_embeds,       #prompt的token向量列表
                prompt_is_token_ids=new_req_data.prompt_is_token_ids,# 混合输入掩码：每个位置是 token id 还是 embed（chat 里混 embedding 时用）
                mm_features=new_req_data.mm_features,
                sampling_params=sampling_params, # 采样参数
                pooling_params=pooling_params,
                generator=generator, # 随机器
                block_ids=new_req_data.block_ids, # 新申请的block_id的列表
                num_computed_tokens=new_req_data.num_computed_tokens, # 已经有kvcache的token数（prefix cache）
                output_token_ids=[], # decode出来的token id列表，为空
                lora_request=new_req_data.lora_request,
            )
            self.requests[req_id] = req_state # 加入这个持久化状态对象
            self.late_interaction_runner.register_request(req_id, pooling_params)

            # ------【核心逻辑】记录请求需要的 prompt logprobs 数量(-1 表示全量 vocab) ------
            # logprobs 要求采样器额外返回「每个位置的概率信息」。它不影响采出什么 token，只影响返回的元数据
            if sampling_params and sampling_params.prompt_logprobs is not None:
                self.num_prompt_logprobs[req_id] = (
                    # 如果没有指定prompt_logprobs， 那么就是词袋大小，指定了就是返回这个每个位置的概率信息
                    self.input_batch.vocab_size
                    if sampling_params.prompt_logprobs == -1 
                    else sampling_params.prompt_logprobs
                )

            # ------【核心逻辑】M-RoPE 模型(如 Qwen2-VL)预先计算多模态 3D 位置 ------
            # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
            if self.uses_mrope:
                self._init_mrope_positions(req_state)

            # ------【核心逻辑】XD-RoPE 模型(如 HunYuan-VL)预先计算扩展维度位置 ------
            # Only relevant for models using XD-RoPE (e.g, HunYuan-VL)
            if self.uses_xdrope_dim > 0:
                self._init_xdrope_positions(req_state)

            # ------【核心逻辑】把新请求加入待添加队列，供后续统一写入常驻 batch ------
            ####################################################
            # 8. 先把这个持久化状态对象，加入一个待添加inputbatch列表，后面统一添加
            ####################################################
            reqs_to_add.append(req_state)
            # ------【投机解码】ngram_gpu 路径跟踪新请求，后续做全张量增量拷贝 ------
            # Track new requests for ngram_gpu full tensor copy
            if is_ngram_gpu:
                ngram_gpu_new_reqs.append(req_state)



        # 到这里，reqs_to_add 列表里面，已经装好了所有new req的持久化状态对象


        # ------【PP+投机解码】记录 PP 末 rank 标志并取出运行/恢复请求与调度好的投机 token ------
        # Update the states of the running/resumed requests.
        is_last_rank = get_pp_group().is_last_rank
        req_data = scheduler_output.scheduled_cached_reqs # 本轮继续的req
        scheduled_spec_tokens = scheduler_output.scheduled_spec_decode_tokens

        # ------【投机解码】ngram_gpu 裁剪前保存调度器分配的草稿长度，供拒绝采样修正 ------
        # Save scheduler-allocated spec lengths before trimming so
        # prev_num_draft_len keeps the optimistic count for rejection correction.
        original_num_spec_per_req: dict[str, int] = {}
        if (
            self.speculative_config is not None
            and self.speculative_config.use_ngram_gpu()
        ):
            for req_id, toks in scheduled_spec_tokens.items():
                original_num_spec_per_req[req_id] = len(toks)
            update_scheduler_for_invalid_drafts(
                self._num_valid_draft_tokens_event,
                self._num_valid_draft_tokens_cpu,
                scheduler_output,
                self.input_batch.req_id_to_index,
            )
        # ------【投机解码+异步 RPC】异步投机解码：先清零上一步草稿长度缓冲 ------
        if self.use_async_spec_decode:
            self.prev_num_draft_tokens.np.fill(0)

        # ------【核心逻辑】遍历运行/恢复请求，逐条提取本步元信息并更新状态 ------
        ############################################################
        # 9. 针对调度器判定，本轮继续的req: 继续的+恢复的
        ############################################################
        for i, req_id in enumerate(req_data.req_ids): # 对于本轮继续的req
            req_state = self.requests[req_id] # 从model_runner的库里取出这个的持久化状态队列
            num_computed_tokens = req_data.num_computed_tokens[i] # 本轮继续的req的已有kvcache的token数量
            new_block_ids = req_data.new_block_ids[i] # 本轮继续的req的新申请的block id列表
            resumed_from_preemption = req_id in req_data.resumed_req_ids # 这个继续的req，是恢复的吗？标志位
            num_output_tokens = req_data.num_output_tokens[i] # 这个继续的req，已经decode出的token数
            req_index = self.input_batch.req_id_to_index.get(req_id) # 获取这个继续req的在inputbatch中的索引

            # ------【投机解码+异步 RPC】异步调度+投机解码：乐观假设草稿全被接受并排队延迟修正 ------
            if req_state.prev_num_draft_len and self.use_async_scheduling:
                # prev_num_draft_len is used in async scheduling mode with
                # spec decode. it indicates if need to update num_computed_tokens
                # of the request. for example:
                # first step: num_computed_tokens = 0, spec_tokens = [],
                # prev_num_draft_len = 0.
                # second step: num_computed_tokens = 100(prompt length),
                # spec_tokens = [a,b], prev_num_draft_len = 0.
                # third step: num_computed_tokens = 100 + 2, spec_tokens = [c,d],
                # prev_num_draft_len = 2.
                # num_computed_tokens in first step and second step doesn't contain
                # the spec tokens length, but in third step it contains the
                # spec tokens length. we only need to update num_computed_tokens
                # when prev_num_draft_len > 0.
                # ------【投机解码+异步 RPC】请求不在 batch 时无法做乐观修正，直接清空草稿长度 ------
                if req_index is None:
                    req_state.prev_num_draft_len = 0
                else:
                    # ------【投机解码+异步 RPC】先用占位 token 撑长输出，接受数待 forward 后回填修正 ------
                    # Optimistically assume all accepted; queue up a correction
                    # to be called after the model forward to preserve async
                    # scheduling. Corrected on GPU in _prepare_inputs.
                    optimistic_num_accepted = req_state.prev_num_draft_len
                    req_state.output_token_ids.extend([-1] * optimistic_num_accepted)

                    deferred_spec_decode_corrections.append(
                        (req_id, optimistic_num_accepted, req_state)
                    )

                    # ------【投机解码+异步 RPC】在上一批索引处记录乐观接受数，供 GPU 侧修正使用 ------
                    prev_req_index = (
                        self.input_batch.prev_req_id_to_index.get(req_id)
                        if self.input_batch.prev_req_id_to_index
                        else None
                    )
                    if prev_req_index is not None:
                        self.prev_num_draft_tokens.np[prev_req_index] = (
                            optimistic_num_accepted
                        )

                    # ------【投机解码】ngram_gpu 需同步累加非投机 token 计数，保持草稿张量对齐 ------
                    if is_ngram_gpu and optimistic_num_accepted > 0:
                        self.input_batch.num_tokens_no_spec[req_index] += (
                            optimistic_num_accepted
                        )


            # ------【核心逻辑】把本步已计算 token 数写回请求状态 ------
            # Update the cached states.
            # 更新model_runner的持久化状态库里面的这个req的状态对象里面的，已经计算的token数
            req_state.num_computed_tokens = num_computed_tokens

            # ------【PP】非末 rank 无直接采样 token，需由调度器下发或走 GPU 广播 ------
            if not is_last_rank:
                if not req_data.new_token_ids:
                    # Async scheduled PP: Sampled tokens propagated via GPU broadcast.
                    new_token_ids: list[int] = []
                else:
                    # Non-async scheduling with PP: The scheduler sends
                    # sampled token ids back because there's no direct communication
                    # between the first-stage worker and the last-stage worker.
                    new_token_ids = req_data.new_token_ids[i]
                    # ------【PP】计算新增的已验证 token 数并追加到输出序列 ------
                    # Add the sampled token(s) from the previous step (if any).
                    # This doesn't include "unverified" tokens like spec tokens.
                    num_new_tokens = (
                        num_computed_tokens + len(new_token_ids) - req_state.num_tokens
                    )
                    if num_new_tokens == 1:
                        # Avoid slicing list in most common case.
                        req_state.output_token_ids.append(new_token_ids[-1])
                    elif num_new_tokens > 0:
                        req_state.output_token_ids.extend(
                            new_token_ids[-num_new_tokens:]
                        )
            # ------【投机解码+核心逻辑】末 rank 截断被乐观撑长/同步失败的输出，对齐真实长度 ------
            elif num_output_tokens < len(req_state.output_token_ids):
                # 如果发现，实际接受的 token 数 < 之前乐观撑长的长度，于是 del 截断掉多塞的部分，对齐真实长度
                '''
                这里是异步调度 + 投机解码

                同步调度 + 投机解码， 采样一步内就知道，接受了几个draft token, 直接append
                异步调度 + 投机解码， 需要先塞 -1 再阶段，这是乐观撑长 + 截断

                异步调度下，为了流水线不断，worker 在「验证结果还没出来」时就得把下一轮的输入准备好。
                而投机解码「接受了几个」必须等 forward 验证完才知道。所以它先乐观假设全部接受（用 -1 占位把 output_token_ids 撑长），等验证结果出来再延迟修正
                '''
                # Some output tokens were discarded due to a sync-KV-load
                # failure, or output_token_ids was inflated by the optimistic
                # extend above (async spec decode). Align the cached state.
                # 下面是截取的动作
                del req_state.output_token_ids[num_output_tokens:]
                if req_index is not None:
                    end_idx = (
                        self.input_batch.num_prompt_tokens[req_index]
                        + num_output_tokens
                    )
                    self.input_batch.num_tokens_no_spec[req_index] = end_idx

            # ------【核心逻辑】非抢占恢复：把新分配的 KV 块追加到现有块表 ------
            # Update the block IDs.
            ############################################## 非抢占的，把新的block id表，更新到 req对应的持久化状态对象里
            if not resumed_from_preemption:
                if new_block_ids is not None:
                    # Append the new blocks to the existing block IDs.
                    for block_ids, new_ids in zip(req_state.block_ids, new_block_ids): # 从持久化状态对象中取出block id列表
                        block_ids.extend(new_ids)
            else:
            ############################################ 抢占恢复的，直接用新块表整体替换旧块表， 因为旧块表废了，被强占就被全部释放block了
                assert req_index is None
                assert new_block_ids is not None
                # The request is resumed from preemption.
                # Replace the existing block IDs with the new ones.
                req_state.block_ids = new_block_ids # 直接更新

            # ------【核心逻辑】请求不在常驻 batch(被抢占/上步未排上)，需重新加入 ------
            #################################################### 这个继续req在inputbatch的索引表上没有索引，重新加入
            if req_index is None:
                # The request is not in the persistent batch.
                # The request was either preempted and resumed later, or was not
                # scheduled in the previous step and needs to be added again.

                # ------【异步 RPC】异步调度下从 all_token_ids 恢复输出 token，保证 input_ids 正确 ------
                if self.use_async_scheduling and num_output_tokens > 0:
                    # We must recover the output token ids for resumed requests in the
                    # async scheduling case, so that correct input_ids are obtained.
                    resumed_token_ids = req_data.all_token_ids[req_id]
                    req_state.output_token_ids = resumed_token_ids[-num_output_tokens:]


                #################################################### 把需重新加入的请求放进待添加队列
                reqs_to_add.append(req_state)
                # Track resumed requests for ngram_gpu full tensor copy
                if is_ngram_gpu:
                    ngram_gpu_new_reqs.append(req_state)
                continue


            # Update the persistent batch.
            ############################################################ 更新inputbatch中，这个继续req的相关信息
            self.input_batch.num_computed_tokens_cpu[req_index] = num_computed_tokens # 已经计算的token数
            if new_block_ids is not None:
                self.input_batch.block_table.append_row(new_block_ids, req_index) # 更新这个req的block id


            # PP优化专属： 非末 rank 不采样
            # ------【PP+chunked prefill】非末 rank 把采样 token 写入 token_ids_cpu 并推进计数 ------
            # For the last rank, we don't need to update the token_ids_cpu
            # because the sampled tokens are already cached.
            if not is_last_rank:
                start_token_index = self.input_batch.num_tokens_no_spec[req_index]
                # For chunked prefill, num_computed_tokens may less
                # than num_tokens_no_spec.
                # Async scheduled PP: no new_token_ids, advance num_tokens_no_spec
                # according to num_computed_tokens.
                end_token_index = max(
                    start_token_index,
                    num_computed_tokens + len(new_token_ids),
                )
                if end_token_index > start_token_index:
                    if new_token_ids:
                        # Add new_token_ids to token_ids_cpu.
                        num_new_tokens = end_token_index - start_token_index
                        tokens_to_append = new_token_ids[-num_new_tokens:]
                        self.input_batch.token_ids_cpu[
                            req_index, start_token_index:end_token_index
                        ] = tokens_to_append
                    self.input_batch.is_token_ids[
                        req_index, start_token_index:end_token_index
                    ] = True
                    self.input_batch.num_tokens_no_spec[req_index] = end_token_index









            # ------【投机解码】把调度好的草稿 token 写入 token_ids_cpu ------
            # Add spec_token_ids to token_ids_cpu.
            self.input_batch.update_req_spec_token_ids(req_state, scheduled_spec_tokens) # 更新本轮的草稿token

            '''
            ngram 是 投机解码里面的一种 草稿token提案方式
            ngram = 连续n个token组成的序列。 gram = 1 token

            投机解码里面，需要 草稿proposer 猜候选token,有两种主流方案：
            EAGLE/Medusa/draft model ------ 用一个小神经网络预测下一批 token, 需要额外模型
            ngram（prompt lookup） ------- 在已生成文本里查找匹配，直接抄，也叫 prompt lookup（提示查找）——本质是「重复模式直接抄答案」。
            MTP = Multi-Token Prediction（多 token 预测），是 DeepSeek-V3 提出的技术，也属于投机解码里的「草稿提案方式」

            MTP ： target 模型原生自带的多个预测模块， 自带，无需额外模型

            在主模型最后一层 hidden state 之上，挂了几个 MTP 模块（每个 = 一个轻量 transformer block + 一个输出头）：

                    第 1 个 MTP 模块 → 预测 t+1 的 token
                    第 2 个 → 预测 t+2 的 token
                    …依次类推
                    一次 forward，主模型算第一个 token 的 logits，
                    MTP 模块复用主模型的 hidden state，逐个往前多猜 N 个 token——这些就是草稿 token，交给验证阶段
                它是自投机（self-speculative）：draft 的头藏在主模型里，forward 主模型时顺带把草稿一起算出来，
                不需要像 EAGLE 那样加载/训练一个独立 draft 模型。


                方案	        草稿来源	                              额外模型
                ngram	        查表匹配重复片段	                        不要
                EAGLE	        独立小 draft 模型（复用 hidden state）	     要（额外训练）
                Medusa	        target 上加多个预测 head	                加 head
                MTP	            target 模型原生自带的多个预测模块	          自带，无需额外模型

            '''
            # ------【投机解码】ngram 裁剪后恢复调度器侧草稿计数，保持 prev_num_draft_len 一致 ------
            # Restore scheduler-side draft count after ngram trimming.
            if original_num_spec_per_req:
                orig = original_num_spec_per_req.get(req_id, 0)
                if orig != req_state.prev_num_draft_len:
                    req_state.prev_num_draft_len = orig




        # ------【核心逻辑】把新增/恢复请求加入常驻 batch(优先填补更小的空槽位) ------
        # Add the new or resumed requests to the persistent batch.
        # The smaller empty indices are filled first.
        ########################################################
        # 10. 我们前面已经更新好了reqs_to_add里面所有的持久化状态对象，现在
        #           把reqs_to_add里面的持久化状态对象，全部添加到inputbatch中
        ########################################################
        for request in reqs_to_add:
            self.input_batch.add_request(request) # 加入inputbatch
            self.input_batch.update_req_spec_token_ids(request, scheduled_spec_tokens) 

        # ------【核心逻辑】压缩 batch 中移除请求留下的空洞，保持索引连续紧凑 ------
        # Condense the batched states if there are gaps left by removed requests
        self.input_batch.condense()
        # ------【核心逻辑】允许 attention 后端按访存模式重排 batch 顺序 ------
        # Allow attention backend to reorder the batch, potentially
        self._may_reorder_batch(scheduler_output)
        # ------【核心逻辑】刷新 batch 元数据，使 pending 更新生效 ------
        # Refresh batch metadata with any pending updates.
        self.input_batch.refresh_metadata()

        # ------【投机解码】batch 稳定后增量更新 ngram_gpu 的全量 token 张量 ------
        # Incrementally update ngram_gpu tensors after batch is stable
        if is_ngram_gpu:
            update_ngram_gpu_tensors_incremental(
                self.input_batch,
                self.token_ids_gpu_tensor,
                self.num_tokens_no_spec_gpu,
                ngram_gpu_new_reqs,
                self.device,
                _pinned_idx_buf=self._ngram_pinned_idx_buf,
                _pinned_val_buf=self._ngram_pinned_val_buf,
            )

        # ------【投机解码+异步 RPC】构造延迟修正闭包：forward 后用真实接受数回填乐观计数 ------
        if deferred_spec_decode_corrections:

            def correct_spec_decode_token_counts():
                # ------【投机解码+异步 RPC】读取有效采样数，逐请求回退乐观多算的接受 token ------
                valid_sampled_token_count = self._get_valid_sampled_token_count()
                if not valid_sampled_token_count:
                    return
                prev_req_id_to_index = self.input_batch.prev_req_id_to_index
                if not prev_req_id_to_index:
                    return
                for (
                    req_id,
                    optimistic_num_accepted,
                    req_state,
                ) in deferred_spec_decode_corrections:
                    prev_req_index = prev_req_id_to_index.get(req_id)
                    if prev_req_index is None:
                        continue
                    num_accepted = valid_sampled_token_count[prev_req_index] - 1
                    correction = optimistic_num_accepted - num_accepted
                    req_state.num_computed_tokens -= correction
                    cur_req_index = self.input_batch.req_id_to_index.get(req_id)
                    if cur_req_index is None:
                        continue
                    self.input_batch.num_computed_tokens_cpu[cur_req_index] -= (
                        correction
                    )
                    if is_ngram_gpu and correction > 0:
                        self.input_batch.num_tokens_no_spec[cur_req_index] -= correction
                        self.num_tokens_no_spec_gpu[cur_req_index] -= correction

            return correct_spec_decode_token_counts
        # ------【核心逻辑】无延迟修正时返回 None，跳过模型执行后的回填步骤 ------
        else:
            return None


















    def _update_states_after_model_execute(
        self, output_token_ids: torch.Tensor, scheduler_output: "SchedulerOutput"
    ) -> None:
        """Update the cached states after model execution.

        This is used for MTP/EAGLE for hybrid models, as in linear attention,
        only the last token's state is kept. In MTP/EAGLE, for draft tokens
        the state are kept util we decide how many tokens are accepted for
        each sequence, and a shifting is done during the next iteration
        based on the number of accepted tokens.
        """
        # ------【投机解码】仅混合架构(线性注意力)的 MTP/EAGLE 需要执行后状态修正 ------
        if not self.speculative_config or not self.model_config.is_hybrid:
            return

        # ------【投机解码】统计每条序列被接受的草稿 token 数(非 -1 位置个数) ------
        # Count the number of accepted tokens for each sequence.
        # Valid tokens are contiguous from position 0, so counting non-(-1)
        # tokens gives us the first -1 position (i.e., number of accepted).
        num_reqs = output_token_ids.size(0)
        self.num_accepted_tokens.gpu[:num_reqs] = (output_token_ids != -1).sum(dim=1)

        # ------【投机解码+内存池/CuMem】align 模式走融合 GPU 后处理，避免 CPU-GPU 同步 ------
        if self.cache_config.mamba_cache_mode == "align":
            # Fused GPU postprocess: state copies + per-request accepted-token
            # update without CPU-GPU sync. The metadata
            # (num_scheduled_tokens, num_draft_tokens, num_computed_tokens) is
            # pre-staged to GPU buffers in _prepare_inputs.
            mamba_utils.postprocess_mamba_align_gpu(
                bufs=self._get_mamba_bufs(),
                num_reqs=num_reqs,
                num_accepted_tokens_gpu=self.num_accepted_tokens.gpu,
                num_accepted_tokens_cpu_tensor=(
                    self.input_batch.num_accepted_tokens_cpu_tensor
                ),
                input_batch=self.input_batch,
                kv_cache_config=self.kv_cache_config,
                forward_context=self.compilation_config.static_forward_context,
                mamba_state_copy_funcs=self.model.get_mamba_state_copy_func(),
            )

            assert self.num_accepted_tokens_event is not None
            # ------【异步 RPC】记录事件标记 GPU 后处理完成，供后续异步等待 ------
            self.num_accepted_tokens_event.record()
        else:
            # ------【异步 RPC】非 align 模式：把接受 token 数异步拷回 CPU 并记录事件 ------
            self.input_batch.num_accepted_tokens_cpu_tensor[:num_reqs].copy_(
                self.num_accepted_tokens.gpu[:num_reqs], non_blocking=True
            )
            assert self.num_accepted_tokens_event is not None
            self.num_accepted_tokens_event.record()

            # ------【核心逻辑】all 模式在 CPU 上做 Mamba 状态后处理(拷贝/对齐) ------
            if self.cache_config.mamba_cache_mode == "all":
                mamba_utils.postprocess_mamba_all(
                    scheduler_output,
                    self.kv_cache_config,
                    self.input_batch,
                    self.requests,
                    self.mamba_state_idx,
                    self.num_spec_tokens,
                    num_reqs,
                )

    def _update_streaming_request(
        self, req_id: str, new_req_data: NewRequestData
    ) -> CachedRequestState:
        """Updates streaming session request from `scheduled_new_reqs`.

        Removes the request from InputBatch (if present), updates the cached
        state, and prepares it for re-addition to the batch.

        NOTE: prompt_token_ids includes intermediate output tokens - tokens
        previously generated but now are input context (part of the prompt).
        """
        # ------【核心逻辑】流式续写：先把该请求从 batch 移除，更新状态后重新加入 ------
        self.input_batch.remove_request(req_id)
        req_state = self.requests[req_id]

        # ------【核心逻辑】用新的 prompt/特征/参数整体替换请求状态(中间输出并入 prompt) ------
        req_state.prompt_token_ids = new_req_data.prompt_token_ids
        req_state.mm_features = new_req_data.mm_features
        req_state.prompt_embeds = new_req_data.prompt_embeds
        req_state.sampling_params = new_req_data.sampling_params
        req_state.pooling_params = new_req_data.pooling_params
        self.late_interaction_runner.register_request(req_id, req_state.pooling_params)
        req_state.block_ids = new_req_data.block_ids
        req_state.num_computed_tokens = new_req_data.num_computed_tokens
        # ------【核心逻辑】重新计算 prompt 长度(token_ids 或 embeds 两种来源) ------
        req_state.num_prompt_tokens = length_from_prompt_token_ids_or_embeds(
            req_state.prompt_token_ids, req_state.prompt_embeds
        )

        # ------【核心逻辑】清空输出 token(已并入 prompt)，开启新一轮生成 ------
        # Clear `output_token_ids` as previous output tokens are now part of
        # `prompt_token_ids`.
        req_state.output_token_ids.clear()

        # ------【核心逻辑】流式更新后重算 M-RoPE 位置，适配新的多模态 prompt ------
        if self.uses_mrope:
            self._init_mrope_positions(req_state)

        return req_state

    def _init_mrope_positions(self, req_state: CachedRequestState):
        # ------【核心逻辑】断言模型支持 M-RoPE 并转为对应接口类型 ------
        model = self.get_model()
        assert supports_mrope(model), "M-RoPE support is not implemented."
        mrope_model = cast(SupportsMRoPE, model)

        # `prompt_embeds` is a passthrough modality (no grid_thw), models'
        # M-RoPE code assumes per-feature grid info, so filter it out. The
        # prompt_embeds positions are treated as text positions for M-RoPE.
        # ------【核心逻辑】过滤掉 prompt_embeds 伪模态，避免无 grid_thw 影响 M-RoPE 计算 ------
        mrope_features = [
            f for f in req_state.mm_features if f.modality != "prompt_embeds"
        ]

        # ------【核心逻辑】token_ids 或 embeds 二选一作为输入长度来源 ------
        if req_state.prompt_token_ids is not None:
            input_tokens = req_state.prompt_token_ids
        elif req_state.prompt_embeds is not None:
            # For embeddings-only inputs, get_mrope_input_positions only
            # needs the sequence length when mm_features is empty (which is
            # the case here since prompt_embeds are filtered out above).
            seq_len = req_state.prompt_embeds.shape[0]
            input_tokens = list(range(seq_len))
        else:
            raise ValueError(
                "M-RoPE requires either prompt_token_ids or prompt_embeds."
            )

        # ------【核心逻辑】调用模型计算 M-RoPE 位置与增量，缓存到请求状态 ------
        req_state.mrope_positions, req_state.mrope_position_delta = (
            mrope_model.get_mrope_input_positions(
                input_tokens,
                mrope_features,
            )
        )

    def _init_xdrope_positions(self, req_state: CachedRequestState):
        # ------【核心逻辑】断言 XD-RoPE 支持并校验 prompt_token_ids 存在 ------
        model = self.get_model()
        xdrope_model = cast(SupportsXDRoPE, model)
        assert req_state.prompt_token_ids is not None, (
            "XD-RoPE requires prompt_token_ids to be available."
        )
        assert supports_xdrope(model), "XD-RoPE support is not implemented."

        # ------【核心逻辑】计算扩展维度 RoPE 位置并缓存到请求状态 ------
        req_state.xdrope_positions = xdrope_model.get_xdrope_input_positions(
            req_state.prompt_token_ids,
            req_state.mm_features,
        )

    def _extract_mm_kwargs(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> BatchedTensorInputs:
        # ------【核心逻辑】仅多模态原始输入模型才需要提取 mm kwargs，否则返回空 ------
        if not scheduler_output or not self.is_multimodal_raw_input_only_model:
            return {}

        # ------【核心逻辑】遍历新请求收集各模态的原始数据项 ------
        mm_kwargs = list[tuple[str, MultiModalKwargsItem]]()
        for req in scheduler_output.scheduled_new_reqs:
            for feature in req.mm_features:
                if feature.data is not None:
                    mm_kwargs.append((feature.modality, feature.data))

        # ------【核心逻辑】把所有模态数据分组打包成 batch 张量并合并返回 ------
        # Input all modalities at once
        mm_kwargs_combined: BatchedTensorInputs = {}
        for _, _, mm_kwargs_batch in group_and_batch_mm_kwargs(
            mm_kwargs,
            device=self.device,
            pin_memory=PIN_MEMORY,
        ):
            mm_kwargs_combined.update(mm_kwargs_batch)

        return mm_kwargs_combined

    def _dummy_mm_kwargs(self, num_seqs: int) -> BatchedTensorInputs:
        # ------【核心逻辑】非多模态原始输入模型返回空 dict ------
        if not self.is_multimodal_raw_input_only_model:
            return {}

        mm_budget = self.mm_budget
        assert mm_budget is not None

        # ------【核心逻辑】无 tower 模态(纯 embedding)时无需 dummy 输入 ------
        if not mm_budget.mm_max_toks_per_item:
            return {}  # No tower modalities (embed-only mode)

        # ------【核心逻辑】取 token 上限最大的模态，生成 dummy batch 供 CUDA graph 捕获 ------
        dummy_modality = mm_budget.get_modality_with_max_tokens()
        return self._get_mm_dummy_batch(dummy_modality, num_seqs)

    def _get_cumsum_and_arange(
        self,
        num_tokens: np.ndarray,
        arange_out: np.ndarray,
        cumsum_dtype: np.dtype | None = None,
    ) -> np.ndarray:
        """Get the cumulative sum and batched arange of the given array.
        E.g., [2, 5, 3] -> [2, 7, 10], arange written to
        arange_out[:10] as [0, 1, 0, 1, 2, 3, 4, 0, 1, 2].
        Equivalent to but faster than:
        np.concatenate([np.arange(n) for n in num_tokens])
        """
        # ------【核心逻辑】求前缀和得到每条请求的 token 偏移(cumsum) ------
        # Step 1. [2, 5, 3] -> [2, 7, 10]
        cu_num_tokens = np.cumsum(num_tokens, dtype=cumsum_dtype)
        total_num_tokens = cu_num_tokens[-1]
        # ------【核心逻辑】用 repeat 展开每条请求的起始偏移，构造批量 arange 的减数 ------
        # Step 2. [2, 7, 10] -> [0, 0, 2, 2, 2, 2, 2, 7, 7, 7]
        cumsums_offsets = np.repeat(cu_num_tokens - num_tokens, num_tokens)
        # ------【核心逻辑】用整体 arange 减去偏移，一次性得到 per-request 局部索引 ------
        # Step 3. [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        np.subtract(
            self.arange_np[:total_num_tokens],
            cumsums_offsets,
            out=arange_out[:total_num_tokens],
        )

        return cu_num_tokens

    def _compute_prev_positions(self, num_reqs: int) -> None:
        """Build prev_positions mapping: current pos -> previous pos (-1 if new).

        Populates self.prev_positions.np[:num_reqs] with the mapping.
        """
        # ------【核心逻辑】构建当前位置到上一批位置的映射，供投机解码复用上一批 token ------
        prev_req_id_to_index = self.input_batch.prev_req_id_to_index
        prev_positions = self.prev_positions.np[:num_reqs]

        # ------【核心逻辑】无上一批映射时全部填 -1，表示都是新请求 ------
        if not prev_req_id_to_index:
            prev_positions.fill(-1)
            return

        # ------【核心逻辑】逐请求查找上一批索引，找不到则置 -1 ------
        for i, req_id in enumerate(self.input_batch.req_ids[:num_reqs]):
            prev_positions[i] = prev_req_id_to_index.get(req_id, -1)





'''
这边来说明一个事情：
    同步调度：
        采样token会同步回传cpu, 写入output_token_ids -> token_ids_cpu -> index_select -> input_ids.cpu. 所以input_ids.cpu里面有老req的真实token
        此时 prev_sampled_token_ids = None, 所以我们直接整条H2D拷走就行

    异步调度：
        异步调度下，老req的采样decode的token，是不会同步回传 cpu的（同步会打断GPU流水线），所以：
            input_ids.cpu 里老req的槽位填的是 占位符（不是真实采样token）
            真实采样token留在GPU的prev_sampled_token_ids 里
            _prepare_input_ids 用scatter把GPU真实token 覆盖到input_ids.gpu 对应槽位。
'''

####################################
# 开始准备本batch的 input ids， 把本轮模型要吃的token id填写到GPU的input_ids.gpu里面
####################################
    def _prepare_input_ids(
        self,
        scheduler_output: "SchedulerOutput",
        num_reqs: int,
        total_num_scheduled_tokens: int,
        cu_num_tokens: np.ndarray,
    ) -> None:
        """Prepare the input IDs for the current batch.

        Carefully handles the `prev_sampled_token_ids` which can be cached
        from the previous engine iteration, in which case those tokens on the
        GPU need to be copied into the corresponding slots into input_ids.

        Uses self.prev_positions[:num_reqs] which maps current pos -> prev pos
        (-1 for new requests).

        输入来源：
            input_ids.cpu           前面 index_select 已填好的本轮所有 token id（从 2D token 表 gather 出来的）
            prev_sampled_token_ids  上一批已经采样的 token（异步调度下还留在 GPU 上）
            _draft_token_ids        投机解码的草稿 token
        """


        ######################################################################
        # 1. 如果是同步调度，GPU侧不会保留旧req的采样结果，每次都会回传cpu， 所以这个为空
        ######################################################################
        if self.input_batch.prev_sampled_token_ids is None:
            # Normal scheduling case
            ####################################
            # 拷贝到GPU
            ####################################
            self.input_ids.copy_to_gpu(total_num_scheduled_tokens)



            # ------【核心逻辑】prompt embeds 模式还需同步拷贝 embeds 与 token 标志 ------
            if self.enable_prompt_embeds:
                self.inputs_embeds.copy_to_gpu(total_num_scheduled_tokens)
                self.is_token_ids.copy_to_gpu(total_num_scheduled_tokens)
            return





        ######################################################################
        # 2. 这里是异步调度，每个req每轮decode出来的token, 会留在GPU侧
        ######################################################################
        # ------【异步 RPC】异步调度路径：复用上一批 GPU 采样 token，避免重复 H2D 拷贝 ------
        # Async scheduling case, where some decode requests from the previous
        # iteration won't have entries in input_ids_cpu and need to be copied
        # on the GPU from prev_sampled_token_ids.
        prev_positions = self.prev_positions.np[:num_reqs]
        scheduled_spec_tokens = scheduler_output.scheduled_spec_decode_tokens
        sample_flattened_indices: list[int] = []
        spec_flattened_indices: list[int] = []
        prev_draft_token_indices: list[int] = []
        prev_indices: list[int] = []
        common_indices_match = True
        max_flattened_index = -1
        total_num_spec_tokens = 0


        '''
        这里来讲一下投机解码的完整闭环：以MTP，自投机为例，每轮猜出3个草稿token
        1. 提案
            基于当前上下文，草稿proposer猜出3个草稿token, [d1, d2, d3]

        2. verify, 大模型一次forward
            大模型并行处理[d1, d2, d3] 这3个位置，一次forward算出每个位置的概率分布
            因为是并行计算，所以验证N个草稿只需要1次前向推理

        3. 逐个对比 大模型的最可能token 和草稿token, 直到第一个不匹配就停止。
                        情况	                结果	                                            本轮输出
                    d1✓ d2✓ d3✗	            接受 2 个，位置3 按大模型分布采样 bonus token	    [d1, d2, d3'] = 3 个
                    d1✓ d2✓ d3✓	            全接受，再额外采样 1 个 bonus token	                [d1, d2, d3, d4'] = 4 个
                    d1✗	                    接受 0 个，只采样 1 个 bonus token	                [d1'] = 1 个

        4. 上下文更新
            新上下文 = 旧上下文 + 本轮接受的token



        异步调度下，本轮的输入，由以下部分：
            1. 上一轮的采样token
            2. 本轮的草稿token
        他们要被一起放入 本轮的输入： input_ids.gpu, 喂给_model_forward

        完整时间线如下：

                第 N-1 轮结束：采样出 token → 存进 prev_sampled_token_ids（GPU 上缓存）
                                                ↓
                第 N 轮 _prepare_inputs：把 prev_sampled_token_ids（上轮输出）
                                        + 新 req 的 prompt + 草稿 token
                                        填进 input_ids.gpu
                                                ↓
                第 N 轮 _model_forward：消费 input_ids.gpu（← 这就是「本轮」的输入）
                                                ↓
                第 N 轮 sample：采样出新 token → 又存进 prev_sampled_token_ids（给第 N+1 轮）

        》》 采样token 和 草稿token 在input_ids里面是连续拼接的，每个decode req 的布局是：
                [anchor(1个采样token)] + [d1, d2, ..., dN(草稿token)]


        3 个 req 拼成的扁平输入序列：

        anchor的意思是上一轮投机解码后，接受序列的最后一个token，也就是接受的草稿token的输出
        草稿token [d1, d2, d3] ，接受d1, d2, 所以大模型采样后的bonus token 为b

        所以上一轮接受的完整序列 [d1, d2, b]， 所以anchor就是b

        [anchor0,     d1,  anchor1, d1, d2,  anchor2, d1, d2]
        0             1      2      3   4      5      6   7
        └─ req0: 1+1 ─┘     └─ req1: 1+2 ─┘  └─ req2: 1+2 ─┘
        每个 req 都是「1 个 anchor + draft_len 个草稿」连续排。这也印证了之前看的 num_scheduled_tokens == draft_len + 1
        '''


        '''
        所以下面的部分，都是异步调度的路径：
        异步路径内部又分了几个子情况
        子情况	                                条件	                                                            动作
        ① 有 prefill 新 req	            num_common_tokens < total_without_spec	                        先整体 H2D 拷贝 input_ids.cpu
        ② 无重叠 req	                num_common_tokens == 0	                                        直接 return（CPU 侧已含全部）
        ③ 批次没重排（常见优化）	       common_indices_match and max_flattened_index == num_common_tokens-1	    单次切片 copy_（GPU→GPU）
        ④ 一般情况	                    其余	                                                用 scatter_ 把采样 token 按索引拷进 input_ids.gpu
        ⑤ 投机解码	                    _draft_token_ids 非空	                                            再 scatter_ 草稿 token

        这段里同时做了「采样 token」和「草稿 token」两件事
        采样 token（2500-2507行）：把 prev_sampled_token_ids（上轮采样输出）scatter 进 input_ids.gpu 的对应位置。

        草稿 token（2512-2521行）：把 _draft_token_ids（本步要验证的草稿）scatter 进 input_ids.gpu 的草稿槽位。


        '''


        # 与前批重叠 = 这个req在上一批里也存在（是持续运行的decode req）, 不是本batch新加入的req


        # ------【核心逻辑+投机解码】为每个 与前批重叠的请求 计算采样/草稿 token 的扁平索引 ------
        # 就是为上一批已经在计算req的，更新他的input_ids
        for cur_index in range(num_reqs):
            prev_index = prev_positions[cur_index]
            if prev_index < 0:
                continue
            prev_indices.append(prev_index)
            req_id = self.input_batch.req_ids[cur_index] # 获取他
            # We need to compute the flattened input_ids index of the
            # last token in each common request.
            draft_len = len(scheduled_spec_tokens.get(req_id, ()))
            total_num_spec_tokens += draft_len # 草稿长度
            flattened_index = cu_num_tokens[cur_index].item() - 1
            # example: cu_num_tokens = [2, 5, 8], draft_tokens = [1, 2, 2]
            # sample_flattened_indices = [0, 2, 5]
            # spec_flattened_indices = [1,   3, 4,    6, 7]
            sample_flattened_indices.append(flattened_index - draft_len) # sample_flattened_indices 本轮扁平input_ids 里的目标位置， anchor放哪里
            spec_flattened_indices.extend(
                range(flattened_index - draft_len + 1, flattened_index + 1)
            )
            start = prev_index * self.prev_num_spec_tokens
            # prev_draft_token_indices is used to find which draft_tokens_id
            # should be copied to input_ids
            # example: prev draft_tokens_id [[1,2], [3,4], [5, 6]]
            # flatten draft_tokens_id [1,2,3,4,5,6]
            # draft_len of each request [1, 2, 1]
            # then prev_draft_token_indices is [0,   2, 3,   4]
            prev_draft_token_indices.extend(range(start, start + draft_len))
            common_indices_match &= prev_index == flattened_index
            max_flattened_index = max(max_flattened_index, flattened_index)




        # ------【核心逻辑】汇总重叠 token 数与去掉投机后的总 token 数 ------
        num_common_tokens = len(sample_flattened_indices)
        total_without_spec = total_num_scheduled_tokens - total_num_spec_tokens
        # ------【核心逻辑】prompt embeds 模式下先刷新 is_token_ids 的 GPU 副本 ------
        if self.enable_prompt_embeds:
            # The multimodal embed path reads is_token_ids.gpu; its .cpu copy is
            # refreshed every step but the async fast paths below only scatter
            # input_ids.gpu, so refresh is_token_ids.gpu here too.
            self.is_token_ids.copy_to_gpu(total_num_scheduled_tokens)
        # ------【核心逻辑】存在新请求时先整体 H2D 拷贝 input_ids，再覆盖重叠部分 ------
        if num_common_tokens < total_without_spec:
            # If not all requests are decodes from the last iteration,
            # we need to copy the input_ids_cpu to the GPU first.
            self.input_ids.copy_to_gpu(total_num_scheduled_tokens) ##############把扁平的input_ids拷贝到gpu
            if self.enable_prompt_embeds:
                self.inputs_embeds.copy_to_gpu(total_num_scheduled_tokens)
        # ------【核心逻辑】与前批无重叠请求时 input_ids.cpu 已含全部，直接返回 ------
        if num_common_tokens == 0:
            # No requests in common with the previous iteration
            # So input_ids.cpu will have all the input ids.
            return
        # ------【核心逻辑】批次未重排的常见情形：单次切片拷贝直接复用上一批采样 token ------
        if common_indices_match and max_flattened_index == (num_common_tokens - 1):
            # Common-case optimization: the batch is unchanged
            # and no reordering happened.
            # The indices are both the same permutation of 0..N-1 so
            # we can copy directly using a single slice.
            self.input_ids.gpu[:num_common_tokens].copy_(
                self.input_batch.prev_sampled_token_ids[:num_common_tokens, 0],
                non_blocking=True,
            )
            return
        # ------【异步 RPC】把索引张量异步上传到 GPU，使 scatter 可以非阻塞执行 ------
        # Upload the index tensors asynchronously so the scatter can be non-blocking.
        sampled_tokens_index_tensor = torch.tensor(
            sample_flattened_indices, dtype=torch.int64, pin_memory=PIN_MEMORY
        ).to(self.device, non_blocking=True)
        prev_common_req_indices_tensor = torch.tensor(
            prev_indices, dtype=torch.int64, pin_memory=PIN_MEMORY
        ).to(self.device, non_blocking=True)
        self.input_ids.gpu.scatter_(
            dim=0,
            index=sampled_tokens_index_tensor,
            src=self.input_batch.prev_sampled_token_ids[
                prev_common_req_indices_tensor, 0
            ],
        )

        # ------【投机解码】草稿 token 为空时跳过后续草稿拷贝 ------
        # Scatter the draft tokens after the sampled tokens are scattered.
        if self._draft_token_ids is None or not spec_flattened_indices:
            return

        # ------【异步 RPC】异步上传草稿 token 的索引张量，准备 GPU 侧 scatter ------
        assert isinstance(self._draft_token_ids, torch.Tensor)
        draft_tokens_index_tensor = torch.tensor(
            spec_flattened_indices, dtype=torch.int64, pin_memory=PIN_MEMORY
        ).to(self.device, non_blocking=True)
        prev_draft_token_indices_tensor = torch.tensor(
            prev_draft_token_indices, dtype=torch.int64, pin_memory=PIN_MEMORY
        ).to(self.device, non_blocking=True)

        # ------【核心逻辑】把草稿 token 转成 int32 以匹配 input_ids 的 dtype ------
        # because input_ids dtype is torch.int32,
        # so convert draft_token_ids to torch.int32 here.
        draft_token_ids = self._draft_token_ids.to(dtype=torch.int32)

        # ------【投机解码】把草稿 token scatter 进 input_ids 对应槽位 ------
        self.input_ids.gpu.scatter_(
            dim=0,
            index=draft_tokens_index_tensor,
            src=draft_token_ids.flatten()[prev_draft_token_indices_tensor],
        )















    def _get_encoder_seq_lens(
        self,
        num_scheduled_tokens: dict[str, int],
        kv_cache_spec: KVCacheSpec,
        num_reqs: int,
        for_cudagraph_capture: bool = False,
    ) -> tuple[torch.Tensor | None, np.ndarray | None]:
        # ------【核心逻辑】非跨注意力(encoder-decoder)规格直接返回 None ------
        if not isinstance(kv_cache_spec, CrossAttentionSpec):
            return None, None

        # ------【CUDA Graph】先把缓冲清零，覆盖图捕获时未被调度的 padding 请求 ------
        # Zero out buffer for padding requests that are not actually scheduled (CGs)
        self.encoder_seq_lens.np[:num_reqs] = 0

        # ------【核心逻辑】遍历本批调度请求，计算各自 encoder 输入长度 ------
        # Build encoder_seq_lens array mapping request indices to
        # encoder lengths for inputs scheduled in this batch
        for req_id in num_scheduled_tokens:
            req_index = self.input_batch.req_id_to_index[req_id]
            req_state = self.requests[req_id]
            # ------【核心逻辑】无多模态特征的请求 encoder 长度为 0 ------
            if req_state.mm_features is None:
                self.encoder_seq_lens.np[req_index] = 0
                continue

            # ------【核心逻辑】累加各特征 mm_position 长度，得到 encoder 应参与注意力的 token 数 ------
            # Get the total number of encoder input tokens for running encoder requests
            # whether encoding is finished or not so that cross-attention knows how
            # many encoder tokens to attend to.
            encoder_input_tokens = sum(
                feature.mm_position.length for feature in req_state.mm_features
            )
            self.encoder_seq_lens.np[req_index] = encoder_input_tokens
        # ------【CUDA Graph】图捕获时用真实最大 encoder 长度，保证 max_seqlen_k 捕获正确 ------
        if for_cudagraph_capture:
            # During CUDA graph capture, we need to use realistic encoder lengths
            # so that max_seqlen_k is captured with the correct value.
            max_encoder_len = getattr(
                self.model_config.hf_config,
                "max_source_positions",
                self.max_encoder_len,
            )
            self.encoder_seq_lens.np[:num_reqs] = max_encoder_len

        # ------【核心逻辑】把 encoder 长度拷到 GPU 并切片出本批实际使用的部分 ------
        self.encoder_seq_lens.copy_to_gpu(num_reqs)
        encoder_seq_lens = self.encoder_seq_lens.gpu[:num_reqs]
        encoder_seq_lens_cpu = self.encoder_seq_lens.np[:num_reqs]

        return encoder_seq_lens, encoder_seq_lens_cpu










################################
# 开始从InputBatch中提取输入缓冲区信息，构造模型输入
################################
    def _prepare_inputs(
        self,
        scheduler_output: "SchedulerOutput",
        num_scheduled_tokens: np.ndarray,
    ) -> tuple[
        torch.Tensor,
        SpecDecodeMetadata | None,
    ]:
        """
        Returns:
            tuple[logits_indices, spec_decode_metadata]
        """
        # ------【核心逻辑】校验本 step 确实有 token 与请求需要调度 ------
        ################################################################
        # 1. 确认本轮确实有调度需求
        ################################################################
        total_num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        assert total_num_scheduled_tokens > 0



        ################################################################
        # 2. 获取目前持久化状态池 inputbatch的运行req数，要有req去跑
        ################################################################
        num_reqs = self.input_batch.num_reqs
        assert num_reqs > 0

        # ------【异步 RPC】先发起 block table 的 H2D 拷贝，与后续 CPU 计算重叠 ------
        # OPTIMIZATION: Start copying the block table first.
        # This way, we can overlap the copy with the following CPU operations.
        ################################################################
        # 3. 发起block table 从cpu->gpu的拷贝，后面让attention的算子读取kvcache的时候用的。因为我们的inputbatch是cpu的实例
        ################################################################
        self.input_batch.block_table.commit_block_table(num_reqs)

        # Get request indices.
        # E.g., [2, 5, 3] -> [0, 0, 1, 1, 1, 1, 1, 2, 2, 2]
        # ------【核心逻辑】用 np.repeat 把每个请求索引展开到每个调度 token ------
        ################################################################
        # 4. 计算req_indices，可知，每个被调度的token属于哪个req id
        ################################################################
        req_indices = np.repeat(self.arange_np[:num_reqs], num_scheduled_tokens)

        # cu_num_tokens: [2, 5, 3] -> [2, 7, 10]
        # self.query_pos.np[:10]: [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        # ------【核心逻辑】计算各请求 token 的累积偏移与请求内位置 ------
        cu_num_tokens = self._get_cumsum_and_arange(
            num_scheduled_tokens, self.query_pos.np
        )

        # ------【核心逻辑】请求内位置 + 已计算 token 数得到绝对位置 ------
        ################################################################
        # 5. 获得positions 缓冲区， 每个token在各自req的绝对位置
        ################################################################
        # Get positions.
        positions_np = (
            self.input_batch.num_computed_tokens_cpu[req_indices]
            + self.query_pos.np[: cu_num_tokens[-1]]
        )

        # ------【核心逻辑】按需计算 M-RoPE / XD-RoPE 旋转位置编码 ------
        # Calculate M-RoPE positions.
        # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
        if self.uses_mrope:
            self._calc_mrope_positions(scheduler_output)

        # Calculate XD-RoPE positions.
        # Only relevant for models using XD-RoPE (e.g, HunYuan-VL)
        if self.uses_xdrope_dim > 0:
            self._calc_xdrope_positions(scheduler_output)



        # Get token indices.
        # E.g., [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        # -> [0, 1, M, M + 1, M + 2, M + 3, M + 4, 2 * M, 2 * M + 1, 2 * M + 2]
        # where M is the max_model_len.
        # ------【核心逻辑】把位置换算成一维 token 索引（请求偏移 × max_len） ------
        ################################################################
        # 6. 计算一维token id列表，获取每个req的token_ids填充max_model_len后拼成一维后的坐标索引
        ################################################################
        token_indices = (
            positions_np + req_indices * self.input_batch.token_ids_cpu.shape[1]
        )
        token_indices_tensor = torch.from_numpy(token_indices) # 转成tensor

        # ------【核心逻辑】用 torch.index_select 高效批量抽取输入 token id ------
        # NOTE(woosuk): We use torch.index_select instead of np.take here
        # because torch.index_select is much faster than np.take for large
        # tensors.
        ################################################################
        # 7. 这边是把二维持久表转化成一维输入, 写入input_ids缓冲区
        ################################################################
        torch.index_select(
            self.input_batch.token_ids_cpu_tensor.flatten(),
            0,
            token_indices_tensor,
            out=self.input_ids.cpu[:total_num_scheduled_tokens], # input_ids
        )
        # ------【核心逻辑】同步抽取 prompt_embeds 场景下的 token/embeds 标志 ------
        if self.enable_prompt_embeds:
            is_token_ids = self.input_batch.is_token_ids_tensor.flatten()
            torch.index_select(
                is_token_ids,
                0,
                token_indices_tensor,
                out=self.is_token_ids.cpu[:total_num_scheduled_tokens],
            )

        # ------【核心逻辑】把请求级 prompt_embeds 逐段拷入预分配张量对应槽位 ------
        # 出现场景是「用户/模型直接提供 embedding 向量
        # 多模态模型， 图像，音频经过各自的encoder得到向量
        # Because we did not pre-allocate a massive prompt_embeds CPU tensor on
        # the InputBatch, we need to fill in the prompt embeds into the expected
        # spots in the GpuModelRunner's pre-allocated prompt_embeds tensor.
        if self.input_batch.req_prompt_embeds:
            output_idx = 0
            for req_idx in range(num_reqs):
                num_sched = num_scheduled_tokens[req_idx]

                # Skip if this request doesn't have embeddings
                if req_idx not in self.input_batch.req_prompt_embeds:
                    output_idx += num_sched
                    continue

                # Skip if no tokens scheduled
                if num_sched <= 0:
                    output_idx += num_sched
                    continue

                req_embeds = self.input_batch.req_prompt_embeds[req_idx]
                start_pos = self.input_batch.num_computed_tokens_cpu[req_idx]

                # Skip if trying to read beyond available embeddings
                if start_pos >= req_embeds.shape[0]:
                    output_idx += num_sched
                    continue

                # Copy available embeddings
                end_pos = start_pos + num_sched
                actual_end = min(end_pos, req_embeds.shape[0])
                actual_num_sched = actual_end - start_pos

                if actual_num_sched > 0:
                    self.inputs_embeds.cpu[
                        output_idx : output_idx + actual_num_sched
                    ].copy_(req_embeds[start_pos:actual_end])

                output_idx += num_sched




        ################################################################
        # 8. 开始准备 query_start_loc
        ################################################################
        '''
        padding = 把「不规则（ragged）的 batch 数据」补齐成「统一/规则的形状」，让 GPU kernel（尤其 CUDA graph）能用固定形状高效执行
        LLM 推理里 batch 天然不规则：不同 req 长度不同、每步调度的 req 数/token 数都在变。
        
        padding 就是把这些变长的东西填到一个固定的、满足 kernel 约束的形状里。
        '''
        # ------【核心逻辑】构造 query_start_loc（含 pad 使非递减）供注意力内核使用 ------
        # Prepare the attention metadata.
        self.query_start_loc.np[0] = 0
        self.query_start_loc.np[1 : num_reqs + 1] = cu_num_tokens
        # Note: pad query_start_loc to be non-decreasing, as kernels
        # like FlashAttention requires that
        # 填充的坐标值，需要不减小。flashattention需要
        self.query_start_loc.np[num_reqs + 1 :].fill(cu_num_tokens[-1])
        self.query_start_loc.copy_to_gpu() # 转移到GPU
        query_start_loc = self.query_start_loc.gpu[: num_reqs + 1]





        # ------【投机解码】乐观假设所有 draft token 都被接受，预计算 seq_lens ------
        # Compute optimistic seq_lens (assumes all draft tokens from previous
        # iteration accepted). Store in optimistic_seq_lens_cpu for use by
        # _build_attention_metadata (max_seq_len) and discard_request_mask.
        # seq_lens (GPU) will be computed later using the same optimistic values.
        torch.add(
            self.input_batch.num_computed_tokens_cpu_tensor[:num_reqs],
            torch.from_numpy(num_scheduled_tokens),
            out=self.optimistic_seq_lens_cpu[:num_reqs],
        )
        self.optimistic_seq_lens_cpu[num_reqs:].fill_(0)

        # ------【投机解码】建立当前请求→上一轮请求的映射，供 GPU 状态回填 ------
        # Build prev_positions mapping: current pos -> prev pos (-1 if new).
        # Used for gathering from previous iteration's GPU tensors.
        prev_req_id_to_index = self.input_batch.prev_req_id_to_index
        self._compute_prev_positions(num_reqs)








        '''
        「是否采样」= 这个 req 本轮是「还在吃 prompt（prefill，不采样）」还是「已经进入 decode（要采样生成新 token）」；
        用「本步算完的 seq_len」对比「目标 token 总数」来判断，没吃满 prompt 的 req 采样结果直接丢弃。
        '''
        # ------【核心逻辑】读取每个请求的总 token 数，用于判定是否采样 ------
        num_tokens = [self.requests[r].num_tokens for r in self.input_batch.req_ids]
        num_tokens_np = np.array(num_tokens, dtype=np.int32)

        # ------【chunked prefill】标记未完整预填充的请求，采样结果会被丢弃 ------
        # Record which requests should not be sampled,
        # so that we could clear the sampled tokens before returning
        self.discard_request_mask.np[:num_reqs] = (
            self.optimistic_seq_lens_cpu[:num_reqs].numpy() < num_tokens_np
        )
        self.discard_request_mask.copy_to_gpu(num_reqs)

        # ------【投机解码+异步 RPC】同步上一轮被接受的 draft token 数（异步调度需按 prev_positions 重映射） ------
        # Sync num_accepted_tokens from CPU (set by
        # _update_states_after_model_execute for hybrid models).
        # Skipped under async scheduling (non-align): the CPU copy races with
        # the in-flight D2H copy and with input-batch row moves.
        needs_cpu_accepted_counts = self.num_accepted_tokens_event is not None and not (
            self.use_async_scheduling and self.cache_config.mamba_cache_mode != "align"
        )
        if needs_cpu_accepted_counts:
            assert self.num_accepted_tokens_event is not None
            self.num_accepted_tokens_event.synchronize()
            # Async mode: condense() reordered indices, use prev_positions mapping
            if self.use_async_scheduling and prev_req_id_to_index:
                prev_idx = self.prev_positions.np[:num_reqs]
                new_mask = prev_idx < 0
                self.num_accepted_tokens.np[:num_reqs] = (
                    self.input_batch.num_accepted_tokens_cpu[
                        np.where(new_mask, 0, prev_idx)
                    ]
                )
                self.num_accepted_tokens.np[:num_reqs][new_mask] = 1
                self.input_batch.num_accepted_tokens_cpu[:num_reqs] = (
                    self.num_accepted_tokens.np[:num_reqs]
                )
            else:
                # Non-async mode: use values directly
                self.num_accepted_tokens.np[:num_reqs] = (
                    self.input_batch.num_accepted_tokens_cpu[:num_reqs]
                )
            self.num_accepted_tokens.np[num_reqs:].fill(1)
            self.num_accepted_tokens.copy_to_gpu()
        else:
            # Default to 1; update_num_computed_tokens_for_batch_change below
            # corrects rows that had drafts from valid_sampled_token_count.
            self.num_accepted_tokens.np.fill(1)
            self.num_accepted_tokens.gpu.fill_(1)

        # ------【投机解码】Mamba 混合模型在 spec-decode 前预处理状态索引 ------
        if self.mamba_prev_last_scheduled_idx is not None:
            mamba_utils.preprocess_mamba_all_specdec(
                scheduler_output,
                self.input_batch,
                self.mamba_state_idx,
                num_reqs,
                self.mamba_prev_last_scheduled_idx,
            )

        # ------【投机解码】异步 spec-decode 在 GPU 上按上一轮实际接受数修正 num_computed_tokens ------
        # Update num_computed_tokens on GPU. In async spec decode,
        # CPU values are optimistic (all drafts accepted). The kernel
        # corrects on GPU using the previous step's
        # valid_sampled_token_count_gpu. Otherwise, just copy from CPU.
        if (
            self.use_async_spec_decode
            and self.valid_sampled_token_count_gpu is not None
            and prev_req_id_to_index
        ):
            self.prev_positions.copy_to_gpu(num_reqs)
            self.prev_num_draft_tokens.copy_to_gpu()
            cpu_values = self.input_batch.num_computed_tokens_cpu_tensor[:num_reqs].to(
                device=self.device, non_blocking=True
            )
            update_num_computed_tokens_for_batch_change(
                self.num_computed_tokens,
                self.num_accepted_tokens.gpu[:num_reqs],
                self.prev_positions.gpu[:num_reqs],
                self.valid_sampled_token_count_gpu,
                self.prev_num_draft_tokens.gpu,
                cpu_values,
            )
        else:
            self.num_computed_tokens[:num_reqs].copy_(
                self.input_batch.num_computed_tokens_cpu_tensor[:num_reqs],
                non_blocking=True,
            )








        ############################################################
        # 开始拷贝到GPU
        ############################################################
        # ------【核心逻辑】把各索引/位置/seq_len 批量拷上 GPU 并填充张量 ------
        self.req_indices.np[:total_num_scheduled_tokens] = req_indices
        self.req_indices.copy_to_gpu(total_num_scheduled_tokens)

        req_indices_gpu = self.req_indices.gpu[:total_num_scheduled_tokens]

        self.query_pos.copy_to_gpu(total_num_scheduled_tokens)

        self.num_scheduled_tokens.np[:num_reqs] = num_scheduled_tokens
        self.num_scheduled_tokens.copy_to_gpu(num_reqs)

        num_scheduled_tokens_gpu = self.num_scheduled_tokens.gpu[:num_reqs]

        self.positions[:total_num_scheduled_tokens] = (
            self.num_computed_tokens[req_indices_gpu].to(torch.int64)
            + self.query_pos.gpu[:total_num_scheduled_tokens]
        )

        self.seq_lens[:num_reqs] = (
            self.num_computed_tokens[:num_reqs] + num_scheduled_tokens_gpu
        )
        self.seq_lens[num_reqs:].fill_(0)





        ############################################################
        # 开始计算每个token的槽位slot,得到他们的物理地址， 这个是用来 写kvcache用的
        ############################################################
        # ------【核心逻辑】按 query_start_loc 与位置计算每个 token 的 KV 槽位映射 ------
        self.input_batch.block_table.compute_slot_mapping(
            num_reqs,
            self.query_start_loc.gpu[: num_reqs + 1],
            self.positions[:total_num_scheduled_tokens],
        )









        # ------【核心逻辑】按调度结果准备输入 id（含投机 token 与多模态掩码） ------
        # Copy the tensors to the GPU.
        ####################################
        # 开始准备模型输入， 开始把本轮的数据，异步下的上一轮的采样token, 都开始往输入缓冲区 input_ids去拷贝了。
        ####################################
        self._prepare_input_ids(
            scheduler_output,
            num_reqs,
            total_num_scheduled_tokens,
            cu_num_tokens,
        )






        # ------【核心逻辑】把 M-RoPE / XD-RoPE 位置张量异步拷上 GPU ------
        if self.uses_mrope:
            # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
            self.mrope_positions.gpu[:, :total_num_scheduled_tokens].copy_(
                self.mrope_positions.cpu[:, :total_num_scheduled_tokens],
                non_blocking=True,
            )
        elif self.uses_xdrope_dim > 0:
            # Only relevant for models using XD-RoPE (e.g, HunYuan-VL)
            self.xdrope_positions.gpu[:, :total_num_scheduled_tokens].copy_(
                self.xdrope_positions.cpu[:, :total_num_scheduled_tokens],
                non_blocking=True,
            )
        # ------【投机解码】异步 spec-decode 下按 GPU/CPU 已计算 token 差修正 RoPE 位置 ------
        if self.use_async_spec_decode and (self.uses_mrope or self.uses_xdrope_dim > 0):
            drift = self.num_computed_tokens[req_indices_gpu].to(
                torch.int64
            ) - self.input_batch.num_computed_tokens_cpu_tensor[req_indices].to(
                device=self.device, dtype=torch.int64, non_blocking=True
            )
            target = self.mrope_positions if self.uses_mrope else self.xdrope_positions
            target.gpu[:, :total_num_scheduled_tokens] += drift

        # ------【投机解码】有 draft token 时构造 spec-decode 元数据，否则按普通解码取 logits 索引 ------
        use_spec_decode = len(scheduler_output.scheduled_spec_decode_tokens) > 0
        if not use_spec_decode:
            # NOTE(woosuk): Due to chunked prefills, the batch may contain
            # partial requests. While we should not sample any token
            # from these partial requests, we do so for simplicity.
            # We will ignore the sampled tokens from the partial requests.
            # TODO: Support prompt logprobs.
            logits_indices = query_start_loc[1:] - 1
            spec_decode_metadata = None
            num_sampled_tokens = np.ones(num_reqs, dtype=np.int32)
        else:
            # ------【投机解码】统计每请求 draft token 数并构造 spec-decode 元数据 ------
            # Get the number of draft tokens for each request.
            # Iterate over the dictionary rather than all requests since not all
            # requests have draft tokens.
            num_draft_tokens = np.zeros(num_reqs, dtype=np.int32)
            # For chunked prefills, use -1 as mask rather than 0, as guided
            # decoding may rollback speculative tokens.
            num_decode_draft_tokens = np.full(num_reqs, -1, dtype=np.int32)
            for (
                req_id,
                draft_token_ids,
            ) in scheduler_output.scheduled_spec_decode_tokens.items():
                req_idx = self.input_batch.req_id_to_index[req_id]
                draft_len = len(draft_token_ids)
                num_draft_tokens[req_idx] = draft_len
                if num_scheduled_tokens[req_idx] == draft_len + 1:
                    num_decode_draft_tokens[req_idx] = draft_len
            # 投机解码的元数据，记录投机解码验证阶段要用到的全部索引 + 草稿token id， 最终喂给rejection_sampler拒绝采样器

            '''
            logits 是 vocab_size的原始分数。但是 logits索引里面的索引，指的是 位置维度

            模型 forward 后，一个 batch 的 hidden states 是：
                    hidden_states: [num_tokens, hidden_size]   ← 每个 token 位置一个向量


            对某一个位置算 logits，才得到 vocab_size 的分数：
                    compute_logits(hidden_state[i]) → [vocab_size]   ← 这一行才是打分


            所以 logits 其实可以看作一个二维结构：
                      位置0    位置1   位置2  ... 位置N-1     ← 位置维度（token 序列）
            logits: [  1行 ] [ 1行 ] [ 1行 ] ...          ← 每行 = vocab_size 个分数（vocab 维度）


            logits_indices 是在位置维度上挑位置，不是挑 vocab。

            因为不需要每个 token 位置都算 logits：
                prefill 中间位置：不采样，不用算 logits
                decode 位置 / 草稿位置 / bonus 位置：要采样，才需要算 logits

            所以用索引选出「需要采样的那些位置」
            '''
            spec_decode_metadata = self._calc_spec_decode_metadata( 
                num_draft_tokens, cu_num_tokens
            )
            logits_indices = spec_decode_metadata.logits_indices
            num_sampled_tokens = num_draft_tokens + 1
            # ------【投机解码+CUDA Graph】为 DECODE-only 图的注意力后端（如 GDN）记录 decode draft 数 ------
            # For DECODE only cuda graph of some attention backends (e.g., GDN).
            self.num_decode_draft_tokens.np[:num_reqs] = num_decode_draft_tokens
            self.num_decode_draft_tokens.np[num_reqs:].fill(-1)
            self.num_decode_draft_tokens.copy_to_gpu()

        # ------【LoRA】热切换本 step 各请求激活的 LoRA 适配器 ------
        # Hot-Swap lora model
        if self.lora_config:
            assert (
                np.sum(num_sampled_tokens)
                <= self.vllm_config.scheduler_config.max_num_batched_tokens
            )
            self.set_active_loras(
                self.input_batch, num_scheduled_tokens, num_sampled_tokens
            )




        '''
        这些位置会从 hidden_states 里挑出来算 logits（hidden_states[logits_indices]），得到 [选中位置数, vocab_size] 的 logits

        logits 是并行一次性算好的


        先厘清：每个位置的 logits 预测「下一个」token
        自回归里，位置 i 的 logits 预测位置 i+1 的 token。输入 [b, d1, d2]（位置 0,1,2）：


        位置:   0(b)       1(d1)      2(d2)      3(还没算)
        输入:   b          d1         d2
        logits: logits@0   logits@1   logits@2
                ↓预测       ↓预测       ↓预测
                位置1=该是d1? 位置2=该是d2? 位置3=采样bonus

                
        logits 是并行算的，不是串行等验证       
                
        三个位置的 logits 在 forward 里一次并行算好：


        forward 一次性并行算：logits@0, logits@1, logits@2 全部出来
        然后验证才串行地去读这些已经算好的 logits：

        读 logits@0 → 验证 d1（对比 argmax == d1？）
        若接受，读 logits@1 → 验证 d2
        若接受，读 logits@2 → 采样 bonus

        ---------------------------------------------------------------------------

        logits@b 的作用是验证 d1，不是直接采样：

        位置	        logits 用途	                        动作
        logits@0 (b)	验证 d1	                    对比 argmax(logits@0) 和 d1，一致→接受
        logits@1 (d1)	验证 d2	                    对比 argmax(logits@1) 和 d2
        logits@2 (d2)	采样 bonus	                从 logits@2 分布抽新 token

    ---------------------------------------------------

        对于贪心采样，就是直接从logits里面选分数最大的token id和草稿token 比较就行。
        如果不是贪心采样，就是采样 拒绝采样算法：

        接受草稿的概率  =  min(1, q(draft)/p(draft))
                q(draft) = 目标模型给这个草稿token 的概率
                p(draft) = 草稿模型给这个草稿token 的概率， 草稿proposer, MTP 头， 提出这个草稿token给他分配的概率
        
        '''

        return (
            logits_indices, # 在input_ids中需要采样输出的token位置
            spec_decode_metadata, # 投机解码的元数据，包含
                '''
                1. 草稿token 本身
                2. 前缀和，定位每个req的范围，cumsum, 累计求和，每组的个数转换成每组的边界位置，因为draft_token_ids 是把所有 req 的草稿拍平成一维数组
                3. logits索引，那些token需要拥有采样分数，用来采样输出token
                
                '''
        )













########################
# 构建注意力元数据
# 把前面算好的一堆 buffer 组装成 attention 后端要的元数据对象
        # 对单 group（无混合 KV、无投机、非 CUDA graph）来说，
        # 就是一次格式转换：把公共 buffer 打包成 CommonAttentionMetadata，再交给 builder.build() 转成后端要的 shape/对象

        # CommonAttentionMetadata 注意力元数据，其实就是：扁平化input_ids的各种统计数据结构 + block_table+slotmapping
########################
    def _build_attention_metadata(
        self,
        num_tokens: int,
        num_reqs: int,
        max_query_len: int,
        num_tokens_padded: int | None = None,
        num_reqs_padded: int | None = None,
        ubatch_slices: UBatchSlices | None = None,
        logits_indices: torch.Tensor | None = None,
        use_spec_decode: bool = False,
        for_cudagraph_capture: bool = False,
        num_scheduled_tokens: dict[str, int] | None = None,
        cascade_attn_prefix_lens: list[list[int]] | None = None,
        slot_mappings: dict[int, torch.Tensor] | None = None,
    ) -> tuple[PerLayerAttnMetadata, CommonAttentionMetadata | None]:
        """
        Returns:
            tuple[attn_metadata, 
            spec_decode_common_attn_metadata] # 公共元数据
        """
        # ------【核心逻辑】无 KV cache 组（如纯 Mamba/无注意力）时直接返回空元数据 ------
        # Attention metadata is not needed for attention free models
        if len(self.kv_cache_config.kv_cache_groups) == 0:
            return {}, None

        # ------【CUDA Graph】用 padded 数回退，保证图捕获维度对齐 ------
        num_tokens_padded = num_tokens_padded or num_tokens
        num_reqs_padded = num_reqs_padded or num_reqs
        assert num_reqs_padded is not None and num_tokens_padded is not None

        # ------【核心逻辑】初始化每层注意力元数据容器（微批则按切片建列表） ------
        attn_metadata: PerLayerAttnMetadata = {} # 每层注意力元数据
        if ubatch_slices is not None:
            attn_metadata = [dict() for _ in range(len(ubatch_slices))]

        # ------【CUDA Graph】捕获时用 max_model_len，运行时用乐观 seq_len 上限 ------
        if for_cudagraph_capture:
            # For some attention backends (e.g. FA) with sliding window models we need
            # to make sure the backend see a max_seq_len that is larger to the sliding
            # window size when capturing to make sure the correct kernel is selected.
            max_seq_len = self.max_model_len
        else:
            max_seq_len = self.optimistic_seq_lens_cpu.numpy()[:num_reqs].max().item()

        kv_cache_groups = self.kv_cache_config.kv_cache_groups

        # ------【CUDA Graph+内存池】取各 KV 组的 block table 并用空块填充 padding 行 ------
        def _get_block_table(kv_cache_gid: int):
            assert num_reqs_padded is not None and num_tokens_padded is not None
            kv_cache_spec = kv_cache_groups[kv_cache_gid].kv_cache_spec
            if isinstance(kv_cache_spec, EncoderOnlyAttentionSpec):
                blk_table_tensor = torch.zeros(
                    (num_reqs_padded, 1),
                    dtype=torch.int32,
                    device=self.device,
                )
            else:
                blk_table = self.input_batch.block_table[kv_cache_gid]
                blk_table_tensor = blk_table.get_device_tensor(num_reqs_padded)

            # Fill unused block table entries with NULL_BLOCK_ID (null block)
            # for CUDAGraph padding. Block 0 is reserved for padding.
            blk_table_tensor[num_reqs:num_reqs_padded].fill_(NULL_BLOCK_ID)
            return blk_table_tensor

        # ------【核心逻辑】缓存第 0 组的 block table 与槽位映射供公共元数据使用 ------
        assert slot_mappings is not None
        block_table_gid_0 = _get_block_table(0)
        slot_mapping_gid_0 = slot_mappings[0]

        # ------【EP/EPLB+异步 RPC】为专家路由异步拷贝注意力槽位映射到私有缓冲，防止下一轮覆盖 ------
        if self.routed_experts_initialized:
            # Copy this step's attention slot_mapping into our private
            # device buffer. The shared ``slot_mappings[attn_gid]`` is
            # owned by the attention block table and will be overwritten
            # by the next ``_prepare_inputs``; we need a stable snapshot
            # because the async D2H may still be in flight on the copy
            # stream when the next step runs.
            slot_mapping_attn = slot_mappings[self.routed_experts_capturer.attn_gid]
            self.routed_experts_slot_mapping_device[:num_tokens].copy_(
                slot_mapping_attn[:num_tokens]
            )

        num_computed_tokens_cpu = self.input_batch.num_computed_tokens_cpu_tensor[
            :num_reqs_padded
        ]
        num_prompt_tokens_cpu = self.input_batch.num_prompt_tokens_cpu_tensor[
            :num_reqs_padded
        ]
        seq_lens_cpu = self.optimistic_seq_lens_cpu[:num_reqs_padded]
        seq_lens_cpu_upper_bound = seq_lens_cpu

        # ------【chunked prefill】读取 CPU 侧长度并判定每个请求是否仍在 prefill 阶段 ------
        # is_prefilling: True if request is still in prefill phase.
        # Used by mamba backends to distinguish actual decodes from
        # short extends.
        is_prefilling = num_computed_tokens_cpu < num_prompt_tokens_cpu
        # Zero out padded rows so stale data from condense() doesn't
        # misclassify padding as prefill in CUDA graph mode.
        is_prefilling[num_reqs:] = False

        # ------【投机解码+异步 RPC】异步模式下以 GPU 张量为准，置 CPU 引用为空 ------
        if self.use_async_spec_decode:
            # GPU tensors are authoritative in async mode.
            seq_lens_cpu = None
            num_computed_tokens_cpu = None

        # Compute mm_prefix bidirectional ranges before building
        # attention metadata so builders handle them during build().
        # By default, ranges exceeding sliding_window are skipped to prevent
        # early tokens from attending across the entire image span. Models that
        # clamp mm_prefix to the sliding window *in-kernel* (e.g. Gemma4, which
        # needs HF's (causal OR blockwise) AND sliding_window on sliding layers)
        # opt out of the skip so the bidirectional range survives for images
        # larger than the window; the kernel then bounds it per-query.
        # ------【核心逻辑】为多模态 prefix 模型预先计算图像双向注意力范围（超滑窗默认跳过） ------
        req_doc_ranges: dict[int, list[tuple[int, int]]] | None = None
        if self.is_mm_prefix_lm:
            req_doc_ranges = {}
            hf_text_config = self.model_config.hf_text_config
            _bidi_sw = getattr(hf_text_config, "sliding_window", None)
            _clamps_in_kernel = getattr(
                self.model, "mm_prefix_clamp_sliding_window", False
            )
            for req_id in self.input_batch.req_ids:
                image_doc_ranges = []
                req_state = self.requests[req_id]
                for mm_feature in req_state.mm_features:
                    if mm_feature.modality == "audio":
                        continue
                    pos_info = mm_feature.mm_position
                    img_doc_range = pos_info.extract_embeds_range()
                    for r in img_doc_range:
                        if (
                            not _clamps_in_kernel
                            and _bidi_sw is not None
                            and (r[1] - r[0] + 1) > _bidi_sw
                        ):
                            continue
                        image_doc_ranges.append(r)
                req_idx = self.input_batch.req_id_to_index[req_id]
                req_doc_ranges[req_idx] = image_doc_ranges

        # ------【核心逻辑】R-SWA 参考滑窗注意力把每请求 prompt 长度传给后端 ------
        # Reference Sliding Window Attention (R-SWA): pass per-request prompt
        # lengths so the attention backend can keep the prefix globally visible.
        # The backend owns the persistent CUDA-graph-safe GPU buffer.
        rswa_prefix_lens = None
        if self.model_config.rswa_window is not None:
            rswa_prefix_lens = num_prompt_tokens_cpu

        # ------【核心逻辑】ReplaySSM 模式下读取各请求 decode 基础偏移 ------
        replayssm_decode_base_cpu = None
        if self.cache_config.use_replayssm:
            replayssm_decode_base_cpu = (
                self.input_batch.replayssm_decode_base_cpu_tensor[:num_reqs_padded]
            )


        #########################################
        # 1. 把公共信息打包成一个对象
        #########################################
        # ------【核心逻辑】汇总公共注意力元数据（位置/seq_len/block table 等） ------
        cm_base = CommonAttentionMetadata(
            query_start_loc=self.query_start_loc.gpu[: num_reqs_padded + 1],
            query_start_loc_cpu=self.query_start_loc.cpu[: num_reqs_padded + 1],

            seq_lens=self.seq_lens[:num_reqs_padded],
            _seq_lens_cpu=seq_lens_cpu,

            _num_computed_tokens_cpu=num_computed_tokens_cpu,

            seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
            replayssm_decode_base_cpu=replayssm_decode_base_cpu,
            num_reqs=num_reqs_padded,
            num_actual_tokens=num_tokens_padded,
            max_query_len=max_query_len,
            max_seq_len=max_seq_len,

            block_table_tensor=block_table_gid_0,
            slot_mapping=slot_mapping_gid_0,

            causal=True,
            is_prefilling=is_prefilling,

            positions=self.positions[:num_tokens_padded],
            mm_req_doc_ranges=req_doc_ranges,
            rswa_prefix_lens=rswa_prefix_lens,
        )

        # ------【TP】DCP 上下文并行下计算本地序列长度并拷上 GPU ------
        if self.dcp_world_size > 1:
            self.dcp_local_seq_lens.cpu[:num_reqs] = get_dcp_local_seq_lens(
                self.optimistic_seq_lens_cpu[:num_reqs],
                self.dcp_world_size,
                self.dcp_rank,
                self.parallel_config.cp_kv_cache_interleave_size,
            )
            self.dcp_local_seq_lens.cpu[num_reqs:].fill_(0)
            self.dcp_local_seq_lens.copy_to_gpu(num_reqs_padded)

            cm_base.dcp_local_seq_lens = self.dcp_local_seq_lens.gpu[:num_reqs_padded]
            cm_base.dcp_local_seq_lens_cpu = self.dcp_local_seq_lens.cpu[
                :num_reqs_padded
            ]

        # ------【前缀缓存】KV 共享 fast prefill 下准备 logits 索引并 pad 到图捕获尺寸 ------
        if logits_indices is not None and self.cache_config.kv_sharing_fast_prefill:
            cm_base.num_logits_indices = logits_indices.size(0)
            cm_base.logits_indices_padded = self._prepare_kv_sharing_fast_prefill(
                logits_indices
            )

        # ------【核心逻辑】按 (KV spec, builder 类型) 缓存注意力元数据，跨混合 KV 组复用 ------
        # Cache attention metadata builds across hybrid KV-cache groups
        # The only thing that changes between different hybrid KV-cache groups when the
        # same metadata builder and KVCacheSpec is the same is the block table, so we
        # can cache the attention metadata builds and just update the block table using
        # `builder.update_block_table` if the builder supports it.
        cached_attn_metadata: dict[
            tuple[KVCacheSpec, type[AttentionMetadataBuilder]], AttentionMetadata
        ] = {}

        def _build_attn_group_metadata(
            kv_cache_gid: int,
            attn_gid: int,
            common_attn_metadata: CommonAttentionMetadata,
            ubid: int | None = None,
        ) -> None:
            # ------【核心逻辑】取当前注意力组对应的 builder 与 (spec, builder) 缓存键 ------
            attn_group = self.attn_groups[kv_cache_gid][attn_gid]
            builder = attn_group.get_metadata_builder(ubid or 0)
            kv_cache_spec = kv_cache_groups[kv_cache_gid].kv_cache_spec
            if isinstance(kv_cache_spec, UniformTypeKVCacheSpecs):
                kv_cache_spec = kv_cache_spec.kv_cache_specs[attn_group.layer_names[0]]
            cache_key = (kv_cache_spec, type(builder))

            # ------【cascade attention】读取该组 cascade 注意力的公共前缀长度（无则 0） ------
            cascade_attn_prefix_len = (
                cascade_attn_prefix_lens[kv_cache_gid][attn_gid]
                if cascade_attn_prefix_lens
                else 0
            )

            # ------【投机解码】为 Mamba2/GDN/线性注意力传入 draft 接受数等额外元数据 ------
            extra_attn_metadata_args = {}
            if use_spec_decode and isinstance(
                builder,
                (
                    Mamba2AttentionMetadataBuilder,
                    GDNAttentionMetadataBuilder,
                    BailingLinearAttentionMetadataBuilder,
                ),
            ):
                assert ubid is None, (
                    "UBatching not supported with GDN or linear attn yet"
                )
                extra_attn_metadata_args = dict(
                    num_accepted_tokens=self.num_accepted_tokens.gpu[:num_reqs_padded],
                    num_decode_draft_tokens_cpu=self.num_decode_draft_tokens.cpu[
                        :num_reqs_padded
                    ],
                )
                if (
                    isinstance(builder, Mamba2AttentionMetadataBuilder)
                    and self.mamba_prev_last_scheduled_idx is not None
                ):
                    extra_attn_metadata_args["prev_last_scheduled_idx"] = (
                        self.mamba_prev_last_scheduled_idx.gpu[:num_reqs_padded]
                    )

            # ------【CUDA Graph+核心逻辑】按场景构建/复用/图捕获注意力元数据并缓存 ------
            if for_cudagraph_capture:
                attn_metadata_i = builder.build_for_cudagraph_capture(
                    common_attn_metadata
                )
            elif (
                cache_key in cached_attn_metadata
                and builder.supports_update_block_table
            ):
                attn_metadata_i = builder.update_block_table(
                    cached_attn_metadata[cache_key],
                    common_attn_metadata.block_table_tensor,
                    common_attn_metadata.slot_mapping,
                )
            else:
                ###########################################
                # 调用backend的builder.build()生成最终元数据，
                # 每个 attn group 有一个 builder（如 FlashAttention、Mamba 等），调用它把 CommonAttentionMetadata 转成后端专属的元数据
                ###########################################
                attn_metadata_i = builder.build(
                    common_prefix_len=cascade_attn_prefix_len,
                    common_attn_metadata=common_attn_metadata,
                    **extra_attn_metadata_args,
                )
                if builder.supports_update_block_table:
                    cached_attn_metadata[cache_key] = attn_metadata_i

            # ------【核心逻辑】把构建好的元数据赋给该组所有层以共享 ------
            if ubid is None:
                assert isinstance(attn_metadata, dict)
                attn_metadata_dict = attn_metadata
            else:
                assert isinstance(attn_metadata, list)
                attn_metadata_dict = attn_metadata[ubid]

            for layer_name in attn_group.layer_names:
                attn_metadata_dict[layer_name] = attn_metadata_i

        # ------【核心逻辑】逐 KV 组准备注意力元数据，同组内层共享 ------
        # Prepare the attention metadata for each KV cache group and make layers
        # in the same group share the same metadata.
        ############################################################
        # 逐 KV 组「浅拷贝 + 按组替换」
        ############################################################
        spec_decode_common_attn_metadata = None
        for kv_cache_gid, kv_cache_group in enumerate(kv_cache_groups):
            cm = copy(cm_base)  # shallow copy

            # ------【核心逻辑】浅拷贝公共元数据，按组更新 encoder seq_lens / block table / slot mapping ------
            # Basically only the encoder seq_lens, block_table and slot_mapping change
            # for each kv_cache_group.
            cm.encoder_seq_lens, cm.encoder_seq_lens_cpu = self._get_encoder_seq_lens(
                num_scheduled_tokens or {},
                kv_cache_group.kv_cache_spec,
                num_reqs_padded,
                for_cudagraph_capture=for_cudagraph_capture,
            )
            if kv_cache_gid > 0:
                cm.block_table_tensor = _get_block_table(kv_cache_gid)# 换这组的 block table
                cm.slot_mapping = slot_mappings[kv_cache_gid]         # 换这组的 slot mapping
                # 因为不同 KV 组只有 block_table 和 slot_mapping 不同，其余公共信息复用。这就是为什么之前 _get_slot_mappings 要按 group 分好。

            # ------【投机解码】把 drafter 所在 KV 组的元数据保存给投机解码使用 ------
            if self.speculative_config and spec_decode_common_attn_metadata is None:
                if isinstance(
                    self.drafter,
                    (
                        EagleProposer,
                        DFlashProposer,
                        Gemma4Proposer,
                        ExtractHiddenStatesProposer,
                    ),
                ):
                    if self.drafter.kv_cache_gid == kv_cache_gid:
                        spec_decode_common_attn_metadata = cm
                else:
                    spec_decode_common_attn_metadata = cm
            # ------【投机解码】为多组 proposer（如 MTP/Gemma4）记录各组的 block table ------
            # Capture per-group block tables for multi-group proposers.
            if self.speculative_config and isinstance(self.drafter, Step3p5MTPProposer):
                self.drafter.set_per_group_attn_metadata(
                    kv_cache_gid, cm.block_table_tensor, cm.slot_mapping
                )
            elif self.speculative_config and isinstance(self.drafter, Gemma4Proposer):
                self.drafter.set_per_group_block_table(
                    kv_cache_gid, cm.block_table_tensor
                )

            # ------【核心逻辑】遍历注意力组构建元数据（微批则先拆分再逐片构建） ------
            for attn_gid in range(len(self.attn_groups[kv_cache_gid])):
                if ubatch_slices is not None:
                    for ubid, _cm in enumerate(split_attn_metadata(ubatch_slices, cm)):
                        _build_attn_group_metadata(kv_cache_gid, attn_gid, _cm, ubid)

                else:
                    _build_attn_group_metadata(kv_cache_gid, attn_gid, cm)

        # ------【投机解码】drafter 用 piecewise 图，需去 padding 的注意力元数据 ------
        if spec_decode_common_attn_metadata is not None and (
            num_reqs != num_reqs_padded or num_tokens != num_tokens_padded
        ):
            # Currently the drafter still only uses piecewise cudagraphs (and modifies
            # the attention metadata in directly), and therefore does not want to use
            # padded attention metadata.
            spec_decode_common_attn_metadata = (
                spec_decode_common_attn_metadata.unpadded(num_tokens, num_reqs)
            )

        return attn_metadata, spec_decode_common_attn_metadata

    def _compute_cascade_attn_prefix_lens(
        self,
        num_scheduled_tokens: np.ndarray,
        num_computed_tokens: np.ndarray,
        num_common_prefix_blocks: list[int],
    ) -> list[list[int]] | None:
        """
        Returns:
            Optional[cascade_attn_prefix_lens]
                cascade_attn_prefix_lens is 2D:
                ``[kv_cache_group_id][attn_group_idx]``,
                None if we should not use cascade attention
        """

        # ------【核心逻辑】初始化各 KV 组的 cascade 前缀长度二维表 ------
        use_cascade_attn = False
        num_kv_cache_groups = len(self.kv_cache_config.kv_cache_groups)
        cascade_attn_prefix_lens: list[list[int]] = [
            [] for _ in range(num_kv_cache_groups)
        ]

        # ------【cascade attention】逐注意力组计算 cascade 公共前缀长度并汇总是否启用 ------
        for kv_cache_gid in range(num_kv_cache_groups):
            for attn_group in self.attn_groups[kv_cache_gid]:
                if isinstance(attn_group.kv_cache_spec, EncoderOnlyAttentionSpec):
                    cascade_attn_prefix_len = 0
                else:
                    # 0 if cascade attention should not be used
                    cascade_attn_prefix_len = self._compute_cascade_attn_prefix_len(
                        num_scheduled_tokens,
                        num_computed_tokens,
                        num_common_prefix_blocks[kv_cache_gid],
                        attn_group.kv_cache_spec,
                        attn_group.get_metadata_builder(),
                    )
                cascade_attn_prefix_lens[kv_cache_gid].append(cascade_attn_prefix_len)
                use_cascade_attn |= cascade_attn_prefix_len > 0

        return cascade_attn_prefix_lens if use_cascade_attn else None

    def _compute_cascade_attn_prefix_len(
        self,
        num_scheduled_tokens: np.ndarray,
        num_computed_tokens: np.ndarray,
        num_common_prefix_blocks: int,
        kv_cache_spec: KVCacheSpec,
        attn_metadata_builder: AttentionMetadataBuilder,
    ) -> int:
        """Compute the length of the common prefix for cascade attention.

        NOTE(woosuk): The common prefix length returned by this function
        represents the length used specifically for cascade attention, not the
        actual number of tokens shared between requests. When cascade attention
        is disabled (use_cascade=False), this function returns 0 even if
        requests share common tokens. Additionally, the common prefix length is
        truncated to a multiple of the block size and may be further truncated
        due to implementation details explained below.

        Args:
            num_scheduled_tokens: Number of tokens scheduled per request.
            num_common_prefix_blocks: Number of shared KV cache blocks.

        Returns:
            int: Length of common prefix in tokens.
        """

        # ------【cascade attention】按公共 KV 块数换算前缀长度，为 0 直接返回 ------
        common_prefix_len = num_common_prefix_blocks * kv_cache_spec.block_size
        if common_prefix_len == 0:
            # Common case.
            return 0

        # NOTE(woosuk): Cascade attention uses two attention kernels: one
        # for the common prefix and the other for the rest. For the first
        # kernel, we concatenate all the query tokens (possibly from
        # different requests) and treat them as if they are from the same
        # request. Then, we use bi-directional attention to process the
        # common prefix in the KV cache. Importantly, this means that the
        # first kernel does not do any masking.

        # Consider the following example:
        # Request 1's input query: [D, E, X]
        # Request 1's kv cache: [A, B, C, D, E, X]
        # Request 1's num_computed_tokens: 3 (i.e., [A, B, C])
        # Request 2's input query: [E, Y]
        # Request 2's kv cache: [A, B, C, D, E, Y]
        # Request 2's num_computed_tokens: 4 (i.e., [A, B, C, D])

        # If we use [A, B, C, D, E] as the common prefix, then the
        # first kernel will compute the bi-directional attention between
        # input query [D, E, X, E, Y] and common prefix [A, B, C, D, E].
        # However, this is wrong because D in Request 1 should not attend to
        # E in the common prefix (i.e., we need masking).
        # To avoid this, [A, B, C, D] should be the common prefix.
        # That is, the common prefix should be capped by the minimum
        # num_computed_tokens among the requests, and plus one to include
        # the first token of the query.

        # In practice, we use [A, B, C] as the common prefix, instead of
        # [A, B, C, D] (i.e., the common prefix is capped by the minimum
        # num_computed_tokens, without plus one).
        # This is because of an implementation detail: We want to always
        # use two kernels for cascade attention. Let's imagine:
        # Request 3's input query: [D]
        # Request 3's kv cache: [A, B, C, D]
        # Request 3's num_computed_tokens: 3 (i.e., [A, B, C])
        # If we use [A, B, C, D] as the common prefix for Request 1-3,
        # then Request 3 will be processed only by the first kernel,
        # and the second kernel will get an empty input. While this is not
        # a fundamental problem, our current implementation does not support
        # this case.
        # ------【cascade attention】前缀按最小已计算 token 截断并对齐到 block 大小 ------
        common_prefix_len = min(common_prefix_len, num_computed_tokens.min())
        # common_prefix_len should be a multiple of the block size.
        common_prefix_len = (
            common_prefix_len // kv_cache_spec.block_size * kv_cache_spec.block_size
        )
        # ------【核心逻辑】识别该 KV spec 是否带滑窗或局部注意力 ------
        use_sliding_window = isinstance(kv_cache_spec, SlidingWindowSpec) or (
            isinstance(kv_cache_spec, FullAttentionSpec)
            and kv_cache_spec.sliding_window is not None
        )
        use_local_attention = isinstance(kv_cache_spec, ChunkedLocalAttentionSpec) or (
            isinstance(kv_cache_spec, FullAttentionSpec)
            and kv_cache_spec.attention_chunk_size is not None
        )
        assert isinstance(kv_cache_spec, AttentionSpec)
        # ------【cascade attention】由 builder 依据前缀长度/头数/SM 数等启发式决定是否启用 ------
        use_cascade = attn_metadata_builder.use_cascade_attention(
            common_prefix_len=common_prefix_len,
            query_lens=num_scheduled_tokens,
            num_query_heads=self.num_query_heads,
            num_kv_heads=kv_cache_spec.num_kv_heads,
            use_alibi=self.use_alibi,
            use_sliding_window=use_sliding_window,
            use_local_attention=use_local_attention,
            num_sms=self.num_sms,
            dcp_world_size=self.dcp_world_size,
        )
        return common_prefix_len if use_cascade else 0

    def _calc_mrope_positions(self, scheduler_output: "SchedulerOutput"):
        # ------【核心逻辑】逐请求读取已计算/待调度/prompt token 数 ------
        mrope_pos_ptr = 0
        for index, req_id in enumerate(self.input_batch.req_ids):
            req = self.requests[req_id]
            assert req.mrope_positions is not None

            num_computed_tokens = self.input_batch.num_computed_tokens_cpu[index]
            num_scheduled_tokens = scheduler_output.num_scheduled_tokens[req_id]
            num_prompt_tokens = length_from_prompt_token_ids_or_embeds(
                req.prompt_token_ids, req.prompt_embeds
            )

            # ------【chunked prefill】把调度 token 拆成 prompt 与 completion 两段 ------
            if num_computed_tokens + num_scheduled_tokens > num_prompt_tokens:
                prompt_part_len = max(0, num_prompt_tokens - num_computed_tokens)
                completion_part_len = max(0, num_scheduled_tokens - prompt_part_len)
            else:
                prompt_part_len = num_scheduled_tokens
                completion_part_len = 0

            assert num_scheduled_tokens == prompt_part_len + completion_part_len

            # ------【核心逻辑】prompt 段的 M-RoPE 位置用预计算结果直接拷贝 ------
            if prompt_part_len > 0:
                # prompt's mrope_positions are pre-computed
                dst_start = mrope_pos_ptr
                dst_end = mrope_pos_ptr + prompt_part_len
                src_start = num_computed_tokens
                src_end = num_computed_tokens + prompt_part_len

                self.mrope_positions.cpu[:, dst_start:dst_end] = req.mrope_positions[
                    :, src_start:src_end
                ]
                mrope_pos_ptr += prompt_part_len

            # ------【核心逻辑】completion 段按增量实时计算 M-RoPE 位置 ------
            if completion_part_len > 0:
                # compute completion's mrope_positions on-the-fly
                dst_start = mrope_pos_ptr
                dst_end = mrope_pos_ptr + completion_part_len

                assert req.mrope_position_delta is not None
                MRotaryEmbedding.get_next_input_positions_tensor(
                    out=self.mrope_positions.np,
                    out_offset=dst_start,
                    mrope_position_delta=req.mrope_position_delta,
                    context_len=num_computed_tokens + prompt_part_len,
                    num_new_tokens=completion_part_len,
                )

                mrope_pos_ptr += completion_part_len

    def _calc_xdrope_positions(self, scheduler_output: "SchedulerOutput"):
        # ------【核心逻辑】逐请求读取已计算/待调度/prompt token 数 ------
        xdrope_pos_ptr = 0
        for index, req_id in enumerate(self.input_batch.req_ids):
            req = self.requests[req_id]
            assert req.xdrope_positions is not None

            num_computed_tokens = self.input_batch.num_computed_tokens_cpu[index]
            num_scheduled_tokens = scheduler_output.num_scheduled_tokens[req_id]
            num_prompt_tokens = length_from_prompt_token_ids_or_embeds(
                req.prompt_token_ids, req.prompt_embeds
            )

            # ------【chunked prefill】把调度 token 拆成 prompt 与 completion 两段 ------
            if num_computed_tokens + num_scheduled_tokens > num_prompt_tokens:
                prompt_part_len = max(0, num_prompt_tokens - num_computed_tokens)
                completion_part_len = max(0, num_scheduled_tokens - prompt_part_len)
            else:
                prompt_part_len = num_scheduled_tokens
                completion_part_len = 0

            assert num_scheduled_tokens == prompt_part_len + completion_part_len

            # ------【核心逻辑】prompt 段的 XD-RoPE 位置用预计算结果直接拷贝 ------
            if prompt_part_len > 0:
                # prompt's xdrope_positions are pre-computed
                dst_start = xdrope_pos_ptr
                dst_end = xdrope_pos_ptr + prompt_part_len
                src_start = num_computed_tokens
                src_end = num_computed_tokens + prompt_part_len

                self.xdrope_positions.cpu[:, dst_start:dst_end] = req.xdrope_positions[
                    :, src_start:src_end
                ]
                xdrope_pos_ptr += prompt_part_len

            # ------【核心逻辑】completion 段按增量实时计算 XD-RoPE 位置 ------
            if completion_part_len > 0:
                # compute completion's xdrope_positions on-the-fly
                dst_start = xdrope_pos_ptr
                dst_end = xdrope_pos_ptr + completion_part_len

                XDRotaryEmbedding.get_next_input_positions_tensor(
                    out=self.xdrope_positions.np,
                    out_offset=dst_start,
                    context_len=num_computed_tokens + prompt_part_len,
                    num_new_tokens=completion_part_len,
                )

                xdrope_pos_ptr += completion_part_len

    def _calc_spec_decode_metadata(
        self,
        num_draft_tokens: np.ndarray,
        cu_num_scheduled_tokens: np.ndarray,
    ) -> SpecDecodeMetadata:
        # Inputs:
        # cu_num_scheduled_tokens:  [  4, 104, 107, 207, 209]
        # num_draft_tokens:         [  3,   0,   2,   0,   1]
        # Outputs:
        # cu_num_draft_tokens:      [  3,   3,   5,   5,   6]
        # logits_indices:           [  0,   1,   2,   3, 103, 104, 105, 106,
        #                            206, 207, 208]
        # target_logits_indices:    [  0,   1,   2,   5,   6,   9]
        # bonus_logits_indices:     [  3,   4,   7,   8,  10]

        # ------【投机解码】用 cumsum+arange+repeat 计算所有采样 token 的 logits 索引 ------
        # Compute the logits indices.
        # [4, 1, 3, 1, 2]
        num_sampled_tokens = num_draft_tokens + 1

        # Step 1.
        # cu_num_sampled_tokens: [4, 5, 8, 9, 11]
        # _arange_scratch[:11]: [0, 1, 2, 3, 0, 0, 1, 2, 0, 0, 1]
        cu_num_sampled_tokens = self._get_cumsum_and_arange(
            num_sampled_tokens, self._arange_scratch, cumsum_dtype=np.int32
        )
        # Step 2. [0, 0, 0, 0, 103, 104, 104, 104, 206, 207, 207]
        logits_indices = np.repeat(
            cu_num_scheduled_tokens - num_sampled_tokens, num_sampled_tokens
        )
        # Step 3. [0, 1, 2, 3, 103, 104, 105, 106, 206, 207, 208]
        logits_indices += self._arange_scratch[: cu_num_sampled_tokens[-1]]

        # ------【投机解码】bonus logits 索引 = 每请求采样段末尾 ------
        # Compute the bonus logits indices.
        bonus_logits_indices = cu_num_sampled_tokens - 1

        # ------【投机解码】计算 draft token 对应的 target logits 索引 ------
        # Compute the draft logits indices.
        # cu_num_draft_tokens: [3, 3, 5, 5, 6]
        # _arange_scratch[:6]: [0, 1, 2, 0, 1, 0]
        cu_num_draft_tokens = self._get_cumsum_and_arange(
            num_draft_tokens, self._arange_scratch, cumsum_dtype=np.int32
        )
        # [0, 0, 0, 5, 5, 9]
        target_logits_indices = np.repeat(
            cu_num_sampled_tokens - num_sampled_tokens, num_draft_tokens
        )
        # [0, 1, 2, 5, 6, 9]
        target_logits_indices += self._arange_scratch[: cu_num_draft_tokens[-1]]

        # ------【异步 RPC】把各类索引异步拷上 GPU，避免同步阻塞 ------
        cu_num_draft_tokens = async_tensor_h2d(cu_num_draft_tokens, device=self.device)
        cu_num_sampled_tokens = async_tensor_h2d(
            cu_num_sampled_tokens, device=self.device
        )
        logits_indices = async_tensor_h2d(logits_indices, device=self.device)
        target_logits_indices = async_tensor_h2d(
            target_logits_indices, device=self.device
        )
        bonus_logits_indices = async_tensor_h2d(
            bonus_logits_indices, device=self.device
        )

        # ------【投机解码】按索引从 input_ids 抽取 draft token id ------
        # Compute the draft token ids.
        # draft_token_indices:      [  1,   2,   3, 105, 106, 208]
        draft_token_ids = self.input_ids.gpu[logits_indices]
        draft_token_ids = draft_token_ids[target_logits_indices + 1]

        return SpecDecodeMetadata(
            draft_token_ids=draft_token_ids,
            num_draft_tokens=num_draft_tokens.tolist(),
            cu_num_draft_tokens=cu_num_draft_tokens,
            cu_num_sampled_tokens=cu_num_sampled_tokens,
            target_logits_indices=target_logits_indices,
            bonus_logits_indices=bonus_logits_indices,
            logits_indices=logits_indices,
        )

    def _prepare_kv_sharing_fast_prefill(
        self,
        logits_indices: torch.Tensor,
    ) -> torch.Tensor:
        # ------【前缀缓存】把 logits 索引写入共享缓冲，pad 位填末索引避免越界 ------
        assert self.kv_sharing_fast_prefill_logits_indices is not None
        num_logits = logits_indices.shape[0]
        assert num_logits > 0
        self.kv_sharing_fast_prefill_logits_indices[:num_logits].copy_(logits_indices)
        # There might have leftover indices in logits_indices[num_logits:]
        # from previous iterations, whose values may be greater than the
        # batch size in the current iteration. To ensure indices are always
        # valid, fill the padded indices with the last index. Broadcast the
        # scalar GPU-side to avoid a D2H sync on `.item()`.
        self.kv_sharing_fast_prefill_logits_indices[num_logits:] = logits_indices[-1]
        # ------【CUDA Graph】按 logits 数量派发 decoder 图并取 pad 后的索引张量 ------
        # Dispatch for the decoder portion of the model.
        _, batch_desc = self.cudagraph_dispatcher.dispatch(
            num_logits, invalid_modes={CUDAGraphMode.FULL}
        )
        num_logits_padded = batch_desc.num_tokens
        logits_indices_padded = self.kv_sharing_fast_prefill_logits_indices[
            :num_logits_padded
        ]
        return logits_indices_padded

    def _batch_mm_inputs_from_scheduler(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> tuple[
        list[str],
        list[tuple[str, MultiModalKwargsItem]],
        list[tuple[str, PlaceholderRange]],
    ]:
        """Batch multimodal inputs from scheduled encoder inputs.

        Args:
            scheduler_output: The scheduler output containing scheduled encoder
                inputs.

        Returns:
            A tuple of (mm_hashes, mm_kwargs, mm_lora_refs) where:
            - mm_hashes: List of multimodal hashes for each item
            - mm_kwargs: List of multimodal kwargs for each item
            - mm_lora_refs: List of (req_id, placeholder_range) for each item
        """
        # ------【核心逻辑】无调度到的编码器输入时直接返回空 ------
        scheduled_encoder_inputs = scheduler_output.scheduled_encoder_inputs
        if not scheduled_encoder_inputs:
            return [], [], []

        # ------【核心逻辑】初始化 mm hash / kwargs / lora 引用三个列表 ------
        mm_hashes = list[str]()
        mm_kwargs = list[tuple[str, MultiModalKwargsItem]]()
        # Multimodal LoRA reference info to map each multimodal item
        # back to its request & position
        mm_lora_refs = list[tuple[str, PlaceholderRange]]()
        # ------【多模态编码器】遍历调度到的编码器输入，收集 hash/kwargs/位置引用 ------
        for req_id, encoder_input_ids in scheduled_encoder_inputs.items():
            req_state = self.requests[req_id]

            for mm_input_id in encoder_input_ids:
                mm_feature = req_state.mm_features[mm_input_id]
                if mm_feature.data is None:
                    continue

                mm_hashes.append(mm_feature.identifier)
                mm_kwargs.append((mm_feature.modality, mm_feature.data))
                mm_lora_refs.append((req_id, mm_feature.mm_position))

        return mm_hashes, mm_kwargs, mm_lora_refs

    def _cache_encoder_output(
        self,
        mm_hash: str,
        output: torch.Tensor,
        ec_manager_metadata: "EncoderCacheManagerMetadata | None",
        free_encoder_mm_hashes: list[str],
    ) -> None:
        """Store an encoder output for later multimodal embedding gather."""
        del ec_manager_metadata, free_encoder_mm_hashes
        # ------【前缀缓存】把编码器输出存入 hash 缓存，并可选同步到外部 connector ------
        self.encoder_cache[mm_hash] = output
        self.maybe_save_ec_to_connector(self.encoder_cache, mm_hash)

    def _execute_mm_encoder(
        self, scheduler_output: "SchedulerOutput"
    ) -> list[torch.Tensor]:
        # ------【多模态编码器】从调度结果批量收集多模态输入，无则返回 ------
        mm_hashes, mm_kwargs, mm_lora_refs = self._batch_mm_inputs_from_scheduler(
            scheduler_output
        )

        if not mm_kwargs:
            return []

        # ------【核心逻辑】prompt_embeds 直通模态直接注入缓存，无需跑编码器 ------
        # `prompt_embeds` is a passthrough modality, the tensor is already in
        # the model embedding space, so no encoder runs. Inject each
        # `prompt_embeds` tensor directly into the encoder cache here so that
        # `_gather_mm_embeddings` can splice it via the standard `is_mm_embed`
        # path.
        pe_indices = [
            i
            for i, (modality, _) in enumerate(mm_kwargs)
            if modality == "prompt_embeds"
        ]
        if pe_indices:
            for i in pe_indices:
                pe_tensor = mm_kwargs[i][1]["embedding"].data
                assert isinstance(pe_tensor, torch.Tensor)

                self._cache_encoder_output(
                    mm_hashes[i],
                    pe_tensor.to(self.device),
                    scheduler_output.ec_manager_metadata,
                    scheduler_output.free_encoder_mm_hashes,
                )
            # Filter out `prompt_embeds` items from mm_kwargs/mm_hashes/mm_lora_refs
            # since they don't require further encoder processing.
            mm_hashes = [h for i, h in enumerate(mm_hashes) if i not in pe_indices]
            mm_kwargs = [k for i, k in enumerate(mm_kwargs) if i not in pe_indices]
            mm_lora_refs = [
                r for i, r in enumerate(mm_lora_refs) if i not in pe_indices
            ]
            if not mm_kwargs:
                return []  # nothing left to encode after filtering out `prompt_embeds`

        # ------【核心逻辑】依据可观测配置决定是否统计编码器耗时 ------
        should_time = bool(
            self.observability_config
            and self.observability_config.enable_mm_processor_stats
            and scheduler_output.scheduled_encoder_inputs
        )

        # Batch mm inputs as much as we can: if a request in the batch has
        # multiple modalities or a different modality than the previous one,
        # we process it separately to preserve item order.
        # FIXME(ywang96): This is a hacky way to deal with multiple modalities
        # in the same batch while still being able to benefit from batching
        # multimodal inputs. The proper solution should be reordering the
        # encoder outputs.
        model = cast(SupportsMultiModal, self.model)

        # ------【LoRA】为编码器独立构建 tower/connector 的 LoRA 映射并激活 ------
        if self.lora_config and self.lora_manager.supports_tower_connector_lora():
            # Build LoRA mappings independently for encoder inputs
            # (encoder batch structure is different from main batch)
            prompt_lora_mapping = []
            token_lora_mapping = []
            lora_requests = set()
            encoder_token_counts = []

            for req_id, pos_info in mm_lora_refs:
                req_idx = self.input_batch.req_id_to_index[req_id]
                lora_id = int(self.input_batch.request_lora_mapping[req_idx])

                # Prefer pos_info.get_num_embeds to count precise MM embedding tokens.
                num_tokens = self.model.get_num_mm_encoder_tokens(  # type: ignore[attr-defined]
                    pos_info.get_num_embeds()
                )
                prompt_lora_mapping.append(lora_id)
                token_lora_mapping.extend([lora_id] * num_tokens)
                encoder_token_counts.append(num_tokens)

                if lora_id > 0:
                    lora_request = self.input_batch.lora_id_to_lora_request.get(lora_id)
                    if lora_request is not None:
                        lora_requests.add(lora_request)

            # Set tower adapter mapping
            tower_mapping = LoRAMapping(
                tuple(token_lora_mapping),
                tuple(prompt_lora_mapping),
                is_prefill=True,
                type=LoRAMappingType.TOWER,
            )
            self.lora_manager.set_active_adapters(lora_requests, tower_mapping)

            # Only set connector mapping if the model actually has a connector.
            # Some multimodal models inherit a stub `get_num_mm_connector_tokens`
            # from `SupportsMultiModal`, which returns None and should not be
            # treated as a signal that connector LoRA is supported.
            mm_mapping = (
                self.model.get_mm_mapping()  # type: ignore[attr-defined]
                if hasattr(self.model, "get_mm_mapping")
                else None
            )
            if (
                mm_mapping is not None
                and mm_mapping.connector
                and hasattr(self.model, "get_num_mm_connector_tokens")
            ):
                post_op_counts = [
                    self.model.get_num_mm_connector_tokens(num_tokens)  # type: ignore[attr-defined]
                    for num_tokens in encoder_token_counts
                ]

                connector_token_mapping = np.repeat(
                    np.array(prompt_lora_mapping, dtype=np.int32),
                    np.array(post_op_counts, dtype=np.int32),
                )
                connector_mapping = LoRAMapping(
                    index_mapping=tuple(connector_token_mapping.tolist()),
                    prompt_mapping=tuple(prompt_lora_mapping),
                    is_prefill=True,
                    type=LoRAMappingType.CONNECTOR,
                )

                self.lora_manager.set_active_adapters(
                    lora_requests,
                    connector_mapping,
                )

        # ------【核心逻辑】初始化编码器输出列表与分组索引 ------
        encoder_outputs: list[torch.Tensor] = []
        # Track the current index in mm_kwargs/mm_lora_refs to map groups to request IDs
        current_item_idx = 0
        # ------【多模态编码器】按 modality 分组批量多模态输入以复用编码器 ------
        for modality, num_items, mm_kwargs_batch in group_and_batch_mm_kwargs(
            mm_kwargs, device=self.device, pin_memory=PIN_MEMORY
        ):
            batch_outputs: MultiModalEmbeddings

            # EVS and dynamic res video related change.
            # (ekhvedchenia): Temporary hack to limit peak memory usage when
            # processing multimodal data. This solves the issue with scheduler
            # putting too many video samples into a single batch. Scheduler
            # uses pruned vision tokens count to compare it versus compute
            # budget which is incorrect (Either input media size or non-pruned
            # output vision tokens count should be considered)
            # dynamic res video for nemotron temporarily uses this hack via
            # requires_sequential_video_encoding
            # because it doesn't yet support video batching.
            # TODO(ywang96): Fix memory profiling to take EVS into account and
            # remove this hack.
            # ------【显存 profiling】多模态剪枝/动态分辨率视频逐条编码以限制显存峰值 ------
            if (
                (
                    self.is_multimodal_pruning_enabled
                    or self.requires_sequential_video_encoding
                )
                and modality == "video"
                and num_items > 1
            ):
                batch_outputs_lst = list[torch.Tensor]()
                for video_idx in range(num_items):
                    video_mm_kwargs_item = mm_kwargs[current_item_idx + video_idx]
                    with self.timed_encoder_operation(
                        should_time, mm_lora_refs, current_item_idx + video_idx, 1
                    ):
                        _, _, micro_batch_mm_inputs = next(
                            group_and_batch_mm_kwargs(
                                [video_mm_kwargs_item],
                                device=self.device,
                                pin_memory=PIN_MEMORY,
                            )
                        )

                        micro_batch_outputs = model.embed_multimodal(
                            **micro_batch_mm_inputs
                        )

                        batch_outputs_lst.extend(micro_batch_outputs)

                batch_outputs = batch_outputs_lst
            else:
                # Run the encoder.
                # `batch_outputs` is either of the following:
                # 1. A tensor of shape (num_items, feature_size, hidden_size)
                # in case feature_size is fixed across all multimodal items.
                # 2. A list or tuple (length: num_items) of tensors,
                # each of shape (feature_size, hidden_size) in case the feature
                # size is dynamic depending on the input multimodal items.

                # ------【多模态编码器+CUDA Graph】优先用编码器 CUDA Graph 执行，否则直接调用 embed_multimodal ------
                with self.timed_encoder_operation(
                    should_time, mm_lora_refs, current_item_idx, num_items
                ):
                    cudagraph_output = None
                    if (
                        self.encoder_cudagraph_manager is not None
                        and self.encoder_cudagraph_manager.supports_modality(modality)
                    ):
                        cudagraph_output = self.encoder_cudagraph_manager.execute(
                            mm_kwargs_batch,
                        )

                    if cudagraph_output is not None:
                        batch_outputs = cudagraph_output
                    else:
                        batch_outputs = model.embed_multimodal(**mm_kwargs_batch)

            # ------【核心逻辑】校验编码器输出条数并累加进结果 ------
            sanity_check_mm_encoder_outputs(batch_outputs, expected_num_items=num_items)
            encoder_outputs.extend(batch_outputs)

            current_item_idx += num_items

        # ------【前缀缓存】按 mm_hash 把编码结果写入缓存供后续 gather 复用 ------
        # Cache the encoder outputs by mm_hash
        for mm_hash, output in zip(mm_hashes, encoder_outputs):
            self._cache_encoder_output(
                mm_hash,
                output,
                scheduler_output.ec_manager_metadata,
                scheduler_output.free_encoder_mm_hashes,
            )
            logger.debug("Finish execute for mm hash %s", mm_hash)

        return encoder_outputs

    def _get_encoder_output_from_cache(self, mm_hash: str) -> torch.Tensor | None:
        """Return a cached encoder output for multimodal
        embedding gather."""
        # ------【前缀缓存】从 hash 缓存读取多模态编码输出 ------
        return self.encoder_cache.get(mm_hash, None)

    def _gather_mm_embeddings(
        self,
        scheduler_output: "SchedulerOutput",
        shift_computed_tokens: int = 0,
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        total_num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens

        # ------【核心逻辑】初始化多模态 embedding 列表与 is_mm_embed 掩码 ------
        mm_embeds = list[torch.Tensor]()
        is_mm_embed = torch.zeros(
            total_num_scheduled_tokens,
            dtype=torch.bool,
            device="cpu",
            pin_memory=PIN_MEMORY,
        )

        # ------【核心逻辑】记录请求起始偏移与是否需重算 RoPE 的标志 ------
        req_start_idx = 0
        should_sync_mrope_positions = False
        should_sync_xdrope_positions = False

        for req_id in self.input_batch.req_ids:
            mm_embeds_req: list[torch.Tensor] = []

            num_scheduled_tokens = scheduler_output.num_scheduled_tokens[req_id]
            req_state = self.requests[req_id]
            num_computed_tokens = req_state.num_computed_tokens + shift_computed_tokens

            mm_features = req_state.mm_features
            # ------【多模态编码器】取该请求在当前窗口内的多模态特征范围 ------
            lo, hi = get_mm_features_in_window(
                mm_features,
                start=num_computed_tokens,
                end=num_computed_tokens + num_scheduled_tokens,
            )
            for i in range(lo, hi):
                mm_feature = mm_features[i]
                pos_info = mm_feature.mm_position
                start_pos = pos_info.offset
                num_encoder_tokens = pos_info.length

                start_idx = max(num_computed_tokens - start_pos, 0)
                end_idx = min(
                    num_computed_tokens - start_pos + num_scheduled_tokens,
                    num_encoder_tokens,
                )
                assert start_idx < end_idx
                # ------【核心逻辑】把 token 窗口换算成编码器输出的 embedding 切片区间 ------
                curr_embeds_start, curr_embeds_end = (
                    pos_info.get_embeds_indices_in_range(start_idx, end_idx)
                )
                # If there are no embeddings in the current range, we skip
                # gathering the embeddings.
                if curr_embeds_start == curr_embeds_end:
                    continue

                mm_hash = mm_feature.identifier
                # ------【前缀缓存】按 hash 取编码输出；drafter 前瞻未编码特征退回 token embedding ------
                encoder_output = self._get_encoder_output_from_cache(mm_hash)
                if encoder_output is None:
                    # A feature starting at/after the processed boundary is only
                    # reached via the drafter's +1 look-ahead and might not be
                    # encoded yet; fall back to the token embedding for drafting.
                    if (
                        start_pos
                        >= req_state.num_computed_tokens + num_scheduled_tokens
                    ):
                        continue
                    raise RuntimeError(f"Encoder cache miss for {mm_hash}.")

                # ------【核心逻辑】按是否 embed 定位切片方式，从编码输出取 embedding ------
                if (is_embed := pos_info.is_embed) is not None:
                    is_embed = is_embed[start_idx:end_idx]
                    mm_embeds_item = encoder_output[curr_embeds_start:curr_embeds_end]
                else:
                    mm_embeds_item = encoder_output[start_idx:end_idx]

                req_start_pos = req_start_idx + start_pos - num_computed_tokens
                # ------【核心逻辑】设置/叠加 is_mm_embed 掩码并标记 embedding 模态 ------
                # OR mask for overlapping mm_features (use_audio_in_video)
                if is_embed is None:
                    is_mm_embed[req_start_pos + start_idx : req_start_pos + end_idx] = (
                        True
                    )
                else:
                    is_mm_embed[
                        req_start_pos + start_idx : req_start_pos + end_idx
                    ] |= is_embed
                set_mm_embedding_modality(mm_embeds_item, mm_feature.modality)
                mm_embeds_req.append(mm_embeds_item)

            # ------【多模态编码器】剪枝场景下重算 M-RoPE 位置并同步回请求状态 ------
            if self.is_multimodal_pruning_enabled and self.uses_mrope:
                assert req_state.mrope_positions is not None
                should_sync_mrope_positions = True
                old_mm_embeds_req = mm_embeds_req
                mm_embeds_req, new_mrope_positions, new_delta = (
                    self.model.recompute_mrope_positions(
                        input_ids=req_state.prompt_token_ids,
                        multimodal_embeddings=mm_embeds_req,
                        mrope_positions=req_state.mrope_positions,
                        num_computed_tokens=req_state.num_computed_tokens,
                    )
                )
                mm_embeds_req = [
                    copy_mm_embedding_modality(src, dst)
                    for src, dst in zip(old_mm_embeds_req, mm_embeds_req)
                ]
                req_state.mrope_positions.copy_(new_mrope_positions)
                req_state.mrope_position_delta = new_delta

            # ------【核心逻辑】累加该请求 embedding 并推进请求起始偏移 ------
            mm_embeds.extend(mm_embeds_req)
            req_start_idx += num_scheduled_tokens

        # ------【核心逻辑】按需重算并同步 M-RoPE/XD-RoPE 位置到 GPU ------
        if should_sync_mrope_positions:
            self._calc_mrope_positions(scheduler_output)
            self.mrope_positions.copy_to_gpu(total_num_scheduled_tokens)

        if should_sync_xdrope_positions:
            self._calc_xdrope_positions(scheduler_output)
            self.xdrope_positions.copy_to_gpu(total_num_scheduled_tokens)

        return mm_embeds, is_mm_embed

    def get_model(self) -> nn.Module:
        # ------【核心逻辑】防御性校验：模型未初始化时访问直接报错 ------
        if not hasattr(self, "model"):
            raise ValueError("Cannot get model before model has been initialized")
        # ------【CUDA Graph】返回前先解包 CUDA Graph / UBatch 包装层，拿到原始 nn.Module ------
        if isinstance(
            self.model, (CUDAGraphWrapper, UBatchWrapper, BreakableCUDAGraphWrapper)
        ):
            # get raw model out of the cudagraph wrapper.
            return self.model.unwrap()
        return self.model

    def get_draft_model(self) -> nn.Module | None:
        # ------【投机解码】无投机解码配置时 drafter 属性不存在，直接返回 None ------
        drafter = getattr(self, "drafter", None)
        if drafter is None:
            return None
        # ------【投机解码+CUDA Graph】取出 draft 模型并解包 CUDA Graph 包装层 ------
        model = getattr(drafter, "model", None)
        if isinstance(
            model, (CUDAGraphWrapper, UBatchWrapper, BreakableCUDAGraphWrapper)
        ):
            return cast(nn.Module, model.unwrap())
        return cast(nn.Module | None, model)

    def get_supported_generation_tasks(self) -> list[GenerationTask]:
        model = self.get_model()
        supported_tasks = list[GenerationTask]()

        # ------【核心逻辑】文本生成模型支持基础 generate 任务 ------
        if is_text_generation_model(model):
            supported_tasks.append("generate")

        # ------【核心逻辑】语音转写模型单独返回 transcription，避免额外任务误报 ------
        if supports_transcription(model):
            if model.supports_transcription_only:
                return ["transcription"]

            supported_tasks.append("transcription")

        # ------【核心逻辑】实时语音模型额外支持 realtime 任务 ------
        if supports_realtime(model):
            supported_tasks.append("realtime")

        return supported_tasks

    def get_supported_pooling_tasks(self) -> list[PoolingTask]:
        model = self.get_model()
        # ------【核心逻辑】非 pooling 模型无 pooling 任务，返回空列表 ------
        if not is_pooling_model(model):
            return []

        # ------【核心逻辑】pooling 任务列表由模型自带 pooler 模块声明 ------
        return list(model.pooler.get_supported_tasks())

    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        tasks = list[SupportedTask]()

        # ------【核心逻辑】按 runner_type 分派到 generation / pooling 两类任务查询 ------
        if self.model_config.runner_type == "generate":
            tasks.extend(self.get_supported_generation_tasks())
        if self.model_config.runner_type == "pooling":
            tasks.extend(self.get_supported_pooling_tasks())

        return tuple(tasks)

    def sync_and_gather_intermediate_tensors(
        self,
        num_tokens: int,
        intermediate_tensors: IntermediateTensors | None,
        sync_self: bool,
    ) -> IntermediateTensors:
        assert self.intermediate_tensors is not None

        tp = self.vllm_config.parallel_config.tensor_parallel_size
        is_rs = is_residual_scattered_for_sp(self.vllm_config, num_tokens)

        # When sequence parallelism is enabled, the "residual" tensor is
        # sharded across TP ranks. All-gather it here because downstream
        # QKV + Attention needs the full residual before the SP split point.
        if sync_self:
            assert intermediate_tensors is not None
            for k, v in intermediate_tensors.items():
                # ------【TP+PP】判断 residual 是否被 SP 切分，需先全量汇聚再写入持久缓冲 ------
                is_scattered = k == "residual" and is_rs
                if is_scattered:
                    local_len = num_tokens // tp
                    # ------【NCCL 通信】all-gather 把各 TP rank 的 residual 分片拼回完整张量 ------
                    v = get_tp_group().all_gather(v[:local_len], dim=0)

                # ------【PP】非阻塞拷贝进持久中间张量，供下一 pipeline stage 读取 ------
                self.intermediate_tensors[k][:num_tokens].copy_(
                    v[:num_tokens], non_blocking=True
                )

        # ------【PP】按 num_tokens 切片构造 IntermediateTensors 作为本 rank 输出 ------
        return IntermediateTensors(
            {k: v[:num_tokens] for k, v in self.intermediate_tensors.items()}
        )

    def eplb_step(self, is_dummy: bool = False, is_profile: bool = False) -> None:
        """
        Step for the EPLB (Expert Parallelism Load Balancing) state.
        """
        # ------【EP/EPLB】未开启 EPLB 或被抑制时跳过，不做负载均衡统计 ------
        if not self.parallel_config.enable_eplb or self.eep_eplb_suppressed:
            return

        assert self.eplb_state is not None
        assert self._moe_model is not None
        # ------【EP/EPLB】推进一次负载均衡状态机，记录专家负载的 balance 指标 ------
        self.eplb_state.step(
            is_dummy,
            is_profile,
            log_stats=self.parallel_config.eplb_config.log_balancedness,
        )

    def setup_eplb_from_mapping(
        self,
        expanded_physical_to_logical: torch.Tensor,
        old_num_physical_experts: int,
    ) -> None:
        assert self._moe_model is not None

        # ------【EP/EPLB】用新旧物理专家映射重建 EPLB 状态，支持专家扩缩容后的重平衡 ------
        self.eplb_state = EplbState.from_mapping(
            model=self._moe_model,
            model_config=self.model_config,
            device=self.device,
            parallel_config=self.parallel_config,
            expanded_physical_to_logical=expanded_physical_to_logical,
            num_valid_physical_experts=old_num_physical_experts,
        )

    def _pool(
        self,
        hidden_states: torch.Tensor,
        num_scheduled_tokens: int,
        num_scheduled_tokens_np: np.ndarray,
        kv_connector_output: KVConnectorOutput | None,
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput:
        num_reqs = self.input_batch.num_reqs
        # ------【核心逻辑】pooling batch 必须整批都是 pooling 请求，逐条校验保证一致性 ------
        assert num_reqs == len(self.input_batch.pooling_params), (
            "Either all or none of the requests in a batch must be pooling request"
        )

        # ------【核心逻辑】截取有效 token 的 hidden states 与 CPU 侧序列长度 ------
        hidden_states = hidden_states[:num_scheduled_tokens]
        seq_lens_cpu = self.optimistic_seq_lens_cpu[:num_reqs]

        # ------【核心逻辑】构建 pooling cursor，标记每条请求在张量里的起止位置 ------
        pooling_metadata = self.input_batch.get_pooling_metadata()
        pooling_metadata.build_pooling_cursor(
            num_scheduled_tokens_np,
            seq_lens_cpu,
            device=hidden_states.device,
            query_start_loc_gpu=self.query_start_loc.gpu[: num_reqs + 1],
        )

        model = cast(VllmModelForPooling, self.model)
        # ------【核心逻辑】调用 pooling 模型的 pooler，把 hidden states 聚合为嵌入向量 ------
        raw_pooler_output: PoolerOutput = model.pooler(
            hidden_states=hidden_states, pooling_metadata=pooling_metadata
        )

        # ------【核心逻辑】用 seq_len 是否等于 prompt_len 判定该请求已完整生成完毕 ------
        finished_mask = [
            seq_len == prompt_len
            for seq_len, prompt_len in zip(seq_lens_cpu, pooling_metadata.prompt_lens)
        ]
        # ------【核心逻辑】对 late-interaction 等特殊 pooler 输出做后处理 ------
        raw_pooler_output = self.late_interaction_runner.postprocess_pooler_output(
            raw_pooler_output=raw_pooler_output,
            pooling_params=pooling_metadata.pooling_params,
            req_ids=self.input_batch.req_ids,
            finished_mask=finished_mask,
        )

        # ------【核心逻辑】构造基础 ModelRunnerOutput，复制 req 映射避免返回后被修改 ------
        model_runner_output = ModelRunnerOutput(
            req_ids=self.input_batch.req_ids.copy(),
            req_id_to_index=self.input_batch.req_id_to_index.copy(),
            kv_connector_output=kv_connector_output,
        )

        # ------【核心逻辑】无 pooler 输出或全部未完成时，同步后返回空 pooler 结果 ------
        if raw_pooler_output is None or not any(finished_mask):
            self._sync_device()
            model_runner_output.pooler_output = [None] * num_reqs
            return model_runner_output

        # ------【异步 RPC】非 CUDA 类设备不支持 stream/event 异步包装，同步拷贝后返回 ------
        if not current_platform.is_cuda_alike():
            # cpu/xpu runners cannot use the CUDA stream/event-based wrapper.
            model_runner_output.pooler_output = _copy_pooler_output_to_cpu(
                raw_pooler_output=raw_pooler_output,
                finished_mask=finished_mask,
            )
            self._sync_device()
            return model_runner_output

        # ------【异步 RPC】CUDA 设备返回异步输出对象，pooler 结果在独立 stream 上非阻塞拷贝 ------
        return AsyncGPUPoolingModelRunnerOutput(
            model_runner_output=model_runner_output,
            raw_pooler_output=raw_pooler_output,
            finished_mask=finished_mask,
            async_output_copy_stream=self._get_or_create_async_output_copy_stream(),
        )

    def _pad_for_sequence_parallelism(self, num_scheduled_tokens: int) -> int:
        # Pad tokens to multiple of tensor_parallel_size when
        # enabled collective fusion for SP
        tp_size = self.vllm_config.parallel_config.tensor_parallel_size
        # ------【TP+chunked prefill】开启 SP 融合且多 TP rank 时，token 数向上取整到 TP 大小倍数 ------
        if self.compilation_config.pass_config.enable_sp and tp_size > 1:
            return round_up(num_scheduled_tokens, tp_size)
        return num_scheduled_tokens

    def _prepare_mm_inputs(
        self, num_tokens: int
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        # ------【核心逻辑】模型需要原始 token id 时才切片 input_ids，否则置空省显存 ------
        if self.model.requires_raw_input_tokens:
            input_ids = self.input_ids.gpu[:num_tokens]
        else:
            input_ids = None

        # ------【核心逻辑】多模态输入统一走 inputs_embeds 路径，切片到当前 batch 长度 ------
        inputs_embeds = self.inputs_embeds.gpu[:num_tokens]
        return input_ids, inputs_embeds






    def _preprocess(
        self,
        scheduler_output: "SchedulerOutput",
        num_input_tokens: int,  # Padded 后该batch的token数
        intermediate_tensors: IntermediateTensors | None = None,
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor,
        IntermediateTensors | None,
        dict[str, Any],
        ECConnectorOutput | None,
    ]:

        # 该batch的真实token 数
        num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens



        # ------【PP】判断本 rank 是否为流水线首段，首段才做 embedding 与多模态编码 ------
        is_first_rank = get_pp_group().is_first_rank
        is_encoder_decoder = self.model_config.is_encoder_decoder

        # Clamp speculative scheduler placeholders (-1) before embedding lookup.
        # ------【投机解码】把调度器预留的 -1 占位 token clamp 成 0，避免 embedding 越界 ------
        if self.speculative_config is not None:
            self.input_ids.gpu[:num_input_tokens].clamp_(min=0)

        # _prepare_inputs may reorder the batch, so we must gather multi
        # modal outputs after that to ensure the correct order
        ec_connector_output = None

        # ------【chunked prefill+多模态】首段且支持多模态时，先跑视觉编码器并收集嵌入 ------
        if self.supports_mm_inputs and is_first_rank and not is_encoder_decoder:
            # Run the multimodal encoder if any.
            with self.maybe_get_ec_connector_output(
                scheduler_output,
                encoder_cache=self.encoder_cache,
            ) as ec_connector_output:
                self._execute_mm_encoder(scheduler_output)
                mm_embeds, is_mm_embed = self._gather_mm_embeddings(scheduler_output)

            # NOTE(woosuk): To unify token ids and soft tokens (vision
            # embeddings), we always use embeddings (rather than token ids)
            # as input to the multimodal model, even when the input is text.
            # ------【多模态+核心逻辑】prompt 里预置 embedding 时，只对 token-id 位置做 embedding ------
            if self.enable_prompt_embeds and self.input_batch.req_prompt_embeds:
                # Some positions carry precomputed prompt_embeds: they are
                # already in self.inputs_embeds and marked is_token_ids=False.
                # Embed only the token-id positions (zeroing the placeholder ids
                # at prompt_embeds positions so the embedding gather cannot read
                # out-of-range ids), and write them back without clobbering the
                # prompt_embeds positions.
                is_token_ids = self.is_token_ids.gpu[:num_scheduled_tokens]
                # ------【多模态+核心逻辑】非 token-id 位置用 0 占位，只对真实 id 做 embedding 避免越界 ------
                safe_input_ids = torch.where(
                    is_token_ids,
                    self.input_ids.gpu[:num_scheduled_tokens],
                    0,
                )
                inputs_embeds_scheduled = self.model.embed_input_ids(
                    safe_input_ids,
                    multimodal_embeddings=mm_embeds,
                    is_multimodal=is_mm_embed,
                )
                # ------【核心逻辑】用 where 只回填 token-id 位置的 embedding，保留预置 embedding 位置 ------
                target = self.inputs_embeds.gpu[:num_scheduled_tokens]
                self.inputs_embeds.gpu[:num_scheduled_tokens] = torch.where(
                    is_token_ids.unsqueeze(-1),
                    inputs_embeds_scheduled,
                    target,
                )
            else:
                # ------【多模态+核心逻辑】无预置 embedding 时，整批做 multimodal embedding ------
                inputs_embeds_scheduled = self.model.embed_input_ids(
                    self.input_ids.gpu[:num_scheduled_tokens],
                    multimodal_embeddings=mm_embeds,
                    is_multimodal=is_mm_embed,
                )

                # TODO(woosuk): Avoid the copy. Optimize.
                self.inputs_embeds.gpu[:num_scheduled_tokens].copy_(
                    inputs_embeds_scheduled
                )

            # ------【核心逻辑】统一走 embedding 输入路径，并合并多模态专用 kwargs ------
            input_ids, inputs_embeds = self._prepare_mm_inputs(num_input_tokens)
            model_kwargs = {
                **self._init_model_kwargs(),
                **self._extract_mm_kwargs(scheduler_output),
            }
        # ------【核心逻辑】纯 prompt_embeds（无视觉）时，仅把 token-id 位置补成 embedding ------
        elif self.enable_prompt_embeds and is_first_rank:
            # Get the input embeddings for the tokens that are not input embeds,
            # then put them into the appropriate positions.
            # TODO(qthequartermasterman): Since even when prompt embeds are
            # enabled, (a) not all requests will use prompt embeds, and (b)
            # after the initial prompt is processed, the rest of the generated
            # tokens will be token ids, it is not desirable to have the
            # embedding layer outside of the CUDA graph all the time. The v0
            # engine avoids this by "double compiling" the CUDA graph, once
            # with input_ids and again with inputs_embeds, for all num_tokens.
            # If a batch only has token ids, then including the embedding layer
            # in the CUDA graph will be more performant (like in the else case
            # below).
            is_token_ids = self.is_token_ids.np[:num_scheduled_tokens]
            token_ids_idx_np = np.nonzero(is_token_ids)[0]
            # Some tokens ids may need to become embeds
            # ------【核心逻辑】只对仍是 token-id 的位置做 embedding，其余保持预置 embedding ------
            if token_ids_idx_np.size > 0:
                # ------【异步 RPC】索引张量异步 H2D 到 GPU，再按索引取 token 做 embedding ------
                token_ids_idx = async_tensor_h2d(token_ids_idx_np, device=self.device)
                token_ids = self.input_ids.gpu[token_ids_idx]
                tokens_to_embeds = self.model.embed_input_ids(input_ids=token_ids)
                self.inputs_embeds.gpu[token_ids_idx] = tokens_to_embeds

            # ------【核心逻辑】全走 embedding 输入，token id 输入置空 ------
            inputs_embeds = self.inputs_embeds.gpu[:num_input_tokens]
            model_kwargs = self._init_model_kwargs()
            input_ids = None
        else:
            # For text-only models, we use token ids as input.
            # While it is possible to use embeddings as input just like the
            # multimodal models, it is not desirable for performance since
            # then the embedding layer is not included in the CUDA graph.
            # ------【CUDA Graph+核心逻辑】纯文本模型直接传 token id，让 embedding 层留在 CUDA Graph 内 ------
            ####################################################################
            # 1. 纯文本llm， input_ids padding cuda graph
            ####################################################################
            input_ids = self.input_ids.gpu[:num_input_tokens] # 延展 [真实token + padding token]
            inputs_embeds = None
            model_kwargs = self._init_model_kwargs()






        # ------【核心逻辑】按模型使用的旋转位置编码类型选取对应 positions 张量 ------
        ##################################
        # position padding， 在 cuda graph
        ##################################
        if self.uses_mrope:
            positions = self.mrope_positions.gpu[:, :num_input_tokens]
        elif self.uses_xdrope_dim > 0:
            positions = self.xdrope_positions.gpu[:, :num_input_tokens]
        else:
            positions = self.positions[:num_input_tokens]# 选出每个token的req内位置
            # ------【核心逻辑】padding 出来的额外 token 位置清零，避免读到脏数据 ------
            if num_input_tokens > num_scheduled_tokens:
                self.positions[num_scheduled_tokens:num_input_tokens].zero_()

        # ------【PP】首段无上游中间张量；非首段从上一 stage 汇聚并同步中间张量 ------
        if is_first_rank:
            intermediate_tensors = None
        else:
            assert intermediate_tensors is not None
            intermediate_tensors = self.sync_and_gather_intermediate_tensors(
                num_input_tokens, intermediate_tensors, True
            )

        # ------【PP+多模态】encoder-decoder 模型运行 encoder，把输出注入 decoder 的 kwargs ------
        if is_encoder_decoder and scheduler_output.scheduled_encoder_inputs:
            # Run the encoder, just like we do with other multimodal inputs.
            # For an encoder-decoder model, our processing here is a bit
            # simpler, because the outputs are just passed to the decoder.
            # We are not doing any prompt replacement. We also will only
            # ever have a single encoder input.
            encoder_outputs = self._execute_mm_encoder(scheduler_output)
            model_kwargs.update({"encoder_outputs": encoder_outputs})

        # ------【核心逻辑】汇总所有前向输入（ids/embeds/positions/中间张量/kwargs）一次性返回 ------
        return (
            input_ids, # 输入的一维token id
            inputs_embeds,
            positions, # 每个token的req内位置
            intermediate_tensors,
            model_kwargs,
            ec_connector_output,
        )








    def _sample(
        self,
        logits: torch.Tensor | None,
        spec_decode_metadata: SpecDecodeMetadata | None,
    ) -> SamplerOutput:
        # Sample the next token and get logprobs if needed.
        sampling_metadata = self.input_batch.sampling_metadata
        # Update output token ids with tokens sampled in last step
        # if async scheduling and required by current sampling params.
        # ------【异步 RPC】先把上一步已采样的 token 同步进 output token id，供 penalty 类采样参数使用 ------
        self.input_batch.update_async_output_token_ids()
        # ------【核心逻辑】无投机解码时直接走普通采样器，从 logits 采样并算 logprobs ------
        if spec_decode_metadata is None:
            return self.sampler(
                logits=logits,
                sampling_metadata=sampling_metadata,
            )

        # Update spec_token_ids with real draft tokens from pre step only when
        # output_token_ids is needed (penalties or bad_words are in use).
        # ------【投机解码+异步 RPC】异步调度下把上一步的真实 draft token 同步进 spec_token_ids ------
        if self.use_async_scheduling and self._draft_token_req_ids is not None:
            draft_token_ids_cpu, _ = self._get_draft_token_ids_cpu()
            self.input_batch.update_async_spec_token_ids(draft_token_ids_cpu)

        # ------【投机解码】取 draft 模型概率，交给拒绝采样器做接受/拒绝校验 ------
        draft_probs = self._get_spec_decode_draft_probs(spec_decode_metadata)
        sampler_output = self.rejection_sampler(
            spec_decode_metadata,
            draft_probs,
            logits,
            sampling_metadata,
        )
        return sampler_output

    def _bookkeeping_sync(
        self,
        scheduler_output: "SchedulerOutput",
        sampler_output: SamplerOutput,
        logits: torch.Tensor | None,
        hidden_states: torch.Tensor,
        num_scheduled_tokens: int,
    ) -> tuple[
        dict[str, int],
        LogprobsLists | None,
        list[list[int]],
        dict[str, LogprobsTensors | None],
        list[str],
        dict[str, int],
        list[int],
    ]:
        # ------【核心逻辑】按需统计 logits 中的 NaN 数量（调试/监控用，默认关闭） ------
        num_nans_in_logits = {}
        if envs.VLLM_COMPUTE_NANS_IN_LOGITS:
            num_nans_in_logits = self._get_nans_in_logits(logits)

        num_reqs = self.input_batch.num_reqs
        # ------【chunked prefill】挑出需要丢弃本步采样结果的请求（非末段 prefill chunk） ------
        discard_sampled_tokens_req_indices = np.nonzero(
            self.discard_request_mask.np[:num_reqs]
        )[0]
        # ------【投机解码】把被丢弃请求的 ngram 生成器 offset 回退，撤销上一步投机产出的 4 个 token ------
        for i in discard_sampled_tokens_req_indices:
            gen = self.input_batch.generators.get(int(i))
            if gen is not None:
                gen.set_offset(gen.get_offset() - 4)

        # Copy some objects so they don't get modified after returning.
        # This is important when using async scheduling.
        # ------【异步 RPC】复制 req 映射快照，避免返回后被异步调度并发修改 ------
        req_ids_output_copy = self.input_batch.req_ids.copy()
        req_id_to_index_output_copy = self.input_batch.req_id_to_index.copy()

        num_sampled_tokens = sampler_output.sampled_token_ids.shape[0]
        sampled_token_ids = sampler_output.sampled_token_ids
        logprobs_tensors = sampler_output.logprobs_tensors
        invalid_req_indices = []
        logprobs_lists = None
        # ------【异步 RPC】同步调度分支：把采样结果立即拷回 CPU 并解析成 list ------
        if not self.use_async_scheduling:
            # Sync scheduling: issue routed experts D2H into the pinned
            # CPU buffer BEFORE ``_to_list`` below. ``_to_list`` does
            # ``event.synchronize()`` on the async copy stream which
            # waits for every D2H queued on the default stream since
            # the last sync, so this enqueue is naturally covered
            # without requiring its own synchronize.
            # ------【EP/EPLB+异步 RPC】先把路由专家结果 D2H 拷进 pinned 缓冲，复用 _to_list 的同步点 ------
            if self.routed_experts_initialized:
                buf = self.routed_experts_capturer.get_device_buffer()
                total = scheduler_output.total_num_scheduled_tokens
                self.routed_experts_cpu[:total].copy_(buf[:total], non_blocking=True)
                self.routed_experts_slot_mapping_cpu[:total].copy_(
                    self.routed_experts_slot_mapping_device[:total],
                    non_blocking=True,
                )

            # Get the valid generated tokens.
            max_gen_len = sampled_token_ids.shape[-1]
            # ------【投机解码】单 token 采样时直接转 list，再清除被丢弃请求的采样结果 ------
            if max_gen_len == 1:
                # No spec decode tokens.
                valid_sampled_token_ids = self._to_list(sampled_token_ids)
                # Mask out the sampled tokens that should not be sampled.
                for i in discard_sampled_tokens_req_indices:
                    valid_sampled_token_ids[int(i)].clear()

                if logprobs_tensors is not None:
                    logprobs_lists = logprobs_tensors.tolists()
            else:
                # Includes spec decode tokens.
                # ------【投机解码】多候选时用拒绝采样解析出被接受/丢弃的 token 序列与 logprobs ------
                valid_sampled_token_ids, logprobs_lists = RejectionSampler.parse_output(
                    sampled_token_ids,
                    self.input_batch.vocab_size,
                    discard_sampled_tokens_req_indices,
                    logprobs_tensors=logprobs_tensors,
                )
        else:
            # ------【异步 RPC】异步调度分支：不拷回 CPU，直接记录无效请求集合 ------
            valid_sampled_token_ids = []
            invalid_req_indices = discard_sampled_tokens_req_indices.tolist()
            invalid_req_indices_set = set(invalid_req_indices)

            # Cache the sampled tokens on the GPU and avoid CPU sync.
            # These will be copied into input_ids in the next step
            # when preparing inputs.
            # With spec decoding, this is done in propose_draft_token_ids().
            # ------【异步 RPC】把采样 token 缓存在 GPU 上，下一步直接消费，避免一次 CPU 同步 ------
            if self.input_batch.prev_sampled_token_ids is None:
                assert sampled_token_ids.shape[-1] == 1
                self.input_batch.prev_sampled_token_ids = sampled_token_ids
            # ------【异步 RPC】同步记录上一步的 req_id 到索引映射，供下一步 input 准备时反查 ------
            self.input_batch.prev_req_id_to_index = {
                req_id: i
                for i, req_id in enumerate(self.input_batch.req_ids)
                if i not in invalid_req_indices_set
            }

        # Cache the sampled tokens in the model runner, so that the scheduler
        # doesn't need to send them back.
        # NOTE(woosuk): As an exception, when using PP, the scheduler sends
        # the sampled tokens back, because there's no direct communication
        # between the first-stage worker and the last-stage worker.
        # ------【核心逻辑】把采样出的 token 逐条写回输入 batch 与请求状态，供调度器复用免回传 ------
        req_ids = self.input_batch.req_ids
        for req_idx in range(num_sampled_tokens):
            # ------【异步 RPC】异步分支用 -1 占位有效请求，无效请求置 None 跳过 ------
            if self.use_async_scheduling:
                sampled_ids = [-1] if req_idx not in invalid_req_indices_set else None
            else:
                sampled_ids = valid_sampled_token_ids[req_idx]

            num_sampled_ids: int = len(sampled_ids) if sampled_ids else 0

            if not sampled_ids:
                continue

            # ------【核心逻辑】定位该请求已生成的 token 区间，并校验不超过 max_model_len ------
            start_idx = self.input_batch.num_tokens_no_spec[req_idx]
            end_idx = start_idx + num_sampled_ids
            assert end_idx <= self.max_model_len, (
                "Sampled token IDs exceed the max model length. "
                f"Total number of tokens: {end_idx} > max_model_len: "
                f"{self.max_model_len}"
            )

            # ------【核心逻辑】同步更新 CPU 侧 token_ids 与 is_token_ids 标记，推进 num_tokens_no_spec ------
            self.input_batch.token_ids_cpu[req_idx, start_idx:end_idx] = sampled_ids
            self.input_batch.is_token_ids[req_idx, start_idx:end_idx] = True
            self.input_batch.num_tokens_no_spec[req_idx] = end_idx

            # ------【核心逻辑】把新 token 追加到请求级输出序列，供最终返回给客户端 ------
            req_id = req_ids[req_idx]
            req_state = self.requests[req_id]
            req_state.output_token_ids.extend(sampled_ids)

        # Compute prompt logprobs if needed.
        # ------【核心逻辑】按需从 hidden states 计算 prompt 段的 logprobs 字典 ------
        prompt_logprobs_dict = self._get_prompt_logprobs_dict(
            hidden_states[:num_scheduled_tokens],
            scheduler_output.num_scheduled_tokens,
        )

        return (
            num_nans_in_logits,
            logprobs_lists,
            valid_sampled_token_ids,
            prompt_logprobs_dict,
            req_ids_output_copy,
            req_id_to_index_output_copy,
            invalid_req_indices,
        )

    @contextmanager
    def synchronize_input_prep(self):
        # ------【核心逻辑】未启用异步 input 准备事件时，直接透传不附加任何同步 ------
        if self.prepare_inputs_event is None:
            yield
            return

        # Ensure prior step has finished with reused CPU tensors.
        # This is required in the async scheduling case because
        # the CPU->GPU transfer happens async.
        # ------【异步 RPC】进入前等待上一轮 input 准备完成，避免 CPU 缓冲被并发复用 ------
        self.prepare_inputs_event.synchronize()
        try:
            yield
        finally:
            # ------【异步 RPC】退出时记录本次事件，供下一轮同步点等待 ------
            self.prepare_inputs_event.record()





    def _model_forward(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **model_kwargs: dict[str, Any],
    ) -> Any:
        """Helper method to call the model forward pass.

        This method can be overridden by subclasses for model execution.
        Motivation: We can inspect only this method versus
        the whole execute_model, which has additional logic.

        Args:
            input_ids: Input token IDs
            positions: Token positions
            intermediate_tensors: Tensors from previous pipeline stages
            inputs_embeds: Input embeddings (alternative to input_ids)
            **model_kwargs: Additional model arguments

        Returns:
            Model output tensor
        """
        # ------【核心逻辑】单点调用模型 forward：把 ids/embeds/positions/中间张量统一喂给模型 ------
        return self.model(
            input_ids=input_ids, 
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            **model_kwargs,
        )

    @staticmethod
    def _is_uniform_decode(
        max_num_scheduled_tokens: int,
        uniform_decode_query_len: int,
        num_tokens: int,
        num_reqs: int,
        force_uniform_decode: bool | None = None,
    ) -> bool:
        """
        Checks if it's a decode batch with same amount scheduled tokens
        across all requests.
        """
        # ------【CUDA Graph】uniform decode 判定：每条请求 token 数一致且等于 query 长度，才可走统一 CUDA Graph ------
        return (
            (
                (max_num_scheduled_tokens == uniform_decode_query_len)
                and (num_tokens == max_num_scheduled_tokens * num_reqs)
            )
            if force_uniform_decode is None
            else force_uniform_decode
        )









    def _determine_batch_execution_and_padding(
        self,
        num_tokens: int, # 这个batch的总token数
        num_reqs: int, # 这个batch的req数
        num_scheduled_tokens_np: np.ndarray, # 这个batch的各个req的token列表
        max_num_scheduled_tokens: int, # 这个batch的最大token数
        use_cascade_attn: bool,
        allow_microbatching: bool = True,
        force_eager: bool = False,
        # For cudagraph capture TODO(lucas): Refactor how we capture cudagraphs (will
        # be improved in model runner v2)
        force_uniform_decode: bool | None = None, 
        force_has_lora: bool | None = None,
        force_num_active_loras: int | None = None,
        num_encoder_reqs: int = 0,
    ) -> tuple[
        CUDAGraphMode, # FULL / PIECEWISE
        BatchDescriptor, # graph key -> 给wrapper看的，去找他的graph
        bool,
        torch.Tensor | None,
        CUDAGraphStat | None,
    ]:
        '''
        padding(token数，TP 2的倍数 + 调度器的档位数) + dispatch

        这个函数，就是输入batch， 然后padding，然后给调度器进行匹配调度
        '''
        # ------【CUDA Graph】判定是否为 uniform decode batch（各请求 token 数一致），决定可用的图模式 ------
        ###############################################
        # 1. 决定是否是纯decode
        ###############################################
        uniform_decode = self._is_uniform_decode(
            max_num_scheduled_tokens=max_num_scheduled_tokens,
            uniform_decode_query_len=self.uniform_decode_query_len,
            num_tokens=num_tokens,
            num_reqs=num_reqs,
            force_uniform_decode=force_uniform_decode,
        )
        # Encoder-decoder models only support CG for decoder_step > 0 (no enc_output
        # is present). Also, chunked-prefill is disabled, so batch are uniform.
        # ------【CUDA Graph】encoder-decoder 带编码器输入时禁止 FULL 图，只能 eager/分段回放 ------
        has_encoder_output = (
            self.model_config.is_encoder_decoder and num_encoder_reqs > 0
        )

        # Compute LoRA state for cudagraph dispatch
        # ------【LoRA+CUDA Graph】统计活跃 LoRA 数量，作为 CUDA Graph 分发的关键维度 ------
        num_active_loras = (
            force_num_active_loras
            if force_num_active_loras is not None
            else len(self.input_batch.lora_id_to_lora_request)
        )
        has_lora = num_active_loras > 0 if force_has_lora is None else force_has_lora

        # ------【TP+CUDA Graph】按 SP 要求把 token 数向上取整到 TP 倍数，得到 padded 长度 ------
        ###############################################
        # 2. padding, 把这个batch的token总数，向上取整到TP倍数，不然不好分
        ###############################################
        num_tokens_padded = self._pad_for_sequence_parallelism(num_tokens) # 被padding后的batch的总token

        # ------【CUDA Graph】封装 dispatch 调用：根据 token/LoRA/uniform 状态选图模式并设置禁用模式 ------
        ###############################################
        # 3. 根据token形状， 纯decode/非纯decode, 来调用调度器，输出 mode, key
        ###############################################
        def dispatch_cudagraph(num_tokens, disable_full=False, valid_modes=None):
            return self.cudagraph_dispatcher.dispatch(
                num_tokens=num_tokens, # 该batch的总token
                has_lora=has_lora,
                uniform_decode=uniform_decode, # 是否是纯decode
                num_active_loras=num_active_loras,
                valid_modes={CUDAGraphMode.NONE} if force_eager else valid_modes,
                invalid_modes={CUDAGraphMode.FULL} if disable_full else None,
            )

        # ------【CUDA Graph】首次分发：cascade attention 或编码器输出时禁用 FULL 图 ------
        cudagraph_mode, batch_descriptor = dispatch_cudagraph(
            num_tokens_padded, disable_full=use_cascade_attn or has_encoder_output # 输入padding后的总token数
        )

        # 返回的是 mode, key, 所以根据key的规格，再次重置一下这个batch的二次padding后的token数
        num_tokens_padded = batch_descriptor.num_tokens

        # ------【TP+chunked prefill】SP 模式下强制校验 padded token 数是 TP 大小的整数倍 ------
        if self.compilation_config.pass_config.enable_sp:
            assert (
                batch_descriptor.num_tokens
                % self.vllm_config.parallel_config.tensor_parallel_size
                == 0
            ), (
                "Sequence parallelism requires num_tokens to be "
                "a multiple of tensor parallel size"
            )

        # Extra coordination when running data-parallel since we need to coordinate
        # across ranks
        # ------【DP+CUDA Graph】多 DP rank 时协调各 rank 的 padding 与图模式，保证 batch 形状一致 ------
        should_ubatch, num_tokens_across_dp = False, None
        if self.vllm_config.parallel_config.data_parallel_size > 1:
            should_ubatch, num_tokens_across_dp, synced_cudagraph_mode = (
                coordinate_batch_across_dp(
                    num_tokens_unpadded=num_tokens,
                    parallel_config=self.parallel_config,
                    allow_microbatching=allow_microbatching,
                    num_tokens_padded=num_tokens_padded,
                    uniform_decode=uniform_decode,
                    cudagraph_mode=cudagraph_mode.value,
                )
            )

            # Extract DP-synced values
            # ------【DP】取本 rank 被 DP 协调后的 padded token 数，用于二次分发 batch_descriptor ------
            if num_tokens_across_dp is not None:
                dp_rank = self.parallel_config.data_parallel_rank
                num_tokens_padded = int(num_tokens_across_dp[dp_rank].item())
                # Re-dispatch with DP padding so we have the correct batch_descriptor
                cudagraph_mode, batch_descriptor = dispatch_cudagraph(
                    num_tokens_padded,
                    valid_modes={CUDAGraphMode(synced_cudagraph_mode)},
                )
                # Assert to make sure the agreed upon token count is correct otherwise
                # num_tokens_across_dp will no-longer be valid
                assert batch_descriptor.num_tokens == num_tokens_padded

        # ------【CUDA Graph+显存 profiling】按需统计本次 batch 的 padding 量与图模式，供观测指标上报 ------
        cudagraph_stats = None
        if self.vllm_config.observability_config.cudagraph_metrics:
            cudagraph_stats = CUDAGraphStat(
                num_unpadded_tokens=num_tokens,
                num_padded_tokens=batch_descriptor.num_tokens,
                num_paddings=batch_descriptor.num_tokens - num_tokens,
                runtime_mode=str(cudagraph_mode),
            )

        # ------【CUDA Graph】汇总图模式、batch 描述、ubatch 与 DP 协调结果返回给主流程 ------
        return (
            cudagraph_mode, # mode
            batch_descriptor, # key
            should_ubatch,
            num_tokens_across_dp,
            cudagraph_stats, # 图信息
        )















    def _register_layerwise_nvtx_hooks(self) -> None:
        """
        Register layerwise NVTX hooks if --enable-layerwise-nvtx-tracing is enabled
        to trace detailed information of each layer or module in the model.
        """

        # ------【核心逻辑】仅当开启层粒度 NVTX 追踪且未注册过时才执行注册 ------
        if (
            self.vllm_config.observability_config.enable_layerwise_nvtx_tracing
            and not self.layerwise_nvtx_hooks_registered
        ):
            # ------【CUDA Graph】CUDA Graph 回放时层粒度 NVTX 标记会缺失，仅打印一次提示 ------
            if self.compilation_config.cudagraph_mode != CUDAGraphMode.NONE:
                logger.debug_once(
                    "layerwise NVTX tracing is not supported when CUDA graph is "
                    "turned off; you may observe part or all of the model "
                    "missing NVTX markers"
                )

            # In STOCK_TORCH_COMPILE mode, after registering hooks here,
            # the __call__ function of nn.module will be recompiled with
            # fullgraph=True. Since nvtx.range_push/pop are not traceable
            # by torch dynamo, we can't register hook functions here
            # because hook functions will also be traced by torch dynamo.
            # ------【核心逻辑】torch.compile 全图模式下 hook 不可被 trace，跳过注册避免编译失败 ------
            if (
                self.vllm_config.compilation_config.mode
                == CompilationMode.STOCK_TORCH_COMPILE
            ):
                logger.debug_once(
                    "layerwise NVTX tracing is not supported when "
                    "CompilationMode is STOCK_TORCH_COMPILE, skipping "
                    "function hooks registration"
                )
            else:
                # ------【核心逻辑】在模型各层注册 NVTX hook，开启层粒度耗时追踪 ------
                pyt_hooks = PytHooks()
                pyt_hooks.register_hooks(self.model, self.model.__class__.__name__)
                self.layerwise_nvtx_hooks_registered = True







    def _get_slot_mappings(
        self,
        num_tokens_padded: int,
        num_reqs_padded: int,
        num_tokens_unpadded: int,
        ubatch_slices: "UBatchSlices | None" = None,
    ) -> tuple[
        dict[int, torch.Tensor] | None,
        dict[str, torch.Tensor] | list[dict[str, torch.Tensor]] | None,
    ]:
        """
        Build slot mappings in both formats needed by the system.

        Args:
            num_tokens_padded: Total number of tokens (padded)
            num_reqs_padded: Total number of requests (padded)
            num_tokens_unpadded: Actual number of tokens (unpadded)
            ubatch_slices: Optional ubatch slicing info for DBO

        Returns:
            A tuple of:
            - slot_mappings_by_gid: dict[int, torch.Tensor] for attention metadata
            - slot_mappings_by_layer: dict[str, torch.Tensor] or list for ForwardContext
        """
        # ------【核心逻辑】无 KV cache 配置时无需 slot mapping，直接返回空 ------
        if not (
            hasattr(self, "kv_cache_config")
            and self.kv_cache_config is not None
            and len(self.kv_cache_config.kv_cache_groups) > 0
        ):
            return None, None

        def _get_slot_mapping(kv_cache_gid: int):
            assert num_reqs_padded is not None and num_tokens_padded is not None
            kv_cache_spec = self.kv_cache_config.kv_cache_groups[
                kv_cache_gid
            ].kv_cache_spec
            # ------【核心逻辑】仅编码器 attention 的组用全零 slot，无需真实 block table 映射 ------
            if isinstance(kv_cache_spec, EncoderOnlyAttentionSpec):
                slot_mapping = torch.zeros(
                    (num_tokens_padded,),
                    dtype=torch.int64,
                    device=self.device,
                )
            else:
                # ------【核心逻辑】从 block table 取每个 token 对应 KV 槽位映射的 GPU 张量 ------
                blk_table = self.input_batch.block_table[kv_cache_gid]
                slot_mapping = blk_table.slot_mapping.gpu[:num_tokens_padded]

            # Fill unused with -1. Needed for reshape_and_cache in full cuda
            # graph mode. `blk_table_tensor` -1 to match mamba PAD_SLOT_ID
            # ------【CUDA Graph】padding 出的空槽填 -1，避免 reshape_and_cache 写越界 ------
            slot_mapping[num_tokens_unpadded:num_tokens_padded].fill_(-1)

            return slot_mapping

        # ------【核心逻辑】按 KV cache 组 id 逐组构建 slot mapping 字典 ------
        slot_mappings_by_gid = {
            gid: _get_slot_mapping(gid)
            for gid, _ in enumerate(self.kv_cache_config.kv_cache_groups)
        }

        # ------【核心逻辑】把组级映射展开成「层名 -> slot mapping」，供每层 attention 直接取用 ------
        slot_mappings_by_layer: dict[str, torch.Tensor] = {}
        for gid, kv_cache_group in enumerate(self.kv_cache_config.kv_cache_groups):
            slot_mapping = slot_mappings_by_gid[gid]
            for layer_name in kv_cache_group.layer_names:
                slot_mappings_by_layer[layer_name] = slot_mapping

        # ------【chunked prefill】存在 ubatch 切片时，按每个子 batch 的 token 区间切出独立映射 ------
        if ubatch_slices is not None:
            result: list[dict[str, torch.Tensor]] = []
            for ubatch in ubatch_slices:
                sliced_mappings: dict[str, torch.Tensor] = {}
                for layer_name, slot_mapping in slot_mappings_by_layer.items():
                    sliced_mappings[layer_name] = slot_mapping[ubatch.token_slice]
                result.append(sliced_mappings)
            return slot_mappings_by_gid, result

        return slot_mappings_by_gid, slot_mappings_by_layer

    def _is_all_reqs_chunked_prefill(self) -> bool:
        """Check if all scheduled requests are marked to discard sampled tokens.

        This is true when `discard_request_mask` is set for every scheduled
        request (e.g., for chunked prefill requests that are not the last
        prefill chunk)."""
        num_reqs = self.input_batch.num_reqs
        # ------【chunked prefill】检查所有请求是否都被标记为丢弃采样结果（全是中间 prefill chunk） ------
        return bool(self.discard_request_mask.np[:num_reqs].all())








########################
# model_runner 开始执行一次 batch调度任务
########################
    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput", # 调度任务batch， 增量信息
        intermediate_tensors: IntermediateTensors | None = None,
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput | IntermediateTensors | None:

        
        # ------【核心逻辑】校验上一步状态已清理，execute_model 与 sample_tokens 必须交替调用 ------
        if self.execute_model_state is not None:
            raise RuntimeError(
                "State error: sample_tokens() must be called "
                "after execute_model() returns None."
            )

        # If ngram_gpu is used, we need to copy the scheduler_output to avoid
        # the modification has influence on the scheduler_output in engine core process.
        # The replace is much faster than deepcopy.
        # ------【投机解码】ngram_gpu 会就地改写调度结果，先浅拷贝两份避免污染引擎核心进程 ------
        if (
            self.speculative_config is not None
            and self.speculative_config.use_ngram_gpu()
        ):
            num_scheduled_tokens_copy = scheduler_output.num_scheduled_tokens.copy()
            spec_decode_tokens_copy = (
                scheduler_output.scheduled_spec_decode_tokens.copy()
            )
            scheduler_output = replace(
                scheduler_output,
                num_scheduled_tokens=num_scheduled_tokens_copy,
                scheduled_spec_decode_tokens=spec_decode_tokens_copy,
            )

        # ------【PD 分离+核心逻辑】有 KV 传输组时，先处理跨实例 KV 抢占迁移的元数据 ------
        if has_kv_transfer_group():
            kv_connector_metadata = scheduler_output.kv_connector_metadata
            assert kv_connector_metadata is not None
            get_kv_transfer_group().handle_preemptions(kv_connector_metadata)


        ############## 本轮batch的总计算的tokens数
        num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens






        # ------【异步 RPC】进入 preprocess 作用域，先等待上一轮 input 准备完成再推进 ------
        with (
            record_function_or_nullcontext("gpu_model_runner: preprocess"), # model_runner 预处理部分
            self.synchronize_input_prep(),
        ):




            
            # Update persistent batch states.
            # ------【核心逻辑】更新持久 batch 状态，返回延迟状态修正闭包供后续调用 ------
            ########################################################################
            # 1. 更新inputBatch持久化信息： SchedulerOutput->inputbatch
            ########################################################################
            deferred_state_corrections_fn = self._update_states(scheduler_output)








            # ------【PD 分离+多模态】EC 生产者只跑编码器并返回空输出，解码交给消费者 ------
            if has_ec_transfer() and not get_ec_transfer().is_consumer:
                with self.maybe_get_ec_connector_output(
                    scheduler_output,
                    encoder_cache=self.encoder_cache,
                ) as ec_connector_output:
                    self._execute_mm_encoder(scheduler_output)
                    return make_empty_encoder_model_runner_output(scheduler_output)

            # ------【核心逻辑】无待处理 token 时直接空跑返回，避免走完整前向 ------
            if not num_scheduled_tokens:
                if (
                    self.parallel_config.distributed_executor_backend
                    == "external_launcher"
                    and self.parallel_config.data_parallel_size > 1
                ):
                    # this is a corner case when both external launcher
                    # and DP are enabled, num_scheduled_tokens could be
                    # 0, and has_unfinished_requests in the outer loop
                    # returns True. before returning early here we call
                    # dummy run to ensure coordinate_batch_across_dp
                    # is called into to avoid out of sync issues.
                    # ------【DP】external launcher + DP 边界情形，先 dummy run 保证各 rank 协调对齐 ------
                    self._dummy_run(1)
                if not has_kv_transfer_group():
                    # Return empty ModelRunnerOutput if no work to do.
                    return EMPTY_MODEL_RUNNER_OUTPUT
                return self.kv_connector_no_forward(scheduler_output, self.vllm_config)

            # ------【前缀缓存】kv-sharing-fast-prefill 与 prompt logprobs 互斥，此处显式校验 ------
            if self.cache_config.kv_sharing_fast_prefill:
                assert not self.num_prompt_logprobs, (
                    "--kv-sharing-fast-prefill produces incorrect "
                    "logprobs for prompt tokens, tokens, please disable "
                    "it when the requests need prompt logprobs"
                )









            ########################################################################
            # 2. 开始从inputbatch里面提取本次要执行的缓冲区信息，准备好input_ids
            ########################################################################
            num_reqs = self.input_batch.num_reqs # 本轮的req数量
            req_ids = self.input_batch.req_ids # batch的req列表
            # ------【核心逻辑】汇总每条请求本步调度的 token 数，得到 CPU 侧逐请求数组 ------
            tokens = [scheduler_output.num_scheduled_tokens[i] for i in req_ids] # batch内的每个req的要计算的token数量
            num_scheduled_tokens_np = np.array(tokens, dtype=np.int32) # 转成numpy格式
            max_num_scheduled_tokens = int(num_scheduled_tokens_np.max()) # 计算数量最多的req的计算token数
            num_tokens_unpadded = scheduler_output.total_num_scheduled_tokens # batch要计算的tokens数

            # ------【核心逻辑】准备 input_ids/positions 等持久缓冲，并返回 logits 索引与投机元数据 ------
            logits_indices, spec_decode_metadata = self._prepare_inputs( # 开始更新缓冲区
                scheduler_output,
                num_scheduled_tokens_np,
            )






            cascade_attn_prefix_lens = None
            # Disable cascade attention when using microbatching (DBO)
            # ------【前缀缓存】开启 cascade attention 且非 ubatch 时，预计算公共前缀长度 ------
            if self.cascade_attn_enabled and not self.parallel_config.use_ubatching:
                # Pre-compute cascade attention prefix lengths
                cascade_attn_prefix_lens = self._compute_cascade_attn_prefix_lens(
                    num_scheduled_tokens_np,
                    self.input_batch.num_computed_tokens_cpu[:num_reqs],
                    scheduler_output.num_common_prefix_blocks,
                )

            # ------【CUDA Graph】分发 CUDA Graph 模式并确定 padding 后的 batch 形状 ------
            ############################################################
            # 把我们处理好的本次的batch， 进行cuda graph的调度处理：
            # 这个batch 本身包含信息：
            #   1. batch形状
            #   2. 纯decode / 非纯decode(纯prefill,混合，都叫这个)

            # padding, 调度后，返回的：
            #   1. mode (FULL/PIECEWISE)
            #   2. key  (desc)
            ############################################################
            (
                cudagraph_mode, # mode
                batch_desc, # key
                should_ubatch,
                num_tokens_across_dp,
                cudagraph_stats,
            ) = self._determine_batch_execution_and_padding(
                num_tokens=num_tokens_unpadded, # 未padding前的tokens数
                num_reqs=num_reqs, # req数
                num_scheduled_tokens_np=num_scheduled_tokens_np, # 每个req的计算token列表
                max_num_scheduled_tokens=max_num_scheduled_tokens, # 每个req的最大计算列表
                use_cascade_attn=cascade_attn_prefix_lens is not None,
                num_encoder_reqs=len(scheduler_output.scheduled_encoder_inputs),
            )

            logger.debug(
                "Running batch with cudagraph_mode: %s, batch_descriptor: %s, "
                "should_ubatch: %s, num_tokens_across_dp: %s",
                cudagraph_mode,
                batch_desc,
                should_ubatch,
                num_tokens_across_dp,
            )

            num_tokens_padded = batch_desc.num_tokens # 被padding后的这个batch的总token数
            num_reqs_padded = ( # 被padding的总req数量
                batch_desc.num_reqs if batch_desc.num_reqs is not None else num_reqs
            )
            # ------【DP+chunked prefill】按需创建 ubatch 切片，把大 batch 拆成多个子 batch 顺序前向 ------
            ubatch_slices, ubatch_slices_padded = maybe_create_ubatch_slices(
                should_ubatch,
                num_scheduled_tokens_np,
                num_tokens_padded,
                num_reqs_padded,
                self.parallel_config.num_ubatches,
            )

            logger.debug(
                "ubatch_slices: %s, ubatch_slices_padded: %s",
                ubatch_slices,
                ubatch_slices_padded,
            )

            # True if any attention backend handles KV cache update separately
            # from forward() (i.e., forward_includes_kv_cache_update=False). When true,
            # slot_mappings must use padded dimensions to match the key/value tensors.
            # ------【核心逻辑】检测是否有后端把 KV cache 更新从 forward 里拆出，需用 padded 维度对齐 ------
            has_separate_kv_update = not all(
                all(
                    g.backend.forward_includes_kv_cache_update
                    for g in self.attn_groups[id]
                )
                for id, spec in enumerate(self.kv_cache_config.kv_cache_groups)
                if not isinstance(spec.kv_cache_spec, EncoderOnlyAttentionSpec)
            )
            # ------【CUDA Graph】FULL 图模式必须用固定 padded 维度，eager/分段则用真实长度 ------
            #################################################
            # flag标志位， True表示，这个batch，调度器判定为使用FULL完整图-》 必须使用固定padded形状
                        # False表示，这个batch, 调度器判定为使用PIECEWISE 分段图，用真实长度
            #################################################
            '''
            FULL, PIECEWISE

            cuda graph， 本质就是录像回放， 把一串 GPU kernel 的启动参数、内存地址、执行顺序全部录下来，之后 replay 就是"原样重放这段录像"

                录像里每个 kernel 的形状是写死的。回放时如果形状变了，它不会自适应——要么算错，要么崩
            
            关键问题：模型 forward 里，哪些算子的形状是"死的"，哪些是"活的"？

            -------------------------------------------------------------------
            一个模型forward = 静态部分 + 一个动态部分：
                静态部分： linear, RMSNorm, RoPE, MLP         只依赖 num_tokens 也就是batch的token总数，他们的处理颗粒度是一个token
                动态部分： attention,                         依赖batch结构， num_reqs, 每个req的seq_lens, query_len，

            注意这个区别：

            同样 num_tokens = 256，可以是 256 个请求 × 各 1 token，也可以是 1 个请求 × 256 token，或任意混合。

            linear/norm 只看"总共 256 个 token"，所以只要 token 数固定，它们就是静态的。
            attention 的 kernel 启动参数（grid、分块）取决于"这 256 个 token 怎么分给请求"，所以即便 token 数固定，batch 结构变，attention 就是动态的。
            -------------------------------------------------------------------

            FULL = 连 attention 一起录进去 → 形状全要锁死

                    FULL 把整个 forward（含 attention）录成一段录像。既然 attention 也被录了，它的启动参数（num_reqs、每请求 token 分布）就全被写死了。

                    所以回放 FULL 图时，batch 必须和录制时一模一样：

                    num_tokens 要 pad 到固定值
                    num_reqs 要 pad 到固定值
                    甚至每请求的 token 分布都要一致（这就是为什么 decode 用 FULL：每条请求固定 1 token，结构天然固定）
                    这就是"FULL 必须 padding 到固定形状"的原因——attention 被录死了，所以整个 batch 结构都得锁死。

                    

            PIECEWISE = 只录静态部分，attention 现场直播 → 只需锁 token 数
                    PIECEWISE 把模型在 attention 处切断：

                    linear/norm/MLP 这些静态段录成 graph（只依赖 num_tokens → 只 pad token 数）
                    attention 不录，每次回放时现场 eager 跑，从 forward context 读真实的 batch 结构（seq_lens / block_table / slot_mapping）
                    所以 PIECEWISE 回放时：

                    num_tokens 要 pad（因为静态段的 kernel 依赖它）
                    num_reqs 随便变（attention 是现场跑的，读多少请求都行）


                    
                    所以这就是为什么我们非纯decode， 需要PIECEWISE，原因是：

                    我们启动flashattention， 它里面每个program，自己知道要从哪个地址来读取数据，这一步算子内部实现，是和batch的内部形状无关的。
                    但是，我们在启动的时候，需要计算grid网格，来得到各自的program的输入的地址，grid的形状这个是要依赖seq_len的，这个就和batch的内部形状有关了，

                --------------------------------------------------------------------------------------
                    就是说，grid网格的形状， 也就是实际我们启动多少个program的形状，这个是依赖batch的内部形状，所以这个无法烤死，
                    除非是纯decode的batch，反正grid的网格划分是固定的值

                    --------------------------------------------------------

                    我彻底懂了，grid就是program的组织形式，graph录制FULL 后， 就相当于把启动的这么多个program，每个读取哪个地址，这些都记录下来了，
                    如果这个时候两个batch, 都是非纯decode， 第一个batch启动的grid形状，不适用于第二个batch的grid形状。
                    所以失败。必须使用PIECEWISE，不录制attention， 每次还是依然动态计算各自的grid的形状

                    而纯decode的情况下，只要两个batch被padding到相同形状，触发对应的FULL的graph，他们的grid网格是通用的，所以就可以使用FULL

                    ---------------------------------------------------------

                    关于调度器这边的padding，也分成两步padding, 


                        padding	                                目的	                            补到什么
                        SP/TP padding	                sequence parallelism 分数据	                token 数补到 TP 大小的整数倍
                                                                                                (只保证总数，不保证每个req平分)
                        cudagraph padding	                复用已捕获的图	                        补到捕获档位 cudagraph_capture_sizes
                                                                                                （decode走req轴填充，非decode走token轴填充）

            '''
            pad_attn = cudagraph_mode == CUDAGraphMode.FULL # 看一下是不是FULL, 要不要吧attn也录入graph

            # ------【核心逻辑】Mamba 对齐缓存模式下，先应用延迟状态修正再预处理 mamba 状态拷贝 ------
            if self.cache_config.mamba_cache_mode == "align":
                # preprocess_mamba reads req_state.num_computed_tokens (CPU)
                # to decide copy operations, so we must apply deferred
                # corrections before it runs.
                if deferred_state_corrections_fn:
                    deferred_state_corrections_fn()
                    deferred_state_corrections_fn = None
                mamba_bufs = self._get_mamba_bufs()
                mamba_utils.preprocess_mamba(
                    scheduler_output,
                    self.kv_cache_config,
                    self.cache_config,
                    self.mamba_state_idx,
                    self.input_batch,
                    self.requests,
                    self.compilation_config.static_forward_context,
                    self.model.get_mamba_state_copy_func(),
                    mamba_bufs.preprocess,
                    align_ctx=mamba_bufs.postprocess_align,
                )
                # preprocess_mamba resets num_accepted_tokens_cpu to 1
                # for requests whose state was copied to a new block.
                # Re-sync to GPU so the mamba kernel reads from the
                # correct initial state slot (init_token_idx = 0).
                # ------【核心逻辑】mamba 状态拷到新块后把 accepted token 数重新同步回 GPU，保证从初始槽读 ------
                self.num_accepted_tokens.np[:num_reqs] = (
                    self.input_batch.num_accepted_tokens_cpu[:num_reqs]
                )
                self.num_accepted_tokens.copy_to_gpu(num_reqs)

                # Stage per-request inputs for the fused postprocess kernel
                # only when that kernel will actually run. The kernel is
                # gated on spec-decode + hybrid (see MambaBuffers.create);
                # without it, ``mamba_bufs.postprocess_align`` is None and
                # the staging buffers don't exist.
                # ------【投机解码】仅当融合后处理 kernel 会运行时，才把逐请求输入 stage 到 GPU ------
                if mamba_bufs.postprocess_align is not None:
                    mamba_utils.stage_postprocess_inputs_to_gpu(
                        mamba_bufs.postprocess_align,
                        scheduler_output,
                        self.input_batch.req_ids,
                        num_reqs,
                        self.requests,
                        self.mamba_state_idx,
                    )

            # ------【投机解码】本步是否携带投机 draft token，决定 attention 元数据是否含投机维度 ------
            use_spec_decode = len(scheduler_output.scheduled_spec_decode_tokens) > 0
            ubatch_slices_attn = ubatch_slices_padded if pad_attn else ubatch_slices






            ################################################################################################
            # 读取已经计算好的 slot_mapping.gpu, 整理成下游需要的格式
            # slot_mapping = blk_table.slot_mapping.gpu[:num_tokens_padded]   # 读已算好的结果
            # slot_mapping[num_tokens_unpadded:num_tokens_padded].fill_(-1)    # padding 部分填 -1

            '''
            然后做三件事：

                padding 填 -1（CUDA graph full 模式需要固定形状，空槽填 -1 防止写越界）
                按 KV group 分组（slot_mappings_by_gid：每个 KV cache group 一份）
                展开成「层名 → slot mapping」（slot_mappings_by_layer：每层 attention 直接取）+ ubatch 切片
            '''

            '''
            ubatch, = micro-batch, 微批次， = 把一个大的batch拆成多个小的子批次 sub-batch, 在前向里顺序处理
            好处是：节省峰值激活显存。
            假设一个 batch 有 8192 个 token，如果一次性前向，中间激活（activation）要一次性占满显存。
            拆成 2 个 ubatch（各 4096 token）顺序跑，峰值激活减半


            '''

            ################################################################################################
            slot_mappings_by_group, slot_mappings = self._get_slot_mappings( 
                num_tokens_padded=num_tokens_padded
                if pad_attn or has_separate_kv_update
                else num_tokens_unpadded,
                num_reqs_padded=(
                    num_reqs_padded if pad_attn or has_separate_kv_update else num_reqs
                ),
                num_tokens_unpadded=num_tokens_unpadded,
                ubatch_slices=ubatch_slices_padded,
            )











            ########################################################################
            # 构建 attention 元数据（含投机解码公共元数据），为前向做好准备 ，就是一次格式转换
            # 这里的元数据，
            ########################################################################
            '''

            它不是 token 内容，但它比「纯规格参数」多了一样东西——地址引用。里面装着 attention kernel 必须知道的：

                字段	                            作用	                                是数据还是规格？
                slot_mapping	            每个 token 的 KV 要写到哪个物理槽	                地址（引用 buffer）
                block_table	                每个 req 的 KV block 物理地址表	                    地址
                seq_lens	                每个 req 多长	                                    规格
                positions	                每个 token 的位置	                                数据/规格
                query_start_loc	            token区间划分	                                    规格
                
            所以准确说：metadata = 「运行时规格 + 指向 KV cache 地址的指针」的打包。

            attention kernel 靠它才知道「把当前这步的 K/V 写到显存哪个位置、prefill 和 decode 分别怎么算」。
            '''
            attn_metadata, spec_decode_common_attn_metadata = (
                self._build_attention_metadata(
                    num_tokens=num_tokens_unpadded, # 未padding的实际batch的token数
                    num_tokens_padded=num_tokens_padded if pad_attn else None, # 被padding后的batch的token数
                    num_reqs=num_reqs, # batch内的原本的req数
                    num_reqs_padded=num_reqs_padded if pad_attn else None, # batch内被padding后的req数量（如果是FULL 纯decode）
                    max_query_len=max_num_scheduled_tokens, # 每个req的最大q token数
                    ubatch_slices=ubatch_slices_attn,
                    logits_indices=logits_indices,
                    use_spec_decode=use_spec_decode,
                    num_scheduled_tokens=scheduler_output.num_scheduled_tokens,
                    cascade_attn_prefix_lens=cascade_attn_prefix_lens,
                    slot_mappings=slot_mappings_by_group,
                )
            )





            ########################################################################
            # 统一预处理出模型前向所需的全部输入（ids/embeds/positions/中间张量/kwargs）
            ########################################################################
            (
                input_ids, # token id（核心）(已经被padding后的)
                inputs_embeds, # 直接传 embedding 时用（多模态/embed 输入，非 token id）
                positions, # 位置（核心）（已经被padding后的）
                intermediate_tensors, # PP 的中间激活传递
                model_kwargs, # model 专属 kwargs（token_type_ids、encoder_outputs 等）
                ec_connector_output,  # encoder-decoder connector 输出
            ) = self._preprocess(
                scheduler_output, num_tokens_padded, intermediate_tensors # 传入填充值
            )









        # Encoder-decoder models can only compile the pure decode steps where no
        # encoder inputs are present. Use eager for the first pass.
        # ------【CUDA Graph】encoder-decoder 带编码器输入时无法走编译图，标记用 eager 前向 ------
        num_encoder_reqs = len(scheduler_output.scheduled_encoder_inputs)
        has_encoder_input = (
            self.model_config.is_encoder_decoder and num_encoder_reqs > 0
        )











        # Run the model.
        # Use persistent buffers for CUDA graphs.
        # When spec decode is enabled, defer connector finalization
        # (wait_for_save + clear metadata) until after draft model runs.
        # ------【投机解码+PD 分离】投机解码时延迟 kv connector 收尾，等 draft 模型跑完再 wait/clear ------
        defer_kv_connector_finalize = self.speculative_config is not None
        # Update the EPLB meta.
        # ------【EP/EPLB】前向开始前准备 EPLB 状态，按本步 token 分布决定专家放置 ------
        if self.eplb_state is not None:
            self.eplb_state.prepare_forward(
                self.model_config,
                num_tokens_unpadded,
                ubatch_slices_padded,
            )
        # ------【CUDA Graph】设置前向上下文（图模式/batch 描述/槽映射），再调用模型 forward ------
        with (

            # 设置cuda graph的前向上下文
            set_forward_context(
                attn_metadata, # 注意力后端元数据
                self.vllm_config,
                num_tokens=num_tokens_padded, # padding后的batch token数
                num_tokens_across_dp=num_tokens_across_dp,
                cudagraph_runtime_mode=cudagraph_mode, # mode
                batch_descriptor=batch_desc, # key
                ubatch_slices=ubatch_slices_padded,
                slot_mapping=slot_mappings,
                skip_compiled=has_encoder_input,
            ),

            record_function_or_nullcontext("gpu_model_runner: forward"),
            self.maybe_get_kv_connector_output(
                scheduler_output,
                defer_finalize=defer_kv_connector_finalize,
            ) as kv_connector_output,
        ):


            ################################################################################################
            # 正式开始我们的前向推理
            ################################################################################################
            model_output = self._model_forward(
                input_ids=input_ids, # padding后的输入batch
                positions=positions, # padding后的token在batch的布局
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=inputs_embeds,
                **model_kwargs,
            )




        with record_function_or_nullcontext("gpu_model_runner: postprocess"):
            # ------【投机解码】EAGLE3 用辅助 hidden states 时拆出主/辅两份输出 ------
            if self.use_aux_hidden_state_outputs:
                # True when EAGLE 3 is used.
                hidden_states, aux_hidden_states = model_output
            else:
                # Common case.
                hidden_states = model_output
                aux_hidden_states = None

            if not self.broadcast_pp_output:
                # Common case.
                # ------【PP】非末段 rank 直接把中间张量返回给下一 stage，不再算 logits ------
                if not get_pp_group().is_last_rank:
                    # Return the intermediate tensors.
                    assert isinstance(hidden_states, IntermediateTensors)
                    self.kv_connector_output = kv_connector_output
                    return hidden_states

                # ------【核心逻辑】pooling 模型走 _pool 聚合输出，直接返回 embedding ------
                if self.is_pooling_model:
                    # Return the pooling output.
                    return self._pool(
                        hidden_states,
                        num_scheduled_tokens,
                        num_scheduled_tokens_np,
                        kv_connector_output,
                    )

                # ------【核心逻辑】末段按 logits_indices 取 hidden states，计算最终 logits ------
                sample_hidden_states = hidden_states[logits_indices]
                logits = self.model.compute_logits(sample_hidden_states)
            else:
                # Rare case.
                assert not self.is_pooling_model

                # ------【PP+NCCL 通信】广播 PP 输出模式下，非末段把 hidden states 发送给末段 ------
                sample_hidden_states = hidden_states[logits_indices]
                if not get_pp_group().is_last_rank:
                    all_gather_tensors = {
                        "residual": not is_residual_scattered_for_sp(
                            self.vllm_config, num_tokens_padded
                        )
                    }
                    get_pp_group().send_tensor_dict(
                        hidden_states.tensors,
                        all_gather_group=get_tp_group(),
                        all_gather_tensors=all_gather_tensors,
                    )
                    logits = None
                else:
                    logits = self.model.compute_logits(sample_hidden_states)

                # ------【PP+NCCL 通信】末段算好 logits 后广播给所有 PP rank，保证各 rank 拿到一致结果 ------
                model_output_broadcast_data: dict[str, Any] = {}
                if logits is not None:
                    model_output_broadcast_data["logits"] = logits.contiguous()

                broadcasted = get_pp_group().broadcast_tensor_dict(
                    model_output_broadcast_data, src=len(get_pp_group().ranks) - 1
                )
                assert broadcasted is not None
                logits = broadcasted["logits"]

        # ------【异步 RPC】把前向产物打包进 execute_model_state，供随后的 sample_tokens 使用 ------
        self.execute_model_state = ExecuteModelState(
            scheduler_output,
            logits,
            spec_decode_metadata,
            spec_decode_common_attn_metadata,
            hidden_states,
            sample_hidden_states,
            aux_hidden_states,
            ec_connector_output,
            cudagraph_stats,
            slot_mappings,
        )
        self.kv_connector_output = kv_connector_output

        # Now the batch has been launched we can wait for corrections from the
        # previous model forward without breaking async scheduling.
        # ------【异步 RPC】batch 已发射后再应用上一轮前向的延迟状态修正，避免打断异步调度 ------
        if deferred_state_corrections_fn:
            deferred_state_corrections_fn()

        return None
























    def _input_fits_in_drafter(
        self, common_attn_metadata: CommonAttentionMetadata | None
    ) -> bool:
        # ------【投机解码】无 attention 元数据时视为不适用，直接返回 False ------
        if common_attn_metadata is None:
            return False
        assert self.speculative_config is not None
        # DFlash queries one extra token (the bonus token) beyond num_spec_tokens
        # ------【投机解码】draft 模型需一次查询 num_spec_tokens（DFlash 额外 +1 bonus）个 token ------
        num_drafter_query_tokens = self.num_spec_tokens + (
            1 if self.speculative_config.use_dflash() else 0
        )
        # ------【投机解码】校验当前序列长度加 draft 查询长度不超过 draft 模型最大长度 ------
        return (
            common_attn_metadata.max_seq_len + num_drafter_query_tokens
            <= self.effective_drafter_max_model_len
        )

    @torch.inference_mode
    def sample_tokens(
        self, grammar_output: "GrammarOutput | None"
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput | IntermediateTensors:
        if self.execute_model_state is None:
            # ------【PD 分离】取出 kv_connector 输出并清空，供后续阶段/rank 透传使用 ------
            kv_connector_output = self.kv_connector_output
            self.kv_connector_output = None
            # receive sampled token ids from the last PP rank.
            # ------【PP + 异步 RPC】非末级 PP rank 接收上一 rank 广播的采样 token id，为下轮输入做准备 ------
            if self.use_async_scheduling and not get_pp_group().is_last_rank:
                self._pp_receive_prev_sampled_token_ids_to_input_batch()
            # ------【PP】本 rank 无执行状态时提前返回，仅透传 kv_connector_output ------
            # In case of PP with kv transfer, we need to pass through the
            # kv_connector_output
            return ModelRunnerOutput.with_kv_conn_output_only(kv_connector_output)

        # ------【核心逻辑】解包本 step 执行阶段暂存的调度/隐藏态/采样元数据等临时状态 ------
        # Unpack ephemeral state.
        (
            scheduler_output,
            logits,
            spec_decode_metadata,
            spec_decode_common_attn_metadata,
            hidden_states,
            sample_hidden_states,
            aux_hidden_states,
            ec_connector_output,
            cudagraph_stats,
            slot_mappings,
        ) = self.execute_model_state
        # ------【核心逻辑】清空临时状态引用，避免下轮误用上一 step 的数据 ------
        # Clear ephemeral state.
        self.execute_model_state = None

        # ------【结构化输出/grammar】将约束解码位掩码应用到 logits，屏蔽非法 token ------
        # Apply structured output bitmasks if present.
        if grammar_output is not None:
            apply_grammar_bitmask(
                scheduler_output, grammar_output, self.input_batch, logits
            )

        # ------【投机解码 + 核心逻辑】对 logits 采样，产出本 step 的采样 token 与 logprob ------
        with record_function_or_nullcontext("gpu_model_runner: sample"):
            sampler_output = self._sample(logits, spec_decode_metadata)

        # ------【核心逻辑】采样后更新内部状态（丢弃掩码、请求输出等 bookkeeping 前置） ------
        self._update_states_after_model_execute(
            sampler_output.sampled_token_ids, scheduler_output
        )
        # ------【PP + 异步 RPC】末级 PP rank 把采样 token 广播给其余 rank，供异步调度对齐输入 ------
        if self.use_async_scheduling:
            pp = get_pp_group()
            # For torchrun external_launcher PP mode with broadcast_pp_output=True,
            # PP outputs have been broadcasted to all ranks at logits computation.
            # Therefore, here is no need to send sampled token ids again in this case.
            if not self.broadcast_pp_output and pp.world_size > 1 and pp.is_last_rank:
                self._pp_broadcast_prev_sampled_token_ids(
                    sampler_output.sampled_token_ids
                )

        # ------【投机解码】重置草稿 token/概率缓存与计数，防止复用上一 step 的陈旧数据 ------
        self._draft_token_ids = None
        self._draft_probs = None
        self._draft_prob_req_ids = None
        self._draft_token_req_ids = None
        self.valid_sampled_token_count_gpu = None
        self.input_batch.prev_sampled_token_ids = None

        # ------【投机解码】嵌套函数：调用草稿模型 propose 并用独立流异步拷贝草稿 token 到 CPU ------
        def propose_draft_token_ids(sampled_token_ids):
            assert spec_decode_common_attn_metadata is not None
            with record_function_or_nullcontext("gpu_model_runner: draft"):
                self._draft_token_ids = self.propose_draft_token_ids(
                    scheduler_output,
                    sampled_token_ids,
                    self.input_batch.sampling_metadata,
                    hidden_states,
                    sample_hidden_states,
                    aux_hidden_states,
                    spec_decode_metadata,
                    spec_decode_common_attn_metadata,
                    slot_mappings,
                )
                self._copy_draft_token_ids_to_cpu(scheduler_output)

        # ------【投机解码】初始化投机配置与「bookkeeping 后再 draft」标志 ------
        spec_config = self.speculative_config
        draft_after_bookkeeping = False
        if spec_config is not None:
            # ------【投机解码】判断草稿器是否适合当前 batch，并区分 GPU/CPU 两种 token 输入路径 ------
            # Decide whether to run the drafter or zero out draft tokens.
            input_fits_in_drafter = self._input_fits_in_drafter(
                spec_decode_common_attn_metadata
            )
            # Whether the drafter runs a GPU model forward (and thus carries
            # TP/EP/DP collectives), independent of padded-batch timing.
            drafter_runs_model_forward = (
                spec_config.use_eagle()
                or spec_config.uses_draft_model()
                or spec_config.uses_extract_hidden_states()
            )
            use_gpu_toks = (
                drafter_runs_model_forward
                and not spec_config.disable_padded_drafter_batch
            )
            # ------【投机解码 + DP】EAGLE/DraftModel 直接消费 GPU 采样 token，无需等待 bookkeeping ------
            if use_gpu_toks:
                # EAGLE/DraftModel speculative decoding can use the GPU sampled tokens
                # as inputs, and does not need to wait for bookkeeping to finish.
                assert isinstance(
                    self.drafter,
                    EagleProposer
                    | DFlashProposer
                    | DraftModelProposer
                    | ExtractHiddenStatesProposer
                    | Gemma4Proposer,
                )
                # ------【投机解码】GPU 路径直接复用采样张量作为草稿模型输入 ------
                sampled_token_ids = sampler_output.sampled_token_ids
                # ------【投机解码】草稿器可运行则立即 propose，否则走 padded 路径准备下一 token ------
                if input_fits_in_drafter:
                    propose_draft_token_ids(sampled_token_ids)
                else:
                    if self.valid_sampled_token_count_event is not None:
                        assert spec_decode_common_attn_metadata is not None
                        next_token_ids, valid_sampled_tokens_count = (
                            self.drafter.prepare_next_token_ids_padded(
                                sampled_token_ids,
                                self.requests,
                                self.input_batch,
                                self.discard_request_mask.gpu,
                            )
                        )
                        self._copy_valid_sampled_token_count(
                            next_token_ids, valid_sampled_tokens_count
                        )
                    # ------【DP + 投机解码】DP 各 rank 判断不一致时 dummy_run 一次，防止集合通信悬挂 ------
                    if self.parallel_config.data_parallel_size > 1:
                        # Prevent hang when DP ranks disagree on input_fits_in_drafter
                        self.drafter.dummy_run(num_tokens=1)
            # ------【投机解码】ngram GPU 草稿器同样走 GPU token 路径，异步拷贝有效计数到 CPU ------
            elif (
                spec_config.use_ngram_gpu()
                and not spec_config.disable_padded_drafter_batch
            ):
                assert isinstance(self.drafter, NgramProposerGPU)
                sampled_token_ids = sampler_output.sampled_token_ids
                if input_fits_in_drafter:
                    propose_draft_token_ids(sampled_token_ids)
                elif self.valid_sampled_token_count_event is not None:
                    assert spec_decode_common_attn_metadata is not None
                    next_token_ids, valid_sampled_tokens_count, _ = (
                        self.drafter.update_token_ids_ngram(
                            sampled_token_ids,
                            self.input_batch,
                            self.token_ids_gpu_tensor,
                            self.num_tokens_no_spec_gpu,
                            self.discard_request_mask.gpu,
                        )
                    )
                    self._copy_valid_sampled_token_count(
                        next_token_ids, valid_sampled_tokens_count
                    )
            else:
                # ------【投机解码】其余 CPU 草稿器需在 bookkeeping 后运行，先置标志延后 ------
                # These drafters consume CPU sampled tokens, so they run
                # after bookkeeping.
                draft_after_bookkeeping = True

            # ------【投机解码】草稿器不可运行时清零草稿 token，避免调度器复用上一步陈旧草稿 ------
            if not input_fits_in_drafter:
                # Zero out draft tokens so the scheduler doesn't schedule
                # stale drafts from the previous step.
                # For Nemotron-H: it is necessary to zero out the draft tokens,
                # otherwise the stale tokens will corrupt Mamba recurrent
                # state and logprobs for sequences near max_model_len.
                self._draft_token_ids = torch.zeros(
                    1, device=self.device, dtype=torch.int32
                ).expand(len(self.input_batch.req_ids), self.num_spec_tokens)
                self._draft_probs = None
                self._draft_prob_req_ids = None
                self._copy_draft_token_ids_to_cpu(scheduler_output, zeros_only=True)

        # ------【核心逻辑】执行同步 bookkeeping，汇总 logprob、有效 token 与失效请求索引 ------
        with record_function_or_nullcontext("gpu_model_runner: bookkeep"):
            (
                num_nans_in_logits,
                logprobs_lists,
                valid_sampled_token_ids,
                prompt_logprobs_dict,
                req_ids_output_copy,
                req_id_to_index_output_copy,
                invalid_req_indices,
            ) = self._bookkeeping_sync(
                scheduler_output,
                sampler_output,
                logits,
                hidden_states,
                scheduler_output.total_num_scheduled_tokens,
            )

        # ------【投机解码 + DP】CPU 草稿器（ngram 等）在 bookkeeping 后用有效 token 运行 propose ------
        if draft_after_bookkeeping:
            # ngram and other speculative decoding methods use the sampled
            # tokens on the CPU, so they are run after bookkeeping.
            if input_fits_in_drafter:
                propose_draft_token_ids(valid_sampled_token_ids)
            elif (
                drafter_runs_model_forward
                and self.parallel_config.data_parallel_size > 1
            ):
                # Prevent hang when DP ranks disagree on input_fits_in_drafter
                assert isinstance(
                    self.drafter,
                    EagleProposer
                    | DFlashProposer
                    | DraftModelProposer
                    | ExtractHiddenStatesProposer
                    | Gemma4Proposer,
                )
                self.drafter.dummy_run(num_tokens=1)

        # ------【PD 分离】草稿模型运行完后 finalize kv_connector，等待保存并清理元数据 ------
        # Finalize KV connector (wait_for_save + clear metadata) after
        # draft model runs. Deferred from target model forward to allow
        # draft model to also save its KV cache.
        if spec_config is not None:
            self.finalize_kv_connector()

        # ------【EP/EPLB】执行专家并行负载均衡一步，统计路由并可能触发专家重平衡 ------
        with record_function_or_nullcontext("gpu_model_runner: eplb"):
            self.eplb_step()

        # ------【PD 分离】再次取出可能被 draft 阶段修改的 kv_connector 输出并清空 ------
        # self.kv_connector_output may be modified during drafting
        kv_connector_output = self.kv_connector_output
        self.kv_connector_output = None

        # ------【核心逻辑】组装同步 ModelRunnerOutput（token/logprob/kv/ec 等结果） ------
        with record_function_or_nullcontext("gpu_model_runner: ModelRunnerOutput"):
            output = ModelRunnerOutput(
                req_ids=req_ids_output_copy,
                req_id_to_index=req_id_to_index_output_copy,
                sampled_token_ids=valid_sampled_token_ids,
                logprobs=logprobs_lists,
                prompt_logprobs_dict=prompt_logprobs_dict,
                kv_connector_output=kv_connector_output,
                ec_connector_output=ec_connector_output
                if self.supports_mm_inputs
                else None,
                num_nans_in_logits=num_nans_in_logits,
                cudagraph_stats=cudagraph_stats,
                routed_experts=None,
            )

        # ------【EP/EPLB】同步路径：把已同步到 CPU 的专家路由数据包装为 numpy 返回 ------
        if not self.use_async_scheduling:
            if self.routed_experts_initialized:
                # Sync path: D2H was issued in ``_bookkeeping_sync`` and
                # synchronized by ``_to_list``'s event.synchronize(), so
                # the pinned buffers are ready to be wrapped as numpy.
                total = scheduler_output.total_num_scheduled_tokens
                output.routed_experts = RoutedExpertsLists(
                    routing_data=self.routed_experts_cpu[:total].numpy(),
                    slot_mapping=self.routed_experts_slot_mapping_cpu[:total].numpy(),
                )
            return output

        with record_function_or_nullcontext(
            "gpu_model_runner: AsyncGPUModelRunnerOutput"
        ):
            # Async path: produce a device-side snapshot that the async
            # copy stream can D2H later. Both tensors must be private
            # clones because:
            #   - ``routing_data`` source is the shared capturer buffer,
            #     which the next forward overwrites on the default stream.
            #   - ``slot_mapping`` source is our own
            #     ``routed_experts_slot_mapping_device``, which the
            #     next ``_prepare_inputs`` overwrites on the default
            #     stream while the D2H is still pending on the copy
            #     stream.
            # Without clones, the copy stream would read torn data.
            # ------【EP/EPLB + 异步 RPC】对专家路由张量做私有快照，避免下轮 forward 覆盖导致拷贝撕裂 ------
            routed_experts_snapshot = self.get_routed_experts(
                scheduler_output.total_num_scheduled_tokens
            )

            # ------【异步 RPC】构造异步输出，携带采样张量与拷贝流，供后续非阻塞 D2H ------
            async_output = AsyncGPUModelRunnerOutput(
                model_runner_output=output,
                sampled_token_ids=sampler_output.sampled_token_ids,
                logprobs_tensors=sampler_output.logprobs_tensors,
                invalid_req_indices=invalid_req_indices,
                async_output_copy_stream=self._get_or_create_async_output_copy_stream(),
                vocab_size=self.input_batch.vocab_size,
                routed_experts=routed_experts_snapshot,
                check_ep_fault=self.check_ep_fault,
            )
        # ------【异步 RPC】异步调度时把 CPU 采样 token 引用及事件保存到 input_batch ------
        with record_function_or_nullcontext(
            "gpu_model_runner: set_async_sampled_token_ids"
        ):
            # Save ref of sampled_token_ids CPU tensor if the batch contains
            # any requests with sampling params that require output ids.
            self.input_batch.set_async_sampled_token_ids(
                async_output.sampled_token_ids_cpu,
                async_output.async_copy_ready_event,
            )

        return async_output

    def _pp_broadcast_prev_sampled_token_ids(
        self, sampled_token_ids: torch.Tensor
    ) -> None:
        """Broadcast sampled token ids (GPU) from last PP stage"""
        # ------【PP】仅末级 PP rank 负责广播，先校验自身 rank ------
        pp = get_pp_group()
        assert pp.is_last_rank
        # ------【PP】校验广播张量形状为 [num_reqs,1]，匹配异步调度输入约定 ------
        # `prev_sampled_token_ids` is expected to have shape [num_reqs, 1].
        assert sampled_token_ids.dim() == 2 and sampled_token_ids.shape[-1] == 1, (
            "PP+async expects sampled_token_ids to have shape [num_reqs, 1]"
        )
        # ------【chunked prefill + NCCL 通信】全 chunked prefill 跳过广播，否则跨 PP rank 集合广播 ------
        # Skip for chunked prefill: sampled tokens are dummy
        # and will be discarded, no need to broadcast.
        if not self._is_all_reqs_chunked_prefill():
            torch.distributed.broadcast(
                sampled_token_ids, src=pp.rank, group=pp.device_group
            )

    def _pp_receive_prev_sampled_token_ids_to_input_batch(self) -> None:
        """Receive sampled token ids broadcast from last PP stage"""
        # ------【PP】非末级 PP rank 才接收广播，先校验自身 rank ------
        pp = get_pp_group()
        assert not pp.is_last_rank
        # ------【PP】分配接收缓冲区，形状 [num_reqs,1] 与广播张量对应 ------
        num_reqs = self.input_batch.num_reqs
        # `prev_sampled_token_ids` is expected to have shape [num_reqs, 1].
        recv = torch.empty((num_reqs, 1), dtype=torch.int32, device=self.device)
        # ------【chunked prefill + NCCL 通信】全 chunked prefill 跳过，否则从末级 rank 集合广播接收 ------
        # skip for chunked prefill.
        if not self._is_all_reqs_chunked_prefill():
            torch.distributed.broadcast(recv, src=pp.last_rank, group=pp.device_group)
        # ------【PP + 异步 RPC】把接收到的采样 token 写入 input_batch，供下轮输入拼接 ------
        self.input_batch.prev_sampled_token_ids = recv

        # construct `prev_req_id_to_index` here so `_prepare_input_ids`
        # can map req_id -> previous batch row
        # ------【核心逻辑】计算本 step 丢弃请求索引集合，跳过这些请求不建立映射 ------
        discard_req_indices = np.nonzero(self.discard_request_mask.np[:num_reqs])[0]
        discard_req_indices_set = set(discard_req_indices)
        prev_req_id_to_index: dict[str, int] = {}
        # ------【PP + 异步 RPC】遍历存活请求：推进本地输出长度并占位，同时记录 req_id → 上一 batch 行号 ------
        for i, req_id in enumerate(self.input_batch.req_ids):
            if i in discard_req_indices_set:
                continue
            prev_req_id_to_index[req_id] = i
            # PP+async scheduling: advance per-request local cached output length by
            # appending a placeholder (-1) token id.
            if (req_state := self.requests.get(req_id)) is not None:
                req_state.output_token_ids.append(-1)
            pos = self.input_batch.num_tokens_no_spec[i]
            self.input_batch.is_token_ids[i, pos] = True
            self.input_batch.num_tokens_no_spec[i] = pos + 1
        self.input_batch.prev_req_id_to_index = prev_req_id_to_index

    def take_draft_token_ids(self) -> DraftTokenIds | None:
        # ------【投机解码】无投机 token 或无草稿请求 id 时直接返回 None ------
        if not self.num_spec_tokens or not self._draft_token_req_ids:
            return None
        # ------【投机解码】从 CPU 取出草稿 token 与请求 id，封装为 DraftTokenIds 返回 ------
        draft_token_ids, req_ids = self._get_draft_token_ids_cpu()
        return DraftTokenIds(req_ids, draft_token_ids)

    def _copy_draft_token_ids_to_cpu(
        self, scheduler_output: "SchedulerOutput", zeros_only: bool = False
    ) -> None:
        # ------【投机解码】记录上一 step 的投机 token 数，供下轮增量处理使用 ------
        if torch.is_tensor(self._draft_token_ids):
            assert isinstance(self._draft_token_ids, torch.Tensor)
            self.prev_num_spec_tokens = self._draft_token_ids.shape[1]
        # ------【异步 RPC + 结构化输出/grammar】异步调度下仅在需要时拷贝草稿 token（约束/惩罚/坏词） ------
        # Check if we need to copy draft tokens to CPU. In async scheduling,
        # we only copy when needed for structured output, penalties or bad_words.
        if self.use_async_scheduling and not (
            scheduler_output.has_structured_output_requests
            or self.input_batch.sampling_metadata.output_token_ids
        ):
            return
        # ------【投机解码】记录草稿 token 对应的请求 id，供 CPU 端按请求对齐使用 ------
        # We must also set the corresponding request ids.
        self._draft_token_req_ids = self.input_batch.req_ids.copy()

        # ------【异步 RPC】校验草稿 token 为张量且拷贝流/事件/CPU 缓冲均已就绪 ------
        draft_token_ids: torch.Tensor = self._draft_token_ids
        if not torch.is_tensor(draft_token_ids):
            return
        assert self.draft_token_ids_event is not None
        assert self.draft_token_ids_copy_stream is not None
        assert self.draft_token_ids_cpu is not None
        # ------【异步 RPC】取默认流与草稿张量形状，用于拷贝区间裁剪 ------
        default_stream = torch.cuda.current_stream()
        num_reqs = draft_token_ids.shape[0]
        num_spec_tokens = draft_token_ids.shape[1]
        # ------【异步 RPC】在专用拷贝流上非阻塞拷贝（或清零）草稿 token 到 CPU，并记录事件 ------
        with torch.cuda.stream(self.draft_token_ids_copy_stream):
            if not zeros_only:
                # Trigger async copy of draft token ids to cpu.
                self.draft_token_ids_copy_stream.wait_stream(default_stream)
                self.draft_token_ids_cpu[:num_reqs, :num_spec_tokens].copy_(
                    draft_token_ids, non_blocking=True
                )
            else:
                # No copy needed, just zero-out cpu tensor.
                self.draft_token_ids_cpu[:num_reqs, :num_spec_tokens] = 0
            self.draft_token_ids_event.record()

    def _get_draft_token_ids_cpu(self) -> tuple[list[list[int]], list[str]]:
        # ------【投机解码】草稿 token 为 list 时直接返回（CPU 草稿器路径） ------
        if isinstance(self._draft_token_ids, list):
            return self._draft_token_ids, self.input_batch.req_ids
        req_ids = self._draft_token_req_ids
        # ------【投机解码】无草稿请求 id 时返回空列表 ------
        if req_ids is None:
            return [], []
        assert self.draft_token_ids_event is not None
        assert self.draft_token_ids_cpu is not None
        # ------【异步 RPC】同步等待草稿 token 的 D2H 拷贝事件完成 ------
        self.draft_token_ids_event.synchronize()
        assert isinstance(self._draft_token_ids, torch.Tensor)
        num_spec_tokens = self._draft_token_ids.shape[1]
        # ------【异步 RPC】把 CPU 草稿 token 转为 list 并连同请求 id 返回 ------
        return self.draft_token_ids_cpu[
            : len(req_ids), :num_spec_tokens
        ].tolist(), req_ids

    def _copy_valid_sampled_token_count(
        self, next_token_ids: torch.Tensor, valid_sampled_tokens_count: torch.Tensor
    ) -> None:
        # ------【异步 RPC】无有效计数事件（非 padded 草稿）时直接返回 ------
        if self.valid_sampled_token_count_event is None:
            return

        # ------【异步 RPC】在独立拷贝流上等待默认流，实现与草稿模型输入准备重叠 ------
        default_stream = torch.cuda.current_stream()
        # Initialize a new stream to overlap the copy operation with
        # prepare_input of draft model.
        with torch.cuda.stream(self.valid_sampled_token_count_copy_stream):
            self.valid_sampled_token_count_copy_stream.wait_stream(default_stream)  # type: ignore
            # ------【异步 RPC】非阻塞拷贝有效采样计数到 CPU 并记录事件 ------
            counts = valid_sampled_tokens_count
            counts_cpu = self.valid_sampled_token_count_cpu
            assert counts_cpu is not None
            counts_cpu[: counts.shape[0]].copy_(counts, non_blocking=True)
            self.valid_sampled_token_count_event.record()

        # ------【异步 RPC】异步投机解码时暂存 GPU 计数，供 _prepare_inputs 做 GPU 端修正 ------
        if self.use_async_spec_decode:
            # Stash for GPU-side correction in _prepare_inputs.
            self.valid_sampled_token_count_gpu = valid_sampled_tokens_count
        # ------【投机解码】把下一 token id（带 batch 维）写入 input_batch 作为上一采样结果 ------
        self.input_batch.prev_sampled_token_ids = next_token_ids.unsqueeze(1)

    def _get_valid_sampled_token_count(self) -> list[int]:
        # ------【异步 RPC】无计数事件或上轮采样 token 时返回空列表 ------
        # Wait until valid_sampled_tokens_count is copied to cpu,
        prev_sampled_token_ids = self.input_batch.prev_sampled_token_ids
        sampled_count_event = self.valid_sampled_token_count_event
        if sampled_count_event is None or prev_sampled_token_ids is None:
            return []

        counts_cpu = self.valid_sampled_token_count_cpu
        assert counts_cpu is not None
        # ------【异步 RPC】同步等待计数拷贝完成，按上轮 token 数裁剪并返回 list ------
        sampled_count_event.synchronize()
        return counts_cpu[: prev_sampled_token_ids.shape[0]].tolist()

    def _get_spec_decode_draft_probs(
        self, spec_decode_metadata: SpecDecodeMetadata
    ) -> torch.Tensor | None:
        # ------【投机解码】无缓存草稿概率时返回 None，走传统拒绝采样路径 ------
        if self._draft_probs is None or self._draft_prob_req_ids is None:
            return None

        # ------【投机解码】建立 req_id → 缓存概率行号 的映射，便于按请求取草稿概率 ------
        row_by_req_id = {
            req_id: idx for idx, req_id in enumerate(self._draft_prob_req_ids)
        }
        draft_probs_rows: list[torch.Tensor] = []
        # ------【投机解码】按当前 batch 逐请求截取草稿概率，缺失则告警并回退 ------
        for req_id, num_draft in zip(
            self.input_batch.req_ids, spec_decode_metadata.num_draft_tokens
        ):
            if num_draft == 0:
                continue
            row_idx = row_by_req_id.get(req_id)
            if row_idx is None:
                logger.warning(
                    "Missing cached draft probabilities for request %s; "
                    "falling back to legacy speculative rejection behavior.",
                    req_id,
                )
                return None
            draft_probs_rows.append(self._draft_probs[row_idx, :num_draft])

        if not draft_probs_rows:
            return None
        # ------【投机解码】拼接各请求草稿概率为连续张量返回 ------
        return torch.cat(draft_probs_rows, dim=0).contiguous()

    def propose_draft_token_ids(
        self,
        scheduler_output: "SchedulerOutput",
        sampled_token_ids: torch.Tensor | list[list[int]],
        sampling_metadata: SamplingMetadata,
        hidden_states: torch.Tensor,
        sample_hidden_states: torch.Tensor,
        aux_hidden_states: list[torch.Tensor] | None,
        spec_decode_metadata: SpecDecodeMetadata | None,
        common_attn_metadata: CommonAttentionMetadata,
        slot_mappings: dict[str, torch.Tensor] | list[dict[str, torch.Tensor]] | None,
    ) -> list[list[int]] | torch.Tensor:
        # ------【投机解码】读取调度 token 数与投机配置，确定本步需生成的投机 token 数 ------
        num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        spec_config = self.speculative_config
        assert spec_config is not None
        num_spec_tokens_to_schedule = scheduler_output.num_spec_tokens_to_schedule
        # ------【投机解码】重置草稿概率缓存，避免复用上一 step 的概率 ------
        self._draft_probs = None
        self._draft_prob_req_ids = None
        # ------【投机解码】ngram 草稿器：基于已见 token 的 n-gram 匹配生成草稿序列 ------
        if spec_config.method == "ngram":
            from vllm.v1.spec_decode.ngram_proposer import NgramProposer

            assert isinstance(sampled_token_ids, list)
            assert isinstance(self.drafter, NgramProposer)
            draft_token_ids = self.drafter.propose(
                num_spec_tokens_to_schedule,
                sampled_token_ids,
                self.input_batch.num_tokens_no_spec,
                self.input_batch.token_ids_cpu,
                slot_mappings=slot_mappings,
            )
        # ------【投机解码】自定义草稿器：直接调用用户类 propose 生成草稿 token ------
        elif spec_config.method == "custom_class":
            assert isinstance(sampled_token_ids, list)
            draft_token_ids = cast(Any, self.drafter).propose(
                sampled_token_ids,
                self.input_batch.num_tokens_no_spec,
                self.input_batch.token_ids_cpu,
                slot_mappings=slot_mappings,
            )
        # ------【投机解码 + 异步 RPC】GPU ngram 更新 token 状态，并异步拷贝有效计数到 CPU ------
        elif spec_config.use_ngram_gpu():
            assert isinstance(self.drafter, NgramProposerGPU)
            (
                next_token_ids,
                valid_sampled_tokens_count,
                valid_sampled_token_ids_gpu,
            ) = self.drafter.update_token_ids_ngram(
                sampled_token_ids,
                self.input_batch,
                self.token_ids_gpu_tensor,
                self.num_tokens_no_spec_gpu,
                self.discard_request_mask.gpu,
            )
            self._copy_valid_sampled_token_count(
                next_token_ids, valid_sampled_tokens_count
            )

            # ------【投机解码】GPU ngram 生成草稿 token 与有效草稿计数 ------
            batch_size = next_token_ids.shape[0]

            draft_token_ids, num_valid_draft_tokens = self.drafter.propose(
                num_spec_tokens_to_schedule,
                self.num_tokens_no_spec_gpu[:batch_size],
                self.token_ids_gpu_tensor[:batch_size],
                valid_sampled_token_ids_gpu,
                valid_sampled_tokens_count,
            )

            # ------【投机解码】缓存有效草稿数供调度器端裁剪草稿长度 ------
            # Cache valid draft counts for scheduler-side trimming.
            self._num_valid_draft_tokens = num_valid_draft_tokens

            # ------【异步 RPC】在专用流上异步拷贝有效草稿数到 CPU ------
            # Async D2H copy on a dedicated stream.
            copy_num_valid_draft_tokens(
                self._num_valid_draft_tokens_cpu,
                self._num_valid_draft_tokens_copy_stream,
                self._num_valid_draft_tokens_event,
                self._num_valid_draft_tokens,
                self.input_batch.num_reqs,
            )
        # ------【投机解码】suffix 草稿器：基于后缀匹配生成草稿 token ------
        elif spec_config.method == "suffix":
            assert isinstance(sampled_token_ids, list)
            assert isinstance(self.drafter, SuffixDecodingProposer)
            draft_token_ids = self.drafter.propose(
                num_spec_tokens_to_schedule,
                self.input_batch,
                sampled_token_ids,
                slot_mappings=slot_mappings,
            )
        elif spec_config.method == "medusa":
            assert isinstance(sampled_token_ids, list)
            assert isinstance(self.drafter, MedusaProposer)

            # ------【投机解码】medusa 用采样位置隐藏态作输入，按有无草稿 token 决定截取/索引 ------
            if sample_hidden_states.shape[0] == len(sampled_token_ids):
                # The input to the target model does not include draft tokens.
                hidden_states = sample_hidden_states
            else:
                indices = []
                offset = 0
                assert spec_decode_metadata is not None, (
                    "No spec decode metadata for medusa"
                )
                for num_draft, tokens in zip(
                    spec_decode_metadata.num_draft_tokens, sampled_token_ids
                ):
                    indices.append(offset + len(tokens) - 1)
                    offset += num_draft + 1
                indices = async_tensor_h2d(indices, device=self.device)
                hidden_states = sample_hidden_states[indices]

            # ------【投机解码】medusa 多头预测草稿 token ------
            draft_token_ids = self.drafter.propose(
                num_speculative_tokens=num_spec_tokens_to_schedule,
                target_hidden_states=hidden_states,
                sampling_metadata=sampling_metadata,
                slot_mappings=slot_mappings,
            )
        elif spec_config.uses_extract_hidden_states():
            assert isinstance(self.drafter, ExtractHiddenStatesProposer)
            assert isinstance(sampled_token_ids, torch.Tensor), (
                "sampled_token_ids should be a torch.Tensor for "
                "extract_hidden_states method."
            )
            if not self.use_aux_hidden_state_outputs or aux_hidden_states is None:
                raise ValueError(
                    "aux_hidden_states are required when using `extract_hidden_states`"
                )
            # ------【投机解码】从辅助隐藏态截取本步 token 作为草稿器目标隐藏态 ------
            target_hidden_states = [h[:num_scheduled_tokens] for h in aux_hidden_states]

            # ------【投机解码】extract_hidden_states 草稿器基于目标隐藏态生成草稿 ------
            draft_token_ids = self.drafter.propose(
                num_speculative_tokens=num_spec_tokens_to_schedule,
                sampled_token_ids=sampled_token_ids,
                target_hidden_states=target_hidden_states,
                common_attn_metadata=common_attn_metadata,
                slot_mappings=slot_mappings,
            )
            # ------【投机解码 + 异步 RPC】准备下一 token 并异步拷贝有效计数到 CPU ------
            next_token_ids, valid_sampled_tokens_count = (
                self.drafter.prepare_next_token_ids_padded(
                    sampled_token_ids,
                    self.requests,
                    self.input_batch,
                    self.discard_request_mask.gpu,
                )
            )
            self._copy_valid_sampled_token_count(
                next_token_ids, valid_sampled_tokens_count
            )

        # ------【投机解码】EAGLE/DFlash/DraftModel 分支：校验草稿器类型并准备输入 ------
        elif (
            spec_config.use_eagle()
            or spec_config.use_dflash()
            or spec_config.uses_draft_model()
        ):
            assert isinstance(
                self.drafter,
                EagleProposer | DFlashProposer | DraftModelProposer | Gemma4Proposer,
            )

            # ------【投机解码】禁用 padded batch：用 CPU 侧有效采样 token 列表准备下一 token ------
            if spec_config.disable_padded_drafter_batch:
                # When padded-batch is disabled, the sampled_token_ids should be
                # the cpu-side list[list[int]] of valid sampled tokens for each
                # request, with invalid requests having empty lists.
                assert isinstance(sampled_token_ids, list), (
                    "sampled_token_ids should be a python list when"
                    "padded-batch is disabled."
                )
                next_token_ids = self.drafter.prepare_next_token_ids_cpu(
                    sampled_token_ids,
                    self.requests,
                    self.input_batch,
                    scheduler_output.num_scheduled_tokens,
                )
            else:
                # ------【投机解码 + 异步 RPC】padded batch：用 GPU 张量准备下一 token 并异步拷贝有效计数 ------
                # When using padded-batch, the sampled_token_ids should be
                # the gpu tensor of sampled tokens for each request, of shape
                # (num_reqs, num_spec_tokens + 1) with rejected tokens having
                # value -1.
                assert isinstance(sampled_token_ids, torch.Tensor), (
                    "sampled_token_ids should be a torch.Tensor when"
                    "padded-batch is enabled."
                )
                next_token_ids, valid_sampled_tokens_count = (
                    self.drafter.prepare_next_token_ids_padded(
                        sampled_token_ids,
                        self.requests,
                        self.input_batch,
                        self.discard_request_mask.gpu,
                    )
                )
                self._copy_valid_sampled_token_count(
                    next_token_ids, valid_sampled_tokens_count
                )

            # ------【投机解码】允许模型覆盖喂给草稿器的隐藏态（如 DeepSeek V4 MTP 残差） ------
            # Let the target override the hidden state fed to the drafter
            # (e.g. DeepSeek V4 MTP needs the pre-hc_head residual). Safe to
            # rebind here: hidden_states was already consumed for sampling
            # above and is not used again in this branch.
            alt = getattr(
                self.get_model(), "get_mtp_target_hidden_states", lambda: None
            )()
            if alt is not None:
                hidden_states = alt

            num_rejected_tokens_gpu = None
            # ------【投机解码】无投机元数据时用本步全部 token 作为目标 token/位置/隐藏态 ------
            if spec_decode_metadata is None:
                token_indices_to_sample = None
                # input_ids can be None for multimodal models.
                target_token_ids = self.input_ids.gpu[:num_scheduled_tokens]
                target_positions = self._get_positions(num_scheduled_tokens)
                if self.use_aux_hidden_state_outputs:
                    assert aux_hidden_states is not None
                    target_hidden_states = torch.cat(
                        [h[:num_scheduled_tokens] for h in aux_hidden_states], dim=-1
                    )
                else:
                    target_hidden_states = hidden_states[:num_scheduled_tokens]
            else:
                # ------【投机解码】有投机元数据时按 padded/非 padded 准备目标 token、位置与隐藏态 ------
                if spec_config.disable_padded_drafter_batch:
                    token_indices_to_sample = None
                    common_attn_metadata, token_indices = self.drafter.prepare_inputs(
                        common_attn_metadata,
                        sampled_token_ids,
                        spec_decode_metadata.num_draft_tokens,
                    )
                    target_token_ids = self.input_ids.gpu[token_indices]
                    target_positions = self._get_positions(token_indices)
                    if self.use_aux_hidden_state_outputs:
                        assert aux_hidden_states is not None
                        target_hidden_states = torch.cat(
                            [h[token_indices] for h in aux_hidden_states], dim=-1
                        )
                    else:
                        target_hidden_states = hidden_states[token_indices]
                else:
                    (
                        common_attn_metadata,
                        token_indices_to_sample,
                        num_rejected_tokens_gpu,
                    ) = self.drafter.prepare_inputs_padded(
                        common_attn_metadata,
                        spec_decode_metadata,
                        valid_sampled_tokens_count,
                    )
                    total_num_tokens = common_attn_metadata.num_actual_tokens
                    # When padding the batch, token_indices is just a range
                    target_token_ids = self.input_ids.gpu[:total_num_tokens]
                    target_positions = self._get_positions(total_num_tokens)
                    if self.use_aux_hidden_state_outputs:
                        assert aux_hidden_states is not None
                        target_hidden_states = torch.cat(
                            [h[:total_num_tokens] for h in aux_hidden_states], dim=-1
                        )
                    else:
                        target_hidden_states = hidden_states[:total_num_tokens]

            # ------【核心逻辑】多模态模型为草稿器收集多模态嵌入输入 ------
            if self.supports_mm_inputs and self.drafter.supports_mm_inputs:
                mm_embed_inputs = self._gather_mm_embeddings(
                    scheduler_output,
                    shift_computed_tokens=1,
                )
            else:
                mm_embed_inputs = None

            # ------【投机解码】EAGLE/DFlash/DraftModel 统一调用 propose 生成草稿 token ------
            draft_token_ids = self.drafter.propose(
                num_speculative_tokens=num_spec_tokens_to_schedule,
                target_token_ids=target_token_ids,
                target_positions=target_positions,
                target_hidden_states=target_hidden_states,
                next_token_ids=next_token_ids,
                token_indices_to_sample=token_indices_to_sample,
                sampling_metadata=sampling_metadata,
                common_attn_metadata=common_attn_metadata,
                mm_embed_inputs=mm_embed_inputs,
                num_rejected_tokens_gpu=num_rejected_tokens_gpu,
                slot_mappings=slot_mappings,
            )
            # ------【投机解码】若草稿器提供草稿概率则缓存，供拒绝采样使用 ------
            if hasattr(self.drafter, "take_last_draft_probs"):
                draft_probs = self.drafter.take_last_draft_probs()
                if draft_probs is not None:
                    self._draft_probs = draft_probs
                    self._draft_prob_req_ids = self.input_batch.req_ids.copy()

        return draft_token_ids

    def update_config(self, overrides: dict[str, Any]) -> None:
        # ------【核心逻辑】限定可覆盖配置白名单，运行时仅允许热更新 load_config/model_config ------
        allowed_config_names = {"load_config", "model_config"}
        for config_name, config_overrides in overrides.items():
            # ------【核心逻辑】逐项校验覆盖键，非白名单配置直接报错防止误改 ------
            if config_name not in allowed_config_names:
                allowed = ", ".join(sorted(allowed_config_names))
                raise ValueError(
                    f"Config override '{config_name}' is not supported. "
                    f"Supported configs: {allowed}"
                )
            # ------【核心逻辑】用覆盖项生成新配置并写回实例属性完成热更新 ------
            config = getattr(self, config_name)
            new_config = update_config(config, config_overrides)
            setattr(self, config_name, new_config)








    @instrument(span_name="Loading (GPU)")
    def load_model(self, load_dummy_weights: bool = False) -> None:
        """
        Args:
            load_dummy_weights: load dummy weights instead of real weights.
        """
        logger.info_once(
            "Starting to load model %s...",
            self.model_config.model,
            scope="global",
        )

        # ------【EP/EPLB】初始化专家并行负载均衡状态，用于 MoE 模型的专家路由与负载统计 ------
        if self.parallel_config.enable_eplb:
            self.eplb_state = EplbState(self.parallel_config, self.device)
            eplb_models = 0

        # ------【显存 profiling + TP/PP】在显存分析器内加载主模型权重（由 loader 按 TP/PP 切分并放置到各设备）──
        try:
            with DeviceMemoryProfiler() as m: # 设备内存实际测试器
                time_before_load = time.perf_counter()
                if load_dummy_weights:
                    self.load_config.load_format = "dummy"

                ##########################################################################################################
                #  1. 获取模型加载器， 加载模型：构造模型实例 + 拷贝权重
                ##########################################################################################################
                model_loader = get_model_loader(self.load_config) # modelrunner里面构造一个model_loader 模型加载器
                self.model = model_loader.load_model( # 加载模型 = 构造模型实例 + 拷贝权重
                    vllm_config=self.vllm_config, model_config=self.model_config
                )




                # ------【LoRA】加载 LoRA 低秩适配器并合并到主模型 ------
                if self.lora_config:
                    self.model = self.load_lora_model( # 加载lora微调模型
                        self.model, self.vllm_config, self.device
                    )

                
                # ------【投机解码】加载草稿模型（draft model），供投机采样生成候选 token ------
                if hasattr(self, "drafter"):
                    logger.info_once("Loading drafter model...")
                    if hasattr(self.drafter, "load_model"):
                        self.drafter.load_model(self.model) # 加载草稿模型
                    # ------【投机解码 + EP/EPLB】草稿模型若为 MoE，同样纳入专家负载均衡管理 ------
                    if (
                        hasattr(self.drafter, "model")
                        and is_mixture_of_experts(self.drafter.model)
                        and self.parallel_config.enable_eplb
                    ):
                        assert not self.parallel_config.enable_elastic_ep, (
                            "Elastic EP is not supported with drafter model."
                        )
                        spec_config = self.vllm_config.speculative_config
                        assert spec_config is not None
                        assert spec_config.draft_model_config is not None
                        logger.info_once(
                            "EPLB is enabled for drafter model %s.",
                            spec_config.draft_model_config.model,
                        )
                        if self.eplb_state is None:
                            self.eplb_state = EplbState(
                                self.parallel_config, self.device
                            )
                        self.eplb_state.add_model(
                            self.drafter.model,
                            spec_config.draft_model_config,
                        )
                        assert hasattr(self.drafter, "set_eplb_state")
                        self.drafter.set_eplb_state(self.eplb_state)
                        eplb_models += 1

                # ------【投机解码】配置 EAGLE3 辅助隐藏层输出，供草稿模型复用主模型隐藏态 ------
                self._setup_eagle3_aux_hidden_state_outputs()

                # ------【EP/EPLB】解析 MoE 模型（解包 VLM 包装层），并将主模型加入专家负载均衡 ------
                # Resolve the MoE model, unwrapping VLM wrappers if needed.
                # VLM models (e.g. KimiK25ForConditionalGeneration) wrap the
                # actual MoE language model but don't implement
                # MixtureOfExperts themselves.
                moe_candidate = self.model
                if not is_mixture_of_experts(moe_candidate) and isinstance(
                    moe_candidate, SupportsMultiModal
                ):
                    moe_candidate = moe_candidate.get_language_model()
                if is_mixture_of_experts(moe_candidate):
                    self._moe_model = moe_candidate

                if (
                    self._moe_model is not None
                    and self.parallel_config.enable_eplb
                    and not load_dummy_weights
                ):
                    logger.info_once(
                        "EPLB is enabled for model %s.",
                        self.model_config.model,
                    )
                    assert self.eplb_state is not None
                    self.eplb_state.add_model(
                        self._moe_model,
                        self.model_config,
                    )
                    eplb_models += 1

                # ------【显存 profiling】记录加载结束时间与模型实际显存占用 ------
                time_after_load = time.perf_counter()


            self.model_memory_usage = m.consumed_memory # 模型权重占用显存
        # ------【显存 profiling】显存不足时给出降低显存占用的友好提示并重新抛出 ------
        except torch.cuda.OutOfMemoryError as e:
            msg = (
                "Failed to load model - not enough GPU memory. "
                "Try lowering --gpu-memory-utilization to free memory for weights, "
                "increasing --tensor-parallel-size, or using --quantization. "
                "See https://docs.vllm.ai/en/latest/configuration/conserving_memory/ "
                "for more tips."
            )
            combined_msg = f"{msg} (original error: {e})"
            logger.error(combined_msg)
            raise e

        
        # ------【显存 profiling】打印模型加载耗时与显存占用日志 ------
        logger.info_once(
            "Model loading took %s GiB memory and %.6f seconds",
            format_gib(self.model_memory_usage),
            time_after_load - time_before_load,
        )

        # ------【核心逻辑】读取多模态配置，标记是否启用多模态剪枝与顺序视频编码 ------
        # 多模态配置
        mm_config = self.model_config.multimodal_config
        self.is_multimodal_pruning_enabled = (
            supports_multimodal_pruning(self.get_model())
            and mm_config is not None
            and mm_config.is_multimodal_pruning_enabled()
        )
        self.requires_sequential_video_encoding = hasattr(
            self.get_model(), "requires_sequential_video_encoding"
        )  # Temporary hack for dynamic res video w/o support for bs>1 yet

        # ------【EP/EPLB】异步 EPLB 模式下启动后台负载均衡循环 ------
        if (
            self._moe_model is not None
            and self.parallel_config.enable_eplb
            and not load_dummy_weights
            and self.eplb_state is not None
            and self.eplb_state.is_async
        ):
            self.eplb_state.start_async_loop()

        # ------【CUDA Graph】stock torch.compile 模式：整图编译后直接返回，不再走后续 CUDA Graph 包装 ------
        if (
            self.vllm_config.compilation_config.mode
            == CompilationMode.STOCK_TORCH_COMPILE
        ):
            from vllm.env_override import _apply_constrain_to_fx_strides_patch

            _apply_constrain_to_fx_strides_patch()
            backend = self.vllm_config.compilation_config.init_backend(self.vllm_config)
            compilation_counter.stock_torch_compile_count += 1
            self.model.compile(fullgraph=True, backend=backend)
            return


        # for other compilation modes, cudagraph behavior is controlled by
        # CudagraphWrapper and CudagraphDispatcher of vllm.

        # ------【CUDA Graph】按 cudagraph 模式为模型选择包装器：breakable / full / ubatching ------
        # wrap the model with full cudagraph wrapper if needed.
        # 包装一层cudagraph模式
        ##################################################################################
        # 加载完模型后，如果我们启用了cudagraph_mode, 那么包装一层cudagraphwrapper
        ##################################################################################
        cudagraph_mode = self.compilation_config.cudagraph_mode
        '''
        cudagraph_mode 是一个 enum值，这个值可以是二元组，表示我们启用何种模式图

        class CUDAGraphMode(enum.Enum):
            NONE = 0
            PIECEWISE = 1
            FULL = 2
            FULL_DECODE_ONLY = (FULL, NONE)      # 二元组
            FULL_AND_PIECEWISE = (FULL, PIECEWISE)  # 二元组

            def decode_mode(self): ...   # 取元组第一个
            def mixed_mode(self): ...    # 取元组第二个
        '''
        assert cudagraph_mode is not None
        if (
            is_breakable_cudagraph_enabled()
            and cudagraph_mode != CUDAGraphMode.NONE # 启用了cuda graph
            and not self.parallel_config.use_ubatching # ubatch是微批次
        ):
            self.model = BreakableCUDAGraphWrapper(self.model, self.vllm_config)
            drafter = getattr(self, "drafter", None)
            if drafter is not None and hasattr(drafter, "model"):
                drafter.model = BreakableCUDAGraphWrapper(
                    drafter.model, self.vllm_config
                )
        elif (
            cudagraph_mode.has_full_cudagraphs() # FULL_AND_PIECEWISE
            and not self.parallel_config.use_ubatching # 不使用微批次
        ):
            #########################################
            # 开始包装我们的模型
            #########################################
            self.model = CUDAGraphWrapper( # 在self.model外面包一层cudagraph包装器
                self.model, self.vllm_config, runtime_mode=CUDAGraphMode.FULL
            )
        elif self.parallel_config.use_ubatching:
            if cudagraph_mode.has_full_cudagraphs():
                self.model = UBatchWrapper(
                    self.model, self.vllm_config, CUDAGraphMode.FULL, self.device
                )
            else:
                self.model = UBatchWrapper(
                    self.model, self.vllm_config, CUDAGraphMode.NONE, self.device
                )

        # ------【显存 profiling】初始化卸载器，管理受限显存下的权重/KV 卸载 ------
        get_offloader().post_init()




















    def _setup_eagle3_aux_hidden_state_outputs(self) -> None:
        # ------【投机解码】未启用 EAGLE3 辅助隐藏态输出时直接返回，跳过后续配置 ------
        if not self.use_aux_hidden_state_outputs:
            return

        # ------【投机解码】校验主模型支持 EAGLE3 接口，不支持则报错 ------
        if not supports_eagle3(self.get_model()):
            raise RuntimeError(
                "Model does not support EAGLE3 interface but "
                "aux_hidden_state_outputs was requested"
            )
        # ------【投机解码】优先从投机配置取辅助层索引，缺失时回退模型默认层 ------
        # Try to get auxiliary layers from speculative config,
        # otherwise use model's default layers
        aux_layers = self._get_eagle3_aux_layers_from_config()
        if aux_layers:
            logger.info(
                "Using auxiliary layers from speculative config: %s", aux_layers
            )
        else:
            aux_layers = self.model.get_eagle3_default_aux_hidden_state_layers()

        # ------【投机解码】把辅助隐藏层索引注入主模型，供 EAGLE3 草稿模型复用主模型隐藏态 ------
        self.model.set_aux_hidden_state_layers(aux_layers)

    def _get_eagle3_aux_layers_from_config(self) -> tuple[int, ...] | None:
        """Extract Eagle3 auxiliary layer indices from speculative config.

        These indices specify which hidden states from the base model should
        be used as auxiliary inputs for the Eagle3 drafter model during
        speculative decoding.

        Returns:
            Tuple of layer indices if found in draft model config,
            None otherwise.
        """
        # ------【投机解码】无草稿模型配置时不存在辅助层，直接返回 None ------
        if not (self.speculative_config and self.speculative_config.draft_model_config):
            return None

        # ------【投机解码】读取草稿模型 HF 配置，从中解析 EAGLE3 辅助层声明 ------
        hf_config = self.speculative_config.draft_model_config.hf_config

        # ------【投机解码】优先取标准字段 eagle_aux_hidden_state_layer_ids ------
        layer_ids = getattr(hf_config, "eagle_aux_hidden_state_layer_ids", None)
        # ------【投机解码】标准字段缺失时回退解析 dflash/eagle 嵌套配置 ------
        if not layer_ids:
            dflash_config = getattr(hf_config, "dflash_config", None)
            eagle_config = getattr(hf_config, "eagle_config", None)

            # ------【投机解码】DFlash 的 target_layer_ids 语义需 +1 对齐层编号 ------
            if dflash_config and isinstance(dflash_config, dict):
                # Add 1 to convert DFlash's aux layer id semantics
                layer_ids = [
                    i + 1 for i in (dflash_config.get("target_layer_ids") or [])
                ]

            # ------【投机解码】再尝试从 eagle_config 读辅助层字段作为最后回退 ------
            if eagle_config and isinstance(eagle_config, dict):
                layer_ids = eagle_config.get("eagle_aux_hidden_state_layer_ids")

        # ------【投机解码】结果校验为序列后转 tuple 返回，否则返回 None ------
        if layer_ids and isinstance(layer_ids, (list, tuple)):
            return tuple(layer_ids)

        return None

    def reload_weights(
        self,
        weights_iterator: Iterable[tuple[str, torch.Tensor]] | None = None,
        weights_path: str | None = None,
        is_checkpoint_format: bool = True,
    ) -> None:
        """
        Reload weights from a weights iterator or from disk

        Args:
            weights_iterator: weights to load into model
            weights_path: path to load weights from if weights_iterator is not
                provided. Use path of original model if neither is provided.
            is_checkpoint_format: set to False if weights have already been
                processed into kernel format (repacking, renaming, etc.)
        """
        # TODO(@kylesayrs): generalize to all runners and loaders
        # ------【权重传输】参数校验：非 checkpoint 格式必须由调用方提供迭代器，否则告警 ------
        # argument validation
        if weights_iterator is None and not is_checkpoint_format:
            logger.warning(
                "Reloading from disk means that weights will be in checkpoint format. "
                "Please use `is_checkpoint_format=True` "
                "to avoid weight reloading errors"
            )

        # ------【权重传输】收集模型全部待加载参数名（LoRA 场景去掉 base_layer 前缀） ------
        model = self.get_model()
        weights_to_load = {
            name.replace(".base_layer.", ".") if self.lora_config else name
            for name, _ in model.named_parameters()
        }
        # ------【权重传输】记录重载开始时间用于后续耗时统计 ------
        counter_before_reloading = time.perf_counter()

        # ------【权重传输】未提供迭代器时，按 load_format 从磁盘读取原始 checkpoint 权重 ------
        # load weights from disk if none are provided
        if weights_iterator is None:
            model_loader = get_model_loader(self.load_config)
            if not hasattr(model_loader, "get_all_weights"):
                raise NotImplementedError(
                    f"Model reloading with `{self.load_config.load_format}` format"
                )

            # ------【权重传输】指定新路径时切换模型来源并清空旧 revision，防止沿用旧仓库版本 ------
            if weights_path is not None:
                # The revision belongs to the model we are reloading away from,
                # so it must not be carried over to the new path.
                self.model_config.model = weights_path
                self.model_config.revision = None
            weights_iterator = model_loader.get_all_weights(self.model_config, model)
            weights_iterator = cast(
                Iterable[tuple[str, torch.Tensor]], weights_iterator
            )

        # ------【权重传输】按格式分支执行权重写入 ------
        # begin loading weights
        logger.info_once("Reloading weights inplace...")
        if is_checkpoint_format:
            # ------【权重传输】checkpoint 原始格式：经层式重载框架逐层加载权重 ------
            # load weights from checkpoint/ original model format
            initialize_layerwise_reload(model)
            loaded_weights = model.load_weights(weights_iterator)
            finalize_layerwise_reload(model, self.model_config)

        else:
            # ------【权重传输】kernel 格式（已切分/重打包）：逐参数就地 copy 写入 ------
            # load weights from kernel format
            logger.warning_once(
                "Reloading with `is_checkpoint_format=True` requires that "
                "weights be in kernel format and already sharded",
            )
            loaded_weights = set()
            for name, loaded_weight in weights_iterator:
                param = _get_parameter_for_reload(model, name)  # TODO: buffers?
                param.copy_(loaded_weight)
                loaded_weights.add(name)

        # ------【LoRA】重载权重后重置 LoRA 状态，避免旧适配器参数残留 ------
        self.reset_lora_state()

        # ------【权重传输】统计重载耗时并打印日志 ------
        # logging and validation
        counter_after_reloading = time.perf_counter()
        diff_seconds = counter_after_reloading - counter_before_reloading
        logger.info_once(
            "Reloading and processing weights took %.2f seconds",
            diff_seconds,
        )
        # ------【权重传输】非量化模型校验是否有权重未成功加载，发现则告警 ------
        if self.model_config.quantization is None and loaded_weights is not None:
            weights_not_loaded = weights_to_load - loaded_weights
            if weights_not_loaded:
                logger.warning(
                    "Following weights were not loaded from checkpoint: %s",
                    weights_not_loaded,
                )

        # ------【核心逻辑】权重更新后清空编码器与多模态缓存，防止陈旧特征被复用 ------
        self.reset_encoder_cache()
        self.reset_mm_cache()

    def _get_prompt_logprobs_dict(
        self,
        hidden_states: torch.Tensor,
        num_scheduled_tokens: dict[str, int],
    ) -> dict[str, LogprobsTensors | None]:
        num_prompt_logprobs_dict = self.num_prompt_logprobs
        # ------【核心逻辑】无请求需要 prompt logprobs 时直接返回空字典 ------
        if not num_prompt_logprobs_dict:
            return {}

        prompt_logprobs_dict: dict[str, LogprobsTensors | None] = {}

        # ------【核心逻辑】逐个请求计算 prompt token 的 logprob（稀有特性，重在可维护） ------
        # Since prompt logprobs are a rare feature, prioritize simple,
        # maintainable loop over optimal performance.
        completed_prefill_reqs = []
        for req_id, num_prompt_logprobs in num_prompt_logprobs_dict.items():
            num_tokens = num_scheduled_tokens.get(req_id)
            # ------【核心逻辑】请求在 prefill 阶段被抢占时无调度 token，跳过 ------
            if num_tokens is None:
                # This can happen if the request was preempted in prefill stage.
                continue

            # Get metadata for this request.
            request = self.requests[req_id]
            # ------【核心逻辑】prompt logprobs 与 prompt embedding 互斥，跳过 embedding 请求 ------
            if request.prompt_token_ids is None:
                # Prompt logprobs is incompatible with prompt embeddings
                continue

            num_prompt_tokens = len(request.prompt_token_ids)
            # ------【异步 RPC】prompt token ids 非阻塞 H2D 拷贝到设备 ------
            prompt_token_ids = async_tensor_h2d(
                request.prompt_token_ids, device=self.device
            )

            # ------【核心逻辑】首次创建覆盖整个 prompt 的空 CPU 结果张量，后续分块填充 ------
            # Set up target LogprobsTensors object.
            logprobs_tensors = request.in_progress_prompt_logprobs_cpu
            if logprobs_tensors is None:
                # Create empty logprobs CPU tensors for the entire prompt.
                # If chunked, we'll copy in slice by slice.
                logprobs_tensors = LogprobsTensors.empty_cpu(
                    num_prompt_tokens - 1, num_prompt_logprobs + 1
                )
                request.in_progress_prompt_logprobs_cpu = logprobs_tensors

            # ------【chunked prefill】按已计算 token 数确定本次要取多少个 logit ------
            # Determine number of logits to retrieve.
            start_idx = request.num_computed_tokens
            start_tok = start_idx + 1
            num_remaining_tokens = num_prompt_tokens - start_tok
            # ------【chunked prefill】判断是否为最后一个 chunk，决定本次 logit 数并登记完成 ------
            if num_tokens <= num_remaining_tokens:
                # This is a chunk, more tokens remain.
                # In the == case, there are no more prompt logprobs to produce
                # but we want to defer returning them to the next step where we
                # have new generated tokens to return.
                num_logits = num_tokens
            else:
                # This is the last chunk of prompt tokens to return.
                num_logits = num_remaining_tokens
                completed_prefill_reqs.append(req_id)
                prompt_logprobs_dict[req_id] = logprobs_tensors

            if num_logits <= 0:
                # This can happen for the final chunk if we prefilled exactly
                # (num_prompt_tokens - 1) tokens for this request in the prior
                # step. There are no more prompt logprobs to produce.
                continue

            # ------【核心逻辑】按请求在批次中的偏移切出隐藏态并计算 logits ------
            # Get the logits corresponding to this req's prompt tokens.
            # If this is a partial request (i.e. chunked prefill),
            # then there is prompt logprob generated for each index.
            req_idx = self.input_batch.req_id_to_index[req_id]
            offset = self.query_start_loc.np[req_idx].item()
            prompt_hidden_states = hidden_states[offset : offset + num_logits]
            logits = self.model.compute_logits(prompt_hidden_states)

            # Get the "target" tokens for each index. For prompt at index i,
            # the token at prompt index i+1 is the "sampled" token we want
            # to gather the logprob for.
            tgt_token_ids = prompt_token_ids[start_tok : start_tok + num_logits]

            # Compute prompt scores respecting logprobs_mode.
            # NOTE: prompt tokens skip sampling processors, so
            # processed_* and raw_* yield the same scores here.
            # ------【核心逻辑】按 logprobs_mode 计算分数并 gather 出 top-k logprob ------
            if self.model_config.logprobs_mode in ("raw_logits", "processed_logits"):
                scores = logits.to(torch.float32)
            else:
                scores = self.sampler.compute_logprobs(logits)
            token_ids, logprobs, ranks, _ = self.sampler.gather_logprobs(
                scores, num_prompt_logprobs, tgt_token_ids
            )

            # ------【异步 RPC】结果非阻塞 GPU->CPU 拷贝到对应 chunk 切片 ------
            # Transfer GPU->CPU async.
            chunk_slice = slice(start_idx, start_idx + num_logits)
            logprobs_tensors.logprob_token_ids[chunk_slice].copy_(
                token_ids, non_blocking=True
            )
            logprobs_tensors.logprobs[chunk_slice].copy_(logprobs, non_blocking=True)
            logprobs_tensors.selected_token_ranks[chunk_slice].copy_(
                ranks, non_blocking=True
            )

        # ------【核心逻辑】移除已完成 prefill 的请求并释放其 CPU 张量引用 ------
        # Remove requests that have completed prefill from the batch
        # num_prompt_logprobs_dict.
        for req_id in completed_prefill_reqs:
            del num_prompt_logprobs_dict[req_id]
            self.requests[req_id].in_progress_prompt_logprobs_cpu = None

        # ------【异步 RPC】同步设备，确保非阻塞拷贝完成后再返回结果 ------
        # Must synchronize the non-blocking GPU->CPU transfers.
        if prompt_logprobs_dict:
            self._sync_device()

        return prompt_logprobs_dict

    def _get_nans_in_logits(
        self,
        logits: torch.Tensor | None,
    ) -> dict[str, int]:
        try:
            # ------【核心逻辑】logits 为 None 时所有请求记为 0 个 NaN ------
            if logits is None:
                return {req_id: 0 for req_id in self.input_batch.req_ids}

            num_nans_in_logits = {}
            # ------【核心逻辑】按 token 统计每行 logits 的 NaN 个数并搬回 CPU ------
            num_nans_for_index = logits.isnan().sum(dim=-1).cpu().numpy()
            # ------【核心逻辑】把每个请求的 NaN 计数映射回 req_id ------
            for req_id in self.input_batch.req_ids:
                req_index = self.input_batch.req_id_to_index[req_id]
                num_nans_in_logits[req_id] = (
                    int(num_nans_for_index[req_index])
                    if num_nans_for_index is not None and req_index < logits.shape[0]
                    else 0
                )
            # ------【核心逻辑】环境变量开启时对 NaN logits 主动抛错定位问题 ------
            if envs.VLLM_RAISE_ON_LOGIT_NANS:
                raise_if_nan_logits(num_nans_in_logits)
            return num_nans_in_logits
        except IndexError:
            return {}

    @contextmanager
    def maybe_randomize_inputs(
        self, input_ids: torch.Tensor | None, inputs_embeds: torch.Tensor | None
    ):
        """
        Randomize input_ids if VLLM_RANDOMIZE_DP_DUMMY_INPUTS is set.
        This is to help balance expert-selection
         - during profile_run
         - during DP rank dummy run
        """

        dp_size = self.vllm_config.parallel_config.data_parallel_size
        # ------【DP】仅当开启环境变量且 DP 规模>1 时，对 dummy 输入做随机化 ------
        randomize_inputs = envs.VLLM_RANDOMIZE_DP_DUMMY_INPUTS and dp_size > 1
        if not randomize_inputs:
            yield
        elif input_ids is not None:

            @functools.cache
            def rand_input_ids() -> torch.Tensor:
                return torch.randint_like(
                    self.input_ids.gpu,
                    low=0,
                    high=self.model_config.get_vocab_size(),
                )

            # ------【DP+EP/EPLB】随机化 input_ids 使各 DP rank 命中不同专家，均衡专家选择 ------
            logger.debug_once("Randomizing dummy input_ids for DP Rank")
            input_ids.copy_(rand_input_ids()[: input_ids.size(0)], non_blocking=True)
            yield
            input_ids.fill_(0)
        else:

            @functools.cache
            def rand_inputs_embeds() -> torch.Tensor:
                return torch.randn_like(
                    self.inputs_embeds.gpu,
                )

            # ------【DP】embedding 输入同样随机化并跑完 dummy 后清零 ------
            assert inputs_embeds is not None
            logger.debug_once("Randomizing dummy inputs_embeds for DP Rank")
            inputs_embeds.copy_(
                rand_inputs_embeds()[: inputs_embeds.size(0)], non_blocking=True
            )
            yield
            inputs_embeds.fill_(0)

    def _get_mm_dummy_batch(
        self,
        modality: str,
        max_items_per_batch: int,
    ) -> BatchedTensorInputs:
        """Dummy data for profiling and precompiling multimodal models."""
        assert self.mm_budget is not None

        # ------【显存 profiling】生成单个模态的 dummy 多模态输入用于显存估计 ------
        # Don't use `max_items_per_batch` here to avoid redundant computation
        dummy_mm_inputs = self.mm_registry.get_dummy_mm_inputs(
            self.model_config,
            mm_counts={modality: 1},
            cache=self.mm_budget.cache,
        )
        dummy_mm_item = dummy_mm_inputs["mm_kwargs"][modality][0]

        # We use the cache so that the item is saved to the cache,
        # but not read from the cache
        assert dummy_mm_item is not None, "Item should not already be cached"

        return next(
            mm_kwargs_batch
            for _, _, mm_kwargs_batch in group_and_batch_mm_kwargs(
                [(modality, dummy_mm_item)] * max_items_per_batch,
                device=self.device,
                pin_memory=PIN_MEMORY,
            )
        )




    ########################################################
    # 实际按照配置，跑一次profile 的 dummy run
    ########################################################
    '''
    这个函数有两个作用：
        1. profiling 测显存
        2. capture阶段的各种图捕获

    所以会有3个调用场景：
        1. profiling阶段，构造纯prefill batch
        2. warmup, capture的捕图阶段，由desc.uniform觉得
            true -> 纯decode
            false -> 混合
        3. flashinfer warmup 专用，decode + 1个prefill的特殊形状
    '''
    @torch.inference_mode()
    def _dummy_run(
        self,
        num_tokens: int, # 假前向的batch的token数
        cudagraph_runtime_mode: CUDAGraphMode | None = None, # cudagraph模式mode
        force_attention: bool = False, # True表示创建 attention metadata, 当mode = None的时候用来预热attention后端
        uniform_decode: bool = False, # 是否是均匀decode batch
        allow_microbatching: bool = True,
        skip_eplb: bool = False,
        is_profile: bool = False, # 设置为 profile run
        create_mixed_batch: bool = False, # 构建混合batch
        remove_lora: bool = True,
        is_graph_capturing: bool = False, # 是图捕获使能
        num_active_loras: int = 0,
        profile_seq_lens: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run a dummy forward pass to warm up/profile run or capture the
        CUDA graph for the model.

        Args:
            num_tokens: Number of tokens to run the dummy forward pass.
            cudagraph_runtime_mode: used to control the behavior.
                - if not set will determine the cudagraph mode based on using
                    the self.cudagraph_dispatcher.
                - CUDAGraphMode.NONE: No cudagraph, for warm up and profile run
                - CUDAGraphMode.PIECEWISE: Piecewise cudagraph.
                - CUDAGraphMode.FULL: Full cudagraph, attention metadata is
                    needed.
            force_attention: If True, always create attention metadata. Used to
                warm up attention backend when mode is NONE.
            uniform_decode: If True, the batch is a uniform decode batch.
            skip_eplb: If True, skip EPLB state update.
            is_profile: If True, this is a profile run.
            create_mixed_batch: If True, create a mixed batch with both decode
                (1 token) and prefill (multiple tokens) requests.
            remove_lora: If False, dummy LoRAs are not destroyed after the run
            num_active_loras: Number of distinct active LoRAs to capture for.
                LoRA is activated when num_active_loras > 0.
            profile_seq_lens: If provided, use this value for seq_lens instead
                of max_query_len. Used to profile attention workspace that
                scales with context length.
        """
        mm_config = self.vllm_config.model_config.multimodal_config
        # ------【核心逻辑】纯多模态编码器模型无需 LM dummy 前向，直接返回空张量 ------
        if mm_config and mm_config.mm_encoder_only:
            # The current dummy run only covers LM execution, so we can skip it.
            # mm encoder dummy run may need to add in the future.
            return torch.tensor([]), torch.tensor([])

        # ------【CUDA Graph】校验传入的 cudagraph 运行时模式合法 ------
        assert (
            cudagraph_runtime_mode is None
            or cudagraph_runtime_mode.is_valid_runtime_mode()
        )

        # If cudagraph_mode.decode_mode() == FULL and
        # cudagraph_mode.separate_routine(). This means that we are using
        # different graphs and/or modes for mixed prefill-decode batches vs.
        # uniform decode batches. A uniform decode batch means that all
        # requests have identical query length, except a potential virtual
        # request (shorter) in the batch account for padding.
        # Uniform decode batch could either be common pure decode, where
        # max_query_len == 1, or speculative decode, where
        # max_query_len == 1 + num_spec_decode_tokens.

        # When setting max_query_len = 1, we switch to and capture the optimized
        # routine of FA2 for pure decode, i.e., Flashdecode + an optimization
        # for GQA/MQA.
        # ------【CUDA Graph】统一 decode 批次用固定 query_len，否则用 num_tokens ------

        #########################################################一个req的最长可以到整个预算
        max_query_len = self.uniform_decode_query_len if uniform_decode else num_tokens

        # ------【核心逻辑】构造 dummy 批次的每请求 token 分配，满足总数==num_tokens ------
        # Set num_scheduled_tokens based on num_tokens and max_num_seqs
        # for dummy run with LoRA so that the num_reqs collectively
        # has num_tokens in total.









        '''
        _dummy_run, 假跑，本身既做显存profiling, 也做cudagraph的捕获。

        num_scheduled_tokens_list， 给cudagraph捕获构造一个 固定形状的 batch
        '''
        # 判断dummy batch的每个req的token分配
        assert num_tokens <= self.max_num_tokens
        max_num_reqs = self.scheduler_config.max_num_seqs


        #############################
        # 捕获的图：prefill+decode 混合图
        # 前段 N 个 decode 请求（各 1 token）+ 1 个 prefill 请求（多 token）
        #############################
        # ------【CUDA Graph】混合 prefill-decode 批次：前段 decode 请求 + 一个 prefill 请求 ------
        if create_mixed_batch:
            assert not uniform_decode
            # Create mixed batch:
            # first half decode tokens, second half one prefill
            # 一半的token用来作为decode, 一半的token塞在一个prefill里面
            num_decode_tokens = min(max_num_reqs - 1, num_tokens // 2)
            num_prefill_tokens = num_tokens - num_decode_tokens
            num_reqs = num_decode_tokens + 1

            # Create decode requests (1 token each) followed by prefill request
            num_scheduled_tokens_list = [1] * num_decode_tokens + [num_prefill_tokens]
            # Note: Overriding max_query_len to be the prefill tokens
            max_query_len = num_prefill_tokens
        # ------【CUDA Graph】统一 decode 批次：每个请求固定 max_query_len 个 token ------
        #############################
        # 捕获的图：纯decode图
        # 每个请求固定 max_query_len（=1 或 1+投机 token 数）
        #############################
        elif uniform_decode:
            assert not create_mixed_batch
            num_reqs = min(max_num_reqs, cdiv(num_tokens, max_query_len))
            num_scheduled_tokens_list = [max_query_len] * num_reqs
            if num_tokens % max_query_len != 0:
                num_scheduled_tokens_list[-1] = num_tokens % max_query_len
        # ------【核心逻辑】通用批次：token 尽可能均匀分到各请求 ------
        ############################
        # 捕获的图：通用/prefill 图
        # token 尽量均匀分到各请求
        ############################
        else:
            # 这里才是我们的逻辑
            ########################################################
            # 1. 指定dummy batch的参数
            ########################################################
            num_reqs = min(num_tokens, max_num_reqs) # batch的req数
            min_tokens_per_req = num_tokens // num_reqs # 每个req的平均token预算
            num_scheduled_tokens_list = [min_tokens_per_req] * num_reqs # 构造一个每个req的token数的list
            num_scheduled_tokens_list[-1] += num_tokens % num_reqs # 余数塞给最后一个req

        assert sum(num_scheduled_tokens_list) == num_tokens
        assert len(num_scheduled_tokens_list) == num_reqs














        # ------【核心逻辑】转成 numpy 数组并记录未 padding 的真实 token 数 ------
        num_scheduled_tokens = np.array(num_scheduled_tokens_list, dtype=np.int32) # 每个req分几个token
        num_tokens_unpadded = int(num_scheduled_tokens.sum()) # 真实总数（每个req未padding）

        num_sampled_tokens = np.ones(num_reqs, dtype=np.int32) # 每个req decode 1个的list

        # ------【CUDA Graph】确定批次执行方式（是否 ubatch/DP 切分/是否 padding） ------
        ###################### 
        # padding，然后调度
        # 这里是主动捕获，所以，直接拿调度器里面预设的档位key来，这里重复padding 档位。而调度器的档位数值一开始初始化的时候按照tp对齐了，所以
        # 这里是有点多余的
        ###################################################
        _cudagraph_mode, batch_desc, should_ubatch, num_tokens_across_dp, _ = (
            '''
            给定一个batch的真实形状，他来决定：
                1. 用哪种图 cudagraph_mode : FULL/PIECEWISE/NONE
                2. padding到多少token, CUDA graph要求形状固定，所以真实token数要向上补到某个档位
                    
                不是必须的。是否 padding 完全取决于 dispatch 命中了哪种模式。看 dispatch() 的三个出口（cudagraph_dispatcher.py:283-326）：

                命中模式	                    padding 程度	                    原因
                FULL	                token + 请求数都要 pad	            图形状完全写死，必须精确匹配 (num_tokens, num_reqs)
                PIECEWISE	            只 pad token 数	                    用 relaxed key num_reqs=None（318），请求数不限，但 token 数仍要补到桶
                NONE（eager）	        完全不额外 pad	                    没图可回放，直接 BatchDescriptor(num_tokens) 原样返回（326）

                

            所以对"混合 / 纯 prefill"来说
            关键看有没有命中的图：

            命中了 FULL 图 → 必须 pad（token 和请求数都 pad）。
            只命中 PIECEWISE → 只 pad token 数。
            没命中 / cudagraph 关了 / 超了 max_cudagraph_capture_size → 直接 NONE，不 pad。

            '''
            self._determine_batch_execution_and_padding(
                num_tokens=num_tokens_unpadded, # 未padding的这个batch的总token数
                num_reqs=num_reqs, # batch的请求数
                num_scheduled_tokens_np=num_scheduled_tokens, # 每个req分几个token作为prompt的numpy数组
                max_num_scheduled_tokens=max_query_len, # 最大被调度的token数
                use_cascade_attn=False,
                allow_microbatching=allow_microbatching,
                force_eager=is_profile
                or (cudagraph_runtime_mode == CUDAGraphMode.NONE),
                # `force_uniform_decode` is used for cudagraph capture; because for
                # capturing mixed prefill-decode batches, we sometimes use
                # num_tokens == num_reqs which looks like a uniform decode batch to the
                # dispatcher; but we actually want to capture a piecewise cudagraph
                force_uniform_decode=uniform_decode,
                # `force_has_lora` is used for cudagraph capture; because LoRA is
                # activated later in the context manager, but we need to know the
                # LoRA state when determining the batch descriptor for capture
                force_has_lora=num_active_loras > 0,
                # `force_num_active_loras` is used for cudagraph capture; because we
                # need to capture graphs for specific num_active_loras counts
                force_num_active_loras=num_active_loras,
            )
        )

        # ------【CUDA Graph】调用方未指定时采用推断出的模式，否则校验二者一致 ------
        if cudagraph_runtime_mode is None:
            cudagraph_runtime_mode = _cudagraph_mode
        else:
            assert cudagraph_runtime_mode == _cudagraph_mode, (
                f"Cudagraph runtime mode mismatch in dummy_run. "
                f"Expected {_cudagraph_mode}, but got {cudagraph_runtime_mode}."
            )

        # ------【CUDA Graph】取 padding 后的 token/请求数，供固定形状图使用 ------
        num_tokens_padded = batch_desc.num_tokens
        num_reqs_padded = (
            batch_desc.num_reqs if batch_desc.num_reqs is not None else num_reqs
        )
        # ------【DP】计算 DCP 上下文长度并生成 ubatch 切片 ------
        dcp_dummy_context_len = get_dcp_dummy_context_len(
            self.dcp_world_size,
            self.parallel_config.cp_kv_cache_interleave_size,
            hasattr(self, "kv_cache_config"),
            create_mixed_batch,
            is_graph_capturing,
            uniform_decode,
        )
        ubatch_slices, ubatch_slices_padded = maybe_create_ubatch_slices(
            should_ubatch,
            num_scheduled_tokens,
            num_tokens_padded,
            num_reqs_padded,
            self.vllm_config.parallel_config.num_ubatches,
        )
        logger.debug(
            "ubatch_slices: %s, ubatch_slices_padded: %s",
            ubatch_slices,
            ubatch_slices_padded,
        )

        attn_metadata: PerLayerAttnMetadata | None = None

        # ------【核心逻辑】为 dummy 批次生成 slot 映射，用于 KV cache 写入地址 ------
        ########################################################
        # 2. 创建kvcache槽位，
        # vllm的策略是通过把模型权重算上，然后让激活值达到peak， 
        # 这样就可以计算出我们的kvcache的最大可用容量，所以dummy batch并不需要占满所有的kvcache， 只要把batch的最大token数，req数打满，让激活值最大就行了
        ########################################################
        slot_mappings_by_group, slot_mappings = self._get_slot_mappings(
            num_tokens_padded=num_tokens_padded,
            num_reqs_padded=num_reqs_padded,
            num_tokens_unpadded=num_tokens_unpadded,
            ubatch_slices=ubatch_slices_padded,
        )

        # ------【核心逻辑】dummy 运行无真实 KV 槽位，填 -1 让 concat_and_cache 跳过写入 ------
        # Dummy runs have no real slot assignments — fill with -1 so
        # concat_and_cache kernels skip the KV write.
        if slot_mappings_by_group is not None:
            for sm in slot_mappings_by_group.values():
                sm.fill_(-1)

        # _dummy_run shares pinned CPU buffers (seq_lens, query_start_loc,
        # etc.) with execute_model.  It must participate in the same event
        # protocol so that back-to-back dummy/real steps don't overwrite
        # pinned memory while a prior non_blocking H2D DMA is still reading.
        # ------【异步 RPC】与真实执行共享 pinned 缓冲，走同一事件协议避免 H2D DMA 覆盖 ------
        with self.synchronize_input_prep():
            # If force_attention is True, we always capture attention.
            # Otherwise, it only happens for cudagraph_runtime_mode=FULL.
            # ------【CUDA Graph】仅 FULL 图或强制时才构造 attention 元数据 ------
            '''
            FULL图捕获，需要捕获attention, 所以需要记录下attention的metadata地址
            PIECEWISE图捕获，不录制attention， 所以前向推理直接传参
            '''
            if force_attention or cudagraph_runtime_mode == CUDAGraphMode.FULL:
                if profile_seq_lens is not None:
                    seq_lens = profile_seq_lens  # type: ignore[assignment]
                elif create_mixed_batch:
                    # In the mixed batch mode (used for FI warmup), we use
                    # shorter sequence lengths to run faster.
                    # TODO(luka) better system for describing dummy batches
                    if dcp_dummy_context_len > 0:
                        seq_lens = torch.tensor(  # type: ignore[assignment]
                            [1 + dcp_dummy_context_len] * num_decode_tokens
                            + [num_prefill_tokens + dcp_dummy_context_len],
                            dtype=torch.int,
                        )
                    else:
                        seq_lens = torch.tensor(  # type: ignore[assignment]
                            [1] * num_decode_tokens + [num_prefill_tokens + 1],
                            dtype=torch.int,
                        )
                elif dcp_dummy_context_len > 0:
                    seq_lens = max_query_len + dcp_dummy_context_len  # type: ignore[assignment]
                else:
                    seq_lens = max_query_len  # type: ignore[assignment]
                # ------【CUDA Graph】写入 dummy 序列长度并拷贝到设备，构造固定形状输入 ------
                self.optimistic_seq_lens_cpu[:num_reqs] = seq_lens
                self.optimistic_seq_lens_cpu[num_reqs:].fill_(0)
                self.seq_lens.copy_(self.optimistic_seq_lens_cpu, non_blocking=True)

                # ------【核心逻辑】计算累计 token 数填充 query_start_loc 前缀和 ------
                cum_num_tokens = self._get_cumsum_and_arange(
                    num_scheduled_tokens, self.query_pos.np
                )
                self.query_start_loc.np[1 : num_reqs + 1] = cum_num_tokens
                self.query_start_loc.np[num_reqs + 1 : num_reqs_padded + 1].fill(
                    cum_num_tokens[-1]
                )
                self.query_start_loc.copy_to_gpu()

                # ------【DP】为 DCP 构造 dummy 上下文元数据（position/query_start_loc 等） ------
                prepare_dcp_dummy_context_metadata(
                    input_batch=self.input_batch,
                    kv_cache_config=getattr(self, "kv_cache_config", None),
                    query_pos=self.query_pos,
                    positions=self.positions,
                    query_start_loc=self.query_start_loc,
                    num_reqs=num_reqs,
                    num_tokens_unpadded=num_tokens_unpadded,
                    dcp_dummy_context_len=dcp_dummy_context_len,
                )

                # ------【核心逻辑】同步 block_table 到设备，让已清理行对元数据构建可见 ------
                # Sync block table CPU->GPU so cleared rows from
                # remove_request() are visible to the attention metadata
                # builder. Without this, stale block IDs from finished
                # requests can corrupt Mamba state.
                self.input_batch.block_table.commit_block_table(num_reqs_padded)

                # ------【CUDA Graph】FULL 图按 padding 形状构建 attention 元数据 ------
                pad_attn = cudagraph_runtime_mode == CUDAGraphMode.FULL
                attn_metadata, _ = self._build_attention_metadata(
                    num_tokens=num_tokens_unpadded, # 原来的batch token数
                    num_tokens_padded=num_tokens_padded if pad_attn else None, # padding后的batch tokens数
                    num_reqs=num_reqs_padded, # padding后的req数
                    max_query_len=max_query_len, # 每个req的最大q长度
                    ubatch_slices=(ubatch_slices_padded if pad_attn else ubatch_slices),
                    # FULL replay reads capture-time metadata buffers. Re-stage them
                    # from the zeroed dummy block tables instead of retaining state
                    # indices from the previous real batch.
                    for_cudagraph_capture=( # 需要录制这个metadata的地址
                        is_graph_capturing
                        or cudagraph_runtime_mode == CUDAGraphMode.FULL
                    ),
                    slot_mappings=slot_mappings_by_group,# slot mapping
                    use_spec_decode=self.speculative_config is not None,
                )

        # ------【LoRA】在 LoRA 上下文管理器内激活 dummy LoRA 以捕获相应图 ------
        with self.maybe_dummy_run_with_lora(
            self.lora_config,
            num_scheduled_tokens,
            num_sampled_tokens,
            remove_lora,
            num_active_loras,
        ):
            # Make sure padding doesn't exceed max_num_tokens
            assert num_tokens_padded <= self.max_num_tokens
            model_kwargs = self._init_model_kwargs()
            # ------【核心逻辑】按是否多模态/embedding 选择输入 id 或 embeds ------
            if self.supports_mm_inputs and not self.model_config.is_encoder_decoder:
                input_ids, inputs_embeds = self._prepare_mm_inputs(num_tokens_padded)

                model_kwargs = {
                    **model_kwargs,
                    **self._dummy_mm_kwargs(num_reqs),
                }
            elif self.enable_prompt_embeds:
                input_ids = None
                inputs_embeds = self.inputs_embeds.gpu[:num_tokens_padded]
                model_kwargs = self._init_model_kwargs()
            else:
                input_ids = self.input_ids.gpu[:num_tokens_padded]
                inputs_embeds = None

            # ------【核心逻辑】按位置编码类型（mrope/xdrope/普通）取对应 position 张量 ------
            if self.uses_mrope:
                positions = self.mrope_positions.gpu[:, :num_tokens_padded]
            elif self.uses_xdrope_dim > 0:
                positions = self.xdrope_positions.gpu[:, :num_tokens_padded]
            else:
                positions = self.positions[:num_tokens_padded]

            # ------【PP】非首 rank 需准备/接收中间张量做流水线 dummy 前向 ------
            if get_pp_group().is_first_rank:
                intermediate_tensors = None
            else:
                if self.intermediate_tensors is None:
                    self.intermediate_tensors = (
                        self.model.make_empty_intermediate_tensors(
                            batch_size=self.max_num_tokens,
                            dtype=self.model_config.dtype,
                            device=self.device,
                        )
                    )

                intermediate_tensors = self.sync_and_gather_intermediate_tensors(
                    num_tokens_padded, None, False
                )

            # ------【CUDA Graph】ubatch 模式把 padded 值调整为单 ubatch 的形状 ------
            if ubatch_slices_padded is not None:
                # Adjust values to reflect a single ubatch.
                # TODO(sage,lucas): this is cruft that should be addressed in
                #  the padding refactor.
                num_tokens_padded = ubatch_slices_padded[0].num_tokens
                if num_tokens_across_dp is not None:
                    num_tokens_across_dp[:] = num_tokens_padded

            # ------【CUDA Graph】在随机化输入与 forward 上下文中执行一次模型前向 ------



            with (
                self.maybe_randomize_inputs(input_ids, inputs_embeds),
                set_forward_context( # 设置cuda graph录制的上下文
                    attn_metadata,
                    self.vllm_config,
                    num_tokens=num_tokens_padded, # 假batch输入
                    num_tokens_across_dp=num_tokens_across_dp,
                    cudagraph_runtime_mode=cudagraph_runtime_mode,# mode
                    batch_descriptor=batch_desc, # key
                    ubatch_slices=ubatch_slices_padded,
                    slot_mapping=slot_mappings, # slot mapping, 写入kvcache 地址
                ),
            ):
                ########################################################
                # 3. 这边开始前向推理 dummy batch
                ########################################################
                outputs = self.model(
                    input_ids=input_ids,
                    positions=positions,
                    intermediate_tensors=intermediate_tensors,
                    inputs_embeds=inputs_embeds,
                    **model_kwargs,
                )

            # ------【投机解码】EAGLE3 辅助输出开启时输出是 (hidden_states, aux) 二元组 ------
            if self.use_aux_hidden_state_outputs:
                hidden_states, _ = outputs
            else:
                hidden_states = outputs

            # ------【投机解码】对草稿模型做 dummy 前向以捕获其图 ------
            if self.speculative_config and (
                self.speculative_config.use_eagle()
                or self.speculative_config.uses_draft_model()
                or self.speculative_config.uses_extract_hidden_states()
            ):
                assert isinstance(
                    self.drafter,
                    EagleProposer
                    | DFlashProposer
                    | DraftModelProposer
                    | ExtractHiddenStatesProposer
                    | Gemma4Proposer,
                )
                assert self.speculative_config is not None
                # Eagle currently only supports PIECEWISE cudagraphs.
                # Therefore only use cudagraphs if the main model uses PIECEWISE
                # NOTE(lucas): this is a hack, need to clean up.
                use_cudagraphs = (
                    (
                        is_graph_capturing
                        and cudagraph_runtime_mode == CUDAGraphMode.PIECEWISE
                    )
                    or (
                        not is_graph_capturing
                        and cudagraph_runtime_mode != CUDAGraphMode.NONE
                    )
                ) and not self.speculative_config.enforce_eager

                # Note(gnovack) - We need to disable cudagraphs for one of the two
                # lora cases when cudagraph_specialize_lora is enabled. This is a
                # short term mitigation for issue mentioned in
                # https://github.com/vllm-project/vllm/issues/28334
                if (
                    self.compilation_config.cudagraph_specialize_lora
                    and num_active_loras > 0
                ):
                    use_cudagraphs = False

                self.drafter.dummy_run(
                    num_tokens,
                    use_cudagraphs=use_cudagraphs,
                    is_graph_capturing=is_graph_capturing,
                    slot_mappings=slot_mappings,
                )

        # ------【CUDA Graph】首次 dynamo 追踪完成后注册 NVTX 钩子，避免钩子被追踪断图 ------
        # We register layerwise NVTX hooks here after the first dynamo tracing is
        # done to avoid nvtx operations in hook functions being traced by
        # torch dynamo and causing graph breaks.
        # Note that for DYNAMO_ONCE and VLLM_COMPILE mode,
        # compiled model's dynamo tracing is only done once and the compiled model's
        # __call__ function is replaced by calling the compiled function.
        # So it's safe to register hooks here. Hooks will be registered to
        # both compiled and uncompiled models but they will never
        # be called on the compiled model execution path.
        self._register_layerwise_nvtx_hooks()

        # This is necessary to avoid blocking DP.
        # For dummy runs, we typically skip EPLB since we don't have any real
        # requests to process.
        # However, in DP settings, there may be cases when some DP ranks do
        # not have any requests to process, so they're executing dummy batches.
        # In such cases, we still have to trigger EPLB to make sure
        # ranks execute the rearrangement in synchronization.
        # ------【EP/EPLB】DP 场景下即使 dummy 批次也要触发 EPLB 以保持各 rank 同步重排 ------
        if not skip_eplb:
            self.eplb_step(is_dummy=True, is_profile=is_profile)

        # ------【核心逻辑】取每个请求最后一个 token 的 logits 作为采样结果返回 ------
        logit_indices = np.cumsum(num_scheduled_tokens) - 1
        logit_indices_device = torch.from_numpy(logit_indices).to(
            self.device, non_blocking=True
        )


        ########################################################
        # 返回结果
        ########################################################
        return hidden_states, hidden_states[logit_indices_device]





























    @torch.inference_mode()
    def _dummy_sampler_run(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        # The dummy hidden states may contain special values,
        # like `inf` or `nan`.
        # To avoid breaking the sampler, we use a random tensor here instead.

        mm_config = self.vllm_config.model_config.multimodal_config
        # ------【核心逻辑】纯编码器模型无需采样器 warmup，直接返回 ------
        if mm_config and mm_config.mm_encoder_only:
            # MM Encoder only model no need to run sampler.
            return torch.tensor([])

        # ------【显存 profiling】用随机张量替换可能含 inf/nan 的 dummy 隐藏态，保证采样器不崩 ------
        hidden_states = torch.rand_like(hidden_states)

        # ------【核心逻辑】计算 logits 并据此构造 dummy 采样元数据 ------
        logits = self.model.compute_logits(hidden_states)
        num_reqs = logits.size(0)

        dummy_tensors = lambda v: torch.full((num_reqs,), v, device=self.device)

        dummy_metadata = SamplingMetadata(
            temperature=dummy_tensors(0.5),
            all_greedy=False,
            all_random=False,
            top_p=dummy_tensors(0.9),
            top_k=dummy_tensors(logits.size(1) - 1),
            generators={},
            max_num_logprobs=None,
            logprob_token_ids=None,
            no_penalties=True,
            prompt_token_ids=None,
            frequency_penalties=dummy_tensors(0.1),
            presence_penalties=dummy_tensors(0.1),
            repetition_penalties=dummy_tensors(0.1),
            output_token_ids=[[] for _ in range(num_reqs)],
            spec_token_ids=[[] for _ in range(num_reqs)],
            allowed_token_ids_mask=None,
            bad_words_token_ids={},
            logitsprocs=LogitsProcessors(),
        )
        # ------【显存 profiling】跑一次采样器，并额外预热 forward_native 路径 ------
        try:
            sampler_output = self.sampler(
                logits=logits, sampling_metadata=dummy_metadata
            )
            # Also warm forward_native (taken when generators dict is non-empty),
            # but skip the extra call in 'processed_logits' / 'processed_logprobs'
            # modes — there TopKTopPSampler binds forward = forward_native at
            # init time, so the warmup call is redundant and only inflates peak
            # memory during profile_run.
            # No .clone() of logits: warmup output is discarded, so any in-place
            # mutation by forward_native does not affect correctness.
            if self.sampler.logprobs_mode not in PROCESSED_LOGPROBS_MODES:
                self.sampler(
                    logits=logits,
                    sampling_metadata=replace(
                        dummy_metadata,
                        generators={
                            0: torch.Generator(device=self.device).manual_seed(0)
                        },
                    ),
                )
        # ------【显存 profiling】采样器 OOM 时给出降配提示并重新抛出 ------
        except RuntimeError as e:
            if "out of memory" in str(e):
                raise RuntimeError(
                    "CUDA out of memory occurred when warming up sampler with "
                    f"{num_reqs} dummy requests. Please try lowering "
                    "`max_num_seqs` or `gpu_memory_utilization` when "
                    "initializing the engine."
                ) from e
            else:
                raise e
        # ------【投机解码】dummy 预热拒绝采样器，覆盖投机解码路径 ------
        if self.speculative_config:
            draft_token_ids = [[0] for _ in range(num_reqs)]
            dummy_spec_decode_metadata = SpecDecodeMetadata.make_dummy(
                draft_token_ids, self.device
            )

            num_tokens = sum(len(ids) for ids in draft_token_ids)
            draft_probs = None
            if (
                self.speculative_config.rejection_sample_method == "standard"
                and self.speculative_config.draft_sample_method == "probabilistic"
            ):
                draft_probs = torch.rand(
                    num_tokens,
                    logits.shape[-1],
                    device=self.device,
                    dtype=torch.float32,
                )
                draft_probs = torch.softmax(draft_probs, dim=-1)
            logits = torch.randn(
                num_tokens + num_reqs,
                logits.shape[-1],
                device=self.device,
                dtype=logits.dtype,
            )
            self.rejection_sampler(
                dummy_spec_decode_metadata,
                draft_probs,
                logits,
                dummy_metadata,
            )
        return sampler_output

    def _dummy_pooler_run_task(
        self,
        hidden_states: torch.Tensor,
        task: PoolingTask,
    ) -> PoolerOutput:
        # ------【核心逻辑】把隐藏态 token 均匀拆成若干 dummy 池化请求 ------
        num_tokens = hidden_states.shape[0]
        max_num_reqs = self.scheduler_config.max_num_seqs
        num_reqs = min(num_tokens, max_num_reqs)
        min_tokens_per_req = num_tokens // num_reqs
        num_scheduled_tokens_np = np.full(num_reqs, min_tokens_per_req)
        num_scheduled_tokens_np[-1] += num_tokens % num_reqs
        assert np.sum(num_scheduled_tokens_np) == num_tokens
        assert len(num_scheduled_tokens_np) == num_reqs

        req_num_tokens = num_tokens // num_reqs

        dummy_prompt_lens = torch.from_numpy(num_scheduled_tokens_np)
        dummy_token_ids = torch.zeros(
            (num_reqs, req_num_tokens), dtype=torch.int32, device=self.device
        )

        model = cast(VllmModelForPooling, self.get_model())
        dummy_pooling_params = PoolingParams(task=task)
        dummy_pooling_params.verify(self.model_config)
        to_update = model.pooler.get_pooling_updates(task)
        to_update.apply(dummy_pooling_params)

        # ------【核心逻辑】构造 dummy 池化元数据并构建游标 ------
        dummy_metadata = PoolingMetadata(
            prompt_lens=dummy_prompt_lens,
            prompt_token_ids=dummy_token_ids,
            prompt_token_ids_cpu=dummy_token_ids.cpu(),
            pooling_params=[dummy_pooling_params] * num_reqs,
            pooling_states=[PoolingStates() for i in range(num_reqs)],
        )

        dummy_metadata.build_pooling_cursor(
            num_scheduled_tokens_np,
            seq_lens_cpu=dummy_prompt_lens,
            device=hidden_states.device,
        )

        # ------【显存 profiling】跑一次 pooler 并在 OOM 时给出降配提示 ------
        try:
            return model.pooler(
                hidden_states=hidden_states, pooling_metadata=dummy_metadata
            )
        except RuntimeError as e:
            if "out of memory" in str(e):
                raise RuntimeError(
                    "CUDA out of memory occurred when warming up pooler "
                    f"({task=}) with {num_reqs} dummy requests. Please try "
                    "lowering `max_num_seqs` or `gpu_memory_utilization` when "
                    "initializing the engine."
                ) from e
            else:
                raise e

    @torch.inference_mode()
    def _dummy_pooler_run(
        self,
        hidden_states: torch.Tensor,
    ) -> PoolerOutput:
        mm_config = self.vllm_config.model_config.multimodal_config
        if mm_config and mm_config.mm_encoder_only:
            # MM Encoder only model not need to run pooler.
            return torch.tensor([])

        # Find the task that has the largest output for subsequent steps
        supported_pooling_tasks = self.get_supported_pooling_tasks()

        if not supported_pooling_tasks:
            raise RuntimeError(
                f"Model {self.model_config.model} does not support "
                "any pooling tasks. See "
                "https://docs.vllm.ai/en/latest/models/pooling_models.html "
                "to learn more."
            )

        # ------【显存 profiling】遍历所有池化任务，找出输出最大者以兜底显存峰值 ------
        output_size = dict[PoolingTask, float]()
        for task in supported_pooling_tasks:
            # Run a full batch with each task to ensure none of them OOMs
            output = self._dummy_pooler_run_task(hidden_states, task)
            output_size[task] = sum(o.nbytes for o in output if o is not None)
            del output  # Allow GC

        max_task = max(output_size.items(), key=lambda x: x[1])[0]
        return self._dummy_pooler_run_task(hidden_states, max_task)












    ########################################################
    # 来执行一次dummy的profile run
    ########################################################
    def profile_run(self) -> None:
        # ------【显存 profiling】先用多模态编码器跑 dummy，估算 encoder 与缓存显存 ------
        # Profile with multimodal encoder & encoder cache.
        # 多模态相关，跳过
        if self.supports_mm_inputs:
            mm_config = self.model_config.multimodal_config
            if mm_config is not None and mm_config.skip_mm_profiling:
                logger.info(
                    "Skipping memory profiling for multimodal encoder and "
                    "encoder cache."
                )
            else:
                mm_budget = self.mm_budget
                assert mm_budget is not None

                if (encoder_budget := mm_budget.get_encoder_budget()) > 0:
                    if not mm_budget.mm_max_toks_per_item:
                        # All modality limits are 0 — embedding-only mode.
                        # Budget is non-zero for embedding storage, but
                        # there's no encoder to profile.
                        logger.info(
                            "Skipping encoder profiling for embedding-only "
                            "mode (all modality limits=0 with "
                            "enable_mm_embeds=True).",
                        )
                    else:
                        # NOTE: Currently model is profiled with a single
                        # non-text modality with the max possible input
                        # tokens even when it supports multiple.
                        dummy_modality = mm_budget.get_modality_with_max_tokens()
                        max_mm_items_per_batch = mm_budget.mm_max_items_per_batch[
                            dummy_modality
                        ]

                        logger.info_once(
                            "Encoder cache will be initialized with a "
                            "budget of %s tokens, and profiled with "
                            "%s %s items of the maximum feature size.",
                            encoder_budget,
                            max_mm_items_per_batch,
                            dummy_modality,
                        )

                        # ------【显存 profiling】生成最大尺寸的 dummy 多模态批次并跑编码器 ------
                        # Create dummy batch of multimodal inputs.
                        batched_dummy_mm_inputs = self._get_mm_dummy_batch(
                            dummy_modality,
                            max_mm_items_per_batch,
                        )

                        # Run multimodal encoder.
                        dummy_encoder_outputs = self.model.embed_multimodal(
                            **batched_dummy_mm_inputs
                        )

                        sanity_check_mm_encoder_outputs(
                            dummy_encoder_outputs,
                            expected_num_items=max_mm_items_per_batch,
                        )
                        for i, output in enumerate(dummy_encoder_outputs):
                            self.encoder_cache[f"tmp_{i}"] = output

        # ------【显存 profiling】用最大 token 数做 dummy 前向，预分配通信缓冲并触发显存分配 ------
        # Add `is_profile` here to pre-allocate communication buffers
        ########################################################
        # 实际跑一次
        ########################################################
        hidden_states, last_hidden_states = self._dummy_run(
            self.max_num_tokens, is_profile=True 
        )


        if get_pp_group().is_last_rank:
            if self.is_pooling_model:
                output = self._dummy_pooler_run(hidden_states)
            else:
                output = self._dummy_sampler_run(last_hidden_states)
        else:
            output = None
        # ------【显存 profiling】同步设备后释放临时张量并清空缓存触发回收 ------
        self._sync_device()
        del hidden_states, output
        self.encoder_cache.clear()
        gc.collect()































    def _init_minimal_kv_cache_for_profiling(self) -> None:
        from vllm.v1.core.kv_cache_utils import (
            get_kv_cache_config_from_groups,
            get_kv_cache_groups,
        )

        # ------【显存 profiling】生成 KV cache 规格并计算最小块数，用于图捕获显存测量 ------
        kv_cache_spec = self.get_kv_cache_spec()
        KVCacheSpecRegistry.check_kv_cache_spec_registry(kv_cache_spec)
        kv_cache_groups = get_kv_cache_groups(self.vllm_config, kv_cache_spec)
        # the minimum number of blocks required is 1 block *per sequence*
        min_blocks = (
            min(self.max_num_reqs, self.compilation_config.max_cudagraph_capture_size)
            or 1
        )

        # ------【显存 profiling】临时改写 override 生成最小配置后立即还原，仅用于测量 ------
        # Temporarily change num_gpu_blocks_override to allocate a minimal KV cache
        saved_override = self.cache_config.num_gpu_blocks_override
        self.cache_config.num_gpu_blocks_override = min_blocks
        minimal_config = get_kv_cache_config_from_groups(
            self.vllm_config, kv_cache_groups, available_memory=0
        )
        self.cache_config.num_gpu_blocks_override = saved_override

        self.initialize_kv_cache(minimal_config, is_profiling=True)
        self.cache_config.num_gpu_blocks = minimal_config.num_blocks

        logger.debug("Initialized minimal KV cache for CUDA graph profiling")

    @staticmethod
    @contextmanager
    def _freeze_gc():
        # ------【显存 profiling】捕获前冻结 GC，避免回收打断图捕获导致显存抖动 ------
        gc.collect()
        should_freeze = not envs.VLLM_ENABLE_CUDAGRAPH_GC
        if should_freeze:
            gc.freeze()
        try:
            yield
        finally:
            if should_freeze:
                gc.unfreeze()
                gc.collect()

    def shutdown(self) -> None:
        """Release GPU tensors (model weights, KV caches, workspace) so that
        memory is reclaimable when running in the same process."""
        from vllm.model_executor.layers.rotary_embedding import _ROPE_DICT
        from vllm.v1.worker.workspace import reset_workspace_manager

        # ------【显存 profiling】先清理 profiling KV cache 并同步设备 ------
        # Calls torch.accelerator.synchronize()
        self._cleanup_profiling_kv_cache()
        if current_platform.is_rocm():
            # Drop captured graphs before distributed teardown. On ROCm, delayed
            # graph destruction can surface HSA faults in the next engine startup.
            CUDAGraphWrapper.clear_all_graphs()
            BreakableCUDAGraphWrapper.clear_all_graphs()
            self.encoder_cudagraph_manager = None
        # ------【核心逻辑】释放模型、静态前向上下文与 RoPE 缓存，便于同进程内回收显存 ------
        self.compilation_config.static_forward_context.clear()
        self.model = None  # type: ignore[assignment]
        _ROPE_DICT.clear()

        reset_workspace_manager()
        if current_platform.is_rocm() or current_platform.is_xpu():
            gc.collect()
            torch.accelerator.empty_cache()
            torch.accelerator.synchronize()

    def _cleanup_profiling_kv_cache(self) -> None:
        # ------【显存 profiling】先同步设备再逐项释放 KV cache 张量引用 ------
        torch.accelerator.synchronize()
        if hasattr(self, "kv_caches") and self.kv_caches:
            for i in range(len(self.kv_caches)):
                self.kv_caches[i] = None  # type: ignore
            self.kv_caches.clear()
        if hasattr(self, "cross_layers_kv_cache"):
            self.cross_layers_kv_cache = None
            self.cross_layers_attn_backend = None
        if hasattr(self, "attn_groups"):
            self.attn_groups.clear()
        if hasattr(self, "kv_cache_config"):
            delattr(self, "kv_cache_config")
        self.cache_config.num_gpu_blocks = None

        # ------【核心逻辑】清理各层静态前向上下文中的 kv_cache 与量化 scale 视图 ------
        for layer in self.compilation_config.static_forward_context.values():
            if hasattr(layer, "kv_cache"):
                kv_cache = layer.kv_cache
                layer.kv_cache = (
                    torch.tensor([]) if isinstance(kv_cache, torch.Tensor) else []
                )
            # Clean up quantized KV cache scale views
            # (int8_per_token_head, fp8_per_token_head)
            if hasattr(layer, "impl"):
                if hasattr(layer.impl, "_k_scale_cache"):
                    layer.impl._k_scale_cache = None
                if hasattr(layer.impl, "_v_scale_cache"):
                    layer.impl._v_scale_cache = None

        # ------【显存 profiling】触发 GC 与缓存清空，回收 profiling 期间占用的显存 ------
        gc.collect()
        torch.accelerator.empty_cache()

        logger.debug("Cleaned up profiling KV cache and CUDA graphs")

    @torch.inference_mode()
    def _create_encoder_cudagraph_manager(self) -> "EncoderCudaGraphManager | None":
        # ------【CUDA Graph】仅当开启多模态编码器图捕获且支持多模态输入时才创建 ------
        if not (
            self.compilation_config.cudagraph_mm_encoder and self.supports_mm_inputs
        ):
            return None

        # Use get_model() to unwrap CUDAGraphWrapper/UBatchWrapper, because
        # @runtime_checkable Protocol isinstance() checks do not work through
        # __getattr__ forwarding.
        from vllm.model_executor.models.interfaces import (
            SupportsEncoderCudaGraph,
            supports_encoder_cudagraph,
        )
        from vllm.v1.worker.encoder_cudagraph import (
            EncoderCudaGraphManager,
        )

        # ------【CUDA Graph】用 get_model 解包 wrapper，再判断是否支持编码器图捕获 ------
        raw_model = self.get_model()
        if not supports_encoder_cudagraph(raw_model):
            return None

        return EncoderCudaGraphManager(
            vllm_config=self.vllm_config,
            device=self.device,
            dtype=self.dtype,
            model=cast(SupportsEncoderCudaGraph, raw_model),
        )

    @torch.inference_mode()
    def _maybe_init_encoder_cudagraph_manager(self) -> None:
        # ------【CUDA Graph】懒初始化编码器图管理器，成功后打日志 ------
        if self.encoder_cudagraph_manager is None:
            self.encoder_cudagraph_manager = self._create_encoder_cudagraph_manager()
            if self.encoder_cudagraph_manager is not None:
                logger.info("Initialized EncoderCudaGraphManager for vision encoder")

    @torch.inference_mode()
    def profile_cudagraph_memory(self) -> int:
        # ------【显存 profiling】用最小 KV cache 做基准，测量 CUDA 图本身占用的显存 ------
        with set_current_vllm_config(self.vllm_config):
            self._init_minimal_kv_cache_for_profiling()

        # ------【显存 profiling】记录捕获计数，供 finally 阶段还原 ------
        saved_num_cudagraph_captured = compilation_counter.num_cudagraph_captured

        capture_descs = self.cudagraph_dispatcher.get_capture_descs()
        # Use a temporary manager for memory profiling. The persistent manager
        # is initialized later so it does not keep profiling-only graph state.
        encoder_cudagraph_manager = self._create_encoder_cudagraph_manager()

        # ------【CUDA Graph】统计需捕获的解码器/编码器图数量，为 0 则跳过 ------
        decoder_graphs = sum(len(descs) for _, descs in capture_descs)
        encoder_graphs = (
            encoder_cudagraph_manager.get_num_graphs_to_capture()
            if encoder_cudagraph_manager is not None
            else 0
        )
        total_graphs = decoder_graphs + encoder_graphs
        if total_graphs == 0:
            logger.debug("No CUDA graphs will be captured, skipping profiling")
            self._cleanup_profiling_kv_cache()
            return 0

        graph_groups = [
            *(
                f"{mode.name}={len(descs)} (largest={descs[0].num_tokens})"
                for mode, descs in capture_descs
                if descs
            ),
        ]
        if encoder_graphs > 0:
            graph_groups.append(
                f"ENCODER={encoder_graphs} "
                f"(largest={encoder_cudagraph_manager.token_budgets[-1]})"
            )

        logger.info("Profiling CUDA graph memory: %s", ", ".join(graph_groups))

        # ------【显存 profiling】用临时图内存池做测量，避免污染主池引发碎片化 ------
        # Use a temporary pool for profiling to avoid fragmentation in the main pool.
        profiling_pool = current_platform.graph_pool_handle()
        encoder_profiling_pool = current_platform.graph_pool_handle()
        original_pools: dict[int, Any] = {}
        all_wrappers = list(CUDAGraphWrapper._all_instances) + list(
            BreakableCUDAGraphWrapper._all_instances
        )
        for instance in all_wrappers:
            original_pools[id(instance)] = instance.graph_pool
            instance.graph_pool = profiling_pool

        shared_memory_estimate = {}
        per_graph_estimate = {}
        encoder_memory_estimate = 0

        # On ROCm, capture these throwaway profiling graphs on vLLM's dedicated
        # compute stream instead of the fresh side stream graph_capture()
        # allocates by default. torch's allocator pools free blocks per stream,
        # so a side-stream forward strands a persistent aiter scratch buffer in
        # a separate pool, shifting the physical placement of the real KV cache
        # allocated afterward and slowing bandwidth-bound decode ~20%. The
        # graphs are discarded, so a side stream is unnecessary here.
        # Use current_stream(), not torch.cuda.current_stream(): before vLLM
        # initializes its dedicated stream, torch returns the per-thread default
        # stream (cuda_stream=0), which cannot be used for cudagraph capture.
        # cap_ctx=None keeps the side-stream path on CUDA.
        cap_ctx = (
            GraphCaptureContext(current_stream())
            if current_platform.is_rocm()
            else None
        )

        # Cleanup-only guard: CUDA graph capture errors should still propagate
        # because encoder graph capture is opt-in.
        # ------【CUDA Graph】开启捕获开关，在冻结 GC + 图捕获上下文中逐模式采样显存增量 ------
        try:
            set_cudagraph_capturing_enabled(True)
            with (
                self._freeze_gc(),
                graph_capture(device=self.device, graph_capture_context=cap_ctx),
            ):
                torch.accelerator.synchronize()
                torch.accelerator.empty_cache()

                # ------【显存 profiling】每个模式抓前 2 个图，分别测首图与增量图显存 ------
                for mode, descs in capture_descs:
                    profile_descs = descs[:2]
                    mem_samples: list[int] = []

                    for i, desc in enumerate(profile_descs):
                        mem_before = torch.accelerator.get_memory_info()[0]
                        self._warmup_and_capture(
                            desc,
                            cudagraph_runtime_mode=mode,
                            profile_seq_lens=(
                                min(
                                    self.max_model_len,
                                    self.max_num_tokens // desc.num_tokens,
                                )
                                if mode == CUDAGraphMode.FULL and i == 0
                                else None
                            ),
                        )
                        torch.accelerator.synchronize()
                        free_after = torch.accelerator.get_memory_info()[0]
                        mem_samples.append(mem_before - free_after)

                    first_capture = mem_samples[0]
                    # Use at least 1 MiB per graph for driver overhead
                    per_graph = max(
                        mem_samples[1] if len(mem_samples) > 1 else 0, 1 << 20
                    )

                    shared_memory_estimate[mode] = first_capture
                    per_graph_estimate[mode] = per_graph * (len(descs) - 1)

                    logger.debug(
                        "Estimated %s CUDA graph memory: "
                        "%.2f MiB first-capture + (%d-1) × %.2f MiB per-graph",
                        mode.name,
                        first_capture / (1 << 20),
                        len(descs),
                        per_graph / (1 << 20),
                    )

                # ------【显存 profiling】单独测量编码器图占用的显存 ------
                if encoder_cudagraph_manager is not None:
                    mem_before = torch.accelerator.get_memory_info()[0]
                    encoder_cudagraph_manager.capture(graph_pool=encoder_profiling_pool)
                    torch.accelerator.synchronize()
                    free_after = torch.accelerator.get_memory_info()[0]
                    encoder_memory_estimate = max(mem_before - free_after, 0)

                    logger.debug(
                        "Estimated encoder CUDA graph memory: %.2f MiB for %d graphs",
                        encoder_memory_estimate / (1 << 20),
                        encoder_graphs,
                    )
        # ------【显存 profiling】无论成败清理一次性图并还原捕获状态/图内存池 ------
        finally:
            set_cudagraph_capturing_enabled(False)
            CUDAGraphWrapper.clear_all_graphs()
            BreakableCUDAGraphWrapper.clear_all_graphs()
            if encoder_cudagraph_manager is not None:
                encoder_cudagraph_manager.clear()
            all_wrappers = list(CUDAGraphWrapper._all_instances) + list(
                BreakableCUDAGraphWrapper._all_instances
            )
            for instance in all_wrappers:
                if id(instance) in original_pools:
                    instance.graph_pool = original_pools[id(instance)]
            for key_set in self.cudagraph_dispatcher.cudagraph_keys.values():
                key_set.clear()
            self.cudagraph_dispatcher.keys_initialized = False
            self.maybe_remove_all_loras(self.lora_config)
            self._cleanup_profiling_kv_cache()
            compilation_counter.num_cudagraph_captured = saved_num_cudagraph_captured

        # FULL and PIECEWISE graphs share the global pool at runtime and are
        # never replayed concurrently, so the pool overlays their memory.
        # Take the max to avoid double-counting the overlap.
        # ------【显存 profiling】FULL/PIECEWISE 共享池取最大，避免重复计入重叠部分 ------
        decoder_estimate = max(shared_memory_estimate.values(), default=0) + sum(
            per_graph_estimate.values()
        )
        # Encoder graphs use a manager-local pool at runtime, separate from the
        # decoder pool, so add their estimate instead of overlaying it.
        total_estimate = decoder_estimate + encoder_memory_estimate
        logger.info(
            "Estimated CUDA graph memory: %.2f GiB total",
            total_estimate / (1 << 30),
        )

        return int(total_estimate)







    # 捕获模型图
    @instrument(span_name="Capture model")
    def capture_model(self) -> int:
        # 若未启用cuda graph, 直接退出
        if self.compilation_config.cudagraph_mode == CUDAGraphMode.NONE:
            logger.warning(
                "Skipping CUDA graph capture. To turn on CUDA graph capture, "
                "ensure `cudagraph_mode` was not manually set to `NONE`"
            )
            return 0

        # ------【CUDA Graph】若启用则先初始化编码器图管理器 ------
        # Initialize encoder CUDA graph manager if enabled.
        self._maybe_init_encoder_cudagraph_manager()

        compilation_counter.num_gpu_runner_capture_triggers += 1

        start_time = time.perf_counter()

        # Trigger CUDA graph capture for specific shapes.
        # Capture the large shapes first so that the smaller shapes
        # can reuse the memory pool allocated for the large shapes.
        # ------【CUDA Graph】开启全局捕获开关，大形状优先捕获以便小形状复用其内存池 ------
        set_cudagraph_capturing_enabled(True)

        # Setup torch profiler for graph capture traces (conditional)
        from vllm.distributed.parallel_state import get_world_group

        local_rank = get_world_group().local_rank
        enable_profiler = (
            local_rank == 0
        ) and self.vllm_config.profiler_config.capture_torch_profiler
        if enable_profiler:
            trace_dir = (
                self.vllm_config.profiler_config.torch_profiler_dir + "/capture_traces"
            )
            profiler = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    trace_dir,
                    worker_name=f"graph_capture_rank_{local_rank}",
                    use_gzip=True,
                ),
            )
            logger.info_once(
                "Rank %d: Torch profiler enabled for CUDA graph capture, "
                "traces will be saved to: %s",
                local_rank,
                trace_dir,
            )
        else:
            profiler = nullcontext()
            logger.info_once(
                "Rank %d: Torch profiler disabled for CUDA graph capture", local_rank
            )

        # ------【显存 profiling】记录捕获前空闲显存，冻结 GC 后进入图捕获上下文 ------
        ########################################################
        # 开始主动捕获
        ########################################################
        with self._freeze_gc(), graph_capture(device=self.device):
            torch.accelerator.synchronize()
            torch.accelerator.empty_cache()
            start_free_gpu_memory = torch.accelerator.get_memory_info()[0]

            # ------【CUDA Graph】按运行时模式遍历所有批次描述符逐个捕获图 ------
            for (
                runtime_mode, # mode
                batch_descs, # key
            ) in self.cudagraph_dispatcher.get_capture_descs(): # 对调度器库里面的每一个(mode, key), 都进行主动捕获
                self._capture_cudagraphs(
                    batch_descriptors=batch_descs,
                    cudagraph_runtime_mode=runtime_mode,
                    profiler=profiler,
                )
                torch.accelerator.synchronize()

            # ------【CUDA Graph】捕获编码器图（若有） ------
            # Capture encoder CUDA graphs if enabled
            if self.encoder_cudagraph_manager is not None:
                encoder_graph_pool = current_platform.graph_pool_handle()
                self.encoder_cudagraph_manager.capture(graph_pool=encoder_graph_pool)

            torch.accelerator.synchronize()
            end_free_gpu_memory = torch.accelerator.get_memory_info()[0]

        # ------【CUDA Graph】关闭捕获开关，此后任何意外图捕获都会报错 ------
        # Disable cudagraph capturing globally, so any unexpected cudagraph
        # capturing will be detected and raise an error after here.
        # Note: We don't put it into graph_capture context manager because
        # we may do lazy capturing in future that still allows capturing
        # after here.
        set_cudagraph_capturing_enabled(False)

        torch.accelerator.synchronize()
        torch.accelerator.empty_cache()

        # ------【显存 profiling】锁定工作区，防止执行期被动态调整尺寸 ------
        # Lock workspace to prevent resizing during execution.
        # Max workspace sizes should have been captured during warmup/profiling.
        lock_workspace()

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        cuda_graph_size = start_free_gpu_memory - end_free_gpu_memory
        # This usually takes 5~20 seconds.
        logger.info_once(
            "Graph capturing finished in %.0f secs, took %.2f GiB",
            elapsed_time,
            cuda_graph_size / (1 << 30),
        )
        return cuda_graph_size


    # 对一个key， model_runner进行预热和捕获
    def _warmup_and_capture(
        self,
        desc: BatchDescriptor,
        cudagraph_runtime_mode: CUDAGraphMode,
        profile_seq_lens: int | None = None,
        allow_microbatching: bool = False,
        num_warmups: int | None = None, # warmup次数
        profiler: AbstractContextManager[Any] | None = None,
    ):
        if profiler is None:
            profiler = nullcontext()
        if num_warmups is None:
            num_warmups = self.compilation_config.cudagraph_num_of_warmups # 获取配置的warmup次数
        # ------【CUDA Graph】FULL 图需强制构造 attention 元数据 ------
        force_attention = cudagraph_runtime_mode == CUDAGraphMode.FULL



        # ------【CUDA Graph】先跑若干次 warmup 让 kernel/显存状态稳定后再捕获 ------
        # 开始warmup
        for _ in range(num_warmups):
            # 直接假跑
            self._dummy_run(
                desc.num_tokens, # 这个档位的key的batch的tokens总数
                cudagraph_runtime_mode=CUDAGraphMode.NONE, # 不开cudagraph
                force_attention=force_attention, # True 表示mode是FULL，纯decode
                uniform_decode=desc.uniform, # 是否均匀
                allow_microbatching=allow_microbatching,
                skip_eplb=True,
                remove_lora=False,
                num_active_loras=desc.num_active_loras,
                profile_seq_lens=profile_seq_lens,
            )

        # ------【CUDA Graph】warmup 可能用辅助流，同步确保其完成后再开始捕获 ------
        if num_warmups > 0:
            # Warmups may use auxiliary streams. Ensure all of their work has
            # completed before beginning CUDA graph capture.
            torch.accelerator.synchronize()



        # ------【CUDA Graph】在 profiler 与 record_function 内做真正的图捕获 dummy 前向 ------
        # 开始真正的捕获
        with (
            profiler,
            torch.profiler.record_function(
                f"capture_{desc.num_tokens}_{cudagraph_runtime_mode.name}"
            ),
        ):
            # 假跑
            self._dummy_run(
                desc.num_tokens, # key的tokens数
                cudagraph_runtime_mode=cudagraph_runtime_mode, # mode
                uniform_decode=desc.uniform, # 是否均匀
                allow_microbatching=allow_microbatching,
                skip_eplb=True,
                remove_lora=False,
                num_active_loras=desc.num_active_loras,
                is_graph_capturing=True, # 捕获使能
                profile_seq_lens=profile_seq_lens,
            )

    def _capture_cudagraphs(
        self,
        batch_descriptors: list[BatchDescriptor], # 里面的多个key档位
        cudagraph_runtime_mode: CUDAGraphMode, # 一个mode
        profiler: AbstractContextManager[Any] | None = None,
    ):
        assert (
            cudagraph_runtime_mode != CUDAGraphMode.NONE
            and cudagraph_runtime_mode.is_valid_runtime_mode()
        ), f"Invalid cudagraph runtime mode: {cudagraph_runtime_mode}"

        if not batch_descriptors:
            return

        # ------【CUDA Graph】同一组描述符共享 uniform/decode 标志 ------
        uniform_decode = batch_descriptors[0].uniform

        # Only rank 0 should print progress bar during capture
        if is_global_first_rank():
            batch_descriptors = tqdm(
                batch_descriptors,
                disable=not self.load_config.use_tqdm_on_load,
                desc="Capturing CUDA graphs ({}, {})".format(
                    "decode" if uniform_decode else "mixed prefill-decode",
                    cudagraph_runtime_mode.name,
                ),
            )

        # ------【CUDA Graph】逐个描述符 warmup+捕获；跳过 EPLB 避免记录 dummy 指标 ------
        # We skip EPLB here since we don't want to record dummy metrics
        # 开始针对各个key进行捕获
        for batch_desc in batch_descriptors:
            # We currently only capture ubatched graphs when its a FULL
            # cudagraph, a uniform decode batch, and the number of tokens
            # is above the threshold. Otherwise we just capture a non-ubatched
            # version of the graph
            # ------【CUDA Graph】仅 FULL+统一 decode+超阈值时才启用 ubatch 图捕获 ------
            allow_microbatching = (
                self.parallel_config.use_ubatching
                and cudagraph_runtime_mode == CUDAGraphMode.FULL
                and uniform_decode
                and check_ubatch_thresholds(
                    config=self.vllm_config.parallel_config,
                    num_tokens=batch_desc.num_tokens,
                    uniform_decode=uniform_decode,
                )
            )

            # 开始针对这个key, 进行预热和捕获
            self._warmup_and_capture(
                batch_desc,
                cudagraph_runtime_mode=cudagraph_runtime_mode,
                allow_microbatching=allow_microbatching,
                profiler=profiler,
            )
            torch.accelerator.synchronize()
        # ------【LoRA】捕获结束后清理所有 dummy LoRA ------
        self.maybe_remove_all_loras(self.lora_config)






    # kvcache tensor显存分配前，初始化注意力后端
    def initialize_attn_backend(
        self,
        kv_cache_config: KVCacheConfig, # 配置信息
        is_profiling: bool = False,
    ) -> None:
        """
        Initialize the attention backends and attention metadata builders.
        """
        assert len(self.attn_groups) == 0, "Attention backends are already initialized"

        class AttentionGroupKey(NamedTuple):
            """
            一个命名元组，用作 KV cache 组内部对 attention 组做去重的键

            Deduplication key for attention groups within a KV cache group.
            同一个 KV cache 组里，可能存在多个层共享同一种后端/spec，这个键就是用来把"本质相同"的层合并成一个 attention 组的去重依据。


            Splits on per-rank ``num_heads_q`` in addition to backend + spec
            去重的依据除了 backend（后端类）和 spec（KV cache 规格）之外，还额外按每 rank 的 num_heads_q（Q 头数）来拆分。


            so layers with different Q-head counts (e.g. a spec-decode draft
            with fewer attention heads than its target) get separate metadata
            builders. 
            这样做的目的是：Q 头数不同的层（例如投机解码里的草稿模型，头数比目标模型少）会被分到不同的 metadata builder。
            
            
            The builders' scratch (e.g. ``softmax_segm_*`` in
            ``triton_attn``, ``num_qo_heads`` in FlashInfer) is sized by
            ``num_heads_q`` and assumes uniformity within the group; see
            ``get_num_attention_heads_from_layers`` in
            ``vllm/v1/attention/backends/utils.py``.

            因为 metadata builder 内部的 scratch 缓冲区（比如 triton_attn 里的 softmax_segm_* 分段 softmax 缓冲区、FlashInfer 里的 num_qo_heads）
            都是按 num_heads_q 来定尺寸的，并且默认一个组内所有层的 Q 头数是一致的。

            ------------------------------------------------------------------------------------------------------------------

            是 Impl 实例。每层都要单独建一个 self.impl（attention.py:409-423），
            因为 Impl 持有权重级参数（scale、sliding_window、head_size、layer 各自的 QKV 权重）。

            而 metadata builder 不需要每层一个。因为 attention 元数据（seq_lens、block_table、slot_mapping 这些 batch 级描述信息）
            对所有层是一样的——第 0 层和第 27 层算 attention 时，输入序列结构完全相同，只是 KV cache 内容不同、权重不同。

            所以 28 层里"后端类 + spec + heads"都相同的，就合并成一个组，只建 1 个 metadata builder，而不是建 28 个。
            """

            ### 这里是对我们model里面这么多的attention后端进行分组：
            '''
            对注意力后端进行分组的好处是，每组构建一个builder，来构建一个组的atten后端通用的metadata

                这就是分组的核心收益。看 line 3501-3502：


                for layer_name in attn_group.layer_names:
                    attn_metadata_dict[layer_name] = attn_metadata_i
                一次 build() 的结果，直接赋给同组的 28 个层名。
            '''
            attn_backend: type[AttentionBackend]
            kv_cache_spec: KVCacheSpec
            num_heads_q: int

        def get_attn_backends_for_group(
            kv_cache_group_spec: KVCacheGroupSpec,
        ) -> tuple[dict[AttentionGroupKey, list[str]], set[type[AttentionBackend]]]:
            layer_type = cast(type[Any], AttentionLayerBase)
            layers = get_layers_from_vllm_config(
                self.vllm_config, layer_type, kv_cache_group_spec.layer_names
            )
            attn_backends = {}
            attn_backend_layers = defaultdict(list)
            # Dedupe based on full class name; this is a bit safer than
            # using the class itself as the key because when we create dynamic
            # attention backend subclasses (e.g. ChunkedLocalAttention) unless
            # they are cached correctly, there will be different objects per
            # layer.
            for layer_name in kv_cache_group_spec.layer_names:
                attn_backend = layers[layer_name].get_attn_backend()

                if layer_name in self.kv_sharing_fast_prefill_eligible_layers:
                    attn_backend = create_fast_prefill_custom_backend(
                        "FastPrefill",
                        attn_backend,  # type: ignore[arg-type]
                    )

                full_cls_name = attn_backend.full_cls_name()
                layer_kv_cache_spec = kv_cache_group_spec.kv_cache_spec
                if isinstance(layer_kv_cache_spec, UniformTypeKVCacheSpecs):
                    layer_kv_cache_spec = layer_kv_cache_spec.kv_cache_specs[layer_name]
                # Non-Attention layer types (e.g. Mamba1, ShortConv) do not
                # expose ``num_heads``; fall back to 0 so they cluster as
                # before. Such layers never coexist with Attention in a
                # single KV cache group (different KVCacheSpec), so the
                # fallback can never spuriously merge them with attention
                # layers.
                num_heads_q = getattr(layers[layer_name], "num_heads", 0)
                key = (full_cls_name, layer_kv_cache_spec, num_heads_q)
                attn_backends[key] = AttentionGroupKey(
                    attn_backend, layer_kv_cache_spec, num_heads_q
                )
                attn_backend_layers[key].append(layer_name)
            return (
                {attn_backends[k]: v for k, v in attn_backend_layers.items()},
                set(group_key.attn_backend for group_key in attn_backends.values()),
            )

        def create_attn_groups(
            attn_backends_map: dict[AttentionGroupKey, list[str]],
            kv_cache_group_id: int,
        ) -> list[AttentionGroup]:
            attn_groups: list[AttentionGroup] = []
            for key, layer_names in attn_backends_map.items():
                attn_group = AttentionGroup(
                    key.attn_backend,
                    layer_names,
                    key.kv_cache_spec,
                    kv_cache_group_id,
                )

                attn_groups.append(attn_group)
            return attn_groups

        # ------【核心逻辑】遍历每个 KV cache 组，解析其注意力后端并按去重键分组 ------
        attention_backend_maps = []
        attention_backend_list = []
        for kv_cache_group_spec in kv_cache_config.kv_cache_groups:
            attn_backends = get_attn_backends_for_group(kv_cache_group_spec)
            attention_backend_maps.append(attn_backends[0])
            attention_backend_list.append(attn_backends[1])

        # ------【CUDA Graph】在初始化 metadata builder 前解析 cudagraph 模式 ------
        # Resolve cudagraph_mode before actually initialize metadata_builders
        ######################################################
        # 解析每个group的atten的约束，然后把满足所有atten backend 约束的（mode, keys）加入dispatcher
        ######################################################
        self._check_and_update_cudagraph_mode(
            attention_backend_list,
            kv_cache_config.kv_cache_groups,
            is_profiling=is_profiling,
        )

        # ------【TP/PP】校验注意力后端对 PCP&DCP 等上下文并行的兼容性 ------
        # Check if attention backend supports PCP&DCP and related features.
        check_attention_cp_compatibility(self.vllm_config)

        # ------【核心逻辑】为每个后端组创建 AttentionGroup 加入 attn_groups ------
        for i, attn_backend_map in enumerate(attention_backend_maps):
            self.attn_groups.append(create_attn_groups(attn_backend_map, i))




    def initialize_metadata_builders(
        self, kv_cache_config: KVCacheConfig, kernel_block_sizes: list[int]
    ) -> None:
        """
        Create the metadata builders for all KV cache groups and attn groups.
        """
        # ------【核心逻辑】为每个 KV cache 组的每个 attn 组创建元数据构建器 ------
        for kv_cache_group_id in range(len(kv_cache_config.kv_cache_groups)):
            for attn_group in self.attn_groups[kv_cache_group_id]:
                attn_group.create_metadata_builders(
                    self.vllm_config,
                    self.device,
                    kernel_block_sizes[kv_cache_group_id]
                    if kv_cache_group_id < len(kernel_block_sizes)
                    else None,
                    num_metadata_builders=1
                    if not self.parallel_config.use_ubatching
                    else self.parallel_config.num_ubatches,
                )
        # Calculate reorder batch threshold (if needed)
        # Note (tdoublep): do this *after* constructing builders,
        # because some of them change the threshold at init time.
        # ------【核心逻辑】构建器建好后计算重排阈值（部分后端会改动阈值） ------
        self.calculate_reorder_batch_threshold()

        # ------【投机解码】为草稿模型初始化注意力后端 ------
        # Initialize drafter attention backend
        if self.speculative_config and (
            self.speculative_config.use_eagle()
            or self.speculative_config.uses_draft_model()
        ):
            assert isinstance(
                self.drafter,
                EagleProposer | DFlashProposer | DraftModelProposer | Gemma4Proposer,
            )
            self.drafter.initialize_attn_backend(kv_cache_config, kernel_block_sizes)




    def _check_and_update_cudagraph_mode(
        self,
        attention_backends: list[set[type[AttentionBackend]]], # 每种类型的atten后端的集合列表
        kv_cache_groups: list[KVCacheGroupSpec],
        is_profiling: bool = False,
    ) -> None:
        """
        Resolve the cudagraph_mode when there are multiple attention
        groups with potential conflicting CUDA graph support.

        Then initialize the cudagraph_dispatcher based on the resolved
        cudagraph_mode.
        """
        # ------【CUDA Graph】取所有注意力后端中最弱的图支持等级，作为整体约束 ------
        min_cg_support = AttentionCGSupport.ALWAYS
        min_cg_attn_backend = None

        for attn_backend_set, kv_cache_group in zip(
            attention_backends, kv_cache_groups
        ):
            for attn_backend in attn_backend_set:
                builder_cls = attn_backend.get_builder_cls()

                cg_support = builder_cls.get_cudagraph_support(
                    self.vllm_config, kv_cache_group.kv_cache_spec
                )
                if cg_support.value < min_cg_support.value:
                    min_cg_support = cg_support
                    min_cg_attn_backend = attn_backend.__name__
        # ------【CUDA Graph】综合最弱支持等级与并行规模解析最终 cudagraph 模式与捕获尺寸 ------

        '''
        约束 = 每个 attention backend 声明的 cudagraph 支持等级 AttentionCGSupport，4 档（backend.py:606-621）：

            等级	                            值	                                含义
            ALWAYS	                            3	                    支持混合 prefill/decode 的 FULL 图
            UNIFORM_BATCH	                    2	                    只支持 query 长度一致的 batch（如 spec-decode）
            UNIFORM_SINGLE_TOKEN_DECODE	        1	                    只支持 query_len==1 的纯 decode
            NEVER	                            0	                    不支持 cudagraph


        每个 backend 的 builder 类通过 _cudagraph_support 类变量声明自己的等级，get_cudagraph_support() 读它（backend.py:650-657）。比如 FlashAttention/FlashInfer 是 ALWAYS，某些新后端可能只有 UNIFORM_BATCH。

        为什么叫"每个组的约束"：一个模型可能有多个 group、多个 backend（如 FlashInfer + TritonAttn 混用）。cudagraph 是整体录制/回放的，所以要用所有组里最弱的那个作为全局上限。这就是 gpu_model_runner.py:9131-9145 那段：
        '''

        '''
            如何得到 (mode, key) 库
                分两步：先 resolve 出 mode，再按 mode 填 key 库。
        '''
        cudagraph_mode = self.compilation_config.resolve_cudagraph_mode_and_sizes(
            min_cg_support,
            min_cg_attn_backend,
            self.uniform_decode_query_len,
            use_v2_model_runner=False,
            tensor_parallel_size=self.parallel_config.tensor_parallel_size,
            kv_cache_config=self.kv_cache_config,
            max_num_reqs=self.max_num_reqs,
            is_profiling=is_profiling,
        )
        # Trigger cudagraph dispatching keys initialization after
        # resolved cudagraph mode.
        # ------【CUDA Graph】模式确定后初始化 dispatcher 的捕获键 ------
        
        ######################################################################################
        # 开始初始化调度器里面的(mode, key)库
        ######################################################################################
        self.cudagraph_dispatcher.initialize_cudagraph_keys(
            cudagraph_mode, self.uniform_decode_query_len
        )

        # ------【投机解码】为草稿模型同步初始化其 cudagraph 键 ------
        # Initialize drafter's cudagraph dispatcher if using spec decode.
        if self.speculative_config and (
            self.speculative_config.use_eagle()
            or self.speculative_config.uses_draft_model()
            or self.speculative_config.uses_extract_hidden_states()
        ):
            assert isinstance(
                self.drafter,
                EagleProposer
                | DFlashProposer
                | DraftModelProposer
                | ExtractHiddenStatesProposer
                | Gemma4Proposer,
            )
            self.drafter.initialize_cudagraph_keys(cudagraph_mode)

    def calculate_reorder_batch_threshold(self) -> None:
        """
        Choose the minimum reorder batch threshold from all attention groups.
        Backends should be able to support lower threshold then what they request
        just may have a performance penalty due to that backend treating decodes
        as prefills.
        """
        # ------【核心逻辑】取所有注意力组中最小的重排阈值，保证各后端都能正确重排 ------
        min_none_high = lambda a, b: a if b is None else b if a is None else min(a, b)

        reorder_batch_thresholds: list[int | None] = [
            group.get_metadata_builder().reorder_batch_threshold
            for group in self._attn_group_iterator()
        ]
        # If there are no attention groups (attention-free model) or no backend
        # reports a threshold, leave reordering disabled.
        # ------【核心逻辑】无注意力组或未报告阈值时禁用重排 ------
        if len(reorder_batch_thresholds) == 0:
            self.reorder_batch_threshold = None
            return
        self.reorder_batch_threshold = reduce(min_none_high, reorder_batch_thresholds)  # type: ignore[assignment]

    def may_reinitialize_input_batch(
        self, kv_cache_config: KVCacheConfig, kernel_block_sizes: list[int]
    ) -> None:
        """
        Re-initialize the input batch if the block sizes are different from
        what it was originally created with. This happens when the final
        block size (determined after model loading) differs from the
        placeholder used during __init__, or when there are multiple
        KV cache groups.

        Args:
            kv_cache_config: The KV cache configuration.
            kernel_block_sizes: The kernel block sizes for each KV cache group.
        """
        # ------【核心逻辑】从最终 KV cache 配置收集 block 大小等参数，判断是否需要重建输入批次 ------
        block_sizes = []
        max_num_blocks = []
        slot_mapping_modes = []
        max_model_len = max(self.max_model_len, self.max_encoder_len)
        for kv_cache_group in kv_cache_config.kv_cache_groups:
            kv_cache_spec = kv_cache_group.kv_cache_spec
            kv_cache_spec_kind = get_kv_cache_spec_kind(kv_cache_spec)
            if kv_cache_spec_kind == KVCacheSpecKind.ENCODER_ONLY_ATTENTION:
                continue
            block_size = kv_cache_spec.block_size
            block_sizes.append(block_size)
            if kv_cache_spec_kind == KVCacheSpecKind.MAMBA:
                slot_mapping_modes.append(SlotMappingMode.NONE)
            else:
                slot_mapping_modes.append(SlotMappingMode.TOKEN_TO_KV_SLOT)
            max_num_blocks_per_req = kv_cache_spec.max_num_blocks_per_req(
                self.vllm_config, max_model_len
            )
            max_num_blocks.append(max_num_blocks_per_req)

        # ------【内存池/CuMem】参数与占位值不一致时用最终 block 大小重建 InputBatch ------
        if (
            block_sizes != self._init_block_sizes
            or kernel_block_sizes != self._init_kernel_block_sizes
            or max_num_blocks != self._init_max_num_blocks
            or slot_mapping_modes != self._init_slot_mapping_modes
        ):
            self._init_block_sizes = block_sizes
            self._init_kernel_block_sizes = kernel_block_sizes
            self._init_max_num_blocks = max_num_blocks
            self._init_slot_mapping_modes = slot_mapping_modes
            self.input_batch = InputBatch(
                max_num_reqs=self.max_num_reqs,
                max_model_len=max_model_len,
                max_num_batched_tokens=self.max_num_tokens,
                device=self.device,
                vocab_size=self.model_config.get_vocab_size(),
                block_sizes=block_sizes,
                kernel_block_sizes=kernel_block_sizes,
                max_num_blocks_per_req=max_num_blocks,
                num_spec_tokens=self.num_spec_tokens,
                logitsprocs=self.input_batch.logitsprocs,
                logitsprocs_need_output_token_ids=self.input_batch.logitsprocs_need_output_token_ids,
                is_pooling_model=self.is_pooling_model,
                cp_kv_cache_interleave_size=self.parallel_config.cp_kv_cache_interleave_size,
                reasoning_config=self.vllm_config.reasoning_config,
                use_replayssm=self.cache_config.use_replayssm,
                slot_mapping_modes=slot_mapping_modes,
            )

        assert self._init_block_sizes == block_sizes, (
            f"InputBatch block_sizes {self._init_block_sizes} != "
            f"kv_cache block_sizes {block_sizes}"
        )
        assert self._init_kernel_block_sizes == kernel_block_sizes, (
            f"InputBatch kernel_block_sizes {self._init_kernel_block_sizes} "
            f"!= kv_cache kernel_block_sizes {kernel_block_sizes}"
        )

    ############################################################################################
    # 创建各个layer的kv cache tensor
    ############################################################################################
    def _allocate_kv_cache_tensors(
        self, kv_cache_config: KVCacheConfig
    ) -> dict[str, torch.Tensor]:
        """
        Initializes the KV cache buffer with the correct size. The buffer needs
        to be reshaped to the desired shape before being used by the models.

        Args:
            kv_cache_config: The KV cache config
        Returns:
            dict[str, torch.Tensor]: A map between layer names to their
            corresponding memory buffer for KV cache.
        """
        # ------【内存池/CuMem】按配置分配 KV cache 底层字节缓冲，packed 张量共享同一 backing ------
        kv_cache_raw_tensors: dict[str, torch.Tensor] = {}
        packed_backing: torch.Tensor | None = None


        ############################################################################################
        # 配置里面的每个张量
        ############################################################################################
        for kv_cache_tensor in kv_cache_config.kv_cache_tensors:
            if kv_cache_tensor.block_stride > 0:
                # Allocate once; all packed tensors alias the same backing.
                if packed_backing is None:
                    packed_backing = torch.zeros(
                        kv_cache_tensor.size,
                        dtype=torch.int8,
                        device=self.device,
                    )
                tensor = packed_backing
            else:
                ############################################################################################
                # 创建GPU上的张量
                ############################################################################################
                tensor = torch.zeros(
                    kv_cache_tensor.size, dtype=torch.int8, device=self.device
                )

                ############################################################################################
                # 标记共享tensor形状的layer有哪些
                ############################################################################################
            for layer_name in kv_cache_tensor.shared_by:
                kv_cache_raw_tensors[layer_name] = tensor

        # ------【核心逻辑】校验每个需要的层都分配到了 KV cache 张量 ------
        layer_names = set()
        for group in kv_cache_config.kv_cache_groups:
            for layer_name in group.layer_names:
                if layer_name in self.runner_only_attn_layers:
                    continue
                layer_names.add(layer_name)
        assert layer_names == set(kv_cache_raw_tensors.keys()), (
            "Some layers are not correctly initialized"
        )
        return kv_cache_raw_tensors










    def _attn_group_iterator(self) -> Iterator[AttentionGroup]:
        return itertools.chain.from_iterable(self.attn_groups)

    def _kv_cache_spec_attn_group_iterator(self) -> Iterator[AttentionGroup]:
        if not self.kv_cache_config.kv_cache_groups:
            return
        for attn_groups in self.attn_groups:
            yield from attn_groups


    ############################################################################################
    # 单layer的tensor，reshape出block维度
    ############################################################################################
    def _reshape_kv_cache_tensors(
        self,
        kv_cache_raw_tensors: dict[str, torch.Tensor],
        kernel_block_sizes: list[int],
    ) -> dict[str, torch.Tensor]:
        """
        Reshape the KV cache tensors to the desired shape and dtype.

        Args:
            kv_cache_raw_tensors: The KV cache buffer of each layer, with
                correct size but uninitialized shape.
            kernel_block_sizes: The kernel block sizes for each KV cache group.
        Returns:
            Dict[str, torch.Tensor]: A map between layer names to their
            corresponding memory buffer for KV cache.
        """
        kv_caches: dict[str, torch.Tensor] = {}
        has_attn, has_mamba = False, False

        # Map layer names to (offset, block_stride) within the packed
        # backing tensor so we can create strided views per layer.
        # ------【内存池/CuMem】记录 packed backing 内各层的偏移与块 stride，便于建 strided 视图 ------
        layer_packing: dict[str, tuple[int, int]] = {}
        for kv_tensor in self.kv_cache_config.kv_cache_tensors:
            if kv_tensor.block_stride > 0:
                for ln in kv_tensor.shared_by:
                    layer_packing[ln] = (kv_tensor.offset, kv_tensor.block_stride)
        # ------【核心逻辑】按层把原始字节缓冲 reshape 成注意力/Mamba 所需的 KV cache 形状 ------
        for group in self._kv_cache_spec_attn_group_iterator():
            kv_cache_spec = group.kv_cache_spec
            attn_backend = group.backend
            if group.kv_cache_group_id == len(kernel_block_sizes):
                # There may be a last group for layers without kv cache.
                continue
            kernel_block_size = kernel_block_sizes[group.kv_cache_group_id]
            for layer_name in group.layer_names:
                if layer_name in self.runner_only_attn_layers:
                    continue
                raw_tensor = kv_cache_raw_tensors[layer_name]
                packing = layer_packing.get(layer_name)
                if packing is not None:
                    _, blk_stride = packing
                    num_blocks = raw_tensor.numel() // blk_stride
                else:
                    assert raw_tensor.numel() % kv_cache_spec.page_size_bytes == 0
                    num_blocks = raw_tensor.numel() // kv_cache_spec.page_size_bytes
                if isinstance(kv_cache_spec, AttentionSpec):
                    has_attn = True
                    num_blocks_per_kv_block = (
                        kv_cache_spec.block_size // kernel_block_size
                    )
                    kernel_num_blocks = num_blocks * num_blocks_per_kv_block

                    # For MLA with compression, storage_block_size != block_size
                    if kv_cache_spec.storage_block_size != kv_cache_spec.block_size:
                        shape_block_size = kv_cache_spec.storage_block_size
                    else:
                        shape_block_size = kernel_block_size

                    # Skipped layers (--kv-cache-dtype-skip-layers) need
                    # the unquantized shape.
                    layer_cache_dtype_str = (
                        "auto"
                        if kv_cache_spec.kv_quant_mode == KVQuantMode.NONE
                        else getattr(
                            kv_cache_spec,
                            "cache_dtype_str",
                            None,
                        )
                        or self.cache_config.cache_dtype
                    )
                    kv_cache_shape = attn_backend.get_kv_cache_shape(
                        kernel_num_blocks,
                        shape_block_size,
                        kv_cache_spec.num_kv_heads,
                        kv_cache_spec.head_size,
                        cache_dtype_str=layer_cache_dtype_str,
                    )
                    try:
                        kv_cache_stride_order = attn_backend.get_kv_cache_stride_order()
                        assert len(kv_cache_stride_order) == len(kv_cache_shape)
                    except (AttributeError, NotImplementedError):
                        kv_cache_stride_order = tuple(range(len(kv_cache_shape)))
                    raw_tensor = kv_cache_raw_tensors[layer_name]
                    kv_caches[layer_name] = _reshape_attention_kv_cache(
                        raw_tensor,
                        kv_cache_spec,
                        kv_cache_shape,
                        kv_cache_stride_order,
                        kernel_num_blocks,
                        packing,
                    )

                elif isinstance(kv_cache_spec, MambaSpec):
                    has_mamba = True
                    raw_tensor = kv_cache_raw_tensors[layer_name]
                    page_size_bytes = kv_cache_spec.page_size_bytes
                    # Hold a single contiguous [num_blocks, 1, 1, page_size_bytes]
                    # int8 page view per layer; the layer's bind_kv_cache unpacks
                    # each block's bytes into its conv/ssm state views. Keeping
                    # one tensor per layer lets the KV connector register it
                    # without special-casing Mamba.
                    kv_caches[layer_name] = raw_tensor[
                        : num_blocks * page_size_bytes
                    ].view(num_blocks, 1, 1, page_size_bytes)
                else:
                    raise NotImplementedError

        # Reconcile divergent KV layouts to blocks-first. Triggered by hybrid
        # attention/mamba models, and by encoder-decoder models whose shared
        # decoder/cross-attention allocation mixes K/V-first and blocks-first
        # backends (see _has_mixed_attention_kv_layout).
        # ------【核心逻辑】混合 attention/mamba 布局时统一到 blocks-first 布局 ------
        if has_attn and (
            has_mamba or self._has_mixed_attention_kv_layout(kernel_block_sizes)
        ):
            self._update_hybrid_attention_mamba_layout(kv_caches, kernel_block_sizes)

        return kv_caches





















    def _has_mixed_attention_kv_layout(self, kernel_block_sizes: list[int]) -> bool:
        """Whether attention groups disagree on the physical KV cache layout.

        Encoder-decoder models (e.g. Whisper) share one raw KV allocation
        between a decoder self-attention layer (K/V-first ROCM_ATTN, block dim
        1) and a cross-attention layer (blocks-first, block dim 0). Mixed block
        dims mean a block ID maps to different bytes per layer, so the shared
        buffer must be normalized to a single (blocks-first) layout.
        """
        # ------【核心逻辑】收集各 attention 组的 block 维度，判断是否存在混合 KV 布局 ------
        block_dims: set[int] = set()
        for group in self._kv_cache_spec_attn_group_iterator():
            kv_cache_spec = group.kv_cache_spec
            if not isinstance(kv_cache_spec, AttentionSpec):
                continue
            if group.kv_cache_group_id == len(kernel_block_sizes):
                continue
            block_dims.add(
                group.backend.get_kv_cache_block_dim(
                    kernel_block_sizes[group.kv_cache_group_id],
                    kv_cache_spec.num_kv_heads,
                    kv_cache_spec.head_size,
                    cache_dtype_str=self.cache_config.cache_dtype,
                )
            )
        return len(block_dims) > 1

    def _update_hybrid_attention_mamba_layout(
        self, kv_caches: dict[str, torch.Tensor], kernel_block_sizes: list[int]
    ) -> None:
        """
        Update the layout of attention layers from (2, num_blocks, ...) to
        (num_blocks, 2, ...).

        Args:
            kv_caches: The KV cache buffer of each layer.
            kernel_block_sizes: The kernel block sizes for each KV cache group.
        """

        # ------【核心逻辑】把 K/V-first 的注意力层用 as_strided 重排为 blocks-first 布局 ------
        for group in self._kv_cache_spec_attn_group_iterator():
            kv_cache_spec = group.kv_cache_spec
            if not isinstance(kv_cache_spec, AttentionSpec):
                continue
            block_dim = group.backend.get_kv_cache_block_dim(
                kernel_block_sizes[group.kv_cache_group_id],
                kv_cache_spec.num_kv_heads,
                kv_cache_spec.head_size,
                cache_dtype_str=self.cache_config.cache_dtype,
            )
            # block_dim: 0 means (num_blocks, 2, ...); 1 means (2, num_blocks, ...).
            if block_dim == 0:
                continue
            assert block_dim == 1
            for layer_name in group.layer_names:
                kv_cache = kv_caches[layer_name]
                hidden_size = kv_cache.shape[2:].numel()
                kv_cache.as_strided_(
                    size=kv_cache.shape,
                    stride=(hidden_size, 2 * hidden_size, *kv_cache.stride()[2:]),
                )





    ############################################################################################
    # 实际划分每个group的各个layer的kvcache tensor， 也就是我们的初始化
    ############################################################################################
    def initialize_kv_cache_tensors(
        self, kv_cache_config: KVCacheConfig, kernel_block_sizes: list[int]
    ) -> dict[str, torch.Tensor]:
        """
        Initialize the memory buffer for KV cache.

        Args:
            kv_cache_config: The KV cache config
            kernel_block_sizes: The kernel block sizes for each KV cache group.

        Returns:
            Dict[str, torch.Tensor]: A map between layer names to their
            corresponding memory buffer for KV cache.
        """

        # ------【PD 分离】优先分配针对 kv-connector 传输优化的统一 KV cache ------
        # Try creating KV caches optimized for kv-connector transfers
        cache_dtype = self.cache_config.cache_dtype
        if self.use_uniform_kv_cache(self.attn_groups):
            kv_caches, cross_layers_kv_cache, attn_backend = (
                self.allocate_uniform_kv_caches(
                    kv_cache_config,
                    self.attn_groups,
                    cache_dtype,
                    self.device,
                    kernel_block_sizes,
                )
            )
            self.cross_layers_kv_cache = cross_layers_kv_cache
            self.cross_layers_attn_backend = attn_backend
        else:
            # ------【核心逻辑】回退通用路径：先分配原始缓冲再 reshape 成层形状 ------
            # Fallback to the general case
            # Initialize the memory buffer for KV cache
            ############################################################################################
            # 1. 最通用的分配显存的方法, 先划分每个layer的tensor
            ############################################################################################
            kv_cache_raw_tensors = self._allocate_kv_cache_tensors(kv_cache_config)

            # Change the memory buffer to the desired shape
            ############################################################################################
            # 2. 对这些个kv cache tensor， reshape出各自的block的维度，然后添加到kv_caches，这个就是我们的block显存池
            ############################################################################################
            kv_caches = self._reshape_kv_cache_tensors(
                kv_cache_raw_tensors, kernel_block_sizes
            )

        # ------【核心逻辑】实现跨层 KV cache 共享：共享层直接引用目标层张量 ------
        # Set up cross-layer KV cache sharing
        for layer_name, target_layer_name in self.shared_kv_cache_layers.items():
            logger.debug("%s reuses KV cache of %s", layer_name, target_layer_name)
            kv_caches[layer_name] = kv_caches[target_layer_name]

        num_attn_module = (
            2 if self.model_config.hf_config.model_type == "longcat_flash" else 1
        )
        # ------【核心逻辑】把 KV cache 绑定到各层静态前向上下文 ------
        ################################################
        # 3. 把划分好的tensor，绑定到model_runner
        ################################################
        bind_kv_cache(
            kv_caches,
            self.compilation_config.static_forward_context,
            self.kv_caches,
            num_attn_module,
        )
        return kv_caches















    def maybe_add_kv_sharing_layers_to_kv_cache_groups(
        self, kv_cache_config: KVCacheConfig
    ) -> None:
        """
        Add layers that re-use KV cache to KV cache group of its target layer.
        Mapping of KV cache tensors happens in `initialize_kv_cache_tensors()`
        """
        # ------【核心逻辑】无跨层 KV 共享时直接返回 ------
        if not self.shared_kv_cache_layers:
            # No cross-layer KV sharing, return
            return

        # ------【核心逻辑】把共享 KV 的层并入其目标层的 KV cache 组 ------
        add_kv_sharing_layers_to_kv_cache_groups(
            self.shared_kv_cache_layers,
            kv_cache_config.kv_cache_groups,
            self.runner_only_attn_layers,
        )

        # ------【前缀缓存】You-Only-Cache-Once：标记仅 prefill 生成 KV 的层以提前退出 ------
        if self.cache_config.kv_sharing_fast_prefill:
            # In You Only Cache Once (https://arxiv.org/abs/2405.05254) or other
            # similar KV sharing setups, only the layers that generate KV caches
            # are involved in the prefill phase, enabling prefill to early exit.
            attn_layers = get_layers_from_vllm_config(self.vllm_config, Attention)
            for layer_name in reversed(attn_layers):
                if layer_name in self.shared_kv_cache_layers:
                    self.kv_sharing_fast_prefill_eligible_layers.add(layer_name)
                else:
                    break










    def initialize_kv_cache(
        self,
        kv_cache_config: KVCacheConfig, # 这个就是已经分配好显存方案的配置文件，各个group的张量布局
        is_profiling: bool = False,
    ) -> None:
        """
        Initialize KV cache based on `kv_cache_config`.
        Args:
            kv_cache_config: Configuration for the KV cache, including the KV
            cache size of each layer
        """
        # ------【核心逻辑】深拷贝配置，避免后续修改影响调用方持有的配置对象 ------
        kv_cache_config = deepcopy(kv_cache_config)
        self.kv_cache_config = kv_cache_config



        self._mamba_bufs = None
        # ------【核心逻辑】先补充 encoder-only 层与 KV 共享层到 KV cache 组 ------
        self.may_add_encoder_only_layers_to_kv_cache_config()
        self.maybe_add_kv_sharing_layers_to_kv_cache_groups(kv_cache_config)


        # ------【核心逻辑】初始化注意力后端, 以及对应符合要求的cuda graph dispatcher （mode, key）库
        ########################
        # 给模型里面的后端分组，然后针对每组atten, 每组对应一个metadata builder（还未创建）, 而不是每个atten 拥有一个builder
        # 然后根据所有group的attention的约束，来初始化dispatcher的库
        ######################## 
        self.initialize_attn_backend(kv_cache_config, is_profiling=is_profiling)


        # 初始化 Mamba SSU 后端 ------
        initialize_mamba_ssu_backend(
            self.vllm_config.mamba_config, self.kv_cache_config
        )
        # The kernel block size for all KV cache groups. For example, if
        # kv_cache_manager uses block_size 256 for a given group, but the attention
        # backends for that group only supports block_size 64, we will return
        # kernel_block_size 64 and split the 256-token-block to 4 blocks with 64
        # tokens each.
        # ------【内存池/CuMem】计算内核实际支持的 block 大小（可能比管理器块更小需拆分） ------
        kernel_block_sizes = prepare_kernel_block_sizes(
            kv_cache_config, self.attn_groups
        )
        self._kernel_block_sizes = kernel_block_sizes

        # create metadata builders
        # ------【核心逻辑】创建元数据构建器并重初始化输入批次 ------
        self.initialize_metadata_builders(kv_cache_config, kernel_block_sizes)

        # Reinitialize need to after initialize_attn_backend
        self.may_reinitialize_input_batch(kv_cache_config, kernel_block_sizes)







        # ------【内存池/CuMem】分配并 reshape 出最终 KV cache 张量 ------
        ############################################################################################
        # 1. 划分不同group的各layer的kvcache tensor区域
        ############################################################################################
        kv_caches = self.initialize_kv_cache_tensors(
            kv_cache_config, kernel_block_sizes
        )

        if (
            self.speculative_config
            and self.speculative_config.uses_extract_hidden_states()
        ):
            assert isinstance(self.drafter, ExtractHiddenStatesProposer)
            # validate all draft model layers belong to the same kv cache
            # group
            self.drafter.validate_same_kv_cache_group(kv_cache_config)

        # ------【PD 分离】存在 KV 传输组时注册 KV cache 供跨节点传输使用 ------
        if has_kv_transfer_group() and not is_profiling:
            kv_transfer_group = get_kv_transfer_group()
            if self.cross_layers_kv_cache is not None:
                assert self.cross_layers_attn_backend is not None
                kv_transfer_group.register_cross_layers_kv_cache(
                    self.cross_layers_kv_cache, self.cross_layers_attn_backend
                )
            else:
                kv_transfer_group.register_kv_caches(kv_caches)
            kv_transfer_group.set_host_xfer_buffer_ops(copy_kv_blocks)

















    def get_routed_experts(
        self,
        num_tokens: int,
    ) -> RoutedExpertsTensors | None:
        # ------【EP/EPLB】未初始化路由专家捕获器时返回 None ------
        if not self.routed_experts_initialized:
            return None

        # ------【EP/EPLB】取出设备侧路由数据与 slot 映射，克隆快照返回 ------
        device_buffer = self.routed_experts_capturer.get_device_buffer()
        return RoutedExpertsTensors(
            routing_data=device_buffer[:num_tokens].clone(),
            slot_mapping=self.routed_experts_slot_mapping_device[:num_tokens].clone(),
        )

    def init_routed_experts_capturer(self):
        logger.info(
            "Initializing routed experts capturer, enable_return_routed_experts: %s",
            self.model_config.enable_return_routed_experts,
        )
        # ------【EP/EPLB】创建路由专家捕获器并绑定到模型，用于记录 token->专家路由 ------
        self.routed_experts_capturer = RoutedExpertsCapturer(
            max_num_batched_tokens=self.scheduler_config.max_num_batched_tokens,
            vllm_config=self.vllm_config,
            kv_cache_config=self.kv_cache_config,
        )
        bind_routed_experts_capturer(self.model, self.routed_experts_capturer)

        # Pinned CPU buffer for non-blocking D2H of ``routing_data`` on
        # the sync scheduling path. Shape / dtype mirror the device
        # capturer exactly so ``copy_`` is a straight memcpy.
        # ------【异步 RPC】分配 pinned CPU 缓冲，非阻塞 D2H 同步路由数据 ------
        self.routed_experts_cpu = torch.empty(
            self.routed_experts_capturer.device_buffer.shape,
            dtype=self.routed_experts_capturer.device_buffer.dtype,
            device="cpu",
            pin_memory=PIN_MEMORY,
        )
        # ``slot_mapping`` dtype is fixed to int64 by
        # ``block_table.slot_mapping``; we mirror that here.
        max_tokens = self.scheduler_config.max_num_batched_tokens
        self.routed_experts_slot_mapping_cpu = torch.empty(
            (max_tokens,),
            dtype=torch.int64,
            device="cpu",
            pin_memory=PIN_MEMORY,
        )
        # Private device buffer so the shared ``block_table.slot_mapping``
        # can be overwritten by the next ``_prepare_inputs`` while the
        # D2H is still pending on the copy stream. Written in
        # ``_prepare_inputs``, read in ``_bookkeeping_sync`` (sync path)
        # or cloned into a snapshot (async path).
        # ------【异步 RPC】私有设备缓冲避免被下一次 _prepare_inputs 覆盖，保证 D2H 期间数据有效 ------
        self.routed_experts_slot_mapping_device = torch.empty(
            (max_tokens,),
            dtype=torch.int64,
            device=self.device,
        )
        self.routed_experts_initialized = True

    def may_add_encoder_only_layers_to_kv_cache_config(self) -> None:
        """
        Add encoder-only layers to the KV cache config.
        """
        # ------【核心逻辑】把 encoder-only 注意力层收集为独立 KV cache 组追加到配置 ------
        block_size = self.vllm_config.cache_config.block_size
        encoder_only_attn_specs: dict[AttentionSpec, list[str]] = defaultdict(list)
        attn_layers = get_layers_from_vllm_config(self.vllm_config, Attention)
        for layer_name, attn_module in attn_layers.items():
            if attn_module.attn_type == AttentionType.ENCODER_ONLY:
                attn_spec: AttentionSpec = EncoderOnlyAttentionSpec(
                    block_size=block_size,
                    num_kv_heads=attn_module.num_kv_heads,
                    head_size=attn_module.head_size,
                    dtype=self.kv_cache_dtype,
                )
                encoder_only_attn_specs[attn_spec].append(layer_name)
                self.runner_only_attn_layers.add(layer_name)
        # ------【核心逻辑】仅支持单一 encoder-only 规格，追加为新的 KV cache 组 ------
        if len(encoder_only_attn_specs) > 0:
            assert len(encoder_only_attn_specs) == 1, (
                "Only support one encoder-only attention spec now"
            )
            spec, layer_names = encoder_only_attn_specs.popitem()
            self.kv_cache_config.kv_cache_groups.append(
                KVCacheGroupSpec(layer_names=layer_names, kv_cache_spec=spec)
            )

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        """
        Generates the KVCacheSpec by parsing the kv cache format from each
        Attention module in the static forward context.
        Returns:
            KVCacheSpec: A dictionary mapping layer names to their KV cache
            format. Layers that do not need KV cache are not included.
        """
        # ------【PD 分离】EC 传输的非消费者端无需本地 KV cache，返回空 ------
        if has_ec_transfer() and not get_ec_transfer().is_consumer:
            return {}
        kv_cache_spec: dict[str, KVCacheSpec] = {}
        layer_type = cast(type[Any], AttentionLayerBase)
        attn_layers = get_layers_from_vllm_config(self.vllm_config, layer_type)
        # ------【核心逻辑】逐层生成 KV cache 规格，跳过跨层共享与无需 KV 的层 ------
        for layer_name, attn_module in attn_layers.items():
            if isinstance(attn_module, Attention) and (
                kv_tgt_layer := attn_module.kv_sharing_target_layer_name
            ):
                # The layer doesn't need its own KV cache and will use that of
                # the target layer. We skip creating a KVCacheSpec for it, so
                # that KV cache management logic will act as this layer does
                # not exist, and doesn't allocate KV cache for the layer. This
                # enables the memory saving of cross-layer kv sharing, allowing
                # a given amount of memory to accommodate longer context lengths
                # or enable more requests to be processed simultaneously.
                self.shared_kv_cache_layers[layer_name] = kv_tgt_layer
                continue
            # Skip modules that don't need KV cache (eg encoder-only attention)
            if spec := attn_module.get_kv_cache_spec(self.vllm_config):
                if isinstance(spec, AttentionSpec):
                    backend = attn_module.get_attn_backend()
                    # indexes_kv_by_block_stride() -> get_kv_cache_stride_order()
                    # -> get_kv_cache_layout() needs the current vLLM config.
                    with set_current_vllm_config(self.vllm_config):
                        indexes = backend.indexes_kv_by_block_stride()
                    spec = replace(spec, indexes_kv_by_block_stride=indexes)
                kv_cache_spec[layer_name] = spec

        return kv_cache_spec

    def _to_list(self, sampled_token_ids: torch.Tensor) -> list[list[int]]:
        # This is a short term mitigation for issue mentioned in
        # https://github.com/vllm-project/vllm/issues/22754.
        # `tolist` would trigger a cuda wise stream sync, which
        # would block other copy ops from other cuda streams.
        # A cuda event sync would avoid such a situation. Since
        # this is in the critical path of every single model
        # forward loop, this has caused perf issue for a disagg
        # setup.
        # ------【异步 RPC】用事件同步替代流同步，避免阻塞其它 copy 流（disagg 关键路径） ------
        pinned = self.sampled_token_ids_pinned_cpu[: sampled_token_ids.shape[0]]
        pinned.copy_(sampled_token_ids, non_blocking=True)
        self.transfer_event.record()
        self.transfer_event.synchronize()
        return pinned.tolist()

    def get_encoder_timing_stats(self) -> dict[str, dict[str, float | int]]:
        """
        Get encoder timing stats for all requests and clear the registry.

        Returns:
            Dictionary mapping request_id to stats dict.
        """
        # ------【异步 RPC】加锁快照编码器耗时统计并清空注册表，避免并发读写 ------
        with self._encoder_timing_lock:
            stats = {
                req_id: stats_obj.to_dict()
                for req_id, stats_obj in self.encoder_timing_registry.items()
            }
            self.encoder_timing_registry.clear()
            return stats

    @contextmanager
    def timed_encoder_operation(
        self,
        should_time: bool,
        group_lora_refs: list[tuple[str, Any]],
        current_item_idx: int,
        num_items: int,
    ):
        """
        Context manager to time encoder forward operations.

        Args:
            should_time: Whether timing is enabled
            group_lora_refs: Full list of (request_id, pos_info) tuples
            current_item_idx: Starting index for this group
            num_items: Number of items in this group
        """
        # ------【核心逻辑】未开启计时时直接透传 yield ------
        if not should_time:
            yield
            return

        # ------【核心逻辑】截取本组请求 id，并同步设备后记录起始时间 ------
        group_refs = group_lora_refs[current_item_idx : current_item_idx + num_items]
        group_request_ids = {req_id for req_id, _ in group_refs}

        torch.accelerator.synchronize()
        start_time = time.perf_counter()

        try:
            yield
        finally:
            torch.accelerator.synchronize()
            elapsed = time.perf_counter() - start_time

            per_request_time = elapsed / max(len(group_request_ids), 1)

            # ------【异步 RPC】把本次编码器耗时均摊到各请求并加锁累加统计 ------
            with self._encoder_timing_lock:
                for req_id in group_request_ids:
                    if req_id not in self.encoder_timing_registry:
                        self.encoder_timing_registry[req_id] = EncoderTimingStats()

                    stats = self.encoder_timing_registry[req_id]
                    stats.encoder_forward_secs += per_request_time
                    stats.num_encoder_calls += 1


@dataclass
class EncoderTimingStats:
    """Per-request timing statistics for encoder forward pass."""

    encoder_forward_secs: float = 0.0
    """Time spent in vision encoder forward pass (seconds)."""

    num_encoder_calls: int = 0
    """Number of times encoder was called for this request."""

    def to_dict(self) -> dict[str, float | int]:
        return {
            "encoder_forward_secs": self.encoder_forward_secs,
            "num_encoder_calls": self.num_encoder_calls,
        }
