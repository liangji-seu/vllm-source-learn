# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import multiprocessing
import os
import pickle
import queue
import signal
import threading
import time
import traceback
import weakref
from collections import deque
from collections.abc import Callable, Sequence
from concurrent.futures import Future, InvalidStateError
from contextlib import suppress
from dataclasses import dataclass
from enum import Enum, auto
from functools import partial
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
from multiprocessing.synchronize import Lock as LockType
from threading import Thread
from typing import Any, cast

import cloudpickle
import torch

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.distributed import destroy_distributed_environment, destroy_model_parallel
from vllm.distributed.device_communicators.shm_broadcast import Handle, MessageQueue
from vllm.distributed.kv_transfer.kv_connector.utils import KVOutputAggregator
from vllm.distributed.parallel_state import (
    get_dcp_group,
    get_dp_group,
    get_ep_group,
    get_inner_dp_world_group,
    get_pcp_group,
    get_pp_group,
    get_tp_group,
    model_parallel_is_initialized,
)
from vllm.envs import enable_envs_cache
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.tracing import instrument, maybe_init_worker_tracer
from vllm.utils import numa_utils
from vllm.utils.network_utils import (
    get_distributed_init_method,
    get_ip,
    get_loopback_ip,
    get_open_port,
)
from vllm.utils.ompmultiprocessing import OMPProcessManager
from vllm.utils.system_utils import (
    _maybe_force_spawn,
    decorate_logs,
    get_mp_context,
    set_process_title,
)
from vllm.utils.torch_utils import (
    OMP_NUM_THREADS_SET_BY_VLLM,
    set_torch_threads_for_runtime,
    startup_omp_num_threads,
)
from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
from vllm.v1.executor.abstract import Executor, FailureCallback
from vllm.v1.executor.vllm_net_devices import set_worker_net_device
from vllm.v1.outputs import AsyncModelRunnerOutput, DraftTokenIds, ModelRunnerOutput
from vllm.v1.worker.worker_base import WorkerWrapperBase

logger = init_logger(__name__)


class FutureWrapper(Future):
    '''
    有序异步 RPC
    '''
    def __init__(
        self,
        futures_queue: deque["FutureWrapper"],
        get_response: Callable[[], Any],
        aggregate: Callable = lambda x: x,
    ):
        # ------【异步 RPC】构造时把自己压入 FIFO 队列：executor 按入队顺序 drain 结果 ------
        self.futures_queue = futures_queue
        self.get_response = get_response
        self.aggregate = aggregate
        super().__init__()
        self.futures_queue.appendleft(self)

    def result(self, timeout=None):
        if timeout is not None:
            raise RuntimeError("timeout not implemented")

        # ------【异步 RPC】顺序性保证：先 drain 队列中比自己更早的 Future 再取自身结果 ------
        # Drain any futures ahead of us in the queue.
        while not self.done():
            future = self.futures_queue.pop()
            future._wait_for_response()
        return super().result()

    def _wait_for_response(self):
        # ------【异步 RPC】真正取回结果：调 get_response 读响应队列，聚合后 set_result ------
        try:
            response = self.aggregate(self.get_response())
            with suppress(InvalidStateError):
                self.set_result(response)
        except Exception as e:
            with suppress(InvalidStateError):
                self.set_exception(e)


class MultiprocExecutor(Executor):
    """
    === 类说明 ===
        继承: Executor (ABC)
        适用: distributed_executor_backend == "mp", 单机多卡场景
        职责: 单机多进程执行器。为每个 GPU 创建一个独立的 Worker 进程,
              通过 MessageQueue (共享内存广播) 实现 Scheduler → Worker 的
              指令分发和结果回收。

    === 架构: 1 Scheduler 进程 + N Worker 进程 ===
        Scheduler (本进程)                  Worker 0    Worker 1    ...
        │                                    │           │
        ├─ rpc_broadcast_mq ────broadcast───→│←──input──→│←──input──→│
        │                                    │           │
        │  response_mqs[0] ←────dequeue──────│           │
        │  response_mqs[1] ←────dequeue─────────────────│
        │  ...
        │
        通信: Scheduler → Worker 用 MessageQueue 广播 SchedulerOutput
              Worker → Scheduler 各用独立的 response_mq 回传结果

    === 核心成员属性 (新增) ===
        workers: list[WorkerProcHandle]       — Worker 进程句柄列表 (每个 GPU 一个)
        rpc_broadcast_mq: MessageQueue        — 广播消息队列 (Scheduler → 所有 Worker)
        response_mqs: list[MessageQueue]      — 响应消息队列 (每个 Worker 一个, Worker → Scheduler)
        futures_queue: deque[FutureWrapper]   — 异步 Future 队列
        is_failed: bool                       — 执行器是否已故障 (worker 死亡)
        failure_callback: FailureCallback     — 故障回调 (通知 EngineCore)
        monitor_workers: bool                 — 是否启动 Worker 健康监控线程
        output_rank: int                      — 输出结果的 Worker rank (通常 TP rank 0 + PP last stage)
        world_size / local_world_size         — 全局（单个模型副本的）/本地（本机） Worker 数量 （单机情况两个相等）

    === 核心方法 ===
        —— 初始化 ——
            _init_executor()           — 创建 Worker 进程 + IPC 基础设施 + 启动健康监控
        —— RPC ——
            collective_rpc(method, ...) — 覆写: 通过 rpc_broadcast_mq 广播, response_mq 收结果
        —— Worker 动作 ——
            execute_model(scheduler_output) — 覆写: collective_rpc("execute_model", ...)
        —— 生命周期 ——
            shutdown()                 — 关闭所有 Worker 进程 + 清理 IPC 资源
            check_health()             — 覆写: 向 Worker 发 health check RPC
            start_worker_monitor()     — 启动后台线程监控 Worker 进程存活

    === 继承自 Executor.__init__ ===
        vllm_config, model_config, cache_config, lora_config, load_config,
        parallel_config, scheduler_config, device_config, speculative_config,
        observability_config, is_sleeping, kv_output_aggregator
    """
    supports_pp: bool = True

    def __init__(self, vllm_config: VllmConfig, monitor_workers: bool = True):
        # ------【进程管理】记录是否开启 worker 健康监控，其余配置字段交给基类 Executor 初始化 ------
        self.monitor_workers = monitor_workers
        super().__init__(vllm_config)

    # 基类调用子类的初始化
    def _init_executor(self) -> None:

        # Call self.shutdown at exit to clean up
        # and ensure workers will be terminated.


        # weakref库，弱引用库，不增加对象引用的情况下访问对象
        # finalize, 这个库里面的注册清理回调的工具
        # ------【进程管理】注册析构钩子：executor 被 GC 时自动 shutdown 清理 worker 进程 ------
        # 我们这边注册multiprocexecutor对象的退出钩子函数：shutdown
        self._finalizer = weakref.finalize(self, self.shutdown) # 对象析构的钩子，如果实例对象被GC回收了，自动调用shutdown()清理worker进程

        # ------【进程管理】初始化故障状态与回调槽位：worker 意外死亡时置位并通知 EngineCore ------
        self.is_failed = False
        self.failure_callback: FailureCallback | None = None




        # ------【TP + PP + PCP】读取张量/流水线/prefill-context 并行度并校验 world_size ------
        # 获取并行参数
        tp_size, pp_size, pcp_size = self._get_parallel_sizes()
        assert self.world_size == tp_size * pp_size * pcp_size, (
            f"world_size ({self.world_size}) must be equal to the "
            f"tensor_parallel_size ({tp_size}) x pipeline"
            f"_parallel_size ({pp_size}) x prefill_context"
            f"_parallel_size ({pcp_size}). "
        )




        # 设置多进程worker的环境：
        # 1. 强制用spawn 而不是fork
        # 2. 设置OMP_NUM_THREADS 控制pytorch CPU线程数
        '''
        PyTorch 默认的 intra-op 线程数 = CPU 核数（比如 128）。
        如果本机跑了 8 个 Worker 进程，每个 Worker 都抢 128 个线程 → 128 × 8 = 大量 CPU 争抢，性能反而差。

        vLLM 的做法：startup_omp_num_threads 根据 local_world_size 均分 CPU 核数。比如 128 核 / 8 Workers = 每 Worker 16 线程。
        子进程继承这个环境变量后 torch 启动时就只用 16 个线程。
        ? 这里还不是很理解
        '''
        # ------【进程管理】设置多进程环境：强制 spawn 启动 + 按 world_size 均分 CPU 线程数 ------
        set_multiprocessing_worker_envs(self.local_world_size) # 设置每个worker的pytorch c++ 计算的线程数，这样每个worker的pytorch计算互不影响





        # 设置worker们的分布式通信环境
        # use the loopback address get_loopback_ip() for communication.
        '''
            distributed_init_method = get_distributed_init_method(get_loopback_ip(), get_open_port())
            # → "tcp://127.0.0.1:29500"
            作用：让本机所有 Worker 进程互相发现、建立 NCCL 集合通信。
        '''
        # 就是返回一个socket通信的地址：tcp://127.0.0.1:29500
        # ------【NCCL 通信】生成 rendezvous 地址(tcp://127.0.0.1:端口)，供 worker 初始化分布式进程组 ------
        distributed_init_method = get_distributed_init_method( # 给 torch.distributed.init_process_group() 的 rendezvous 地址
            get_loopback_ip(), get_open_port() # 127.0.0.1 可用端口号
        )

        # ------【异步 RPC】广播/响应队列占位：Scheduler→Worker 指令走广播 MQ，Worker→Scheduler 走响应 MQ ------
        # 构造调度任务的广播的环形缓冲区
        self.rpc_broadcast_mq: MessageQueue | None = None

        # 连接信息
        scheduler_output_handle: Handle | None = None 

        '''
        ┌─ 父进程 (Executor/Scheduler) ─────────────────────────────┐
        │                                                             │
        │  MessageQueue (writer 端)                                    │
        │    ├── 共享内存缓冲区 ──────────────────────┐                │
        │    ├── ZMQ PUB socket 广播"数据到了" ──────┼───┐            │
        │    └── SpinCondition 通知                   │   │            │
        │                                              │   │           │
        │  export_handle() 把这些资源的"描述符"导出:      │   │           │
        │    buffer_handle     → 共享内存的 fd/大小/名称  │   │           │
        │    local_subscribe_addr → "tcp://127.0.0.1:xx" │   │          │
        │    local_notify_addr   → "ipc:///tmp/xxx"      │   │          │
        └──────────────────────────────────┬──────────────┼───┼────────┘
                                           │              │   │
                                    Handle 传参          │   │
                                        │              │   │
        ┌─ 子进程 (Worker rank=0) ────────┼──────────────┼───┼────────┐
        │                                 ▼              │   │        │
        │  create_from_handle(handle, rank=0)            │   │        │
        │    → buffer_handle 拿去 mmap 映射共享内存 ◄─────┘   │        │
        │    → local_subscribe_addr 拿去 connect ZMQ ◄────────┘        │
        │                                                             │
        │  读数据：ZMQ 通知 → 读共享内存 → 拿到 SchedulerOutput          │
        └─────────────────────────────────────────────────────────────┘
        
        '''
        # Initialize worker and set up message queues for SchedulerOutputs
        # and ModelRunnerOutputs

        '''
        每个节点都有自己的 Executor，不是只有 rank=0 才有 Executor。每个 Executor 管理本节点上的 Worker 进程。区别在于：

            node_rank_within_dp == 0：这个 Executor 额外负责创建广播 MessageQueue，把 SchedulerOutput 推送出去给同 DP 组的其他节点
            node_rank_within_dp != 0：这个 Executor 不创建广播队列，它只管理自己的 Worker，SchedulerOutput 从 leader 那边收
        '''
        # ------【DP】仅 DP 组 leader 节点创建广播 MQ：非 leader 节点从 leader 收 SchedulerOutput ------
        # 只有 DP 组内的 leader 节点创建广播消息队列：一个模型副本，只有一个主节点发布消息
        if self.parallel_config.node_rank_within_dp == 0: # DP组内的节点编号
            # For leader node within each dp rank,
            # each dp will have its own leader multiproc executor.
            '''
            1. max_chunk_bytes 用于控制进程间通信的数据传输方式：
                当序列化后的数据总大小小于 max_chunk_bytes 时，通过共享内存环形缓冲区（ShmRingBuffer）直接传输，实现零拷贝；
            2. 当数据总大小达到或超过 max_chunk_bytes 时，共享内存仅写入一个溢出标记（overflow flag = 1），
                实际数据改由 rpc_broadcast_mq 关联的 ZMQ XPUB/SUB 套接字进行传输，以避免共享内存预分配过大。

                max_chunk_bytes 的本质是「环形缓冲区单个槽位的固定大小」
                ——数据区布局是 max_chunks × max_chunk_bytes（见 shm_broadcast.py:313），一条消息必须塞进一个槽。塞不下的就溢出到 ZMQ
            '''
            max_chunk_bytes = envs.VLLM_MQ_MAX_CHUNK_BYTES_MB * 1024 * 1024 # 消息分片字节长度
            mq_connect_ip = get_ip() # 本机的真实网卡ip（用来多节点通信的，因为别的机器，肯定需要真正的ip）单机用不上, 多机用来节点通信
            logger.info(
                "DP group leader: node_rank=%d, node_rank_within_dp=%d, "
                "master_addr=%s, mq_connect_ip=%s (local), "
                "world_size=%d, local_world_size=%d",
                self.parallel_config.node_rank,
                self.parallel_config.node_rank_within_dp,
                self.parallel_config.master_addr,
                mq_connect_ip,
                self.world_size,
                self.local_world_size,
            )

            ##########################################################################################
            # 1. 创建发送任务的广播队列
            ##########################################################################################
            self.rpc_broadcast_mq = MessageQueue( # 创建广播环形缓冲区
                self.world_size, # 卡数量
                self.local_world_size, # 节点内卡的数量
                max_chunk_bytes=max_chunk_bytes, # 广播环形缓冲区，单个slot槽位的字节数
                connect_ip=mq_connect_ip, # 节点ip
            )
            scheduler_output_handle = self.rpc_broadcast_mq.export_handle() # 导出连接信息，都是些地址信息


        
        # Create workers
        # ------【进程管理】准备多进程启动上下文、跨进程共享锁与待就绪列表，并预置成败标志 ------
        context = get_mp_context() # python的multiprocessing模块的进程启动上下文
        shared_worker_lock = context.Lock() # 跨进程互斥锁，多个worker进程共享，唯一用途：保护多模态的共享内存缓存（mm_processor_cache_type='shm'）。
        unready_workers: list[UnreadyWorkerProcHandle] = []
        success = False

        # ------【进程管理】逐个 GPU 拉起 Worker 子进程，并搭建 ready/death 两条 IPC 管道 ------
        # 创建N个worker进程 + 搭建IPC通信链路
        try:
            global_start_rank = (
                self.local_world_size * self.parallel_config.node_rank_within_dp
            )
            # When using fork, keep track of socket file descriptors that are
            # inherited by the worker, so that we can close them in subsequent
            # workers
            inherited_fds: list[int] | None = (
                [] if context.get_start_method() == "fork" else None
            )

            # For CPU backend only, to setup OpenMP threads affinity
            # CPU 专属功能——把每个 Worker 进程的 OpenMP 线程绑定到特定的 CPU 核心上，防止线程乱跳。GPU 后端这个类什么都不做
            # ------【NUMA 亲和/进程管理】构建 OpenMP 线程绑定管理器：CPU 后端按 rank 绑核，GPU 后端为空操作 ------
            cpu_omp_manager = OMPProcessManager(self.vllm_config)


            # ------【进程管理】逐个 GPU 构建 Worker 子进程句柄，并在 CPU 后端为其配置 OpenMP 绑核 ------
            # 对于每一个卡的worker
            for local_rank in range(self.local_world_size):
                global_rank = global_start_rank + local_rank # 定义全局rank号
                '''
                driver worker 负责多做两件事：模型参数加载（只加载一次，然后广播给同 TP 组的其他 Worker）和流水线并行的调度协调。
                普通 Worker 等着从 driver 那边收参数就行。
                '''
                is_driver_worker = self._is_driver_worker(global_rank)

                with cpu_omp_manager.configure_omp_envs( # GPU环境下，这个可以跳过
                    rank=global_rank, local_rank=local_rank
                ):

                    ##########################################################################################
                    # 2. 创建并启动Worker进程
                    ##########################################################################################
                    unready_worker_handle = WorkerProc.make_worker_process( # 创建进程实例并启动，还未启动完成
                        vllm_config=self.vllm_config,
                        local_rank=local_rank, # 单卡的单机id
                        rank=global_rank, # 单卡的跨机全局id
                        distributed_init_method=distributed_init_method, # executor的连接地址
                        input_shm_handle=scheduler_output_handle, # 把连接信息传递给每一个worker进程
                        shared_worker_lock=shared_worker_lock, # 多模态，跳过
                        is_driver_worker=is_driver_worker, #该worker是否是TP组的driver worker
                        inherited_fds=inherited_fds,
                    )


                unready_workers.append(unready_worker_handle)

                # ------【进程管理】fork 模式下记录新 worker 管道 fd，供后续 worker 关闭继承句柄防止 EOF 失效 ------
                # 这几个是给fork兜底的，spawn不会执行，没有socket资源需要继承
                if inherited_fds is not None:
                    inherited_fds.append(unready_worker_handle.death_writer.fileno())
                    inherited_fds.append(unready_worker_handle.ready_pipe.fileno())

            # Workers must be created before wait_for_ready to avoid
            # deadlock, since worker.init_device() does a device sync.

            # ------【进程管理】就绪握手：阻塞等待每个 worker 通过 ready_pipe 回传 READY + 响应 MQ 句柄 ------
            # Wait for all local workers to be ready.
            # 等待所有worker子进程启动完成连接
            self.workers = WorkerProc.wait_for_ready(unready_workers)

            # The workers have inherited their thread count (see
            # set_multiprocessing_worker_envs); this process only schedules, so
            # it gets no benefit from torch intra-op parallelism, just CPU
            # contention with them.
            # ------【进程管理】Scheduler 进程只调度不算矩阵，关掉 torch 的 CPU 并行避免与 worker 抢核 ------
            set_torch_threads_for_runtime() # 用户手动设了 OMP_NUM_THREADS: return   # 尊重用户，不动

            # ------【进程管理】启动后台线程监控 worker 存活，异常退出时触发 failure_callback ------
            # Start background thread to monitor worker health if not in headless mode.
            if self.monitor_workers:
                self.start_worker_monitor()

            # ------【异步 RPC + DP】收集响应队列：本节点直连，跨节点经 driver worker 的 peer 句柄 ------
            self.response_mqs = [] # 收集每个worker的响应通道
            # Only leader node have remote response mqs
            if self.parallel_config.node_rank_within_dp == 0: # 主executor in DP

                # 对于这个executor的模型副本下的所有worker
                for rank in range(self.world_size):
                    # 如果他是本节点的worker
                    if rank < self.local_world_size:
                        local_message_queue = self.workers[rank].worker_response_mq # 直接获取他们的回复队列
                        assert local_message_queue is not None
                        self.response_mqs.append(local_message_queue)

                    # 他是其他节点的worker
                    else:
                        remote_message_queue = self.workers[0].peer_worker_response_mqs[ # # 从远程 handle 连接 【待理解】
                            rank
                        ]
                        assert remote_message_queue is not None
                        self.response_mqs.append(remote_message_queue)

            # Ensure message queues are ready. Will deadlock if re-ordered
            # Must be kept consistent with the WorkerProc.


            # 确保广播，接受结果的消息队列都是可用的
            # Wait for all input mqs to be ready.
            '''
            ① worker 连上来 + subscribe          → 订阅消息自动发给 executor（XPUB）
            ② executor 收够 N 条订阅消息          = 确认「所有 worker 都连上、都订阅了」
            ③ executor 广播 "READY"             → executor 主动发（这一步才对应你说的"发布消息"）
            ④ worker 收到 "READY"              = 确认「通道能真正送达」
            '''
            # ------【异步 RPC】等待广播/响应队列握手就绪：确保所有 worker 已订阅、通道可送达 ------
            if self.rpc_broadcast_mq is not None:
                self.rpc_broadcast_mq.wait_until_ready()
            # Wait for all remote response mqs to be ready.
            for response_mq in self.response_mqs:
                response_mq.wait_until_ready() # 等待每个worker的响应通道都就绪

            # ------【异步 RPC】Future 先进先出队列：保证非阻塞 RPC 结果按发出顺序被取走 ------
            # 异步RPC的future FIFI队列
            self.futures_queue = deque[FutureWrapper]()

            # ------【进程管理】后置初始化钩子 + 标记启动成功：失败时 finally 分支负责清理已拉起进程 ------
            self._post_init_executor() # 空的，无后处理

            success = True # 执行器启动成功
        finally:
            if not success:
                # ------【进程管理】启动失败兜底：先关 death_writer 发退出信号，再强制终止残余 worker 进程 ------
                # Clean up the worker procs if there was a failure.
                # Close death_writers first to signal workers to exit
                for uw in unready_workers:
                    if uw.death_writer is not None:
                        uw.death_writer.close()
                        uw.death_writer = None
                self._ensure_worker_termination([uw.proc for uw in unready_workers])

        # ------【TP + PP】记录唯一回传结果的 worker rank：TP rank=0 且 PP 最后一层 ------
        self.output_rank = self._get_output_rank()










    def get_response_mqs(self, unique_reply_rank: int = -1) -> list[MessageQueue]:
        # ------【异步 RPC】校验 reply rank 合法后返回对应 worker 的响应队列：-1 表示全部 ------
        assert unique_reply_rank >= -1 and unique_reply_rank < self.world_size, (
            f"unique_reply_rank must be -1 or < world_size,"
            f"unique_reply_rank = {unique_reply_rank}, "
            f"world_size={self.world_size}"
        )
        ranks = (
            [unique_reply_rank] if unique_reply_rank != -1 else range(self.world_size)
        )
        return [self.workers[rank].worker_response_mq for rank in ranks]

    def _get_parallel_sizes(self) -> tuple[int, int, int]:
        # ------【TP + PP + PCP】从 parallel_config 读出全局/本地 world_size 与各并行度，校验整除性 ------
        self.world_size = self.parallel_config.world_size
        assert self.world_size % self.parallel_config.nnodes_within_dp == 0, (
            f"global world_size ({self.parallel_config.world_size}) must be "
            f"divisible by nnodes_within_dp "
            f"({self.parallel_config.nnodes_within_dp}). "
        )
        self.local_world_size = self.parallel_config.local_world_size
        tp_size = self.parallel_config.tensor_parallel_size
        pp_size = self.parallel_config.pipeline_parallel_size
        pcp_size = self.parallel_config.prefill_context_parallel_size
        return tp_size, pp_size, pcp_size

    def _post_init_executor(self) -> None:
        pass

    def _is_driver_worker(self, rank: int) -> bool:
        # ------【TP】TP 组内 rank 为 0 的 worker 当 driver：负责加载并广播权重给同组其它 rank ------
        return rank % self.parallel_config.tensor_parallel_size == 0

    def start_worker_monitor(self, inline=False) -> None:
        # ------【进程管理】持有 worker 列表并弱引用 executor，供后台线程在 worker 死亡时安全回调 ------
        workers = self.workers
        self_ref = weakref.ref(self)

        # Monitors worker process liveness. If any die unexpectedly,
        # logs an error, shuts down the executor and invokes the failure
        # callback to inform the engine.
        # ------【进程管理】阻塞等待任一 worker sentinel 触发，判死并 shutdown + 回调 EngineCore ------
        def monitor_workers():
            sentinels = [h.proc.sentinel for h in workers]
            died = multiprocessing.connection.wait(sentinels)
            _self = self_ref()
            if not _self or getattr(_self, "shutting_down", False):
                logger.debug("MultiprocWorkerMonitor: shutdown already initiated")
                return
            _self.is_failed = True
            proc = next(h.proc for h in workers if h.proc.sentinel == died[0])
            logger.error(
                "Worker proc %s died unexpectedly (exit code: %s), "
                "shutting down executor.",
                proc.name,
                proc.exitcode,
            )
            _self.shutdown()
            callback = _self.failure_callback
            if callback is not None:
                _self.failure_callback = None
                callback()

        # ------【进程管理】inline 直接同步执行监控，否则起守护线程异步监控 ------
        if not inline:
            Thread(
                target=monitor_workers, daemon=True, name="MultiprocWorkerMonitor"
            ).start()
            return

        monitor_workers()

    def register_failure_callback(self, callback: FailureCallback):
        # ------【进程管理】注册故障回调：若已失败立即触发，否则存起来待 worker 死亡时调用 ------
        if self.is_failed:
            callback()
        else:
            self.failure_callback = callback

    def execute_model(  # type: ignore[override]
        self, scheduler_output: SchedulerOutput, non_block: bool = False
    ) -> ModelRunnerOutput | None | Future[ModelRunnerOutput | None]:
        # ------【异步 RPC】把 SchedulerOutput 广播给 worker 执行前向，只回收 output_rank 的结果 ------
        return self.collective_rpc(
            "execute_model",
            args=(scheduler_output,),
            unique_reply_rank=self.output_rank,
            non_block=non_block,
            timeout=envs.VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS,
            kv_output_aggregator=self.kv_output_aggregator,
        )

    def sample_tokens(  # type: ignore[override]
        self, grammar_output: GrammarOutput | None, non_block: bool = False
    ) -> ModelRunnerOutput | Future[ModelRunnerOutput]:
        # ------【异步 RPC + 结构化输出】广播采样指令（携带 grammar 约束），仅回收 output_rank 结果 ------
        return self.collective_rpc(
            "sample_tokens",
            args=(grammar_output,),
            unique_reply_rank=self.output_rank,
            non_block=non_block,
            timeout=envs.VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS,
            kv_output_aggregator=self.kv_output_aggregator,
        )

    def execute_dummy_batch(self) -> None:
        # ------【CUDA Graph + 异步 RPC】广播执行 dummy batch，用于预热/捕获 CUDA Graph ------
        self.collective_rpc("execute_dummy_batch", unique_reply_rank=self.output_rank)

    def take_draft_token_ids(self) -> DraftTokenIds | None:
        # ------【投机解码 + 异步 RPC】只从 output_rank 取 draft token，减少跨 worker 通信开销 ------
        # OPTIMIZATION: Get output only from a single worker (output_rank)
        return self.collective_rpc(
            "take_draft_token_ids", unique_reply_rank=self.output_rank
        )

    def collective_rpc(  # type: ignore[override]
        self,
        method: str | Callable,
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict | None = None,
        non_block: bool = False,
        unique_reply_rank: int | None = None,
        kv_output_aggregator: KVOutputAggregator | None = None,
    ) -> Any:
        """Returns single result if unique_reply_rank and/or kv_output_aggregator
        is provided, otherwise list."""
        # ------【异步 RPC】前置检查：leader 节点才有广播 MQ；失败态直接抛错 ------
        assert self.rpc_broadcast_mq is not None, (
            "collective_rpc should not be called on follower node"
        )
        if self.is_failed:
            raise RuntimeError("Executor failed.")

        # ------【异步 RPC】计算超时截止时间并规整 kwargs 为字典 ------
        deadline = None if timeout is None else time.monotonic() + timeout
        kwargs = kwargs or {}

        # ------【PD 分离】若带 KV 聚合器则收集所有 worker 的 KV 传输结果再合并，而非只取单 rank ------
        if kv_output_aggregator is not None:
            output_rank = None
            aggregate: Callable[[Any], Any] = partial(
                kv_output_aggregator.aggregate, output_rank=unique_reply_rank or 0
            )
        else:
            output_rank = unique_reply_rank
            aggregate = lambda x: x

        # ------【异步 RPC】把 (方法, 参数, 结果 rank) 打包后广播进 MessageQueue，唤醒所有 worker ------
        if isinstance(method, str):
            send_method = method
        else:
            send_method = cloudpickle.dumps(method, protocol=pickle.HIGHEST_PROTOCOL)
        self.rpc_broadcast_mq.enqueue((send_method, args, kwargs, output_rank)) # 执行器发出 RPC广播 给workers

        # ------【异步 RPC】选定等待结果的响应队列：指定 output_rank 只等单个，否则等全部 worker ------
        response_mqs: Sequence[MessageQueue] = self.response_mqs
        if output_rank is not None:
            response_mqs = (response_mqs[output_rank],)


        # ------【异步 RPC】定义取结果回调：逐个 dequeue 响应队列，检查 SUCCESS/FAILURE 并解包 ------
        # 定义如何从响应队列里面获取异步RPC的回复
        def get_response():
            responses = []
            for mq in response_mqs:
                dequeue_timeout = (
                    None if deadline is None else max(0.0, deadline - time.monotonic())
                )
                try:
                    status, result = mq.dequeue(timeout=dequeue_timeout)
                except TimeoutError as e:
                    raise TimeoutError(f"RPC call to {method} timed out.") from e
                if status != WorkerProc.ResponseStatus.SUCCESS:
                    raise RuntimeError(
                        f"Worker failed with error '{result}', please check the"
                        " stack trace above for the root cause"
                    )
                responses.append(result)
            return responses[0] if output_rank is not None else responses


        # ------【异步 RPC】包装成 Future 入队：non_block 立即返回，否则调用 result() 阻塞等待 ------
        # executor 发出RPC，远程过程调用，给worker， 肯定不能阻塞等待，给挂一个回调任务
        # 这个回调任务就是
        # 发出RPC后，不阻塞等待结果，直接包装一个Future，然后立刻返回。不等worker
        # 等executor真正需要结果时，调用future.result(), 他会先drain掉队里所有比自己老的future, 然后再取自己
        future = FutureWrapper(
            self.futures_queue, get_response=get_response, aggregate=aggregate
        )

        return future if non_block else future.result()

    @staticmethod
    def _ensure_worker_termination(worker_procs: list[BaseProcess]):
        """Ensure that all worker processes are terminated. Assumes workers have
        received termination requests. Waits for processing, then sends
        termination and kill signals if needed."""

        # ------【进程管理】轮询等待进程退出，超时返回 False；解释器晚期 time 为 None 时只看存活态 ------
        def wait_for_termination(procs, timeout):
            if not time:
                # If we are in late stage shutdown, the interpreter may replace
                # `time` with `None`.
                return all(not proc.is_alive() for proc in procs)
            start_time = time.time()
            while time.time() - start_time < timeout:
                if all(not proc.is_alive() for proc in procs):
                    return True
                time.sleep(0.1)
            return False

        # ------【进程管理】过滤出仍存活的 worker 进程并记录初始数量 ------
        active_procs = lambda: [proc for proc in worker_procs if proc.is_alive()]
        initial_count = len(active_procs())

        # ------【进程管理】先给宽限期让 worker 自行清理退出，全部退出则直接返回 ------
        # Give processes time to clean themselves up properly first
        logger.info(
            "[shutdown] Executor: waiting for worker exit count=%d",
            initial_count,
        )
        if wait_for_termination(
            active_procs(), timeout=envs.VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS
        ):
            logger.info_once("[shutdown] Executor: all workers exited gracefully")
            return

        # Send SIGTERM if still running
        # ------【进程管理】宽限期后仍存活则发送 SIGTERM 请求优雅终止 ------
        remaining = active_procs()
        logger.warning(
            "[shutdown] Executor: workers still running after grace period; "
            "sending SIGTERM count=%d",
            len(remaining),
        )
        for p in remaining:
            p.terminate()
        if not wait_for_termination(active_procs(), 4):
            # ------【进程管理】SIGTERM 后仍未退出则升级为 SIGKILL 强制杀死 ------
            # Send SIGKILL if still running
            remaining = active_procs()
            logger.warning(
                "[shutdown] Executor: workers still running after SIGTERM; "
                "sending SIGKILL count=%d",
                len(remaining),
            )
            for p in remaining:
                p.kill()

    def shutdown(self):
        """Properly shut down the executor and its workers"""
        # ------【进程管理】幂等关闭：置位 shutting_down 防止重复执行 ------
        if not getattr(self, "shutting_down", False):
            worker_count = len(getattr(self, "workers", None) or [])
            logger.debug(
                "[shutdown] Executor: start worker_count=%d",
                worker_count,
            )
            self.shutting_down = True

            # ------【进程管理】先关 death_writer 发退出信号，再强制终止所有 worker 进程 ------
            # Make sure all the worker processes are terminated first.
            if workers := getattr(self, "workers", None):
                for w in workers:
                    # Close death_writer to signal child processes to exit
                    if w.death_writer is not None:
                        w.death_writer.close()
                        w.death_writer = None
                self._ensure_worker_termination([w.proc for w in workers])

                # ------【异步 RPC】关闭每个 worker 的响应队列，释放共享内存与 ZMQ 资源 ------
                for w in workers:
                    # Shutdown response queues
                    if w.worker_response_mq is not None:
                        w.worker_response_mq.shutdown()
                        w.worker_response_mq = None

        # ------【异步 RPC】关闭广播与全部响应 MessageQueue，释放 IPC 资源 ------
        if rpc_broadcast_mq := getattr(self, "rpc_broadcast_mq", None):
            rpc_broadcast_mq.shutdown()
            self.rpc_broadcast_mq = None
        if response_mqs := getattr(self, "response_mqs", None):
            for mq in response_mqs:
                mq.shutdown()
            self.response_mqs = []

        logger.debug_once("[shutdown] Executor: complete")

    def check_health(self) -> None:
        # ------【异步 RPC + 进程管理】广播健康检查 RPC，超时未响应即触发故障处理 ------
        self.collective_rpc("check_health", timeout=10)
        return

    def _get_output_rank(self) -> int:
        # Only returns ModelRunnerOutput from TP rank=0 and PP rank=-1
        # (the first TP worker of the last PP stage).
        # Example:
        # Assuming TP=8, PP=4, then the world_size=32
        # 0-7, PP rank 0
        # 8-15, PP rank 1
        # 16-23, PP rank 2
        # 24-31, PP rank 3
        # so world_size - tp_size = 32 - 8 = 24 should be PP rank = -1 (i.e. 3)
        # ------【TP + PP】世界尺寸减去最后 TP 组大小 = 最后 PP stage 的 TP rank 0，即唯一回传结果的 rank ------
        return (
            self.world_size
            - self.parallel_config.tensor_parallel_size
            * self.parallel_config.prefill_context_parallel_size
        )

    @classmethod
    def supports_async_scheduling(cls) -> bool:
        # ------【异步 RPC】声明该执行器支持异步调度，引擎可下放 non_block 任务 ------
        return True


@dataclass
class UnreadyWorkerProcHandle:
    """WorkerProcess handle before READY."""

    proc: BaseProcess
    rank: int
    ready_pipe: Connection
    death_writer: Connection | None = None


@dataclass
class WorkerProcHandle:
    proc: BaseProcess
    rank: int
    # The worker process writes to this MQ in single-node mode
    worker_response_mq: MessageQueue | None
    # This is only non empty on driver node,
    # the peer worker process i writes to MQ
    # `peer_worker_response_mqs[i]`
    # 跨节点（多机）TP/PP 的产物
    # 它只在「一个模型副本太大、一台机放不下、被拆到多台物理机上」时才出现
    # driver worker 收集同模型副本其他 worker 结果
    peer_worker_response_mqs: list[MessageQueue | None] # 这个是主worker用来收集同伴的结果的
    death_writer: Connection | None = None

    @classmethod
    def from_unready_handle(
        cls,
        unready_handle: UnreadyWorkerProcHandle,
        worker_response_mq: MessageQueue | None,
        peer_worker_response_mqs: list[MessageQueue | None],
    ) -> "WorkerProcHandle":
        # ------【异步 RPC】把就绪前的句柄 + 握手拿到的响应 MQ 组装成最终 WorkerProcHandle ------
        return cls(
            proc=unready_handle.proc,
            rank=unready_handle.rank,
            worker_response_mq=worker_response_mq,
            peer_worker_response_mqs=peer_worker_response_mqs,
            death_writer=unready_handle.death_writer,
        )


class WorkerProc:
    """
    === 类说明 ===
        继承: object
        职责: 单个 Worker 进程的完整生命周期管理。在子进程中完成:
              ① 初始化分布式环境 (torch.distributed)

              ② 加载模型到 GPU
              ③ 建立与 Scheduler 的 IPC 通信 (MessageQueue)

              ④ 运行 worker_busy_loop() 主循环, 处理 RPC 调用

              ⑤ 优雅关闭

    === 架构: 一个 WorkerProc 对象 = 一个独立的 OS 进程 ===
        Scheduler 进程 (父)                  Worker 进程 (子)
        │                                     │
        │  make_worker_process()               │
        │  → proc.start() ─────spawn/fork────→ │
        │                                     │
        │                                     ├─ WorkerProc.__init__()
        │                                     │   ├─ WorkerWrapperBase.init_worker()
        │                                     │   │   └─ torch.distributed.init_process_group()
        │                                     │   ├─ worker.init_device()  ← GPU 初始化
        │                                     │   ├─ worker.load_model()   ← 加载模型权重
        │                                     │   └─ _init_message_queues() ← IPC 建立
        │                                     │
        │                                     ├─ ready_writer.send("READY")
        │                                     │   └─ 附带 response_mq 的 handle，注册回复队列
        │                                     │
        │  wait_for_ready() ← 收到 READY ─────│
        │                                     │
        │                                     └─ worker_busy_loop()
        │                                         while True:
        │                                           msg = rpc_broadcast_mq.dequeue() # 弹出任务
        │                                           result = worker.execute_model(msg)
        │                                           response_mq.enqueue(result)

    === 通信: 三条管道 ===
        rpc_broadcast_mq   — Scheduler → Worker: 接收广播 (execute_model, sample_tokens 等)
        worker_response_mq — Worker → Scheduler: 回传 ModelRunnerOutput
        death_pipe         — 父进程退出检测: 父进程死 → pipe EOF → Worker 自动终止
        ready_pipe         — Worker → 父进程: 发送 "READY" 就绪信号 + response_mq 的 handle

    === 核心方法 ===
        —— 进程创建 (静态) ——
            make_worker_process()  — @staticmethod: 创建子进程, 返回 UnreadyWorkerProcHandle
            wait_for_ready()       — @staticmethod: 阻塞等待所有 Worker 就绪
            worker_main()          — @staticmethod: 子进程入口函数 (target of Process)
        —— 实例方法 ——
            __init__()             — 在子进程中运行: init_device → load_model → init 消息队列
            enqueue_output()       — 将 Worker 输出打包为 (SUCCESS/FAILURE, result) 入队
            handle_output()        — 异步调度: 放入 async_output_queue; 同步调度: 直接 enqueue
            shutdown()             — 关闭消息队列 + Worker + 销毁分布式环境
            monitor_death_pipe()   — 启动后台线程, 监听父进程存活 (父死则自毁)
        —— 内部类 ——
            ResponseStatus         — Enum: SUCCESS / FAILURE

    === 核心成员属性 ===
        rank: int                  — 全局 rank (含 DP 偏移)
        worker: WorkerWrapperBase  — 包装了实际的 Worker 对象 (GpuWorker 等)
        rpc_broadcast_mq           — 接收 Scheduler 广播的消息队列
        worker_response_mq         — 向 Scheduler 回传结果的消息队列
        use_async_scheduling       — 是否启用异步调度
        async_output_queue         — 异步调度的输出队列 (线程间传递)
        peer_response_handles      — 多节点 DP 时, 远程 Worker 的 response handle 列表
    """

    def _init_message_queues(
        self, input_shm_handle: Handle, vllm_config: VllmConfig
    ) -> None:
        # ------【异步 RPC】按单机/多机 DP 选择建队列方式：单机直连共享内存，多机走分布式组远程广播 ------
        if vllm_config.parallel_config.nnodes_within_dp == 1:
            # Initialize MessageQueue for receiving SchedulerOutput
            self.rpc_broadcast_mq = MessageQueue.create_from_handle(
                input_shm_handle, self.worker.rank
            )

            # ------【异步 RPC】单机：新建 1 对 1 响应队列回传输出，无需 peer 句柄 ------
            # Initializes a message queue for sending the model output
            self.worker_response_mq = MessageQueue(1, 1)
            self.peer_response_handles = []
        else:
            # ------【异步 RPC + DP】多机：经分布式组创建跨节点广播/响应 MQ，driver 收集各 rank 句柄 ------
            # Initialize remote MessageQueue for receiving SchedulerOutput across nodes
            self.rpc_broadcast_mq = get_inner_dp_world_group().create_mq_broadcaster(
                external_writer_handle=input_shm_handle,
                # Since there is external_writer_handle from executor proc,
                # where the ready signal from actual writer is sent out of the
                # create_mq_broadcaster method and after this setup, we make it
                # non blocking. The handshake will be triggered when
                # worker.rpc_broadcast_mq.wait_until_ready() is called
                blocking=False,
            )
            # Initializes remote message queue for sending the model output to the
            # driver worker, exposing peer_response_handles for driver worker
            # that include handles for all ranks
            self.worker_response_mq, self.peer_response_handles = (
                get_inner_dp_world_group().create_single_reader_mq_broadcasters(
                    reader_rank_in_group=0
                )
            )

    @instrument(span_name="Worker init")
    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        input_shm_handle: Handle, # worker进程构建的时候，按照执行器传给他的连接信息，进行连接
        shared_worker_lock: LockType,
        is_driver_worker: bool,
    ):
        # ------【NCCL 通信】构造 WorkerWrapper 并 init_worker：各 rank 携带自己的初始化参数建立分布式进程组 ------
        self.rank = rank



        ##########################################################################################
        # 0. 构建调用转发的中间装饰器 WorkerWrapperBase
        ##########################################################################################
        wrapper = WorkerWrapperBase(rpc_rank=local_rank, global_rank=rank) # 构造一个装饰器的实例，用来选择合适的Worker，并解析我们的指令
        # TODO: move `init_worker` to executor level as a collective rpc call
        all_kwargs: list[dict] = [
            {} for _ in range(vllm_config.parallel_config.world_size)
        ]
        all_kwargs[local_rank] = {
            "vllm_config": vllm_config,
            "local_rank": local_rank,
            "rank": rank,
            "distributed_init_method": distributed_init_method,
            "is_driver_worker": is_driver_worker,
            "shared_worker_lock": shared_worker_lock,
        }

        
        ##########################################################################################
        # 1. 在这里利用wrapper装饰器，构建worker的实例
        ##########################################################################################
        wrapper.init_worker(all_kwargs) 
        self.worker = wrapper

        # ------【进程管理】设置进程名与日志前缀，带上 DP/TP/PP/EP 等并行 rank 便于排障 ------
        self.setup_proc_title_and_log_prefix(
            enable_ep=vllm_config.parallel_config.enable_expert_parallel
        )

        # ------【进程管理】初始化 GPU 设备上下文：为 ModelRunner 分配设备、创建 runner 实例 ------
        # Load model
        
        ##########################################################################################
        # 2. 驱动内部Worker开始init_device, 为GPUModelRunner准备环境， 然后构造model_runner实例
        ##########################################################################################
        self.worker.init_device()



        # Update process title now that parallel groups are initialized
        self.setup_proc_title_and_log_prefix(
            enable_ep=vllm_config.parallel_config.enable_expert_parallel
        )
        # ------【CUDA Graph】加载模型权重，内部会把前向计算捕获成 CUDA Graph 加速回放 ------
        if envs.VLLM_ELASTIC_EP_SCALE_UP_LAUNCH:
            self.worker.elastic_ep_execute("load_model")
        else:
            # 驱动Worker开始load_model(), 获得了self.model, 还在外面包了一层cuda graph


            ##########################################################################################
            # 3. 驱动Worker开始load_model(), 获得了self.model,完成模型实例构建 + 权重加载
            ##########################################################################################
            self.worker.load_model()


        # ------【异步 RPC】若开启异步调度，额外起一个输出拷贝线程，把结果异步写回响应队列 ------
        # 启动一个异步输出拷贝线程
        scheduler_config = vllm_config.scheduler_config
        self.use_async_scheduling = scheduler_config.async_scheduling

        '''
        主线程 worker_busy_loop          → 收指令 → 执行模型 → 得到 output
                                        → handle_output(output)
                                            ├─ 同步调度: 直接 enqueue_output() 发回 scheduler
                                            └─ 异步调度: 塞进 async_output_queue（立刻返回，继续下一步）
        '''
        if self.use_async_scheduling: # 如果使用异步调度
            self.async_output_queue: queue.Queue = queue.Queue()
            self.async_output_copy_thread = Thread(
                target=self.async_output_busy_loop,
                daemon=True,
                name="WorkerAsyncOutputCopy",
            )
            self.async_output_copy_thread.start()

        # ------【显存 profiling】按注意力后端更新 block size，供 KVCache 分配规划 ------
        # Set block size based on the attention backends
        current_platform.update_block_size_for_backend(vllm_config)

        # ------【异步 RPC】建立与 Scheduler 的广播/响应 MessageQueue（多机时用分布式组创建远程 MQ） ------
        # Initialize message queues after init_device() since multi-node setups
        # (nnodes_within_dp > 1) require distributed groups to be initialized
        self._init_message_queues(input_shm_handle, vllm_config)

        # ------【核心逻辑】启动后冻结环境变量读取，避免后续运行时重复解析 env 影响性能 ------
        # Enable environment variable cache (e.g. assume no more
        # environment variable overrides after this point)
        enable_envs_cache()
















    # 工厂函数，返回一个未启动的worker进程实例
    @staticmethod
    def make_worker_process(
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        input_shm_handle,  # Receive SchedulerOutput
        shared_worker_lock: LockType,
        is_driver_worker: bool,
        inherited_fds: list[int] | None = None,
    ) -> UnreadyWorkerProcHandle:
        # ------【进程管理】建两条单向管道：ready 通知父进程就绪，death 供子进程探测父进程退出 ------
        context = get_mp_context() # 获取启动的上下文环境

        # Ready pipe to communicate readiness from child to parent
        ready_reader, ready_writer = context.Pipe(duplex=False) # 管道1：ready
        # Death pipe to let child detect parent process exit
        death_reader, death_writer = context.Pipe(duplex=False)# 管道2：death

        # ------【进程管理】fork 模式下把本 worker 的两条管道 fd 并入继承清单，供后续 worker 关闭 ------
        if inherited_fds is not None:
            inherited_fds = inherited_fds.copy()
            inherited_fds.extend((ready_reader.fileno(), death_writer.fileno()))
        # ------【进程管理】组装子进程启动参数：配置 + 连接句柄 + 两条管道 + fork 下需关闭的继承 fd ------
        process_kwargs = {
            "vllm_config": vllm_config,
            "local_rank": local_rank,
            "rank": rank,
            "distributed_init_method": distributed_init_method,
            "input_shm_handle": input_shm_handle,
            "ready_pipe": ready_writer,
            "death_pipe": death_reader,
            "shared_worker_lock": shared_worker_lock,
            "is_driver_worker": is_driver_worker,
            # Have the worker close parent end of this worker's pipes too
            "inherited_fds": inherited_fds if inherited_fds is not None else [],
        }
        # ------【进程管理】构建并启动 Worker 子进程（入口 worker_main），可选 NUMA 绑核 ------
        # Run EngineCore busy loop in background process.
        proc = context.Process( # 构建一个进程实例，入口为worker_main
            target=WorkerProc.worker_main,
            kwargs=process_kwargs,
            name=f"VllmWorker-{rank}",
            daemon=True,
        )

        # Apply NUMA binding if configured
        with numa_utils.configure_subprocess(
            vllm_config, local_rank, process_kind="worker"
        ):
            proc.start()

        # ------【进程管理】父进程关闭子进程端，保留 death_writer 以便父退出时子进程收到 EOF ------
        # Close child ends of pipes here in the parent
        ready_writer.close()
        death_reader.close()
        # Keep death_writer open in parent - when parent exits,
        # death_reader in child will get EOFError
        return UnreadyWorkerProcHandle(proc, rank, ready_reader, death_writer)

    @staticmethod
    def wait_for_response_handle_ready(
        handles: dict[str, Any], proc_handle: UnreadyWorkerProcHandle
    ) -> WorkerProcHandle:
        # ------【异步 RPC】从握手消息里的 handle 重建响应 MQ：本地 reader 直接映射，peer 按远程订阅地址连接 ------
        response_handle = handles["handle"]
        worker_response_mq: MessageQueue | None = None
        if len(response_handle.local_reader_ranks) > 0:
            worker_response_mq = MessageQueue.create_from_handle(response_handle, 0)
        # ------【异步 RPC + DP】把各 peer worker 的响应 handle 连接成本地 MessageQueue 列表（driver 用） ------
        peer_response_handles = handles["peer_response_handles"]
        peer_worker_response_mqs = [
            MessageQueue.create_from_handle(handle, -1)
            if handle.remote_subscribe_addr is not None
            else None
            for handle in peer_response_handles
        ]
        return WorkerProcHandle.from_unready_handle(
            proc_handle,
            worker_response_mq,
            peer_worker_response_mqs=peer_worker_response_mqs,
        )

    @staticmethod
    def wait_for_ready(
        unready_proc_handles: list[UnreadyWorkerProcHandle],
    ) -> list[WorkerProcHandle]:
        e = Exception(
            "WorkerProc initialization failed due to an exception in a "
            "background process. See stack trace for root cause."
        )

        # ------【进程管理】把每个待就绪 worker 的 ready_pipe 映射成句柄，预置结果列表按 rank 填充 ------
        pipes = {handle.ready_pipe: handle for handle in unready_proc_handles}
        ready_proc_handles: list[WorkerProcHandle | None] = [None] * len(
            unready_proc_handles
        )
        # ------【进程管理】循环等待各 worker 就绪：recv 到 READY 后组装 WorkerProcHandle 并关管道 ------
        while pipes:
            ready = multiprocessing.connection.wait(pipes.keys())
            for pipe in ready:
                assert isinstance(pipe, Connection)
                try:
                    # Wait until the WorkerProc is ready.
                    unready_proc_handle = pipes.pop(pipe)
                    response: dict[str, Any] = pipe.recv()
                    if response["status"] != "READY":
                        raise e

                    idx = unready_proc_handle.rank % len(ready_proc_handles)
                    ready_proc_handles[idx] = WorkerProc.wait_for_response_handle_ready(
                        response, unready_proc_handle
                    )
                except EOFError:
                    e.__suppress_context__ = True
                    raise e from None

                finally:
                    # Close connection.
                    pipe.close()

        return cast(list[WorkerProcHandle], ready_proc_handles)

    def shutdown(self):
        # ------【异步 RPC】关闭广播/响应 MessageQueue 并释放 worker 资源 ------
        if self.rpc_broadcast_mq is not None:
            self.rpc_broadcast_mq.shutdown()
        if self.worker_response_mq is not None:
            self.worker_response_mq.shutdown()
        self.worker.shutdown()
        self.rpc_broadcast_mq = None
        self.worker_response_mq = None
        # ------【NCCL 通信】销毁模型并行组与 torch 分布式环境，结束进程组 ------
        destroy_model_parallel()
        destroy_distributed_environment()

    def monitor_death_pipe(self, death_pipe, shutdown_requested: threading.Event):
        if death_pipe is None:
            return

        # ------【进程管理】后台阻塞 recv 父进程 death 管道：父进程退出即 EOF，主动关闭本 worker 队列 ------
        def death_pipe_monitor(queues_to_shutdown: list[MessageQueue]):
            try:
                # This will block until parent process exits (pipe closes)
                death_pipe.recv()
            except EOFError:
                logger.info_once("Parent process exited, terminating worker queues")
                shutdown_requested.set()
                for mq in queues_to_shutdown:
                    if mq is not None:
                        mq.shutdown()
            except Exception as e:
                logger.warning("Death monitoring error: %s", e)

        # ------【进程管理】直接传队列引用避免 GC 问题，起守护线程执行监控 ------
        # Pass queue references directly to avoid gc issues if passing self
        Thread(
            target=death_pipe_monitor,
            args=([self.rpc_broadcast_mq, self.worker_response_mq],),
            daemon=True,
            name="DeathPipeMonitor",
        ).start()


    # 工厂函数，也是worker进程的启动的入口函数，立刻构建workerProc实例对象
    @staticmethod
    def worker_main(*args, **kwargs):
        """Worker initialization and execution loops.
        This runs a background process"""

        # ------【进程管理】注册 SIGTERM/SIGINT 处理：置位 shutdown_requested 并抛 SystemExit 优雅退出 ------
        # Signal handler used for graceful termination.
        # SystemExit exception is only raised once to allow this and worker
        # processes to terminate without error
        shutdown_requested = threading.Event()

        def signal_handler(signum, frame):
            nonlocal shutdown_requested
            if not shutdown_requested.is_set():
                shutdown_requested.set()
                logger.debug(
                    "WorkerProc handling signal %d, raising SystemExit", signum
                )
                raise SystemExit()

        # Either SIGTERM or SIGINT will terminate the worker
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

        # ------【进程管理】提前发布逻辑→物理 GPU 映射并设置网卡设备环境变量，供后续拓扑工具使用 ------
        # Publish the logical-to-physical mapping early so topology helpers
        # work before init_device (needed by set_worker_net_device below).
        assigned_physical_gpu_ids = kwargs[
            "vllm_config"
        ].parallel_config.assigned_physical_gpu_ids
        if assigned_physical_gpu_ids is not None:
            from vllm.platforms.interface import set_assigned_physical_gpu_ids

            set_assigned_physical_gpu_ids(assigned_physical_gpu_ids)

        # Set net device env vars for the worker if VLLM_GPU_NIC_PCIE_MAPPING is set
        set_worker_net_device(kwargs.get("local_rank", 0), kwargs["vllm_config"])

        # ------【进程管理】取出 ready/death 管道，并关闭 fork 继承来的其它 worker 管道避免 EOF 失效 ------
        worker = None
        ready_writer = kwargs.pop("ready_pipe")
        death_pipe = kwargs.pop("death_pipe", None)

        # Close inherited pipes from parent (incl. other worker pipes)
        # Explicitly passing in existing pipes and closing them makes the pipe
        # behave when using fork. Otherwise, a hidden reference to the pipes
        # exist in the child process and prevents EOF closure.
        for fd in kwargs.pop("inherited_fds", []):
            try:
                os.close(fd)
            except Exception as e:
                logger.warning("Error closing inherited connection: %s: %s", type(e), e)

        try:
            # ------【进程管理】初始化 worker 侧 tracing，为分布式观测打点 ------
            # Initialize tracer
            rank = kwargs.get("rank", 0)
            maybe_init_worker_tracer(
                instrumenting_module_name="vllm.worker",
                process_kind="worker",
                process_name=f"Worker_{rank}",
            )

            # ------【进程管理】构建 WorkerProc（含 init_device/load_model/建队列），随后监听父进程存活 ------
            worker = WorkerProc(*args, **kwargs) # 构建workproc实例对象
            assert worker.worker_response_mq is not None
            if kwargs["vllm_config"].parallel_config.numa_bind:
                numa_utils.log_current_affinity_state(f"Worker_{worker.rank}")

            worker.monitor_death_pipe(death_pipe, shutdown_requested)

            # ------【异步 RPC】就绪握手：回传 READY 与响应 MQ 句柄，再等待广播/响应队列真正可用 ------
            # Send READY once we know everything is loaded # 发送READY给executor，告知回复的MessageQueue的联系方式
            ready_writer.send(
                {
                    "status": WorkerProc.READY_STR,
                    "handle": worker.worker_response_mq.export_handle(),
                    "peer_response_handles": worker.peer_response_handles,
                }
            )

            # Ensure message queues are ready. Will deadlock if re-ordered.
            # Must be kept consistent with the Executor
            if worker.rpc_broadcast_mq is not None:
                worker.rpc_broadcast_mq.wait_until_ready()
            worker.worker_response_mq.wait_until_ready()
            ready_writer.close()
            ready_writer = None

            # ------【异步 RPC】进入主循环：持续 dequeue 广播指令、执行、回写结果 ------
            worker.worker_busy_loop() # 开始进入工作循环










        # ------【进程管理】异常兜底：worker 启动/运行失败时置位关闭标志，避免 ZMQ 析构时二次抛异常 ------
        except Exception:
            # NOTE: if an Exception arises in busy_loop, we send
            # a FAILURE message over the MQ RPC to notify the Executor,
            # which triggers system shutdown.
            # TODO(rob): handle case where the MQ itself breaks.

            if ready_writer is not None:
                logger.exception("WorkerProc failed to start.")
            elif shutdown_requested.is_set():
                logger.debug_once(
                    "[shutdown] WorkerProc: exiting after shutdown request"
                )
            else:
                logger.exception("WorkerProc failed.")

            # The parent sends a SIGTERM to all worker processes if
            # any worker dies. Set this value so we don't re-throw
            # SystemExit() to avoid zmq exceptions in __del__.
            shutdown_requested.set()

        except SystemExit as e:
            # ------【进程管理】SIGTERM/SIGINT 触发的退出必须重新抛出，不能被吞掉 ------
            # SystemExit is raised on SIGTERM or SIGKILL, which usually indicates that
            # the graceful shutdown process did not succeed
            if shutdown_requested.is_set():
                logger.debug_once(
                    "[shutdown] WorkerProc: terminated by shutdown signal"
                )
            else:
                logger.warning("WorkerProc was terminated")
            # SystemExit must never be ignored
            raise e

        finally:
            # ------【进程管理】兜底清理：关闭剩余管道并 shutdown worker，释放分布式/队列资源 ------
            if ready_writer is not None:
                ready_writer.close()
            if death_pipe is not None:
                death_pipe.close()
            # Clean up once worker exits busy loop
            if worker is not None:
                worker.shutdown()

    class ResponseStatus(Enum):
        SUCCESS = auto()
        FAILURE = auto()

    # 把output塞回我们的Worker到executor的返回缓冲去里面
    def enqueue_output(self, output: Any):
        """Prepares output from the worker and enqueues it to the
        worker_response_mq. If the output is an Exception, it is
        converted to a FAILURE response.
        """
        # ------【异步 RPC】异步输出先物化 get_output，异常则降级为失败原因 ------
        if isinstance(output, AsyncModelRunnerOutput):
            try:
                output = output.get_output()
            except Exception as e:
                logger.exception("Error getting async model runner output")
                output = e

        # ------【异步 RPC】把结果打包成 (SUCCESS/FAILURE, 内容) 状态对 ------
        if isinstance(output, Exception):
            result = (WorkerProc.ResponseStatus.FAILURE, str(output))
        else:
            result = (WorkerProc.ResponseStatus.SUCCESS, output) 
        # ------【异步 RPC】写入响应队列回传给 Scheduler ------
        if (response_mq := self.worker_response_mq) is not None:
            response_mq.enqueue(result) # 同步调度，直接往我们的Worker的回复队列里面塞就行了


    ###############
    # 所以这个handle_output是分发结果用的。
    ###############
    def handle_output(self, output: Any):
        """Handles output from the worker. If async scheduling is enabled,
        it is passed to the async_output_busy_loop thread. Otherwise, it is
        enqueued directly to the worker_response_mq.
        """
        # ------【异步 RPC】按调度模式分流输出：异步入队交给拷贝线程，同步直接写响应队列 ------
        if self.use_async_scheduling: # PP + step_with_batch_queue + v2, 我们是异步调度
            # 分发结果
            self.async_output_queue.put(output)
        else:# 我们使用同步调度
            self.enqueue_output(output)

    def async_output_busy_loop(self):
        """Entrypoint for the thread which handles outputs asynchronously."""

        # set device to the worker device for the thread.
        # a thread will not inherit the context of the main thread.
        # when calling any cuda runtime functions, it will implicitly
        # create a new cuda context on device 0, consuming extra memory.
        # here we set the device to the worker device for the thread,
        # enforcing the context to be the same as the main thread.
        from vllm.platforms import current_platform

        # ------【异步 RPC + CUDA Graph】子线程不继承主线程 CUDA context，需手动绑定同一设备避免额外显存 ------
        if hasattr(self.worker, "device"):
            current_platform.set_device(self.worker.device)

        # ------【异步 RPC】持续从输出队列取结果并写回响应队列，把 worker 主循环从序列化中解放出来 ------
        while True:
            output = self.async_output_queue.get()
            self.enqueue_output(output)

    # Worker进程的RPC服务循环
    def worker_busy_loop(self):
        """Main busy loop for Multiprocessing Workers"""
        assert self.rpc_broadcast_mq is not None

        while True:
            # ------【异步 RPC】阻塞 dequeue 广播指令：拿到要执行的 method 与目标结果 rank ------
            # 持续从广播队列里面接受指令
            #########################
            # 即便是在异步调度下，也是从指令广播队列里面，一次执行一个指令,这个队列里面是：

            # (forward A, sample A, forward B, sample B, ....., forward Z, sample Z)

            #########################
            method, args, kwargs, output_rank = self.rpc_broadcast_mq.dequeue(
                indefinite=True
            )
            # ------【异步 RPC】按 method 分发到 worker：字符串反射取方法 / 字节反序列化，执行后回写结果 ------
            try:
                if isinstance(method, str):
                    func = getattr(self.worker, method) # 根据method构造func方法
                elif isinstance(method, bytes):
                    func = partial(cloudpickle.loads(method), self.worker)

                #########################
                # 执行完的结果的返回是output，可能是前向，也可能是采样的结果
                #########################
                output = func(*args, **kwargs) # 执行方法

                # 如果当前的rank是尾rank
                if output_rank is None or self.rank == output_rank:
                    self.handle_output(output) # 处理func的结果



            # ------【异步 RPC】异常转失败响应：把不可序列化的异常转成字符串回传给 executor ------
            except Exception as e:
                # Notes have been introduced in python 3.11
                if hasattr(e, "add_note"):
                    e.add_note(traceback.format_exc())
                logger.exception("WorkerProc hit an exception.")
                # exception might not be serializable, so we convert it to
                # string, only for logging purpose.
                if output_rank is None or self.rank == output_rank:
                    self.handle_output(e)



    @staticmethod
    def setup_proc_title_and_log_prefix(enable_ep: bool) -> None:
        # Check if parallel groups are initialized first
        # ------【进程管理】并行组未初始化时用默认进程名，已初始化则拼接各并行维度 rank 便于排障 ------
        if not model_parallel_is_initialized():
            # Parallel groups not yet initialized, use default process name
            set_process_title(name="Worker")
            decorate_logs("Worker")
            return

        # ------【DP/TP/PP/EP】读取各并行组大小与 rank，用于拼出可读的进程名与日志前缀 ------
        dp_size = get_dp_group().world_size
        dp_rank = get_dp_group().rank_in_group
        pp_size = get_pp_group().world_size
        pp_rank = get_pp_group().rank_in_group
        pcp_size = get_pcp_group().world_size
        pcp_rank = get_pcp_group().rank_in_group
        tp_size = get_tp_group().world_size
        tp_rank = get_tp_group().rank_in_group
        dcp_size = get_dcp_group().world_size
        dcp_rank = get_dcp_group().rank_in_group
        process_name = "Worker"
        if dp_size > 1:
            process_name += f"_DP{dp_rank}"
        if pp_size > 1:
            process_name += f"_PP{pp_rank}"
        if pcp_size > 1:
            process_name += f"_PCP{pcp_rank}"
        if tp_size > 1:
            process_name += f"_TP{tp_rank}"
        if dcp_size > 1:
            process_name += f"_DCP{dcp_rank}"
        if enable_ep:
            ep_rank = get_ep_group().rank_in_group
            process_name += f"_EP{ep_rank}"
        set_process_title(name=process_name)
        decorate_logs(process_name)


# 这是worker的multiprocessing多进程池的启动环境，包括进程spawn, 每个进程的线程数
def set_multiprocessing_worker_envs(local_world_size: int = 1):
    """Set up environment variables that should be used when there are workers
    in a multiprocessing environment. This should be called by the parent
    process before worker processes are created"""

    # ------【进程管理】强制 spawn 启动避免 fork 死锁；CPU 后端或用户已设线程数则直接返回 ------
    _maybe_force_spawn()

    if current_platform.is_cpu() or "OMP_NUM_THREADS" in os.environ: # os.environ里面没有OMP_NUM_THREADS，这个不是vllm的自定义环境变量
        return

    # Choose the workers' thread count here, before they start, since a worker
    # must not set its own: `torch.set_num_threads()` spawns the thread pool
    # eagerly, and doing that part way through a worker's startup either races
    # the dlopen of shared objects or, in a forked worker, deadlocks (libgomp
    # is not fork-safe).
    # ------【进程管理】按 world_size 均分 CPU 核数写入 OMP_NUM_THREADS，隔离各 worker 的线程争抢 ------
    num_threads = startup_omp_num_threads(local_world_size) # 计算每个进程的合适的线程核心数
    os.environ["OMP_NUM_THREADS"] = str(num_threads) # 设置好这个vllm的环境变量
    os.environ[OMP_NUM_THREADS_SET_BY_VLLM] = "1" # 已经设置过的标志位

    # A spawned worker picks the count up from the environment when it imports
    # torch. A forked worker instead inherits it from this process, so set it
    # here too. This is safe as long as we don't *use* the pool before forking:
    # a forked child whose parent had run a parallel region deadlocks, whereas
    # one whose parent merely sized the pool does not.
    # ------【进程管理】父进程也同步设好线程数：spawn 子进程读环境变量，fork 子进程直接继承该值 ------
    torch.set_num_threads(num_threads) # 这里设置的是pytorch c++底层在worker进程内部 做cpu计算时开的工作线程，
    # 这样可以防止多个worker的内部线程 互相抢核心，相当于把多个worker的pytorch库的运算的核心给隔离开了
    logger.debug(
        "Set OMP_NUM_THREADS=%d for %d worker process(es).",
        num_threads,
        local_world_size,
    )
