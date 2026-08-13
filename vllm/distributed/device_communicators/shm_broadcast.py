# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import copyreg
import functools
import io
import os
import pickle
import shutil
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from multiprocessing import shared_memory
from pickle import PickleBuffer
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import patch

import torch
import torch.distributed as dist
import zmq
from torch.distributed import ProcessGroup
from zmq import (  # type: ignore
    IPV6,  # type: ignore
    PUB,
    SUB,
    SUBSCRIBE,
    XPUB,
    XPUB_VERBOSE,
    Context,
)

import vllm.envs as envs
from vllm.distributed.utils import StatelessProcessGroup, sched_yield
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.network_utils import (
    get_ip,
    get_open_port,
    get_open_zmq_inproc_path,
    get_open_zmq_ipc_path,
    is_valid_ipv6_address,
)

logger = init_logger(__name__)


SPINLOOP_EXT_ENABLED = False
if envs.VLLM_USE_SPINLOOP_EXT:
    try:
        from vllm.spinloop import spinloop

        SPINLOOP_EXT_ENABLED = True
    except ImportError:
        logger.warning(
            "spinloop extension could not be loaded, disabling VLLM_USE_SPINLOOP_EXT!"
        )
SPINLOOP_TIMEOUT_SECONDS = 0.1

if TYPE_CHECKING:
    from _typeshed import SizedBuffer

VLLM_RINGBUFFER_WARNING_INTERVAL = envs.VLLM_RINGBUFFER_WARNING_INTERVAL
# Cap on how long an idle reader parks before re-reading the authoritative SHM
# written-flag. Bounds lost-notify recovery latency to ~5s while the periodic
# wakeup stays negligible (one flag check per reader every 5s).
SHM_READER_RECHECK_INTERVAL_MS = 5000


from_bytes_big = functools.partial(int.from_bytes, byteorder="big")


# Memory fence for cross-process shared memory visibility.
# Required for correct producer-consumer synchronization when using
# shared memory without locks.
_memory_fence_lock = threading.Lock()


def memory_fence():
    """
    Full memory barrier for shared memory synchronization.

    Ensures all prior memory writes are visible to other processes before
    any subsequent reads. This is critical for lock-free producer-consumer
    patterns using shared memory.

    Implementation acquires and immediately releases a lock. Python's
    threading.Lock provides sequentially consistent memory barrier semantics
    across all major platforms (POSIX, Windows). This is a lightweight
    operation (~20ns) that guarantees:
    - All stores before the barrier are visible to other threads/processes
    - All loads after the barrier see the latest values
    """
    # Lock acquire/release provides full memory barrier semantics.
    # Using context manager ensures lock release even on exceptions.
    with _memory_fence_lock:
        pass


def to_bytes_big(value: int, size: int) -> bytes:
    return value.to_bytes(size, byteorder="big")


LONG_WAIT_TIME_LOG_MSG = (
    "No available shared memory broadcast block found "
    "in %d seconds. This typically happens "
    "when some processes are hanging or doing some "
    "time-consuming work (e.g. compilation, "
    "weight/kv cache quantization)."
)


class SpinCondition:
    """
    This class implements an interface similar to a threading.Condition. It
    allows a writer to notify readers to wake up and read from the shared memory
    buffer. This notification is done over a zmq socket.

    For optimal performance under load we don't want the readers to need to poll
    the zmq socket for every read. So the `wait` method here will return
    immediately when reads are frequent, and will only enter "idle mode" and
    await a notification on the zmq socket after a period of inactivity. This
    allows the readers to spin quickly, hence "SpinCondition".

    To support clean shutdown, a separate thread in the reader's process must be
    able to wake the reader so that it can exit. A separate cancel() method is
    implemented with an in-process socket to allow this interruption.
    """

    def __init__(
        self,
        is_reader: bool,
        context: zmq.Context,
        notify_address: str,
        busy_loop_s: float = 1,
    ):
        self.is_reader = is_reader

        if is_reader:
            # ------【异步 RPC】记录最近一次从共享内存读取的时间，用于判断该自旋还是进入空闲 ------
            # Time of last shm buffer read
            self.last_read = time.monotonic()

            # ------【异步 RPC】忙等窗口：这段时间内读端自旋而非阻塞，避免高吞吐下的唤醒开销 ------
            # Time to keep busy-looping on the shm buffer before going idle
            self.busy_loop_s = busy_loop_s

            # ------【ZMQ 通信】读端订阅写端的通知消息，用 SUB 接收“有新数据”的轻量唤醒 ------
            # Readers subscribe to write notifications
            self.local_notify_socket: zmq.Socket = context.socket(SUB)
            # Set zmq.CONFLATE to only keep the last message that the socket
            # receives. This prevents us from piling up notification messages
            # under high load when we aren't polling the socket.
            self.local_notify_socket.setsockopt(zmq.CONFLATE, 1)
            # Subscribe to all messages on the socket
            self.local_notify_socket.setsockopt_string(SUBSCRIBE, "")
            self.local_notify_socket.connect(notify_address)

            # ------【异步 RPC】进程内 PAIR 对：monitor 线程可通过它打断读端的阻塞等待 ------
            # Readers require a process-local socket to poll for cancellation
            cancel_path = get_open_zmq_inproc_path()
            self.write_cancel_socket: zmq.Socket = context.socket(zmq.PAIR)
            self.write_cancel_socket.bind(cancel_path)
            self.read_cancel_socket: zmq.Socket = context.socket(zmq.PAIR)
            self.read_cancel_socket.connect(cancel_path)

            # ------【异步 RPC】Poller 同时监听通知与取消两类事件，让阻塞等待可被任一唤醒 ------
            # Poller allows waiting on either `.notify()` or `.cancel()`
            self.poller = zmq.Poller()
            self.poller.register(self.read_cancel_socket, zmq.POLLIN)
            self.poller.register(self.local_notify_socket, zmq.POLLIN)
        else:
            # ------【ZMQ 通信】写端用 PUB 广播写通知，所有读端 SUB 都能收到 ------
            # Writer side publishes write notifications
            self.local_notify_socket: zmq.Socket = context.socket(PUB)  # type: ignore
            # ------【ZMQ 通信】高水位设为 1：忙时只保留最新一次通知，避免积压大量 ping ------
            # Set high water mark to 1 - we don't need to send a massive amount of
            # pings during busy operation. PUB sockets will silently drop subsequent
            # messages after the high water mark is reached.
            self.local_notify_socket.setsockopt(zmq.SNDHWM, 1)
            self.local_notify_socket.bind(notify_address)

            # ------【异步 RPC】写端用不到的字段置为占位值，保持两类对象的接口一致 ------
            self.last_read = 0
            self.busy_loop_s = 0
            self.read_cancel_socket = None
            self.write_cancel_socket = None
            self.poller = None

    def record_read(self):
        # ------【异步 RPC】读端每次消费后刷新读时间戳，重新开启一段忙等窗口 ------
        self.last_read = time.monotonic()

    def cancel(self):
        # ------【异步 RPC】发取消 ping 唤醒空闲读端，供同进程 monitor 线程用于干净关停 ------
        # Sends cancellation ping that will cause the reader to wake up.
        # This is done from a monitor thread in the same process as the reader.
        if self.is_reader:
            logger.debug("Canceling waiting reads on SHM Buffer")
            self.write_cancel_socket.send(b"\x00")

    def wait(self, timeout_ms: int | None = None) -> None:
        """Wait for data on the shared memory buffer.

        Yields the scheduler then returns immediately if it has been less than
        self.busy_loop_s since the last read.

        Otherwise, enters idle mode and awaits a socket ping for at most
        `timeout_ms` milliseconds, or indefinitely if timeout_ms is None.
        """
        assert self.is_reader, "Only readers can wait"

        # ------【异步 RPC】若距上次读取仍在忙等窗口内，只让出调度器立即返回，自旋不阻塞 ------
        current_time = time.monotonic()
        if current_time <= self.last_read + self.busy_loop_s:
            sched_yield()
        else:
            # ------【异步 RPC】空闲模式：阻塞在 Poller 上等通知或取消，避免空转烧 CPU ------
            events = dict(self.poller.poll(timeout=timeout_ms))

            # ------【异步 RPC】按事件类型分流：取消 / 数据通知 / 超时，各自走不同处理 ------
            if self.read_cancel_socket in events:
                logger.debug("Poller received cancel event")
            elif self.local_notify_socket in events:
                logger.debug("Poller received notify event")
                # Since zmq.CONFLATE is set, there will only be one notification
                # to read from the socket
                self.local_notify_socket.recv(flags=zmq.NOBLOCK, copy=False)
            else:
                logger.debug("Poller timed out")

    def notify(self):
        """Notifies all readers to wake up"""
        # ------【ZMQ 通信】写端广播一次通知，唤醒所有空闲读端来消费新数据 ------
        assert not self.is_reader, "Only writers can notify"
        self.local_notify_socket.send(b"\x00")


SHM_PATH = "/dev/shm"


def check_shm_free_space(required_bytes: int, shm_path: str = SHM_PATH) -> None:
    """Raise if ``shm_path`` cannot fit a ``required_bytes`` shared segment.

    Args:
        required_bytes: Size of the shared-memory segment to be created.
        shm_path: Mount point backing POSIX shared memory; skipped if absent.

    Raises:
        RuntimeError: If ``required_bytes`` exceeds the free space.
    """
    # ------【显存 profiling】/dev/shm 不存在（如非 Linux）时跳过校验，直接放行 ------
    if not os.path.isdir(shm_path):
        return
    # ------【显存 profiling】查询共享内存挂载点剩余空间，判断能否容纳所需段 ------
    free_bytes = shutil.disk_usage(shm_path).free
    if required_bytes <= free_bytes:
        return
    mib = 1 << 20
    raise RuntimeError(
        f"Insufficient space in {shm_path}: {required_bytes / mib:.0f} MiB "
        f"required, {free_bytes / mib:.0f} MiB free. Increase {shm_path} "
        "(e.g. --shm-size or --ipc=host)."
    )


class ShmRingBuffer:
    def __init__(
        self,
        n_reader: int,
        max_chunk_bytes: int,
        max_chunks: int,
        name: str | None = None,
    ):
        """
        A shared memory ring buffer implementation for broadcast communication.
        Essentially, it is a queue where only one will `enqueue` and multiple
        will `dequeue`. The max size of each item, together with the max number
        of items that can be stored in the buffer are known in advance.
        In this case, we don't need to synchronize the access to
         the buffer.

        Buffer memory layout:
                  data                                 metadata
                    |                                      |
                    | (current_idx)                        | (current_idx)
                    v                                      v
        +-------------------------------+----------------------------------------+
        | chunk0 | chunk1 | ... | chunk | metadata0 | metadata1 | ... | metadata |
        +-------------------------------+----------------------------------------+
        | max_chunks x max_chunk_bytes  | max_chunks x (1 + n_reader) bytes      |

        metadata memory layout: each byte is a flag, the first byte is the written
        flag, and the rest are reader flags. The flags are set to 0 by default.
        +--------------+--------------+--------------+-----+--------------+
        | written_flag | reader0_flag | reader1_flag | ... | readerN_flag |
        +--------------+--------------+--------------+-----+--------------+

        The state of metadata is as follows:

        (case 1) 0???...???: the block is not written yet, cannot read, can write
        (case 2) 1000...000: the block is just written, can read, cannot write
        (case 3) 1???...???: the block is written and read by some readers, can read if not read, cannot write
        (case 4) 1111...111: the block is written and read by all readers, cannot read, can write

        State transition for readers:

        When a reader finds a block that it can read (case 2 or 3), it can yield the block for caller to read.
        Only after the caller finishes reading the block, the reader can mark the block as read.
        Readers only mark the block as read (from 0 to 1), the writer marks the block as ready to read (from 1 to 0).

        State transition for writer:

        When the writer writes to a block (case 1 or 4), it first resets the written flag to 0, converting either case
        to case 1. Then it can yield the block for caller to write. After the caller finishes writing the block, the writer
        can reset the reader flags to 0, and mark the block as written (from 0 to 1).
        NOTE: the order is important here, first reset the reader flags (so that we are still in case 1), then mark the block as written. The state transition is atomic. If we do it in the reverse order, it will go through case 3 and then back to case 2, and readers might read the intermediate case 3, which is not correct.

        During creation, `name` is None and the buffer is created. We can pass the
        created object to other processes by pickling it. The other processes will
        get the name of the shared memory and open it, so that they can access the
        same shared memory buffer.
        """  # noqa
        # ------【异步 RPC】每个 chunk 元数据 = 1 个写标志 + n_reader 个读标志，据此算总字节与偏移 ------
        self.n_reader = n_reader
        self.metadata_size = 1 + n_reader
        self.max_chunk_bytes = max_chunk_bytes
        self.max_chunks = max_chunks
        self.total_bytes_of_buffer = (
            self.max_chunk_bytes + self.metadata_size
        ) * self.max_chunks
        # ------【异步 RPC】内存布局：前面是数据区，后面紧跟元数据区，二者各占一段连续偏移 ------
        self.data_offset = 0
        self.metadata_offset = self.max_chunk_bytes * self.max_chunks

        if name is None:
            # ------【进程管理】name 为空表示创建端：先校验剩余空间，再真正分配共享内存段 ------
            # we are creating a buffer
            self.is_creator = True
            check_shm_free_space(self.total_bytes_of_buffer)
            self.shared_memory = shared_memory.SharedMemory(
                create=True, size=self.total_bytes_of_buffer
            )
            assert self.shared_memory.buf is not None, "Buffer was not created"
            # ------【异步 RPC】元数据区整体清零，保证所有写/读标志初始为“未写/未读” ------
            # initialize the metadata section to 0
            with self.shared_memory.buf[self.metadata_offset :] as metadata_buffer:
                torch.frombuffer(metadata_buffer, dtype=torch.uint8).fill_(0)
        else:
            # ------【进程管理】name 非空表示附加端：通过 name 打开已有共享内存而非新建 ------
            # we are opening an existing buffer
            self.is_creator = False
            # ------【进程管理】打补丁禁用资源追踪，避免附加他人创建的共享内存被误释放 ------
            # fix to https://stackoverflow.com/q/62748654/9191338
            # Python incorrectly tracks shared memory even if it is not
            # created by the process. The following patch is a workaround.
            with patch(
                "multiprocessing.resource_tracker.register",
                lambda *args, **kwargs: None,
            ):
                try:
                    self.shared_memory = shared_memory.SharedMemory(name=name)
                    # See https://docs.python.org/3/library/multiprocessing.shared_memory.html # noqa
                    # Some platforms allocate memory based on page size,
                    # so the shared memory block size may be larger or equal
                    # to the requested size. The size parameter is ignored
                    # when attaching to an existing block.
                    assert self.shared_memory.size >= self.total_bytes_of_buffer
                except FileNotFoundError:
                    # ------【进程管理】跨节点反序列化时共享内存不存在，此对象不再使用，静默忽略 ------
                    # we might deserialize the object in a different node
                    # in this case, this object is not used,
                    # and we should suppress the error
                    pass

    def handle(self):
        # ------【进程管理】导出最小描述符，供 pickle 后在其他进程重建并附加同一共享内存 ------
        return (
            self.n_reader,
            self.max_chunk_bytes,
            self.max_chunks,
            self.shared_memory.name,
        )

    def __reduce__(self):
        # ------【进程管理】pickle 只传 handle 参数，反序列化端据此重新打开而非复制整个缓冲区 ------
        return (
            self.__class__,
            self.handle(),
        )

    def __del__(self):
        # ------【进程管理】析构时关闭映射；仅创建端负责 unlink，避免附加端误删共享段 ------
        if hasattr(self, "shared_memory"):
            self.shared_memory.close()
            if self.is_creator:
                self.shared_memory.unlink()

    @contextmanager
    def get_data(self, current_idx: int):
        # ------【异步 RPC】按 chunk 索引算数据区切片，直接映射共享内存字节给调用方读写 ------
        start = self.data_offset + current_idx * self.max_chunk_bytes
        end = start + self.max_chunk_bytes
        assert self.shared_memory.buf is not None, "Buffer has been closed"
        with self.shared_memory.buf[start:end] as buf:
            yield buf

    @contextmanager
    def get_metadata(self, current_idx: int):
        # ------【异步 RPC】按 chunk 索引算元数据区切片，暴露写/读标志位供无锁状态机判断 ------
        start = self.metadata_offset + current_idx * self.metadata_size
        end = start + self.metadata_size
        assert self.shared_memory.buf is not None, "Buffer has been closed"
        with self.shared_memory.buf[start:end] as buf:
            yield buf


def _rebuild_tensor(buf: Any, shape: tuple[int, ...], dtype_str: str) -> torch.Tensor:
    """Rebuild a tensor from an out-of-band pickle buffer.

    Counterpart of `_reduce_tensor`. Note that pickle passes the original
    buffer-providing object from `loads(buffers=...)` straight to this
    function (no `PickleBuffer` wrapper on the receiving side), so `buf` is
    a `zmq.Frame`, a `memoryview` of a shared-memory ring chunk, or `bytes`
    if the buffer was serialized in-band.
    """
    # ------【异步 RPC】由 dtype 字符串还原 torch.dtype，用于把字节视图重铸成张量 ------
    dtype = getattr(torch, dtype_str)
    assert isinstance(dtype, torch.dtype)
    # ------【异步 RPC】ZMQ 帧内存可独立存活，直接零拷贝别名，让张量强引用该帧保活 ------
    if isinstance(buf, zmq.Frame):
        # ZMQ frames own their message memory independently of any context,
        # so the tensor can safely alias it with zero copies. The tensor's
        # storage keeps the frame (and thus its bytes) alive via a strong
        # reference for as long as the tensor is.
        try:
            return torch.frombuffer(buf, dtype=torch.uint8).view(dtype).view(shape)
        except ValueError:
            # Empty or read-only frame buffer; fall through to the copy path.
            pass
    # ------【异步 RPC】环形缓冲 chunk 会被写端复用，必须拷贝出来；bytearray 保持结果张量可写 ------
    # Shared-memory ring buffer chunks are reused by the writer once all
    # readers have marked them read, so we must copy out of them. bytearray
    # (vs bytes) keeps the resulting tensor writable, matching normal tensor
    # semantics.
    raw = bytearray(buf)
    # ------【异步 RPC】空缓冲意味着含 0 维的占位张量，直接构造空张量返回 ------
    if not raw:
        assert 0 in shape
        return torch.empty(shape, dtype=dtype)
    return torch.frombuffer(raw, dtype=torch.uint8).view(dtype).view(shape)


def _reduce_tensor(tensor: torch.Tensor):
    """Reduce a CPU tensor to a `PickleBuffer` for out-of-band pickling.

    `torch.Tensor.__reduce_ex__` copies the tensor bytes into the pickle
    byte stream via `torch.serialization` and never emits a `PickleBuffer`,
    which defeats the out-of-band buffer handling in `MessageQueue.enqueue`.
    This reducer instead exposes the tensor's memory directly, so large
    tensors (e.g. `prompt_embeds` in `SchedulerOutput`) traverse the queue
    without being copied into and back out of the pickled message.
    """
    # ------【异步 RPC】零拷贝快速路径：CPU 连续张量暴露 uint8 视图给 PickleBuffer，免去 pickle 拷贝 ------
    if (
        tensor.device.type == "cpu"
        and tensor.layout == torch.strided
        and not tensor.requires_grad
    ):
        # ------【异步 RPC】尝试把张量转成 uint8 numpy 视图以暴露原始字节 ------
        try:
            # The uint8 view exposes the raw bytes via the buffer protocol,
            # including for dtypes numpy doesn't recognize (bfloat16, fp8, ...).
            # reshape(-1) first so that 0-dim tensors can be viewed as well.
            raw = tensor.contiguous().reshape(-1).view(torch.uint8).numpy()
        except RuntimeError:
            # Exotic tensors (e.g. with the conjugate bit set) that don't
            # support aliasing views; let torch handle them.
            pass
        else:
            # ------【异步 RPC】打包重建函数与 PickleBuffer，让 pickle 走带外缓冲而不拷贝进主串 ------
            dtype_str = str(tensor.dtype).removeprefix("torch.")
            return _rebuild_tensor, (PickleBuffer(raw), tuple(tensor.shape), dtype_str)

    # ------【异步 RPC】兜底路径：交给 torch 默认的拷贝式序列化 ------
    # Fall back to torch's default (copying) reduction.
    return tensor.__reduce_ex__(pickle.HIGHEST_PROTOCOL)

@dataclass
class Handle:
    local_reader_ranks: list[int] = field(default_factory=list)  # 哪些 rank 走本地共享内存读取（同机 Worker）
    buffer_handle: tuple[int, int, int, str] | None = None       # 共享内存描述符 (shm_fd, size, chunk_bytes, name)，子进程 mmap 映射
    local_subscribe_addr: str | None = None                      # 本机 ZMQ SUB 地址，子进程连接接收"新数据到了"通知
    local_notify_addr: str | None = None                         # 本机 SpinCondition 通知地址，轻量跨进程唤醒
    remote_subscribe_addr: str | None = None                     # 跨节点远程 DP 的 ZMQ SUB 订阅地址
    remote_addr_ipv6: bool = False                               # 远程地址是否 IPv6


class MessageQueue:
    def __init__(
        self,
        n_reader,  # number of all readers
        n_local_reader,  # number of local readers through shared memory
        local_reader_ranks: list[int] | None = None,
        # Default of 24MiB chosen to be large enough to accommodate grammar
        # bitmask tensors for large batches (1024 requests).
        max_chunk_bytes: int = 1024 * 1024 * 24,
        max_chunks: int = 10,
        connect_ip: str | None = None,
    ):
        # ------【进程管理】默认本机读者 rank 为前 n_local_reader 个；否则校验数量一致 ------
        if local_reader_ranks is None:
            local_reader_ranks = list(range(n_local_reader))
        else:
            assert len(local_reader_ranks) == n_local_reader
        # ------【DP】读者分为本机(共享内存)与远程(网络)两类，据此初始化不同通道 ------
        self.n_local_reader = n_local_reader
        n_remote_reader = n_reader - n_local_reader
        self.n_remote_reader = n_remote_reader
        self.shutting_down = False
        context = Context()

        if n_local_reader > 0:
            # ------【DP】本机读者走共享内存环(小数据) + XPUB 套接字(大数据溢出)，两条通道分工 ------
            # for local readers, we will:
            # 1. create a shared memory ring buffer to communicate small data
            # 2. create a publish-subscribe socket to communicate large data
            self.buffer = ShmRingBuffer(n_local_reader, max_chunk_bytes, max_chunks)

            # ------【ZMQ 通信】用 XPUB 而非 PUB：可收到订阅消息，从而在握手时确认读者数量 ------
            # XPUB is very similar to PUB,
            # except that it can receive subscription messages
            # to confirm the number of subscribers
            self.local_socket = context.socket(XPUB)
            # set the verbose option so that we can receive every subscription
            # message. otherwise, we will only receive the first subscription
            # see http://api.zeromq.org/3-3:zmq-setsockopt for more details
            # ------【ZMQ 通信】XPUB_VERBOSE 让每次订阅变更都上报，便于逐个确认读者已连接 ------
            self.local_socket.setsockopt(XPUB_VERBOSE, True)
            local_subscribe_addr = get_open_zmq_ipc_path()
            logger.debug("Binding to %s", local_subscribe_addr)
            self.local_socket.bind(local_subscribe_addr)

            # ------【异步 RPC】写端 current_idx 指向下一个待写入的 chunk 槽位 ------
            self.current_idx = 0

            # ------【异步 RPC】创建写端 SpinCondition，负责在写完数据后通知本机读者唤醒 ------
            # Create the notification side of the SpinCondition
            local_notify_addr = get_open_zmq_ipc_path()
            self._spin_condition = SpinCondition(
                is_reader=False, context=context, notify_address=local_notify_addr
            )
        else:
            # ------【DP】无本机读者时本地通道全部置空，仅保留远程通道 ------
            self.buffer = None  # type: ignore
            local_subscribe_addr = None
            self.local_socket = None
            self.current_idx = -1
            local_notify_addr = None
            self._spin_condition = None  # type: ignore

        # ------【DP】远程读者通道走 TCP+XPUB：跨节点广播，需处理 IPv6 地址格式与端口分配 ------
        remote_addr_ipv6 = False
        if n_remote_reader > 0:
            # for remote readers, we will:
            # create a publish-subscribe socket to communicate large data
            if not connect_ip:
                connect_ip = get_ip()
            self.remote_socket = context.socket(XPUB)
            self.remote_socket.setsockopt(XPUB_VERBOSE, True)
            remote_subscribe_port = get_open_port()
            # ------【ZMQ 通信】IPv6 地址需开启 IPV6 选项并加方括号，否则 ZMQ 解析失败 ------
            if is_valid_ipv6_address(connect_ip):
                self.remote_socket.setsockopt(IPV6, 1)
                remote_addr_ipv6 = True
                connect_ip = f"[{connect_ip}]"
            socket_addr = f"tcp://{connect_ip}:{remote_subscribe_port}"
            self.remote_socket.bind(socket_addr)
            remote_subscribe_addr = f"tcp://{connect_ip}:{remote_subscribe_port}"
        else:
            remote_subscribe_addr = None
            self.remote_socket = None

        # ------【异步 RPC】新建的队列默认就是写端，读者端由 create_from_handle 另设 ------
        self._is_writer = True
        self._is_local_reader = False
        self.local_reader_rank = -1
        # rank does not matter for remote readers
        self._is_remote_reader = False

        # ------【进程管理】把各通道地址/描述符打包成 Handle，供广播给所有读者进程 ------
        self.handle = Handle(
            local_reader_ranks=local_reader_ranks,
            buffer_handle=self.buffer.handle() if self.buffer is not None else None,
            local_subscribe_addr=local_subscribe_addr,
            local_notify_addr=local_notify_addr,
            remote_subscribe_addr=remote_subscribe_addr,
            remote_addr_ipv6=remote_addr_ipv6,
        )

        logger.debug("vLLM message queue communication handle: %s", self.handle)

    def export_handle(self) -> Handle:
        # ------【进程管理】返回 Handle 供写端广播，读者据此附加共享内存并连接套接字 ------
        return self.handle

    @staticmethod
    def create_from_handle(handle: Handle, rank) -> "MessageQueue":
        # ------【进程管理】绕过 __init__ 直接分配对象，再由 Handle 填充通道与角色信息 ------
        self = MessageQueue.__new__(MessageQueue)
        self.handle = handle
        self._is_writer = False

        context = Context()

        # ------【DP】rank 在本机读者名单内则走共享内存 + 本机 SUB，否则走远程 SUB ------
        if rank in handle.local_reader_ranks:
            # ------【进程管理】用 Handle 里的描述符附加到同一共享内存，而非新建 ------
            assert handle.buffer_handle is not None
            self.buffer = ShmRingBuffer(*handle.buffer_handle)
            self.current_idx = 0
            self.local_reader_rank = handle.local_reader_ranks.index(rank)
            self._is_local_reader = True
            self._is_remote_reader = False

            # ------【ZMQ 通信】本机读者以 SUB 连接写端 XPUB，订阅全部消息接收广播数据 ------
            self.local_socket = context.socket(SUB)
            self.local_socket.setsockopt_string(SUBSCRIBE, "")
            socket_addr = handle.local_subscribe_addr
            logger.debug("Connecting to %s", socket_addr)
            self.local_socket.connect(socket_addr)

            # ------【异步 RPC】创建读端 SpinCondition，等待写端的通知唤醒 ------
            self.remote_socket = None
            assert isinstance(handle.local_notify_addr, str)
            self._spin_condition = SpinCondition(
                is_reader=True, context=context, notify_address=handle.local_notify_addr
            )
        else:
            # ------【DP】远程读者无共享内存，全部走 TCP SUB 接收广播 ------
            self.buffer = None  # type: ignore
            self.current_idx = -1
            self.local_reader_rank = -1
            self._is_local_reader = False
            self._is_remote_reader = True

            self.local_socket = None

            # ------【ZMQ 通信】按 Handle 里的地址(可能 IPv6)连接远程写端，订阅广播消息 ------
            self.remote_socket = context.socket(SUB)
            self.remote_socket.setsockopt_string(SUBSCRIBE, "")
            if handle.remote_addr_ipv6:
                self.remote_socket.setsockopt(IPV6, 1)
            socket_addr = handle.remote_subscribe_addr
            logger.debug("Connecting to %s", socket_addr)
            self.remote_socket.connect(socket_addr)
            self._spin_condition = None  # type: ignore

        # ------【进程管理】重置关停标志，返回已按 Handle 配置好的队列对象 ------
        self.shutting_down = False
        return self

    def wait_until_ready(self):
        """This is a collective operation. All processes (including the
        readers and the writer) should call this function.
        """
        # ------【ZMQ 通信】写端等待所有读者订阅到位，实现 PUB/SUB 的连接握手屏障 ------
        if self._is_writer:
            # wait for all readers to connect

            # local readers
            # ------【ZMQ 通信】逐个 recv 消费 XPUB 上报的订阅消息，确认每个本机读者已连接 ------
            for i in range(self.n_local_reader):
                # wait for subscription messages from all local readers
                self.local_socket.recv()
            if self.n_local_reader > 0:
                # send a message to all local readers
                # to make sure the publish channel is working
                self.local_socket.send(b"READY")

            # remote readers
            # ------【ZMQ 通信】同样等待所有远程读者订阅，确保发布通道可用 ------
            for i in range(self.n_remote_reader):
                # wait for subscription messages from all remote readers
                self.remote_socket.recv()
            if self.n_remote_reader > 0:
                # send a message to all remote readers
                # to make sure the publish channel is working
                self.remote_socket.send(b"READY")
        elif self._is_local_reader:
            # wait for the writer to send a message
            # ------【ZMQ 通信】读者阻塞等待写端的 READY 包，确认订阅已建立 ------
            recv = self.local_socket.recv()
            assert recv == b"READY"
        elif self._is_remote_reader:
            # wait for the writer to send a message
            recv = self.remote_socket.recv()
            assert recv == b"READY"

    def shutdown(self):
        """If this is an idle reader, wakes it up so it can clean up and shut
        down"""
        # ------【进程管理】置关停标志并取消等待，唤醒空闲读端使其退出阻塞循环 ------
        self.shutting_down = True
        if self._spin_condition is not None:
            self._spin_condition.cancel()

    @contextmanager
    def acquire_write(self, timeout: float | None = None):
        assert self._is_writer, "Only writers can acquire write"
        # ------【异步 RPC】记录起始时间与告警计数，用于超时与长时间等待的日志控制 ------
        start_time = time.monotonic()
        n_warning = 1
        while True:
            # ------【异步 RPC】映射当前槽位的元数据，进入无锁自旋判定循环 ------
            with self.buffer.get_metadata(self.current_idx) as metadata_buffer:

                # ------【异步 RPC】校验本槽是否可写：已写且仍有读者未读则不可写 ------
                def check():
                    memory_fence()
                    read_count = sum(metadata_buffer[1:])
                    written_flag = metadata_buffer[0]
                    return not (written_flag and read_count != self.buffer.n_reader)

                # ------【异步 RPC】可选 spinloop 扩展：CPU 指令级自旋替代 Python 轮询降延迟 ------
                if SPINLOOP_EXT_ENABLED and not check():
                    spinloop(metadata_buffer, check, timeout=SPINLOOP_TIMEOUT_SECONDS)

                if not check():
                    # ------【异步 RPC】槽位被占(已写且未读尽)：让出 CPU 避免忙等烧核 ------
                    # this block is written and not read by all readers
                    # for writers, `self.current_idx` is the next block to write
                    # if this block is not ready to write,
                    # we need to wait until it is read by all readers

                    # Release the processor to other threads
                    sched_yield()

                    # if we time out, raise an exception
                    # ------【异步 RPC】超过超时阈值则抛异常，防止写端无限等待挂死的读者 ------
                    elapsed = time.monotonic() - start_time
                    if timeout is not None and elapsed > timeout:
                        raise TimeoutError

                    # if we wait for a long time, log a message
                    # ------【异步 RPC】等待过久按间隔打日志，提示可能有人挂起或做重活 ------
                    if elapsed > VLLM_RINGBUFFER_WARNING_INTERVAL * n_warning:
                        logger.info(
                            LONG_WAIT_TIME_LOG_MSG, VLLM_RINGBUFFER_WARNING_INTERVAL
                        )
                        n_warning += 1

                    continue
                # ------【异步 RPC】找到可写槽：先清写标志防止读者误读旧数据，再交数据区给调用方 ------
                # found a block that is either
                # (1) not written
                # (2) read by all readers

                # mark the block as not written
                metadata_buffer[0] = 0
                # let caller write to the buffer
                with self.buffer.get_data(self.current_idx) as buf:
                    yield buf

                # caller has written to the buffer
                # NOTE: order is important here
                # first set the read flags to 0
                # then set the written flag to 1
                # otherwise, the readers may think they already read the block
                # ------【异步 RPC】写完后先复位所有读标志再置写标志，顺序不能反，否则读者会读到中间态 ------
                for i in range(1, self.buffer.n_reader + 1):
                    # set read flag to 0, meaning it is not read yet
                    metadata_buffer[i] = 0
                # ------【异步 RPC】内存屏障保证数据先于写标志可见，避免弱内存序下读者读到半成品 ------
                # Memory fence here ensures the order of the buffer and flag
                # writes. This guarantees that when `metadata_buffer[0] = 1` is
                # visible to readers, `buf` can be completely ready. Without
                # this, some CPU architectures with weak ordering may incur
                # memory inconsistency.
                memory_fence()
                # mark the block as written
                metadata_buffer[0] = 1
                # Memory fence ensures the write is visible to readers on other cores
                # before we proceed. Without this, readers may spin indefinitely
                # waiting for a write that's stuck in our CPU's store buffer.
                memory_fence()
                # ------【异步 RPC】环形推进写指针到下一槽位，回绕实现循环复用 ------
                self.current_idx = (self.current_idx + 1) % self.buffer.max_chunks
                break

    class ReadTimeoutWithWarnings:
        def __init__(self, timeout: float | None, should_warn: bool) -> None:
            # ------【异步 RPC】记录开始时刻并算截止时间，无超时则用整型上限代替 ------
            self.started = time.monotonic()
            self.deadline = sys.maxsize if timeout is None else self.started + timeout

            # if should_warn, we need to wake up periodically to log
            # ------【异步 RPC】需要告警时算出周期性唤醒间隔，供 Poller 超时分片使用 ------
            self.warning_wait_time_ms: int | None = (
                VLLM_RINGBUFFER_WARNING_INTERVAL * 1000 if should_warn else None
            )

            self._should_warn = should_warn
            self.n_warning = 1
            self.timeout = timeout

        def timeout_ms(self) -> int:
            """Returns a timeout, capped at the recheck interval, that is:
            - min(time to deadline, time to next warning) if we're logging warnings
            - time to deadline, if we're not logging warnings
            - recheck interval if the timeout is None and we're not logging warnings
            - raise TimeoutError if we are past the deadline
            """
            # ------【异步 RPC】先把 Poller 超时上限压到周期重查间隔，保证周期性能醒着读写标志 ------
            wait_ms = SHM_READER_RECHECK_INTERVAL_MS
            if self.warning_wait_time_ms is not None:
                wait_ms = min(wait_ms, self.warning_wait_time_ms)
            if self.timeout is None:
                return wait_ms
            # ------【异步 RPC】有超时则算剩余时间，归零即抛 TimeoutError，否则取更小者 ------
            time_left_ms = int((self.deadline - time.monotonic()) * 1000)
            if time_left_ms <= 0:
                raise TimeoutError
            return min(wait_ms, time_left_ms)

        def should_warn(self) -> bool:
            """Returns true if it's time to log a warning for a timeout that is not
            indefinite"""
            # ------【异步 RPC】按固定间隔判断是否到告警时刻，命中则递增计数并触发日志 ------
            if self._should_warn:
                elapsed = time.monotonic() - self.started
                if elapsed >= VLLM_RINGBUFFER_WARNING_INTERVAL * self.n_warning:
                    self.n_warning += 1
                    return True
            return False

    @contextmanager
    def acquire_read(
        self,
        timeout: float | None = None,
        indefinite: bool = False,
    ):
        assert self._is_local_reader, "Only readers can acquire read"
        # ------【异步 RPC】构造带告警与重查分片的读超时控制器 ------
        read_timeout = self.ReadTimeoutWithWarnings(
            timeout=timeout, should_warn=not indefinite
        )
        with self.buffer.get_metadata(self.current_idx) as metadata_buffer:
            while True:

                # ------【异步 RPC】校验本槽是否可读：已写且本读者尚未读过 ------
                def check():
                    memory_fence()
                    read_flag = metadata_buffer[self.local_reader_rank + 1]
                    written_flag = metadata_buffer[0]
                    return not (not written_flag or read_flag)

                # ------【异步 RPC】可选 spinloop 扩展：指令级自旋等待写标志翻转为 1 ------
                if SPINLOOP_EXT_ENABLED and not check():
                    spinloop(
                        metadata_buffer[0 : self.local_reader_rank + 1],
                        check,
                        timeout=SPINLOOP_TIMEOUT_SECONDS,
                    )

                if not check():
                    # ------【异步 RPC】槽未就绪(未写或已读)：阻塞在 SpinCondition 上等写端通知 ------
                    # this block is either
                    # (1) not written
                    # (2) already read by this reader

                    # for readers, `self.current_idx` is the next block to read
                    # if this block is not ready,
                    # we need to wait until it is written
                    self._spin_condition.wait(timeout_ms=read_timeout.timeout_ms())

                    # ------【进程管理】关停中被唤醒则抛异常退出读循环，实现干净停机 ------
                    if self.shutting_down:
                        raise RuntimeError("cancelled")

                    # if we wait for a long time, log a message
                    # ------【异步 RPC】等待过久按间隔打日志，暴露可能的写端挂起 ------
                    if read_timeout.should_warn():
                        logger.info(
                            LONG_WAIT_TIME_LOG_MSG, VLLM_RINGBUFFER_WARNING_INTERVAL
                        )

                    continue
                # ------【异步 RPC】找到可读槽：把数据区交调用方，读毕再回写自己的读标志 ------
                # found a block that is not read by this reader
                # let caller read from the buffer
                with self.buffer.get_data(self.current_idx) as buf:
                    try:
                        yield buf
                    finally:
                        # caller has read from the buffer; set the read flag.
                        # ------【异步 RPC】读完置本读者读标志为 1，让写端知道该读者已完成 ------
                        metadata_buffer[self.local_reader_rank + 1] = 1
                        # Memory fence ensures the read flag is visible to the writer.
                        # Without this, writer may not see our read completion and
                        # could wait indefinitely for all readers to finish.
                        memory_fence()
                        # ------【异步 RPC】推进读指针并记录读时间，重新开启忙等窗口 ------
                        next_idx = self.current_idx + 1
                        self.current_idx = next_idx % self.buffer.max_chunks
                        self._spin_condition.record_read()
                break

    def enqueue(self, obj, timeout: float | None = None):
        """Write to message queue with optional timeout (in seconds)"""
        assert self._is_writer, "Only writers can enqueue"
        # ------【异步 RPC】预留主 pickle 串槽位，并预置 6 字节头(缓冲计数 2 + 主串长度 4) ------
        all_buffers: list[SizedBuffer] = [b""]
        total_bytes = 6  # 2 bytes for oob buffer count, 4 for main buffer size

        def oob_callback(buf: PickleBuffer) -> bool:
            # ------【异步 RPC】大于 1MiB 的缓冲走带外通道，小缓冲内联进主串省去额外开销 ------
            raw_buf = buf.raw()
            if len(raw_buf) < 1024 * 1024:
                # In-line buffers smaller than 1MiB.
                return True
            all_buffers.append(raw_buf)
            nonlocal total_bytes
            total_bytes += len(raw_buf) + 4
            return False

        # ------【异步 RPC】注册 torch.Tensor 的自定义 reducer，让张量字节走带外缓冲而非拷进主串 ------
        # CPU tensors are routed through `_reduce_tensor` so that their
        # bytes are emitted as out-of-band buffers instead of being
        # copied into the pickle stream by torch's default reducer.
        # Start from `copyreg.dispatch_table` to preserve globally
        # registered reducers (e.g. `re.Pattern`); the per-pickler
        # dispatch table would otherwise shadow them.
        dispatch_table = dict(copyreg.dispatch_table)
        dispatch_table[torch.Tensor] = _reduce_tensor
        # ------【异步 RPC】用带外回调把对象 pickle 进内存流，大缓冲被抽到 all_buffers ------
        with io.BytesIO() as bio:
            pickler = pickle.Pickler(
                bio,
                protocol=pickle.HIGHEST_PROTOCOL,
                buffer_callback=oob_callback,
            )
            pickler.dispatch_table = dispatch_table
            pickler.dump(obj)
            all_buffers[0] = bio.getvalue()
        if self.n_local_reader > 0:
            # ------【异步 RPC】数据放不下一个 chunk 时走溢出：标记后改由套接字直接发多段消息 ------
            if total_bytes + len(all_buffers[0]) >= self.buffer.max_chunk_bytes:
                with self.acquire_write(timeout) as buf:
                    buf[0] = 1  # overflow
                self.local_socket.send_multipart(all_buffers, copy=False)
            else:
                # ------【异步 RPC】能放下则写入环形槽：按 2 字节计数 + 每段 4 字节长度 布局封包 ------
                # Byte 0: 0
                # Bytes 1-2: Count of buffers
                # Then each buffer follows, preceded by 4 bytes containing its length:
                # [4 byte int L][L bytes of buffer content] ...
                with self.acquire_write(timeout) as buf:
                    buf[0] = 0  # not overflow
                    offset = 3
                    buf[1:offset] = to_bytes_big(len(all_buffers), 2)  # oob buf count
                    for buffer in all_buffers:
                        buf_len = len(buffer)
                        # prepend each buffer with 4 bytes containing its size.
                        buf_offset = offset + 4
                        buf[offset:buf_offset] = to_bytes_big(buf_len, 4)
                        buf[buf_offset : (offset := buf_offset + buf_len)] = buffer

            # ------【异步 RPC】写完后广播通知，唤醒本机空闲读者消费新数据 ------
            self._spin_condition.notify()

        if self.n_remote_reader > 0:
            # ------【DP】远程读者直接经 TCP 套接字发多段消息(零拷贝)，绕过共享内存 ------
            self.remote_socket.send_multipart(all_buffers, copy=False)

    def dequeue(
        self,
        timeout: float | None = None,
        indefinite: bool = False,
    ):
        """Read from message queue with optional timeout (in seconds)"""
        # ------【DP】本机读者优先从共享内存槽读取，溢出时才回落到套接字 ------
        if self._is_local_reader:
            with self.acquire_read(timeout, indefinite) as buf:
                overflow = buf[0] == 1
                if not overflow:
                    # ------【异步 RPC】按头部格式解包：读计数，再按每段 4 字节长度还原各缓冲 ------
                    offset = 3
                    buf_count = from_bytes_big(buf[1:offset])
                    all_buffers = []
                    for i in range(buf_count):
                        buf_offset = offset + 4
                        buf_len = from_bytes_big(buf[offset:buf_offset])
                        offset = buf_offset + buf_len
                        all_buffers.append(buf[buf_offset:offset])
                    # ------【异步 RPC】用带外缓冲直接重建对象，避免把张量字节再拷一遍 ------
                    obj = pickle.loads(all_buffers[0], buffers=all_buffers[1:])
            if overflow:
                # ------【异步 RPC】溢出数据走本地套接字多段接收 ------
                obj = MessageQueue.recv(self.local_socket, timeout)
        elif self._is_remote_reader:
            # ------【DP】远程读者直接从套接字接收多段消息 ------
            obj = MessageQueue.recv(self.remote_socket, timeout)
        else:
            raise RuntimeError("Only readers can dequeue")
        return obj

    @staticmethod
    def recv(socket: zmq.Socket, timeout: float | None) -> Any:
        # ------【ZMQ 通信】把秒级超时转成非负毫秒给 ZMQ poll，超时即抛异常 ------
        # Ensure non-negative timeout passed to zmq poll.
        timeout_ms = None if timeout is None else max(0, int(timeout * 1000))
        if not socket.poll(timeout=timeout_ms):
            raise TimeoutError
        # ------【异步 RPC】零拷贝接收多段消息，首段为主串其余为带外缓冲，直接重建对象 ------
        recv, *recv_oob = socket.recv_multipart(copy=False)
        return pickle.loads(recv, buffers=recv_oob)

    def broadcast_object(self, obj=None):
        # ------【核心逻辑】写端入队广播，读端出队接收，统一成对称的广播接口 ------
        if self._is_writer:
            self.enqueue(obj)
            return obj
        return self.dequeue()

    @staticmethod
    def create_from_process_group_single_reader(
        pg: ProcessGroup,
        max_chunk_bytes,
        max_chunks,
        reader_rank: int = 0,
        blocking: bool = False,
    ) -> tuple["MessageQueue", list[Handle]]:
        """
        Creates a MessageQueue for a process group with a single reader.

        This method is designed for scenarios where only one process (the reader)
        will consume messages, and all other processes are writers. It sets up
        the shared memory buffer and communication handles accordingly, and
        gathers the handles from all processes to the reader.

        Args:
            pg (ProcessGroup): The torch distributed process group.
            max_chunk_bytes (int): Maximum size in bytes for each chunk in the buffer.
            max_chunks (int): Maximum number of chunks in the buffer.
            reader_rank (int, optional): The global rank that will act as the reader.
                Defaults to 0.
            blocking (bool, optional): If True, blocks until all processes are ready.
                Defaults to False.

        Returns:
            tuple[MessageQueue, list[Handle]]:
            The MessageQueue instance for the calling process,
            and a list of handles (only non-empty for the reader process).
        """
        from vllm.platforms.interface import get_assigned_physical_gpu_ids

        # ------【进程管理】用分配的物理 GPU 数推断每节点进程数，据此判断与读者是否同机 ------
        assigned_physical_gpu_ids = get_assigned_physical_gpu_ids()
        if assigned_physical_gpu_ids is not None:
            local_size = len(assigned_physical_gpu_ids)
        else:
            local_size = current_platform.device_count()
        rank = dist.get_rank()
        same_node = rank // local_size == reader_rank // local_size
        # ------【异步 RPC】单读者场景：同机则建共享内存通道，跨机则退化为纯网络通道 ------
        buffer_io = MessageQueue(
            n_reader=1,
            n_local_reader=1 if same_node else 0,
            max_chunk_bytes=max_chunk_bytes,
            max_chunks=max_chunks,
        )
        handle = buffer_io.export_handle()
        # ------【进程管理】各进程把 Handle 汇聚到读者端，读者据此掌握全部写端信息 ------
        handles = [None] * dist.get_world_size(pg) if rank == reader_rank else None
        dist.gather_object(handle, handles, dst=reader_rank, group=pg)
        if blocking:
            buffer_io.wait_until_ready()
        return buffer_io, cast(list[Handle], handles or [])

    @staticmethod
    def create_from_process_group(
        pg: ProcessGroup | StatelessProcessGroup,
        max_chunk_bytes,
        max_chunks,
        writer_rank: int = 0,
        external_writer_handle=None,
        blocking: bool = True,
    ) -> "MessageQueue":
        """
        Creates a MessageQueue for a distributed process group with one writer and
        multiple readers.

        This method is designed for scenarios where one process (the writer) sends
        messages, and all other processes (the readers) receive messages. It sets up
        the shared memory buffer and socket communication handles accordingly, and
        broadcasts the handle from the writer to all readers.

        Args:
            pg (ProcessGroup | StatelessProcessGroup): The torch distributed process
                group.
            max_chunk_bytes (int): Maximum size in bytes for each chunk in the buffer.
            max_chunks (int): Maximum number of chunks in the buffer.
            writer_rank (int, optional): The global rank that will act as the writer.
                Defaults to 0.
            external_writer_handle (Handle, optional): Used when there is a handle
                from an external Message Queue. If provided, use this handle to init
                PG writer message queue instead of creating a new one. Defaults to None.
            blocking (bool, optional): If True, blocks until all processes are ready.
                Defaults to True.

        Returns:
            MessageQueue: The MessageQueue instance for the calling process.

        """
        # ------【进程管理】统一从 torch ProcessGroup 或轻量 StatelessProcessGroup 取组内 rank/规模 ------
        if isinstance(pg, ProcessGroup):
            group_rank = dist.get_rank(pg)
            group_world_size = dist.get_world_size(pg)
            global_ranks = dist.get_process_group_ranks(pg)
        else:
            group_rank = pg.rank
            group_world_size = pg.world_size
            global_ranks = list(range(pg.world_size))
        from vllm.distributed.parallel_state import in_the_same_node_as

        # ------【NUMA 亲和】按是否与写端同机区分本机/远程读者，本机走共享内存以省去网络开销 ------
        status = in_the_same_node_as(pg, source_rank=writer_rank)
        if group_rank == writer_rank:
            if external_writer_handle is not None:
                # ------【进程管理】复用了外部传入的 Handle，直接据此构造写端队列 ------
                buffer_io = MessageQueue.create_from_handle(
                    external_writer_handle, group_rank
                )
            else:
                # ------【NUMA 亲和】写端统计同机读者名单，据此确定本地读者数量与 rank ------
                same_node_ranks = [i for i, s in enumerate(status) if s]
                n_reader = group_world_size - 1
                n_local_reader = len(same_node_ranks) - 1
                local_reader_ranks = [i for i in same_node_ranks if i != writer_rank]
                buffer_io = MessageQueue(
                    n_reader=n_reader,
                    n_local_reader=n_local_reader,
                    local_reader_ranks=local_reader_ranks,
                    max_chunk_bytes=max_chunk_bytes,
                    max_chunks=max_chunks,
                )
            handle = buffer_io.export_handle()
            # ------【进程管理】写端把 Handle 广播给所有读者，读者据此附加共享内存并连套接字 ------
            if isinstance(pg, ProcessGroup):
                dist.broadcast_object_list(
                    [handle], src=global_ranks[writer_rank], group=pg
                )
            else:
                pg.broadcast_obj(handle, writer_rank)
        else:
            # ------【进程管理】读端接收写端广播的 Handle，再用它重建本地队列对象 ------
            if isinstance(pg, ProcessGroup):
                recv = [None]
                dist.broadcast_object_list(
                    recv, src=global_ranks[writer_rank], group=pg
                )
                handle = recv[0]  # type: ignore
            else:
                handle = pg.broadcast_obj(None, writer_rank)
            buffer_io = MessageQueue.create_from_handle(handle, group_rank)
        if blocking:
            buffer_io.wait_until_ready()
        return buffer_io
