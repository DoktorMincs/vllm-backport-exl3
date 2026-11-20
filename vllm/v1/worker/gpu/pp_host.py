# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host-relayed pipeline-parallel handoff: ``VLLM_PP_TRANSPORT=host``.

Why this exists
---------------
On this deployment (TP2 x PP5, 10x RTX 3090 on sm_86, no GPUDirect P2P) the
decode step spends 7-8 ms per PP stage boundary inside NCCL: the PP groups
are strided (``[0,2,4,6,8]`` / ``[1,3,5,7,9]``), so every hop crosses a socket,
NCCL has to stage through host memory anyway, and the receive path also pays a
blocking gloo round-trip for the tensor metadata (``recv_object``) plus an
NCCL all-gather to undo the TP sharding of each tensor.

This module replaces *only* the inter-stage byte movement with a host-side
relay. The reference engine (Project Maya) uses the same idea, but it runs
every rank in one process and can therefore hand a host-mapped buffer to both
GPUs: ``xfer_put`` writes D2H into mapped memory, sets a ``seq`` word after
``__threadfence_system()`` and ``xfer_get`` spins on it from the consumer GPU
(``src/ds4/ds4_kernels.cu``). That zero-copy shape is NOT available here:

* vLLM runs one process per rank, and ``cudaHostRegister`` (which is what makes
  a host address usable by a GPU) is only accepted by the first process that
  touched the physical page. See ``vllm/v1/kv_offload/cpu/gpu_worker.py``.
  Sibling ranks cannot register the same pages, so a mapped buffer cannot be
  shared between two rank processes.

What we do instead (per directed PP edge, sender -> receiver):

    GPU (main stream)
      -> D2H on the edge's tx side stream, into a *private pinned* buffer
      -> [relay thread] waits on the tx CUDA event, memcpy into a /dev/shm
         slot, then publishes ``seq``
      -> [receiver] spins on ``seq``, memcpy into its own *private pinned*
         buffer, acks the slot, H2D on the edge's rx side stream
      -> main stream waits on the rx event

Both endpoints therefore keep the "own pinned buffer per rank" property, the
flag is a plain 8-byte word in the shared segment (written last), and the
"short spin" is on the CPU: a GPU-side spin would need a custom kernel, and
the flag is not device-mapped in this design (see ``Risks`` in the docstring
of :class:`HostPPTransport`).

Stream ordering, explicitly
---------------------------
Top of every step, ``wait_send()`` (called from ``Worker.execute_model`` next
to the NCCL path's ``_pp_send_work`` wait):

* ``main_stream.wait_event(previous tx event)`` -- the forward pass queued
  after this point cannot replay into the graph output buffers the previous
  handoff is still copying out. Pure stream dependency, no CPU block. The
  first PP rank needs this explicitly: it has no ``recv_tensors`` to piggyback
  on.

Sender, in ``send_tensors``:

1. ``_wait_flushed()`` -- the relay must have copied the previous payload out
   of the pinned buffer before the GPU refills it (a CPU wait that in steady
   state returns immediately; the NCCL path pays the same kind of wait on its
   send handles).
2. ``tx_stream.wait_stream(main_stream)`` -- the payload was produced on main.
3. ``copies -> tx_stream.record_event()``.

The relay thread calls ``event.synchronize()`` before touching the pinned
buffer: that is the D2H completion gate.

Receiver, in ``recv_tensors``:

1. ``previous rx event .synchronize()`` -- the previous H2D must be done
   before the pinned staging buffer is overwritten by the CPU.
2. spin on ``seq`` -> host memcpy -> ack the slot.
3. ``rx_stream.wait_stream(main_stream)`` (allocator ordering for the freshly
   allocated destinations), ``copies -> rx_stream.record_event()``.
4. ``main_stream.wait_event(rx event)`` before returning, so the model that
   consumes the tensors is ordered after the H2D.
5. TP all-gather (when the tensor was sharded by the sender) runs on the main
   stream, i.e. after step 4 -- same point in the stream order as the NCCL
   path's ``comm_postprocess``.

CUDA graph capture
------------------
``send_tensors`` / ``recv_tensors`` are called from ``Worker.execute_model``,
*around* the model forward, never from inside a captured region (capture goes
through ``model_runner.capture_model()``, and the memory-profile dummy run
through ``profile_run()``). The transport is therefore graph-safe at its call
site as long as it is never invoked while the current stream is capturing;
:meth:`HostPPTransport._assert_not_capturing` enforces that with a hard error
instead of silently poisoning a graph. ``VLLM_PP_HOST_FORCE_EAGER=1`` is
provided as a bring-up knob that turns off compilation/graphs on this path
(the config change happens in ``Worker.__init__`` before the runner is built).

Validation status (be honest)
-----------------------------
* Validated here, on CPU only: imports and ``py_compile``; the metadata and
  layout helpers; and the full send -> relay -> recv frame protocol (seq/ack,
  slot rotation, TP-shard layout, dtype views, empty and odd-sized tensors,
  overflow rejection) driven with fake CUDA streams/events, fake PP ranks and
  a fake TP group (two transport instances in one process, scratch tests, not
  committed).
* NOT validated: anything that needs a GPU -- the real D2H/H2D copies, the
  stream ordering on real streams, the gloo handshake between rank processes,
  CUDA-graph interplay and, above all, the latency. No CUDA context was
  created while writing this module.
* Not implemented on purpose: the token broadcast (``PPHandler``) and
  ``pp_share`` keep using NCCL for now; the host transport carries exactly
  what ``send_tensor_dict`` carries (the ``IntermediateTensors`` dict, which
  *includes* the ``pp_share`` payload, since ``PPSharePlan.send`` folds its
  tensors into that dict).
"""

from __future__ import annotations

import ctypes
import mmap
import os
import pickle
import threading
import time
from dataclasses import dataclass, field

import numpy as np
import torch

from vllm import envs
from vllm.distributed.parallel_state import GroupCoordinator, get_pp_group
from vllm.logger import init_logger
from vllm.sequence import IntermediateTensors

logger = init_logger(__name__)

# Per-slot header (cache-line sized), then the pickled metadata, then the
# payload. All offsets are 64-byte aligned so a payload slice can be viewed as
# any dtype.
_HDR_BYTES = 64
_META_CAP = 1 << 16  # 64 KiB of pickled per-tensor metadata
_ALIGN = 64

# Header word indexes (8-byte words at the start of each slot).
_W_SEQ = 0
_W_NBYTES = 1
_W_META_LEN = 2
_W_ACK = 3
_W_RESERVED = 4


def _round_up(n: int, a: int = _ALIGN) -> int:
    return (n + a - 1) // a * a


def pack_meta(entries: list[tuple]) -> bytes:
    """Serialize the per-tensor layout of one step.

    Each entry is ``(key, shape, dtype_str, sharded, shard_rank, offset,
    nbytes, nshards)``. ``shard_rank`` is the TP rank the slice belongs to when
    ``sharded`` (the receiver all-gathers ``nshards`` pieces), else -1.
    """
    return pickle.dumps(entries, protocol=4)


def unpack_meta(blob: bytes) -> list[tuple]:
    entries = pickle.loads(blob)
    if not isinstance(entries, list):
        raise ValueError(f"corrupt PP host metadata: {type(entries)!r}")
    return entries


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


def _dtype_from_name(name: str) -> torch.dtype:
    dtype = getattr(torch, name, None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"unknown dtype {name!r} in PP host metadata")
    return dtype


def _should_shard(
    key: str,
    numel: int,
    group: GroupCoordinator | None,
    overrides: dict[str, bool] | None,
) -> bool:
    """Same rule as ``GroupCoordinator._should_use_all_gather``: a tensor whose
    flat size divides by the group size is sent as one slice per rank and
    rebuilt with an all-gather on the receiver. Empty tensors are never
    sharded (the NCCL path skips them instead of all-gathering)."""
    if group is None or numel == 0:
        return False
    use = numel % group.world_size == 0
    if overrides is not None:
        use = overrides.get(key, use)
    return use


@dataclass
class _TxEdge:
    """Outgoing edge (this rank -> next PP rank): the shared segment."""

    path: str
    capacity: int
    slots: int
    slot_stride: int = 0
    size: int = 0
    mm: mmap.mmap | None = None
    words: np.ndarray = field(default=None)  # type: ignore[assignment]
    base: int = 0  # data address of slot 0

    def __post_init__(self) -> None:
        self.slot_stride = _HDR_BYTES + _META_CAP + self.capacity
        self.size = self.slot_stride * self.slots
        fd = os.open(self.path, os.O_CREAT | os.O_RDWR | os.O_TRUNC, 0o600)
        try:
            os.ftruncate(fd, self.size)
            self.mm = mmap.mmap(fd, self.size)
        finally:
            os.close(fd)
        assert self.mm is not None
        # One flat u64 view over the whole segment: every slot's header words
        # live at ``slot * slot_stride // 8``. (slot_stride is 64-aligned.)
        self.words = np.frombuffer(self.mm, dtype=np.uint64)
        self.base = ctypes.addressof(ctypes.c_char.from_buffer(self.mm))

    def _w(self, slot: int, word: int) -> int:
        return slot * (self.slot_stride // 8) + word

    def seq(self, slot: int) -> int:
        return int(self.words[self._w(slot, _W_SEQ)])

    def ack(self, slot: int) -> int:
        return int(self.words[self._w(slot, _W_ACK)])

    def publish(self, slot: int, seq: int, nbytes: int, meta: bytes) -> None:
        """Write metadata + counters, then the sequence word last."""
        assert len(meta) <= _META_CAP
        meta_off = slot * self.slot_stride + _HDR_BYTES
        ctypes.memmove(self.base + meta_off, meta, len(meta))
        self.words[self._w(slot, _W_NBYTES)] = nbytes
        self.words[self._w(slot, _W_META_LEN)] = len(meta)
        # Release: on x86-64 store/store ordering already guarantees the data
        # above is visible before this word; the explicit ordering is what
        # matters (a torn payload is caught by the receiver's nbytes/meta
        # checks in recv_tensors).
        self.words[self._w(slot, _W_SEQ)] = seq

    def close(self) -> None:
        # The numpy view exports a buffer on the mmap; release it first.
        self.words = None  # type: ignore[assignment]
        if self.mm is not None:
            self.mm.close()
            self.mm = None


@dataclass
class _RxEdge:
    """Incoming edge (previous PP rank -> this rank)."""

    path: str
    size: int
    capacity: int
    slots: int
    slot_stride: int = 0
    mm: mmap.mmap | None = None
    words: np.ndarray = field(default=None)  # type: ignore[assignment]
    base: int = 0

    def __post_init__(self) -> None:
        self.slot_stride = _HDR_BYTES + _META_CAP + self.capacity
        assert self.size == self.slot_stride * self.slots, (
            "PP host edge size mismatch: sender and receiver disagree on "
            f"capacity/slots ({self.size} != {self.slot_stride * self.slots})"
        )
        fd = os.open(self.path, os.O_RDWR)
        try:
            self.mm = mmap.mmap(fd, self.size)
        finally:
            os.close(fd)
        assert self.mm is not None
        self.words = np.frombuffer(self.mm, dtype=np.uint64)
        self.base = ctypes.addressof(ctypes.c_char.from_buffer(self.mm))

    def _w(self, slot: int, word: int) -> int:
        return slot * (self.slot_stride // 8) + word

    def seq(self, slot: int) -> int:
        return int(self.words[self._w(slot, _W_SEQ)])

    def meta_len(self, slot: int) -> int:
        return int(self.words[self._w(slot, _W_META_LEN)])

    def nbytes(self, slot: int) -> int:
        return int(self.words[self._w(slot, _W_NBYTES)])

    def ack(self, slot: int, seq: int) -> None:
        self.words[self._w(slot, _W_ACK)] = seq

    def meta_off(self, slot: int) -> int:
        return slot * self.slot_stride + _HDR_BYTES

    def data_addr(self, slot: int) -> int:
        return self.base + slot * self.slot_stride + _HDR_BYTES + _META_CAP

    def close(self) -> None:
        self.words = None  # type: ignore[assignment]
        if self.mm is not None:
            self.mm.close()
            self.mm = None


class HostPPTransport:
    """Host-relayed PP handoff, see the module docstring for the design and
    the (deliberately explicit) list of what is and is not validated.

    One instance per rank; every rank owns at most one outgoing edge (to the
    next PP rank) and one incoming edge (from the previous one). Both PP ends
    are absent on the first/last rank.

    Risks (all real, none measured):

    * Deadlock -- the relay is step-synchronous: one send per step, one recv
      per step. If one side skips a step (e.g. a rank disagreeing on
      ``forward_pass``) the two ends drift and the receiver spins until
      ``VLLM_PP_HOST_TIMEOUT_S``. The transport cannot detect drift on its
      own; the timeout error names the edge and the sequence numbers.
    * Buffer size -- the capacity is sized from the model's
      ``make_empty_intermediate_tensors`` at ``max_num_batched_tokens``
      (i.e. bytes/token x token budget) times a headroom factor, capped by
      ``VLLM_PP_HOST_MAX_BYTES``. An oversized step raises at send time
      instead of truncating.
    * Host memory -- each rank holds capacity x (1 TX + 1 RX + slots x shm)
      bytes: ~120 MB per rank at a 64 MiB cap. ``/dev/shm`` must have room for
      ``slots x capacity`` per edge or the segment creation fails at boot.
    * Graph capture -- see the module docstring; a hard guard refuses to run
      while a stream capture is in progress.
    """

    def __init__(
        self,
        *,
        device: torch.device,
        max_num_batched_tokens: int,
        template_bytes: int,
        slots: int | None = None,
        max_bytes: int | None = None,
        timeout_s: float | None = None,
        shm_dir: str | None = None,
    ) -> None:
        self.device = device
        self.main_stream = torch.cuda.current_stream(device)
        self.tx_stream = torch.cuda.Stream(device)
        self.rx_stream = torch.cuda.Stream(device)

        cap = int(max_bytes if max_bytes is not None else envs.VLLM_PP_HOST_MAX_BYTES)
        # bytes/token x token budget already; 1.5x covers aux-hidden-state
        # relays and the fact that the local template is the *incoming* layout.
        want = max(1 << 20, _round_up(int(template_bytes * 1.5)))
        capacity = _round_up(min(int(cap), want))
        if want > capacity:
            logger.warning(
                "HostPPTransport: payload up to %.1f MiB but capacity is capped "
                "at %.1f MiB (VLLM_PP_HOST_MAX_BYTES=%d); a step above the cap "
                "will raise at send time.",
                want / 2**20,
                capacity / 2**20,
                cap,
            )
        self.capacity = capacity
        self.slots = int(slots if slots is not None else envs.VLLM_PP_HOST_SLOTS)
        self.timeout_s = float(
            timeout_s if timeout_s is not None else envs.VLLM_PP_HOST_TIMEOUT_S
        )
        self.max_num_batched_tokens = int(max_num_batched_tokens)
        self.shm_dir = shm_dir or envs.VLLM_PP_HOST_DIR

        pp = get_pp_group()
        self.pp = pp
        self.rank_in_group = pp.rank_in_group
        self.global_rank = pp.rank
        self.prev = self.rank_in_group - 1 if self.rank_in_group > 0 else None
        self.next = (
            self.rank_in_group + 1 if self.rank_in_group + 1 < pp.world_size else None
        )

        self._tx: _TxEdge | None = None
        self._rx: _RxEdge | None = None
        self._tx_pinned: torch.Tensor | None = None
        self._tx_pinned_addr = 0
        self._rx_pinned: torch.Tensor | None = None
        self._rx_pinned_addr = 0

        self._sent_seq = 0
        self._recv_seq = 0
        self._flushed_seq = 0
        self._last_tx_event: torch.cuda.Event | None = None
        self._last_rx_event: torch.cuda.Event | None = None

        self._pending: list[tuple] = []
        self._cv = threading.Condition()
        self._thread: threading.Thread | None = None
        self._closed = False
        self._relay_error: BaseException | None = None

        # Handshake, once per edge. Each rank first receives its incoming edge
        # from the previous rank, then creates its outgoing edge and tells the
        # next one. That order is a chain, so it cannot deadlock.
        if self.prev is not None:
            info = pp.recv_object(src=self.prev)
            self._attach_incoming(info)
        if self.next is not None:
            info = self._create_outgoing()
            pp.send_object(info, dst=self.next)

        logger.info(
            "HostPPTransport rank=%d (pp_group world=%d): prev=%s next=%s "
            "capacity=%.1f MiB slots=%d host_bytes=%.1f MiB",
            self.global_rank,
            pp.world_size,
            None if self.prev is None else pp.ranks[self.prev],
            None if self.next is None else pp.ranks[self.next],
            self.capacity / 2**20,
            self.slots,
            (self._segment_bytes() * self.slots + 2 * self.capacity) / 2**20,
        )

    # ------------------------------------------------------------------ setup

    def _segment_bytes(self) -> int:
        return _HDR_BYTES + _META_CAP + self.capacity

    def _path_for(self, src: int, dst: int) -> str:
        return os.path.join(
            self.shm_dir,
            f"vllm_pp_host_{os.getpid()}_{src}_{dst}.mmap",
        )

    def _create_outgoing(self) -> dict:
        src = self.global_rank
        dst = self.pp.ranks[self.next]  # type: ignore[index]
        path = self._path_for(src, dst)
        edge = _TxEdge(path=path, capacity=self.capacity, slots=self.slots)
        self._tx = edge
        self._tx_pinned = torch.empty(
            self.capacity, dtype=torch.uint8, pin_memory=True
        )
        self._tx_pinned_addr = self._tx_pinned.data_ptr()
        return {
            "path": path,
            "size": edge.size,
            "capacity": self.capacity,
            "slots": self.slots,
        }

    def _attach_incoming(self, info: dict) -> None:
        capacity = int(info["capacity"])
        if capacity > self.capacity:
            # The sender's layout needs more than our template: grow locally
            # (the shm segment is the sender's, so its size is authoritative).
            logger.warning(
                "HostPPTransport: incoming edge capacity %.1f MiB > local %.1f "
                "MiB; growing the receive buffer.",
                capacity / 2**20,
                self.capacity / 2**20,
            )
            self.capacity = capacity
        self._rx = _RxEdge(
            path=str(info["path"]),
            size=int(info["size"]),
            capacity=capacity,
            slots=int(info["slots"]),
        )
        self._rx_pinned = torch.empty(
            self.capacity, dtype=torch.uint8, pin_memory=True
        )
        self._rx_pinned_addr = self._rx_pinned.data_ptr()

    # ------------------------------------------------------------- guard rail

    @staticmethod
    def _assert_not_capturing(where: str) -> None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                f"HostPPTransport.{where} was called while a CUDA graph was "
                "being captured. VLLM_PP_TRANSPORT=host is not capture-safe; "
                "run with VLLM_PP_HOST_FORCE_EAGER=1 (or --enforce-eager)."
            )

    # -------------------------------------------------------------------- send

    def wait_send(self) -> None:
        """Order the caller's main stream after the previous step's D2H.

        This is the host transport's counterpart of the NCCL path's
        ``_pp_send_work`` wait, and it must run at the *top* of the step: the
        forward pass queued afterwards would otherwise replay into the graph
        output buffers that the previous handoff is still copying out. The
        wait is a stream dependency, so it does not block the CPU.
        """
        if self._last_tx_event is not None:
            self.main_stream.wait_event(self._last_tx_event)

    def send_tensors(
        self,
        tensors: dict[str, torch.Tensor],
        all_gather_tensors: dict[str, bool] | None = None,
    ) -> None:
        if self._tx is None or self._tx_pinned is None:
            raise RuntimeError("HostPPTransport.send_tensors on the last PP rank")
        self._assert_not_capturing("send_tensors")

        # (1) The next forward must not overwrite graph buffers the previous
        # step's D2H is still reading: order the *main* stream after it. The
        # worker also calls wait_send() at the top of the step (rank 0 has no
        # recv to piggyback on); this is the same, idempotent dependency.
        self.wait_send()

        # The relay must have copied the previous payload out of the pinned
        # buffer before the GPU overwrites it. In steady state this is already
        # true and costs no wait; it is the CPU-side analogue of the NCCL
        # path's handle wait.
        self._wait_flushed()

        tp = _tp_group()
        entries: list[tuple] = []
        off = 0
        # The payload was produced on the main stream: every copy below runs on
        # the tx side stream, ordered after it once.
        self.tx_stream.wait_stream(self.main_stream)
        with torch.cuda.stream(self.tx_stream):
            for key, t in tensors.items():
                if not isinstance(t, torch.Tensor):
                    raise TypeError(
                        f"HostPPTransport only carries tensors; key {key!r} "
                        f"holds {type(t)!r}. Use VLLM_PP_TRANSPORT=nccl for "
                        "this model."
                    )
                tt = t if t.is_contiguous() else t.contiguous()
                esize = tt.element_size()
                sharded = _should_shard(key, tt.numel(), tp, all_gather_tensors)
                src = tt
                shard_rank = -1
                if sharded:
                    assert tp is not None
                    shard_rank = tp.rank_in_group
                    width = tt.numel() // tp.world_size
                    src = tt.reshape(-1)[shard_rank * width : (shard_rank + 1) * width]
                nbytes = src.numel() * esize
                end = _round_up(off + nbytes)
                if end > self.capacity:
                    raise RuntimeError(
                        f"HostPPTransport payload overflow: key {key!r} needs "
                        f"{nbytes} B at offset {off}, capacity is "
                        f"{self.capacity} B ({len(tensors)} tensors, "
                        f"max_num_batched_tokens={self.max_num_batched_tokens})."
                        " Raise VLLM_PP_HOST_MAX_BYTES."
                    )
                entries.append(
                    (
                        key,
                        tuple(tt.shape),
                        _dtype_name(tt.dtype),
                        sharded,
                        shard_rank,
                        off,
                        nbytes,
                        tp.world_size if sharded else 1,
                    )
                )
                if nbytes:
                    dst = self._tx_pinned[off : off + nbytes].view(tt.dtype).view(
                        src.shape
                    )
                    # D2H into this rank's private pinned buffer.
                    dst.copy_(src, non_blocking=True)
                # 64-byte stride between tensors so each slice can be viewed as
                # any dtype on the receive side.
                off = end

        meta = pack_meta(entries)
        slot = self._sent_seq % self.slots
        self._sent_seq += 1
        seq = self._sent_seq

        ev = self.tx_stream.record_event()
        self._last_tx_event = ev
        with self._cv:
            self._pending.append((slot, seq, ev, off, meta))
            self._cv.notify()
        self._ensure_thread()

    def _wait_flushed(self) -> None:
        if self._relay_error is not None:
            raise RuntimeError(
                "HostPPTransport relay thread died; see the traceback logged "
                "above"
            ) from self._relay_error
        deadline = time.monotonic() + self.timeout_s
        with self._cv:
            while self._flushed_seq < self._sent_seq:
                if self._relay_error is not None:
                    raise RuntimeError(
                        "HostPPTransport relay thread died; see the traceback "
                        "logged above"
                    ) from self._relay_error
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RuntimeError(
                        "HostPPTransport: relay thread did not drain the send "
                        f"buffer (flushed={self._flushed_seq}, "
                        f"sent={self._sent_seq}) within {self.timeout_s}s"
                    )
                self._cv.wait(min(0.05, remaining))

    def _wait_slot_free(self, slot: int, seq: int) -> None:
        """A slot may be reused once the receiver acked the previous message
        that used it (``ack >= seq - slots``)."""
        assert self._tx is not None
        want = seq - self.slots
        deadline = time.monotonic() + self.timeout_s
        while self._tx.ack(slot) < want:
            if time.monotonic() > deadline:
                raise RuntimeError(
                    f"HostPPTransport: receiver never acked slot {slot} "
                    f"(sent seq {seq}, ack {self._tx.ack(slot)}, "
                    f"peer={self.pp.ranks[self.next]}). The PP step counts of "
                    "the two stages have drifted, or the peer died."
                )
            time.sleep(1e-5)

    def _ensure_thread(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._relay_loop, name="vllm-pp-host-relay", daemon=True
        )
        self._thread.start()

    def _relay_loop(self) -> None:
        assert self._tx is not None and self._tx_pinned is not None
        torch.cuda.set_device(self.device)
        edge = self._tx
        while True:
            with self._cv:
                while not self._pending and not self._closed:
                    self._cv.wait(0.25)
                if self._closed and not self._pending:
                    return
                slot, seq, ev, nbytes, meta = self._pending.pop(0)
            try:
                self._wait_slot_free(slot, seq)
                # D2H completion gate: the pinned buffer now holds the step.
                ev.synchronize()
                if nbytes:
                    ctypes.memmove(
                        edge.base + slot * edge.slot_stride + _HDR_BYTES + _META_CAP,
                        self._tx_pinned_addr,
                        nbytes,
                    )
                # Metadata and counters, then the sequence word last: the
                # receiver only trusts a frame whose seq it is waiting for.
                edge.publish(slot, seq, nbytes, meta)
            except BaseException as exc:  # noqa: BLE001 - surfaced in the caller
                logger.exception("HostPPTransport relay failed at seq %d", seq)
                with self._cv:
                    self._relay_error = exc
                    self._cv.notify_all()
                return
            finally:
                with self._cv:
                    self._flushed_seq = seq
                    self._cv.notify_all()

    # -------------------------------------------------------------------- recv

    def recv_tensors(
        self,
        all_gather_tensors: dict[str, bool] | None = None,
    ) -> IntermediateTensors:
        """Receive one step and return GPU tensors already ordered after the
        H2D.

        ``all_gather_tensors`` is accepted for signature symmetry with
        ``irecv_tensor_dict`` but not consulted: which tensors are sharded is
        decided by the sender and carried in the frame metadata (the sender
        applies exactly the same ``_should_use_all_gather`` rule).
        """
        if self._rx is None or self._rx_pinned is None:
            raise RuntimeError("HostPPTransport.recv_tensors on the first PP rank")
        self._assert_not_capturing("recv_tensors")

        # The previous H2D must be complete before the CPU overwrites the
        # pinned staging buffer.
        if self._last_rx_event is not None:
            self._last_rx_event.synchronize()

        seq = self._recv_seq + 1
        slot = self._recv_seq % self.slots
        self._recv_seq = seq
        self._spin_for_seq(slot, seq)

        nbytes = self._rx.nbytes(slot)
        meta_len = self._rx.meta_len(slot)
        if meta_len > _META_CAP or nbytes > self._rx.capacity:
            raise RuntimeError(
                f"HostPPTransport: corrupt frame on edge {self.prev}->"
                f"{self.rank_in_group} (seq {seq}, nbytes {nbytes}, "
                f"meta_len {meta_len}, capacity {self._rx.capacity})"
            )
        meta = ctypes.string_at(self._rx.base + self._rx.meta_off(slot), meta_len)
        if nbytes:
            ctypes.memmove(
                self._rx_pinned_addr,
                self._rx.data_addr(slot),
                nbytes,
            )
        # The frame is in our private buffer: let the sender reuse the slot.
        self._rx.ack(slot, seq)

        entries = unpack_meta(meta)
        tp = _tp_group()
        out: dict[str, torch.Tensor] = {}
        pending_gather: list[tuple] = []

        with torch.cuda.stream(self.rx_stream):
            self.rx_stream.wait_stream(self.main_stream)
            for key, shape, dtype_name, sharded, shard_rank, off, nb, nshards in entries:
                dtype = _dtype_from_name(dtype_name)
                if sharded:
                    # Same layout as the NCCL path: each TP rank on the sender
                    # side shipped one flat slice; rebuild with an all-gather.
                    full = torch.empty(shape, dtype=dtype, device=self.device)
                    view = full.reshape(nshards, -1)[shard_rank]
                    if nb:
                        src = self._rx_pinned[off : off + nb].view(dtype)
                        view.copy_(src, non_blocking=True)
                    # Allocated under rx_stream, read by the main stream in the
                    # all-gather below: both streams must be recorded.
                    full.record_stream(self.rx_stream)
                    full.record_stream(self.main_stream)
                    out[key] = full
                    pending_gather.append((key, view, shape))
                else:
                    t = torch.empty(shape, dtype=dtype, device=self.device)
                    if nb:
                        src = (
                            self._rx_pinned[off : off + nb]
                            .view(dtype)
                            .view(shape)
                        )
                        t.copy_(src, non_blocking=True)
                    t.record_stream(self.rx_stream)
                    t.record_stream(self.main_stream)
                    out[key] = t
            self._last_rx_event = self.rx_stream.record_event()

        # Consume only after the H2D landed; the all-gather below therefore
        # runs at the same point as the NCCL path's comm_postprocess.
        self.main_stream.wait_event(self._last_rx_event)

        if pending_gather:
            assert tp is not None
            for key, view, shape in pending_gather:
                out[key] = tp.all_gather(view, dim=0).reshape(shape)

        return IntermediateTensors(out)

    def _spin_for_seq(self, slot: int, seq: int) -> None:
        assert self._rx is not None
        deadline = time.monotonic() + self.timeout_s
        spins = 0
        while self._rx.seq(slot) < seq:
            spins += 1
            if spins < 8192:
                # Short busy wait, but yield the GIL regularly: this rank's own
                # relay thread (its outgoing edge) has to keep running while we
                # wait for the upstream stage.
                if (spins & 63) == 0:
                    time.sleep(0)
            else:
                time.sleep(2e-5)
            if time.monotonic() > deadline:
                raise RuntimeError(
                    f"HostPPTransport: timed out after {self.timeout_s}s "
                    f"waiting for seq {seq} on edge {self.prev}->"
                    f"{self.rank_in_group} (slot {slot}, seen "
                    f"{self._rx.seq(slot)}). The sender is stuck or the two "
                    "stages' step counts have drifted."
                )

    # ------------------------------------------------------------------- close

    def close(self) -> None:
        with self._cv:
            self._closed = True
            self._cv.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        if self._tx is not None:
            self._tx.close()
            try:
                os.unlink(self._tx.path)
            except OSError:
                pass
        if self._rx is not None:
            self._rx.close()


def _tp_group() -> GroupCoordinator | None:
    from vllm.distributed.parallel_state import get_tp_group

    try:
        return get_tp_group()
    except AssertionError:
        return None
