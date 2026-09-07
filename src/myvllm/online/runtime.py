from __future__ import annotations

import queue
import threading
import uuid
from collections import OrderedDict
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any

from myvllm.engine.llm_engine import LLMEngine
from myvllm.engine.output import RequestOutput
from myvllm.sampling_parameters import SamplingParams


@dataclass
class _Command:
    kind: str
    payload: dict[str, Any]
    future: Future


@dataclass(frozen=True)
class OnlineRequestSpec:
    prompt: str
    sampling_params: SamplingParams
    request_id: str | None = None


class OnlineRequest:
    """Thread-safe consumer handle for one online request's committed outputs."""

    def __init__(
        self,
        runtime: "OnlineEngine",
        request_id: str,
        output_queue: queue.Queue[RequestOutput],
    ):
        self.runtime = runtime
        self.request_id = request_id
        self._output_queue = output_queue
        self._release_lock = threading.Lock()
        self._released = False

    def next_output(self, timeout: float | None = None) -> RequestOutput:
        output = self._output_queue.get(timeout=timeout)
        if output.finished and not output.error:
            self.release(timeout=timeout)
        return output

    def iter_outputs(self, timeout: float | None = None):
        while True:
            output = self.next_output(timeout)
            yield output
            if output.finished:
                return

    def cancel(self, timeout: float | None = None) -> bool:
        return self.runtime.abort(self.request_id, timeout=timeout)

    def release(self, timeout: float | None = None) -> bool:
        """Forget engine/channel state after a terminal event is observable."""
        with self._release_lock:
            if self._released:
                return True
            released = self.runtime.release(self.request_id, timeout=timeout)
            if released:
                self._released = True
            return released


class OnlineEngine:
    """Single-writer background loop around ``LLMEngine``.

    Callers submit commands through a queue. Only ``_run`` invokes mutating engine
    methods or advances the scheduler/model, so HTTP handler concurrency cannot
    race GPU state.
    """

    def __init__(
        self,
        engine: LLMEngine,
        close_engine: bool = True,
        max_terminal_history: int = 256,
    ):
        if max_terminal_history < 0:
            raise ValueError("max_terminal_history must be non-negative")
        self.engine = engine
        self.close_engine = close_engine
        self.max_terminal_history = max_terminal_history
        self._commands: queue.Queue[_Command] = queue.Queue()
        self._channels: dict[str, queue.Queue[RequestOutput]] = {}
        self._snapshot_lock = threading.Lock()
        self._statuses: dict[str, dict] = {}
        self._terminal_history: OrderedDict[str, dict] = OrderedDict()
        self._submitted_requests = 0
        self._released_requests = 0
        self._stats: dict = {"healthy": True, "online_running": False}
        self._healthy = True
        self._running = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def is_running(self) -> bool:
        return self._running.is_set()

    def start(self) -> None:
        if self.is_running:
            return
        self._running.set()
        self._thread = threading.Thread(
            target=self._run, name="minivllm-online-engine", daemon=True
        )
        self._thread.start()

    def submit(
        self,
        prompt: str,
        sampling_params: SamplingParams,
        request_id: str | None = None,
        timeout: float | None = None,
    ) -> OnlineRequest:
        self._require_running()
        request_id = request_id or f"online-{uuid.uuid4().hex}"
        future = Future()
        self._commands.put(
            _Command(
                "add",
                {
                    "prompt": prompt,
                    "sampling_params": sampling_params,
                    "request_id": request_id,
                },
                future,
            )
        )
        return future.result(timeout=timeout)

    def submit_batch(
        self,
        prompts: list[str],
        sampling_params: SamplingParams,
        request_ids: list[str] | None = None,
        timeout: float | None = None,
    ) -> list[OnlineRequest]:
        """Atomically admit a batch before the engine advances a scheduler step."""
        self._require_running()
        if not prompts:
            raise ValueError("submit_batch requires at least one prompt")
        if request_ids is None:
            request_ids = [f"online-{uuid.uuid4().hex}" for _ in prompts]
        if len(request_ids) != len(prompts):
            raise ValueError("request_ids must match the number of prompts")
        if len(set(request_ids)) != len(request_ids):
            raise ValueError("request_ids must be unique within a batch")

        return self.submit_requests(
            [
                OnlineRequestSpec(prompt, sampling_params, request_id)
                for prompt, request_id in zip(prompts, request_ids)
            ],
            timeout=timeout,
        )

    def submit_requests(
        self,
        requests: list[OnlineRequestSpec],
        timeout: float | None = None,
    ) -> list[OnlineRequest]:
        """Atomically admit request specs before the engine advances a step."""
        self._require_running()
        if not requests:
            raise ValueError("submit_requests requires at least one request")
        request_ids = [
            spec.request_id or f"online-{uuid.uuid4().hex}" for spec in requests
        ]
        if len(set(request_ids)) != len(request_ids):
            raise ValueError("request_ids must be unique within a batch")

        future = Future()
        self._commands.put(
            _Command(
                "add_batch",
                {
                    "requests": [
                        {
                            "prompt": spec.prompt,
                            "sampling_params": spec.sampling_params,
                            "request_id": request_id,
                        }
                        for spec, request_id in zip(requests, request_ids)
                    ]
                },
                future,
            )
        )
        return future.result(timeout=timeout)

    def abort(self, request_id: str, timeout: float | None = None) -> bool:
        self._require_running()
        future = Future()
        self._commands.put(_Command("abort", {"request_id": request_id}, future))
        return future.result(timeout=timeout)

    def release(self, request_id: str, timeout: float | None = None) -> bool:
        self._require_running()
        future = Future()
        self._commands.put(_Command("release", {"request_id": request_id}, future))
        return future.result(timeout=timeout)

    def status(self, request_id: str) -> dict | None:
        with self._snapshot_lock:
            status = self._statuses.get(request_id)
            if status is None:
                status = self._terminal_history.get(request_id)
            return dict(status) if status is not None else None

    def stats(self) -> dict:
        with self._snapshot_lock:
            return dict(self._stats)

    def close(self, timeout: float | None = 10.0) -> None:
        if not self.is_running:
            return
        future = Future()
        self._commands.put(_Command("stop", {}, future))
        future.result(timeout=timeout)
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def _require_running(self) -> None:
        if not self.is_running:
            raise RuntimeError("online engine is not running")
        if not self._healthy:
            raise RuntimeError("online engine is unhealthy")

    def _run(self) -> None:
        try:
            self._refresh_snapshots()
            while self._running.is_set():
                if self.engine.scheduler.is_finished():
                    command = self._commands.get()
                    if self._handle_command(command):
                        break
                else:
                    if self._drain_commands():
                        break
                    result = self.engine.step()
                    self._publish_step(result.outputs)
                    self._refresh_snapshots()
        except BaseException as exc:
            self._healthy = False
            self._fail_all(exc)
        finally:
            self._running.clear()
            if self.close_engine and hasattr(self.engine, "exit"):
                try:
                    self.engine.exit()
                except Exception:
                    self._healthy = False
            self._refresh_snapshots()

    def _drain_commands(self) -> bool:
        while True:
            try:
                command = self._commands.get_nowait()
            except queue.Empty:
                return False
            if self._handle_command(command):
                return True

    def _handle_command(self, command: _Command) -> bool:
        try:
            if command.kind == "add":
                handle = self._admit_request(command.payload)
                self._submitted_requests += 1
                command.future.set_result(handle)
                return False
            if command.kind == "add_batch":
                handles: list[OnlineRequest] = []
                try:
                    for payload in command.payload["requests"]:
                        handles.append(self._admit_request(payload))
                except BaseException:
                    self._rollback_admission(handles)
                    raise
                self._submitted_requests += len(handles)
                command.future.set_result(handles)
                return False
            if command.kind == "abort":
                request_id = command.payload["request_id"]
                aborted = self.engine.abort_request(request_id)
                if aborted:
                    self._publish(
                        RequestOutput(
                            request_id=request_id,
                            token_ids=(),
                            finished=True,
                            finish_reason="cancelled",
                        )
                    )
                self._update_status(request_id)
                command.future.set_result(aborted)
                return False
            if command.kind == "release":
                request_id = command.payload["request_id"]
                with self._snapshot_lock:
                    status = self._statuses.get(request_id)
                    if status is None:
                        status = self._terminal_history.get(request_id)
                if status is None:
                    try:
                        status = self.engine.get_request_status(request_id)
                    except KeyError:
                        status = None
                if status is None:
                    command.future.set_result(False)
                    return False
                if not status.get("finished", False):
                    raise RuntimeError(f"request is not terminal: {request_id}")
                forgotten = self.engine.forget_request(request_id)
                self._channels.pop(request_id, None)
                with self._snapshot_lock:
                    self._statuses.pop(request_id, None)
                    if self.max_terminal_history:
                        self._terminal_history[request_id] = dict(status)
                        self._terminal_history.move_to_end(request_id)
                        while len(self._terminal_history) > self.max_terminal_history:
                            self._terminal_history.popitem(last=False)
                    self._released_requests += int(forgotten)
                # Make the release counters/registry sizes visible before the
                # consumer unblocks from the synchronous release handshake.
                self._refresh_snapshots()
                command.future.set_result(True)
                return False
            if command.kind == "stop":
                for request_id in list(self._channels):
                    if self.engine.abort_request(request_id):
                        self._publish(
                            RequestOutput(
                                request_id=request_id,
                                token_ids=(),
                                finished=True,
                                finish_reason="cancelled",
                            )
                        )
                command.future.set_result(True)
                return True
            raise ValueError(f"unknown online command: {command.kind}")
        except BaseException as exc:
            command.future.set_exception(exc)
            return False
        finally:
            self._refresh_snapshots()

    def _admit_request(self, payload: dict[str, Any]) -> OnlineRequest:
        request_id = self.engine.add_request(**payload)
        output_queue: queue.Queue[RequestOutput] = queue.Queue()
        self._channels[request_id] = output_queue
        self._update_status(request_id)
        return OnlineRequest(self, request_id, output_queue)

    def _rollback_admission(self, handles: list[OnlineRequest]) -> None:
        for handle in handles:
            request_id = handle.request_id
            self.engine.abort_request(request_id)
            self.engine.forget_request(request_id)
            self._channels.pop(request_id, None)
            with self._snapshot_lock:
                self._statuses.pop(request_id, None)

    def _publish_step(self, outputs: tuple[RequestOutput, ...]) -> None:
        for output in outputs:
            self._publish(output)
            self._update_status(output.request_id)

    def _publish(self, output: RequestOutput) -> None:
        channel = self._channels.get(output.request_id)
        if channel is not None:
            channel.put(output)

    def _update_status(self, request_id: str) -> None:
        try:
            status = self.engine.get_request_status(request_id)
        except KeyError:
            return
        with self._snapshot_lock:
            self._statuses[request_id] = status

    def _refresh_snapshots(self) -> None:
        stats = self.engine.get_stats() if hasattr(self.engine, "get_stats") else {}
        with self._snapshot_lock:
            self._stats = {
                **stats,
                "healthy": self._healthy,
                "online_running": self._running.is_set(),
                "online_requests": {
                    "submitted": self._submitted_requests,
                    "released": self._released_requests,
                    "active_channels": len(self._channels),
                    "retained_terminal_statuses": len(self._terminal_history),
                    "terminal_history_limit": self.max_terminal_history,
                },
            }

    def _fail_all(self, exc: BaseException) -> None:
        message = f"{type(exc).__name__}: {exc}"
        for request_id, channel in self._channels.items():
            channel.put(
                RequestOutput(
                    request_id=request_id,
                    token_ids=(),
                    finished=True,
                    finish_reason="error",
                    error=message,
                )
            )
        while True:
            try:
                command = self._commands.get_nowait()
            except queue.Empty:
                break
            command.future.set_exception(RuntimeError(message))
