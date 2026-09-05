import queue
import threading
import time

import pytest

from myvllm.online import OnlineEngine, OnlineRequest
from myvllm.sampling_parameters import SamplingParams

from test_engine_requests import FakeRunner, make_engine


def consume(handle, first_output=None):
    outputs = [] if first_output is None else [first_output]
    if first_output is not None and first_output.finished:
        return outputs
    for output in handle.iter_outputs(timeout=2):
        outputs.append(output)
    return outputs


def test_fatal_error_terminal_is_delivered_while_running_flag_remains_set():
    # Reproduce the exact fatal-window ordering deterministically: _healthy is
    # already false and _fail_all has published, but _run.finally has not cleared
    # _running yet. Error delivery must never depend on that transient flag.
    for index in range(200):
        engine = make_engine(speculative=False)
        online = OnlineEngine(engine, close_engine=False)
        channel = queue.Queue()
        request_id = f"fatal-{index}"
        online._channels[request_id] = channel
        online._running.set()
        online._healthy = False
        handle = OnlineRequest(online, request_id, channel)

        online._fail_all(RuntimeError("controlled fatal failure"))
        assert online.is_running
        output = handle.next_output(timeout=0.1)

        assert output.finished
        assert output.finish_reason == "error"
        assert output.error == "RuntimeError: controlled fatal failure"
        assert online._commands.empty()


@pytest.mark.parametrize("speculative", [False, True])
def test_online_runtime_streams_committed_tokens_and_matches_final_state(speculative):
    engine = make_engine(speculative)
    online = OnlineEngine(engine, close_engine=False)
    online.start()
    try:
        handle = online.submit(
            "1,2,3",
            SamplingParams(temperature=0, max_tokens=10, ignore_eos=True),
            request_id=f"stream-{speculative}",
        )
        outputs = consume(handle)
        streamed = [token for output in outputs for token in output.token_ids]
        expected_engine = make_engine(speculative)
        expected = expected_engine.generate(
            ["1,2,3"],
            SamplingParams(temperature=0, max_tokens=10, ignore_eos=True),
        )["token_ids"][0]
        assert streamed == expected
        assert handle.request_id not in engine.requests
        assert outputs[-1].finished
        assert outputs[-1].finish_reason == "length"
        sizes = [len(output.token_ids) for output in outputs if output.token_ids]
        if speculative:
            assert any(size > 1 for size in sizes)
        else:
            assert all(size == 1 for size in sizes)
    finally:
        online.close()


def test_submit_batch_admits_all_requests_before_first_engine_step():
    engine = make_engine(speculative=False)
    online = OnlineEngine(engine, close_engine=False)
    online.start()
    try:
        handles = online.submit_batch(
            ["1,2,3", "4,5"],
            SamplingParams(temperature=0, max_tokens=4, ignore_eos=True),
            request_ids=["batch-a", "batch-b"],
        )
        outputs = [consume(handle) for handle in handles]

        assert all(items[-1].finished for items in outputs)
        assert any(
            step["active_batch_size"] == 2
            for step in online.stats()["recent_decode_steps"]
        )
    finally:
        online.close()


def test_submit_batch_rejects_invalid_request_ids_before_admission():
    engine = make_engine(speculative=False)
    online = OnlineEngine(engine, close_engine=False)
    online.start()
    try:
        params = SamplingParams(temperature=0, max_tokens=1)
        with pytest.raises(ValueError, match="match the number"):
            online.submit_batch(["1", "2"], params, request_ids=["only-one"])
        with pytest.raises(ValueError, match="unique"):
            online.submit_batch(["1", "2"], params, request_ids=["same", "same"])
        assert not engine.requests
    finally:
        online.close()


def test_consumed_terminal_requests_have_bounded_retention_and_no_kv_leak():
    engine = make_engine(speculative=False)
    online = OnlineEngine(engine, close_engine=False, max_terminal_history=17)
    online.start()
    try:
        for index in range(300):
            handle = online.submit(
                "1,2",
                SamplingParams(temperature=0, max_tokens=1, ignore_eos=True),
                f"short-{index}",
            )
            outputs = consume(handle)
            assert outputs[-1].finished

        stats = online.stats()
        assert not engine.requests
        assert not online._channels
        assert len(online._terminal_history) == 17
        assert stats["requests"] == {
            "total": 0,
            "waiting": 0,
            "running": 0,
            "finished": 0,
        }
        assert stats["online_requests"] == {
            "submitted": 300,
            "released": 300,
            "active_channels": 0,
            "retained_terminal_statuses": 17,
            "terminal_history_limit": 17,
        }
        assert online.status("short-0") is None
        assert online.status("short-299")["finish_reason"] == "length"
        engine.scheduler.block_manager.assert_consistent()
        assert len(engine.scheduler.block_manager.free_block_ids) == len(
            engine.scheduler.block_manager.blocks
        )
    finally:
        online.close()


class BlockingDecodeRunner(FakeRunner):
    def __init__(self):
        self.decode_entered = threading.Event()
        self.release_decode = threading.Event()
        self.block_once = True

    def call(self, method, *args):
        if method == "run" and not args[1] and self.block_once:
            self.block_once = False
            self.decode_entered.set()
            assert self.release_decode.wait(timeout=2)
        return super().call(method, *args)


def test_late_request_joins_continuous_batch_and_idle_loop_does_not_spin():
    engine = make_engine(speculative=False)
    runner = BlockingDecodeRunner()
    engine.model_runner = runner
    online = OnlineEngine(engine, close_engine=False)
    online.start()
    try:
        first = online.submit(
            "1,2", SamplingParams(temperature=0, max_tokens=8), "first"
        )
        first_prefill = first.next_output(timeout=2)
        assert first_prefill.request_id == "first"
        assert runner.decode_entered.wait(timeout=2)

        second_holder = {}

        def submit_second():
            second_holder["handle"] = online.submit(
                "3,4", SamplingParams(temperature=0, max_tokens=5), "second"
            )

        submit_thread = threading.Thread(target=submit_second)
        submit_thread.start()
        time.sleep(0.02)
        runner.release_decode.set()
        submit_thread.join(timeout=2)
        assert not submit_thread.is_alive()
        second = second_holder["handle"]

        first_outputs = consume(first, first_prefill)
        second_outputs = consume(second)
        assert all(output.request_id == "first" for output in first_outputs)
        assert all(output.request_id == "second" for output in second_outputs)
        assert any(
            step["active_batch_size"] == 2
            for step in online.stats()["recent_decode_steps"]
        )

        steps_when_idle = engine.metrics.engine_steps
        time.sleep(0.05)
        assert engine.metrics.engine_steps == steps_when_idle
    finally:
        online.close()


def test_online_cancel_running_request_publishes_terminal_and_frees_kv():
    engine = make_engine(speculative=False)
    runner = BlockingDecodeRunner()
    engine.model_runner = runner
    online = OnlineEngine(engine, close_engine=False)
    online.start()
    try:
        handle = online.submit(
            "1,2,3", SamplingParams(temperature=0, max_tokens=20), "cancel-me"
        )
        first = handle.next_output(timeout=2)
        assert runner.decode_entered.wait(timeout=2)

        cancelled = {}

        def cancel():
            cancelled["value"] = handle.cancel(timeout=2)

        cancel_thread = threading.Thread(target=cancel)
        cancel_thread.start()
        time.sleep(0.02)
        runner.release_decode.set()
        cancel_thread.join(timeout=2)
        assert cancelled["value"]
        outputs = consume(handle, first)
        assert outputs[-1].finished
        assert outputs[-1].finish_reason == "cancelled"
        assert not engine.scheduler.running
        engine.scheduler.block_manager.assert_consistent()
        assert len(engine.scheduler.block_manager.free_block_ids) == len(
            engine.scheduler.block_manager.blocks
        )
    finally:
        online.close()
