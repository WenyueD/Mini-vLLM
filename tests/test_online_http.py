import http.client
import io
import json
import queue
import threading
import time
from types import SimpleNamespace

from myvllm.engine.output import RequestOutput
from myvllm.online import OnlineEngine
from myvllm.online.server import (
    CompletionRequestHandler,
    create_server,
    format_sse,
)
from myvllm.sampling_parameters import SamplingParams

from test_engine_requests import FakeRunner, make_engine, target
from test_online_runtime import BlockingDecodeRunner


def start_stack(engine):
    online = OnlineEngine(engine, close_engine=False)
    online.start()
    server = create_server(online, port=0, request_timeout=2)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return online, server, thread


def stop_stack(online, server, thread):
    server.shutdown()
    server.server_close()
    thread.join(timeout=2)
    online.close()


def request(server, method, path, body=None):
    connection = http.client.HTTPConnection(
        "127.0.0.1", server.server_address[1], timeout=3
    )
    encoded = None if body is None else json.dumps(body)
    headers = {"Content-Type": "application/json"} if body is not None else {}
    connection.request(method, path, body=encoded, headers=headers)
    response = connection.getresponse()
    payload = response.read()
    status = response.status
    content_type = response.getheader("Content-Type")
    connection.close()
    return status, content_type, payload


def test_sse_formatter_has_one_event_and_blank_line():
    assert format_sse({"x": 1}) == b'data: {"x": 1}\n\n'
    assert format_sse("[DONE]") == b"data: [DONE]\n\n"


def test_health_stats_non_stream_and_sse_protocol():
    engine = make_engine(speculative=True)
    engine.config["model_name_or_path"] = "fake-model"
    online, server, thread = start_stack(engine)
    try:
        status, _, body = request(server, "GET", "/health")
        assert status == 200
        assert json.loads(body) == {"status": "ok"}

        status, _, body = request(
            server,
            "POST",
            "/v1/completions",
            {
                "prompt": "1,2,3",
                "temperature": 0,
                "max_tokens": 7,
                "request_id": "non-stream",
            },
        )
        assert status == 200
        completion = json.loads(body)
        assert completion["id"] == "non-stream"
        assert completion["object"] == "text_completion"
        assert completion["choices"][0]["finish_reason"] == "length"
        assert completion["usage"]["completion_tokens"] == 7

        status, content_type, body = request(
            server,
            "POST",
            "/v1/completions",
            {
                "prompt": "4,5",
                "temperature": 0,
                "max_tokens": 8,
                "stream": True,
                "request_id": "streaming",
            },
        )
        assert status == 200
        assert content_type.startswith("text/event-stream")
        events = [part for part in body.decode().split("\n\n") if part]
        assert events[-1] == "data: [DONE]"
        chunks = [json.loads(event[len("data: ") :]) for event in events[:-1]]
        assert all(chunk["id"] == "streaming" for chunk in chunks)
        assert chunks[-1]["choices"][0]["finish_reason"] == "length"
        streamed_text = "".join(chunk["choices"][0]["text"] for chunk in chunks)
        expected_engine = make_engine(speculative=True)
        expected = expected_engine.generate(
            ["4,5"], SamplingParams(temperature=0, max_tokens=8)
        )["text"][0]
        assert streamed_text == expected
        assert "streaming" not in engine.requests

        status, _, body = request(server, "GET", "/stats")
        assert status == 200
        stats = json.loads(body)
        assert stats["engine_steps"] > 0
        assert stats["decode_steps"] > 0
    finally:
        stop_stack(online, server, thread)


def test_injected_stream_write_disconnect_aborts_and_releases_request():
    class DisconnectingWriter:
        closed = False

        def write(self, _data):
            raise BrokenPipeError("client disconnected")

        def flush(self):
            pass

    class Handle:
        request_id = "disconnect"

        def __init__(self):
            self.cancel_calls = 0
            self.release_calls = 0

        def iter_outputs(self, timeout=None):
            yield RequestOutput("disconnect", (7,), False)

        def cancel(self, timeout=None):
            self.cancel_calls += 1
            return True

        def release(self, timeout=None):
            self.release_calls += 1
            return True

    handle = Handle()
    engine = SimpleNamespace(
        speculative_config=SimpleNamespace(enabled=False),
        tokenizer=SimpleNamespace(decode=lambda ids: ",".join(map(str, ids))),
        config={},
    )
    online = SimpleNamespace(engine=engine, submit=lambda **_kwargs: handle)
    handler = CompletionRequestHandler.__new__(CompletionRequestHandler)
    handler.server = SimpleNamespace(
        online_engine=online,
        request_timeout=1,
        stream_lookbehind_tokens=0,
    )
    handler.wfile = DisconnectingWriter()
    handler._read_json = lambda: {
        "prompt": "1",
        "stream": True,
        "temperature": 0,
    }
    handler.send_response = lambda *_args: None
    handler.send_header = lambda *_args: None
    handler.end_headers = lambda: None

    handler._handle_completion()

    assert handle.cancel_calls >= 1
    assert handle.release_calls == 1


def test_stream_timeout_after_headers_uses_sse_error_without_second_status():
    class Handle:
        request_id = "timeout-stream"

        def __init__(self):
            self.cancel_calls = 0
            self.release_calls = 0

        def iter_outputs(self, timeout=None):
            raise queue.Empty
            yield

        def cancel(self, timeout=None):
            self.cancel_calls += 1
            return True

        def release(self, timeout=None):
            self.release_calls += 1
            return True

    handle = Handle()
    engine = SimpleNamespace(
        speculative_config=SimpleNamespace(enabled=False),
        tokenizer=SimpleNamespace(decode=lambda ids: ""),
        config={},
    )
    online = SimpleNamespace(engine=engine, submit=lambda **_kwargs: handle)
    handler = CompletionRequestHandler.__new__(CompletionRequestHandler)
    handler.server = SimpleNamespace(
        online_engine=online,
        request_timeout=0.01,
        stream_lookbehind_tokens=0,
    )
    handler.wfile = io.BytesIO()
    handler._read_json = lambda: {
        "prompt": "1",
        "stream": True,
        "temperature": 0,
    }
    statuses = []
    handler.send_response = lambda status, *_args: statuses.append(status)
    handler.send_header = lambda *_args: None
    handler.end_headers = lambda: None

    handler._handle_completion()

    assert statuses == [200]
    events = handler.wfile.getvalue().decode()
    assert '"type": "timeout"' in events
    assert events.endswith("data: [DONE]\n\n")
    assert handle.cancel_calls >= 1
    assert handle.release_calls == 1


def test_streaming_lookbehind_handles_non_prefix_cumulative_decode():
    class RewritingTokenizer:
        def encode(self, prompt):
            return [int(value) for value in prompt.split(",")]

        def decode(self, token_ids):
            values = list(token_ids)
            if not values:
                return ""
            if len(values) == 1:
                return f"first:{values[0]}"
            return "final:" + "|".join(map(str, values))

    tokenizer = RewritingTokenizer()
    assert not tokenizer.decode([1, 2]).startswith(tokenizer.decode([1]))
    engine = make_engine(speculative=False)
    engine.tokenizer = tokenizer
    online, server, thread = start_stack(engine)
    try:
        status, _, body = request(
            server,
            "POST",
            "/v1/completions",
            {
                "prompt": "1,2",
                "temperature": 0,
                "max_tokens": 3,
                "stream": True,
                "request_id": "rewrite",
            },
        )
        assert status == 200
        events = [part for part in body.decode().split("\n\n") if part]
        chunks = [json.loads(event[6:]) for event in events[:-1]]
        streamed_text = "".join(chunk["choices"][0]["text"] for chunk in chunks)
        expected_tokens = []
        history = [1, 2]
        for _ in range(3):
            token = target(history)
            expected_tokens.append(token)
            history.append(token)
        assert streamed_text == tokenizer.decode(expected_tokens)
        assert events[-1] == "data: [DONE]"
    finally:
        stop_stack(online, server, thread)


def test_online_spec_server_rejects_positive_temperature_explicitly():
    online, server, thread = start_stack(make_engine(speculative=True))
    try:
        status, _, body = request(
            server,
            "POST",
            "/v1/completions",
            {"prompt": "1,2", "temperature": 0.5},
        )
        assert status == 400
        assert "temperature=0" in json.loads(body)["error"]
    finally:
        stop_stack(online, server, thread)


class FailDecodeRunner(FakeRunner):
    def call(self, method, *args):
        if method == "run" and not args[1]:
            raise RuntimeError("injected target failure")
        return super().call(method, *args)


def test_sse_engine_error_is_framed_and_terminated_with_done():
    engine = make_engine(speculative=False)
    engine.model_runner = FailDecodeRunner()
    online, server, thread = start_stack(engine)
    try:
        status, _, body = request(
            server,
            "POST",
            "/v1/completions",
            {
                "prompt": "1,2",
                "temperature": 0,
                "max_tokens": 5,
                "stream": True,
                "request_id": "will-fail",
            },
        )
        assert status == 200
        events = [part for part in body.decode().split("\n\n") if part]
        assert events[-1] == "data: [DONE]"
        error = json.loads(events[-2][len("data: ") :])
        assert "injected target failure" in error["error"]["message"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_http_cancel_endpoint_aborts_running_stream_and_frees_blocks():
    engine = make_engine(speculative=False)
    runner = BlockingDecodeRunner()
    engine.model_runner = runner
    online, server, thread = start_stack(engine)
    stream_result = {}

    def stream_request():
        stream_result["response"] = request(
            server,
            "POST",
            "/v1/completions",
            {
                "prompt": "1,2,3",
                "temperature": 0,
                "max_tokens": 20,
                "stream": True,
                "request_id": "cancel-http",
            },
        )

    stream_thread = threading.Thread(target=stream_request)
    stream_thread.start()
    try:
        assert runner.decode_entered.wait(timeout=2)
        cancel_result = {}

        def cancel_request():
            cancel_result["response"] = request(
                server,
                "POST",
                "/v1/requests/cancel-http/cancel",
                {},
            )

        cancel_thread = threading.Thread(target=cancel_request)
        cancel_thread.start()
        deadline = time.monotonic() + 2
        while online._commands.empty() and time.monotonic() < deadline:
            time.sleep(0.001)
        assert not online._commands.empty()
        runner.release_decode.set()
        cancel_thread.join(timeout=3)
        stream_thread.join(timeout=3)
        assert not cancel_thread.is_alive()
        assert not stream_thread.is_alive()
        status, _, body = cancel_result["response"]
        assert status == 200
        assert json.loads(body)["cancelled"] is True
        _, _, stream_body = stream_result["response"]
        assert stream_body.decode().rstrip().endswith("data: [DONE]")
        assert online.status("cancel-http")["finish_reason"] == "cancelled"
        assert "cancel-http" not in engine.requests
        assert "cancel-http" not in online._channels
        engine.scheduler.block_manager.assert_consistent()
        assert len(engine.scheduler.block_manager.free_block_ids) == len(
            engine.scheduler.block_manager.blocks
        )
    finally:
        stop_stack(online, server, thread)
