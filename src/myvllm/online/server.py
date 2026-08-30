from __future__ import annotations

import argparse
import json
import queue
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from myvllm.engine.llm_engine import LLMEngine
from myvllm.engine.output import RequestOutput
from myvllm.online.runtime import OnlineEngine
from myvllm.sampling_parameters import SamplingParams


def format_sse(payload: dict[str, Any] | str) -> bytes:
    data = payload if isinstance(payload, str) else json.dumps(payload)
    return f"data: {data}\n\n".encode("utf-8")


def _completion_chunk(
    request_id: str,
    text: str,
    finish_reason: str | None,
    model: str,
) -> dict:
    return {
        "id": request_id,
        "object": "text_completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "text": text,
                "logprobs": None,
                "finish_reason": finish_reason,
            }
        ],
    }


class CompletionHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        server_address,
        online_engine: OnlineEngine,
        request_timeout: float = 300.0,
        stream_lookbehind_tokens: int = 8,
    ):
        if stream_lookbehind_tokens < 0:
            raise ValueError("stream_lookbehind_tokens must be non-negative")
        self.online_engine = online_engine
        self.request_timeout = request_timeout
        self.stream_lookbehind_tokens = stream_lookbehind_tokens
        super().__init__(server_address, CompletionRequestHandler)


class CompletionRequestHandler(BaseHTTPRequestHandler):
    server: CompletionHTTPServer
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:
        if self.path == "/health":
            stats = self.server.online_engine.stats()
            healthy = stats.get("healthy", False) and stats.get("online_running", False)
            self._send_json(
                HTTPStatus.OK if healthy else HTTPStatus.SERVICE_UNAVAILABLE,
                {"status": "ok" if healthy else "unhealthy"},
            )
            return
        if self.path == "/stats":
            self._send_json(HTTPStatus.OK, self.server.online_engine.stats())
            return
        self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path == "/v1/completions":
            self._handle_completion()
            return
        if self.path.startswith("/v1/requests/") and self.path.endswith("/cancel"):
            request_id = self.path[len("/v1/requests/") : -len("/cancel")]
            self._handle_cancel(request_id)
            return
        self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})

    def _handle_completion(self) -> None:
        handle = None
        completed = False
        self._response_started = False
        try:
            payload = self._read_json()
            prompt = payload.get("prompt")
            if not isinstance(prompt, str):
                raise ValueError("prompt must be a string")
            stream = payload.get("stream", False)
            if not isinstance(stream, bool):
                raise ValueError("stream must be a boolean")
            temperature = payload.get("temperature", 0)
            if not isinstance(temperature, (int, float)):
                raise ValueError("temperature must be numeric")
            spec_config = self.server.online_engine.engine.speculative_config
            if spec_config.enabled and temperature != 0:
                raise ValueError(
                    "online speculative decoding supports only temperature=0; "
                    "use a target-only server for positive temperatures"
                )
            params = SamplingParams(
                temperature=float(temperature),
                max_tokens=int(payload.get("max_tokens", 16)),
                ignore_eos=bool(payload.get("ignore_eos", False)),
                max_model_length=payload.get("max_model_length"),
            )
            handle = self.server.online_engine.submit(
                prompt=prompt,
                sampling_params=params,
                request_id=payload.get("request_id"),
                timeout=self.server.request_timeout,
            )
            if stream:
                completed = self._stream_completion(handle)
            else:
                completed = self._non_stream_completion(handle)
        except ValueError as exc:
            if self._response_started:
                self._try_send_sse_error(str(exc), "invalid_request_error")
            else:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
        except queue.Empty:
            if handle is not None:
                try:
                    handle.cancel(timeout=5)
                except Exception:
                    pass
            if self._response_started:
                self._try_send_sse_error("request timeout", "timeout")
            else:
                self._send_json(
                    HTTPStatus.GATEWAY_TIMEOUT, {"error": "request timeout"}
                )
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            if handle is not None and not completed:
                try:
                    handle.cancel(timeout=5)
                except Exception:
                    pass
        except Exception as exc:
            if handle is not None and not completed:
                try:
                    handle.cancel(timeout=5)
                except Exception:
                    pass
            if self._response_started:
                self._try_send_sse_error(f"{type(exc).__name__}: {exc}", "server_error")
            elif not self.wfile.closed:
                self._send_json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {"error": f"{type(exc).__name__}: {exc}"},
                )
        finally:
            if handle is not None and not completed:
                try:
                    handle.cancel(timeout=5)
                except Exception:
                    pass
                try:
                    handle.release(timeout=5)
                except Exception:
                    pass

    def _non_stream_completion(self, handle) -> bool:
        token_ids = []
        finish_reason = None
        for output in handle.iter_outputs(timeout=self.server.request_timeout):
            if output.error:
                raise RuntimeError(output.error)
            token_ids.extend(output.token_ids)
            finish_reason = output.finish_reason
        tokenizer = self.server.online_engine.engine.tokenizer
        text = tokenizer.decode(token_ids)
        status = self.server.online_engine.status(handle.request_id) or {}
        prompt_tokens = status.get("num_prompt_tokens", 0)
        body = _completion_chunk(
            handle.request_id,
            text,
            finish_reason,
            self._model_name(),
        )
        body["usage"] = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": len(token_ids),
            "total_tokens": prompt_tokens + len(token_ids),
        }
        self._send_json(HTTPStatus.OK, body)
        return True

    def _stream_completion(self, handle) -> bool:
        self._response_started = True
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        tokenizer = self.server.online_engine.engine.tokenizer
        all_token_ids = []
        emitted_text = ""
        lookbehind = self.server.stream_lookbehind_tokens
        for output in handle.iter_outputs(timeout=self.server.request_timeout):
            if output.error:
                self.wfile.write(
                    format_sse(
                        {
                            "error": {
                                "message": output.error,
                                "type": "engine_error",
                            }
                        }
                    )
                )
                self.wfile.flush()
                self.wfile.write(format_sse("[DONE]"))
                self.wfile.flush()
                return True
            all_token_ids.extend(output.token_ids)
            stable_token_ids = (
                all_token_ids
                if output.finished
                else all_token_ids[: max(0, len(all_token_ids) - lookbehind)]
            )
            stable_text = tokenizer.decode(stable_token_ids)
            if not stable_text.startswith(emitted_text):
                self.wfile.write(
                    format_sse(
                        {
                            "error": {
                                "message": (
                                    "tokenizer rewrote text beyond the streaming "
                                    "lookbehind window"
                                ),
                                "type": "detokenization_rewrite",
                            }
                        }
                    )
                )
                self.wfile.flush()
                self.wfile.write(format_sse("[DONE]"))
                self.wfile.flush()
                return True
            text_delta = stable_text[len(emitted_text) :]
            emitted_text = stable_text
            if text_delta or output.finished:
                self.wfile.write(
                    format_sse(
                        _completion_chunk(
                            handle.request_id,
                            text_delta,
                            output.finish_reason if output.finished else None,
                            self._model_name(),
                        )
                    )
                )
                self.wfile.flush()
        self.wfile.write(format_sse("[DONE]"))
        self.wfile.flush()
        return True

    def _try_send_sse_error(self, message: str, error_type: str) -> bool:
        """Best-effort terminal SSE framing after HTTP headers were committed."""
        try:
            self.wfile.write(
                format_sse({"error": {"message": message, "type": error_type}})
            )
            self.wfile.flush()
            self.wfile.write(format_sse("[DONE]"))
            self.wfile.flush()
            return True
        except Exception:
            return False

    def _handle_cancel(self, request_id: str) -> None:
        if not request_id:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "missing request_id"})
            return
        cancelled = self.server.online_engine.abort(
            request_id, timeout=self.server.request_timeout
        )
        self._send_json(
            HTTPStatus.OK if cancelled else HTTPStatus.NOT_FOUND,
            {"id": request_id, "cancelled": cancelled},
        )

    def _read_json(self) -> dict:
        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length)
            value = json.loads(body or b"{}")
        except (ValueError, json.JSONDecodeError) as exc:
            raise ValueError("invalid JSON body") from exc
        if not isinstance(value, dict):
            raise ValueError("JSON body must be an object")
        return value

    def _model_name(self) -> str:
        return str(
            self.server.online_engine.engine.config.get(
                "model_name_or_path", "minivllm"
            )
        )

    def _send_json(self, status: int, payload: dict) -> None:
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, format: str, *args) -> None:
        return


def create_server(
    online_engine: OnlineEngine,
    host: str = "127.0.0.1",
    port: int = 8000,
    request_timeout: float = 300.0,
    stream_lookbehind_tokens: int = 8,
) -> CompletionHTTPServer:
    return CompletionHTTPServer(
        (host, port),
        online_engine,
        request_timeout=request_timeout,
        stream_lookbehind_tokens=stream_lookbehind_tokens,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve MiniVLLM completions")
    parser.add_argument("--config", required=True, help="Path to engine JSON config")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    engine = LLMEngine(config)
    online = OnlineEngine(engine)
    online.start()
    server = create_server(online, args.host, args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        online.close()


if __name__ == "__main__":
    main()
