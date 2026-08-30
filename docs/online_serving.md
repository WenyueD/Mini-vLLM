# Online serving architecture

MiniVLLM online serving is a queue-driven wrapper around the existing engine. It
does not move speculative algorithms into HTTP code and does not allow concurrent
model-runner calls.

## Layers

1. `LLMEngine` owns `Sequence`, Scheduler, paged KV, model execution, stable
   `request_id`, abort, incremental `RequestOutput`, and engine-step metrics.
2. `OnlineEngine` owns one background thread. Add/cancel commands enter a
   thread-safe queue and committed outputs leave per-request queues. Only this
   thread mutates `LLMEngine`; it blocks on the command queue while idle.
3. `CompletionHTTPServer` translates `/v1/completions`, SSE, cancellation,
   `/health`, and `/stats`. Handler threads never call `step()` or `model_runner`.
4. The existing proposer → batched verifier → rejection sampler → KV transaction
   chain remains below the engine boundary and is unchanged by serving concerns.

Requests arriving during a model step wait in the command queue. They are inserted
before the next step, so waiting requests can prefill and subsequently join running
requests in a later decode batch. This is dynamic continuous batching. A client
that sends the next request only after the previous response completes exercises
serial inference behind HTTP, not continuous batching.

## Output and cancellation contracts

Each `LLMEngine.step()` output contains only tokens newly committed during that
step. For speculative requests, these are precisely the target-verified
`emitted_token_ids`; provisional draft tokens never reach a response queue.
`generate()` consumes the same events to preserve the offline API.

Abort removes waiting/running requests, rolls back an active Sequence transaction
and KV reservation, deallocates retained blocks, and emits a terminal event with
`finish_reason="cancelled"`. The HTTP cancellation endpoint and streaming socket
failure both enqueue this operation; they do not mutate the scheduler directly.

A terminal request remains in the engine registry and its output channel until a
consumer dequeues the terminal event. That dequeue sends a serialized `release`
command: the engine `Sequence` and channel are then removed, while only a bounded
terminal status/usage snapshot is retained (256 entries by default). Thus
`requests.total/finished` describe currently retained, not lifetime, engine
objects. `online_requests.submitted/released` are lifetime counters;
`active_channels` and `retained_terminal_statuses` expose current retention.
Fatal engine errors are the exception: their terminal error event bypasses the
release command because the command loop is already unhealthy. This guarantees
the error remains deliverable regardless of the brief interval between failure
publication and clearing the runtime's running flag; the stopped runtime owns the
remaining cleanup.

## Metrics

Every decode model batch produces one `SpeculativeBatchObservation`, including
active/speculative batch sizes, proposed/accepted totals, per-request acceptance
length, and per-position proposed/accepted arrays. `EngineMetrics` maintains
weighted global totals and a bounded step history. Coordinator
`sequence_records` remains an algorithm-level count and is not presented as engine
steps.

The `/stats` endpoint exposes this data. A decode `active_batch_size > 1` proves
that overlapping requests reached a shared model batch; HTTP concurrency alone
does not.

SSE text uses cumulative detokenization with an eight-token lookbehind. Text is
emitted only once it is outside that mutable suffix, and the terminal chunk flushes
the complete decode. If a tokenizer rewrites text beyond the retained suffix, the
server emits a `detokenization_rewrite` error and `[DONE]` instead of silently
concatenating incorrect text. This is a deliberately small OpenAI-style endpoint,
not a claim of complete OpenAI API or arbitrary-tokenizer streaming compatibility.
Once SSE headers have been committed, timeouts and handler errors are also framed
as an SSE error followed by `[DONE]`; the handler never attempts to write a second
HTTP status line.

## API policy

The server uses Python's standard library to avoid a FastAPI/ASGI dependency for
this small repository. A speculative server accepts only `temperature=0` and
returns HTTP 400 for positive temperatures. A target-only server retains the
ordinary positive-temperature path.
