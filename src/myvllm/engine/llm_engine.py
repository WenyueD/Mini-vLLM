import atexit
from itertools import count
import torch.distributed as dist
import time
import torch.multiprocessing as mp

from myvllm.engine.sequence import Sequence
from myvllm.engine.scheduler import Scheduler
from myvllm.engine.metrics import EngineMetrics
from myvllm.engine.output import (
    EngineStepResult,
    RequestOutput,
    SpeculativeBatchObservation,
)
from myvllm.sampling_parameters import SamplingParams
from myvllm.spec_decode import (
    MockProposer,
    NGramProposer,
    Proposer,
    SpecDecodeCoordinator,
    SpeculativeConfig,
)
from myvllm.spec_decode.types import Proposal
from transformers import AutoTokenizer


def worker_process(config, rank, event):
    """Worker process function that initializes ModelRunner and enters loop."""
    # FIRST print before any other code
    import sys
    import os

    sys.stdout = os.fdopen(sys.stdout.fileno(), "w", buffering=1)  # Line buffering
    sys.stderr = os.fdopen(sys.stderr.fileno(), "w", buffering=1)

    from myvllm.engine.model_runner import ModelRunner

    model_runner = ModelRunner(config, rank, event)
    model_runner.loop()


class RunnerDFlash2Proposer(Proposer):
    supports_parallel = True

    def __init__(self, model_runner):
        self.model_runner = model_runner

    def propose(self, sequence: Sequence, max_tokens: int) -> Proposal:
        return self.propose_batch([sequence], [max_tokens])[0]

    def propose_batch(
        self, sequences: list[Sequence], max_tokens: list[int]
    ) -> list[Proposal]:
        return self.model_runner.call("propose_dflash2", sequences, max_tokens)


class LLMEngine:
    def __init__(self, config: dict):
        from myvllm.engine.model_runner import ModelRunner

        self.config = config
        self._closed = False
        self._request_counter = count()
        self.requests: dict[str, Sequence] = {}
        self.metrics = EngineMetrics(config.get("max_metrics_history", 256))
        self.speculative_config = SpeculativeConfig.from_value(
            config.get("speculative_config")
        )
        world_size = config.get("world_size", 1)
        ctx = mp.get_context("spawn")
        self.processes = []
        self.events = []
        for i in range(1, world_size):
            event = ctx.Event()
            process = ctx.Process(target=worker_process, args=(config, i, event))
            self.events.append(event)
            self.processes.append(process)
            process.start()
        # start the engine only on the master thread with rank = 0
        self.model_runner = ModelRunner(config, rank=0, event=self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.get("model_name_or_path", "gpt2")
        )

        # scheduler needs to init after model_runner: when world_size > 1,
        # ModelRunner.__init__ calls dist.init_process_group() which is a
        # collective barrier — rank-0 blocks until all worker ranks have joined.
        # The scheduler should only be created after that rendezvous completes.
        # When world_size == 1 there is no barrier and no real dependency.
        self.scheduler = Scheduler(
            max_num_sequences=config.get("max_num_sequences", 16),
            max_num_batched_tokens=config.get("max_num_batched_tokens", 1024),
            max_cached_blocks=config.get("max_cached_blocks", 1024),
            block_size=config.get("block_size", 256),
            eos=config.get("eos", 50256),
        )
        self.spec_decode_coordinator = None
        if self.speculative_config.enabled:
            if self.speculative_config.proposer == "ngram":
                proposer = NGramProposer(
                    self.speculative_config.ngram_min_size,
                    self.speculative_config.ngram_max_size,
                )
            elif self.speculative_config.proposer == "dflash2":
                proposer = RunnerDFlash2Proposer(self.model_runner)
            else:
                proposer = MockProposer()
            self.spec_decode_coordinator = SpecDecodeCoordinator(
                self.speculative_config, proposer
            )

        self._atexit_callback = self.exit
        atexit.register(self._atexit_callback)

    def exit(self):
        if self._closed:
            return
        self._closed = True
        try:
            model_runner = getattr(self, "model_runner", None)
            if model_runner is not None:
                model_runner.call("exit")
        finally:
            # RunnerDFlash2Proposer owns a second reference to the model runner.
            # Break it before dropping the engine so repeated in-process runs can
            # actually return the model weights to CUDA.
            self.spec_decode_coordinator = None
            if hasattr(self, "model_runner"):
                del self.model_runner
            for process in getattr(self, "processes", []):
                process.join()
            callback = getattr(self, "_atexit_callback", None)
            if callback is not None:
                atexit.unregister(callback)
                self._atexit_callback = None

    # call scheduler to schedule the next batch
    # return scheduled sequences and whether it is for prefilling
    # call model_runner.run() to run the model
    # call postprocessor to process the outputs and update sequences and update block manager
    def _ensure_runtime_state(self) -> None:
        """Keep lightweight test/fake engines constructed with ``__new__`` working."""
        if not hasattr(self, "requests"):
            self.requests = {}
        if not hasattr(self, "_request_counter"):
            self._request_counter = count()
        if not hasattr(self, "metrics"):
            self.metrics = EngineMetrics()

    def step(self) -> EngineStepResult:
        self._ensure_runtime_state()
        scheduled_sequences, is_prefill = self.scheduler.schedule(
            speculative=self.speculative_config.enabled
        )
        num_processed_tokens = 0
        if not scheduled_sequences:
            return EngineStepResult(
                engine_step=self.metrics.engine_steps,
                outputs=(),
                num_processed_tokens=0,
                is_prefill=is_prefill,
            )
        engine_step = self.metrics.next_engine_step()
        completion_offsets = {
            seq.seq_id: len(seq.completion_token_ids) for seq in scheduled_sequences
        }
        spec_results = []
        if is_prefill:
            outputs = self.model_runner.call("run", scheduled_sequences, True)
            if outputs is not None:
                outputs = outputs.cpu().tolist()
            self.scheduler.postprocess(scheduled_sequences, outputs, is_prefill=True)
            num_processed_tokens = sum(
                seq.num_prompt_tokens for seq in scheduled_sequences
            )
        else:
            speculative_sequences = [
                seq
                for seq in scheduled_sequences
                if self.spec_decode_coordinator is not None and seq.is_greedy
            ]
            ordinary_sequences = [
                seq for seq in scheduled_sequences if seq not in speculative_sequences
            ]
            if ordinary_sequences:
                ordinary_outputs = self.model_runner.call(
                    "run", ordinary_sequences, False
                )
                if ordinary_outputs is not None:
                    ordinary_outputs = ordinary_outputs.cpu().tolist()
                self.scheduler.postprocess(ordinary_sequences, ordinary_outputs)
                num_processed_tokens += len(ordinary_sequences)
            if speculative_sequences:
                commit_verified_cache = None
                rollback_verified_cache = None
                if hasattr(self.model_runner, "commit_verified_cache") and hasattr(
                    self.model_runner, "rollback_verified_cache"
                ):
                    commit_verified_cache = lambda accepted: self.model_runner.call(
                        "commit_verified_cache", accepted
                    )
                    rollback_verified_cache = lambda sequence_ids: self.model_runner.call(
                        "rollback_verified_cache", sequence_ids
                    )
                spec_results = self.spec_decode_coordinator.execute(
                    speculative_sequences,
                    self.scheduler.block_manager,
                    lambda batch: self.model_runner.call("verify", batch),
                    self.scheduler.eos,
                    max_verification_tokens=self.scheduler.max_num_batched_tokens,
                    commit_verified_cache=commit_verified_cache,
                    rollback_verified_cache=rollback_verified_cache,
                )
                num_processed_tokens += sum(
                    len(result.emitted_token_ids) for result in spec_results
                )
                self.scheduler.finalize_speculative(speculative_sequences)

        outputs = []
        for seq in scheduled_sequences:
            start = completion_offsets[seq.seq_id]
            committed = tuple(seq.completion_token_ids[start:])
            if committed or seq.is_finished:
                outputs.append(
                    RequestOutput(
                        request_id=seq.request_id,
                        token_ids=committed,
                        finished=seq.is_finished,
                        finish_reason=seq.finish_reason,
                    )
                )

        observation = None
        if not is_prefill:
            per_request = {seq.request_id: 0 for seq in scheduled_sequences}
            proposed_tokens = 0
            accepted_tokens = 0
            per_position_proposed: list[int] = []
            per_position_accepted: list[int] = []
            by_seq_id = {seq.seq_id: seq for seq in scheduled_sequences}
            for result in spec_results:
                seq = by_seq_id[result.sequence_id]
                proposed_tokens += result.proposed_tokens
                accepted_tokens += result.accepted_proposal_tokens
                per_request[seq.request_id] = result.accepted_proposal_tokens
                while len(per_position_proposed) < result.proposed_tokens:
                    per_position_proposed.append(0)
                    per_position_accepted.append(0)
                for position, accepted in enumerate(result.per_position_acceptance):
                    per_position_proposed[position] += 1
                    per_position_accepted[position] += int(accepted)
            observation = SpeculativeBatchObservation(
                engine_step=engine_step,
                active_batch_size=len(scheduled_sequences),
                speculative_batch_size=len(speculative_sequences),
                proposed_tokens=proposed_tokens,
                accepted_tokens=accepted_tokens,
                batch_acceptance_rate=(
                    accepted_tokens / proposed_tokens if proposed_tokens else 0.0
                ),
                per_request_accepted_length=per_request,
                per_position_proposed=tuple(per_position_proposed),
                per_position_accepted=tuple(per_position_accepted),
            )
            self.metrics.record_decode(observation)

        finished_request_ids = [
            seq.request_id for seq in scheduled_sequences if seq.is_finished
        ]
        if finished_request_ids and hasattr(
            self.model_runner, "release_hybrid_requests"
        ):
            self.model_runner.call("release_hybrid_requests", finished_request_ids)

        return EngineStepResult(
            engine_step=engine_step,
            outputs=tuple(outputs),
            num_processed_tokens=num_processed_tokens,
            is_prefill=is_prefill,
            speculative=observation,
        )

    def get_speculative_stats(self, *, include_history: bool = True) -> dict:
        if self.spec_decode_coordinator is None:
            return {
                "enabled": False,
                "proposed_tokens": 0,
                "accepted_tokens": 0,
            }
        return {
            "enabled": True,
            **self.spec_decode_coordinator.metrics.snapshot(include_history=include_history),
        }

    def get_live_stats(self) -> dict:
        """Read current counters without serializing retained diagnostic history."""
        return self.get_stats(include_history=False)

    def get_stats(self, *, include_history: bool = True) -> dict:
        self._ensure_runtime_state()
        result = self.metrics.snapshot(include_history=include_history)
        result["speculative"] = self.get_speculative_stats(include_history=include_history)
        model_runner = getattr(self, "model_runner", None)
        if model_runner is not None and getattr(
            model_runner, "is_qwen35_target", False
        ):
            result["qwen35_verification"] = model_runner.call(
                "get_qwen35_verification_stats"
            )
            if include_history and hasattr(model_runner, "get_target_decision_diagnostics"):
                result["target_decision_diagnostics"] = model_runner.call(
                    "get_target_decision_diagnostics"
                )
            if include_history and hasattr(model_runner, "get_memory_trace"):
                result["memory_trace"] = model_runner.call("get_memory_trace")
        if include_history and model_runner is not None and getattr(
            model_runner, "dflash2_draft_model", None
        ) is not None:
            result["dflash2_profile"] = model_runner.call(
                "get_dflash2_profile_stats"
            )
        result["requests"] = {
            "total": len(self.requests),
            "waiting": len(self.scheduler.waiting),
            "running": len(self.scheduler.running),
            "finished": sum(seq.is_finished for seq in self.requests.values()),
        }
        return result

    # add prompt string to the waiting queue by first transforming it to Sequence object
    def add_request(
        self,
        prompt: str,
        sampling_params: SamplingParams,
        request_id: str | None = None,
    ) -> str:
        self._ensure_runtime_state()
        request_id = request_id or f"request-{next(self._request_counter)}"
        if request_id in self.requests:
            raise ValueError(f"duplicate request_id: {request_id}")
        token_ids = self.tokenizer.encode(prompt)
        if not token_ids:
            raise ValueError("prompt must encode to at least one token")
        sequence = Sequence(
            token_ids=token_ids,
            block_size=self.config["block_size"],
            sampling_params=sampling_params,
            request_id=request_id,
        )
        self.scheduler.add_sequence(sequence)
        self.requests[request_id] = sequence
        return request_id

    def add_prompt(
        self,
        prompt: str,
        sampling_params: SamplingParams,
        request_id: str | None = None,
    ) -> str:
        return self.add_request(prompt, sampling_params, request_id)

    def get_request_status(self, request_id: str) -> dict:
        self._ensure_runtime_state()
        seq = self.requests.get(request_id)
        if seq is None:
            raise KeyError(request_id)
        return {
            "request_id": request_id,
            "status": seq.status.name.lower(),
            "finished": seq.is_finished,
            "finish_reason": seq.finish_reason,
            "num_prompt_tokens": seq.num_prompt_tokens,
            "num_completion_tokens": seq.num_completion_tokens,
        }

    def abort_request(self, request_id: str) -> bool:
        self._ensure_runtime_state()
        seq = self.requests.get(request_id)
        if seq is None:
            return False
        return self.scheduler.abort(seq)

    def forget_request(self, request_id: str) -> bool:
        """Release a terminal request from the engine registry.

        Online consumers call this only after dequeuing the terminal event.  Keeping
        this separate from scheduler finalization prevents a fast producer from
        deleting status/usage before the consumer has observed completion.
        """
        self._ensure_runtime_state()
        seq = self.requests.get(request_id)
        if seq is None:
            return False
        if not seq.is_finished:
            raise RuntimeError(f"request is not terminal: {request_id}")
        del self.requests[request_id]
        return True

    # given a list of prompts
    # add_prompt for each prompt
    # call step until all sequences are finished
    # return the generated texts
    def generate(
        self, prompts: list[str], sampling_params: SamplingParams
    ) -> list[str]:
        request_ids = [self.add_prompt(prompt, sampling_params) for prompt in prompts]
        generated_tokens = {request_id: [] for request_id in request_ids}
        while not self.scheduler.is_finished():
            start_t = time.time()
            step_result = self.step()
            end_t = time.time()
            running_time = end_t - start_t + 1e-10
            if step_result.is_prefill:
                print(
                    step_result.num_processed_tokens,
                    "number of processed tokens",
                    step_result.num_processed_tokens / running_time,
                    "tokens/sec during prefilling",
                )
            else:
                print(
                    step_result.num_processed_tokens,
                    "number of processed tokens",
                    step_result.num_processed_tokens / running_time,
                    "tokens/sec during decoding",
                )
            for output in step_result.outputs:
                generated_tokens[output.request_id].extend(output.token_ids)

        ordered_tokens = [generated_tokens[request_id] for request_id in request_ids]
        output = {
            "text": [self.tokenizer.decode(tokens) for tokens in ordered_tokens],
            "token_ids": ordered_tokens,
        }
        return output
