import atexit
import torch.distributed as dist
import time
import torch.multiprocessing as mp

from myvllm.engine.sequence import Sequence
from myvllm.engine.scheduler import Scheduler
from myvllm.sampling_parameters import SamplingParams
from myvllm.spec_decode import (
    MockProposer,
    NGramProposer,
    SpecDecodeCoordinator,
    SpeculativeConfig,
)
from transformers import AutoTokenizer


def worker_process(config, rank, event):
    """Worker process function that initializes ModelRunner and enters loop."""
    # FIRST print before any other code
    import sys
    import os
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)  # Line buffering
    sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', buffering=1)

    from myvllm.engine.model_runner import ModelRunner

    model_runner = ModelRunner(config, rank, event)
    model_runner.loop()


class LLMEngine:
    def __init__(self, config: dict):
        from myvllm.engine.model_runner import ModelRunner

        self.config = config
        self._closed = False
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
        self.tokenizer = AutoTokenizer.from_pretrained(config.get("model_name_or_path", "gpt2"))
        
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
            eos=config.get("eos", 50256)
        )
        self.spec_decode_coordinator = None
        if self.speculative_config.enabled:
            if self.speculative_config.proposer == "ngram":
                proposer = NGramProposer(
                    self.speculative_config.ngram_min_size,
                    self.speculative_config.ngram_max_size,
                )
            else:
                proposer = MockProposer()
            self.spec_decode_coordinator = SpecDecodeCoordinator(
                self.speculative_config, proposer
            )

        atexit.register(self.exit)


    def exit(self):
        if self._closed:
            return
        self._closed = True
        self.model_runner.call("exit")
        del self.model_runner
        for process in self.processes:
            process.join()

    # call scheduler to schedule the next batch
    # return scheduled sequences and whether it is for prefilling
    # call model_runner.run() to run the model
    # call postprocessor to process the outputs and update sequences and update block manager
    def step(self) -> tuple[list[tuple[int, list[int]]], int, bool]:
        scheduled_sequences, is_prefill = self.scheduler.schedule(
            speculative=self.speculative_config.enabled
        )
        num_processed_tokens = 0
        if not scheduled_sequences:
            return [], num_processed_tokens, is_prefill
        if is_prefill:
            outputs = self.model_runner.call("run", scheduled_sequences, True)
            if outputs is not None:
                outputs = outputs.cpu().tolist()
            self.scheduler.postprocess(
                scheduled_sequences, outputs, is_prefill=True
            )
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
                spec_results = self.spec_decode_coordinator.execute(
                    speculative_sequences,
                    self.scheduler.block_manager,
                    lambda batch: self.model_runner.call("verify", batch),
                    self.scheduler.eos,
                    max_verification_tokens=self.scheduler.max_num_batched_tokens,
                )
                num_processed_tokens += sum(
                    len(result.emitted_token_ids) for result in spec_results
                )
                self.scheduler.finalize_speculative(speculative_sequences)

        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in scheduled_sequences if seq.is_finished]

        return outputs, num_processed_tokens, is_prefill

    def get_speculative_stats(self) -> dict:
        if self.spec_decode_coordinator is None:
            return {
                "enabled": False,
                "proposed_tokens": 0,
                "accepted_tokens": 0,
            }
        return {
            "enabled": True,
            **self.spec_decode_coordinator.metrics.snapshot(),
        }


    # add prompt string to the waiting queue by first transforming it to Sequence object
    def add_prompt(self, prompt: str, sampling_params: SamplingParams) -> None:
        self.scheduler.add_sequence(Sequence(token_ids=self.tokenizer.encode(prompt), block_size=self.config['block_size'],sampling_params=sampling_params))

    # given a list of prompts
    # add_prompt for each prompt
    # call step until all sequences are finished
    # return the generated texts
    def generate(self, prompts: list[str], sampling_params: SamplingParams) -> list[str]:
        for prompt in prompts:
            self.add_prompt(prompt, sampling_params)
        generated_tokens = {}
        while not self.scheduler.is_finished():
            start_t = time.time()
            outputs, num_processed_tokens, is_prefill = self.step()
            end_t = time.time()
            running_time = end_t - start_t + 1e-10
            if is_prefill:
                print(num_processed_tokens, 'number of processed tokens', num_processed_tokens/running_time, "tokens/sec during prefilling")
            else:
                print(num_processed_tokens, 'number of processed tokens', num_processed_tokens/running_time, "tokens/sec during decoding")
            generated_tokens.update({seq_id: tokens for seq_id, tokens in outputs})

        generated_tokens = [generated_tokens[seq_id] for seq_id in sorted(generated_tokens.keys())]
        output = {'text': [self.tokenizer.decode(tokens) for tokens in generated_tokens], 'token_ids': generated_tokens}
        return output
