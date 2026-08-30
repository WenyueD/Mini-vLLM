"""CPU pipeline smoke benchmark using a fake repeating-cycle target.

This validates equality, transactions, and metrics. It is not a model benchmark.
"""

import json
import time

from myvllm.engine.block_manager import BlockManager
from myvllm.engine.sequence import Sequence, SequenceStatus
from myvllm.sampling_parameters import SamplingParams
from myvllm.spec_decode import (
    NGramProposer,
    SpecDecodeCoordinator,
    SpeculativeConfig,
    VerificationResult,
)


def next_token(history: list[int]) -> int:
    return (history[-1] + 1) % 5


def verify(batch):
    results = []
    for seq, proposal in zip(batch.sequences, batch.proposals):
        history = list(seq.token_ids)
        target_tokens = []
        for position in range(len(proposal.token_ids) + 1):
            target_tokens.append(next_token(history))
            if position < len(proposal.token_ids):
                history.append(proposal.token_ids[position])
        results.append(VerificationResult(seq.seq_id, tuple(target_tokens)))
    return results


def main(num_tokens: int = 512) -> None:
    prompt = [0, 1, 2, 3, 4, 0]
    target_only = list(prompt)
    for _ in range(num_tokens):
        target_only.append(next_token(target_only))

    seq = Sequence(
        prompt,
        block_size=4,
        sampling_params=SamplingParams(temperature=0, max_tokens=num_tokens),
    )
    manager = BlockManager(num_blocks=256, block_size=4)
    manager.allocate(seq)
    seq.status = SequenceStatus.RUNNING
    seq.mark_computed(len(seq))
    seq.append_token(next_token(seq.token_ids))
    coordinator = SpecDecodeCoordinator(
        SpeculativeConfig(
            enabled=True,
            proposer="ngram",
            num_speculative_tokens=4,
            ngram_min_size=1,
            ngram_max_size=4,
        ),
        NGramProposer(1, 4),
    )

    started = time.perf_counter()
    while not seq.is_finished:
        coordinator.execute([seq], manager, verify, eos_token_id=-1)
    elapsed = time.perf_counter() - started
    assert seq.token_ids == target_only
    stats = coordinator.metrics.snapshot()
    acceptance_lengths = stats.pop("acceptance_lengths")
    stats["mean_acceptance_length"] = sum(acceptance_lengths) / len(acceptance_lengths)
    print(
        json.dumps(
            {
                "generated_tokens": num_tokens,
                "elapsed_seconds": elapsed,
                "committed_tokens_per_second": num_tokens / elapsed,
                **stats,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
