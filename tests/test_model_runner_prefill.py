import torch

from myvllm.engine.model_runner import ModelRunner


class CapturingSampler:
    def __init__(self):
        self.logits = None

    def __call__(self, logits, temperatures):
        self.logits = logits.clone()
        return logits.argmax(dim=-1)


def run_prefill(indices, logits, num_sequences):
    runner = ModelRunner.__new__(ModelRunner)
    runner.rank = 0
    runner._prefill_sample_indices = indices
    runner.sampler = CapturingSampler()
    runner.prepare_prefill = lambda sequences: torch.tensor([99])
    runner.run_model = lambda input_ids, is_prefill: logits
    runner.prepare_sample = lambda sequences: torch.zeros(len(sequences))
    outputs = runner.run([object()] * num_sequences, is_prefill=True)
    return outputs, runner.sampler.logits


def test_ragged_prefill_selects_one_last_prompt_logit_per_request():
    # q_len=[2,3] is flattened into five rows. Request decisions come from the
    # final row of each segment, indices 1 and 4, not from all five token rows.
    logits = torch.tensor(
        [
            [10.0, 0.0, 0.0],
            [0.0, 20.0, 0.0],
            [30.0, 0.0, 0.0],
            [0.0, 0.0, 40.0],
            [50.0, 0.0, 0.0],
        ]
    )
    outputs, sampled_logits = run_prefill([1, 4], logits, num_sequences=2)
    assert torch.equal(sampled_logits, logits[[1, 4]])
    assert outputs.tolist() == [1, 0]
    assert outputs.numel() == 2


def test_single_request_prefill_selects_its_last_row():
    logits = torch.tensor([[1.0, 0.0], [0.0, 2.0], [3.0, 0.0]])
    outputs, sampled_logits = run_prefill([2], logits, num_sequences=1)
    assert torch.equal(sampled_logits, logits[[2]])
    assert outputs.tolist() == [0]
