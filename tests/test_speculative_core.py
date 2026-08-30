import torch

from myvllm.engine.sequence import Sequence
from myvllm.sampling import logits_to_token_ids
from myvllm.sampling_parameters import SamplingParams
from myvllm.spec_decode import (
    MockProposer,
    NGramProposer,
    Proposal,
    RejectionSampler,
    SpeculativeConfig,
    VerificationResult,
)


def test_temperature_zero_is_legal_and_exact_argmax():
    params = SamplingParams(temperature=0)
    assert params.temperature == 0
    logits = torch.tensor([[1.0, 5.0, 2.0], [3.0, -1.0, 4.0]])
    temperatures = torch.zeros(2)
    expected = torch.tensor([1, 2])
    assert torch.equal(logits_to_token_ids(logits, temperatures), expected)


def test_positive_temperature_path_remains_available():
    params = SamplingParams(temperature=1.0)
    logits = torch.tensor([[float("-inf"), 0.0, float("-inf")]])
    assert logits_to_token_ids(logits, torch.tensor([params.temperature])).item() == 1


def test_ngram_proposer_uses_longest_recent_prompt_match():
    seq = Sequence([7, 8, 9, 10, 7, 8], block_size=4)
    proposal = NGramProposer(2, 4).propose(seq, max_tokens=3)
    assert proposal.token_ids == (9, 10, 7)
    assert proposal.metadata["match"] == 2


def test_ngram_no_match_and_mock_variable_lengths():
    first = Sequence([1, 2, 3], block_size=4)
    second = Sequence([4, 5], block_size=4)
    assert NGramProposer(2, 3).propose(first, 4).token_ids == ()
    mock = MockProposer({first.seq_id: [8, 9], second.seq_id: [6]})
    assert mock.propose(first, 4).token_ids == (8, 9)
    assert mock.propose(second, 4).token_ids == (6,)


def sample(draft, target):
    return RejectionSampler().sample(
        Proposal(4, tuple(draft)), VerificationResult(4, tuple(target))
    )


def test_greedy_rejection_boundaries_and_bonus():
    first = sample([1, 2, 3], [9, 8, 7, 6])
    assert first.accepted_proposal_tokens == 0
    assert first.emitted_token_ids == (9,)
    assert first.per_position_acceptance == (False, False, False)
    assert not first.used_bonus_token

    middle = sample([1, 2, 3], [1, 8, 7, 6])
    assert middle.accepted_proposal_tokens == 1
    assert middle.emitted_token_ids == (1, 8)
    assert middle.per_position_acceptance == (True, False, False)

    all_accepted = sample([1, 2, 3], [1, 2, 3, 4])
    assert all_accepted.accepted_proposal_tokens == 3
    assert all_accepted.emitted_token_ids == (1, 2, 3, 4)
    assert all_accepted.used_bonus_token

    empty = sample([], [5])
    assert empty.emitted_token_ids == (5,)
    assert empty.used_bonus_token


def test_speculative_config_validation():
    assert SpeculativeConfig.from_value({"enabled": True}).enabled
    try:
        SpeculativeConfig(num_speculative_tokens=-1)
    except ValueError:
        pass
    else:
        raise AssertionError("negative K must fail")
