"""Opt-in real-GPU equivalence test; requires local Qwen3-0.6B weights."""

import gc
import os

import pytest
import torch
from transformers import AutoConfig

from myvllm.engine.llm_engine import LLMEngine
from myvllm.sampling_parameters import SamplingParams
from myvllm.spec_decode import Proposal, Proposer


MODEL_PATH = os.environ.get("MINIVLLM_QWEN3_PATH")
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not MODEL_PATH,
    reason="requires CUDA and MINIVLLM_QWEN3_PATH",
)


def engine_config(speculative: bool) -> dict:
    hf = AutoConfig.from_pretrained(MODEL_PATH, local_files_only=True)
    return {
        "max_num_sequences": 2,
        "max_num_batched_tokens": 256,
        "max_cached_blocks": 128,
        "block_size": 16,
        "world_size": 1,
        "model_name_or_path": MODEL_PATH,
        "enforce_eager": True,
        "vocab_size": hf.vocab_size,
        "hidden_size": hf.hidden_size,
        "num_heads": hf.num_attention_heads,
        "head_dim": getattr(hf, "head_dim", hf.hidden_size // hf.num_attention_heads),
        "num_kv_heads": hf.num_key_value_heads,
        "intermediate_size": hf.intermediate_size,
        "num_layers": hf.num_hidden_layers,
        "tie_word_embeddings": hf.tie_word_embeddings,
        "base": hf.rope_theta,
        "rms_norm_epsilon": hf.rms_norm_eps,
        "qkv_bias": False,
        "scale": 1,
        "max_position": hf.max_position_embeddings,
        "ffn_bias": False,
        "max_num_batch_tokens": 512,
        "max_model_length": 256,
        "gpu_memory_utilization": 0.75,
        "eos": hf.eos_token_id,
        "speculative_config": {
            "enabled": speculative,
            "proposer": "ngram",
            "num_speculative_tokens": 4,
            "ngram_min_size": 1,
            "ngram_max_size": 4,
        },
    }


PROMPT = "red blue red blue red blue; continue the pattern:"


class ReplayProposer(Proposer):
    """Replay target-only output so real target verification must accept K > 0.

    Prefill has already committed completion token 0 when the proposer first runs.
    Therefore ``num_completion_tokens`` is exactly the index of the next baseline
    token to propose.
    """

    def __init__(self, baseline_token_ids: list[int]):
        self.baseline_token_ids = baseline_token_ids

    def propose(self, sequence, max_tokens: int) -> Proposal:
        start = sequence.num_completion_tokens
        tokens = tuple(self.baseline_token_ids[start : start + max_tokens])
        return Proposal(
            sequence_id=sequence.seq_id,
            token_ids=tokens,
            metadata={"kind": "replay-oracle", "baseline_start": start},
        )


def generate(speculative: bool, replay_tokens: list[int] | None = None):
    engine = LLMEngine(engine_config(speculative))
    try:
        if replay_tokens is not None:
            assert engine.spec_decode_coordinator is not None
            engine.spec_decode_coordinator.proposer = ReplayProposer(replay_tokens)
        result = engine.generate(
            [PROMPT],
            SamplingParams(
                temperature=0,
                max_tokens=32,
                ignore_eos=True,
                max_model_length=256,
            ),
        )
        return result["token_ids"], engine.get_speculative_stats()
    finally:
        engine.exit()
        gc.collect()
        torch.cuda.empty_cache()


@pytest.fixture(scope="module")
def baseline_tokens():
    token_ids, _ = generate(False)
    return token_ids


def test_qwen3_real_k_plus_one_verification_is_lossless(baseline_tokens):
    assert len(baseline_tokens) == 1
    speculative_tokens, stats = generate(True, replay_tokens=baseline_tokens[0])
    assert speculative_tokens == baseline_tokens
    # These assertions make a K=0-only run fail even if output equality holds.
    assert stats["proposed_tokens"] > 0
    assert stats["accepted_tokens"] > 0
    assert max(stats["acceptance_lengths"], default=0) > 1
    assert len(stats["per_position_proposed"]) > 1
    assert stats["per_position_proposed"][1] > 0
    print(stats)


def test_qwen3_ngram_speculative_smoke_is_lossless(baseline_tokens):
    speculative_tokens, stats = generate(True)
    assert speculative_tokens == baseline_tokens
    # Prompt lookup may legitimately find no continuation for this tokenizer and
    # prompt. The replay test above, not this smoke test, enforces real K > 0.
    print(stats)
