from types import SimpleNamespace

import pytest
import torch
from torch import nn

from myvllm.models.qwen3_8 import Qwen35RotaryEmbedding
from myvllm.spec_decode.dflash2 import DFlash2Attention


@pytest.mark.parametrize("kind", ["target", "draft"])
def test_rope_survives_meta_materialization_and_dtype_conversion(kind):
    config = SimpleNamespace(
        rotary_dim=64, rope_theta=10000000.0, head_dim=64,
        num_attention_heads=2, num_key_value_heads=1, hidden_size=128,
        sliding_window=2048, rms_norm_eps=1e-6,
    )
    cls = Qwen35RotaryEmbedding if kind == "target" else DFlash2Attention
    reference = cls(config)
    with torch.device("meta"):
        model = nn.Sequential(cls(config))
    model.to_empty(device="cpu")
    rope = model[0]
    expected = reference.inv_freq.clone()
    torch.testing.assert_close(rope.inv_freq, expected, rtol=0, atol=0)
    assert "inv_freq" not in rope.state_dict()

    positions = torch.tensor([[0, 1, 122, 129, 10000]])
    for dtype in (torch.bfloat16, torch.float16, torch.float32):
        model.to(dtype=dtype)
        assert rope.inv_freq.dtype == torch.float32
        torch.testing.assert_close(rope.inv_freq, expected, rtol=0, atol=0)
        actual = (rope(positions, dtype) if kind == "target"
                  else rope._position_embedding(positions, dtype))
        wanted = (reference(positions, dtype) if kind == "target"
                  else reference._position_embedding(positions, dtype))
        for value, expected_value in zip(actual, wanted):
            torch.testing.assert_close(value, expected_value, rtol=0, atol=0)
