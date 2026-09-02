from __future__ import annotations

import os
from pathlib import Path

from myvllm.models.qwen3_8_config import (
    QWEN35_DEFAULT_HIDDEN_TAPS,
    load_qwen35_config_dict,
    normalize_qwen35_text_config,
)


def main() -> None:
    path = os.environ.get("MINIVLLM_QWEN38_PATH")
    if not path:
        raise SystemExit("Set MINIVLLM_QWEN38_PATH to the local Qwen3.8 snapshot path")
    snapshot = Path(path)
    raw = load_qwen35_config_dict(path)
    cfg = normalize_qwen35_text_config(raw)
    print(f"snapshot={snapshot}")
    print(f"architectures={raw.get('architectures')}")
    print(f"text_layers={cfg.num_hidden_layers}")
    print(f"hidden_size={cfg.hidden_size}")
    print(f"vocab_size={cfg.vocab_size}")
    print(
        f"full_attention_layers={cfg.full_attention_layers[:8]}...{cfg.full_attention_layers[-3:]}"
    )
    print(f"linear_attention_layers_count={len(cfg.linear_attention_layers)}")
    print(f"hidden_taps_zero_based={QWEN35_DEFAULT_HIDDEN_TAPS}")
    print(f"conv_kernel={cfg.linear_conv_kernel_dim}")
    print(f"linear_key_heads={cfg.linear_num_key_heads}")
    print(f"linear_value_heads={cfg.linear_num_value_heads}")


if __name__ == "__main__":
    main()
