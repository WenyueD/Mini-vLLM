from __future__ import annotations

import os
from pathlib import Path

from myvllm.spec_decode.dflash2 import (
    DFlash2Config,
    audit_dflash2_checkpoint,
    expected_dflash2_weight_shapes,
)


def main() -> None:
    path = os.environ.get("MINIVLLM_DFLASH2_PATH")
    if not path:
        raise SystemExit(
            "Set MINIVLLM_DFLASH2_PATH to the local "
            "incoai/Qwen3.8-27B-DFlash2 snapshot path"
        )
    snapshot = Path(path).expanduser()
    config = DFlash2Config.from_pretrained_path(str(snapshot))
    config.validate_contract()
    expected = expected_dflash2_weight_shapes(config)
    audit = audit_dflash2_checkpoint(str(snapshot), config)

    print(f"snapshot={snapshot}")
    print(f"block_size={config.block_size}")
    print(f"num_speculative_tokens={config.num_speculative_tokens}")
    print(f"target_layer_ids_zero_based={config.target_layer_ids}")
    print(f"selector_top_k={config.selector_top_k}")
    print(f"selector_rank={config.selector_rank}")
    print(f"conv_kernel_size={config.conv_kernel_size}")
    print(f"conv_group_size={config.conv_group_size}")
    print(f"hidden_size={config.hidden_size}")
    print(f"vocab_size={config.vocab_size}")
    print(f"sliding_attention_layers={config.num_sliding_attention_layers}")
    print(f"sliding_window={config.sliding_window}")
    print(f"checkpoint_tensors={audit.checkpoint_tensors}")
    print(f"expected_tensors={audit.expected_tensors}")
    print(f"dtype_counts={audit.dtype_counts}")
    print(f"missing_weights={len(audit.missing_weights)}")
    print(f"unknown_weights={len(audit.unknown_weights)}")
    print(f"shape_mismatches={len(audit.shape_mismatches)}")

    print("\nexpected first 20:")
    for name in sorted(expected)[:20]:
        print(f"  {name}: {expected[name]}")

    for title, values in [
        ("missing", audit.missing_weights),
        ("unknown", audit.unknown_weights),
    ]:
        if values:
            print(f"\n{title} first 30:")
            for name in values[:30]:
                print(f"  {name}")

    if audit.shape_mismatches:
        print("\nshape mismatches first 30:")
        for name, expected_shape, actual_shape in audit.shape_mismatches[:30]:
            print(f"  {name}: expected={expected_shape} checkpoint={actual_shape}")

    if not audit.passed:
        raise SystemExit(1)
    print("dflash2_weight_name_audit=PASS")


if __name__ == "__main__":
    main()
