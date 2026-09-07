import pytest

import scripts.export_oracle_dflash2_fixture as oracle_exporter
from scripts.compare_dflash2_oracle_fixture import (
    oracle_effective_anchor_mismatch,
    oracle_target_bonus_contract_error,
)
from scripts.export_oracle_dflash2_fixture import (
    build_llm_kwargs,
    completion_output_token_ids,
    dflash_capture_contract_error,
    parse_args,
    qualified_type_name,
    select_target_bonus_capture,
    tensor_attr_to_list,
    tensor_rows_to_list,
)


class ExportArgs:
    target_model = "target"
    tensor_parallel_size = 1
    gpu_memory_utilization = 0.9
    max_model_len = 256
    trust_remote_code = True
    enforce_eager = True
    language_model_only = True
    extra_llm_kwargs = "{}"
    dtype = "bfloat16"
    method = "dflash"
    draft_model = "draft"
    num_speculative_tokens = 7


class CpuOnlyTensor:
    def __init__(self, value):
        self.value = value
        self.detached = False
        self.moved_to_cpu = False

    def detach(self):
        self.detached = True
        return self

    def cpu(self):
        self.moved_to_cpu = True
        return self

    def tolist(self):
        assert self.moved_to_cpu
        return self.value

    def __getitem__(self, key):
        raise AssertionError("source tensor must not be indexed before CPU conversion")


class Completion:
    def __init__(self, token_ids):
        self.token_ids = token_ids


class RequestOutput:
    def __init__(self, token_ids):
        self.outputs = [Completion(token_ids)]


def test_tensor_rows_selects_sparse_active_slots_after_cpu_copy():
    value = CpuOnlyTensor([[0], [11], [22]])

    assert tensor_rows_to_list(value, 1, indices=[2]) == [[22]]
    assert value.detached
    assert value.moved_to_cpu


def test_completion_output_does_not_strip_matching_prompt_prefix():
    output = RequestOutput([9419, 11])

    assert completion_output_token_ids(output, [9419]) == [9419, 11]


def test_build_llm_kwargs_enables_language_model_only():
    kwargs = build_llm_kwargs(ExportArgs(), speculative=True)

    assert kwargs["language_model_only"] is True
    assert kwargs["speculative_config"] == {
        "method": "dflash",
        "model": "draft",
        "num_speculative_tokens": 7,
    }


def test_parse_args_accepts_target_preflight_only(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "export_oracle_dflash2_fixture.py",
            "--output",
            "unused.json",
            "--target-preflight-only",
        ],
    )

    args = parse_args()

    assert args.target_preflight_only is True
    assert args.skip_target_preflight is False


def test_main_rejects_conflicting_target_preflight_modes(monkeypatch):
    args = type(
        "Args",
        (),
        {
            "target_model": "target",
            "draft_model": "draft",
            "target_preflight_only": True,
            "skip_target_preflight": True,
        },
    )()
    monkeypatch.setattr(oracle_exporter, "parse_args", lambda: args)

    with pytest.raises(SystemExit, match="mutually exclusive"):
        oracle_exporter.main()


def test_qualified_type_name_reports_runtime_subclass():
    class RuntimeDFlash2Speculator:
        pass

    assert qualified_type_name(RuntimeDFlash2Speculator()).endswith(
        ".RuntimeDFlash2Speculator"
    )


def test_tensor_rows_falls_back_for_compact_per_step_values():
    assert tensor_rows_to_list([[11]], 1, indices=[1022]) == [[11]]


def test_tensor_attr_selects_sparse_active_slots():
    owner = type("Owner", (), {"cached": [[0], [11], [22]]})()

    assert tensor_attr_to_list(owner, "cached", 1, indices=[1]) == [[11]]


def test_oracle_effective_anchor_mismatch_detects_overridden_fixture_anchor():
    assert oracle_effective_anchor_mismatch(
        {"anchor_token_id": 11, "oracle_capture_anchor_token_id": 9419}
    )
    assert not oracle_effective_anchor_mismatch(
        {"anchor_token_id": 11, "oracle_capture_anchor_token_id": 11}
    )


def test_select_target_bonus_capture_skips_prompt_anchor_phase():
    captures = [
        {"num_sampled": [1], "anchor_token_id": 9419},
        {"num_sampled": [1], "anchor_token_id": 11},
    ]

    assert select_target_bonus_capture(
        captures,
        expected_target_first_token_id=11,
        allow_prefill_anchor=False,
    ) == (1, True)


def test_select_target_bonus_capture_reports_diagnostic_fallback():
    captures = [{"num_sampled": [1], "anchor_token_id": 9419}]

    assert select_target_bonus_capture(
        captures,
        expected_target_first_token_id=11,
        allow_prefill_anchor=False,
    ) == (0, False)


def test_select_target_bonus_capture_rejects_prefill_by_default():
    captures = [{"num_sampled": [0], "anchor_token_id": 9419}]

    assert select_target_bonus_capture(
        captures,
        expected_target_first_token_id=11,
        allow_prefill_anchor=False,
    ) == (None, False)


def test_oracle_target_bonus_contract_accepts_formal_fixture():
    fixture = {
        "anchor_token_id": 11,
        "oracle_capture_anchor_token_id": 11,
        "oracle_target_bonus_capture_matched": True,
        "oracle_runtime": {
            "sample_from_anchor": False,
            "sample_from_anchor_attr_present": True,
        },
    }

    assert oracle_target_bonus_contract_error(fixture) is None


def test_oracle_target_bonus_contract_rejects_legacy_anchor_sampling():
    fixture = {
        "anchor_token_id": 11,
        "oracle_capture_anchor_token_id": 9419,
        "oracle_target_bonus_capture_matched": False,
        "oracle_runtime": {
            "sample_from_anchor": None,
            "sample_from_anchor_attr_present": False,
        },
    }

    assert (
        oracle_target_bonus_contract_error(fixture)
        == "target_bonus_capture_not_matched"
    )


def test_export_contract_requires_explicit_sample_from_anchor_false():
    assert (
        dflash_capture_contract_error(
            {"sample_from_anchor_attr_present": False},
            target_bonus_capture_matched=True,
        )
        == "sample_from_anchor_contract_unknown"
    )
    assert (
        dflash_capture_contract_error(
            {
                "sample_from_anchor_attr_present": True,
                "sample_from_anchor": True,
            },
            target_bonus_capture_matched=True,
        )
        == "sample_from_anchor_must_be_false"
    )
    assert (
        dflash_capture_contract_error(
            {
                "sample_from_anchor_attr_present": True,
                "sample_from_anchor": False,
            },
            target_bonus_capture_matched=True,
        )
        is None
    )
