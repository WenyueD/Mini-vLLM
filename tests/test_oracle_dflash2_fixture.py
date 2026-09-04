from scripts.compare_dflash2_oracle_fixture import oracle_effective_anchor_mismatch
from scripts.export_oracle_dflash2_fixture import tensor_attr_to_list, tensor_rows_to_list


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


def test_tensor_rows_selects_sparse_active_slots_after_cpu_copy():
    value = CpuOnlyTensor([[0], [11], [22]])

    assert tensor_rows_to_list(value, 1, indices=[2]) == [[22]]
    assert value.detached
    assert value.moved_to_cpu


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
