from dataclasses import dataclass

from myvllm.spec_decode.types import SpecDecodeBatch


@dataclass(frozen=True)
class VerificationInput:
    input_ids: tuple[int, ...]
    context_lens: tuple[int, ...]
    slot_mappings: tuple[int, ...]
    block_tables: tuple[tuple[int, ...], ...]
    query_lengths: tuple[int, ...]


def prepare_verification_input(
    batch: SpecDecodeBatch, block_size: int
) -> VerificationInput:
    """Pure CPU planner for heterogeneous K+1 paged target verification."""
    if not batch.sequences:
        return VerificationInput((), (), (), (), ())
    input_ids = []
    context_lens = []
    slot_mappings = []
    block_tables = []
    max_num_blocks = max(len(seq.block_table) for seq in batch.sequences)
    for seq, proposal in zip(batch.sequences, batch.proposals):
        if seq.num_tokens - seq.num_computed_tokens != 1:
            raise RuntimeError("verification requires exactly one pending token")
        if tuple(seq.provisional_token_ids) != proposal.token_ids:
            raise RuntimeError("sequence/proposal provisional state mismatch")
        query_tokens = [seq.last_token, *proposal.token_ids]
        padded_table = tuple(
            seq.block_table + [-1] * (max_num_blocks - len(seq.block_table))
        )
        for offset, token_id in enumerate(query_tokens):
            logical_position = seq.num_computed_tokens + offset
            logical_block, block_offset = divmod(logical_position, block_size)
            if logical_block >= len(seq.block_table):
                raise RuntimeError("verification query has no reserved KV slot")
            input_ids.append(token_id)
            context_lens.append(logical_position + 1)
            slot_mappings.append(
                seq.block_table[logical_block] * block_size + block_offset
            )
            block_tables.append(padded_table)
    return VerificationInput(
        input_ids=tuple(input_ids),
        context_lens=tuple(context_lens),
        slot_mappings=tuple(slot_mappings),
        block_tables=tuple(block_tables),
        query_lengths=batch.query_lengths,
    )
