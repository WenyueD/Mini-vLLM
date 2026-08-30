import xxhash
import numpy as np
from collections import deque
from dataclasses import dataclass
from enum import Enum, auto

from myvllm.engine.sequence import Sequence


class ReservationState(Enum):
    ACTIVE = auto()
    COMMITTED = auto()
    ROLLED_BACK = auto()


@dataclass
class KVReservation:
    sequence_id: int
    original_num_blocks: int
    reserved_num_tokens: int
    reserved_num_blocks: int
    state: ReservationState = ReservationState.ACTIVE

class Block:
    def __init__(self, block_id):
        self.block_id = block_id
        self.hash = -1 
        self.ref_count = 0
        self.token_ids = []


    def update(self, h: int, token_ids: list[int]):
        self.hash = h 
        self.token_ids = token_ids

    def reset(self):
        self.hash = -1
        # reset() is only reached via BlockManager._allocate_block, which takes the
        # block off the free list on behalf of exactly one sequence. Allocation is
        # therefore the first reference: leaving this at 0 makes the matching
        # deallocate() drive ref_count to -1, so the block is never freed.
        self.ref_count = 1
        self.token_ids = []

class BlockManager:
    def __init__(self, num_blocks: int, block_size: int):
        # block_size: number of tokens per block
        self.block_size: int = block_size
        # list of all blocks
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        # hash to block id: this is for prefix caching
        self.hash_to_block_id: dict[int, int] = {}
        # free block ids
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        # used block ids
        self.used_block_ids: set[int] = set()
        self._active_reservations: dict[int, KVReservation] = {}

    # given token_ids, compute the hash value
    # use prefix_hash_value to compute the hash in a context-sensitive way
    def compute_hash(self, token_ids: list[int], prefix_hash_value: int) -> int:
        h = xxhash.xxh64()
        if prefix_hash_value != -1:
            h.update(prefix_hash_value.to_bytes(8, 'little'))
        h.update(np.array(token_ids, dtype=np.int32).tobytes())
        return h.intdigest()

    # move this block to used list
    def _allocate_block(self, block_id: int) -> Block:
        block = self.blocks[block_id]
        assert block.ref_count == 0, "Block is already allocated"
        block.reset()
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return block

    def _deallocate_block(self, block_id: int) -> None:
        assert self.blocks[block_id].ref_count == 0, "Block is still in use"
        block = self.blocks[block_id]
        # Clearing token_ids deliberately keeps the prefix cache scoped to blocks that
        # are still referenced: a freed block can no longer match in allocate(), so a
        # cache hit never spans a finished sequence. Reuse across sequences is not
        # enabled yet because the prefill path cannot consume it -- the Triton kernel
        # in layers/attention.py attends only over the K/V computed in that pass and
        # ignores context.block_tables, and qwen3 derives RoPE positions from
        # cu_seqlens_q, so both would be wrong by num_cached_tokens. Enabling reuse
        # means a paged prefill kernel (cu_seqlens_q != cu_seqlens_k) plus a position
        # offset; until then this line is what keeps the engine correct.
        block.token_ids = []
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    # whether we can allocate a block for this sequence
    def can_allocate(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= seq.num_blocks


    def allocate(self, seq: Sequence) -> None:
        h = -1
        for i in range(seq.num_blocks):
            no_cache_found = False

            token_ids = seq.block(i)
            # only compute hash for full blocks, always -1 for partial blocks
            h = self.compute_hash(token_ids=token_ids, prefix_hash_value=h) if len(token_ids) == self.block_size else -1
            block_id = self.hash_to_block_id.get(h, -1)
            
            # if cache miss or hash collision
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                no_cache_found = True

            if not no_cache_found:
                # update sequence information
                seq.num_cached_tokens += self.block_size # which == len(token_ids)
                # update block information, considering the edge case that the block is not allocated yet but with hash code
                if block_id not in self.used_block_ids:
                    # Unreachable while _deallocate_block clears token_ids: a freed
                    # block has token_ids == [] and fails the match above. Kept for
                    # when cross-sequence reuse is enabled.
                    block = self._allocate_block(block_id)
                else:
                    # update block information
                    block = self.blocks[self.hash_to_block_id[h]]
                    block.ref_count += 1
            else:
                # cache miss
                block = self._allocate_block(self.free_block_ids[0])
                block.update(h=h, token_ids=token_ids)
                if h != -1:
                    self.hash_to_block_id[h] = block.block_id
            seq.block_table.append(block.block_id)
        
    def deallocate(self, seq: Sequence) -> None:
        if seq.seq_id in self._active_reservations:
            raise RuntimeError("cannot deallocate a sequence with an active reservation")
        # update block information
        for block_id in seq.block_table:
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        # update sequence information
        seq.block_table = []
        seq.num_cached_tokens = 0

    def _required_blocks(self, num_tokens: int) -> int:
        return (num_tokens + self.block_size - 1) // self.block_size if num_tokens else 0

    def can_reserve(self, seq: Sequence, num_provisional_tokens: int) -> bool:
        """Return whether committed plus provisional positions can be backed.

        Unlike ``can_append``, this method is called before any logical mutation.
        It is therefore the contract used by speculative scheduling.
        """
        if num_provisional_tokens < 0:
            return False
        if seq.block_size != self.block_size:
            raise ValueError("sequence and block manager block sizes must match")
        return self.blocks_needed_for_reserve(seq, num_provisional_tokens) <= len(
            self.free_block_ids
        )

    def blocks_needed_for_reserve(
        self, seq: Sequence, num_provisional_tokens: int
    ) -> int:
        if num_provisional_tokens < 0:
            raise ValueError("provisional token count must be non-negative")
        if seq.block_size != self.block_size:
            raise ValueError("sequence and block manager block sizes must match")
        desired = self._required_blocks(seq.num_tokens + num_provisional_tokens)
        return max(0, desired - len(seq.block_table))

    def reserve(self, seq: Sequence, num_provisional_tokens: int) -> KVReservation:
        """Reserve paged-cache capacity without touching logical token state."""
        if seq.seq_id in self._active_reservations:
            raise RuntimeError("sequence already has an active KV reservation")
        if not self.can_reserve(seq, num_provisional_tokens):
            raise MemoryError(
                f"not enough KV blocks for sequence {seq.seq_id} and "
                f"{num_provisional_tokens} provisional tokens"
            )
        original_num_blocks = len(seq.block_table)
        reserved_num_tokens = seq.num_tokens + num_provisional_tokens
        reserved_num_blocks = self._required_blocks(reserved_num_tokens)
        while len(seq.block_table) < reserved_num_blocks:
            block = self._allocate_block(self.free_block_ids[0])
            # Speculative blocks are deliberately not published to prefix cache.
            seq.block_table.append(block.block_id)
        reservation = KVReservation(
            sequence_id=seq.seq_id,
            original_num_blocks=original_num_blocks,
            reserved_num_tokens=reserved_num_tokens,
            reserved_num_blocks=reserved_num_blocks,
        )
        self._active_reservations[seq.seq_id] = reservation
        return reservation

    def reserve_batch(
        self, requests: list[tuple[Sequence, int]]
    ) -> list[KVReservation]:
        """Atomically reserve a heterogeneous batch or roll it all back."""
        reservations: list[tuple[Sequence, KVReservation]] = []
        try:
            for seq, num_provisional_tokens in requests:
                reservations.append(
                    (seq, self.reserve(seq, num_provisional_tokens))
                )
        except Exception:
            for reserved_seq, reservation in reversed(reservations):
                self.rollback(reserved_seq, reservation)
            raise
        return [reservation for _, reservation in reservations]

    def extend_reservation(
        self,
        seq: Sequence,
        reservation: KVReservation,
        num_provisional_tokens: int,
    ) -> None:
        """Grow a mandatory-token reservation to include a draft proposal."""
        self._require_active_reservation(seq, reservation)
        if num_provisional_tokens < 0:
            raise ValueError("provisional token count must be non-negative")
        desired_tokens = seq.num_tokens + num_provisional_tokens
        desired_blocks = self._required_blocks(desired_tokens)
        additional_blocks = max(0, desired_blocks - len(seq.block_table))
        if additional_blocks > len(self.free_block_ids):
            raise MemoryError("not enough KV blocks to extend reservation")
        while len(seq.block_table) < desired_blocks:
            block = self._allocate_block(self.free_block_ids[0])
            seq.block_table.append(block.block_id)
        reservation.reserved_num_tokens = desired_tokens
        reservation.reserved_num_blocks = desired_blocks

    def commit(
        self,
        seq: Sequence,
        reservation: KVReservation,
        committed_num_tokens: int | None = None,
    ) -> None:
        """Keep blocks for committed tokens and free every reserved tail block."""
        committed_num_tokens = (
            seq.num_tokens if committed_num_tokens is None else committed_num_tokens
        )
        self.validate_commit(seq, reservation, committed_num_tokens, seq.token_ids)
        keep_blocks = self._required_blocks(committed_num_tokens)
        self._release_tail_blocks(seq, keep_blocks)
        self._finalize_committed_blocks(seq, committed_num_tokens, seq.token_ids)
        reservation.state = ReservationState.COMMITTED
        del self._active_reservations[seq.seq_id]

    def validate_commit(
        self,
        seq: Sequence,
        reservation: KVReservation,
        retained_num_tokens: int,
        committed_token_ids: list[int],
    ) -> None:
        """Validate a commit without mutating blocks or transaction state."""
        self._require_active_reservation(seq, reservation)
        if not 0 <= retained_num_tokens <= reservation.reserved_num_tokens:
            raise ValueError("retained KV prefix is outside the reservation")
        if retained_num_tokens > len(committed_token_ids):
            raise ValueError("retained KV cannot exceed committed token metadata")
        keep_blocks = self._required_blocks(retained_num_tokens)
        if keep_blocks > len(seq.block_table):
            raise ValueError("retained KV prefix has no reserved blocks")
        for block_id in seq.block_table[keep_blocks:]:
            if self.blocks[block_id].ref_count <= 0:
                raise AssertionError("reserved tail block has invalid ref_count")

    def rollback(self, seq: Sequence, reservation: KVReservation) -> None:
        """Discard the reservation; stale in-block KV becomes unreachable."""
        self._require_active_reservation(seq, reservation)
        self._release_tail_blocks(seq, reservation.original_num_blocks)
        reservation.state = ReservationState.ROLLED_BACK
        del self._active_reservations[seq.seq_id]

    def slot_for(self, seq: Sequence, logical_token_index: int) -> int:
        if logical_token_index < 0:
            raise IndexError("logical token index must be non-negative")
        logical_block, offset = divmod(logical_token_index, self.block_size)
        if logical_block >= len(seq.block_table):
            raise IndexError("logical token has no reserved KV slot")
        return seq.block_table[logical_block] * self.block_size + offset

    def _release_tail_blocks(self, seq: Sequence, keep_blocks: int) -> None:
        if keep_blocks < 0 or keep_blocks > len(seq.block_table):
            raise ValueError("invalid retained block count")
        while len(seq.block_table) > keep_blocks:
            block_id = seq.block_table.pop()
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count < 0:
                raise AssertionError("KV block ref_count became negative")
            if block.ref_count == 0:
                self._deallocate_block(block_id)

    def _finalize_committed_blocks(
        self,
        seq: Sequence,
        retained_num_tokens: int,
        committed_token_ids: list[int],
    ) -> None:
        """Publish metadata only for complete blocks with valid committed KV."""
        full_blocks = retained_num_tokens // self.block_size
        prefix_hash = -1
        for logical_block in range(full_blocks):
            block_id = seq.block_table[logical_block]
            start = logical_block * self.block_size
            tokens = committed_token_ids[start : start + self.block_size]
            block_hash = self.compute_hash(tokens, prefix_hash)
            block = self.blocks[block_id]
            block.update(block_hash, list(tokens))
            self.hash_to_block_id[block_hash] = block_id
            prefix_hash = block_hash
        if full_blocks < len(seq.block_table):
            partial = self.blocks[seq.block_table[full_blocks]]
            partial.hash = -1
            start = full_blocks * self.block_size
            partial.token_ids = list(committed_token_ids[start:retained_num_tokens])

    def _require_active_reservation(
        self, seq: Sequence, reservation: KVReservation
    ) -> None:
        if reservation.sequence_id != seq.seq_id:
            raise ValueError("reservation belongs to another sequence")
        if reservation.state is not ReservationState.ACTIVE:
            raise RuntimeError("reservation has already been finalized")
        if self._active_reservations.get(seq.seq_id) is not reservation:
            raise RuntimeError("reservation is not active on this block manager")

    def assert_consistent(self) -> None:
        """Expensive CPU invariant check used by randomized tests/benchmarks."""
        assert self.used_block_ids.isdisjoint(self.free_block_ids)
        assert len(self.used_block_ids) + len(self.free_block_ids) == len(self.blocks)
        assert set(self.free_block_ids) | self.used_block_ids == set(range(len(self.blocks)))
        for block in self.blocks:
            if block.block_id in self.used_block_ids:
                assert block.ref_count > 0
            else:
                assert block.ref_count == 0

    # this is to check whether we can append tokens to this sequence
    # when that token would require allocating a new block.
    def can_append(self, seq: Sequence) -> bool:
        # Called after the new token is already counted in seq.num_tokens, so the
        # condition must match append()'s allocation branch: a fresh block is only
        # needed when that token is the first of a new block (num_tokens % size == 1).
        # At num_tokens % size == 0 the token still fits in the block the sequence
        # already holds, and append() merely finalizes its hash.
        if seq.num_tokens % self.block_size == 1:
            return len(self.free_block_ids) > 0
        return True

    # this is the actual work to append tokens to this sequence
    # this is called when the new token has been added to the seq information
    # but no block in gpu has yet allocate for it
    def append(self, seq: Sequence) -> None:
        block_tables = seq.block_table
        last_block_for_seq_id = block_tables[-1]

        # if the last block is now full, compute hash
        if seq.num_tokens % self.block_size == 0:
            h = self.compute_hash(token_ids = seq.block(seq.num_blocks - 1), prefix_hash_value = -1 if len(block_tables) == 1 else self.blocks[block_tables[-2]].hash)
            block = self.blocks[last_block_for_seq_id]
            block.update(h=h, token_ids=seq.block(seq.num_blocks - 1))
            self.hash_to_block_id[h] = block.block_id
        # if one new block is needed
        elif seq.num_tokens % self.block_size == 1:
            # Previous block should be finalized
            assert self.blocks[last_block_for_seq_id].hash != -1
            block = self._allocate_block(self.free_block_ids[0])
            block_tables.append(block.block_id)
        # else, do nothing
        else:
            assert last_block_for_seq_id in self.used_block_ids, "Last block should be allocated"
            assert self.blocks[last_block_for_seq_id].hash == -1, "Last block should be partial block with hash -1"
