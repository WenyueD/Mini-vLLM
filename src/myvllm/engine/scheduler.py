from collections import deque
from myvllm.engine.sequence import Sequence, SequenceStatus
from myvllm.engine.block_manager import BlockManager


class Scheduler:
    def __init__(self, max_num_sequences: int, max_num_batched_tokens: int, max_cached_blocks: int, block_size: int, eos: int):
        # block manager
        self.block_manager = BlockManager(max_cached_blocks, block_size)
        self.max_num_batched_tokens = max_num_batched_tokens
        self.max_num_sequences = max_num_sequences
        # sequence queue
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        self.eos = eos


    def is_finished(self):
        return len(self.waiting) == 0 and len(self.running) == 0
    
    def add_sequence(self, sequence: Sequence):
        # Reject up front what the block manager could never satisfy, otherwise the
        # sequence sits in `waiting` forever and only surfaces as a stalled engine.
        capacity = len(self.block_manager.blocks)
        if sequence.num_blocks > capacity:
            raise ValueError(
                f"Sequence {sequence.seq_id} needs {sequence.num_blocks} blocks "
                f"({len(sequence)} tokens at block_size={self.block_manager.block_size}) "
                f"but the KV cache only holds {capacity}. "
                f"Raise max_cached_blocks or block_size, or shorten the prompt."
            )
        self.waiting.append(sequence)


    def schedule(self, speculative: bool = False) -> tuple[list[Sequence], bool]:
        scheduled_sequences = []
        current_scheduled_tokens = 0
        # An empty schedule is only legitimate when this call freed blocks by
        # preempting, so the next call can make progress. See the guard below.
        preempted = False
        # try schedule for prefilling from waiting queue if not exceeding limits
        while self.waiting and len(scheduled_sequences) < self.max_num_sequences:
            seq = self.waiting[0]
            if self.block_manager.can_allocate(seq) and len(seq) + current_scheduled_tokens <= self.max_num_batched_tokens:
                seq = self.waiting.popleft() # remove from waiting
                self.block_manager.allocate(seq)
                seq.status = SequenceStatus.RUNNING
                self.running.append(seq)
                scheduled_sequences.append(seq)
                current_scheduled_tokens += len(seq)
            else:
                break
        if scheduled_sequences:
            return scheduled_sequences, True
        
        # try schedule for completion from running queue
        speculative_blocks_claimed = 0
        while self.running:
            seq = self.running.popleft()
            # Ordinary decode historically allocates the pending token's block
            # immediately before execution. Speculative decode must be selected
            # without mutation; its coordinator owns reserve/commit/rollback.
            use_speculation = speculative and seq.is_greedy
            if use_speculation:
                needed = self.block_manager.blocks_needed_for_reserve(seq, 0)
                can_schedule = needed <= (
                    len(self.block_manager.free_block_ids) - speculative_blocks_claimed
                )
            else:
                needed = int(seq.num_tokens % self.block_manager.block_size == 1)
                can_schedule = self.block_manager.can_append(seq) and needed <= (
                    len(self.block_manager.free_block_ids)
                    - speculative_blocks_claimed
                )
            if not can_schedule:
                preempted = True
                if self.running:
                    self.running.appendleft(seq)
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                if current_scheduled_tokens >= self.max_num_batched_tokens or len(scheduled_sequences) >= self.max_num_sequences:
                    self.running.appendleft(seq)
                    break
                if not use_speculation:
                    self.block_manager.append(seq)
                else:
                    speculative_blocks_claimed += needed
                scheduled_sequences.append(seq)
                current_scheduled_tokens += 1 # only one token for completion

        # re-add to running queue in the same order
        if scheduled_sequences:
            self.running.extendleft(reversed(scheduled_sequences))
        elif not preempted and (self.waiting or self.running):
            # Nothing was scheduled and nothing was preempted, so no engine state
            # changed: every later schedule() would take the same decisions and
            # LLMEngine.generate() would spin forever. Fail loudly instead.
            raise RuntimeError(
                "Scheduler made no progress: "
                f"{len(self.waiting)} waiting and {len(self.running)} running sequences, "
                f"{len(self.block_manager.free_block_ids)} of "
                f"{len(self.block_manager.blocks)} blocks free. "
                "This means either a sequence that cannot fit in the KV cache, or "
                "blocks leaked because their ref_count never returned to 0."
            )

        return scheduled_sequences, False


    def preempt(self, seq: Sequence) -> None:
        self.block_manager.deallocate(seq)
        seq.status = SequenceStatus.WAITING
        self.waiting.appendleft(seq)        

    def abort(self, seq: Sequence) -> bool:
        """Remove a request and release all provisional and committed KV state."""
        if seq.is_finished:
            return False
        found = False
        if seq in self.waiting:
            self.waiting.remove(seq)
            found = True
        if seq in self.running:
            self.running.remove(seq)
            found = True
        if seq.active_transaction is not None:
            transaction = seq.active_transaction
            seq.rollback(transaction)
            found = True
        if self.block_manager.rollback_active(seq):
            found = True
        if seq.block_table:
            self.block_manager.deallocate(seq)
            found = True
        seq.status = SequenceStatus.ABORTED
        seq.finish_reason = "cancelled"
        return found


    # postprocess after generation to check whether sequences are finished
    # if finished, deallocate blocks
    def postprocess(
        self,
        seqs: list[Sequence],
        token_ids: list[int],
        is_prefill: bool = False,
    ) -> None:
        for seq, token_id in zip(seqs, token_ids):
            if is_prefill:
                seq.mark_computed(len(seq))
            else:
                seq.mark_computed(seq.num_computed_tokens + 1)
            no_completion_budget = seq.num_completion_tokens >= seq.max_tokens
            at_model_limit = (
                seq.max_model_length is not None
                and seq.num_tokens >= seq.max_model_length
            )
            if no_completion_budget or at_model_limit:
                seq.status = SequenceStatus.FINISHED
                seq.finish_reason = "length"
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
                continue
            seq.append_token(token_id)
            # Check stopping conditions:
            # EOS token
            # Reached max_tokens limit (number of completion tokens)
            # Reached max_model_length limit (total sequence length including prompt)
            stop_due_to_eos = not seq.ignore_eos and token_id == self.eos
            stop_due_to_max_tokens = seq.num_completion_tokens >= seq.max_tokens
            stop_due_to_max_length = seq.max_model_length is not None and seq.num_tokens >= seq.max_model_length

            if stop_due_to_eos or stop_due_to_max_tokens or stop_due_to_max_length:
                seq.status = SequenceStatus.FINISHED
                seq.finish_reason = "eos" if stop_due_to_eos else "length"
                self.block_manager.deallocate(seq)
                self.running.remove(seq)

    def finalize_speculative(self, seqs: list[Sequence]) -> None:
        """Remove/deallocate requests whose coordinator applied a stop rule."""
        for seq in seqs:
            if seq.is_finished:
                self.block_manager.deallocate(seq)
                if seq in self.running:
                    self.running.remove(seq)
