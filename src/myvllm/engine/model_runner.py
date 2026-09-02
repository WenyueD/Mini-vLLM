import math
import torch
import pickle
import torch.distributed as dist
from pathlib import Path
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from myvllm.engine.sequence import Sequence
from myvllm.spec_decode.types import (
    SpecDecodeBatch,
    VerificationResult,
)
from myvllm.spec_decode.verification import prepare_verification_input
from myvllm.utils import *
from myvllm.engine.hybrid_cache import HybridCacheManager
from myvllm.models.qwen3_8_config import (
    QWEN35_DEFAULT_HIDDEN_TAPS,
    is_qwen35_config,
    load_qwen35_config_dict,
    normalize_qwen35_text_config,
)


_CONFIG_DTYPES = {
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float32": torch.float32,
    "fp32": torch.float32,
}


def _dtype_from_config(value) -> torch.dtype | None:
    if value is None:
        return None
    if isinstance(value, torch.dtype):
        return value
    try:
        return _CONFIG_DTYPES[str(value)]
    except KeyError as exc:
        raise ValueError(f"Unsupported model dtype: {value}") from exc


class ModelRunner:
    def __init__(self, config: dict, rank: int, event: Event | list[Event]):
        from myvllm.layers.sampler import SamplerLayer
        from myvllm.models.llama import LlamaForCausalLM
        from myvllm.models.qwen3 import Qwen3ForCausalLM
        from myvllm.models.qwen3_8 import Qwen35ForCausalLM

        self.config = config
        self.event = event

        # set distributed config
        self.block_size = config["block_size"]
        self.world_size = config["world_size"]
        self.enforce_eager = config.get("enforce_eager", False)

        self.rank = rank
        self.is_qwen35_target = False
        self.hybrid_cache_manager = None
        dist.init_process_group(
            "nccl", "tcp://localhost:12345", world_size=config["world_size"], rank=rank
        )
        torch.cuda.set_device(rank)

        # set model
        path_str = self.config["model_name_or_path"]
        model_name = Path(path_str).name
        qwen35_text_config = None
        configured_dtype = _dtype_from_config(config.get("dtype"))
        if configured_dtype is not None:
            torch.set_default_dtype(configured_dtype)
        if is_qwen35_config(config) or model_name in {"Qwen3.8-27B", "Qwen3.5-27B"}:
            qwen35_raw_config = config
            if not is_qwen35_config(qwen35_raw_config):
                qwen35_raw_config = load_qwen35_config_dict(path_str)
            qwen35_text_config = normalize_qwen35_text_config(qwen35_raw_config)
            if self.world_size != 1:
                raise NotImplementedError(
                    "Qwen3.5/Qwen3.8 text-only target currently supports "
                    "world_size=1 in Mini-vLLM"
                )
            if config.get("init_on_meta", False):
                with torch.device("meta"):
                    self.model = Qwen35ForCausalLM(qwen35_text_config)
            else:
                self.model = Qwen35ForCausalLM(qwen35_text_config)
            self.config["num_layers"] = qwen35_text_config.num_hidden_layers
            self.config["num_kv_heads"] = qwen35_text_config.num_key_value_heads
            self.config["head_dim"] = qwen35_text_config.head_dim
            self.config["vocab_size"] = qwen35_text_config.vocab_size
            self.config["num_full_attention_layers"] = len(
                qwen35_text_config.full_attention_layers
            )
            self.is_qwen35_target = True
            self.hybrid_cache_manager = HybridCacheManager()
            print(
                "[Rank 0] Qwen3.5/Qwen3.8 text target: "
                f"{qwen35_text_config.num_hidden_layers} layers, "
                f"full_attention={len(qwen35_text_config.full_attention_layers)}, "
                f"linear_attention={len(qwen35_text_config.linear_attention_layers)}, "
                f"hidden_taps={list(QWEN35_DEFAULT_HIDDEN_TAPS)}"
            )
        else:
            match model_name:
                case "Qwen3-0.6B":
                    self.model = Qwen3ForCausalLM(
                        vocab_size=config["vocab_size"],
                        hidden_size=config["hidden_size"],
                        num_heads=config["num_heads"],
                        head_dim=config["head_dim"],
                        scale=config["scale"],
                        num_kv_heads=config["num_kv_heads"],
                        rms_norm_epsilon=config["rms_norm_epsilon"],
                        qkv_bias=config["qkv_bias"],
                        base=config["base"],
                        max_position=config["max_position"],
                        intermediate_size=config["intermediate_size"],
                        ffn_bias=config["ffn_bias"],
                        num_layers=config["num_layers"],
                        tie_word_embeddings=config["tie_word_embeddings"],
                        block_size=self.block_size,
                    )
                case "Llama-3.2-1B-Instruct":
                    self.model = LlamaForCausalLM(
                        vocab_size=config["vocab_size"],
                        hidden_size=config["hidden_size"],
                        head_dim=config["head_dim"],
                        num_qo_heads=config["num_qo_heads"],
                        num_kv_heads=config["num_kv_heads"],
                        has_attn_bias=config["has_attn_bias"],
                        rms_norm_epsilon=config["rms_norm_epsilon"],
                        rope_base=config["rope_base"],
                        max_position_embeddings=config["max_position_embeddings"],
                        intermediate_size=config["intermediate_size"],
                        ffn_bias=config["ffn_bias"],
                        num_layers=config["num_layers"],
                        block_size=self.block_size,
                        tie_word_embeddings=config["tie_word_embeddings"],
                    )
                case _:
                    raise Exception(
                        f"Unsupported model: {config['model_name_or_path']}"
                    )

        # Load weights in GPU (model moved to GPU before loading weights)
        if any(param.is_meta for param in self.model.parameters()):
            self.model.to_empty(device=f"cuda:{rank}")
        else:
            self.model = self.model.cuda(rank)

        # Load pretrained weights if model_name_or_path is provided
        if config.get("model_name_or_path"):
            from myvllm.utils.loader import load_weights_from_checkpoint

            strict_language = bool(config.get("strict_language_weights", True))
            safe_open_device = str(config.get("safe_open_device", "cpu"))
            if safe_open_device == "cuda":
                safe_open_device = f"cuda:{rank}"
            load_weights_from_checkpoint(
                self.model,
                config["model_name_or_path"],
                strict_language=strict_language,
                stream_safetensors=bool(config.get("stream_weights", False)),
                safe_open_device=safe_open_device,
            )

        # Load weights in CPU (move the model to GPU after loading weights)
        # self.model = self.model.cuda(rank)

        self.sampler = SamplerLayer()

        # Store default dtype before it's needed in allocate_kv_cache
        self.default_dtype = next(self.model.parameters()).dtype

        # Debug flag for first decode step
        self._first_decode = False

        # warm up model so that we know peak memory usage
        self.warmup_model()
        # allocate kv cache
        self.allocate_kv_cache()
        # capture cuda graph for decoding
        if not self.enforce_eager:
            self.capture_cudagraph()

        torch.set_default_device(f"cuda:{rank}")
        torch.set_default_dtype(self.default_dtype)

        # IMPORTANT: Set up shared memory and barrier AFTER all model initialization
        # This ensures both ranks complete warmup/allocation before rank 1 enters its event loop
        if self.world_size > 1:
            # Synchronize before setting up shared memory
            dist.barrier()
            if self.rank == 0:
                # Try to clean up existing shared memory first
                try:
                    old_shm = SharedMemory(name="myvllm")
                    old_shm.close()
                    old_shm.unlink()
                except FileNotFoundError:
                    pass  # Doesn't exist, which is fine
                self.shm = SharedMemory(name="myvllm", create=True, size=2**20)
                # Barrier to ensure rank 1 waits until shared memory is created
                dist.barrier()
            else:
                # Wait for rank 0 to create shared memory
                dist.barrier()
                self.shm = SharedMemory(name="myvllm")
                # Don't call self.loop() here - let the spawning code handle it
                # Otherwise we'll be stuck in an infinite loop during __init__

    # only use read when rank != 0
    def read_shm(self):
        assert (
            self.world_size > 1 and self.rank != 0
        ), "read_shm can only be called when world_size > 1 and rank != 0"
        self.event.wait()
        n = int.from_bytes(self.shm.buf[:4], "little")  # read length
        method_name, *args = pickle.loads(self.shm.buf[4 : n + 4])
        self.event.clear()
        return method_name, args

    # only use write when rank == 0
    def write_shm(self, method_name: str, args: tuple):
        assert (
            self.world_size > 1 and self.rank == 0
        ), "write_shm can only be called when world_size > 1 and rank == 0"
        # encode the length first
        # Flatten: (method_name, args) where args is a tuple -> (method_name, *args)
        data = pickle.dumps((method_name, *args))
        n = len(data)
        self.shm.buf[:4] = n.to_bytes(4, "little")
        self.shm.buf[4 : n + 4] = data
        for event in self.event:
            event.set()

    # close shared memory, destroy process group, delete graphs
    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs
            del self.graph_vars
        torch.cuda.synchronize()
        # Check if process group exists before destroying
        if dist.is_initialized():
            dist.destroy_process_group()

    # wait to read method and args from shared memory
    # execute the method with args
    # write results back to shared memory
    def loop(self):
        assert (
            self.world_size > 1 and self.rank != 0
        ), "loop can only be called when world_size > 1 and rank != 0"
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)  # Unpack args when calling
            if method_name == "exit":
                self.exit()
                break

    # will be called by both rank == 0 and rank != 0
    # given method name and args from shared memory
    # execute the method and return results
    def call(self, method_name: str, *args: dict):
        if self.world_size > 1 and self.rank == 0:  # will be called in main engine
            self.write_shm(method_name, args)
        method = getattr(self, method_name, None)
        if method:
            return method(*args)
        raise ValueError(f"Unknown method: {method_name}")

    # cleanup memory
    # compute max number of sequence based on max token and max model length
    # run empty sequence to warm up the model
    # clear memory
    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_tokens = self.config["max_num_batch_tokens"]
        max_model_length = self.config["max_model_length"]
        batch_size = max_tokens // max_model_length
        if self.is_qwen35_target:
            batch_size = 1
        seqs = [
            Sequence(
                token_ids=[0] * max_model_length, block_size=self.config["block_size"]
            )
            for _ in range(batch_size)
        ]
        try:
            self.run(seqs, is_prefill=True)
        finally:
            if self.hybrid_cache_manager is not None:
                for seq in seqs:
                    self.hybrid_cache_manager.release_request(seq.request_id)
        torch.cuda.empty_cache()

    # allocate kv cache memory blocks for model
    def allocate_kv_cache(self):
        # find all available memory
        free_mem, total_mem = torch.cuda.mem_get_info()
        total_free_mem = free_mem * self.config["gpu_memory_utilization"]
        peak_mem_usage = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current_mem_usage = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        # reserve some room for peak memory usage during model execution
        available_mem = total_free_mem - (peak_mem_usage - current_mem_usage)
        target_cache_fraction = float(
            self.config.get("target_cache_memory_fraction", 1.0)
        )
        draft_reservation_fraction = float(
            self.config.get("draft_memory_reservation_fraction", 0.0)
        )
        if draft_reservation_fraction:
            target_cache_fraction = min(
                target_cache_fraction, max(0.0, 1.0 - draft_reservation_fraction)
            )
        available_mem *= target_cache_fraction

        # find parameters to compute kv cache size
        num_layers = self.config.get(
            "num_full_attention_layers", self.config["num_layers"]
        )
        num_kv_heads = self.config["num_kv_heads"] // self.world_size
        head_dim = (
            self.config["head_dim"]
            if "head_dim" in self.config
            else self.config["hidden_size"] // self.config["num_heads"]
        )

        # check whether the current free memory can hold at least one block
        # compute the actual byte required of each block
        block_bytes = (
            self.block_size
            * 2
            * num_layers
            * num_kv_heads
            * head_dim
            * self.default_dtype.itemsize
        )
        num_available_kv_blocks = int(available_mem // block_bytes)
        assert (
            num_available_kv_blocks >= 1
        ), f"Not enough memory to hold at least one block of KV cache on rank {self.rank}"

        # Synchronize max_cached_blocks across all ranks.
        # Each rank independently computed num_available_kv_blocks from its own
        # free GPU memory. Ranks may differ slightly: rank-0 carries extra overhead
        # (NCCL buffers, process-group state) so it often has less free memory than
        # workers. Without sync, the scheduler (which runs only on rank-0) would use
        # rank-0's local value and could allocate more blocks than some rank can hold,
        # causing an OOM on that rank during KV cache writes.
        if self.world_size > 1:
            print(
                f"[Rank {self.rank}] Local max_cached_blocks: {num_available_kv_blocks}"
            )
            per_rank_max_blocks_tensor = torch.tensor(
                num_available_kv_blocks, dtype=torch.long, device=f"cuda:{self.rank}"
            )
            # all_reduce with MIN: every rank learns the most conservative limit,
            # i.e. the block count that even the most memory-constrained rank can serve.
            # This single agreed-upon value is then stored in config so the Scheduler
            # (initialized afterwards on rank-0) never allocates more blocks than any
            # rank can physically hold.
            dist.all_reduce(per_rank_max_blocks_tensor, op=dist.ReduceOp.MIN)
            self.config["max_cached_blocks"] = per_rank_max_blocks_tensor.item()
        else:
            # Single GPU: no cross-rank sync needed; use the local value directly.
            self.config["max_cached_blocks"] = num_available_kv_blocks
        if self.rank == 0:
            print(
                f"[Rank 0] Global max_cached_blocks (min): {self.config['max_cached_blocks']}"
            )

        # allocate max possible kv cache for the model, instead for each sequence
        # this is the key for paged attention: one giant KV cache pool, divided into blocks
        # IMPORTANT: Use zeros() instead of empty() to avoid garbage values
        allocated_kv_cache = torch.zeros(
            2,
            self.config["num_layers"],
            self.config["max_cached_blocks"],
            self.block_size,
            num_kv_heads,
            head_dim,
            device=f"cuda:{self.rank}",
        )
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = allocated_kv_cache[0, layer_id]
                module.v_cache = allocated_kv_cache[1, layer_id]
                layer_id += 1
        if self.rank == 0:
            print(
                f"[Rank 0] KV cache attached to {layer_id} full-attention layers; "
                f"blocks={self.config['max_cached_blocks']}, block_size={self.block_size}, "
                f"target_cache_fraction={target_cache_fraction:.3f}"
            )

    # given seqs
    # prepare the data needed for a prefill forward pass
    # taking prefix cache into consideration:
    # input_ids, positions, cu_seqlens_q/k, slot_mapping (where to write new KV values), block_tables (where to read KV values)
    # cu_seqlens_q = [0, 3, 5, 9]
    #               │  │  │  │
    #               │  │  │  └─ end of seq3 (position 9)
    #               │  │  └──── end of seq2 (position 5)
    #               │  └─────── end of seq1 (position 3)
    #               └────────── start (position 0)
    def prepare_prefill(self, seqs: list[Sequence]) -> torch.Tensor:
        # length: sum of all input_ids after prefix cache
        input_ids = []
        # length: sum of all input_ids after prefix cache
        slot_mappings = []
        # length: num_seqs
        seqlens_q = []
        # length: num_seqs
        seqlens_k = []
        # length: num_seqs + 1
        cu_seqlens_q = [0]
        # length: num_seqs + 1
        cu_seqlens_k = [0]
        # block_tables: num_seqs x num_blocks (padded)
        block_tables = []
        for seq in seqs:
            token_ids = seq.token_ids
            num_cached_tokens = seq.num_cached_tokens
            input_ids.extend(token_ids[num_cached_tokens:])
            seqlens_q.append(len(token_ids) - num_cached_tokens)
            seqlens_k.append(len(token_ids))
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlens_q[-1])
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlens_k[-1])
            if seq.block_table:
                for i, block_id in enumerate(seq.block_table[seq.num_cached_blocks :]):
                    if seq.num_cached_blocks + i != seq.num_blocks - 1:
                        slot_mappings.extend(
                            list(
                                range(
                                    block_id * self.block_size,
                                    (block_id + 1) * self.block_size,
                                )
                            )
                        )
                    else:
                        slot_mappings.extend(
                            list(
                                range(
                                    block_id * self.block_size,
                                    block_id * self.block_size
                                    + seq.last_block_num_tokens,
                                )
                            )
                        )
        if cu_seqlens_q[-1] < cu_seqlens_k[-1]:
            # pad block_tables
            all_block_tables = [seq.block_table for seq in seqs]
            max_num_blocks = max(len(bt) for bt in all_block_tables)
            for i, seq in enumerate(seqs):
                block_table = seq.block_table + [-1] * (
                    max_num_blocks - len(seq.block_table)
                )
                block_tables.append(block_table)
        self._prefill_sample_indices = [end - 1 for end in cu_seqlens_q[1:]]
        input_ids = torch.tensor(input_ids, dtype=torch.long, pin_memory=True).cuda(
            non_blocking=True
        )
        slot_mapping_tensor = torch.tensor(
            slot_mappings, dtype=torch.long, pin_memory=True
        ).cuda(non_blocking=True)

        set_context(
            is_prefill=True,
            cu_seqlens_q=torch.tensor(
                cu_seqlens_q, dtype=torch.int32, pin_memory=True
            ).cuda(non_blocking=True),
            cu_seqlens_k=torch.tensor(
                cu_seqlens_k, dtype=torch.int32, pin_memory=True
            ).cuda(non_blocking=True),
            max_seqlen_q=max(seqlens_q),
            max_seqlen_k=max(seqlens_k),
            slot_mapping=slot_mapping_tensor,
            context_lens=None,
            block_tables=(
                torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(
                    non_blocking=True
                )
                if block_tables
                else None
            ),
        )
        return input_ids

    # prepare input data for decoding
    def prepare_decode(self, seqs: list[Sequence]) -> torch.Tensor:
        input_ids = []
        context_lens = []
        slot_mappings = []
        block_tables = []
        for seq in seqs:
            input_ids.append(seq.last_token)
            context_lens.append(len(seq))
            slot_mappings.append(
                seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1
            )
        all_block_tables = [seq.block_table for seq in seqs]
        max_num_blocks = max(len(bt) for bt in all_block_tables)
        for i, seq in enumerate(seqs):
            block_table = seq.block_table + [-1] * (
                max_num_blocks - len(seq.block_table)
            )
            block_tables.append(block_table)
        input_ids = torch.tensor(input_ids, dtype=torch.long, pin_memory=True).cuda(
            non_blocking=True
        )
        set_context(
            is_prefill=False,
            cu_seqlens_q=None,
            cu_seqlens_k=None,
            max_seqlen_q=0,
            max_seqlen_k=0,
            slot_mapping=torch.tensor(
                slot_mappings, dtype=torch.long, pin_memory=True
            ).cuda(non_blocking=True),
            context_lens=torch.tensor(
                context_lens, dtype=torch.long, pin_memory=True
            ).cuda(non_blocking=True),
            block_tables=(
                torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(
                    non_blocking=True
                )
                if block_tables
                else None
            ),
        )
        return input_ids

    def prepare_speculative(
        self, batch: SpecDecodeBatch
    ) -> tuple[torch.Tensor, tuple[int, ...]]:
        """Prepare variable-q eager verification over existing paged prefixes."""
        plan = prepare_verification_input(batch, self.block_size)

        input_tensor = torch.tensor(
            plan.input_ids, dtype=torch.long, pin_memory=True
        ).cuda(non_blocking=True)
        set_context(
            is_prefill=False,
            cu_seqlens_q=None,
            cu_seqlens_k=None,
            max_seqlen_q=0,
            max_seqlen_k=0,
            slot_mapping=torch.tensor(
                plan.slot_mappings, dtype=torch.long, pin_memory=True
            ).cuda(non_blocking=True),
            context_lens=torch.tensor(
                plan.context_lens, dtype=torch.long, pin_memory=True
            ).cuda(non_blocking=True),
            block_tables=torch.tensor(
                plan.block_tables, dtype=torch.int32, pin_memory=True
            ).cuda(non_blocking=True),
        )
        return input_tensor, plan.query_lengths

    # prepare the temperature
    def prepare_sample(self, seqs: list[Sequence]) -> None:
        return torch.tensor(
            [seq.temperature for seq in seqs], dtype=torch.float32, pin_memory=True
        ).cuda(non_blocking=True)

    # when prefilling, directly compute model forward + logits
    # when decoding, use cuda graph execution to speed up
    # allocate input_ids, positions, slot_mapping, context_lens, block_tables, outputs
    # into graph_variable, and then replay the graph
    @torch.inference_mode()
    def run_model(
        self,
        input_ids: torch.Tensor,
        is_prefill: bool,
        force_eager: bool = False,
        capture_hidden_taps: bool = False,
    ) -> torch.Tensor:
        if is_prefill or self.enforce_eager or force_eager:
            # For varlen prefill, keep input_ids as 1D (concatenated tokens)
            # Do NOT unsqueeze - flash_attn_varlen_func expects 1D input with cu_seqlens
            output = self.model(
                input_ids,
                capture_hidden_taps=capture_hidden_taps,
                return_dict=capture_hidden_taps,
            )
            if capture_hidden_taps:
                return output
            logits = self.model.compute_logits(output)
        else:
            bs = input_ids.size(0)
            context = get_context()

            # finds smallest captured graph that fits the batch size
            graph = self.graphs[next(bs_ for bs_ in self.graphs.keys() if bs_ >= bs)]
            vars = self.graph_vars
            # copy input data into graph variables
            vars["input_ids"][:bs].copy_(input_ids)
            vars["slot_mapping"][:bs].fill_(-1)
            vars["slot_mapping"][:bs].copy_(context.slot_mapping)
            vars["context_lens"].zero_()
            vars["context_lens"][:bs].copy_(context.context_lens)
            vars["block_tables"][
                :bs, : context.block_tables.size(1)
            ] = context.block_tables
            # replay the graph
            graph.replay()
            logits = self.model.compute_logits(vars["outputs"][:bs])

        return logits

    @torch.inference_mode()
    def run_target_with_hidden_taps(
        self,
        input_ids: torch.Tensor,
        is_prefill: bool,
    ):
        """Return logits plus zero-based layer-completion hidden taps."""
        output = self.run_model(
            input_ids,
            is_prefill,
            force_eager=True,
            capture_hidden_taps=True,
        )
        return output

    # prepare prefill
    # prepare sample
    # run model
    # sample logits
    # reset context
    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        if getattr(self, "is_qwen35_target", False):
            try:
                if len(seqs) != 1:
                    raise NotImplementedError(
                        "Qwen3.5/Qwen3.8 cached target runner currently supports B=1"
                    )
                if is_prefill:
                    input_ids = self.prepare_prefill(seqs)
                else:
                    input_ids = self.prepare_decode(seqs)
                output = self.model.forward_with_cache(
                    input_ids.reshape(-1),
                    self.hybrid_cache_manager,
                    seqs[0].request_id,
                )
                logits = output.logits[:, -1, :]
                token_ids = None
                if self.rank == 0:
                    token_ids = self.sampler(logits, self.prepare_sample(seqs))
                return token_ids
            finally:
                reset_context()
        if is_prefill:
            input_ids = self.prepare_prefill(seqs)
        else:
            input_ids = self.prepare_decode(seqs)
        logits = self.run_model(input_ids, is_prefill)
        if is_prefill:
            logits = logits[self._prefill_sample_indices]
        # only sample when rank == 0
        token_ids = None
        if self.rank == 0:
            token_ids = self.sampler(logits, self.prepare_sample(seqs))
        reset_context()
        return token_ids

    def verify(self, batch: SpecDecodeBatch) -> list[VerificationResult]:
        """Greedy K+1 target decisions for each speculative request."""
        try:
            input_ids, query_lengths = self.prepare_speculative(batch)
            logits = self.run_model(input_ids, is_prefill=False, force_eager=True)
            if self.rank != 0:
                return []
            temperatures = torch.zeros(
                logits.shape[0], dtype=torch.float32, device=logits.device
            )
            flat_tokens = self.sampler(logits, temperatures).cpu().tolist()
            results = []
            start = 0
            for seq, query_length in zip(batch.sequences, query_lengths):
                results.append(
                    VerificationResult(
                        sequence_id=seq.seq_id,
                        target_token_ids=tuple(
                            flat_tokens[start : start + query_length]
                        ),
                    )
                )
                start += query_length
            return results
        finally:
            reset_context()

    def release_hybrid_requests(self, request_ids: list[str]) -> None:
        if self.hybrid_cache_manager is None:
            return
        for request_id in request_ids:
            self.hybrid_cache_manager.release_request(request_id)

    # capture the CUDA graph:
    # pre-allocation at maximum sizes: allocated onece and reuse for all graphs
    # capture for different common batch sizes: [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
    # with torch.cuda.graph(graph, self.graph_pool):
    #        run model() and exact sequence of CUDA kernels for running self.model() will be captured
    # (later use graph.replay() to run the captured graph)
    @torch.inference_mode()
    def capture_cudagraph(self) -> None:
        max_bs = self.config["max_num_seqs"]
        max_len = self.config["max_model_length"]
        max_num_blocks = math.ceil(max_len / self.block_size)
        # for decoding, input is always of shape (batch_size, 1)
        input_ids = torch.zeros(max_bs, dtype=torch.long, device=f"cuda:{self.rank}")
        # for paged attention
        # where to write new KV values in the cache
        slot_mapping = torch.zeros(max_bs, dtype=torch.long, device=f"cuda:{self.rank}")
        # how many tokens each sequence has processed
        context_lens = torch.zeros(max_bs, dtype=torch.long, device=f"cuda:{self.rank}")
        # where to read KV values in the cache
        block_tables = torch.zeros(
            max_bs, max_num_blocks, dtype=torch.int32, device=f"cuda:{self.rank}"
        )
        # output logits
        outputs = torch.zeros(
            max_bs, self.config["vocab_size"], device=f"cuda:{self.rank}"
        )

        # graphs to be captured for different batch sizes
        batch_sizes = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        graph_pool = None

        for batch_size in reversed(batch_sizes):
            graph = torch.cuda.CUDAGraph()
            set_context(
                is_prefill=False,
                cu_seqlens_q=None,
                cu_seqlens_k=None,
                max_seqlen_q=0,
                max_seqlen_k=0,
                slot_mapping=slot_mapping[:batch_size],
                context_lens=context_lens[:batch_size],
                block_tables=block_tables[:batch_size],
            )
            outputs[:batch_size] = self.model(input_ids[:batch_size])

            with torch.cuda.graph(graph, graph_pool):
                outputs[:batch_size] = self.model(input_ids[:batch_size])
                if graph_pool is None:
                    graph_pool = graph.pool()
            # store the captured graph
            self.graphs[batch_size] = graph

            # make sure that the capture is done before resetting and next capture
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
