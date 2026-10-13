import math
import torch
import torch.nn.functional as F
import pickle
import torch.distributed as dist
from dataclasses import dataclass
from pathlib import Path
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory
from time import perf_counter
from typing import Any

from myvllm.engine.sequence import Sequence
from myvllm.spec_decode.types import (
    Proposal,
    SpecDecodeBatch,
    VerificationResult,
)
from myvllm.spec_decode.verification import prepare_verification_input
from myvllm.utils import *
from myvllm.engine.hybrid_cache import HybridCacheManager, HybridCacheTransaction
from myvllm.engine.memory import MemoryTrace, module_parameter_bytes
from myvllm.models.qwen3_8_config import (
    QWEN35_DEFAULT_HIDDEN_TAPS,
    is_qwen35_config,
    load_qwen35_config_dict,
    normalize_qwen35_text_config,
)
from myvllm.spec_decode.dflash2 import (
    DFlash2Config,
    DFlash2DraftCache,
    DFlash2DraftModel,
    load_dflash2_weights_from_safetensors,
)
from myvllm.sampling import greedy_token_ids


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


@dataclass(frozen=True)
class PendingQwen35Verification:
    transaction: HybridCacheTransaction
    request_id: str
    hidden_taps: dict[int, torch.Tensor]
    target_token_ids: tuple[int, ...]
    start_position: int
    query_tokens: tuple[int, ...]
    executed_tokens: int = 0


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
        self._pending_verified_cache_transactions = {}
        self._qwen35_verification_stats = {
            "calls": 0,
            "max_batch_size": 0,
            "tensor_batched_token_steps": 0,
            "total_token_steps": 0,
            "last_active_batch_sizes": (),
            "executed_request_token_steps": 0,
            "avoided_request_token_steps": 0,
            "executed_model_steps": 0,
            "avoided_model_steps": 0,
            "first_rejection_position_histogram": {},
        }
        self._target_decision_diagnostics = []
        self.dflash2_draft_model = None
        self.dflash2_draft_cache = None
        self._dflash2_contexts = {}
        self.qwen35_text_config = None
        self._memory_trace = MemoryTrace(
            enabled=bool(
                config.get("memory_trace", config.get("memory_trace_enabled", False))
            )
        )
        dist.init_process_group(
            "nccl", "tcp://localhost:12345", world_size=config["world_size"], rank=rank
        )
        torch.cuda.set_device(rank)

        # set model
        path_str = self.config["model_name_or_path"]
        model_name = Path(path_str).name
        qwen35_raw_config = None
        qwen35_text_config = None
        configured_dtype = _dtype_from_config(config.get("dtype"))
        if configured_dtype is not None:
            torch.set_default_dtype(configured_dtype)
        if is_qwen35_config(config):
            qwen35_raw_config = config
        elif model_name in {"Qwen3.8-27B", "Qwen3.5-27B"} or Path(path_str).is_dir():
            try:
                candidate_config = load_qwen35_config_dict(path_str)
            except Exception:
                if model_name in {"Qwen3.8-27B", "Qwen3.5-27B"}:
                    raise
            else:
                if is_qwen35_config(candidate_config):
                    qwen35_raw_config = candidate_config

        if qwen35_raw_config is not None:
            qwen35_text_config = normalize_qwen35_text_config(qwen35_raw_config)
            self.qwen35_text_config = qwen35_text_config
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
            self.model.enable_experimental_batched_linears(
                mlp=bool(config.get("qwen35_parallel_batched_mlp", False)),
                safe_shapes=bool(config.get(
                    "qwen35_parallel_batched_safe_linears", False
                )),
            )
            self.config["num_layers"] = qwen35_text_config.num_hidden_layers
            self.config["num_kv_heads"] = qwen35_text_config.num_key_value_heads
            self.config["head_dim"] = qwen35_text_config.head_dim
            self.config["vocab_size"] = qwen35_text_config.vocab_size
            self.config["num_full_attention_layers"] = len(
                qwen35_text_config.full_attention_layers
            )
            self.is_qwen35_target = True
            self.hybrid_cache_manager = HybridCacheManager(
                reuse_transaction_kv=self.config.get("qwen35_reuse_transaction_kv", True)
            )
            # Keep cached decode, verification, and replay on the same CUDA
            # kernel shapes. BF16 reductions can otherwise vary with the
            # active batch/context shape and change greedy decisions at ties.
            self.config.setdefault(
                "qwen35_cached_attention_pad_to_length",
                self.config.get("max_model_length"),
            )
            self.config.setdefault(
                "qwen35_cached_batch_pad_to_size",
                min(
                    4,
                    self.config.get(
                        "max_num_sequences", self.config.get("max_num_seqs", 1)
                    ),
                ),
            )
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
        self._record_memory_stage("target_model_allocated")

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
        self._record_memory_stage("target_load")

        self._maybe_load_dflash2_draft(rank)

        # Load weights in CPU (move the model to GPU after loading weights)
        # self.model = self.model.cuda(rank)

        self.sampler = SamplerLayer()

        # Store default dtype before it's needed in allocate_kv_cache
        self.default_dtype = next(self.model.parameters()).dtype

        # Debug flag for first decode step
        self._first_decode = False

        self._dflash2_profile_stats = {
            "context_cache_update_calls": 0,
            "context_cache_update_tokens": 0,
            "context_combine_seconds": 0.0,
            "context_projection_seconds": 0.0,
            "context_cache_install_seconds": 0.0,
            "proposal_calls": 0,
            "proposal_requests": 0,
            "max_proposal_batch_size": 0,
            "tensor_batched_proposal_calls": 0,
            "proposal_stage_seconds": {},
        }
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

    def _speculative_config_dict(self) -> dict:
        value = self.config.get("speculative_config") or {}
        return value if isinstance(value, dict) else getattr(value, "__dict__", {})

    def _memory_device(self):
        return f"cuda:{self.rank}" if torch.cuda.is_available() else None

    def _memory_estimates(self) -> dict[str, Any]:
        estimates: dict[str, Any] = {
            "target_parameter_bytes": module_parameter_bytes(
                getattr(self, "model", None)
            ),
            "draft_parameter_bytes": module_parameter_bytes(
                getattr(self, "dflash2_draft_model", None)
            ),
        }
        hybrid = getattr(self, "hybrid_cache_manager", None)
        if hybrid is not None:
            for key, value in hybrid.estimate_bytes().items():
                estimates[f"hybrid_{key}"] = value
        draft_cache = getattr(self, "dflash2_draft_cache", None)
        if draft_cache is not None:
            for key, value in draft_cache.estimate_bytes().items():
                estimates[f"dflash2_cache_{key}"] = value
        capacity = self.config.get("qwen35_dynamic_cache_capacity")
        if isinstance(capacity, dict):
            estimates.update(
                {f"qwen35_capacity_{key}": value for key, value in capacity.items()}
            )
        return estimates

    def _record_memory_stage(self, stage: str, **extra: Any) -> None:
        trace = getattr(self, "_memory_trace", None)
        if trace is None:
            return
        trace.record(
            stage,
            device=self._memory_device(),
            estimates=self._memory_estimates(),
            extra=extra,
        )

    def get_memory_trace(self) -> list[dict[str, Any]]:
        trace = getattr(self, "_memory_trace", None)
        return [] if trace is None else trace.snapshot()

    def _model_device_for_replay(self) -> torch.device:
        try:
            return next(self.model.parameters()).device
        except (AttributeError, StopIteration):
            return torch.device("cpu")

    def _maybe_load_dflash2_draft(self, rank: int) -> None:
        spec_config = self._speculative_config_dict()
        if spec_config.get("proposer") != "dflash2":
            return
        if not self.is_qwen35_target:
            raise ValueError("DFlash2 proposer requires a Qwen3.5/Qwen3.8 target")
        if self.world_size != 1:
            raise NotImplementedError(
                "DFlash2 integration currently supports world_size=1"
            )
        draft_path = spec_config.get("draft_model_name_or_path") or self.config.get(
            "dflash2_model_name_or_path"
        )
        if not draft_path:
            raise ValueError(
                "DFlash2 proposer requires speculative_config.draft_model_name_or_path"
            )
        self._record_memory_stage("draft_load_start")
        draft_config = DFlash2Config.from_pretrained_path(draft_path)
        draft_config.validate_contract()
        with torch.device("meta"):
            draft_model = DFlash2DraftModel(draft_config)
        draft_model.to_empty(device=f"cuda:{rank}")
        draft_model.eval()
        safe_open_device = str(self.config.get("dflash2_safe_open_device", "cpu"))
        if safe_open_device == "cuda":
            safe_open_device = f"cuda:{rank}"
        load_dflash2_weights_from_safetensors(
            draft_model,
            draft_path,
            safe_open_device=safe_open_device,
        )
        self.dflash2_draft_model = draft_model
        self.dflash2_draft_cache = DFlash2DraftCache(draft_config.sliding_window)
        self._record_memory_stage("draft_load")
        print(
            "[Rank 0] DFlash2 draft loaded: "
            f"layers={draft_config.num_sliding_attention_layers}, "
            f"K={draft_config.num_speculative_tokens}, "
            f"hidden_taps={list(draft_config.target_layer_ids)}"
        )

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
        self._record_memory_stage("warmup_prefill_start")
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
                    self._dflash2_contexts.pop(seq.seq_id, None)
                    if self.dflash2_draft_cache is not None:
                        self.dflash2_draft_cache.release(seq.request_id)
        torch.cuda.empty_cache()
        self._record_memory_stage("warmup_prefill")

    def _generic_kv_cache_modules(self) -> list[Any]:
        return [
            module
            for module in self.model.modules()
            if hasattr(module, "k_cache") and hasattr(module, "v_cache")
        ]

    def _qwen35_dynamic_cache_byte_model(self) -> dict[str, int]:
        qwen_config = self.qwen35_text_config
        if qwen_config is None:
            return {}
        itemsize = self.default_dtype.itemsize
        full_attention_token_bytes = (
            len(qwen_config.full_attention_layers)
            * 2
            * qwen_config.num_key_value_heads
            * qwen_config.head_dim
            * itemsize
        )
        recurrent_per_linear_layer = (
            qwen_config.linear_num_value_heads
            * qwen_config.linear_key_head_dim
            * qwen_config.linear_value_head_dim
            * torch.float32.itemsize
        )
        conv_dim = (
            qwen_config.linear_num_key_heads * qwen_config.linear_key_head_dim * 2
            + qwen_config.linear_num_value_heads * qwen_config.linear_value_head_dim
        )
        conv_per_linear_layer = (
            conv_dim * max(qwen_config.linear_conv_kernel_dim - 1, 0) * itemsize
        )
        linear_request_bytes = len(qwen_config.linear_attention_layers) * (
            recurrent_per_linear_layer + conv_per_linear_layer
        )
        dflash2_token_bytes = 0
        draft_model = getattr(self, "dflash2_draft_model", None)
        if draft_model is not None:
            draft_config = draft_model.config
            dflash2_token_bytes = (
                len(draft_config.target_layer_ids) * draft_config.hidden_size * itemsize
                + torch.int64.itemsize
                + draft_config.num_sliding_attention_layers
                * 2
                * draft_config.num_key_value_heads
                * draft_config.head_dim
                * itemsize
            )
        return {
            "full_attention_token_bytes": full_attention_token_bytes,
            "linear_request_bytes": linear_request_bytes,
            "linear_recurrent_request_bytes": len(qwen_config.linear_attention_layers)
            * recurrent_per_linear_layer,
            "linear_conv_request_bytes": len(qwen_config.linear_attention_layers)
            * conv_per_linear_layer,
            "dflash2_token_bytes": dflash2_token_bytes,
            "token_bytes": full_attention_token_bytes + dflash2_token_bytes,
        }

    def _configure_qwen35_dynamic_cache_capacity(self, available_mem: float) -> None:
        byte_model = self._qwen35_dynamic_cache_byte_model()
        configured_blocks = max(
            1,
            int(self.config.get("max_num_sequences", 1))
            * math.ceil(self.config["max_model_length"] / self.block_size),
        )
        max_sequences = int(self.config.get("max_num_sequences", 1))
        request_state_bytes = max_sequences * byte_model["linear_request_bytes"]
        token_bytes = max(1, byte_model["token_bytes"])
        block_bytes = token_bytes * self.block_size
        available_for_token_cache = max(0.0, available_mem - request_state_bytes)
        capacity_blocks = max(1, int(available_for_token_cache // block_bytes))
        max_cached_blocks = min(configured_blocks, capacity_blocks)
        self.config["max_cached_blocks"] = max_cached_blocks
        self.config["qwen35_uses_generic_kv_cache"] = False
        self.config["qwen35_dynamic_cache_capacity"] = {
            **byte_model,
            "configured_blocks": configured_blocks,
            "max_cached_blocks": max_cached_blocks,
            "available_bytes": int(max(0.0, available_mem)),
            "available_for_token_cache_bytes": int(available_for_token_cache),
            "reserved_request_state_bytes": int(request_state_bytes),
            "block_bytes": int(block_bytes),
            "generic_kv_pool_bytes": 0,
            "generic_kv_cache_layers": 0,
        }
        if self.rank == 0:
            print(
                "[Rank 0] Qwen3.5/Qwen3.8 dynamic cache capacity: "
                f"blocks={max_cached_blocks}/{configured_blocks}, "
                f"block_size={self.block_size}, generic_kv_pool=0"
            )

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

        if self.is_qwen35_target:
            qwen35_fraction = float(
                self.config.get("qwen35_dynamic_cache_memory_fraction", 1.0)
            )
            self._configure_qwen35_dynamic_cache_capacity(
                available_mem * qwen35_fraction
            )
            self._record_memory_stage("target_cache_allocation")
            return

        available_mem *= target_cache_fraction

        # find parameters to compute kv cache size
        cache_modules = self._generic_kv_cache_modules()
        num_layers = len(cache_modules)
        if num_layers <= 0:
            raise RuntimeError("model exposes no paged-attention KV cache modules")
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
            num_layers,
            self.config["max_cached_blocks"],
            self.block_size,
            num_kv_heads,
            head_dim,
            device=f"cuda:{self.rank}",
            dtype=self.default_dtype,
        )
        for layer_id, module in enumerate(cache_modules):
            module.k_cache = allocated_kv_cache[0, layer_id]
            module.v_cache = allocated_kv_cache[1, layer_id]
        if self.rank == 0:
            print(
                f"[Rank 0] KV cache attached to {num_layers} full-attention layers; "
                f"blocks={self.config['max_cached_blocks']}, block_size={self.block_size}, "
                f"target_cache_fraction={target_cache_fraction:.3f}"
            )
        self._record_memory_stage(
            "target_cache_allocation",
            generic_kv_cache_layers=num_layers,
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
    @torch.inference_mode()
    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        if getattr(self, "is_qwen35_target", False):
            if getattr(self, "config", {}).get("qwen35_vectorized_target", False):
                self.model.set_experimental_vectorized_target(
                    not is_prefill, fused_gdn=self.config.get("qwen35_fused_gdn", False),
                    short_prefill_attention=self.config.get("qwen35_short_prefill_attention", False),
                )
            stage_name = "prefill" if is_prefill else "target_decode"
            self._record_memory_stage(
                f"{stage_name}_start",
                batch_size=len(seqs),
            )
            try:
                if not is_prefill:
                    input_ids = self.prepare_decode(seqs).reshape(len(seqs), 1)
                    outputs = self.model.forward_with_cache_batch(
                        input_ids,
                        [1] * len(seqs),
                        self.hybrid_cache_manager,
                        [seq.request_id for seq in seqs],
                        capture_hidden_taps=self.dflash2_draft_model is not None,
                        attention_pad_to_length=self.config.get(
                            "qwen35_cached_attention_pad_to_length"
                        ),
                        batch_pad_to_size=self.config.get(
                            "qwen35_cached_batch_pad_to_size"
                        ),
                    )
                    if self.rank != 0:
                        return None
                    logits = torch.cat(
                        [output.logits[:, -1, :] for output in outputs], dim=0
                    )
                    hidden_states = (
                        torch.cat(
                            [output.hidden_states[:, -1, :] for output in outputs],
                            dim=0,
                        )
                        if all(hasattr(output, "hidden_states") for output in outputs)
                        else None
                    )
                    token_ids = self._sample_qwen35_logits(
                        logits,
                        self.prepare_sample(seqs),
                        hidden_states=hidden_states,
                    )
                    token_ids = token_ids.reshape(-1)
                    if token_ids.numel() != len(seqs):
                        raise RuntimeError(
                            "Qwen3.5/Qwen3.8 batched sampler returned the wrong "
                            "token count"
                        )
                    for seq, row_logits, token_id in zip(seqs, logits, token_ids):
                        self._record_target_decision(
                            seq,
                            row_logits,
                            output_index=seq.num_completion_tokens,
                            source="decode",
                            active_batch_size=len(seqs),
                            input_token_id=seq.last_token,
                            selected_token_id=int(token_id.item()),
                        )
                    for seq, output, token_id in zip(seqs, outputs, token_ids):
                        self._store_dflash2_context_from_target_output(
                            seq,
                            output,
                            int(token_id.item()),
                        )
                    return token_ids

                sampled_tokens = []
                for seq in seqs:
                    input_ids = self.prepare_prefill([seq])
                    output = self.model.forward_with_cache(
                        input_ids.reshape(-1),
                        self.hybrid_cache_manager,
                        seq.request_id,
                        capture_hidden_taps=self.dflash2_draft_model is not None,
                    )
                    if self.rank != 0:
                        continue
                    logits = output.logits[:, -1, :]
                    token_ids = self._sample_qwen35_logits(
                        logits,
                        self.prepare_sample([seq]),
                        hidden_states=(
                            output.hidden_states[:, -1, :]
                            if hasattr(output, "hidden_states")
                            else None
                        ),
                    )
                    token_ids = token_ids.reshape(-1)
                    if token_ids.numel() != 1:
                        raise RuntimeError(
                            "Qwen3.5/Qwen3.8 per-request sampler returned "
                            f"{token_ids.numel()} tokens"
                        )
                    self._record_target_decision(
                        seq,
                        logits[0],
                        output_index=0,
                        source="prefill",
                        active_batch_size=1,
                        input_token_id=seq.last_token,
                        selected_token_id=int(token_ids[0].item()),
                    )
                    sampled_tokens.append(token_ids)
                    self._store_dflash2_context_from_target_output(
                        seq,
                        output,
                        int(token_ids.reshape(-1)[0].item()),
                    )
                if self.rank != 0:
                    return None
                return torch.cat(sampled_tokens) if sampled_tokens else None
            finally:
                self._record_memory_stage(
                    stage_name,
                    batch_size=len(seqs),
                )
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

    def _sample_qwen35_block_outputs(self, outputs):
        """Sample once per model batch, then download only the token IDs."""
        if not self.config.get("qwen35_batch_verification_sampling", True):
            tokens = []
            for output in outputs:
                tokens.append(self._sample_qwen35_logits(
                    output.logits.reshape(-1, output.logits.shape[-1]),
                    torch.zeros(output.logits.shape[1], dtype=torch.float32,
                                device=output.logits.device),
                    hidden_states=output.hidden_states.reshape(
                        -1, output.hidden_states.shape[-1]),
                ).detach().cpu().tolist())
            return tokens, len(outputs)
        groups = {}
        for index, output in enumerate(outputs):
            batch = getattr(output, "verification_batch", None)
            key = None if batch is None else id(batch[0])
            groups.setdefault(key, []).append((index, output))
        tokens = [None] * len(outputs)
        for entries in groups.values():
            first_batch = getattr(entries[0][1], "verification_batch", None)
            rows = [getattr(output, "verification_batch", (None, None, None))[2]
                    if getattr(output, "verification_batch", None) is not None
                    else None for _, output in entries]
            if first_batch is not None and rows == list(range(len(entries))):
                logits, hidden, _ = first_batch
                logits = logits[:len(entries)].reshape(-1, logits.shape[-1])
                hidden = hidden[:len(entries)].reshape(-1, hidden.shape[-1])
            else:
                # Legacy/custom outputs or an incomplete batch: gather once.
                logits = torch.cat([output.logits.reshape(-1, output.logits.shape[-1])
                                    for _, output in entries], dim=0)
                hidden = torch.cat([output.hidden_states.reshape(
                    -1, output.hidden_states.shape[-1]) for _, output in entries], dim=0)
            selected = self._sample_qwen35_logits(
                logits, torch.zeros(logits.shape[0], dtype=torch.float32,
                                    device=logits.device),
                hidden_states=hidden,
            ).detach().cpu().tolist()
            offset = 0
            for index, output in entries:
                count = output.logits.shape[1]
                tokens[index] = selected[offset:offset + count]
                offset += count
        return tokens, len(groups)

    def _sample_qwen35_logits(
        self,
        logits: torch.Tensor,
        temperatures: torch.Tensor,
        *,
        hidden_states: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if hidden_states is None or not torch.all(temperatures == 0):
            return self.sampler(logits, temperatures)
        token_ids = greedy_token_ids(logits)
        if not self.config.get("resolve_greedy_ties_with_fp32_lm_head", False):
            return token_ids
        flat_logits = logits.reshape(-1, logits.shape[-1])
        flat_hidden = hidden_states.reshape(flat_logits.shape[0], -1)
        flat_token_ids = token_ids.reshape(-1)
        lm_head = getattr(self.model, "lm_head", None)
        if lm_head is None:
            return token_ids
        bias = getattr(lm_head, "bias", None)
        for row in range(flat_logits.shape[0]):
            row_logits = flat_logits[row]
            tied_token_ids = (
                (row_logits == row_logits.max()).nonzero(as_tuple=False).reshape(-1)
            )
            if tied_token_ids.numel() <= 1:
                continue
            tied_weight = lm_head.weight.index_select(0, tied_token_ids)
            tied_scores = F.linear(
                flat_hidden[row : row + 1].float(),
                tied_weight.float(),
                None if bias is None else bias.index_select(0, tied_token_ids).float(),
            )
            tied_index = int(greedy_token_ids(tied_scores).item())
            flat_token_ids[row] = tied_token_ids[tied_index]
        return flat_token_ids.reshape(token_ids.shape)

    def _record_target_decision(
        self,
        seq: Sequence,
        logits: torch.Tensor,
        *,
        output_index: int,
        source: str,
        active_batch_size: int,
        input_token_id: int | None = None,
        query_offset: int | None = None,
        selected_token_id: int | None = None,
    ) -> None:
        config = getattr(self, "config", {})
        if not config.get("debug_target_decisions", False):
            return
        history_limit = int(config.get("max_metrics_history", 256))
        if history_limit <= 0:
            return
        flat_logits = logits.detach().reshape(-1)
        top_count = min(2, flat_logits.numel())
        top_values, top_indices = torch.topk(flat_logits, k=top_count)
        token_ids = top_indices.detach().cpu().tolist()
        values = top_values.float().detach().cpu().tolist()
        max_logit = flat_logits.max()
        tied_token_ids = (flat_logits == max_logit).nonzero(as_tuple=False).reshape(-1)
        record = {
            "request_id": seq.request_id,
            "sequence_id": seq.seq_id,
            "output_index": int(output_index),
            "source": source,
            "active_batch_size": int(active_batch_size),
            "input_token_id": (
                int(input_token_id) if input_token_id is not None else None
            ),
            "query_offset": int(query_offset) if query_offset is not None else None,
            "argmax_token_id": int(
                selected_token_id
                if selected_token_id is not None
                else greedy_token_ids(flat_logits).item()
            ),
            "exact_tie_break_policy": (
                "bf16_max_then_fp32_lm_head_tied_tokens"
                if config.get("resolve_greedy_ties_with_fp32_lm_head", False)
                else "highest_token_id"
            ),
            "max_tie_count": int(tied_token_ids.numel()),
            "max_tie_min_token_id": (
                int(tied_token_ids.min().item()) if tied_token_ids.numel() else None
            ),
            "max_tie_max_token_id": (
                int(tied_token_ids.max().item()) if tied_token_ids.numel() else None
            ),
            "top1_token_id": int(token_ids[0]),
            "top1_logit": float(values[0]),
            "top2_token_id": int(token_ids[1]) if top_count > 1 else None,
            "top2_logit": float(values[1]) if top_count > 1 else None,
            "top2_margin": (float(values[0] - values[1]) if top_count > 1 else None),
        }
        history = getattr(self, "_target_decision_diagnostics", None)
        if history is None:
            history = []
            self._target_decision_diagnostics = history
        history.append(record)
        if len(history) > history_limit:
            del history[:-history_limit]

    def get_target_decision_diagnostics(self) -> list[dict]:
        history = getattr(self, "_target_decision_diagnostics", ())
        return [dict(record) for record in history]

    def _store_dflash2_context_from_target_output(
        self,
        seq: Sequence,
        output,
        anchor_token_id: int,
        decision_index: int | None = None,
    ) -> None:
        if self.dflash2_draft_model is None:
            return
        if not output.hidden_taps:
            return
        if decision_index is None:
            decision_index = output.logits.shape[1] - 1
            hidden_taps = {
                layer_id: value[
                    :,
                    -min(
                        value.shape[1],
                        self.dflash2_draft_model.config.sliding_window,
                    ) :,
                    :,
                ].detach()
                for layer_id, value in output.hidden_taps.items()
            }
        else:
            hidden_taps = {
                layer_id: value[:, decision_index : decision_index + 1, :].detach()
                for layer_id, value in output.hidden_taps.items()
            }
        self._dflash2_contexts[seq.seq_id] = {
            "request_id": seq.request_id,
            "anchor_token_id": int(anchor_token_id),
            "anchor_position": int(
                self.hybrid_cache_manager.get_request_state(
                    seq.request_id
                ).logical_length
                - output.logits.shape[1]
                + decision_index
                + 1
            ),
            "hidden_taps": hidden_taps,
        }
        if self.dflash2_draft_cache is not None:
            self.dflash2_draft_cache.update_target_context(
                seq.request_id,
                hidden_taps,
                int(anchor_token_id),
                int(self._dflash2_contexts[seq.seq_id]["anchor_position"]),
            )
            self._install_dflash2_projected_context(
                seq.request_id,
                hidden_taps,
                int(self._dflash2_contexts[seq.seq_id]["anchor_position"])
                - next(iter(hidden_taps.values())).shape[1],
                append=False,
            )

    def _install_dflash2_projected_context(
        self,
        request_id: str,
        hidden_taps: dict[int, torch.Tensor],
        start_position: int,
        *,
        append: bool,
    ) -> None:
        if self.dflash2_draft_model is None or self.dflash2_draft_cache is None:
            return
        context_length = next(iter(hidden_taps.values())).shape[1]
        reference = next(iter(hidden_taps.values()))
        positions = torch.arange(
            start_position,
            start_position + context_length,
            dtype=torch.long,
            device=reference.device,
        ).view(1, -1)
        profile_stages = bool(self.config.get("dflash2_profile_stages", False))
        if profile_stages and reference.is_cuda:
            torch.cuda.synchronize(reference.device)
        started = perf_counter()
        context_states = self.dflash2_draft_model.combine_hidden_taps(hidden_taps)
        if profile_stages:
            if reference.is_cuda:
                torch.cuda.synchronize(reference.device)
            combined = perf_counter()
        projected = self.dflash2_draft_model.project_combined_context_kv(
            context_states,
            positions,
        )
        if profile_stages:
            if reference.is_cuda:
                torch.cuda.synchronize(reference.device)
            projected_at = perf_counter()
        self.dflash2_draft_cache.install_projected_context(
            request_id,
            positions,
            projected,
            append=append,
        )
        if profile_stages:
            if reference.is_cuda:
                torch.cuda.synchronize(reference.device)
            stats = self._dflash2_profile_stats
            stats["context_cache_update_calls"] += 1
            stats["context_cache_update_tokens"] += context_length
            finished = perf_counter()
            stats["context_combine_seconds"] += combined - started
            stats["context_projection_seconds"] += projected_at - combined
            stats["context_cache_install_seconds"] += finished - projected_at

    def _append_dflash2_hidden_history(
        self,
        sequence_id: int,
        request_id: str,
        output,
        token_count: int,
    ) -> dict[int, torch.Tensor]:
        if token_count <= 0:
            return self._dflash2_contexts.get(sequence_id, {}).get("hidden_taps", {})
        previous = self._dflash2_contexts.get(sequence_id, {})
        old_taps = previous.get("hidden_taps", {})
        window = self.dflash2_draft_model.config.sliding_window
        hidden_taps = getattr(output, "hidden_taps", output)
        merged = {}
        for layer_id, value in hidden_taps.items():
            new_value = value[:, :token_count, :].detach()
            old_value = old_taps.get(layer_id)
            if old_value is not None:
                new_value = torch.cat([old_value, new_value], dim=1)
            merged[layer_id] = new_value[:, -min(new_value.shape[1], window) :, :]
        return merged

    @torch.inference_mode()
    def propose_dflash2(
        self,
        sequences: list[Sequence],
        max_tokens: list[int],
    ) -> list[Proposal]:
        if self.dflash2_draft_model is None or self.dflash2_draft_cache is None:
            raise RuntimeError("DFlash2 draft model is not loaded")
        if len(sequences) != len(max_tokens):
            raise ValueError("sequence and proposer budget counts must match")
        budgets = [
            min(
                self.dflash2_draft_model.config.num_speculative_tokens,
                max(0, max_token_count),
            )
            for max_token_count in max_tokens
        ]
        proposals: list[Proposal | None] = [None] * len(sequences)
        active_indices = [index for index, budget in enumerate(budgets) if budget]
        for index, (seq, budget) in enumerate(zip(sequences, budgets)):
            if not budget:
                proposals[index] = Proposal(
                    seq.seq_id,
                    (),
                    {"kind": "dflash2", "block_parallel": True, "budget": 0},
                )
        if not active_indices:
            return [proposal for proposal in proposals if proposal is not None]

        active_sequences = [sequences[index] for index in active_indices]
        contexts = []
        for seq in active_sequences:
            context = self._dflash2_contexts.get(seq.seq_id)
            if context is None or context["request_id"] != seq.request_id:
                raise RuntimeError(
                    "DFlash2 target hidden context is missing for this sequence"
                )
            contexts.append(context)
        request_ids = [seq.request_id for seq in active_sequences]
        context_positions, context_valid_mask, context_kv = (
            self.dflash2_draft_cache.batch_projected_context(request_ids)
        )
        device = next(self.dflash2_draft_model.parameters()).device
        begun_request_ids = []
        self._record_memory_stage(
            "proposal_start",
            batch_size=len(active_sequences),
            max_budget=max(budgets),
        )
        try:
            for request_id in request_ids:
                self.dflash2_draft_cache.begin_proposal(request_id)
                begun_request_ids.append(request_id)
            profile_stages = bool(self.config.get("dflash2_profile_stages", False))
            proposal_batch, debug = self.dflash2_draft_model.forward_block(
                anchor_token_ids=torch.tensor(
                    [context["anchor_token_id"] for context in contexts],
                    dtype=torch.long,
                    device=device,
                ),
                anchor_positions=torch.tensor(
                    [context["anchor_position"] for context in contexts],
                    dtype=torch.long,
                    device=device,
                ),
                hidden_taps=None,
                token_embedding=self.model.model.embed_tokens,
                lm_head=self.model.lm_head,
                context_positions=context_positions,
                context_kv_by_layer=context_kv,
                context_valid_mask=context_valid_mask,
                profile_stages=profile_stages,
                return_debug=bool(self.config.get("dflash2_debug_metadata", False)),
            )
            self._record_dflash2_proposal_batch(
                len(active_sequences),
                debug,
                profile_stages=profile_stages,
            )
            if proposal_batch.token_ids.shape[0] != len(active_sequences):
                raise ValueError("DFlash2 draft model returned the wrong batch size")
            if proposal_batch.valid_lengths.shape[0] != len(active_sequences):
                raise ValueError(
                    "DFlash2 draft valid lengths have the wrong batch size"
                )
            for row, index in enumerate(active_indices):
                seq = sequences[index]
                valid_length = min(
                    int(proposal_batch.valid_lengths[row].item()),
                    budgets[index],
                )
                token_ids = tuple(
                    int(value)
                    for value in proposal_batch.token_ids[row, :valid_length].tolist()
                )
                metadata = {
                    "kind": "dflash2",
                    "block_parallel": True,
                    "proposal_batch_size": len(active_sequences),
                }
                if self.config.get("dflash2_debug_metadata", False):
                    metadata.update(debug)
                proposals[index] = Proposal(seq.seq_id, token_ids, metadata)
                self.dflash2_draft_cache.advance(seq.request_id, valid_length)
            if any(proposal is None for proposal in proposals):
                raise RuntimeError("DFlash2 proposal batching lost a request row")
            self._record_memory_stage(
                "proposal",
                batch_size=len(active_sequences),
                proposed_tokens=sum(len(proposal.token_ids) for proposal in proposals),
            )
            return [proposal for proposal in proposals if proposal is not None]
        except Exception:
            for request_id in begun_request_ids:
                self.dflash2_draft_cache.rollback_proposal(request_id)
            self._record_memory_stage(
                "proposal_rollback",
                batch_size=len(active_sequences),
            )
            raise

    def _record_dflash2_proposal_batch(
        self,
        batch_size: int,
        debug: dict,
        *,
        profile_stages: bool,
    ) -> None:
        stats = getattr(self, "_dflash2_profile_stats", None)
        if stats is None:
            stats = {
                "context_cache_update_calls": 0,
                "context_cache_update_tokens": 0,
                "context_combine_seconds": 0.0,
                "context_projection_seconds": 0.0,
                "context_cache_install_seconds": 0.0,
                "proposal_calls": 0,
                "proposal_requests": 0,
                "max_proposal_batch_size": 0,
                "tensor_batched_proposal_calls": 0,
                "proposal_stage_seconds": {},
            }
            self._dflash2_profile_stats = stats
        stats["proposal_calls"] += 1
        stats["proposal_requests"] += batch_size
        stats["max_proposal_batch_size"] = max(
            stats["max_proposal_batch_size"], batch_size
        )
        stats["tensor_batched_proposal_calls"] += int(batch_size > 1)
        if profile_stages:
            totals = stats["proposal_stage_seconds"]
            for name, seconds in debug["stage_seconds"].items():
                totals[name] = totals.get(name, 0.0) + seconds

    def get_dflash2_profile_stats(self) -> dict:
        stats = getattr(self, "_dflash2_profile_stats", None)
        if stats is None:
            return {}
        result = dict(stats)
        result["proposal_stage_seconds"] = dict(stats["proposal_stage_seconds"])
        calls = stats["proposal_calls"]
        result["proposal_stage_seconds_per_call"] = (
            {
                name: seconds / calls
                for name, seconds in stats["proposal_stage_seconds"].items()
            }
            if calls
            else {}
        )
        return result

    def verify(self, batch: SpecDecodeBatch) -> list[VerificationResult]:
        """Greedy K+1 target decisions for each speculative request."""
        if getattr(self, "is_qwen35_target", False):
            return self._verify_qwen35_cached(batch)
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

    @torch.inference_mode()
    def _verify_qwen35_cached(self, batch: SpecDecodeBatch) -> list[VerificationResult]:
        """Tensor-batch Qwen3.5/Qwen3.8 verification across request caches."""
        if self.config.get("qwen35_vectorized_target", False):
            self.model.set_experimental_vectorized_target(
                True, fused_gdn=self.config.get("qwen35_fused_gdn", False),
                short_prefill_attention=self.config.get("qwen35_short_prefill_attention", False),
                compact_updates=self.config.get("qwen35_compact_verification_copies", False),
            )
        if self.hybrid_cache_manager is None:
            raise RuntimeError("Qwen3.5/Qwen3.8 verifier requires HybridCacheManager")
        if not batch.sequences:
            return []
        sequence_ids = [seq.seq_id for seq in batch.sequences]
        self.rollback_verified_cache(sequence_ids)
        self._record_memory_stage("verify_start", batch_size=len(batch.sequences))
        query_tokens_by_request = []
        for seq, proposal in zip(batch.sequences, batch.proposals):
            if seq.num_tokens - seq.num_computed_tokens != 1:
                raise RuntimeError("verification requires exactly one pending token")
            if tuple(seq.provisional_token_ids) != proposal.token_ids:
                raise RuntimeError("sequence/proposal provisional state mismatch")
            query_tokens = (seq.last_token, *proposal.token_ids)
            query_tokens_by_request.append(query_tokens)
        transactions = {}
        try:
            for seq, query_tokens in zip(batch.sequences, query_tokens_by_request):
                transactions[seq.request_id] = (
                    self.hybrid_cache_manager.begin_transaction(
                        [seq.request_id],
                        {seq.request_id: len(query_tokens)},
                    )
                )
            device = next(self.model.parameters()).device
            query_lengths = [
                len(query_tokens) for query_tokens in query_tokens_by_request
            ]
            input_ids = torch.zeros(
                len(query_lengths),
                max(query_lengths),
                dtype=torch.long,
                device=device,
            )
            for row, query_tokens in enumerate(query_tokens_by_request):
                input_ids[row, : len(query_tokens)] = torch.tensor(
                    query_tokens,
                    dtype=torch.long,
                    device=device,
                )
            parallel_verification = bool(
                self.config.get("qwen35_parallel_verification", False)
            )
            stop_at_mismatch = not parallel_verification and bool(
                self.config.get("qwen35_stop_verification_at_mismatch", False)
            )
            sampled_by_request = {
                seq.request_id: [] for seq in batch.sequences
            }
            proposals_by_request = {
                seq.request_id: proposal.token_ids
                for seq, proposal in zip(batch.sequences, batch.proposals)
            }

            postprocess_sampling_calls = 0
            postprocess_token_downloads = 0
            single_block_reused_requests = 0

            def continue_after_step(
                active_request_ids, step_logits, step_hidden, token_index,
            ):
                nonlocal postprocess_sampling_calls, postprocess_token_downloads
                postprocess_sampling_calls += 1
                postprocess_token_downloads += 1
                temperatures = torch.zeros(
                    step_logits.shape[0], dtype=torch.float32,
                    device=step_logits.device,
                )
                selected = self._sample_qwen35_logits(
                    step_logits.reshape(-1, step_logits.shape[-1]),
                    temperatures,
                    hidden_states=step_hidden.reshape(step_hidden.shape[0], -1),
                ).detach().cpu().tolist()
                keep = []
                for request_id, token_id in zip(active_request_ids, selected):
                    sampled_by_request[request_id].append(int(token_id))
                    proposal = proposals_by_request[request_id]
                    keep.append(
                        token_index < len(proposal)
                        and int(token_id) == proposal[token_index]
                    )
                return keep

            common_forward_kwargs = dict(
                transactions=transactions,
                record_linear_updates=self.config.get("qwen35_replay_state_updates", True),
                capture_hidden_taps=bool(
                    self.config.get("capture_verification_hidden_taps", False)
                    or self.dflash2_draft_model is not None
                ),
                attention_pad_to_length=self.config.get(
                    "qwen35_cached_attention_pad_to_length"
                ),
                batch_pad_to_size=self.config.get("qwen35_cached_batch_pad_to_size"),
            )
            parallel_chunk_size = int(
                self.config.get("qwen35_parallel_verification_chunk_size", 8) or 0
            )
            reuse_chunk_samples = (
                parallel_verification and parallel_chunk_size > 0
                and self.config.get("qwen35_reuse_verification_samples", False)
            )
            parallel_model_calls = None
            if parallel_verification and parallel_chunk_size > 0:
                parts = [[] for _ in batch.sequences]
                reported_lengths = [None for _ in batch.sequences]
                actual_lengths = [0 for _ in batch.sequences]
                active_rows = list(range(len(batch.sequences)))
                parallel_model_calls = 0
                for chunk_start in range(0, max(query_lengths), parallel_chunk_size):
                    active_rows = [
                        row for row in active_rows if query_lengths[row] > chunk_start
                    ]
                    if not active_rows:
                        break
                    active_indices = torch.tensor(active_rows, device=device)
                    chunk_ids = input_ids.index_select(0, active_indices)[
                        :, chunk_start : chunk_start + parallel_chunk_size
                    ]
                    chunk_lengths = [
                        min(parallel_chunk_size, query_lengths[row] - chunk_start)
                        for row in active_rows
                    ]
                    chunk_outputs = self.model.forward_with_cache_parallel_batch(
                        chunk_ids,
                        chunk_lengths,
                        self.hybrid_cache_manager,
                        [batch.sequences[row].request_id for row in active_rows],
                        logit_batch_size=min(
                            len(batch.sequences),
                            common_forward_kwargs["batch_pad_to_size"]
                            or len(batch.sequences),
                        ),
                        **common_forward_kwargs,
                    )
                    micro_batch_size = (
                        common_forward_kwargs["batch_pad_to_size"]
                        or len(active_rows)
                    )
                    parallel_model_calls += sum(
                        len(set(chunk_lengths[start:start + micro_batch_size]))
                        for start in range(0, len(chunk_lengths), micro_batch_size)
                    )
                    chunk_samples, sampling_calls = self._sample_qwen35_block_outputs(chunk_outputs)
                    postprocess_sampling_calls += sampling_calls
                    postprocess_token_downloads += sampling_calls
                    continuing_rows = []
                    for row, chunk_output, tokens in zip(active_rows, chunk_outputs, chunk_samples):
                        parts[row].append(chunk_output)
                        actual_lengths[row] += chunk_output.logits.shape[1]
                        if reuse_chunk_samples:
                            sampled_by_request[batch.sequences[row].request_id].extend(tokens)
                        proposal = batch.proposals[row].token_ids
                        first_rejection = next((
                            position for position, token in enumerate(
                                tokens, chunk_start
                            )
                            if position < len(proposal)
                            and int(token) != proposal[position]
                        ), None)
                        if first_rejection is not None:
                            reported_lengths[row] = first_rejection + 1
                        else:
                            continuing_rows.append(row)
                    active_rows = continuing_rows
                outputs = []
                for row, request_parts in enumerate(parts):
                    if not request_parts:
                        raise RuntimeError("chunked verifier omitted a request")
                    output_class = type(request_parts[0])
                    reported_length = reported_lengths[row] or actual_lengths[row]
                    if (len(request_parts) == 1
                            and self.config.get("qwen35_reuse_single_block_outputs", True)):
                        part = request_parts[0]
                        outputs.append(output_class(
                            logits=part.logits[:, :reported_length],
                            hidden_states=part.hidden_states[:, :reported_length],
                            hidden_taps={layer_id: value[:, :reported_length]
                                         for layer_id, value in part.hidden_taps.items()},
                            cache_transaction_id=part.cache_transaction_id,
                        ))
                        single_block_reused_requests += 1
                        continue
                    outputs.append(output_class(
                        logits=torch.cat(
                            [part.logits for part in request_parts], dim=1
                        )[:, :reported_length],
                        hidden_states=torch.cat(
                            [part.hidden_states for part in request_parts], dim=1
                        )[:, :reported_length],
                        hidden_taps={
                            layer_id: torch.cat(
                                [part.hidden_taps[layer_id] for part in request_parts],
                                dim=1,
                            )[:, :reported_length]
                            for layer_id in request_parts[0].hidden_taps
                        },
                        cache_transaction_id=request_parts[0].cache_transaction_id,
                    ))
            else:
                forward_fn = (
                    self.model.forward_with_cache_parallel_batch
                    if parallel_verification else self.model.forward_with_cache_batch
                )
                outputs = forward_fn(
                    input_ids,
                    query_lengths,
                    self.hybrid_cache_manager,
                    [seq.request_id for seq in batch.sequences],
                    **common_forward_kwargs,
                    **({"continue_after_step": continue_after_step} if stop_at_mismatch else {}),
                )
            if self.rank != 0:
                raise RuntimeError("Qwen3.5/Qwen3.8 cached verifier requires rank 0")
            if len(outputs) != len(batch.sequences):
                raise RuntimeError(
                    "batched cached target returned the wrong output count"
                )
            if stop_at_mismatch or reuse_chunk_samples:
                flat_token_ids = [
                    token_id
                    for seq, output in zip(batch.sequences, outputs)
                    for token_id in sampled_by_request[seq.request_id][:output.logits.shape[1]]
                ]
            else:
                flat_logits = torch.cat(
                    [
                        output.logits.reshape(-1, output.logits.shape[-1])
                        for output in outputs
                    ],
                    dim=0,
                )
                flat_hidden_states = (
                    torch.cat(
                        [
                            output.hidden_states.reshape(-1, output.hidden_states.shape[-1])
                            for output in outputs
                        ],
                        dim=0,
                    )
                    if all(hasattr(output, "hidden_states") for output in outputs)
                    else None
                )
                temperatures = torch.zeros(
                    flat_logits.shape[0],
                    dtype=torch.float32,
                    device=flat_logits.device,
                )
                postprocess_sampling_calls += 1
                postprocess_token_downloads += 1
                flat_token_ids_tensor = self._sample_qwen35_logits(
                    flat_logits,
                    temperatures,
                    hidden_states=flat_hidden_states,
                )
                flat_token_ids = flat_token_ids_tensor.detach().cpu().tolist()
            results = []
            offset = 0
            for row, (seq, query_tokens, output) in enumerate(zip(
                batch.sequences, query_tokens_by_request, outputs
            )):
                executed = output.logits.shape[1]
                for query_offset, input_token_id in enumerate(query_tokens[:executed]):
                    self._record_target_decision(
                        seq,
                        output.logits[0, query_offset, :],
                        output_index=seq.num_completion_tokens + query_offset,
                        source="verify",
                        active_batch_size=sum(
                            output_item.logits.shape[1] > query_offset
                            for output_item in outputs
                        ),
                        input_token_id=input_token_id,
                        query_offset=query_offset,
                        selected_token_id=int(flat_token_ids[offset + query_offset]),
                    )
                transaction = transactions[seq.request_id]
                token_ids = tuple(flat_token_ids[offset : offset + executed])
                offset += executed
                self._pending_verified_cache_transactions[seq.seq_id] = (
                    PendingQwen35Verification(
                        transaction=transaction,
                        request_id=seq.request_id,
                        hidden_taps={
                            layer_id: value.detach()
                            for layer_id, value in output.hidden_taps.items()
                        },
                        target_token_ids=token_ids,
                        start_position=int(
                            transaction.original_states[seq.request_id].logical_length
                        ),
                        query_tokens=tuple(query_tokens[:executed]),
                        executed_tokens=(
                            actual_lengths[row]
                            if parallel_verification and parallel_chunk_size > 0
                            else executed
                        ),
                    )
                )
                results.append(
                    VerificationResult(
                        sequence_id=seq.seq_id,
                        target_token_ids=token_ids,
                        metadata={
                            "hybrid_cache_transaction_id": transaction.transaction_id,
                            "hybrid_cache_request_id": seq.request_id,
                            "verified_tokens": executed,
                            "hidden_tap_layers": tuple(sorted(output.hidden_taps)),
                            "tensor_batched": True,
                            "verification_batch_size": len(batch.sequences),
                        },
                    )
                )
            executed_lengths = [output.logits.shape[1] for output in outputs]
            if parallel_verification and parallel_chunk_size > 0:
                executed_lengths = actual_lengths
            self._record_qwen35_verification_batch(
                executed_lengths,
                planned_lengths=query_lengths,
                proposal_tokens=[proposal.token_ids for proposal in batch.proposals],
                target_tokens=[result.target_token_ids for result in results],
                parallel_verification=parallel_verification,
                parallel_model_calls=parallel_model_calls,
                postprocess_sampling_calls=postprocess_sampling_calls,
                postprocess_token_downloads=postprocess_token_downloads,
                single_block_reused_requests=single_block_reused_requests,
            )
            self._record_memory_stage(
                "verify",
                batch_size=len(batch.sequences),
                verified_tokens=sum(executed_lengths),
            )
            return results
        except Exception:
            for transaction in transactions.values():
                if transaction.state.name == "ACTIVE":
                    self.hybrid_cache_manager.rollback(transaction)
            self.rollback_verified_cache(sequence_ids)
            self._record_memory_stage(
                "verify_rollback", batch_size=len(batch.sequences)
            )
            raise

    def _record_qwen35_verification_batch(
        self,
        query_lengths: list[int],
        *,
        planned_lengths: list[int] | None = None,
        proposal_tokens: list[tuple[int, ...]] | None = None,
        target_tokens: list[tuple[int, ...]] | None = None,
        parallel_verification: bool = False,
        parallel_model_calls: int | None = None,
        postprocess_sampling_calls: int = 0,
        postprocess_token_downloads: int = 0,
        single_block_reused_requests: int = 0,
    ) -> None:
        stats = getattr(self, "_qwen35_verification_stats", None)
        if stats is None:
            stats = {
                "calls": 0,
                "max_batch_size": 0,
                "tensor_batched_token_steps": 0,
                "total_token_steps": 0,
                "last_active_batch_sizes": (),
                "executed_request_token_steps": 0,
                "avoided_request_token_steps": 0,
                "executed_model_steps": 0,
                "avoided_model_steps": 0,
                "first_rejection_position_histogram": {},
                "postprocess_sampling_calls": 0,
                "postprocess_token_downloads": 0,
                "single_block_reused_requests": 0,
            }
            self._qwen35_verification_stats = stats
        planned_lengths = planned_lengths or query_lengths
        active_batch_sizes = tuple(
            sum(query_length > token_index for query_length in query_lengths)
            for token_index in range(max(query_lengths))
        )
        micro_batch_size = int(
            self.config.get("qwen35_cached_batch_pad_to_size")
            or len(query_lengths)
        )
        def model_steps(lengths):
            return sum(
                max(lengths[start:start + micro_batch_size], default=0)
                for start in range(0, len(lengths), micro_batch_size)
            )

        stats["postprocess_sampling_calls"] += postprocess_sampling_calls
        stats["postprocess_token_downloads"] += postprocess_token_downloads
        stats["single_block_reused_requests"] += single_block_reused_requests
        stats["calls"] += 1
        stats["max_batch_size"] = max(
            stats["max_batch_size"],
            max(active_batch_sizes, default=0),
        )
        stats["tensor_batched_token_steps"] += sum(
            batch_size > 1 for batch_size in active_batch_sizes
        )
        stats["total_token_steps"] += len(active_batch_sizes)
        stats["last_active_batch_sizes"] = active_batch_sizes
        stats["executed_request_token_steps"] += sum(query_lengths)
        stats["avoided_request_token_steps"] += (
            sum(planned_lengths) - sum(query_lengths)
        )
        if parallel_verification:
            stats["executed_model_steps"] += (
                parallel_model_calls if parallel_model_calls is not None else sum(
                    len(set(query_lengths[start:start + micro_batch_size]))
                    for start in range(0, len(query_lengths), micro_batch_size)
                )
            )
            stats["avoided_model_steps"] += 0
        else:
            stats["executed_model_steps"] += model_steps(query_lengths)
            stats["avoided_model_steps"] += (
                model_steps(planned_lengths) - model_steps(query_lengths)
            )
        if proposal_tokens is not None and target_tokens is not None:
            histogram = stats["first_rejection_position_histogram"]
            for proposal, tokens in zip(proposal_tokens, target_tokens):
                position = next(
                    (
                        index for index in range(min(len(proposal), len(tokens)))
                        if tokens[index] != proposal[index]
                    ),
                    None,
                )
                key = "all_accepted" if position is None else str(position)
                histogram[key] = histogram.get(key, 0) + 1

    def get_qwen35_verification_stats(self) -> dict:
        stats = getattr(self, "_qwen35_verification_stats", None)
        if stats is None:
            return {
                "calls": 0,
                "max_batch_size": 0,
                "tensor_batched_token_steps": 0,
                "total_token_steps": 0,
                "last_active_batch_sizes": (),
                "executed_request_token_steps": 0,
                "avoided_request_token_steps": 0,
                "executed_model_steps": 0,
                "avoided_model_steps": 0,
                "first_rejection_position_histogram": {},
                "postprocess_sampling_calls": 0,
                "postprocess_token_downloads": 0,
                "single_block_reused_requests": 0,
            }
        return {
            **stats,
            "first_rejection_position_histogram": dict(
                stats["first_rejection_position_histogram"]
            ),
        }

    def _coerce_pending_qwen35_verification(
        self, entry: Any
    ) -> PendingQwen35Verification:
        if isinstance(entry, PendingQwen35Verification):
            return entry
        if len(entry) == 6:
            (
                transaction,
                request_id,
                output,
                target_token_ids,
                start_position,
                query_tokens,
            ) = entry
            return PendingQwen35Verification(
                transaction=transaction,
                request_id=request_id,
                hidden_taps=dict(getattr(output, "hidden_taps", {})),
                target_token_ids=tuple(target_token_ids),
                start_position=int(start_position),
                query_tokens=tuple(query_tokens),
                executed_tokens=len(query_tokens),
            )
        if len(entry) == 5:
            transaction, request_id, output, target_token_ids, start_position = entry
            return PendingQwen35Verification(
                transaction=transaction,
                request_id=request_id,
                hidden_taps=dict(getattr(output, "hidden_taps", {})),
                target_token_ids=tuple(target_token_ids),
                start_position=int(start_position),
                query_tokens=(),
            )
        transaction, request_id = entry
        return PendingQwen35Verification(
            transaction=transaction,
            request_id=request_id,
            hidden_taps={},
            target_token_ids=(),
            start_position=0,
            query_tokens=(),
        )

    def _replay_qwen35_accepted_prefixes(
        self,
        plans: list[tuple[int, PendingQwen35Verification, int]],
        replay_device: torch.device,
    ) -> None:
        replay_plans = [
            (plan, accepted_length)
            for _sequence_id, plan, accepted_length in plans
            if plan.query_tokens and accepted_length > 0
        ]
        if not replay_plans:
            return
        max_length = max(accepted_length for _plan, accepted_length in replay_plans)
        input_ids = torch.zeros(
            len(replay_plans),
            max_length,
            dtype=torch.long,
            device=replay_device,
        )
        query_lengths = []
        request_ids = []
        for row, (plan, accepted_length) in enumerate(replay_plans):
            accepted_tokens = plan.query_tokens[:accepted_length]
            input_ids[row, :accepted_length] = torch.tensor(
                accepted_tokens,
                dtype=torch.long,
                device=replay_device,
            )
            query_lengths.append(accepted_length)
            request_ids.append(plan.request_id)
        if self.config.get("qwen35_vectorized_target", False):
            self.model.set_experimental_vectorized_target(
                True, fused_gdn=self.config.get("qwen35_fused_gdn", False),
                short_prefill_attention=self.config.get("qwen35_short_prefill_attention", False),
            )
        self.model.forward_with_cache_batch(
            input_ids,
            query_lengths,
            self.hybrid_cache_manager,
            request_ids,
            capture_hidden_taps=False,
            attention_pad_to_length=self.config.get(
                "qwen35_cached_attention_pad_to_length"
            ),
            batch_pad_to_size=self.config.get("qwen35_cached_batch_pad_to_size"),
            return_logits=False,
        )

    def commit_verified_cache(
        self, accepted_lengths_by_sequence: dict[int, int]
    ) -> None:
        if self.hybrid_cache_manager is None:
            return
        pending = getattr(self, "_pending_verified_cache_transactions", {})
        plans: list[tuple[int, PendingQwen35Verification, int]] = []
        for sequence_id, accepted_length in accepted_lengths_by_sequence.items():
            entry = pending.get(sequence_id)
            if entry is None:
                continue
            plan = self._coerce_pending_qwen35_verification(entry)
            if accepted_length < 0:
                raise ValueError("accepted verification length cannot be negative")
            if plan.query_tokens and accepted_length > len(plan.query_tokens):
                raise ValueError("accepted verification length exceeds query tokens")
            plans.append((sequence_id, plan, accepted_length))
        if not plans:
            return
        self._record_memory_stage("commit_start", batch_size=len(plans))
        try:
            replay_device = self._model_device_for_replay()
            # A fully accepted query already has its final cache from verification.
            replay_plans = [
                (sequence_id, plan, accepted_length)
                for sequence_id, plan, accepted_length in plans
                if not (
                    plan.query_tokens
                    and accepted_length == len(plan.query_tokens)
                    and accepted_length == plan.executed_tokens
                    and self.config.get("qwen35_reuse_verified_cache", True)
                )
            ]
            model_replay_plans = []
            for _sequence_id, plan, accepted_length in replay_plans:
                if (
                    plan.query_tokens
                    and self.config.get("qwen35_replay_state_updates", True)
                    and hasattr(self.hybrid_cache_manager, "restore_verified_prefix")
                    and self.hybrid_cache_manager.restore_verified_prefix(
                        plan.transaction, plan.request_id, accepted_length,
                        fused=self.config.get("qwen35_fused_prefix_commit", False),
                    )
                ):
                    continue
                model_replay_plans.append((_sequence_id, plan, accepted_length))
                if plan.query_tokens and hasattr(
                    self.hybrid_cache_manager, "restore_original"
                ):
                    self.hybrid_cache_manager.restore_original(
                        plan.transaction,
                        [plan.request_id],
                    )
            self._replay_qwen35_accepted_prefixes(model_replay_plans, replay_device)

            for _sequence_id, plan, accepted_length in plans:
                decision_index = accepted_length - 1
                if (
                    accepted_length > 0
                    and getattr(self, "dflash2_draft_model", None) is not None
                    and plan.hidden_taps
                    and decision_index < len(plan.target_token_ids)
                ):
                    if not hasattr(self, "_dflash2_contexts"):
                        self._dflash2_contexts = {}
                    hidden_taps = self._append_dflash2_hidden_history(
                        _sequence_id,
                        plan.request_id,
                        plan,
                        accepted_length,
                    )
                    self._dflash2_contexts[_sequence_id] = {
                        "request_id": plan.request_id,
                        "anchor_token_id": int(plan.target_token_ids[decision_index]),
                        "anchor_position": plan.start_position + decision_index + 1,
                        "hidden_taps": hidden_taps,
                    }
                    if self.dflash2_draft_cache is not None:
                        self.dflash2_draft_cache.update_target_context(
                            plan.request_id,
                            hidden_taps,
                            int(plan.target_token_ids[decision_index]),
                            plan.start_position + decision_index + 1,
                        )
                        new_hidden_taps = {
                            layer_id: value[:, :accepted_length, :].detach()
                            for layer_id, value in plan.hidden_taps.items()
                        }
                        self._install_dflash2_projected_context(
                            plan.request_id,
                            new_hidden_taps,
                            plan.start_position,
                            append=(
                                plan.request_id
                                in self.dflash2_draft_cache.context_kv_by_request
                            ),
                        )
                        self.dflash2_draft_cache.commit_proposal(plan.request_id)
                else:
                    draft_cache = getattr(self, "dflash2_draft_cache", None)
                    if draft_cache is not None:
                        draft_cache.rollback_proposal(plan.request_id)

            for sequence_id, plan, accepted_length in plans:
                if plan.query_tokens and hasattr(
                    self.hybrid_cache_manager, "finalize_replayed_commit"
                ):
                    self.hybrid_cache_manager.finalize_replayed_commit(
                        plan.transaction,
                        {plan.request_id: accepted_length},
                    )
                else:
                    self.hybrid_cache_manager.commit(
                        plan.transaction,
                        {plan.request_id: accepted_length},
                    )
                pending.pop(sequence_id, None)
            self._record_memory_stage(
                "commit",
                batch_size=len(plans),
                accepted_tokens=sum(length for _sid, _plan, length in plans),
            )
        except Exception:
            for _sequence_id, plan, _accepted_length in plans:
                if plan.transaction.state.name == "ACTIVE":
                    self.hybrid_cache_manager.rollback(plan.transaction)
                draft_cache = getattr(self, "dflash2_draft_cache", None)
                if draft_cache is not None:
                    draft_cache.rollback_proposal(plan.request_id)
            self._record_memory_stage("commit_rollback", batch_size=len(plans))
            raise

    def rollback_verified_cache(self, sequence_ids: list[int]) -> None:
        if self.hybrid_cache_manager is None:
            return
        pending = getattr(self, "_pending_verified_cache_transactions", {})
        self._record_memory_stage("rollback_start", batch_size=len(sequence_ids))
        for sequence_id in sequence_ids:
            entry = pending.pop(sequence_id, None)
            if entry is None:
                context = getattr(self, "_dflash2_contexts", {}).get(sequence_id)
                draft_cache = getattr(self, "dflash2_draft_cache", None)
                if context is not None and draft_cache is not None:
                    draft_cache.rollback_proposal(context["request_id"])
                continue
            plan = self._coerce_pending_qwen35_verification(entry)
            transaction = plan.transaction
            if transaction.state.name == "ACTIVE":
                self.hybrid_cache_manager.rollback(transaction)
            draft_cache = getattr(self, "dflash2_draft_cache", None)
            if draft_cache is not None:
                draft_cache.rollback_proposal(plan.request_id)
        self._record_memory_stage("rollback", batch_size=len(sequence_ids))

    def release_hybrid_requests(self, request_ids: list[str]) -> None:
        if self.hybrid_cache_manager is None:
            return
        self._record_memory_stage("release_start", request_count=len(request_ids))
        for request_id in request_ids:
            self.hybrid_cache_manager.release_request(request_id)
            if self.dflash2_draft_cache is not None:
                self.dflash2_draft_cache.release(request_id)
        self._dflash2_contexts = {
            seq_id: context
            for seq_id, context in self._dflash2_contexts.items()
            if context.get("request_id") not in set(request_ids)
        }
        self._record_memory_stage("release", request_count=len(request_ids))

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
