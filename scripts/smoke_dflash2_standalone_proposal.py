from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from types import MethodType

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from myvllm.engine.hybrid_cache import HybridCacheManager
from myvllm.models.qwen3_8 import Qwen35ForCausalLM, Qwen35RMSNormGated
from myvllm.models.qwen3_8_config import (
    QWEN35_DEFAULT_HIDDEN_TAPS,
    load_qwen35_config_dict,
    normalize_qwen35_text_config,
)
from myvllm.spec_decode.dflash2 import (
    DFlash2Config,
    DFlash2DraftModel,
    load_dflash2_weights_from_safetensors,
)
from myvllm.utils.loader import load_weights_from_checkpoint


DTYPES = {
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float16": torch.float16,
    "fp16": torch.float16,
    "float32": torch.float32,
    "fp32": torch.float32,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one real DFlash2 standalone block proposal."
    )
    parser.add_argument(
        "--target-model",
        default=os.environ.get("MINIVLLM_QWEN38_PATH"),
        help="Local Qwen3.8 snapshot path. Defaults to MINIVLLM_QWEN38_PATH.",
    )
    parser.add_argument(
        "--draft-model",
        default=os.environ.get("MINIVLLM_DFLASH2_PATH"),
        help="Local DFlash2 snapshot path. Defaults to MINIVLLM_DFLASH2_PATH.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16", choices=sorted(DTYPES))
    parser.add_argument("--safe-open-device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument(
        "--reference-gated-rmsnorm-rounding",
        action="store_true",
        help=(
            "Round normalized GDN activations back to the model dtype before "
            "multiplying the SiLU gate, matching Transformers Qwen3.5."
        ),
    )
    parser.add_argument(
        "--dense-target-prefill-for-proposal",
        action="store_true",
        help=(
            "Use the dense target forward for proposal hidden taps while also "
            "running the cached path to initialize verification state."
        ),
    )
    parser.add_argument(
        "--reference-gdn-chunk",
        action="store_true",
        help=(
            "Use the Transformers Qwen3.5 chunk GDN fallback for the dense "
            "target proposal prefill. Requires --dense-target-prefill-for-proposal."
        ),
    )
    parser.add_argument("--prompt", default="Hello")
    parser.add_argument(
        "--workload-json",
        default=None,
        help="Optional benchmark workload JSON containing prompt_token_ids.",
    )
    parser.add_argument(
        "--request-id",
        default="short-000000",
        help="Request key to load from --workload-json.",
    )
    parser.add_argument(
        "--json-output",
        default=None,
        help="Optional path to write a proposal fixture for Oracle alignment.",
    )
    parser.add_argument(
        "--tensor-dump",
        default=None,
        help="Optional torch.save path for hidden-state alignment tensors.",
    )
    parser.add_argument(
        "--hidden-taps-tensor-dump",
        default=None,
        help=(
            "Optional Oracle tensor dump whose hidden_taps replace the Mini "
            "target taps before the DFlash2 proposal."
        ),
    )
    return parser.parse_args()


def bytes_to_gib(value: int) -> float:
    return value / 1024**3


def print_cuda_memory(label: str, device: torch.device) -> None:
    torch.cuda.synchronize(device)
    free, total = torch.cuda.mem_get_info(device)
    print(
        f"{label}: allocated={bytes_to_gib(torch.cuda.memory_allocated(device)):.2f} GiB "
        f"reserved={bytes_to_gib(torch.cuda.memory_reserved(device)):.2f} GiB "
        f"max_allocated={bytes_to_gib(torch.cuda.max_memory_allocated(device)):.2f} GiB "
        f"free={bytes_to_gib(free):.2f} GiB total={bytes_to_gib(total):.2f} GiB",
        flush=True,
    )


def enable_reference_gated_rmsnorm_rounding() -> None:
    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        normalized = x.float()
        normalized = normalized * torch.rsqrt(
            normalized.pow(2).mean(dim=-1, keepdim=True) + self.eps
        )
        normalized = (normalized * self.weight.float()).to(dtype)
        return (normalized * F.silu(gate.float())).to(dtype)

    Qwen35RMSNormGated.forward = forward


def load_prompt(args: argparse.Namespace, tokenizer) -> tuple[str, list[int]]:
    if args.workload_json is None:
        return args.prompt, tokenizer.encode(args.prompt)
    workload = json.loads(Path(args.workload_json).read_text(encoding="utf-8"))
    try:
        request = workload[args.request_id]
        token_ids = [int(token_id) for token_id in request["prompt_token_ids"]]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            f"Invalid workload request {args.request_id!r}: {exc}"
        ) from exc
    return tokenizer.decode(token_ids, skip_special_tokens=False), token_ids


def tensor_fingerprint(value: torch.Tensor) -> dict[str, object]:
    tensor = value.detach().float()
    flat_tokens = tensor.reshape(-1, tensor.shape[-1])
    return {
        "shape": list(tensor.shape),
        "dtype": str(value.dtype),
        "checksum": float(tensor.sum().item()),
        "l2_norm": float(tensor.norm().item()),
        "abs_max": float(tensor.abs().max().item()),
        "first_token_checksum": float(flat_tokens[0].sum().item()),
        "last_token_checksum": float(flat_tokens[-1].sum().item()),
    }


def reference_chunk_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    decay: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    initial_dtype = query.dtype
    query = query * torch.rsqrt(
        (query * query).sum(dim=-1, keepdim=True) + 1e-6
    )
    key = key * torch.rsqrt((key * key).sum(dim=-1, keepdim=True) + 1e-6)
    query, key, value, beta, decay = [
        tensor.transpose(1, 2).contiguous().float()
        for tensor in (query, key, value, beta, decay)
    ]
    batch_size, num_heads, sequence_length, key_dim = key.shape
    value_dim = value.shape[-1]
    chunk_size = 64
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size)) / (key_dim**0.5)
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    decay = F.pad(decay, (0, pad_size))
    total_length = sequence_length + pad_size

    value_beta = value * beta.unsqueeze(-1)
    key_beta = key * beta.unsqueeze(-1)
    query, key, value, key_beta, value_beta = [
        tensor.reshape(
            tensor.shape[0], tensor.shape[1], -1, chunk_size, tensor.shape[-1]
        )
        for tensor in (query, key, value, key_beta, value_beta)
    ]
    decay = decay.reshape(decay.shape[0], decay.shape[1], -1, chunk_size)
    upper = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device)
    )
    decay = decay.cumsum(dim=-1)
    decay_mask = (
        (decay.unsqueeze(-1) - decay.unsqueeze(-2)).tril().exp().float()
    ).tril()
    attention = -((key_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(
        upper, 0
    )
    for index in range(1, chunk_size):
        row = attention[..., index, :index].clone()
        sub = attention[..., :index, :index].clone()
        attention[..., index, :index] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attention = attention + torch.eye(
        chunk_size, dtype=attention.dtype, device=attention.device
    )
    value = attention @ value_beta
    key_cumulative_decay = attention @ (key_beta * decay.exp().unsqueeze(-1))
    state = (
        torch.zeros(
            batch_size,
            num_heads,
            key_dim,
            value_dim,
            dtype=value.dtype,
            device=value.device,
        )
        if initial_state is None
        else initial_state.to(value)
    )
    core = torch.zeros_like(value)
    for index in range(total_length // chunk_size):
        query_chunk = query[:, :, index]
        key_chunk = key[:, :, index]
        value_chunk = value[:, :, index]
        attention = (query_chunk @ key_chunk.transpose(-1, -2)) * decay_mask[
            :, :, index
        ]
        value_prime = key_cumulative_decay[:, :, index] @ state
        value_new = value_chunk - value_prime
        attention_inter = (
            query_chunk * decay[:, :, index, :, None].exp()
        ) @ state
        core[:, :, index] = attention_inter + attention @ value_new
        state = (
            state * decay[:, :, index, -1, None, None].exp()
            + (
                key_chunk
                * (
                    decay[:, :, index, -1, None] - decay[:, :, index]
                ).exp()[..., None]
            ).transpose(-1, -2)
            @ value_new
        )
    core = core.reshape(batch_size, num_heads, -1, value_dim)
    core = core[:, :, :sequence_length]
    return core.transpose(1, 2).contiguous().to(initial_dtype), state


def enable_reference_gdn_chunk(model: Qwen35ForCausalLM) -> None:

    def forward(
        self,
        hidden_states: torch.Tensor,
        recurrent_state: torch.Tensor | None = None,
        conv_state: torch.Tensor | None = None,
        return_final_state: bool = False,
    ):
        batch_size, seq_len, _ = hidden_states.shape
        mixed_qkv = self.in_proj_qkv(hidden_states)
        mixed_qkv, final_conv_state = self._causal_conv(mixed_qkv, conv_state)
        query, key, value = torch.split(
            mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1
        )
        query = query.reshape(
            batch_size, seq_len, self.num_k_heads, self.head_k_dim
        )
        key = key.reshape(batch_size, seq_len, self.num_k_heads, self.head_k_dim)
        value = value.reshape(
            batch_size, seq_len, self.num_v_heads, self.head_v_dim
        )
        if self.num_v_heads // self.num_k_heads > 1:
            repeats = self.num_v_heads // self.num_k_heads
            query = query.repeat_interleave(repeats, dim=2)
            key = key.repeat_interleave(repeats, dim=2)
        beta = self.in_proj_b(hidden_states).sigmoid()
        decay = -self.A_log.float().exp() * F.softplus(
            self.in_proj_a(hidden_states).float() + self.dt_bias
        )
        core, final_state = reference_chunk_gated_delta_rule(
            query,
            key,
            value,
            decay,
            beta,
            recurrent_state,
        )
        z = self.in_proj_z(hidden_states).reshape(
            batch_size, seq_len, self.num_v_heads, self.head_v_dim
        )
        core = self.norm(
            core.reshape(-1, self.head_v_dim), z.reshape(-1, self.head_v_dim)
        )
        output = self.out_proj(core.reshape(batch_size, seq_len, self.value_dim))
        if return_final_state:
            return output, final_state.detach().clone(), final_conv_state
        return output

    for layer in model.model.layers:
        if layer.block_type == "linear_attention":
            layer.linear_attn.forward = MethodType(forward, layer.linear_attn)


def materialize_target(snapshot: Path, device: torch.device, dtype: torch.dtype):
    raw = load_qwen35_config_dict(str(snapshot))
    config = normalize_qwen35_text_config(raw)
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        with torch.device("meta"):
            model = Qwen35ForCausalLM(config)
    finally:
        torch.set_default_dtype(previous_dtype)
    model.to_empty(device=device)
    model.eval()
    return model


def materialize_draft(snapshot: Path, device: torch.device, dtype: torch.dtype):
    config = DFlash2Config.from_pretrained_path(str(snapshot))
    config.validate_contract()
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        with torch.device("meta"):
            model = DFlash2DraftModel(config)
    finally:
        torch.set_default_dtype(previous_dtype)
    model.to_empty(device=device)
    model.eval()
    return model


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    args = parse_args()
    if not args.target_model:
        raise SystemExit("Pass --target-model or set MINIVLLM_QWEN38_PATH")
    if not args.draft_model:
        raise SystemExit("Pass --draft-model or set MINIVLLM_DFLASH2_PATH")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for DFlash2 standalone proposal smoke")

    device = torch.device(args.device)
    dtype = DTYPES[args.dtype]
    target_snapshot = Path(args.target_model).expanduser()
    draft_snapshot = Path(args.draft_model).expanduser()
    safe_open_device = str(device) if args.safe_open_device == "cuda" else "cpu"
    if args.reference_gated_rmsnorm_rounding:
        enable_reference_gated_rmsnorm_rounding()
    if args.reference_gdn_chunk and not args.dense_target_prefill_for_proposal:
        raise SystemExit(
            "--reference-gdn-chunk requires --dense-target-prefill-for-proposal"
        )

    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    print(f"target_snapshot={target_snapshot}", flush=True)
    print(f"draft_snapshot={draft_snapshot}", flush=True)
    print(f"device={device}", flush=True)
    print(f"dtype={dtype}", flush=True)
    print(f"safe_open_device={safe_open_device}", flush=True)
    print(
        "reference_gated_rmsnorm_rounding="
        f"{args.reference_gated_rmsnorm_rounding}",
        flush=True,
    )
    print(
        "dense_target_prefill_for_proposal="
        f"{args.dense_target_prefill_for_proposal}",
        flush=True,
    )
    print(f"reference_gdn_chunk={args.reference_gdn_chunk}", flush=True)
    print(f"prompt={args.prompt!r}", flush=True)
    print(f"workload_json={args.workload_json}", flush=True)
    print(f"request_id={args.request_id}", flush=True)
    print_cuda_memory("before_models", device)

    started = time.time()
    target = materialize_target(target_snapshot, device, dtype)
    load_weights_from_checkpoint(
        target,
        str(target_snapshot),
        strict_language=True,
        stream_safetensors=True,
        safe_open_device=safe_open_device,
    )
    print(f"target_loaded_elapsed={time.time() - started:.2f}s", flush=True)
    print_cuda_memory("after_target_load", device)

    started = time.time()
    draft = materialize_draft(draft_snapshot, device, dtype)
    load_dflash2_weights_from_safetensors(
        draft,
        str(draft_snapshot),
        safe_open_device=safe_open_device,
    )
    print(f"draft_loaded_elapsed={time.time() - started:.2f}s", flush=True)
    print_cuda_memory("after_draft_load", device)

    tensor_capture: dict[str, torch.Tensor] = {}
    draft_layer_trace: list[dict[str, object]] = []
    original_select_path = draft.candidate_selector.select_path

    def capture_select_path(unary_logits, proposal_hidden, anchor_token_ids):
        tensor_capture["proposal_hidden"] = proposal_hidden.detach().cpu()
        return original_select_path(unary_logits, proposal_hidden, anchor_token_ids)

    draft.candidate_selector.select_path = capture_select_path

    def wrap_draft_layer(layer_index: int, original_forward):
        def forward(
            self,
            hidden_states,
            query_positions,
            context_states=None,
            context_positions=None,
            context_kv=None,
            context_valid_mask=None,
        ):
            output = original_forward(
                hidden_states,
                query_positions,
                context_states=context_states,
                context_positions=context_positions,
                context_kv=context_kv,
                context_valid_mask=context_valid_mask,
            )
            draft_layer_trace.append(
                {
                    "layer_index": layer_index,
                    "positions": query_positions.detach().cpu(),
                    "input": hidden_states.detach().cpu(),
                    "output": output.detach().cpu(),
                }
            )
            return output

        return forward

    for layer_index, layer in enumerate(draft.layers):
        layer.forward = MethodType(
            wrap_draft_layer(layer_index, layer.forward),
            layer,
        )

    tokenizer = AutoTokenizer.from_pretrained(str(target_snapshot))
    prompt, token_ids = load_prompt(args, tokenizer)
    if not token_ids:
        raise RuntimeError("prompt encoded to no tokens")
    input_ids = torch.tensor(token_ids, dtype=torch.long, device=device)
    cache = HybridCacheManager()
    trace_layers = tuple(range(target.model.config.num_hidden_layers))
    with torch.inference_mode():
        cached_target_output = target.forward_with_cache(
            input_ids,
            cache,
            "standalone",
            capture_hidden_taps=not args.dense_target_prefill_for_proposal,
            hidden_tap_layers=trace_layers,
        )
        if args.dense_target_prefill_for_proposal:
            if args.reference_gdn_chunk:
                enable_reference_gdn_chunk(target)
            target_output = target(
                input_ids.unsqueeze(0),
                capture_hidden_taps=True,
                hidden_tap_layers=trace_layers,
                return_dict=True,
            )
        else:
            target_output = cached_target_output
        hidden_taps = {
            layer_id: target_output.hidden_taps[layer_id][
                :, -min(target_output.hidden_taps[layer_id].shape[1], draft.config.sliding_window) :, :
            ]
            for layer_id in draft.config.target_layer_ids
        }
        if args.hidden_taps_tensor_dump:
            injected = torch.load(
                args.hidden_taps_tensor_dump,
                map_location=device,
                weights_only=False,
            )["hidden_taps"]
            if isinstance(injected, dict):
                hidden_taps = {
                    int(layer_id): value if value.dim() == 3 else value.unsqueeze(0)
                    for layer_id, value in injected.items()
                }
            else:
                hidden_taps = {
                    int(layer_id): value if value.dim() == 3 else value.unsqueeze(0)
                    for layer_id, value in zip(
                        draft.config.target_layer_ids, injected, strict=True
                    )
                }
        target_layer_boundary_trace = [
            {
                "model_layer_index": layer_id,
                "boundary_index": layer_id + 1,
                **tensor_fingerprint(value),
            }
            for layer_id, value in target_output.hidden_taps.items()
        ]
        anchor_token_id = int(target_output.logits[:, -1, :].argmax(dim=-1).item())
        anchor_position = len(token_ids)
        proposal_batch, debug = draft.forward_block(
            anchor_token_ids=torch.tensor(
                [anchor_token_id],
                dtype=torch.long,
                device=device,
            ),
            anchor_positions=torch.tensor(
                [anchor_position],
                dtype=torch.long,
                device=device,
            ),
            hidden_taps=hidden_taps,
            token_embedding=target.model.embed_tokens,
            lm_head=target.lm_head,
            return_debug=True,
        )

    valid_length = int(proposal_batch.valid_lengths[0].item())
    proposed = proposal_batch.token_ids[0, :valid_length].tolist()
    anchor_top = target_output.logits[:, -1, :].float().topk(2, dim=-1)
    with torch.inference_mode():
        verification_output = target.forward_with_cache(
            torch.tensor(
                [anchor_token_id, *proposed],
                dtype=torch.long,
                device=device,
            ),
            cache,
            "standalone",
        )
    verification_top = verification_output.logits[0].float().topk(2, dim=-1)
    verification_top_token_ids = verification_top.indices.cpu().tolist()
    verification_top_logits = verification_top.values.cpu().tolist()
    proposal_target_top1 = [
        int(values[0]) for values in verification_top_token_ids[:valid_length]
    ]
    first_rejection_index = next(
        (
            index
            for index, (proposal_id, target_id) in enumerate(
                zip(proposed, proposal_target_top1)
            )
            if int(proposal_id) != target_id
        ),
        None,
    )
    debug_json = {
        key: list(value) if isinstance(value, tuple) else value
        for key, value in debug.items()
    }
    fixture = {
        "fixture_schema_version": 2,
        "anchor_semantics": "target_bonus_token",
        "source": "minivllm",
        "reference_gated_rmsnorm_rounding": (
            args.reference_gated_rmsnorm_rounding
        ),
        "dense_target_prefill_for_proposal": (
            args.dense_target_prefill_for_proposal
        ),
        "reference_gdn_chunk": args.reference_gdn_chunk,
        "hidden_taps_tensor_dump": args.hidden_taps_tensor_dump,
        "prompt": prompt,
        "prompt_token_ids": token_ids,
        "anchor_token_id": anchor_token_id,
        "anchor_position": anchor_position,
        "target_layer_ids": list(draft.config.target_layer_ids),
        "hidden_tap_checksums": {
            str(layer_id): float(value.float().sum().item())
            for layer_id, value in hidden_taps.items()
        },
        "target_layer_boundary_trace": target_layer_boundary_trace,
        "target_anchor_top_token_ids": anchor_top.indices[0].cpu().tolist(),
        "target_anchor_top_logits": anchor_top.values[0].cpu().tolist(),
        "proposal_valid_length": valid_length,
        "proposal_token_ids": proposed,
        "verification_top_token_ids": verification_top_token_ids,
        "verification_top_logits": verification_top_logits,
        "proposal_target_top1": proposal_target_top1,
        "first_rejection_index": first_rejection_index,
        **debug_json,
    }
    print(f"prompt_token_ids={token_ids}", flush=True)
    print(f"anchor_token_id={anchor_token_id}", flush=True)
    print(f"anchor_position={anchor_position}", flush=True)
    print(f"hidden_tap_keys={sorted(hidden_taps)}", flush=True)
    print(f"proposal_valid_lengths={proposal_batch.valid_lengths.tolist()}", flush=True)
    print(f"proposal_token_ids={proposed}", flush=True)
    print(f"proposal_target_top1={proposal_target_top1}", flush=True)
    print(f"first_rejection_index={first_rejection_index}", flush=True)
    for key, value in debug.items():
        print(f"{key}={value}", flush=True)
    if args.json_output:
        output_path = Path(args.json_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(fixture, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"json_output={output_path}", flush=True)
    if args.tensor_dump:
        tensor_dump_path = Path(args.tensor_dump)
        tensor_dump_path.parent.mkdir(parents=True, exist_ok=True)
        with torch.inference_mode():
            normalized_context = draft.combine_hidden_taps(hidden_taps)
        torch.save(
            {
                "hidden_taps": {
                    int(layer_id): value.detach().cpu()
                    for layer_id, value in hidden_taps.items()
                },
                "target_layer_boundaries": {
                    int(layer_id + 1): value.detach().cpu()
                    for layer_id, value in target_output.hidden_taps.items()
                },
                "draft_layer_trace": draft_layer_trace,
                "normalized_context": normalized_context.detach().cpu(),
                "proposal_hidden": tensor_capture.get("proposal_hidden"),
            },
            tensor_dump_path,
        )
        print(f"tensor_dump={tensor_dump_path}", flush=True)
    print_cuda_memory("after_proposal", device)
    print("dflash2_standalone_proposal=PASS", flush=True)
    print("dflash2_oracle_alignment=PENDING", flush=True)


if __name__ == "__main__":
    main()
