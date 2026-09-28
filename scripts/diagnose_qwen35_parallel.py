"""Compare Qwen3.8 serial and causal-block target verification layer by layer."""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from myvllm.engine.hybrid_cache import HybridCacheManager
from myvllm.models.qwen3_8 import Qwen35ForCausalLM
from myvllm.models.qwen3_8_config import (
    load_qwen35_config_dict,
    normalize_qwen35_text_config,
)
from myvllm.utils.loader import load_weights_from_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--request", default="short-000003")
    parser.add_argument("--query", type=int, nargs="+", default=[1857, 220, 18, 26, 271])
    parser.add_argument("--pad-to", type=int, default=256)
    parser.add_argument("--micro-batch", type=int, default=2)
    parser.add_argument("--fp32-linears", action="store_true")
    parser.add_argument("--fp32-kernels", action="store_true")
    parser.add_argument("--full-fp32-activations", action="store_true")
    parser.add_argument("--serial-token-kernels", action="store_true")
    parser.add_argument("--trace-layer0", action="store_true")
    parser.add_argument("--batched-mlp", action="store_true")
    parser.add_argument("--batched-safe-linears", action="store_true")
    args = parser.parse_args()

    torch.set_default_dtype(torch.bfloat16)
    torch.cuda.set_device(0)
    config = normalize_qwen35_text_config(load_qwen35_config_dict(args.model))
    with torch.device("meta"):
        model = Qwen35ForCausalLM(config)
    model.to_empty(device="cuda:0")
    model.enable_experimental_batched_linears(
        mlp=args.batched_mlp, safe_shapes=args.batched_safe_linears,
    )
    load_weights_from_checkpoint(
        model, args.model, strict_language=True, stream_safetensors=True,
        safe_open_device="cuda:0",
    )
    model.eval()
    prompt = json.loads(args.workload.read_text())[args.request]["prompt_token_ids"]
    serial_cache = HybridCacheManager()
    parallel_cache = HybridCacheManager()
    all_layers = tuple(range(config.num_hidden_layers))
    ids = torch.tensor([args.query], device="cuda:0", dtype=torch.long)
    with torch.inference_mode():
        model.forward_with_cache(
            torch.tensor(prompt, device="cuda:0", dtype=torch.long),
            serial_cache, "probe", return_logits=False,
        )
        parallel_cache._states["probe"] = serial_cache.get_request_state("probe").clone()
        if args.fp32_linears or args.fp32_kernels or args.full_fp32_activations:
            native_linear = F.linear

            def fp32_linear(input, weight, bias=None):
                output = native_linear(
                    input.float(), weight.float(),
                    None if bias is None else bias.float(),
                )
                return output if args.full_fp32_activations else output.to(input.dtype)

            F.linear = fp32_linear
        if args.fp32_kernels or args.full_fp32_activations:
            native_conv1d = F.conv1d
            native_matmul = torch.matmul

            def fp32_conv1d(input, weight, bias=None, *conv_args, **conv_kwargs):
                output = native_conv1d(
                    input.float(), weight.float(),
                    None if bias is None else bias.float(),
                    *conv_args, **conv_kwargs,
                )
                return output if args.full_fp32_activations else output.to(input.dtype)

            def fp32_matmul(input, other, *mm_args, **mm_kwargs):
                output = native_matmul(
                    input.float(), other.float(), *mm_args, **mm_kwargs,
                )
                return output if args.full_fp32_activations else output.to(input.dtype)

            F.conv1d = fp32_conv1d
            torch.matmul = fp32_matmul
        if args.full_fp32_activations:
            native_embedding_forward = model.model.embed_tokens.forward
            model.model.embed_tokens.forward = (
                lambda input: native_embedding_forward(input).float()
            )
        if args.serial_token_kernels:
            native_linear = F.linear
            native_conv1d = F.conv1d
            native_matmul = torch.matmul

            def token_linear(input, weight, bias=None):
                if input.ndim == 3 and input.shape[1] > 1:
                    return torch.cat([
                        native_linear(input[:, index : index + 1].contiguous(), weight, bias)
                        for index in range(input.shape[1])
                    ], dim=1)
                return native_linear(input, weight, bias)

            def token_conv1d(input, weight, bias=None, *conv_args, **conv_kwargs):
                token_count = input.shape[-1] - weight.shape[-1] + 1
                if token_count > 1:
                    return torch.cat([
                        native_conv1d(
                            input[..., index : index + weight.shape[-1]].contiguous(),
                            weight, bias, *conv_args, **conv_kwargs,
                        )
                        for index in range(token_count)
                    ], dim=-1)
                return native_conv1d(input, weight, bias, *conv_args, **conv_kwargs)

            def token_matmul(input, other, *mm_args, **mm_kwargs):
                if input.ndim == 4 and input.shape[-2] > 1:
                    return torch.cat([
                        native_matmul(
                            input[..., index : index + 1, :].contiguous(),
                            other, *mm_args, **mm_kwargs,
                        )
                        for index in range(input.shape[-2])
                    ], dim=-2)
                return native_matmul(input, other, *mm_args, **mm_kwargs)

            F.linear = token_linear
            F.conv1d = token_conv1d
            torch.matmul = token_matmul
        traces = {"serial": {}, "parallel": {}}
        phase = ["serial"]
        handles = []
        if args.trace_layer0:
            layer = model.model.layers[0]
            modules = {
                "input_norm": layer.input_layernorm,
                "gdn_qkv": layer.linear_attn.in_proj_qkv,
                "gdn_norm": layer.linear_attn.norm,
                "gdn_out": layer.linear_attn.out_proj,
                "post_norm": layer.post_attention_layernorm,
                "mlp_gate": layer.mlp.gate_proj,
                "mlp_up": layer.mlp.up_proj,
                "mlp_down": layer.mlp.down_proj,
            }

            def make_hook(name):
                def hook(_module, _inputs, output):
                    traces[phase[0]].setdefault(name, []).append(output.detach().clone())
                return hook

            handles = [module.register_forward_hook(make_hook(name))
                       for name, module in modules.items()]
        serial = model.forward_with_cache_batch(
            ids, [len(args.query)], serial_cache, ["probe"],
            capture_hidden_taps=True, hidden_tap_layers=all_layers,
            attention_pad_to_length=args.pad_to, batch_pad_to_size=args.micro_batch,
        )[0]
        phase[0] = "parallel"
        parallel = model.forward_with_cache_parallel_batch(
            ids, [len(args.query)], parallel_cache, ["probe"],
            capture_hidden_taps=True, hidden_tap_layers=all_layers,
            attention_pad_to_length=args.pad_to, batch_pad_to_size=args.micro_batch,
        )[0]
        for handle in handles:
            handle.remove()
    if args.trace_layer0:
        for name in modules:
            serial_items = traces["serial"][name]
            parallel_items = traces["parallel"][name]
            for pos in (0, min(3, len(args.query) - 1)):
                a = serial_items[pos][0].float()
                if len(parallel_items) == len(args.query):
                    b = parallel_items[pos][0].float()
                elif parallel_items[0].ndim == 2:
                    per_request_rows = serial_items[0].shape[0] // 2
                    b = parallel_items[0].reshape(
                        2, len(args.query), per_request_rows, -1,
                    )[0, pos, 0].float()
                else:
                    b = parallel_items[0][0, pos].float()
                diff = (a - b).abs()
                print(f"layer0 {name} position={pos} max={diff.max().item():.7f} mean={diff.mean().item():.7f}", flush=True)
    for pos, token in enumerate(args.query):
        s = serial.logits[0, pos].float()
        p = parallel.logits[0, pos].float()
        print(
            f"position={pos} input={token} serial_top={int(s.argmax())} "
            f"parallel_top={int(p.argmax())} max_logit_diff={(s-p).abs().max().item():.6f}",
            flush=True,
        )
        for layer in all_layers:
            a = serial.hidden_taps[layer][0, pos].float()
            b = parallel.hidden_taps[layer][0, pos].float()
            diff = (a - b).abs()
            if pos == 3 or layer < 8 or layer % 8 == 7 or layer == all_layers[-1]:
                print(
                    f"  layer={layer:02d} max={diff.max().item():.6f} "
                    f"mean={diff.mean().item():.6f}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
