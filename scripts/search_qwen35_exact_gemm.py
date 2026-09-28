"""Search A800 BF16 GEMM layouts matching the cached one-token result."""

import argparse

import torch
import torch.nn.functional as F


@torch.inference_mode()
def check(in_features: int, out_features: int, *, batch: int = 2,
          exhaustive: bool = False) -> None:
    torch.manual_seed(20260928)
    tokens = 8
    x = torch.randn(batch, tokens, in_features, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(out_features, in_features, device="cuda", dtype=torch.bfloat16)
    reference = torch.cat([
        F.linear(x[:, index : index + 1].contiguous(), weight)
        for index in range(tokens)
    ], dim=1)

    def report(name: str, result: torch.Tensor) -> None:
        diff = (reference.float() - result.float()).abs()
        count = int(torch.count_nonzero(reference != result))
        print(f"{in_features}->{out_features} {name}: {count}/{reference.numel()} max={diff.max().item():.5f}", flush=True)

    for query_chunk in (2, 3, 4, 5, 6, 7, 8):
        result = torch.cat([
            F.linear(x[:, start : start + query_chunk].contiguous(), weight)
            for start in range(0, tokens, query_chunk)
        ], dim=1)
        report(f"query_chunk={query_chunk}", result)

    if not exhaustive:
        return

    for padded_batch in (4, 8, 16):
        padded = torch.cat([
            x, x.new_zeros(padded_batch - batch, tokens, in_features)
        ], dim=0)
        report(f"padded_batch={padded_batch}", F.linear(padded, weight)[:batch])

    for padded_output in (6144, 8192, 10240, 11264, 12288, 16384, 17408, 20480):
        if padded_output <= out_features:
            continue
        padded_weight = torch.cat([
            weight, weight.new_zeros(padded_output - out_features, in_features)
        ], dim=0)
        report(f"padded_output={padded_output}", F.linear(x, padded_weight)[..., :out_features])
        del padded_weight

    for output_chunk in (1024, 2048, 4096, 5120, 6144, 8192):
        if output_chunk >= out_features:
            continue
        report(f"output_chunk={output_chunk}", torch.cat([
            F.linear(x, part) for part in weight.split(output_chunk, dim=0)
        ], dim=-1))

    report("matmul", torch.matmul(x, weight.T))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=2)
    args = parser.parse_args()
    for shape in (
        (5120, 5120), (5120, 17408), (5120, 6144),
        (5120, 12288), (5120, 1024), (6144, 5120),
        (5120, 10240), (17408, 5120),
    ):
        check(*shape, batch=args.batch)
