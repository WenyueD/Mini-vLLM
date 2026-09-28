"""Probe whether a batched Triton BF16 GEMM matches cached cuBLAS bitwise."""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def gemm(A, W, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
         BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    k = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for offset in range(tl.cdiv(K, BK)):
        kk = offset * BK + k
        a = tl.load(A + m[:, None] * K + kk[None, :],
                    (m[:, None] < M) & (kk[None, :] < K), other=0)
        w = tl.load(W + n[None, :] * K + kk[:, None],
                    (n[None, :] < N) & (kk[:, None] < K), other=0)
        acc = tl.dot(a, w, acc)
    tl.store(C + m[:, None] * N + n[None, :], acc.to(tl.bfloat16),
             (m[:, None] < M) & (n[None, :] < N))


def run(in_features, out_features):
    torch.manual_seed(20260928)
    batch, tokens = 2, 8
    x = torch.randn(batch, tokens, in_features, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(out_features, in_features, device="cuda", dtype=torch.bfloat16)
    reference = torch.cat([
        F.linear(x[:, i : i + 1].contiguous(), weight) for i in range(tokens)
    ], dim=1)
    flat = x.reshape(batch * tokens, in_features)
    for bm in (16, 32):
        for bn in (32, 64, 128):
            for bk in (32, 64, 128):
                result = torch.empty(
                    batch * tokens, out_features, device="cuda", dtype=torch.bfloat16,
                )
                try:
                    gemm[(triton.cdiv(batch * tokens, bm), triton.cdiv(out_features, bn))](
                        flat, weight, result, batch * tokens, out_features,
                        in_features, bm, bn, bk, num_warps=4, num_stages=2,
                    )
                    result = result.reshape(batch, tokens, out_features)
                    different = int(torch.count_nonzero(result != reference))
                    print(
                        f"{in_features}->{out_features} BM={bm} BN={bn} BK={bk} "
                        f"different={different}/{reference.numel()}",
                        flush=True,
                    )
                except Exception as exc:
                    print(f"BM={bm} BN={bn} BK={bk} error={exc}", flush=True)


if __name__ == "__main__":
    run(5120, 10240)
    run(17408, 5120)
