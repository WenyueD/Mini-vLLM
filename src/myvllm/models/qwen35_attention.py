"""Fixed-row cached attention for one-token decode and short-block verification.

QK and PV use identical per-row accumulation for all query lengths (1..8).
Historical KV is read directly, without padding copies or GQA expansion.
"""
from __future__ import annotations

import torch
try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


if triton is not None:
    @triton.jit(do_not_specialize=["lengths", "prefixes"])
    def _cached_qk(
        q_ptr, keys, lengths, prefixes, scores,
        B: tl.constexpr, H: tl.constexpr, HK: tl.constexpr,
        Q: tl.constexpr, D: tl.constexpr, L: tl.constexpr,
        QB: tl.constexpr, QT: tl.constexpr, QH: tl.constexpr,
        SCALE: tl.constexpr, BN: tl.constexpr = 64, BK: tl.constexpr = 32,
    ):
        tile = tl.program_id(0)
        bh = tl.program_id(1)
        batch, head = bh // H, bh % H
        key = keys[0]
        length, prefix = lengths[0], prefixes[0]
        for row in tl.static_range(1, B):
            if batch == row:
                key = keys[row]
                length, prefix = lengths[row], prefixes[row]
        qi = tl.arange(0, 16)
        ni = tile * BN + tl.arange(0, BN)
        ki = tl.arange(0, BK)
        acc = tl.full((16, BN), 0, tl.float32)
        for start in range(tl.cdiv(D, BK)):
            di = start * BK + ki
            q = tl.load(
                q_ptr + batch * QB + qi[:, None] * QT + head * QH + di[None, :],
                (qi[:, None] < Q) & (di[None, :] < D), other=0,
            )
            k = tl.load(
                key + ni[None, :] * HK * D + (head // (H // HK)) * D + di[:, None],
                (ni[None, :] < length) & (di[:, None] < D), other=0,
            )
            acc = tl.dot(q, k, acc)
        # Preserve the existing two BF16 rounding boundaries before softmax.
        rounded = acc.to(q_ptr.dtype.element_ty).to(tl.float32)
        scaled = (rounded * SCALE).to(q_ptr.dtype.element_ty)
        valid = (ni[None, :] < length) & (ni[None, :] < prefix + qi[:, None] + 1)
        scaled = tl.where(valid, scaled, -float("inf"))
        tl.store(
            scores + bh * Q * L + qi[:, None] * L + ni[None, :],
            scaled, (qi[:, None] < Q) & (ni[None, :] < L),
        )


    @triton.jit(do_not_specialize=["lengths"])
    def _cached_pv(
        probs, values, lengths, output,
        B: tl.constexpr, H: tl.constexpr, HK: tl.constexpr,
        Q: tl.constexpr, D: tl.constexpr, L: tl.constexpr,
        BD: tl.constexpr = 64, BK: tl.constexpr = 64,
    ):
        tile = tl.program_id(0)
        bh = tl.program_id(1)
        batch, head = bh // H, bh % H
        value, length = values[0], lengths[0]
        for row in tl.static_range(1, B):
            if batch == row:
                value, length = values[row], lengths[row]
        qi = tl.arange(0, 16)
        di = tile * BD + tl.arange(0, BD)
        ki = tl.arange(0, BK)
        acc = tl.full((16, BD), 0, tl.float32)
        # Keep reduction tiles fixed across decode and verification.
        for start in range(tl.cdiv(L, BK)):
            ni = start * BK + ki
            p = tl.load(
                probs + bh * Q * L + qi[:, None] * L + ni[None, :],
                (qi[:, None] < Q) & (ni[None, :] < L), other=0,
            )
            v = tl.load(
                value + ni[:, None] * HK * D + (head // (H // HK)) * D + di[None, :],
                (ni[:, None] < length) & (di[None, :] < D), other=0,
            )
            acc = tl.dot(p, v, acc)
        tl.store(
            output + batch * Q * H * D + qi[:, None] * H * D + head * D + di[None, :],
            acc.to(output.dtype.element_ty),
            (qi[:, None] < Q) & (di[None, :] < D),
        )


def short_cached_attention(query, cached_rows, prefix_lengths, pad_to_length):
    """Return [B,Q,H,D], reading each request's compact [1,L,HK,D] KV."""
    if triton is None or not query.is_cuda or query.dtype != torch.bfloat16:
        raise RuntimeError("short cached attention requires CUDA BF16 Triton")
    batch, query_length, heads, dim = query.shape
    if not 1 <= query_length <= 8:
        raise ValueError("short cached attention supports 1..8 query positions")
    if len(cached_rows) != batch or len(prefix_lengths) != batch:
        raise ValueError("short attention request metadata does not match batch")
    kv_heads = cached_rows[0][0].shape[2]
    if heads % kv_heads:
        raise ValueError("query heads must be divisible by KV heads")
    lengths = tuple(int(key.shape[1]) for key, _ in cached_rows)
    length = max(max(lengths), int(pad_to_length or 0))
    for (key, value), prefix, context in zip(cached_rows, prefix_lengths, lengths):
        if key.shape != value.shape or key.shape != (1, context, kv_heads, dim):
            raise ValueError("short attention KV shape mismatch")
        if not key.is_contiguous() or not value.is_contiguous():
            raise ValueError("short attention requires contiguous per-request KV")
        if key.device != query.device or value.device != query.device:
            raise ValueError("short attention KV device mismatch")
        if key.dtype != query.dtype or value.dtype != query.dtype:
            raise ValueError("short attention KV dtype mismatch")
        if prefix < 0 or prefix + query_length != context:
            raise ValueError("short attention prefix length mismatch")
    scores = torch.empty((batch, heads, query_length, length),
                         device=query.device, dtype=query.dtype)
    _cached_qk[(triton.cdiv(length, 64), batch * heads)](
        query, tuple(k for k, _ in cached_rows), lengths, tuple(prefix_lengths), scores,
        batch, heads, kv_heads, query_length, dim, length,
        query.stride(0), query.stride(1), query.stride(2), dim ** -0.5,
        num_warps=4, enable_fp_fusion=False,
    )
    probs = torch.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
    output = torch.empty_like(query, memory_format=torch.contiguous_format)
    _cached_pv[(triton.cdiv(dim, 64), batch * heads)](
        probs, tuple(v for _, v in cached_rows), lengths, output,
        batch, heads, kv_heads, query_length, dim, length,
        num_warps=4, enable_fp_fusion=False,
    )
    return output
