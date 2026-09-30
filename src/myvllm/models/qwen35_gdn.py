"""Fused cached GDN recurrence with compact prefix-commit records.

Each program owns a value-column tile of one request/head's FP32 state.
Candidate positions advance causally inside that program. No per-position
state matrices are allocated. Target decode and block verify use the same
reduction layout; disabling FP fusion also makes prefix replay bit-exact.
"""
import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


if triton is not None:
    @triton.jit
    def _mul_rn(left, right):
        return tl.inline_asm_elementwise(
            "mul.rn.f32 $0, $1, $2;", constraints="=f,f,f",
            args=[left, right], dtype=tl.float32, is_pure=True, pack=1,
        )

    @triton.jit
    def _add_rn(left, right):
        return tl.inline_asm_elementwise(
            "add.rn.f32 $0, $1, $2;", constraints="=f,f,f",
            args=[left, right], dtype=tl.float32, is_pure=True, pack=1,
        )

    @triton.jit(do_not_specialize=["steps"])
    def _cached_gdn(
        q_ptr, k_ptr, v_ptr, beta_ptr, decay_ptr, state_ptr,
        out_ptr, final_ptr, delta_ptr,
        steps, heads: tl.constexpr,
        key_dim: tl.constexpr, value_dim: tl.constexpr,
        block_k: tl.constexpr, block_v: tl.constexpr,
    ):
        bh = tl.program_id(0)
        batch = bh // heads
        head = bh % heads
        keys = tl.arange(0, block_k)
        vals = tl.program_id(1) * block_v + tl.arange(0, block_v)
        state_offsets = bh * key_dim * value_dim + keys[:, None] * value_dim + vals[None, :]
        state_mask = (keys[:, None] < key_dim) & (vals[None, :] < value_dim)
        state = tl.load(state_ptr + state_offsets, state_mask, other=0).to(tl.float32)
        for step in range(steps):
            row = (batch * steps + step) * heads + head
            q = tl.load(q_ptr + row * key_dim + keys, keys < key_dim, other=0).to(tl.float32)
            k = tl.load(k_ptr + row * key_dim + keys, keys < key_dim, other=0).to(tl.float32)
            v = tl.load(v_ptr + row * value_dim + vals, vals < value_dim, other=0).to(tl.float32)
            beta = tl.load(beta_ptr + row).to(tl.float32)
            decay = tl.load(decay_ptr + row).to(tl.float32)
            # Explicit round-to-nearest boundaries prevent reassociation when
            # delta is consumed only internally (decode) versus recorded (verify).
            state = _mul_rn(state, decay)
            memory = tl.sum(_mul_rn(state, k[:, None]), axis=0)
            delta = _mul_rn(_add_rn(v, -memory), beta)
            state = _add_rn(state, _mul_rn(k[:, None], delta[None, :]))
            out = tl.sum(_mul_rn(state, q[:, None]), axis=0)
            tl.store(out_ptr + row * value_dim + vals, out, vals < value_dim)
            tl.store(delta_ptr + row * value_dim + vals, delta, vals < value_dim)
        tl.store(final_ptr + state_offsets, state, state_mask)


def fused_cached_gdn(query, key, value, beta, decay, state, *, record_updates=False):
    """Inputs are [B,T,H,D], scalar gates [B,T,H]; decay is already exp(g).

    Return BF16/FP32 core, a new FP32 final state, and optional FP32 deltas.
    The input state is never mutated, including when a caller later rolls back.
    """
    if triton is None or not query.is_cuda:
        raise RuntimeError("fused cached GDN requires CUDA Triton")
    batch, steps, heads, key_dim = query.shape
    value_dim = value.shape[-1]
    if steps < 1 or key.shape != query.shape or value.shape[:3] != query.shape[:3]:
        raise ValueError("incompatible cached GDN query/key/value shapes")
    if beta.shape != query.shape[:3] or decay.shape != beta.shape:
        raise ValueError("cached GDN gates must have shape [batch, steps, heads]")
    if state.shape != (batch, heads, key_dim, value_dim) or state.dtype != torch.float32:
        raise ValueError("cached GDN state must have shape [batch, heads, key, value] in FP32")
    output = torch.empty_like(value, memory_format=torch.contiguous_format)
    final = torch.empty_like(state, memory_format=torch.contiguous_format)
    # Keep identical compiled reductions in decode and verification. Omitting
    # this small store lets the compiler pick a different reduction layout.
    deltas = torch.empty(value.shape, device=value.device, dtype=torch.float32)
    _cached_gdn[(batch * heads, triton.cdiv(value_dim, 16))](
        query.contiguous(), key.contiguous(), value.contiguous(), beta.contiguous(),
        decay.contiguous(), state.contiguous(), output, final, deltas,
        steps, heads, key_dim, value_dim, triton.next_power_of_2(key_dim), 16,
        num_warps=4, enable_fp_fusion=False,
    )
    return output, final, deltas if record_updates else None
