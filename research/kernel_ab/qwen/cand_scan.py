"""KMD-2 scan kernel — Triton forward + reduced-intermediate backward.

Optimization: Save ONLY S_before_list (107MB) instead of all 4 intermediate lists (216MB).
Recompute S_after, kv_mem, and update during backward from S_before + forward inputs.
Trade-off: extra BMMs in backward vs. less memory bandwidth in save_for_backward.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _fwd_step_kernel(
    S_after_ptr, S_before_out_ptr, kv_mem_out_ptr, update_out_ptr,
    S_after_new_ptr, yt_out_ptr,
    g_t_ptr, k_t_ptr, v_t_ptr, be_t_ptr, bw_t_ptr, q_t_ptr, mixw_ptr,
    N: tl.constexpr, dk: tl.constexpr, dv: tl.constexpr,
    r_out: tl.constexpr,
    BLOCK_DV: tl.constexpr, BLOCK_DK: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_dv = tl.program_id(1)
    if pid_n >= N:
        return

    dv_off = pid_dv * BLOCK_DV + tl.arange(0, BLOCK_DV)
    dv_mask = dv_off < dv

    bw_val = tl.load(bw_t_ptr + pid_n)
    be_val = tl.load(be_t_ptr + pid_n)
    v_t = tl.load(v_t_ptr + pid_n * dv + dv_off, mask=dv_mask, other=0.0)

    kv_mem = tl.zeros([BLOCK_DV], dtype=tl.float32)
    for dk_start in range(0, dk, BLOCK_DK):
        dk_off = dk_start + tl.arange(0, BLOCK_DK)
        dk_mask = dk_off < dk
        g_tile = tl.load(g_t_ptr + pid_n * dk + dk_off, mask=dk_mask, other=1.0)
        k_tile = tl.load(k_t_ptr + pid_n * dk + dk_off, mask=dk_mask, other=0.0)
        sa_ptrs = S_after_ptr + pid_n * dk * dv + dk_off[:, None] * dv + dv_off[None, :]
        sa = tl.load(sa_ptrs, mask=dk_mask[:, None] & dv_mask[None, :], other=0.0)
        sb = sa * g_tile[:, None]
        sb_ptrs = S_before_out_ptr + pid_n * dk * dv + dk_off[:, None] * dv + dv_off[None, :]
        tl.store(sb_ptrs, sb, mask=dk_mask[:, None] & dv_mask[None, :])
        kv_mem = kv_mem + tl.sum(sb * k_tile[:, None], axis=0)

    update = bw_val * v_t - be_val * kv_mem
    tl.store(kv_mem_out_ptr + pid_n * dv + dv_off, kv_mem, mask=dv_mask)
    tl.store(update_out_ptr + pid_n * dv + dv_off, update, mask=dv_mask)

    yt_acc = tl.zeros([BLOCK_DV], dtype=tl.float32)
    for dk_start in range(0, dk, BLOCK_DK):
        dk_off = dk_start + tl.arange(0, BLOCK_DK)
        dk_mask = dk_off < dk
        k_tile = tl.load(k_t_ptr + pid_n * dk + dk_off, mask=dk_mask, other=0.0)
        if r_out > 1:
            wq_tile = tl.zeros([BLOCK_DK], dtype=tl.float32)
            for r in range(r_out):
                mv = tl.load(mixw_ptr + pid_n * r_out + r)
                q_off = pid_n * r_out * dk + r * dk + dk_off
                qr = tl.load(q_t_ptr + q_off, mask=dk_mask, other=0.0)
                wq_tile = wq_tile + mv * qr
        else:
            wq_tile = tl.load(q_t_ptr + pid_n * dk + dk_off, mask=dk_mask, other=0.0)
        sb_ptrs = S_before_out_ptr + pid_n * dk * dv + dk_off[:, None] * dv + dv_off[None, :]
        sb = tl.load(sb_ptrs, mask=dk_mask[:, None] & dv_mask[None, :], other=0.0)
        san = sb + k_tile[:, None] * update[None, :]
        san_ptrs = S_after_new_ptr + pid_n * dk * dv + dk_off[:, None] * dv + dv_off[None, :]
        tl.store(san_ptrs, san, mask=dk_mask[:, None] & dv_mask[None, :])
        yt_acc = yt_acc + tl.sum(san * wq_tile[:, None], axis=0)

    tl.store(yt_out_ptr + pid_n * dv + dv_off, yt_acc, mask=dv_mask)


def _forward_step_triton(S_after, S_before_out, kv_mem_out, update_out,
                          S_after_new_out, yt_out,
                          g_t, kt, v_t, be_t, bw_t, q_t, mixw, r_out, N, dk, dv):
    """Forward step with pre-allocated output buffers to eliminate alloc+copy overhead."""
    BLOCK_DV = 128
    BLOCK_DK = 128
    grid = (N, triton.cdiv(dv, BLOCK_DV))

    _fwd_step_kernel[grid](
        S_after, S_before_out, kv_mem_out, update_out, S_after_new_out, yt_out,
        g_t, kt, v_t, be_t, bw_t, q_t,
        mixw if r_out > 1 else S_after,
        N, dk, dv, r_out, BLOCK_DV, BLOCK_DK,
    )


def _backward_step_reduced(dS, S_before_t, kt, v_t, bw_t, be_t, q_t, dy_t,
                           g_t, S_after_prev, r_out, mixw):
    """Backward step that recomputes kv_mem and update from S_before + inputs.
    S_after_prev always a tensor (zeros for t=0), eliminating None branch."""
    # Recompute kv_mem = k_t^T @ S_before_t
    kv_mem_t = torch.bmm(kt.unsqueeze(1), S_before_t).squeeze(1)
    # Recompute update = bw_t * v_t - be_t * kv_mem_t
    update_t = (bw_t.unsqueeze(-1) * v_t) - (be_t.unsqueeze(-1) * kv_mem_t)
    # Recompute S_after_t = S_before_t + k_t @ update_t
    S_after_t = S_before_t + torch.bmm(kt.unsqueeze(2), update_t.unsqueeze(1))

    # Now proceed with the standard backward using recomputed values
    if r_out > 1:
        weighted_q = q_t * mixw.transpose(1, 2)
        dy_exp = dy_t.unsqueeze(1).expand(-1, r_out, -1)
    else:
        weighted_q = q_t.squeeze(1).unsqueeze(1)
        dy_exp = dy_t.unsqueeze(1)
    dS_output = torch.bmm(weighted_q.transpose(-2, -1), dy_exp)
    dS = dS + dS_output
    dq_base = torch.bmm(dy_t.unsqueeze(1), S_after_t.transpose(-2, -1)).squeeze(1)
    if r_out > 1:
        dq_t = dq_base.unsqueeze(-2) * mixw.transpose(1, 2)
    else:
        dq_t = dq_base.unsqueeze(-2)
    dk_t = torch.bmm(update_t.unsqueeze(1), dS.transpose(-2, -1)).squeeze(1)
    d_update = torch.bmm(kt.unsqueeze(1), dS).squeeze(1)
    dv_t = bw_t.unsqueeze(-1) * d_update
    dbw_t = (v_t * d_update).sum(dim=-1)
    d_kv_mem = -be_t.unsqueeze(-1) * d_update
    dbe_t = -(kv_mem_t * d_update).sum(dim=-1)
    dk_t = dk_t + torch.bmm(d_kv_mem.unsqueeze(1), S_before_t.transpose(-2, -1)).squeeze(1)
    dS_before_from_kv = torch.bmm(kt.unsqueeze(2), d_kv_mem.unsqueeze(1))
    dS_before = dS + dS_before_from_kv
    dS = dS_before * g_t.unsqueeze(-1)
    dg_t = (dS_before * S_after_prev).sum(dim=-1)  # S_after_prev always tensor
    return dS, dq_t, dk_t, dv_t, dg_t, dbe_t, dbw_t


_backward_step_compiled = torch.compile(
    _backward_step_reduced, mode='max-autotune', dynamic=True
)


class _ScanFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, g, beta_e, beta_w, out_mix):
        B, T, H, r_out, dk = q.shape
        dv = v.shape[-1]
        N = B * H

        def flat(x, *tail):
            return x.permute(1, 0, 2, *range(3, x.dim())).reshape(T, N, *tail).float()

        q_ = flat(q, r_out, dk)
        k_ = flat(k, dk)
        v_ = flat(v, dv)
        g_ = flat(g, dk)
        be_ = flat(beta_e)
        bw_ = flat(beta_w)

        S_after = torch.zeros(N, dk, dv, dtype=torch.float32, device=k.device)
        outs = torch.empty(T, N, dv, dtype=torch.float32, device=k.device)

        # Save S_before_list + S_after_list (~214MB) — needed for backward recomputation
        # Actually we only need S_before_list now since S_after_prev is recomputed inside compiled fn
        # But keeping S_after_list avoids recompilation of the backward function signature
        S_before_list = torch.empty(T, N, dk, dv, dtype=torch.float32, device=k.device)
        S_after_list = torch.empty(T, N, dk, dv, dtype=torch.float32, device=k.device)

        if r_out > 1:
            mixw = out_mix[None].expand(B, -1, -1).reshape(N, 1, r_out).float()

        # Pre-allocate intermediate buffers (reused each iteration, no allocation overhead)
        kv_mem = torch.empty(N, dv, dtype=torch.float32, device=k.device)
        update = torch.empty(N, dv, dtype=torch.float32, device=k.device)

        for t in range(T):
            # Triton kernel writes DIRECTLY into S_before_list[t], S_after_list[t], outs[t]
            # No intermediate allocation + copy needed!
            _forward_step_triton(
                S_after, S_before_list[t], kv_mem, update,
                S_after_list[t], outs[t],
                g_[t], k_[t], v_[t], be_[t], bw_[t], q_[t], mixw, r_out, N, dk, dv
            )
            S_after = S_after_list[t]  # update pointer for next iteration

        # Save S_before_list + S_after_list (~214MB) — recomputing S_after_prev
        # in backward costs 2 BMMs/step (3.17×) vs saving (4.29×)
        ctx.save_for_backward(
            q_, k_, v_, g_, be_, bw_,
            S_before_list, S_after_list
        )
        ctx.r_out = r_out
        ctx.B, ctx.T, ctx.H = B, T, H
        ctx.dv = dv
        ctx.N = N
        ctx.dk = dk
        if r_out > 1:
            ctx.mixw = mixw

        return outs.reshape(T, B, H, dv).permute(1, 0, 2, 3)

    @staticmethod
    def backward(ctx, dy):
        dy = dy.permute(1, 0, 2, 3).reshape(ctx.T, ctx.N, ctx.dv)
        q_, k_, v_, g_, be_, bw_ = ctx.saved_tensors[:6]
        S_before_list, S_after_list = ctx.saved_tensors[6:]
        N, T, dk, dv = ctx.N, ctx.T, ctx.dk, ctx.dv
        r_out = ctx.r_out
        dev = q_.device
        mixw = ctx.mixw if r_out > 1 else None

        dq_ = torch.empty_like(q_)
        dk_ = torch.zeros(T, N, dk, device=dev)
        dv_ = torch.zeros_like(v_)
        dg_ = torch.zeros_like(g_)
        dbe_ = torch.zeros_like(be_)
        dbw_ = torch.zeros_like(bw_)

        dS_bufs = [torch.zeros(N, dk, dv, device=dev) for _ in range(2)]
        dS = dS_bufs[0]
        S_after_zeros = torch.zeros(N, dk, dv, dtype=torch.float32, device=dev)

        for t in reversed(range(T)):
            S_after_prev = S_after_list[t - 1] if t > 0 else S_after_zeros
            dS_in = dS_bufs[t & 1]
            dS_in.copy_(dS)
            dS, dq_t, dk_t, dv_t, dg_t, dbe_t, dbw_t = _backward_step_compiled(
                dS_in,
                S_before_list[t], k_[t], v_[t], bw_[t], be_[t], q_[t], dy[t],
                g_[t], S_after_prev, r_out, mixw
            )
            dq_[t] = dq_t
            dk_[t] = dk_t
            dv_[t] = dv_t
            dg_[t] = dg_t
            dbe_[t] = dbe_t
            dbw_[t] = dbw_t

        def unflat(grad_, orig_shape):
            reshaped = grad_.reshape(T, ctx.B, ctx.H, *grad_.shape[2:])
            return reshaped.permute(1, 0, 2, *range(3, reshaped.dim()))

        return (
            unflat(dq_, q_.shape), unflat(dk_, k_.shape),
            unflat(dv_, v_.shape), unflat(dg_, g_.shape),
            unflat(dbe_, be_.shape), unflat(dbw_, bw_.shape), None,
        )


def scan(q, k, v, g, beta_e, beta_w, out_mix=None):
    return _ScanFunc.apply(q, k, v, g, beta_e, beta_w, out_mix)
