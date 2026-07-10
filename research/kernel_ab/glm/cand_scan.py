"""KMD-2 scan candidate (glm workspace) — chunk-parallel gated delta rule.

Reference recurrence (per token t):
    S = g_t * S                              # per-key-channel decay
    m = k_t^T S                              # read (decayed)
    S = S - k_t (be_t * m)^T                 # erase
    S = S + k_t (bw_t * v_t)^T               # write
    y = Q_t S (then mix over r_out)          # query (uses updated S)

Equivalent chunk-parallel form (block of C tokens, carry state S0 = S_{s-1}).
Let D_t = diag(g_t), G_{a:b}=D_{s+b-1}..D_{s+a}, gcumF[i]=prod_{l=0}^{i} g_{s+l}.
Unrolling gives S_{s+i} = G_{0:i+1} S0 + sum_{j<=i} G_{j+1:i+1} k_j u_j^T with
u_j = bw_j v_j - be_j m_j, m_j = k_j^T G_{0:j+1} S0 + sum_{l<j} (k_j^T G_{l+1:j+1} k_l) u_l.
The u's solve a unit-lower-triangular system  (I + DiagBe*Kmat) U = bw*v - be*m_inter,
where Kmat[j,l]=k_j^T G_{l+1:j+1} k_l (strictly lower). Up/down key split
kUp=k*gcumF, kDn=k/gcumF makes the between-decay a ratio: Kmat = kUp @ kDn^T.
Outputs y_i = qUp_i @ S0 + (masked A=qUp@kDn^T) @ U. Carry S_new=gcumF_last*(S0+kDn^T@U).
"""
import torch
import torch._inductor.config as _ic
import triton
import triton.language as tl
_ic.coordinate_descent_tuning = True
_ic.max_autotune = True
_ic.max_autotune_pointwise = True

_BF16 = torch.bfloat16


def _bmm(a, b):
    # bf16 matmul; cuBLAS accumulates bf16 in fp32 internally, result back to fp32.
    # Roughly doubles gemm throughput vs fp32/TF32 on the compute-bound fwd+bwd.
    return torch.bmm(a.to(_BF16), b.to(_BF16)).float()


@triton.jit
def _trsm_kernel(L_ptr, rhs_ptr, U_ptr,
                 stride_ln, stride_lc, stride_lcol,
                 stride_rn, stride_rc, stride_rdv,
                 stride_un, stride_uc, stride_udv,
                 BLOCK_DV: tl.constexpr, Cc: tl.constexpr):
    # Batched unit-lower triangular solve U = L^-1 rhs, rank-1-update forward
    # substitution (no dot products / tile indexing): U=rhs; for i: U[j>i]-=L[j,i]*U[i].
    # ~24% faster than cuSOLVER batched trsm for N=32,C=128; trsm is ~23% of fb.
    n = tl.program_id(0)
    b = tl.program_id(1) * BLOCK_DV
    offs_c = tl.arange(0, Cc)
    offs_d = tl.arange(0, BLOCK_DV)
    r_ptrs = rhs_ptr + n * stride_rn + offs_c[:, None] * stride_rc + (b + offs_d[None, :]) * stride_rdv
    U = tl.load(r_ptrs)
    for i in tl.range(0, Cc):
        lcol = tl.load(L_ptr + n * stride_ln + offs_c * stride_lc + i * stride_lcol)
        lcol = tl.where(offs_c > i, lcol, 0.0)
        ui = tl.sum(tl.where(offs_c[:, None] == i, U, 0.0), axis=0)
        U = U - lcol[:, None] * ui[None, :]
    st_ptrs = U_ptr + n * stride_un + offs_c[:, None] * stride_uc + (b + offs_d[None, :]) * stride_udv
    tl.store(st_ptrs, U)


def _trsm_triton(L, rhs):
    N_, C_, DV_ = L.shape[0], L.shape[1], rhs.shape[2]
    U = torch.empty_like(rhs)
    BLOCK_DV = 64
    grid = (N_, DV_ // BLOCK_DV)
    _trsm_kernel[grid](L, rhs, U,
                       L.stride(0), L.stride(1), L.stride(2),
                       rhs.stride(0), rhs.stride(1), rhs.stride(2),
                       U.stride(0), U.stride(1), U.stride(2),
                       BLOCK_DV=BLOCK_DV, Cc=C_, num_warps=4)
    return U


@triton.jit
def _trsm_upper_kernel(L_ptr, g_ptr, dU_ptr,
                       stride_ln, stride_lc, stride_lcol,
                       stride_gn, stride_gc, stride_gdv,
                       stride_dn, stride_dc, stride_ddv,
                       BLOCK_DV: tl.constexpr, Cc: tl.constexpr):
    # Batched unit-upper solve dU=(L^T)^-1 g_U via backward-sub rank-1 update:
    # dU=g_U; for i=C-1..0: di=dU[i]; dU[j<i]-=L[i,j]*di  (M[j,i]=L[i,j], strictly-lower of L row i).
    n = tl.program_id(0)
    b = tl.program_id(1) * BLOCK_DV
    offs_c = tl.arange(0, Cc)
    offs_d = tl.arange(0, BLOCK_DV)
    g_ptrs = g_ptr + n * stride_gn + offs_c[:, None] * stride_gc + (b + offs_d[None, :]) * stride_gdv
    dU = tl.load(g_ptrs)
    for k in tl.range(0, Cc):
        i = Cc - 1 - k
        lrow = tl.load(L_ptr + n * stride_ln + i * stride_lc + offs_c * stride_lcol)
        lrow = tl.where(offs_c < i, lrow, 0.0)
        di = tl.sum(tl.where(offs_c[:, None] == i, dU, 0.0), axis=0)
        dU = dU - lrow[:, None] * di[None, :]
    st_ptrs = dU_ptr + n * stride_dn + offs_c[:, None] * stride_dc + (b + offs_d[None, :]) * stride_ddv
    tl.store(st_ptrs, dU)


def _trsm_upper_triton(L, g_U):
    dU = torch.empty_like(g_U)
    BLOCK_DV = 64
    grid = (L.shape[0], g_U.shape[2] // BLOCK_DV)
    _trsm_upper_kernel[grid](L, g_U, dU,
                             L.stride(0), L.stride(1), L.stride(2),
                             g_U.stride(0), g_U.stride(1), g_U.stride(2),
                             dU.stride(0), dU.stride(1), dU.stride(2),
                             BLOCK_DV=BLOCK_DV, Cc=L.shape[1], num_warps=4)
    return dU


class _TrsmFn(torch.autograd.Function):
    # Opaque cuSOLVER trsm -> custom Triton fwd + hand bwd. No inductor fusion is
    # lost (trsm was already an opaque linalg call), so the autograd.Function
    # boundary is free here (unlike the prior gemm-fusion dead ends).
    @staticmethod
    def forward(ctx, L, rhs, tril_strict):
        U = _trsm_triton(L, rhs)
        ctx.save_for_backward(L, U, tril_strict)
        return U

    @staticmethod
    def backward(ctx, g_U):
        L, U, tril_strict = ctx.saved_tensors
        # dU = (L^T)^-1 g_U  (L unit-lower -> L^T unit-upper); Triton backward-sub trsm.
        dU = _trsm_upper_triton(L, g_U)
        g_L = -(dU @ U.transpose(-1, -2)) * tril_strict      # strictly-lower free part
        g_rhs = dU
        return g_L, g_rhs, None


def _scan_impl(q, k, v, g, beta_e, beta_w, out_mix=None):
    B, T, H, r_out, dk = q.shape
    dv = v.shape[-1]
    N = B * H
    C = 128
    dev = q.device

    def flat(x, *tail):
        return x.permute(1, 0, 2, *range(3, x.dim())).reshape(T, N, *tail).float()

    q_ = flat(q, r_out, dk)            # [T, N, r_out, dk]
    k_ = flat(k, dk)                   # [T, N, dk]
    v_ = flat(v, dv)                   # [T, N, dv]
    g_ = flat(g, dk)                   # [T, N, dk]
    be_ = flat(beta_e)[:, :, None]     # [T, N, 1]
    bw_ = flat(beta_w)[:, :, None]     # [T, N, 1]
    if r_out > 1:
        mixw = out_mix[None].expand(B, -1, -1).reshape(N, 1, r_out).float()  # [N,1,r_out]

    # Pad T to a multiple of C: zero q/k/v + unit decay => no-op steps (S unchanged, y=0).
    if T % C != 0:
        P = C - (T % C)
        z = torch.zeros
        q_ = torch.cat([q_, z(P, N, r_out, dk, device=dev, dtype=torch.float32)], 0)
        k_ = torch.cat([k_, z(P, N, dk, device=dev, dtype=torch.float32)], 0)
        v_ = torch.cat([v_, z(P, N, dv, device=dev, dtype=torch.float32)], 0)
        g_ = torch.cat([g_, torch.ones(P, N, dk, device=dev, dtype=torch.float32)], 0)
        be_ = torch.cat([be_, z(P, N, 1, device=dev, dtype=torch.float32)], 0)
        bw_ = torch.cat([bw_, z(P, N, 1, device=dev, dtype=torch.float32)], 0)
    Tc = q_.shape[0]
    nC = Tc // C

    # Group into chunks: [N, nC, C, ...] (consecutive T indices per chunk).
    q_c = q_.reshape(nC, C, N, r_out, dk).permute(2, 0, 1, 3, 4)
    k_c = k_.reshape(nC, C, N, dk).permute(2, 0, 1, 3)
    v_c = v_.reshape(nC, C, N, dv).permute(2, 0, 1, 3)
    g_c = g_.reshape(nC, C, N, dk).permute(2, 0, 1, 3)
    be_c = be_.reshape(nC, C, N, 1).permute(2, 0, 1, 3)
    bw_c = bw_.reshape(nC, C, N, 1).permute(2, 0, 1, 3)

    gcumF = torch.cumprod(g_c, dim=2)                                  # [N, nC, C, dk]
    eps = 1e-12
    kUp = k_c * gcumF                                                  # decayed key (<=|k|)
    kDn = k_c / gcumF.clamp_min(eps)                                   # inverse-decayed key

    S = torch.zeros(N, dk, dv, dtype=torch.float32, device=dev)        # carry [N, dk, dv]
    tril_strict = torch.tril(torch.ones(C, C, device=dev), diagonal=-1)   # col < row (l < j)
    tril_incl = torch.tril(torch.ones(C, C, device=dev), diagonal=0)      # col <= row (j <= i)
    eye = torch.eye(C, device=dev)

    out_chunks = []
    for c in range(nC):
        vc = v_c[:, c]            # [N, C, dv]
        qcc = q_c[:, c]           # [N, C, r_out, dk]
        gcumF_c = gcumF[:, c]     # [N, C, dk]
        kUp_c = kUp[:, c]         # [N, C, dk]
        kDn_c = kDn[:, c]         # [N, C, dk]
        bec = be_c[:, c, :, 0]    # [N, C]
        bwc = bw_c[:, c, :, 0]    # [N, C]

        # Inter-chunk (carry) contributions: m_inter = kUp @ S ; term1 = qUp @ S
        qUp_c = qcc * gcumF_c.unsqueeze(2)                             # [N, C, r_out, dk]
        qUp_flat = qUp_c.reshape(N, C * r_out, dk)                      # shared left factor for term1 & A
        m_inter = _bmm(kUp_c, S)                                 # [N, C, dv]
        term1 = _bmm(qUp_flat, S).reshape(N, C, r_out, dv)

        # Within-chunk key Gram (strictly lower) -> unit-lower L = I + DiagBe*Kmat
        Kmat = _bmm(kUp_c, kDn_c.transpose(1, 2)) * tril_strict   # [N, C, C], l<j
        L = eye + bec.unsqueeze(-1) * Kmat                              # [N, C, C] unit-lower
        rhs = bwc.unsqueeze(-1) * vc - bec.unsqueeze(-1) * m_inter     # [N, C, dv]
        U = _TrsmFn.apply(L, rhs, tril_strict)

        # Intra-chunk output: term2[i,r] = sum_{j<=i} (qUp_i . kDn_j) u_j
        # Reuse qUp_flat (shared left factor with term1) so inductor can fuse the two qUp@X gemms.
        A = (_bmm(qUp_flat, kDn_c.transpose(1, 2)).reshape(N, C, r_out, C)
             * tril_incl.view(1, C, 1, C))                              # [N, C, r_out, C]
        term2 = _bmm(A.reshape(N, C * r_out, C), U).reshape(N, C, r_out, dv)

        y = term1 + term2                                              # [N, C, r_out, dv]
        if r_out > 1:
            y = (y * mixw[:, :, :, None]).sum(dim=2)                 # [N, C, dv]
        else:
            y = y.squeeze(2)
        out_chunks.append(y)

        # Carry: S_new = gcumF_last * (S0 + kDn^T @ U)
        W = _bmm(kDn_c.transpose(1, 2), U)                          # [N, dk, dv] bmm (cuBLAS bf16, fp32 accum)
        S = gcumF_c[:, -1, :].unsqueeze(-1) * (S + W)                  # [N, dk, dv]

    y_out = torch.cat(out_chunks, dim=1)[:, :T, :]                     # [N, T, dv]
    return y_out.transpose(0, 1).reshape(T, B, H, dv).permute(1, 0, 2, 3)


# Chunk-parallel body has a short outer loop (nC = T/C = 8 train / 32 eval), so
# dynamo unrolls a tractable graph and reduce-overhead CUDA-graph capture removes
# the remaining per-chunk launch overhead. Compile cost is absorbed by the bench's
# warmup calls; cached calls are ~2x faster than the eager chunked version.
scan = torch.compile(_scan_impl, mode="max-autotune-no-cudagraphs")
