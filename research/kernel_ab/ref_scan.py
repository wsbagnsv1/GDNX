"""FROZEN reference for the KMD-2 native recurrence scan. DO NOT EDIT.

This is the exact math of gdn3/kmd2_native.py::KMD2NativeAttn._scan, extracted as
a free function. The A/B kernel search optimizes `scan(...)` in each workspace's
cand_scan.py; the bench checks it against THIS reference (forward output + input
gradients) and only then times it. Any candidate that does not reproduce these
outputs is disqualified — a fast-but-wrong kernel is worthless because it must drop
into the trained checkpoint unchanged.

Interface (all float32 on cuda):
  q      [B, T, H, r_out, dk]   query slots (already scaled + rotated)
  k      [B, T, H, dk]          keys (L2-normed, rotated)
  v      [B, T, H, dv]          values
  g      [B, T, H, dk]          per-channel multiplicative decay in (0, 1]
  beta_e [B, T, H]              erase gate
  beta_w [B, T, H]              write gate
  out_mix[H, r_out]             per-head slot mixing weights (r_out > 1)
returns
  y      [B, T, H, dv]
"""
import torch


def scan(q, k, v, g, beta_e, beta_w, out_mix=None):
    B, T, H, r_out, dk = q.shape
    dv = v.shape[-1]
    N = B * H

    def flat(x, *tail):
        return x.permute(1, 0, 2, *range(3, x.dim())).reshape(T, N, *tail).float()

    q_ = flat(q, r_out, dk); k_ = flat(k, dk); v_ = flat(v, dv)
    g_ = flat(g, dk); be_ = flat(beta_e); bw_ = flat(beta_w)
    if r_out > 1:
        mixw = out_mix[None].expand(B, -1, -1).reshape(N, 1, r_out).float()

    S = torch.zeros(N, dk, dv, dtype=torch.float32, device=k.device)
    outs = []
    for t in range(T):
        S = S * g_[t].unsqueeze(-1)                          # decay (key rows)
        kt = k_[t]                                            # [N, dk]
        kv_mem = torch.bmm(kt.unsqueeze(1), S).squeeze(1)     # S^T k -> [N, dv]
        S = S - torch.bmm(kt.unsqueeze(2), (be_[t].unsqueeze(-1) * kv_mem).unsqueeze(1))
        S = S + torch.bmm(kt.unsqueeze(2), (bw_[t].unsqueeze(-1) * v_[t]).unsqueeze(1))
        yt = torch.bmm(q_[t], S)                              # [N, r_out, dv]
        yt = (yt * mixw.transpose(1, 2)).sum(1) if r_out > 1 else yt.squeeze(1)
        outs.append(yt)
    return torch.stack(outs, 0).reshape(T, B, H, dv).permute(1, 0, 2, 3)
