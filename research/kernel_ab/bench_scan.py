"""Fitness harness for the KMD-2 scan kernel A/B search.

Loads a candidate `scan(...)` (from --cand), checks it against the FROZEN
ref_scan.py on forward output AND input gradients, then — only if correct — times
forward and forward+backward at a train-shaped and an eval-shaped config. Appends
one JSON result line to --leaderboard.

Primary metric: `train_fb_toks` (forward+backward tokens/s at B=2,T=512) — this is
what gates heal throughput. Secondary: `eval_fwd_toks` (B=1,T=2048).

Usage:
  python bench_scan.py --cand glm/cand_scan.py --leaderboard glm/leaderboard.jsonl \
      --note "chunked scan, C=64"

A candidate is DISQUALIFIED (correct=false, not timed) if fwd relMSE >= 2e-3 or
grad relMSE >= 1e-2 on any config. Report shows the actual errors so you can debug.
"""
import argparse, importlib.util, json, os, sys, time
from datetime import datetime, timezone

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
FWD_TOL = 2e-3      # relMSE on forward output
GRAD_TOL = 1e-2     # relMSE on input grads (bwd accumulates more error)
CONFIGS = {  # name: (B, T, H, r_out, dk, dv)
    "train": (2, 512, 16, 4, 128, 128),
    "eval":  (1, 2048, 16, 4, 128, 128),
}


def _load(path, modname):
    spec = importlib.util.spec_from_file_location(modname, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _mk_inputs(cfg, device, seed=0):
    B, T, H, r_out, dk, dv = cfg
    g = torch.Generator(device=device).manual_seed(seed)
    r = lambda *s: torch.randn(*s, device=device, dtype=torch.float32, generator=g)
    q = r(B, T, H, r_out, dk) * (dk ** -0.5)
    k = torch.nn.functional.normalize(r(B, T, H, dk), dim=-1)
    v = r(B, T, H, dv)
    # decay MUST match the real trained model: g spans the full (0,1] with mean ~0.78
    # and reaches ~0 (measured on the native heal ckpt). A benign near-1 decay HIDES
    # the failure of any within-chunk decay-RATIO reformulation (kDn=k/gcumF), whose
    # cumulative product underflows to 0 here. g = exp(-softplus(.)) mirrors the
    # native mechanism g_head=-exp(A_log)*softplus(a+dt_bias).
    decay = torch.exp(-torch.nn.functional.softplus(r(B, T, H, dk) * 1.2 - 0.6))
    beta_e = torch.sigmoid(r(B, T, H))
    beta_w = torch.sigmoid(r(B, T, H))
    out_mix = r(H, r_out)
    return q, k, v, decay, beta_e, beta_w, out_mix


def _relmse(a, b):
    return ((a - b).pow(2).mean() / b.pow(2).mean().clamp_min(1e-12)).item()


def _grads(scan, ins, upstream):
    q, k, v, decay, be, bw, om = (t.clone().requires_grad_(t.dtype.is_floating_point)
                                  for t in ins)
    y = scan(q, k, v, decay, be, bw, om)
    (y.float() * upstream).sum().backward()
    return y.detach(), [t.grad.detach() if t.grad is not None else None
                        for t in (q, k, v, decay, be, bw)]


def check_and_time(cand_scan, device, n_time=8, n_warm=2):
    ref = _load(os.path.join(HERE, "ref_scan.py"), "ref_scan").scan
    out = {"correct": True, "fwd_relmse": {}, "grad_relmse": {}}
    for name, cfg in CONFIGS.items():
        ins = _mk_inputs(cfg, device, seed=hash(name) & 0xffff)
        ups = torch.randn(cfg[0], cfg[1], cfg[2], cfg[5], device=device)
        y_ref, g_ref = _grads(ref, ins, ups)
        y_c, g_c = _grads(cand_scan, ins, ups)
        fwd_err = _relmse(y_c.float(), y_ref)
        gerr = max(_relmse(a.float(), b) for a, b in zip(g_c, g_ref)
                   if a is not None and b is not None)
        out["fwd_relmse"][name] = fwd_err
        out["grad_relmse"][name] = gerr
        if fwd_err >= FWD_TOL or gerr >= GRAD_TOL:
            out["correct"] = False
    if not out["correct"]:
        return out

    def timeit(scan, cfg, backward):
        ins = _mk_inputs(cfg, device, seed=7)
        ups = torch.randn(cfg[0], cfg[1], cfg[2], cfg[5], device=device)
        def run():
            if backward:
                _grads(scan, ins, ups)
            else:
                with torch.no_grad():
                    scan(*ins)
        for _ in range(n_warm): run()
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(n_time): run()
        torch.cuda.synchronize()
        return (time.time() - t0) / n_time

    tB, tT = CONFIGS["train"][0], CONFIGS["train"][1]
    eB, eT = CONFIGS["eval"][0], CONFIGS["eval"][1]
    cand_fb = timeit(cand_scan, CONFIGS["train"], True)
    ref_fb = timeit(ref, CONFIGS["train"], True)
    cand_fe = timeit(cand_scan, CONFIGS["eval"], False)
    ref_fe = timeit(ref, CONFIGS["eval"], False)
    out.update({
        "train_fb_ms": cand_fb * 1e3, "train_fb_toks": tB * tT / cand_fb,
        "eval_fwd_ms": cand_fe * 1e3, "eval_fwd_toks": eB * eT / cand_fe,
        "speedup_fb": ref_fb / cand_fb, "speedup_fwd": ref_fe / cand_fe,
    })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", required=True, help="path to candidate scan module")
    ap.add_argument("--leaderboard", required=True)
    ap.add_argument("--note", default="")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True

    cand_path = args.cand if os.path.isabs(args.cand) else os.path.join(HERE, args.cand)
    lb_path = args.leaderboard if os.path.isabs(args.leaderboard) else os.path.join(HERE, args.leaderboard)
    cand = _load(cand_path, "cand_scan").scan
    res = check_and_time(cand, args.device)
    res["note"] = args.note
    res["ts"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    res["cand"] = os.path.relpath(cand_path, HERE)

    print(json.dumps(res, indent=2))
    with open(lb_path, "a") as f:
        f.write(json.dumps(res) + "\n")
    if res["correct"]:
        print(f"\n=> CORRECT | train fwd+bwd {res['train_fb_toks']:.0f} tok/s "
              f"({res['speedup_fb']:.2f}x ref) | eval fwd {res['eval_fwd_toks']:.0f} tok/s "
              f"({res['speedup_fwd']:.2f}x ref)")
    else:
        print(f"\n=> DISQUALIFIED (wrong output). fwd_relmse={res['fwd_relmse']} "
              f"grad_relmse={res['grad_relmse']} (tol fwd<{FWD_TOL}, grad<{GRAD_TOL})")


if __name__ == "__main__":
    main()
