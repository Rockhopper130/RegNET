"""Phase 1: label-free evaluation of a trained SVF registration model.

Scores a checkpoint with signals it never saw during training. Nothing here is
optimised by the model's loss except where the independence table says otherwise.

Read the tautology warning before quoting anything:

  * exp(v) o exp(-v) inverse consistency and template-ROUTED triplet closure are
    ALGEBRAIC IDENTITIES for a stationary velocity field routed through one
    template. They come out ~0 regardless of how bad the network is. They are
    computed here as a pipeline audit, with a synthetic-velocity control that
    isolates how much of the residual is pure numerics.
  * The informative siblings are PAIRWISE SYMMETRY and DIRECT TRIPLET CLOSURE,
    which use independent forward passes and which nothing in the architecture or
    the loss forces to zero. The model trains on random subject pairs
    (keyreg.py:384), so these are in-distribution and fair.

Usage:
    python eval_labelfree.py --val <...>/full_val.txt --fixed_name seg4_onehot.npy \
        --run svf_E_full --csv results/labelfree_gt.csv
"""
import argparse
import os
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import svfcommon as S
from keyreg import (load_seg, read_list, identity_grid_vol, svf_integrate,
                    compose, dice_per_class, folding_pct)

ap = argparse.ArgumentParser(description="Label-free evaluation of an SVF model")
ap.add_argument("--val", default=f"{S.NEURITE}/full_val.txt")
ap.add_argument("--fixed_name", default="seg4_onehot.npy",
                help="basename swapped into each val path (seg4_onehot_clinical.npy "
                     "for the recon-all-clinical arm)")
ap.add_argument("--run", default="svf_E_full")
ap.add_argument("--ckpt", default="best.pth")
ap.add_argument("--model_kind", default="seg", choices=["seg", "intensity"],
                help="what the NETWORK consumes. Metrics are identical either way: "
                     "Dice always warps the segmentation, NCC always warps the "
                     "intensity. Only the network input differs.")
ap.add_argument("--template_subject", default=S.TEMPLATE_SUBJ)
ap.add_argument("--paired_only", action="store_true",
                help="restrict to subjects that also have the clinical volume, so the "
                     "GT and clinical arms cover the same cohort")
ap.add_argument("--int_sweep", default="", help="e.g. 4,6,8,10,12,14,16")
ap.add_argument("--n_pairs", type=int, default=83)
ap.add_argument("--n_triplets", type=int, default=200)
ap.add_argument("--erode", type=int, default=6)
ap.add_argument("--ncc_win", type=int, default=9)
ap.add_argument("--limit", type=int, default=0, help="probe mode: first N subjects")
ap.add_argument("--csv", default="")
ap.add_argument("--device", default="cuda:0")
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

S.require_shared_tmpdir()
dev = args.device
torch.manual_seed(args.seed)
random.seed(args.seed)

print("=" * 78)
print(f"LABEL-FREE EVALUATION  run={args.run}  fixed={args.fixed_name}")
print("=" * 78, flush=True)

scale_err = S.assert_scale_mm(args.template_subject)
print(f"[guard] SCALE_MM agrees with the affine to {scale_err:.2e} mm", flush=True)

# --------------------------------------------------------------------------- #
# Cohort
# --------------------------------------------------------------------------- #
paths = read_list(args.val)
subs = S.check_lists(paths)
print(f"[guard] split clean: {len(subs)} subjects, no train overlap, template excluded")

if args.paired_only:
    keep, dropped = S.paired_subjects(args.val)
    subs = [s for s in subs if s in set(keep)]
    print(f"[guard] paired-only: {len(subs)} subjects")
if args.limit:
    subs = subs[:args.limit]
    print(f"[probe] limited to {len(subs)} subjects")


def seg_path(subj):
    return os.path.join(S.NEURITE, subj, args.fixed_name)


missing = [s for s in subs if not os.path.exists(seg_path(s))]
assert not missing, f"missing {args.fixed_name} for {len(missing)}: {missing[:5]}"

# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
ck_path = os.path.join(S.RUNS, args.run, args.ckpt)
ck = torch.load(ck_path, map_location=dev, weights_only=False)
if args.model_kind == "seg":
    m = S.build_seg_model(ck, dev)
else:
    from ssl_intensity import IntensitySVF
    a = ck["args"]
    m = IntensitySVF(target=a["target"], delta=a["delta"],
                     int_steps=a["int_steps"]).to(dev)
    m.load_state_dict(ck["model"])
    m.eval()
    assert ck.get("select_on") == "val_ncc", (
        f"SSL checkpoint was selected on {ck.get('select_on')!r}; if that is Dice "
        "the model is secretly label-supervised and its Dice column is void")
    print(f"[guard] SSL checkpoint selected on {ck['select_on']} -- no label leak")
n_steps = ck["args"]["int_steps"]
print(f"[model] {ck_path}  epoch={ck.get('epoch')}  int_steps={n_steps}  "
      f"params={sum(p.numel() for p in m.parameters()):,}", flush=True)

idg = identity_grid_vol(ck["args"]["target"], dev)
tpl_seg = load_seg(os.path.join(S.NEURITE, args.template_subject, "seg4_onehot.npy"),
                   S.TS).unsqueeze(0).to(dev)
tpl_int = S.load_intensity(args.template_subject, device=dev)
tpl_mask = S.brain_mask(tpl_int, erode=args.erode)
print(f"[data] template brain mask covers {tpl_mask.float().mean().item()*100:.1f}% "
      "of the volume", flush=True)

rows = []
cache = {}          # subj -> (vel, grid_fwd, grid_inv) for the consistency section


@torch.no_grad()
def per_subject(subj):
    fix = load_seg(seg_path(subj), S.TS).unsqueeze(0).to(dev)
    I_s = S.load_intensity(subj, device=dev)

    # The ONLY per-model branch in the whole metric path: what the network is fed.
    # Everything downstream is byte-identical, which is what makes the head-to-head
    # a test of supervision rather than of plumbing.
    if args.model_kind == "seg":
        net_m, net_f = tpl_seg, fix
    else:
        net_m, net_f = tpl_int, I_s
    vel, g_f, g_i = S.fwd_inv_grids(m, net_m, net_f, idg)

    # Dice always warps the SEGMENTATION; NCC always warps the INTENSITY -- for both
    # models. So Dice is in-objective for SVF-E and held out for SSL, and NCC is the
    # reverse. The biases are symmetric and stated, not hidden.
    warped = S.warp(tpl_seg, g_f)
    pc = dice_per_class(warped, fix)

    mask_s = S.brain_mask(I_s, erode=args.erode)
    frac = mask_s.float().mean().item()
    assert 0.10 <= frac <= 0.50, f"{subj}: brain mask fraction {frac:.3f} out of range"

    # --- intensity arms. The model consumed only seg4_onehot; aligned_norm is
    # --- untouched by the training signal.
    I_model = S.warp(tpl_int, g_f)          # template intensity carried to subject
    I_rev = S.warp(tpl_int, g_i)            # direction control: must be worse
    ncc_id, _ = S.masked_local_ncc(tpl_int, I_s, mask_s, args.ncc_win)
    ncc_md, exc = S.masked_local_ncc(I_model, I_s, mask_s, args.ncc_win)
    ncc_rv, _ = S.masked_local_ncc(I_rev, I_s, mask_s, args.ncc_win)

    # --- inverse consistency (AUDIT ONLY -- tautological for an SVF)
    ic = S.mm_err(compose(g_f, g_i), idg, tpl_mask)

    # --- numerics control: synthetic velocity, same RMS, same code path. Its true
    # --- IC error is exactly zero, so this is 100% pipeline numerics.
    lin = torch.linspace(-1, 1, ck["args"]["target"], device=dev)
    zz, yy, xx = torch.meshgrid(lin, lin, lin, indexing="ij")
    v_ref = torch.stack([torch.sin(2 * np.pi * xx), torch.sin(2 * np.pi * yy),
                         torch.sin(2 * np.pi * zz)], 0)[None]
    v_ref = v_ref * (vel.pow(2).mean().sqrt() / v_ref.pow(2).mean().sqrt())
    gr_f = idg + svf_integrate(v_ref, n_steps, idg).permute(0, 2, 3, 4, 1)
    gr_i = idg + svf_integrate(-v_ref, n_steps, idg).permute(0, 2, 3, 4, 1)
    ic_ref = S.mm_err(compose(gr_f, gr_i), idg, tpl_mask)

    lj = S.logjac_stats(g_f)
    row = {
        "subject": subj,
        "wm_dice": pc[3], "fg_dice": float(np.mean(pc[1:])),
        "folding_pct": folding_pct(g_f),
        "std_log_det": lj["std_log_det"], "mean_det": lj["mean_det"],
        "mean_disp_mm": S.mean_disp_mm(g_f, idg, tpl_mask),
        "ic_mm": ic["mean"], "ic_p99_mm": ic["p99"],
        "ic_ref_mm": ic_ref["mean"],
        "ic_attributable_mm": ic["mean"] - ic_ref["mean"],
        "ncc_identity": ncc_id, "ncc_model": ncc_md, "ncc_reversed": ncc_rv,
        "ncc_gain": ncc_md - ncc_id,
        "pearson_identity": S.masked_pearson(tpl_int, I_s, mask_s),
        "pearson_model": S.masked_pearson(I_model, I_s, mask_s),
        "nmi_identity": S.masked_nmi(tpl_int, I_s, mask_s),
        "nmi_model": S.masked_nmi(I_model, I_s, mask_s),
        "mse_identity": S.masked_mse(tpl_int, I_s, mask_s),
        "mse_model": S.masked_mse(I_model, I_s, mask_s),
        "mask_frac": frac, "ncc_excluded_frac": exc,
    }
    cache[subj] = (g_f.cpu(), g_i.cpu())
    del fix, vel, g_f, g_i, warped, I_s, I_model, I_rev, v_ref, gr_f, gr_i
    torch.cuda.empty_cache()
    return row


print("\n--- per-subject ---", flush=True)
for i, s in enumerate(subs, 1):
    r = per_subject(s)
    rows.append(r)
    print(f"[{i}/{len(subs)}] {s}  dice {r['wm_dice']:.4f}  disp {r['mean_disp_mm']:.2f}mm"
          f"  ncc {r['ncc_identity']:.3f}->{r['ncc_model']:.3f}"
          f"  fold {r['folding_pct']:.4f}%", flush=True)

# --------------------------------------------------------------------------- #
# Direction control -- hard assert, this is how the 8x flip error got shipped
# --------------------------------------------------------------------------- #
mean_id = float(np.mean([r["ncc_identity"] for r in rows]))
mean_md = float(np.mean([r["ncc_model"] for r in rows]))
mean_rv = float(np.mean([r["ncc_reversed"] for r in rows]))
print(f"\n[control] NCC identity {mean_id:.4f} | model {mean_md:.4f} | "
      f"reversed {mean_rv:.4f}")
assert mean_rv < mean_id, (
    f"reversed-direction control ({mean_rv:.4f}) is not worse than identity "
    f"({mean_id:.4f}) -- the warp direction is probably flipped")
print("[control] reversed-direction control passes (pipeline direction is correct)")

# NOT an assert. A model genuinely CAN be worse than doing nothing -- that is a
# finding, not a bug, and on recon-all-clinical inputs it is exactly what happens.
# The reversed-direction assert above is what distinguishes the two cases: it uses
# the same code path, so if it passes, the pipeline is sound and a sub-identity
# score belongs to the model.
if mean_md <= mean_id:
    print(f"\n{'!' * 74}")
    print(f"FINDING: the model is WORSE than not registering at all.")
    print(f"  model NCC {mean_md:.4f}  vs  no-warp baseline {mean_id:.4f}")
    print(f"  The reversed-direction control passed, so this is the model, not a bug.")
    print(f"  A warp scoring below identity is actively misaligning anatomy --")
    print(f"  Dice alone would not show this.")
    print(f"{'!' * 74}\n", flush=True)
else:
    print(f"[control] model beats the no-warp baseline "
          f"({mean_md:.4f} > {mean_id:.4f})", flush=True)

# --------------------------------------------------------------------------- #
# Mismatched-subject floor: every brain looks like a brain, so an NCC of 0.8 is
# meaningless without knowing what a wrong pairing scores.
# --------------------------------------------------------------------------- #
rng = random.Random(args.seed)
mis = []
for s in subs[:min(len(subs), 40)]:
    other = rng.choice([o for o in subs if o != s])
    g_f, _ = cache[s]
    I_other = S.load_intensity(other, device=dev)
    mo = S.brain_mask(I_other, erode=args.erode)
    with torch.no_grad():
        v, _ = S.masked_local_ncc(S.warp(tpl_int, g_f.to(dev)), I_other, mo, args.ncc_win)
    mis.append(v)
print(f"[floor] mismatched-subject NCC {np.mean(mis):.4f} "
      f"(chance level; model {mean_md:.4f}, identity {mean_id:.4f})", flush=True)

# --------------------------------------------------------------------------- #
# Consistency: tautological vs informative, side by side
# --------------------------------------------------------------------------- #
print("\n--- consistency ---", flush=True)


@torch.no_grad()
def pair_grid(a, b):
    """Directly predicted map between two subjects. In-distribution: the model is
    trained on random subject pairs (keyreg.py:384), so this is a fair test."""
    if args.model_kind == "seg":
        A = load_seg(seg_path(a), S.TS).unsqueeze(0).to(dev)
        B = load_seg(seg_path(b), S.TS).unsqueeze(0).to(dev)
    else:
        A = S.load_intensity(a, device=dev)
        B = S.load_intensity(b, device=dev)
    _, g, _ = S.fwd_inv_grids(m, A, B, idg)
    del A, B
    return g


sym, direct, routed_vs_direct, routed_trip = [], [], [], []
n_pairs = min(args.n_pairs, len(subs))
for i in range(n_pairs):
    a, b = subs[i], subs[(i + 1) % len(subs)]
    if a == b:
        continue
    g_ab, g_ba = pair_grid(a, b), pair_grid(b, a)
    sym.append(S.mm_err(compose(g_ab, g_ba), idg, tpl_mask)["mean"])
    # routed: B -> template -> A, built from the two template maps
    g_fa, g_ia = cache[a]
    g_fb, g_ib = cache[b]
    routed = compose(g_ia.to(dev), g_fb.to(dev))
    routed_vs_direct.append(S.mm_err(routed, g_ab, tpl_mask)["mean"])
    del g_ab, g_ba, routed
    torch.cuda.empty_cache()

if len(subs) >= 3:
    for _ in range(args.n_triplets):
        a, b, c = rng.sample(subs, 3)
        g = compose(compose(pair_grid(a, b), pair_grid(b, c)), pair_grid(c, a))
        direct.append(S.mm_err(g, idg, tpl_mask)["mean"])
        # routed counterpart -- telescopes algebraically, expect ~0
        ga_f, ga_i = cache[a]
        gb_f, gb_i = cache[b]
        gc_f, gc_i = cache[c]
        gr = compose(compose(compose(ga_i.to(dev), gb_f.to(dev)),
                             compose(gb_i.to(dev), gc_f.to(dev))),
                     compose(gc_i.to(dev), ga_f.to(dev)))
        routed_trip.append(S.mm_err(gr, idg, tpl_mask)["mean"])
        del g, gr
        torch.cuda.empty_cache()

# --------------------------------------------------------------------------- #
# int_steps sweep
# --------------------------------------------------------------------------- #
sweep = []
if args.int_sweep:
    print("\n--- int_steps sweep (IC error AND Dice; changing n changes the warp) ---")
    probe = subs[:min(10, len(subs))]
    for n in [int(x) for x in args.int_sweep.split(",")]:
        ics, dcs = [], []
        for s in probe:
            with torch.no_grad():
                fix = load_seg(seg_path(s), S.TS).unsqueeze(0).to(dev)
                _, gf, gi = S.fwd_inv_grids(m, tpl_seg, fix, idg, int_steps=n)
                ics.append(S.mm_err(compose(gf, gi), idg, tpl_mask)["mean"])
                dcs.append(dice_per_class(S.warp(tpl_seg, gf), fix)[3])
                del fix, gf, gi
        sweep.append((n, float(np.mean(ics)), float(np.mean(dcs))))
        print(f"  int_steps={n:<3d} IC {np.mean(ics):.5f} mm   WM Dice {np.mean(dcs):.4f}",
              flush=True)

# --------------------------------------------------------------------------- #
# Summary
# --------------------------------------------------------------------------- #
def ms(k):
    v = np.array([r[k] for r in rows], dtype=float)
    return f"{v.mean():.4f} +/- {v.std():.4f}"


print("\n" + "=" * 78)
print(f"SUMMARY  {args.run}  {args.fixed_name}  n={len(rows)}")
print("=" * 78)
print("\nSUPERVISED-OBJECTIVE METRICS (Dice IS this model's training loss)")
print(f"  WM Dice                {ms('wm_dice')}")
print(f"  FG Dice                {ms('fg_dice')}")
print("\nREGULARITY  (all of these are maximised by the IDENTITY warp --")
print("             read them next to mean_disp_mm, never alone)")
print(f"  folding %              {ms('folding_pct')}")
print(f"  std log det J          {ms('std_log_det')}")
print(f"  mean |disp| mm         {ms('mean_disp_mm')}   <- magnitude")
print("\nAUDIT ONLY -- TAUTOLOGICAL, exp(v) and exp(-v) are exact inverses by")
print("              construction. Not evidence about the model.")
print(f"  inverse consistency mm {ms('ic_mm')}")
print(f"  synthetic-vel control  {ms('ic_ref_mm')}")
print(f"  model-attributable     {ms('ic_attributable_mm')}")
if routed_trip:
    print(f"  routed triplet closure {np.mean(routed_trip):.5f} +/- {np.std(routed_trip):.5f} mm")
print("\nINFORMATIVE CONSISTENCY -- independent forward passes, nothing forces these")
print("                           to zero")
if sym:
    print(f"  pairwise symmetry mm   {np.mean(sym):.4f} +/- {np.std(sym):.4f}")
if direct:
    print(f"  direct triplet mm      {np.mean(direct):.4f} +/- {np.std(direct):.4f}")
if routed_vs_direct:
    print(f"  routed vs direct mm    {np.mean(routed_vs_direct):.4f} +/- "
          f"{np.std(routed_vs_direct):.4f}   <- cost of atlas-routing")
print("\nINTENSITY -- aligned_norm, never seen by this model (it consumes only")
print("             seg4_onehot). Gains are over the no-warp affine baseline.")
print(f"  NCC   identity         {ms('ncc_identity')}")
print(f"  NCC   model            {ms('ncc_model')}")
print(f"  NCC   gain             {ms('ncc_gain')}")
print(f"  NCC   mismatched floor {np.mean(mis):.4f}")
print(f"  Pearson identity/model {ms('pearson_identity')} / {ms('pearson_model')}")
print(f"  NMI     identity/model {ms('nmi_identity')} / {ms('nmi_model')}")
print(f"  MSE     identity/model {ms('mse_identity')} / {ms('mse_model')}")
print()

if args.csv:
    import csv
    os.makedirs(os.path.dirname(os.path.abspath(args.csv)) or ".", exist_ok=True)
    with open(args.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"per-subject CSV -> {args.csv}")
    agg = os.path.splitext(args.csv)[0] + "_consistency.csv"
    with open(agg, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "mean_mm", "std_mm", "n", "tautological"])
        w.writerow(["inverse_consistency", np.mean([r['ic_mm'] for r in rows]),
                    np.std([r['ic_mm'] for r in rows]), len(rows), "YES"])
        if routed_trip:
            w.writerow(["routed_triplet", np.mean(routed_trip), np.std(routed_trip),
                        len(routed_trip), "YES"])
        if sym:
            w.writerow(["pairwise_symmetry", np.mean(sym), np.std(sym), len(sym), "no"])
        if direct:
            w.writerow(["direct_triplet", np.mean(direct), np.std(direct), len(direct), "no"])
        if routed_vs_direct:
            w.writerow(["routed_vs_direct", np.mean(routed_vs_direct),
                        np.std(routed_vs_direct), len(routed_vs_direct), "no"])
        for n, ic, dc in sweep:
            w.writerow([f"int_steps_{n}_ic", ic, 0.0, len(subs[:10]), "YES"])
            w.writerow([f"int_steps_{n}_dice", dc, 0.0, len(subs[:10]), "-"])
    print(f"consistency CSV -> {agg}")
