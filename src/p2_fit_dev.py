"""
Fit calibrators on TRAIN, compare and select on VALIDATION. Test files are never opened.

    python src/p2_fit_dev.py --data-root E:/math_dataset

Key design (new, because syndrome weight is now available):
  The stopping rule stops at syndrome = 0 anyway, so q_now / q_fail only matter on rows with syndrome > 0.
  Models are compared on that decision-relevant region ("syn>0"), and we test whether fitting on that region
  ("c" models) beats fitting on all rows. Selection rule (declared before looking): best validation log loss,
  then the SIMPLEST model not significantly worse (paired frame-bootstrap, 95%).
"""
import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import calib_lib as cl
import p2_common as pc

REPO = pc.REPO
ap = argparse.ArgumentParser()
ap.add_argument("--data-root", default=None)
ap.add_argument("--out-dir", default=str(REPO / "results" / "person2_v3"))
ap.add_argument("--snrs", default="2.00,2.25,2.50,2.75")
ap.add_argument("--k-values", default="1,2,3,5,10")
ap.add_argument("--t-min", type=int, default=5)
ap.add_argument("--n-boot", type=int, default=100)
a = ap.parse_args()
SNRS = [float(s) for s in a.snrs.split(",")]
KS = [int(k) for k in a.k_values.split(",")]
T_MIN = a.t_min
OUT = Path(a.out_dir); OUT.mkdir(parents=True, exist_ok=True)
root = pc.find_root(a.data_root)
pd.set_option("display.width", 220, "display.max_columns", 30)
print(f"data root: {root}\nout: {OUT}")

tr_d = pc.load_group(root, "train", SNRS)
va_d = pc.load_group(root, "validation", SNRS)
tr, va = pc.table(tr_d, KS), pc.table(va_d, KS)
_T0 = time.perf_counter()
nf_va = len(va_d["frame_id"]); T = cl.T
print(f"train frames {len(tr_d['frame_id']):,} (failing {int(tr_d['final_failure'].sum())}) | "
      f"validation frames {nf_va:,} (failing {int(va_d['final_failure'].sum())})")
print("rows with syndrome>0 (validation): %.1f%%" % (100 * (va.syn > 0).mean()))

# =====================================================================================
# q_now
# =====================================================================================
print("\n" + "=" * 90 + "\n[A] q_now  (target: current frame wrong).  Primary evaluation region: syndrome > 0\n" + "=" * 90)
y_tr, y_va = tr["y_now"].to_numpy(), va["y_now"].to_numpy()
pos_tr = tr["syn"].to_numpy() > 0
def fitL(feats, mask):
    return cl.LogisticCal(feats).fit(tr[mask], y_tr[mask])
allm = np.ones(len(tr), bool)
now = {  # name -> (calibrator, fit_region)   ordered simplest -> most complex
    "RAW r": (cl.IdentityRaw(), "none"),
    "M1 spec logit(log r) [all]": (fitL(["log_r"], allm), "all"),
    "M1c logit(log r) [syn>0]": (fitL(["log_r"], pos_tr), "syn>0"),
    "M2 ext (log r,S) [all]": (fitL(["log_r", "S"], allm), "all"),
    "M2c ext (log r,S) [syn>0]": (fitL(["log_r", "S"], pos_tr), "syn>0"),
    "M5c +lsyn [syn>0]": (fitL(["log_r", "S", "lsyn"], pos_tr), "syn>0"),
    "M6c +lsyn,min_abs [syn>0]": (fitL(["log_r", "S", "lsyn", "min_abs"], pos_tr), "syn>0"),
    "M3 isotonic A2 [all]": (cl.IsotonicCal().fit(tr, y_tr), "all"),
    "M3c isotonic A2 [syn>0]": (cl.IsotonicCal().fit(tr[pos_tr], y_tr[pos_tr]), "syn>0"),
}
order_now = list(now)
P_now = {n: c.predict(va) for n, (c, _) in now.items()}
# per-SNR maps (plan: "also fit per-SNR-bin maps") -- reported, not selectable
p_snr = np.zeros(len(va))
for s in SNRS:
    mt, mv = (tr.snr == s).to_numpy() & pos_tr, (va.snr == s).to_numpy()
    p_snr[mv] = cl.LogisticCal(["log_r", "S"]).fit(tr[mt], y_tr[mt]).predict(va[mv])
P_report = dict(P_now); P_report["M4c per-SNR M2c [syn>0]"] = p_snr

mask_now = pc.eval_mask(va, "q_now", T_MIN)
rows = [{"model": n, **pc.score2(y_va[mask_now], p[mask_now])} for n, p in P_report.items()]
tab_now = pd.DataFrame(rows); tab_now.to_csv(OUT / "q_now_val_region_syn_pos.csv", index=False)
print(f"\nvalidation, region syn>0  (rows {mask_now.sum():,}, prevalence {y_va[mask_now].mean():.4f})")
print(tab_now.drop(columns=["n", "ece_fixed10"]).round(5).to_string(index=False))

rows = [{"model": n, **pc.score2(y_va, p)} for n, p in P_report.items()]
pd.DataFrame(rows).to_csv(OUT / "q_now_val_all_rows.csv", index=False)
print("\n(all rows, for comparison with the pilot: ECE_eqmass20 / Brier)")
print(pd.DataFrame(rows)[["model", "ece_eqmass20", "brier", "logloss"]].round(5).to_string(index=False))

rows = []
for s in SNRS:
    mk = mask_now & (va.snr == s).to_numpy()
    for n, p in P_report.items():
        sc = pc.score2(y_va[mk], p[mk]); rows.append({"snr": s, "model": n, "rows": int(mk.sum()),
                                                      "ece": sc["ece_eqmass20"], "brier": sc["brier"],
                                                      "slope": sc["cal_slope"], "intercept": sc["cal_intercept"]})
tab_snr = pd.DataFrame(rows); tab_snr.to_csv(OUT / "q_now_val_by_snr.csv", index=False)
print("\nECE by SNR (region syn>0)"); print(tab_snr.pivot(index="model", columns="snr", values="ece").round(5).to_string())

print(f"\nframe-level bootstrap ({a.n_boot}) ...")
Y = y_va.reshape(nf_va, T); M = mask_now.reshape(nf_va, T)
B = pc.boot_arrays({n: p.reshape(nf_va, T) for n, p in P_now.items()}, Y, M, a.n_boot)
pt_ll = {n: float(tab_now.loc[tab_now.model == n, "logloss"].iloc[0]) for n in order_now}
sel_now, best_now = pc.pick_simplest(order_now, {n: B[n]["logloss"] for n in order_now}, pt_ll)
rows = []
for n in order_now:
    r = {"model": n}
    for m in ("brier", "logloss", "ece"):
        d = B[n][m] - B["RAW r"][m]
        r[f"d_{m}_vs_raw"] = f"{d.mean():+.5f} [{np.percentile(d, 2.5):+.5f},{np.percentile(d, 97.5):+.5f}]"
    rows.append(r)
pd.DataFrame(rows).to_csv(OUT / "q_now_bootstrap_vs_raw.csv", index=False)
print(pd.DataFrame(rows).to_string(index=False))
print(f"\n>>> best validation log loss: {best_now}\n>>> SELECTED q_now (simplest not significantly worse): {sel_now}")

# =====================================================================================
# q_fail
# =====================================================================================
print("\n" + "=" * 90 + f"\n[B] q_fail (target: still wrong at iteration 50).  Region: syndrome>0 and t>={T_MIN}\n" + "=" * 90)
yf_tr, yf_va = tr["y_fail"].to_numpy(), va["y_fail"].to_numpy()
fm_c = pc.fit_mask(tr, "q_fail", "syn>0", T_MIN); fm_a = pc.fit_mask(tr, "q_fail", "all", T_MIN)
mask_f = pc.eval_mask(va, "q_fail", T_MIN)
def fitF(feats, m):
    return cl.LogisticCal(feats).fit(tr[m], yf_tr[m])
ks_rows = []
for k in KS:
    c = fitF(["log_r", f"dlog_k{k}", "t"], fm_c)
    ks_rows.append({"k": k, **cl.score(yf_va[mask_f], c.predict(va)[mask_f])})
ks_tab = pd.DataFrame(ks_rows); ks_tab.to_csv(OUT / "q_fail_k_search.csv", index=False)
best_k = int(ks_tab.loc[ks_tab.logloss.idxmin(), "k"])
print("window search (model F2, validation log loss on the common region):")
print(ks_tab[["k", "logloss", "brier", "pr_auc", "auc"]].round(5).to_string(index=False))
print(f"selected k = {best_k}")
dk, dsk = f"dlog_k{best_k}", f"dlsyn_k{best_k}"
fail = {
    "RAW r": (cl.IdentityRaw(), "none"),
    "F0 log r [syn>0]": (fitF(["log_r"], fm_c), "syn>0"),
    "F1 +dlog_k [syn>0]": (fitF(["log_r", dk], fm_c), "syn>0"),
    "F2 +t [syn>0]": (fitF(["log_r", dk, "t"], fm_c), "syn>0"),
    "F2a +t [all t>=t_min]": (fitF(["log_r", dk, "t"], fm_a), "all"),
    "F4 +t,lsyn [syn>0]": (fitF(["log_r", dk, "t", "lsyn"], fm_c), "syn>0"),
    "F5 +t,lsyn,min_abs,dlsyn [syn>0]": (fitF(["log_r", dk, "t", "lsyn", "min_abs", dsk], fm_c), "syn>0"),
}
order_f = list(fail)
P_f = {n: c.predict(va) for n, (c, _) in fail.items()}
tab_f = pd.DataFrame([{"model": n, **pc.score2(yf_va[mask_f], p[mask_f])} for n, p in P_f.items()])
tab_f.to_csv(OUT / "q_fail_val_region.csv", index=False)
Yf = yf_va.reshape(nf_va, T); Mf = mask_f.reshape(nf_va, T)
pos_frames = int(((Yf * Mf).sum(1) > 0).sum())
print(f"\nvalidation region rows {mask_f.sum():,}, prevalence {yf_va[mask_f].mean():.4f}, failing frames inside region {pos_frames}")
print(tab_f.drop(columns=["n", "ece_fixed10"]).round(5).to_string(index=False))
Bf = pc.boot_arrays({n: p.reshape(nf_va, T) for n, p in P_f.items()}, Yf, Mf, a.n_boot)
pt_llf = {n: float(tab_f.loc[tab_f.model == n, "logloss"].iloc[0]) for n in order_f}
sel_f, best_f = pc.pick_simplest(order_f, {n: Bf[n]["logloss"] for n in order_f}, pt_llf)
rows = []
for n in order_f:
    r = {"model": n}
    for m in ("brier", "logloss", "ece"):
        d = Bf[n][m] - Bf["F0 log r [syn>0]"][m]
        r[f"d_{m}_vs_F0"] = f"{d.mean():+.5f} [{np.percentile(d, 2.5):+.5f},{np.percentile(d, 97.5):+.5f}]"
    rows.append(r)
pd.DataFrame(rows).to_csv(OUT / "q_fail_bootstrap_vs_F0.csv", index=False)
print(pd.DataFrame(rows).to_string(index=False))
print(f"\n>>> best validation log loss: {best_f}\n>>> SELECTED q_fail: {sel_f}")

rows = []
for s in SNRS:
    mk = mask_f & (va.snr == s).to_numpy()
    for n in ("RAW r", "F0 log r [syn>0]", sel_f):
        sc = pc.score2(yf_va[mk], P_f[n][mk])
        rows.append({"snr": s, "model": n, "rows": int(mk.sum()), "ece": sc["ece_eqmass20"], "brier": sc["brier"],
                     "auc": sc["auc"], "pr_auc": sc["pr_auc"]})
pd.DataFrame(rows).to_csv(OUT / "q_fail_val_by_snr.csv", index=False)
print("\nq_fail by SNR"); print(pd.DataFrame(rows).round(4).to_string(index=False))

# =====================================================================================
# plots
# =====================================================================================
fig, ax = plt.subplots(1, len(SNRS), figsize=(4 * len(SNRS), 4), sharey=True)
for a_, s in zip(np.atleast_1d(ax), SNRS):
    mk = mask_now & (va.snr == s).to_numpy()
    for n in ("RAW r", "M2 ext (log r,S) [all]", sel_now):
        rel = cl.reliability(y_va[mk], P_now[n][mk], 15)
        a_.plot(np.maximum(rel.mean_pred, 1e-6), np.maximum(rel.obs_rate, 1e-6), "o-", ms=3, label=n)
    a_.plot([1e-6, 1], [1e-6, 1], "k--"); a_.set_xscale("log"); a_.set_yscale("log"); a_.set_title(f"q_now, {s:.2f} dB")
    a_.set_xlabel("predicted"); a_.grid(alpha=.3, which="both")
np.atleast_1d(ax)[0].set_ylabel("observed"); np.atleast_1d(ax)[0].legend(fontsize=6)
plt.tight_layout(); plt.savefig(OUT / "q_now_reliability_by_snr.png", dpi=150); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
for n in ("RAW r", "F0 log r [syn>0]", sel_f):
    rel = cl.reliability(yf_va[mask_f], P_f[n][mask_f], 15)
    ax[0].plot(np.maximum(rel.mean_pred, 1e-4), np.maximum(rel.obs_rate, 1e-4), "o-", ms=3, label=n)
ax[0].plot([1e-4, 1], [1e-4, 1], "k--"); ax[0].set_xscale("log"); ax[0].set_yscale("log")
ax[0].set_title("q_fail reliability (validation, syn>0, t>=t_min)"); ax[0].legend(fontsize=7); ax[0].grid(alpha=.3, which="both")
from sklearn.metrics import roc_auc_score
for n in ("RAW r", "F0 log r [syn>0]", sel_f):
    pts = []
    for t in range(T_MIN, T + 1):
        mk = mask_f & (va.t == t).to_numpy()
        if mk.sum() > 50 and len(np.unique(yf_va[mk])) == 2:
            pts.append((t, roc_auc_score(yf_va[mk], P_f[n][mk])))
    if pts:
        ax[1].plot(*zip(*pts), label=n)
ax[1].set_xlabel("iteration"); ax[1].set_ylabel("AUC for final failure"); ax[1].set_title("abandon-head AUC by iteration (syn>0)")
ax[1].legend(fontsize=7); ax[1].grid(alpha=.3)
plt.tight_layout(); plt.savefig(OUT / "q_fail_reliability_and_auc.png", dpi=150); plt.close()

# =====================================================================================
# artifacts + model card (interface for Person 3)
# =====================================================================================
meta = dict(train_snrs=SNRS, t_min=T_MIN, log_r_floor=cl.LOG_R_FLOOR)
c_now, reg_now = now[sel_now]; c_fail, reg_f = fail[sel_f]
cl.save_calibrator(c_now, OUT / "q_now.json", REPO, "train", target="cur_err", model_name=sel_now, fit_region=reg_now, **meta)
cl.save_calibrator(c_fail, OUT / "q_fail.json", REPO, "train", target="final_failure", model_name=sel_f, fit_region=reg_f,
                   k=best_k, **meta)
card = {"q_now": {"file": "q_now.json", "name": sel_now, "fit_region": reg_now, "valid_where": "syndrome_weight > 0"},
        "q_fail": {"file": "q_fail.json", "name": sel_f, "fit_region": reg_f, "k": best_k, "t_min": T_MIN,
                   "valid_where": f"syndrome_weight > 0 and iteration >= {T_MIN}"},
        "inputs": "scalar traces (log_r, S, min_abs, syndrome_weight); never info_err, cur_err, final_failure",
        "supported": "AWGN, BPSK, 5G NR BG2 K=110 N=220, decoder 50 iterations, SNR 2.0-2.75 dB (matched)",
        "note": "outside the valid region the calibrators are extrapolating; stopping rules already accept at syndrome 0"}
json.dump(card, open(OUT / "model_card.json", "w", encoding="utf-8"), indent=2)
json.dump({"selected_q_now": sel_now, "selected_q_fail": sel_f, "best_q_now": best_now, "best_q_fail": best_f, "k": best_k,
           "calibration_fit_and_eval_wall_seconds": round(time.perf_counter() - _T0, 1),
           "train_rows": int(len(tr)), "train_frames": int(len(tr_d["frame_id"]))},
          open(OUT / "selection_summary.json", "w", encoding="utf-8"), indent=2)
print(f"\nSaved artifacts and model_card.json in {OUT}\nTest data was NOT opened.")
