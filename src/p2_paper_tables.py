"""
Paper tables and figures from the FINAL test result CSVs (outputs of p2_final_test.py).
Reads only results/person2_v3/final_*.csv; never opens test traces, so it may be run any number of times.

    python src/p2_paper_tables.py
Writes paper_tables.md and fig_*.png into the same folder.
"""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--out-dir", default=str(Path(__file__).resolve().parents[1] / "results" / "person2_v3"))
a = ap.parse_args()
OUT = Path(a.out_dir)
mt = pd.read_csv(OUT / "final_matched_test.csv")
mm = pd.read_csv(OUT / "final_mismatch_test.csv")
st = pd.read_csv(OUT / "final_stopping_test.csv")
sm = pd.read_csv(OUT / "final_stopping_mismatch.csv")
md = []
def ci(v, lo, hi, nd=4): return f"{v:.{nd}f} [{lo:.{nd}f}, {hi:.{nd}f}]"
def table(df):
    try:
        return df.to_markdown(index=False)
    except ImportError:                       # tabulate not installed: plain text block
        return "```\n" + df.to_string(index=False) + "\n```"
def add(title, df, note=None):
    md.append(f"\n### {title}\n"); md.append(table(df))
    if note: md.append(f"\n*{note}*")

# ---------------------------------------------------------------- 1. matched calibration
for head in ("q_now", "q_fail"):
    rows = []
    for s in sorted(mt.snr.unique()):
        r = mt[(mt["head"] == head) & (mt.snr == s)].set_index("model")
        for m in ("raw", "matched"):
            x = r.loc[m]
            rows.append({"SNR": s, "model": m, "ECE [95% CI]": ci(x.ece, x.ece_lo, x.ece_hi), "Brier": f"{x.brier:.4f}",
                         "log loss": f"{x.logloss:.4f}", "slope": f"{x.cal_slope:.2f}", "intercept": f"{x.cal_intercept:+.2f}",
                         "AUC": f"{x.auc:.3f}", "PR-AUC": f"{x.pr_auc:.3f}",
                         "ECE change vs raw": "" if m == "raw" else f"{100 * (x.ece / r.loc['raw'].ece - 1):+.0f}%"})
    add(f"Matched-channel test calibration: {head} (region: syndrome>0{', t>=5' if head == 'q_fail' else ''})", pd.DataFrame(rows),
        "H2 target: ECE at least halved (<= -50%). Percentile-bootstrap ECE intervals are biased upward; compare models with the Brier/log-loss columns too.")

# ---------------------------------------------------------------- 2. stopping on matched test
sel = ["B2 syndrome-zero", "B5 syndrome + cap", "P-cal abandon-only", "P-cal both", "P-raw both", "B4 ideal info-CRC (ceiling)"]
for rho in (0.10, 0.20):
    rows = []
    for s in sorted(st.snr.unique()):
        r = st[(st.snr == s) & (st.rho == rho)].set_index("rule")
        for rule in sel:
            x = r.loc[rule]
            rows.append({"SNR": s, "rule": rule, "params": x.params, "mean it": f"{x.mean_it:.3f}",
                         "iter saving vs B2 %": ci(x.saving_vs_B2_pct, x.saving_lo, x.saving_hi, 1),
                         "total-ops saving % (cx=10)": ci(x.total_ops_saving_vs_B2_pct, x.total_saving_lo, x.total_saving_hi, 1),
                         "cx=3 / cx=25": f"{x.total_saving_pct_cx3:.1f} / {x.total_saving_pct_cx25:.1f}",
                         "excess/budget": f"{int(x.excess_events)}/{x.budget_events:.1f}", "budget met (test)": bool(x.budget_met_on_test),
                         "undetected | accepted": f"{x.undetected_given_accepted:.4f}" if pd.notna(x.undetected_given_accepted) else "",
                         "NACK rate": f"{x.nack_rate:.3f}", "net BLER change": f"{x.net_bler_change:+.4f}"})
    add(f"Stopping on matched test, budget rho = {rho:.0%} (thresholds frozen on validation)", pd.DataFrame(rows),
        "Thresholds minimise mean iterations under the excess-event budget; operation savings are supplementary results (approximate scalar-operation estimates, not measured instructions or energy).")
    d = st[(st.rho == rho) & (st.rule == "P-cal both")][["snr", "pcal_both_faster_than_B5_by", "diff_lo", "diff_hi"]]
    add(f"P-cal both vs B5: paired mean-iteration advantage, rho = {rho:.0%} (positive = P-cal faster)",
        d.rename(columns={"snr": "SNR", "pcal_both_faster_than_B5_by": "iterations", "diff_lo": "CI low", "diff_hi": "CI high"}).round(3))

comp = st.groupby("rule").agg(cells=("budget_met_on_test", "size"), violated=("budget_met_on_test", lambda x: int((~x).sum()))).reset_index()
add("Budget compliance on TEST (all SNR x rho in {5,10,20}%)", comp,
    "Thresholds were tuned to the edge of the budget on validation, so some rules exceed it on test. Rules that violate the budget are not equal-budget comparisons.")

# ---------------------------------------------------------------- 3. mismatch calibration
for head in ("q_now", "q_fail"):
    rows = []
    for s in sorted(mm.snr.unique()):
        r = mm[(mm["head"] == head) & (mm.snr == s)]
        p = r.pivot(index="delta", columns="model", values="ece")
        for dl in p.index:
            rows.append({"SNR": s, "delta dB": int(dl), **{k: (f"{p.loc[dl, k]:.4f}" if pd.notna(p.loc[dl, k]) else "") for k in p.columns},
                         "matched / matched@0": f"{p.loc[dl, 'matched'] / p.loc[0, 'matched']:.1f}x"})
    add(f"Mismatch test: ECE by delta, {head}", pd.DataFrame(rows), "delta = 0 is the paired matched reference (same frames). 'refit' exists only at 2.25 dB.")
asym = []
for head in ("q_now", "q_fail"):
    for model in ("raw", "matched"):
        r = mm[(mm["head"] == head) & (mm.model == model) & (mm.delta != 0)]
        for s in sorted(r.snr.unique()):
            x = r[r.snr == s]
            asym.append({"head": head, "model": model, "SNR": s, "mean ECE delta<0": x[x.delta < 0].ece.mean(),
                         "mean ECE delta>0": x[x.delta > 0].ece.mean(), "worst delta": int(x.loc[x.ece.idxmax(), "delta"])})
add("H4 asymmetry: mean ECE for negative vs positive delta", pd.DataFrame(asym).round(4))

# ---------------------------------------------------------------- 4. stopping under mismatch
sm["ratio"] = sm.excess_vs_budget.str.split("/").apply(lambda t: float(t[0]) / max(float(t[1]), 1e-9))
rows = []
for s in sorted(sm.snr.unique()):
    r = sm[(sm.snr == s) & (sm.rule == "P-cal both")]
    for _, x in r.iterrows():
        rows.append({"SNR": s, "delta": int(x.delta), "calibrators": x.calibrators, "excess/budget": x.excess_vs_budget,
                     "budget met": bool(x.budget_met), "undetected per 1000": f"{x.undetected_per_1000:.2f}", "mean it": f"{x.mean_it:.2f}",
                     "iter saving vs B2 same delta %": f"{x.saving_vs_B2_same_delta_pct:.1f}",
                     "total-ops saving %": f"{x.total_ops_saving_vs_B2_same_delta_pct:.1f}"})
add("Stopping reliability under mismatch: P-cal both with FROZEN matched thresholds (rho = 10%)", pd.DataFrame(rows),
    "Excess is counted against the mismatch run's own full 50-iteration outcome.")
v = sm[(sm.calibrators == "matched") & (sm.delta != 0)].copy()
v["sign"] = np.where(v.delta < 0, "delta<0 (decoder overconfident)", "delta>0 (decoder conservative)")
vs = v.groupby(["rule", "sign"]).agg(cells=("budget_met", "size"), budget_violated=("budget_met", lambda x: int((~x).sum()))).reset_index()
add("Budget violations under mismatch by rule and sign of delta (matched thresholds)", vs)

# ---------------------------------------------------------------- figures
fig, ax = plt.subplots(2, 3, figsize=(14, 7), sharex=True)
for i, head in enumerate(("q_now", "q_fail")):
    for j, s in enumerate((2.00, 2.25, 2.50)):
        for model in ("raw", "matched", "refit"):
            r = mm[(mm["head"] == head) & (mm.snr == s) & (mm.model == model)].sort_values("delta")
            if len(r):
                ax[i, j].errorbar(r.delta, r.ece, yerr=[np.maximum(r.ece - r.ece_lo, 0), np.maximum(r.ece_hi - r.ece, 0)], marker="o", ms=4, capsize=2, label=model)
        ax[i, j].set_title(f"{head}, {s:.2f} dB"); ax[i, j].set_yscale("log"); ax[i, j].grid(alpha=.3, which="both")
        if i == 1: ax[i, j].set_xlabel("assumed-noise mismatch delta (dB)")
        if j == 0: ax[i, j].set_ylabel("ECE (log scale)")
ax[0, 0].legend(fontsize=8)
plt.tight_layout(); plt.savefig(OUT / "fig_ece_vs_delta.png", dpi=150); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
for rho, a_ in zip((0.10, 0.20), ax):
    snrs = sorted(st.snr.unique()); xs = np.arange(len(snrs)); w = 0.2
    for k, (rule, col, lo, hi, lab) in enumerate([("B5 syndrome + cap", "saving_vs_B2_pct", "saving_lo", "saving_hi", "B5 iterations"),
                                                  ("P-cal both", "saving_vs_B2_pct", "saving_lo", "saving_hi", "P-cal iterations"),
                                                  ("B5 syndrome + cap", "total_ops_saving_vs_B2_pct", "total_saving_lo", "total_saving_hi", "B5 total ops"),
                                                  ("P-cal both", "total_ops_saving_vs_B2_pct", "total_saving_lo", "total_saving_hi", "P-cal total ops")]):
        r = st[(st.rho == rho) & (st.rule == rule)].sort_values("snr")
        a_.bar(xs + (k - 1.5) * w, r[col], w, yerr=[np.maximum(r[col] - r[lo], 0), np.maximum(r[hi] - r[col], 0)], capsize=2, label=lab, hatch="//" if "ops" in lab else None)
    a_.axhline(0, color="k", lw=.8); a_.set_xticks(xs); a_.set_xticklabels([f"{s:.2f}" for s in snrs])
    a_.set_xlabel("SNR (dB)"); a_.set_ylabel("saving vs syndrome-zero (%)"); a_.set_title(f"budget rho = {rho:.0%} (test, frozen thresholds)"); a_.grid(alpha=.3)
ax[0].legend(fontsize=7)
plt.tight_layout(); plt.savefig(OUT / "fig_savings_iterations_vs_ops.png", dpi=150); plt.close()

fig, ax = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
for a_, s in zip(ax, (2.00, 2.25, 2.50)):
    for cal in ("matched", "refit"):
        r = sm[(sm.snr == s) & (sm.rule == "P-cal both") & (sm.calibrators == cal)].sort_values("delta")
        if len(r): a_.plot(r.delta, r.ratio, "o-", label=cal)
    a_.axhline(1.0, color="r", ls="--", lw=1); a_.set_title(f"{s:.2f} dB"); a_.set_xlabel("delta (dB)"); a_.grid(alpha=.3)
ax[0].set_ylabel("excess events / budget (rho=10%)"); ax[0].legend(fontsize=8)
plt.tight_layout(); plt.savefig(OUT / "fig_budget_ratio_vs_delta.png", dpi=150); plt.close()

(OUT / "paper_tables.md").write_text("# Final-test tables (generated by p2_paper_tables.py)\n" + "\n".join(md), encoding="utf-8")
print("wrote", OUT / "paper_tables.md", "and fig_ece_vs_delta.png, fig_savings_iterations_vs_ops.png, fig_budget_ratio_vs_delta.png")
print(comp.to_string(index=False)); print(vs.to_string(index=False))
