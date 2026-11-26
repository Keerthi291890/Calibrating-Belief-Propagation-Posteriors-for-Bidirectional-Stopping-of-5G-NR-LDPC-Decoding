"""
Person 3 analysis of the FROZEN final CSVs (reads results/person2_v3/final_*.csv only; never opens trace or test files).

    python src/p3_analysis.py [--in-dir results/person2_v3] [--out-dir results/person3]

Writes tables (csv) and figures (png) for E1 (headroom, from test CSV), E3 (trade-off), E4 (stopping under mismatch),
and a number check.  All statements are EMPIRICAL.  Operation counts are approximate scalar-operation estimates.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
REPO = Path(__file__).resolve().parents[1]
ap.add_argument("--in-dir", default=str(REPO / "results" / "person2_v3"))
ap.add_argument("--out-dir", default=str(REPO / "results" / "person3"))
a = ap.parse_args()
IN, OUT = Path(a.in_dir), Path(a.out_dir); OUT.mkdir(parents=True, exist_ok=True)
t = pd.read_csv(IN / "final_stopping_test.csv")
m = pd.read_csv(IN / "final_stopping_mismatch.csv")
SNRS = sorted(t.snr.unique()); RHOS = sorted(t.rho.unique())
SHORT = {"B1 fixed 50": "B1", "B2 syndrome-zero": "B2", "B4 ideal info-CRC (ceiling)": "B4", "B3 min|LLR| accept": "B3",
         "B5 syndrome + cap": "B5", "A1 mean-entropy accept": "A1", "P-raw accept-only": "P-raw acc", "P-raw abandon-only": "P-raw ab",
         "P-raw both": "P-raw both", "P-cal accept-only": "P-cal acc", "P-cal abandon-only": "P-cal ab", "P-cal both": "P-cal both"}
t["rule_s"] = t.rule.map(SHORT); m["rule_s"] = m.rule.map(SHORT)
m["budget_events"] = m.excess_vs_budget.str.split("/").str[1].astype(float)   # text column "excess/budget"
m["excess_ratio"] = m.excess_events / m.budget_events
t["excess_ratio"] = t.excess_events / t.budget_events
checks = []
def check(name, cond):
    checks.append((name, bool(cond))); print(("[PASS] " if cond else "[FAIL] ") + name)

# ---------------------------------------------------------------- E1: headroom (test CSV, rho-independent rules)
h = t[(t.rho == RHOS[1]) & t.rule_s.isin(["B1", "B2", "B4"])].pivot(index="snr", columns="rule_s", values=["mean_it", "median_it", "p95_it", "p99_it"])
e1 = pd.DataFrame({"B2_mean": h["mean_it"]["B2"], "B4_mean": h["mean_it"]["B4"]})
e1["B4_headroom_vs_B2_pct"] = 100 * (1 - e1.B4_mean / e1.B2_mean)
e1["B2_median"], e1["B4_median"] = h["median_it"]["B2"], h["median_it"]["B4"]
e1["B2_p95"], e1["B2_p99"] = h["p95_it"]["B2"], h["p99_it"]["B2"]
e1.round(3).to_csv(OUT / "p3_e1_headroom_test_csv.csv"); print("\nE1 headroom (B4 ceiling vs B2)\n", e1.round(2))

# ---------------------------------------------------------------- E3: trade-off table + figure
cols = ["snr", "rho", "rule_s", "mean_it", "median_it", "p95_it", "p99_it", "saving_vs_B2_pct", "excess_events", "budget_events",
        "excess_ratio", "budget_met_on_test", "nack_rate", "undetected_per_1000", "net_bler_change", "total_saving_pct_cx10"]
t[cols].round(4).to_csv(OUT / "p3_tradeoff_table.csv", index=False)
fig, ax = plt.subplots(1, len(SNRS), figsize=(4.2 * len(SNRS), 3.8), sharey=False)
mk = {"B3": "s", "B5": "D", "A1": "^", "P-raw both": "x", "P-cal acc": "v", "P-cal ab": "<", "P-cal both": "o"}
for i, s in enumerate(SNRS):
    d = t[t.snr == s]
    for r, mkr in mk.items():
        x = d[d.rule_s == r].sort_values("rho")
        ax[i].plot(x.excess_events / x.failing_frames * 100, x.saving_vs_B2_pct, marker=mkr, label=r, lw=1.2)
    b4 = d[d.rule_s == "B4"].saving_vs_B2_pct.iloc[0]
    ax[i].axhline(b4, color="k", ls="--", lw=.8, label="B4 (info-CRC accept only)")
    for rho in RHOS: ax[i].axvline(rho * 100, color="grey", ls=":", lw=.6)
    ax[i].set(title=f"{s:.2f} dB", xlabel="realised excess events (% of failing frames)")
    if i == 0: ax[i].set_ylabel("iteration saving vs B2 (%)"); ax[i].legend(fontsize=6)
fig.suptitle("E3: saving vs realised excess (test; dotted = budgets 5/10/20%; points beyond a dotted line exceed that budget)", fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "p3_fig_tradeoff.png", dpi=150); plt.close(fig)

# decomposition: accept vs abandon share of saving (P-cal)
dec = []
for s in SNRS:
    for rho in RHOS:
        g = t[(t.snr == s) & (t.rho == rho)].set_index("rule_s").saving_vs_B2_pct
        dec.append(dict(snr=s, rho=rho, accept_only=g["P-cal acc"], abandon_only=g["P-cal ab"], both=g["P-cal both"],
                        abandon_share_of_both_pct=100 * g["P-cal ab"] / g["P-cal both"]))
dec = pd.DataFrame(dec); dec.round(3).to_csv(OUT / "p3_saving_decomposition.csv", index=False); print("\nDecomposition\n", dec.round(2))
fig, ax = plt.subplots(figsize=(6, 3.5)); d10 = dec[dec.rho == 0.10]; w = .27; x = np.arange(len(SNRS))
for j, c in enumerate(["accept_only", "abandon_only", "both"]): ax.bar(x + (j - 1) * w, d10[c], w, label=c)
ax.set_xticks(x, [f"{s:.2f}" for s in SNRS]); ax.set(xlabel="SNR (dB)", ylabel="iteration saving vs B2 (%)", title="P-cal saving by head (rho = 10%)"); ax.legend()
fig.tight_layout(); fig.savefig(OUT / "p3_fig_decomposition.png", dpi=150); plt.close(fig)

# operations sensitivity: P-cal both vs B5 at cx 3/10/25
ops = []
for s in SNRS:
    for rho in RHOS:
        g = t[(t.snr == s) & (t.rho == rho)].set_index("rule_s")
        for cx in (3, 10, 25):
            ops.append(dict(snr=s, rho=rho, cx=cx, pcal_both=g.loc["P-cal both", f"total_saving_pct_cx{cx}"], b5=g.loc["B5", f"total_saving_pct_cx{cx}"]))
ops = pd.DataFrame(ops); ops["pcal_minus_b5"] = ops.pcal_both - ops.b5; ops.round(3).to_csv(OUT / "p3_ops_sensitivity.csv", index=False)
print("\nOps: P-cal both beats B5 in total ops in", int((ops.pcal_minus_b5 > 0).sum()), "of", len(ops), "SNR x rho x cx cells")

# ---------------------------------------------------------------- E4: stopping under mismatch
mm = m[m.rule_s.isin(["P-cal both", "B5", "P-raw both"]) & (m.delta != 0)].copy()
mm["variant"] = mm.rule_s + " (" + mm.calibrators.astype(str) + ")"
piv = mm.pivot_table(index=["variant", "snr"], columns="delta", values="excess_ratio"); piv.round(2).to_csv(OUT / "p3_mismatch_budget_ratio.csv")
print("\nMismatch excess/budget (rho = 10%)\n", piv.round(2))
fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
for k, v in enumerate(["P-cal both (matched)", "B5 (matched)", "P-raw both (matched)"]):
    p = piv.loc[v]; im = ax[k].imshow(p.values, cmap="RdYlGn_r", vmin=0, vmax=2, aspect="auto")
    ax[k].set_xticks(range(p.shape[1]), [f"{c:+d}" for c in p.columns]); ax[k].set_yticks(range(p.shape[0]), [f"{s:.2f}" for s in p.index])
    for (i, j), z in np.ndenumerate(p.values): ax[k].text(j, i, f"{z:.2f}", ha="center", va="center", fontsize=7)
    ax[k].set(title=v + "\nexcess/budget (>1 = over budget)", xlabel="delta (dB)", ylabel="SNR (dB)")
fig.tight_layout(); fig.savefig(OUT / "p3_fig_mismatch_budget_heatmap.png", dpi=150); plt.close(fig)
refit = m[(m.rule_s == "P-cal both") & (m.snr == 2.25) & (m.delta != 0)].pivot(index="delta", columns="calibrators", values="excess_ratio")
refit.round(2).to_csv(OUT / "p3_refit_vs_matched_2.25dB.csv"); print("\nRefit vs matched at 2.25 dB (excess/budget)\n", refit.round(2))

# ---------------------------------------------------------------- number checks (claims used in the text)
pc10 = t[(t.rule_s == "P-cal both") & (t.rho == 0.10)].set_index("snr")
b5_10 = t[(t.rule_s == "B5") & (t.rho == 0.10)].set_index("snr")
check("P-cal both saves 12.6/5.8/3.3/1.7% vs B2 at rho 10%", np.allclose(pc10.saving_vs_B2_pct.round(1).values, [12.6, 5.8, 3.3, 1.7]))
check("B5 saves 9.1/3.4/1.0/0.9% vs B2 at rho 10%", np.allclose(b5_10.saving_vs_B2_pct.round(1).values, [9.1, 3.4, 1.0, 0.9]))
check("P-cal both over budget in 2 of 12 test cells", int((~t[t.rule_s == "P-cal both"].budget_met_on_test).sum()) == 2)
check("B5 over budget in 5 of 12 test cells", int((~t[t.rule_s == "B5"].budget_met_on_test).sum()) == 5)
mp = m[(m.rule_s == "P-cal both") & (m.calibrators == "matched") & (m.delta != 0)]
check("P-cal both matched: 6 of 18 delta!=0 cells over budget", int((~mp.budget_met).sum()) == 6)
check("... all 6 have delta < 0", bool((mp[~mp.budget_met].delta < 0).all()))
check("P-cal both refit: 6 of 6 within budget", bool(m[(m.rule_s == "P-cal both") & (m.calibrators == "refit")].budget_met.all()))
check("P-cal both loses to B5 in total ops (cx=10) in all rho 10% cells", bool((pc10.total_saving_pct_cx10 < b5_10.total_saving_pct_cx10).all()))
check("abandon-only >= 80% of P-cal both saving at rho 10%", bool((dec[dec.rho == 0.10].abandon_share_of_both_pct >= 80).all()))
pd.DataFrame(checks, columns=["check", "pass"]).to_csv(OUT / "p3_number_checks.csv", index=False)
print(f"\n{sum(c for _, c in checks)}/{len(checks)} number checks pass"); raise SystemExit(0 if all(c for _, c in checks) else 1)
