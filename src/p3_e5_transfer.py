"""
E5 (Person 3): threshold transfer across SNR, EXPLORATORY and VALIDATION-ONLY (the sealed test was opened once and must stay closed).

    python src/p3_e5_transfer.py --data-root <root> [--rho 0.10] [--out-dir results/person3]

Tune the P-cal both and B5 parameters on validation_i (same procedure as the freeze), apply them unchanged to validation_j,
report excess/budget_j and iteration saving vs B2.  The diagonal is in-sample (tuned and evaluated on the same frames), so
read the off-diagonal cells; they show how a threshold set for one SNR behaves at another.  Uses the frozen calibrators.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import p2_common as pc
import p3_data as p3d
import p2_stopping as ps
from risk_model import RiskModel

ap = argparse.ArgumentParser()
ap.add_argument("--data-root", default=None); ap.add_argument("--rho", type=float, default=0.10)
ap.add_argument("--model-dir", default=str(pc.REPO / "results" / "person2_v3"))
ap.add_argument("--out-dir", default=str(pc.REPO / "results" / "person3"))
a = ap.parse_args(); root = pc.find_root(a.data_root); out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
rm = RiskModel(a.model_dir); S = pc.SNRS_DEFAULT
ctxs = {}
for s in S:
    d = p3d.load_group(root, "validation", [s]); ctxs[s] = ps.make_ctx(d, rm.k, rm.t_min, rm.q_now_cal, rm.q_fail_cal)
tuned = {s: ps.tune(ctxs[s], rhos=(a.rho,))[a.rho] for s in S}
rows = []
for rule in ("P-cal both", "B5 syndrome + cap"):
    for si in S:
        p = tuned[si][rule]["params"]
        for sj in S:
            c = ctxs[sj]; fn = ps.RULES[rule][1]; pf = ps.per_frame(c, *fn(c, p)); sm = ps.summarize(pf, c["ff"])
            b2 = ps.summarize(ps.per_frame(c, *ps.RULES["B2 syndrome-zero"][1](c, {})), c["ff"])["mean_it"]
            budget = a.rho * int(c["ff"].sum())
            rows.append(dict(rule=rule, tuned_at=si, applied_at=sj, params=str(p), excess=sm["excess_events"], budget=budget,
                             excess_ratio=sm["excess_events"] / budget if budget else np.nan, saving_vs_B2_pct=100 * (1 - sm["mean_it"] / b2)))
df = pd.DataFrame(rows); df.round(4).to_csv(out / "p3_e5_transfer.csv", index=False)
fig, ax = plt.subplots(2, 2, figsize=(9, 7))
for i, rule in enumerate(("P-cal both", "B5 syndrome + cap")):
    g = df[df.rule == rule]
    for j, (col, ttl, cmap, kw) in enumerate((("excess_ratio", "excess/budget (>1 = over)", "RdYlGn_r", dict(vmin=0, vmax=2)), ("saving_vs_B2_pct", "saving vs B2 (%)", "viridis", {}))):
        p = g.pivot(index="tuned_at", columns="applied_at", values=col); im = ax[i, j].imshow(p.values, cmap=cmap, **kw)
        ax[i, j].set_xticks(range(len(S)), [f"{s:.2f}" for s in S]); ax[i, j].set_yticks(range(len(S)), [f"{s:.2f}" for s in S])
        for (r, c), z in np.ndenumerate(p.values): ax[i, j].text(c, r, f"{z:.2f}", ha="center", va="center", fontsize=8)
        ax[i, j].set(title=f"{rule}: {ttl}", xlabel="applied at SNR (dB)", ylabel="tuned at SNR (dB)")
fig.suptitle(f"E5 threshold transfer, validation only, rho = {a.rho:.0%} (diagonal is in-sample)", fontsize=9)
fig.tight_layout(); fig.savefig(out / "p3_fig_e5_transfer.png", dpi=150); print(df.round(3).to_string())
