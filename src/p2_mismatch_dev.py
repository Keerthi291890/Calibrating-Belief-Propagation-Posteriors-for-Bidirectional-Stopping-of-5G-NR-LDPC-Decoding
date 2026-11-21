"""
Mismatch PREVIEW on the refit files (train_mismatch_*, 2.25 dB, 10,000 paired frames per delta).
These files are explicitly TRAINING data, so using them here does not touch the sealed test set.
Frames are split by parity of frame_id (same split for every delta): even = refit-fit half, odd = evaluation half.
Compares, on the evaluation half, RAW r vs the MATCHED calibrators (fitted on matched train) vs REFIT calibrators.
Reference delta = 0 is the matched validation data at 2.25 dB (different frames).

    python src/p2_mismatch_dev.py --data-root E:/math_dataset
"""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import calib_lib as cl
import p2_common as pc
from risk_model import RiskModel

ap = argparse.ArgumentParser()
ap.add_argument("--data-root", default=None)
ap.add_argument("--out-dir", default=str(pc.REPO / "results" / "person2_v3"))
ap.add_argument("--n-boot", type=int, default=50)
a = ap.parse_args()
OUT = Path(a.out_dir); root = pc.find_root(a.data_root)
rm = RiskModel(OUT); K, TM = rm.k, rm.t_min
raw = {"q_now": cl.IdentityRaw(), "q_fail": cl.IdentityRaw()}
matched = {"q_now": rm.q_now_cal, "q_fail": rm.q_fail_cal}
pd.set_option("display.width", 220, "display.max_columns", 30)
print(f"matched models: q_now = {rm.card['q_now']['name']} | q_fail = {rm.card['q_fail']['name']} (k={K}, t_min={TM})")

def subset(d, keep):
    return {k: v[keep] for k, v in d.items() if hasattr(v, "shape") and v.ndim >= 1 and len(v) == len(keep)}

rows = []
va = pc.load_group(root, "validation", [2.25])
rows += pc.head_rows(va, K, TM, {"raw": raw, "matched": matched}, a.n_boot, {"delta": 0, "source": "matched validation 2.25"})
for dl in pc.DELTAS:
    d = pc.load_group(root, "refit", [2.25], delta=dl)
    even = (d["frame_id"] % 2) == 0
    fit, ev = subset(d, even), subset(d, ~even)
    tf = pc.table(fit, (K,))
    refit = {}
    for head, ycol in (("q_now", "y_now"), ("q_fail", "y_fail")):
        region = rm.card[head]["fit_region"]
        m = pc.fit_mask(tf, head, "all" if region in ("all", "none") else "syn>0", TM)
        refit[head] = pc.refit_like(matched[head], tf[m], tf.loc[m, ycol].to_numpy())
    rows += pc.head_rows(ev, K, TM, {"raw": raw, "matched": matched, "refit": {"q_now": refit["q_now"], "q_fail": refit["q_fail"]}},
                         a.n_boot, {"delta": dl, "source": "refit eval half"})
res = pd.DataFrame(rows); res.to_csv(OUT / "mismatch_dev.csv", index=False)
for head in ("q_now", "q_fail"):
    r = res[res["head"] == head]
    print(f"\n=== {head}: region syn>0{' & t>=%d' % TM if head == 'q_fail' else ''} ===")
    print(r.pivot(index="delta", columns="model", values="ece").round(5).to_string()); print("(ECE)")
    print(r.pivot(index="delta", columns="model", values="brier").round(5).to_string()); print("(Brier)")
    print(r[r.model == "matched"][["delta", "bler", "failing_frames", "region_rows", "prevalence", "cal_slope", "cal_intercept"]].round(4).to_string(index=False))

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for a_, head in zip(ax, ("q_now", "q_fail")):
    for mdl in ("raw", "matched", "refit"):
        r = res[(res["head"] == head) & (res.model == mdl)].sort_values("delta")
        a_.errorbar(r.delta, r.ece, yerr=[np.maximum(r.ece - r.ece_lo, 0), np.maximum(r.ece_hi - r.ece, 0)], marker="o", ms=4, capsize=2, label=mdl)
    a_.set_xlabel("assumed-noise mismatch delta (dB)"); a_.set_ylabel("ECE (region syn>0)"); a_.set_title(f"{head}, 2.25 dB (preview)")
    a_.grid(alpha=.3); a_.legend(fontsize=8)
plt.tight_layout(); plt.savefig(OUT / "mismatch_dev_ece_vs_delta.png", dpi=150); plt.close()
print("\nPREVIEW ONLY: one SNR, evaluation half of the refit files. Held-out mismatch results come from the sealed test files.")
