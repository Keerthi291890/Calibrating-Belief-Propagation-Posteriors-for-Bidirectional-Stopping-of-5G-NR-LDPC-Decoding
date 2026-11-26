"""
E1 (Person 3): how much room is there?  Uses TRAIN and VALIDATION files only (never the sealed test).

    python src/p3_e1_headroom.py --data-root <root> [--out-dir results/person3]

Per SNR and split: BLER, iteration distribution under B1 (fixed 50), B2 (syndrome-zero), B4 (ideal info-CRC),
share of B2 iterations spent on failing frames, and the B2-vs-B4 gap.  B4 stops only on correct frames, so it is an
ACCEPT-side bound; it is not a bound for abandonment (abandoning failing frames can save more than B4 does).
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import p2_common as pc
import p3_data as p3d
import p2_stopping as ps

ap = argparse.ArgumentParser()
ap.add_argument("--data-root", default=None)
ap.add_argument("--out-dir", default=str(pc.REPO / "results" / "person3"))
a = ap.parse_args(); root = pc.find_root(a.data_root); out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
rows = []
for split in ("train", "validation"):
    for s in pc.SNRS_DEFAULT:
        d = p3d.load_group(root, split, [s])
        pos = d["syndrome_weight"] > 0; wrong = d["info_err"] > 0; ff = d["final_failure"].astype(bool)
        t2, t4 = ps.first_t(~pos), ps.first_t(~wrong)          # first iteration with syndrome 0 / zero info-bit errors (50+1 if never)
        t2c, t4c = np.minimum(t2, 50), np.minimum(t4, 50)
        rows.append(dict(split=split, snr=s, frames=len(ff), bler=ff.mean(),
                         B2_mean=t2c.mean(), B2_median=np.median(t2c), B2_p95=np.percentile(t2c, 95), B2_p99=np.percentile(t2c, 99),
                         B4_mean=t4c.mean(), B4_median=np.median(t4c), B4_p99=np.percentile(t4c, 99),
                         B4_headroom_vs_B2_pct=100 * (1 - t4c.mean() / t2c.mean()),
                         share_B2_iters_in_failing_frames=t2c[ff].sum() / t2c.sum(),
                         frac_syndrome_zero_but_info_wrong=float(((~pos) & wrong).any(1).mean())))
df = pd.DataFrame(rows); df.round(4).to_csv(out / "p3_e1_headroom.csv", index=False); print(df.round(3).to_string())
