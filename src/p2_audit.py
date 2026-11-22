"""
Statistical audit of the scalar traces (Person 2).  Reads dev/ (train, validation) and the refit files only.
Member 1 already audited structure; this checks what matters for calibration.

    python src/p2_audit.py --data-root E:/math_dataset
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import p2_common as pc
import calib_lib as cl

ap = argparse.ArgumentParser()
ap.add_argument("--data-root", default=None)
a = ap.parse_args()
root = pc.find_root(a.data_root)
print("data root:", root)
fails = []
def check(ok, msg):
    print(("[PASS] " if ok else "[FAIL] ") + msg)
    if not ok:
        fails.append(msg)

print("\n[1] per-file consistency (dev + refit)")
files = [("train", s, None) for s in pc.SNRS_DEFAULT] + [("validation", s, None) for s in pc.SNRS_DEFAULT] \
      + [("refit", 2.25, d) for d in pc.DELTAS]
summ, ids = [], {}
for kind, s, dl in files:
    d = pc._load(pc._path(root, kind, s, dl)); tag = f"{kind} {s:.2f}" + (f" d{dl:+d}" if dl else "")
    n = len(d["frame_id"])
    need = ["frame_id", "snr", "log_r", "S", "min_abs", "mean_abs", "mean_ent", "info_err", "cur_err",
            "stored_r", "syndrome_weight", "final_failure"]
    check(all(k in d for k in need), f"{tag}: required keys")
    check(all(d[k].shape == (n, cl.T) for k in need[2:-1]) and d["final_failure"].shape == (n,), f"{tag}: shapes (n={n})")
    check(all(np.isfinite(d[k].astype(float)).all() for k in need[2:]), f"{tag}: finite")
    check(((d["info_err"] > 0).astype(int) == d["cur_err"].astype(int)).all(), f"{tag}: cur_err == (info_err>0)")
    check((d["cur_err"][:, -1].astype(int) == d["final_failure"].astype(int)).all(), f"{tag}: final_failure == cur_err[:,50]")
    ref = np.log(np.maximum(-np.expm1(-d["S"].astype(float)), 1e-300))
    ok = np.abs(d["log_r"] - ref) < 1e-6 * np.maximum(1, np.abs(ref)) + 1e-8
    check(ok.mean() > 0.999, f"{tag}: log_r agrees with log(-expm1(-S)) on {ok.mean():.4%} of cells")
    ent_ok = d["mean_ent"].astype(float) >= d["S"].astype(float) / cl.K - 1e-6
    check(ent_ok.mean() > 0.999, f"{tag}: mean_ent >= S/110 on {ent_ok.mean():.4%} of cells (consistent with info-bit entropy)")
    check(len(np.unique(d["frame_id"])) == n, f"{tag}: unique frame_id")
    ids[tag] = set(d["frame_id"].tolist())
    ff = d["final_failure"].astype(bool); syn50 = d["syndrome_weight"][:, -1]
    summ.append({"file": tag, "frames": n, "failing": int(ff.sum()), "bler": ff.mean(),
                 "P(syn50=0|correct)": float((syn50[~ff] == 0).mean()),
                 "undetected_errors(syn50=0|fail)": int((syn50[ff] == 0).sum()),
                 "log_r_min": float(d["log_r"].min()), "log_r_p50": float(np.median(d["log_r"])),
                 "id_min": int(d["frame_id"].min())})
print("\n[2] summary"); pd.set_option("display.width", 220)
print(pd.DataFrame(summ).round(5).to_string(index=False))

print("\n[3] leakage: train/validation frame_id overlap per SNR")
for s in pc.SNRS_DEFAULT:
    check(not (ids[f"train {s:.2f}"] & ids[f"validation {s:.2f}"]), f"snr {s:.2f}: no overlap")
for dl in pc.DELTAS[1:]:
    check(ids["refit 2.25 d%+d" % dl] == ids["refit 2.25 d%+d" % pc.DELTAS[0]], f"refit delta {dl:+d}: same frame ids as delta -3")

print("\n[4] decision-relevant population (validation): share of rows with syndrome > 0")
for s in pc.SNRS_DEFAULT:
    d = pc._load(pc._path(root, "validation", s))
    pos = d["syndrome_weight"] > 0
    print(f"  snr {s:.2f}: rows syn>0 = {pos.mean():6.1%};  wrong among them = {d['cur_err'][pos].mean():.3f};  "
          f"mean iteration of first syn==0 = {np.where((~pos).any(1), (~pos).argmax(1) + 1, cl.T).mean():.2f}")
p = root / "bler_summary.csv"
if p.exists():
    print("\n[5] bler_summary.csv (Member 1):"); print(pd.read_csv(p).head(40).to_string(index=False))
print("\nAUDIT:", "ALL PASSED" if not fails else f"{len(fails)} FAILED -> {fails}")
