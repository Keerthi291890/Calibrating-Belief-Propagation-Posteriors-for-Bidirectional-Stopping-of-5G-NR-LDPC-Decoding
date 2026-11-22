"""
Cross-run reproducibility check (to be run by someone who did NOT write the module).

  1) python src/p2_fit_dev.py --data-root <root> --out-dir results/repro_check      (fresh fit, test never opened)
  2) python src/p2_repro_check.py --ref-dir results/person2_v3 --new-dir results/repro_check

Compares the regenerated calibrators with the frozen ones numerically (BLAS differences can change the last digits, so
exact file hashes are not required) and checks that the same models were selected.
"""
import argparse, json
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--ref-dir", required=True); ap.add_argument("--new-dir", required=True)
ap.add_argument("--rtol", type=float, default=1e-5)
a = ap.parse_args(); R, N = Path(a.ref_dir), Path(a.new_dir)
ok = True
def check(cond, msg):
    global ok; ok &= bool(cond); print(("[PASS] " if cond else "[FAIL] ") + msg)

for name in ("q_now.json", "q_fail.json"):
    r, n = json.load(open(R / name)), json.load(open(N / name))
    check(r["method"] == n["method"], f"{name}: method {r['method']}")
    check(r.get("features") == n.get("features") and r.get("feature") == n.get("feature"), f"{name}: same features")
    if r["method"] == "logistic":
        w_r, w_n = np.array(r["coefficients"] + [r["intercept"]]), np.array(n["coefficients"] + [n["intercept"]])
        err = np.max(np.abs(w_r - w_n) / np.maximum(np.abs(w_r), 1e-8))
        check(err < a.rtol, f"{name}: coefficients agree (max relative difference {err:.2e})")
    elif r["method"] == "isotonic":
        check(np.allclose(r["x_thresholds"], n["x_thresholds"], rtol=a.rtol) and np.allclose(r["y_thresholds"], n["y_thresholds"], rtol=a.rtol),
              f"{name}: isotonic thresholds agree")
    for k in ("k", "t_min", "fit_region", "model_name", "target"):
        if k in r: check(r[k] == n.get(k), f"{name}: {k} = {r[k]}")
sr, sn = json.load(open(R / "selection_summary.json")), json.load(open(N / "selection_summary.json"))
for k in ("selected_q_now", "selected_q_fail", "k"):
    check(sr[k] == sn[k], f"selection {k}: {sr[k]}")
print("\nREPRODUCIBLE" if ok else "\nNOT REPRODUCIBLE: investigate before submission")
