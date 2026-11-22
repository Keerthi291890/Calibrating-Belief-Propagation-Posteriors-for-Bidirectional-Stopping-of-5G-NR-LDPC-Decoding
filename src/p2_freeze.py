"""Freeze models, code AND stopping thresholds BEFORE the test files are opened.
Thresholds are tuned on VALIDATION only, per SNR and budget rho in {5,10,20}%, and stored in FROZEN.json.

    python src/p2_freeze.py --data-root E:/math_dataset --i-have-reviewed-validation
"""
import argparse, hashlib, json, datetime
from pathlib import Path
import numpy as np
import calib_lib as cl
import p2_common as pc
import p2_stopping as ps
from risk_model import RiskModel

ap = argparse.ArgumentParser()
ap.add_argument("--data-root", default=None)
ap.add_argument("--out-dir", default=str(pc.REPO / "results" / "person2_v3"))
ap.add_argument("--snrs", default="2.00,2.25,2.50,2.75")
ap.add_argument("--n-edges", default=None)
ap.add_argument("--n-vn", type=int, default=350)
ap.add_argument("--n-checks", type=int, default=150)
ap.add_argument("--cx", type=int, default=ps.CX_DEFAULT)
ap.add_argument("--i-have-reviewed-validation", action="store_true")
a = ap.parse_args()
if not a.i_have_reviewed_validation:
    raise SystemExit("Refusing: review validation results first, then pass --i-have-reviewed-validation.")
OUT = Path(a.out_dir); root = pc.find_root(a.data_root); SNRS = [float(s) for s in a.snrs.split(",")]
rm = RiskModel(OUT)
cost_params = {"n_edges": pc.find_n_edges(root, a.n_edges), "n_checks": a.n_checks, "n_vn": a.n_vn, "cx": a.cx}
va = pc.load_group(root, "validation", SNRS)
ctx = ps.make_ctx(va, rm.k, rm.t_min, rm.q_now_cal, rm.q_fail_cal)
thr = {}
for s in SNRS:
    res = ps.tune(ps.sub(ctx, np.nonzero(va["snr"] == s)[0]))
    thr[f"{s:.2f}"] = {f"{rho:.2f}": {rule: {"params": r["params"], "feasible": r["feasible"], "fragile_budget": r["fragile_budget"],
                                             "val_mean_it": r["mean_it"], "val_excess": r["excess_events"]}
                                      for rule, r in d.items()} for rho, d in res.items()}
h = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
files = [OUT / n for n in ("q_now.json", "q_fail.json", "model_card.json")] + \
        [pc.REPO / "src" / n for n in ("calib_lib.py", "p2_common.py", "risk_model.py", "p2_stopping.py",
                                      "p2_fit_dev.py", "p2_final_test.py")]
fz = {"frozen_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "git_hash": cl.git_hash(pc.REPO),
      "sha256": {p.name: h(p) for p in files}, "thresholds": thr, "cost_model": cost_params,
      "selection": json.load(open(OUT / "selection_summary.json", encoding="utf-8"))}
json.dump(fz, open(OUT / "FROZEN.json", "w", encoding="utf-8"), indent=2)
print("FROZEN.json written (models, code hashes, validation-tuned thresholds). Test may now be opened ONCE.")
print(json.dumps(fz["selection"], indent=2))
for s in thr:
    print(s, "rho=0.10 P-cal both:", thr[s]["0.10"]["P-cal both"], "| B5:", thr[s]["0.10"]["B5 syndrome + cap"]["params"])
