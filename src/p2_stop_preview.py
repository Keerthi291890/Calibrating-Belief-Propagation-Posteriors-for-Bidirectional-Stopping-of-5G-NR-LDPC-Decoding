"""
(1) TAIL calibration: calibration exactly where decisions are made (low q_now for accept, high q_fail for abandon).
(2) Stopping PREVIEW on VALIDATION frames only (Person 3 owns the official evaluation; test stays sealed).
    Thresholds are tuned on validation and reported on validation, so numbers are OPTIMISTIC (in-sample).

    python src/p2_stop_preview.py --data-root E:/math_dataset

Rule semantics (project plan): accept at the first t with syndrome = 0 or score <= eps_a; abandon at the first
t >= t_min with q_fail >= eps_b; accept wins ties; otherwise run to 50.  Never later than syndrome-zero.
Excess = false accept (accepted with errors, full 50-iteration run correct) + false abandon (abandoned, full run correct).
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import calib_lib as cl
import p2_common as pc
import p2_stopping as ps
from risk_model import RiskModel

ap = argparse.ArgumentParser()
ap.add_argument("--data-root", default=None)
ap.add_argument("--out-dir", default=str(pc.REPO / "results" / "person2_v3"))
ap.add_argument("--snrs", default="2.00,2.25,2.50,2.75")
a = ap.parse_args()
SNRS = [float(s) for s in a.snrs.split(",")]
OUT = Path(a.out_dir); root = pc.find_root(a.data_root)
pd.set_option("display.width", 230, "display.max_columns", 30)
rm = RiskModel(OUT); T = cl.T; TM = rm.t_min

tr_d = pc.load_group(root, "train", SNRS)
va_d = pc.load_group(root, "validation", SNRS)
tr, va = pc.table(tr_d, (rm.k,)), pc.table(va_d, (rm.k,))
nf = len(va_d["frame_id"])
pos_tr = tr["syn"].to_numpy() > 0
m2c = cl.LogisticCal(["log_r", "S"]).fit(tr[pos_tr], tr["y_now"].to_numpy()[pos_tr])    # simple comparator

res = rm.predict(va_d)
Q = {"raw r": np.exp(np.maximum(va_d["log_r"], cl.LOG_R_FLOOR)),
     "M2c (log r,S)": m2c.predict(va).reshape(nf, T),
     f"selected ({rm.card['q_now']['name'][:2]})": res["q_now"]}
QF = {"raw r": Q["raw r"], f"selected ({rm.card['q_fail']['name'][:2]})": res["q_fail"]}
syn = va_d["syndrome_weight"]; pos = syn > 0
ie = va_d["info_err"]; wrong = ie > 0
ff = va_d["final_failure"].astype(bool)
tt = np.broadcast_to(np.arange(1, T + 1), (nf, T))

# ------------------------------------------------------------------ 1. tail calibration
print("=" * 100 + "\n[1] TAIL calibration, validation, rows with syndrome>0\n" + "=" * 100)
def tail_table(Qd, y, mask, edges, title):
    rows = []
    for n, q in Qd.items():
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = mask & (q > lo) & (q <= hi)
            if m.sum() == 0:
                continue
            fr = np.unique(np.nonzero(m)[0]).size
            rows.append({"score": n, "bin": f"({lo:g},{hi:g}]", "rows": int(m.sum()), "frames": fr,
                         "mean_pred": q[m].mean(), "observed": y[m].mean(), "events": int(y[m].sum()),
                         "obs/pred": y[m].mean() / max(q[m].mean(), 1e-12)})
    print(f"\n{title}"); df = pd.DataFrame(rows); print(df.round(5).to_string(index=False)); return df
t1 = tail_table(Q, wrong, pos, [0, 1e-3, 1e-2, 0.05, 0.1, 0.3, 0.7, 1.0],
                "q_now: ACCEPT side. obs/pred > 1 means the score UNDER-states error risk (unsafe for accepting).")
mf = pos & (tt >= TM)
yfm = np.broadcast_to(ff[:, None], (nf, T))
t2 = tail_table(QF, yfm, mf, [0, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0],
                "q_fail: ABANDON side. obs/pred < 1 in high bins means the score OVER-states failure (false abandons).")
t1.to_csv(OUT / "tail_q_now_val.csv", index=False); t2.to_csv(OUT / "tail_q_fail_val.csv", index=False)

# ------------------------------------------------------------------ 2. stopping preview
print("\n" + "=" * 100 + "\n[2] Stopping PREVIEW on validation (in-sample thresholds; test sealed)\n" + "=" * 100)
ctx = ps.make_ctx(va_d, rm.k, TM, rm.q_now_cal, rm.q_fail_cal)
rows = []
for s in SNRS:
    idx = np.nonzero(va_d["snr"] == s)[0]
    res = ps.tune(ps.sub(ctx, idx))
    base = res[0.10]["B2 syndrome-zero"]["mean_it"]
    for rho, d in res.items():
        for rule, r in d.items():
            rows.append({"snr": s, "rho": rho, "rule": rule, "params": str(r["params"]), "budget_met": r["feasible"],
                         "failing_frames": int(ctx["ff"][idx].sum()), "budget_events": round(r["budget_events"], 1),
                         **{k: v for k, v in r.items() if k not in ("params", "feasible", "budget_events")},
                         "saving_vs_B2_pct": 100 * (1 - r["mean_it"] / base)})
R = pd.DataFrame(rows); R.to_csv(OUT / "stop_preview_val.csv", index=False)
cols = ["rule", "params", "budget_met", "mean_it", "p95_it", "p99_it", "false_accept", "false_abandon", "excess_events",
        "undetected_per_1000", "net_bler_change", "saving_vs_B2_pct"]
for s in SNRS:
    for rho in (0.10, 0.20):
        r = R[(R.snr == s) & (R.rho == rho)]
        print(f"\n--- {s:.2f} dB, rho={rho:.0%}: {r.failing_frames.iloc[0]} failing frames, budget {r.budget_events.iloc[0]} excess events ---")
        print(r[cols].round(3).to_string(index=False))
print("\nbudget_met=False means NO threshold of that rule met the budget: ignore its iteration numbers.")
print("In-sample (thresholds tuned on the same validation frames). 5% budget rows are in stop_preview_val.csv.")
