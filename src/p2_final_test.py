"""
FINAL evaluation on the sealed test files. Refuses to run unless FROZEN.json exists, the frozen files are unchanged,
and the test set has not been used before (override logged).

    python src/p2_final_test.py --data-root E:/math_dataset

Matched: main test per SNR (raw vs frozen calibrators).  Mismatch: for SNR in {2.00,2.25,2.50} and each delta, the
mismatch test file vs the SAME frames from the main test (paired delta=0 reference); at 2.25 dB additionally
REFIT calibrators fitted on the whole train_mismatch file of that delta (training data, not test).
"""
import argparse, hashlib, json, datetime
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
ap.add_argument("--n-boot", type=int, default=200)
ap.add_argument("--allow-rerun", action="store_true")
a = ap.parse_args()
OUT = Path(a.out_dir); root = pc.find_root(a.data_root)
fz_path = OUT / "FROZEN.json"
if not fz_path.exists():
    raise SystemExit("No FROZEN.json: run p2_freeze.py first.")
fz = json.load(open(fz_path, encoding="utf-8"))
h = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
for name, digest in fz["sha256"].items():
    p = (OUT / name) if name.endswith(".json") else (pc.REPO / "src" / name)
    if h(p) != digest:
        raise SystemExit(f"{name} changed since freeze. Models/code must not change after freezing.")
log = OUT / "TEST_ACCESS_LOG.txt"
if log.exists() and "RUN" in log.read_text() and not a.allow_rerun:
    raise SystemExit("Test set already used once (see TEST_ACCESS_LOG.txt). Use --allow-rerun only for bug fixes; it is logged.")
with open(log, "a") as f:
    f.write(f"RUN {datetime.datetime.utcnow().isoformat()}Z git={cl.git_hash(pc.REPO)} frozen={fz['frozen_utc']} rerun={a.allow_rerun}\n")
pc.unlock_test(fz_path)

rm = RiskModel(OUT); K, TM = rm.k, rm.t_min
raw = {"q_now": cl.IdentityRaw(), "q_fail": cl.IdentityRaw()}
matched = {"q_now": rm.q_now_cal, "q_fail": rm.q_fail_cal}
pd.set_option("display.width", 230, "display.max_columns", 30)
def subset(d, keep):
    return {k: v[keep] for k, v in d.items() if hasattr(v, "shape") and v.ndim >= 1 and len(v) == len(keep)}

print("\n[1] matched test, per SNR")
rows = []
mains = {}
for s in pc.SNRS_DEFAULT:
    d = pc.load_group(root, "test", [s]); mains[s] = d
    rows += pc.head_rows(d, K, TM, {"raw": raw, "matched": matched}, a.n_boot, {"snr": s})
mt = pd.DataFrame(rows); mt.to_csv(OUT / "final_matched_test.csv", index=False)
for head in ("q_now", "q_fail"):
    print(f"\n--- {head} ---")
    print(mt[mt["head"] == head][["snr", "model", "frames", "failing_frames", "region_rows", "prevalence", "brier", "ece", "ece_lo", "ece_hi",
                                  "logloss", "cal_slope", "cal_intercept", "auc", "pr_auc"]].round(5).to_string(index=False))

print("\n[2] mismatch test (paired with the same main-test frames)")
rows = []
for s in (2.00, 2.25, 2.50):
    for dl in pc.DELTAS:
        dm = pc.load_group(root, "test", [s], delta=dl)
        ids = set(dm["frame_id"].tolist())
        base = subset(mains[s], np.isin(mains[s]["frame_id"], list(ids)))
        models = {"raw": raw, "matched": matched}
        if s == 2.25:
            dr = pc.load_group(root, "refit", [2.25], delta=dl); tr_ = pc.table(dr, (K,)); refit = {}
            for head, ycol in (("q_now", "y_now"), ("q_fail", "y_fail")):
                reg = rm.card[head]["fit_region"]
                m = pc.fit_mask(tr_, head, "all" if reg in ("all", "none") else "syn>0", TM)
                refit[head] = pc.refit_like(matched[head], tr_[m], tr_.loc[m, ycol].to_numpy())
            models["refit"] = refit
        rows += pc.head_rows(dm, K, TM, models, a.n_boot, {"snr": s, "delta": dl})
        if dl == pc.DELTAS[0]:
            rows += pc.head_rows(base, K, TM, {"raw": raw, "matched": matched}, a.n_boot, {"snr": s, "delta": 0})
mm = pd.DataFrame(rows).sort_values(["head", "snr", "delta"]); mm.to_csv(OUT / "final_mismatch_test.csv", index=False)
for head in ("q_now", "q_fail"):
    for s in (2.00, 2.25, 2.50):
        r = mm[(mm["head"] == head) & (mm.snr == s)]
        print(f"\n--- {head}, {s:.2f} dB: ECE by delta (delta=0 is the paired matched reference) ---")
        print(r.pivot(index="delta", columns="model", values="ece").round(5).to_string())

def boot_stop(ctx, rule_params, n_boot, seed=0):
    """rule_params: {rule: params}. Returns rows with frame-bootstrap CIs; paired savings vs B2 and P-cal-vs-B5 diff."""
    nf = ctx["nf"]; rng = np.random.default_rng(seed); boots = [rng.integers(0, nf, nf) for _ in range(n_boot)]
    pfs = {r: ps.per_frame(ctx, *ps.RULES[r][1](ctx, p)) for r, p in rule_params.items()}
    b2 = pfs["B2 syndrome-zero"]["t_stop"]; b5 = pfs["B5 syndrome + cap"]["t_stop"]; pc_ = pfs["P-cal both"]["t_stop"]
    rows = {}
    for r, pf in pfs.items():
        t = pf["t_stop"]
        mi = np.array([t[ix].mean() for ix in boots])
        sv = np.array([100 * (1 - t[ix].mean() / b2[ix].mean()) for ix in boots])
        rows[r] = {"summary": ps.summarize(pf, ctx["ff"]), "mean_it_ci": (np.percentile(mi, 2.5), np.percentile(mi, 97.5)),
                   "saving_pct": 100 * (1 - t.mean() / b2.mean()), "saving_ci": (np.percentile(sv, 2.5), np.percentile(sv, 97.5))}
    d = np.array([b5[ix].mean() - pc_[ix].mean() for ix in boots])
    return rows, (b5.mean() - pc_.mean(), np.percentile(d, 2.5), np.percentile(d, 97.5))

print("\n[3] stopping on matched test (thresholds FROZEN from validation)")
rows = []
for s in pc.SNRS_DEFAULT:
    ctx = ps.make_ctx(mains[s], K, TM, rm.q_now_cal, rm.q_fail_cal)
    fails = int(ctx["ff"].sum())
    for rho in ps.RHOS:
        params = {r: fz["thresholds"][f"{s:.2f}"][f"{rho:.2f}"][r]["params"] for r in ps.RULES}
        feas = {r: fz["thresholds"][f"{s:.2f}"][f"{rho:.2f}"][r]["feasible"] for r in ps.RULES}
        res, dif = boot_stop(ctx, params, a.n_boot)
        for r, v in res.items():
            sm = v["summary"]
            rows.append({"snr": s, "rho": rho, "rule": r, "params": str(params[r]), "feasible_on_validation": feas[r],
                         "failing_frames": fails, "budget_events": rho * fails,
                         "budget_met_on_test": sm["excess_events"] <= rho * fails, **sm,
                         "mean_it_lo": v["mean_it_ci"][0], "mean_it_hi": v["mean_it_ci"][1],
                         "saving_vs_B2_pct": v["saving_pct"], "saving_lo": v["saving_ci"][0], "saving_hi": v["saving_ci"][1],
                         "pcal_both_faster_than_B5_by": dif[0], "diff_lo": dif[1], "diff_hi": dif[2]})
st = pd.DataFrame(rows); st.to_csv(OUT / "final_stopping_test.csv", index=False)
show = ["rule", "params", "feasible_on_validation", "budget_met_on_test", "mean_it", "mean_it_lo", "mean_it_hi", "false_accept", "false_abandon",
        "undetected_per_1000", "net_bler_change", "saving_vs_B2_pct", "saving_lo", "saving_hi"]
for s in pc.SNRS_DEFAULT:
    for rho in (0.10, 0.20):
        r = st[(st.snr == s) & (st.rho == rho)]
        print(f"\n--- test {s:.2f} dB, rho={rho:.0%}: {r.failing_frames.iloc[0]} failing frames, budget {r.budget_events.iloc[0]:.1f} events ---")
        print(r[show].round(3).to_string(index=False))
        print(f"    P-cal both faster than B5 by {r.pcal_both_faster_than_B5_by.iloc[0]:.3f} iterations "
              f"[{r.diff_lo.iloc[0]:.3f}, {r.diff_hi.iloc[0]:.3f}] (positive = P-cal better)")

print("\n[4] stopping reliability under mismatch (matched thresholds, rho=10%, excess counted vs the mismatch run's own full run)")
sel = ["B2 syndrome-zero", "B5 syndrome + cap", "P-raw both", "P-cal accept-only", "P-cal abandon-only", "P-cal both"]
rows = []
for s in (2.00, 2.25, 2.50):
    ctx_main = ps.make_ctx(mains[s], K, TM, rm.q_now_cal, rm.q_fail_cal)
    for dl in [0] + pc.DELTAS:
        if dl == 0:
            dm = pc.load_group(root, "test", [s], delta=pc.DELTAS[0])
            idx = np.nonzero(np.isin(mains[s]["frame_id"], dm["frame_id"]))[0]; ctx = ps.sub(ctx_main, idx); variants = {"matched": ctx}
        else:
            dm = pc.load_group(root, "test", [s], delta=dl); variants = {"matched": ps.make_ctx(dm, K, TM, rm.q_now_cal, rm.q_fail_cal)}
            if s == 2.25:
                dr = pc.load_group(root, "refit", [2.25], delta=dl); tr_ = pc.table(dr, (K,)); rf = {}
                for head, ycol in (("q_now", "y_now"), ("q_fail", "y_fail")):
                    reg = rm.card[head]["fit_region"]
                    mm_ = pc.fit_mask(tr_, head, "all" if reg in ("all", "none") else "syn>0", TM)
                    rf[head] = pc.refit_like({"q_now": rm.q_now_cal, "q_fail": rm.q_fail_cal}[head], tr_[mm_], tr_.loc[mm_, ycol].to_numpy())
                variants["refit"] = ps.make_ctx(dm, K, TM, rf["q_now"], rf["q_fail"])
        for vname, c in variants.items():
            fails = int(c["ff"].sum())
            base = ps.summarize(ps.per_frame(c, *ps.RULES["B2 syndrome-zero"][1](c, {})), c["ff"])["mean_it"]
            for r in sel:
                if vname == "refit" and not r.startswith("P-cal"):
                    continue
                p = fz["thresholds"][f"{s:.2f}"]["0.10"][r]["params"]
                sm = ps.summarize(ps.per_frame(c, *ps.RULES[r][1](c, p)), c["ff"])
                rows.append({"snr": s, "delta": dl, "calibrators": vname, "rule": r, "params": str(p), "failing_frames_at_delta": fails,
                             "excess_vs_budget": f"{sm['excess_events']}/{0.10 * fails:.1f}", "budget_met": sm["excess_events"] <= 0.10 * fails,
                             **sm, "saving_vs_B2_same_delta_pct": 100 * (1 - sm["mean_it"] / base)})
ms = pd.DataFrame(rows); ms.to_csv(OUT / "final_stopping_mismatch.csv", index=False)
for s in (2.00, 2.25, 2.50):
    r = ms[(ms.snr == s) & (ms.rule == "P-cal both")]
    print(f"\n--- {s:.2f} dB, P-cal both, frozen thresholds ---")
    print(r[["delta", "calibrators", "failing_frames_at_delta", "excess_vs_budget", "budget_met", "mean_it", "undetected_per_1000",
             "net_bler_change", "saving_vs_B2_same_delta_pct"]].round(3).to_string(index=False))
print("\nDone. Test set used. Files: final_matched_test.csv, final_mismatch_test.csv, final_stopping_test.csv, final_stopping_mismatch.csv, TEST_ACCESS_LOG.txt")
