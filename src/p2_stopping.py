"""
Stopping rules, metrics and threshold tuning (shared by preview, freeze and the final test).

Semantics (project plan): accept at the first t with syndrome = 0 or score <= eps_a; abandon at the first t >= t_min
with q_fail >= eps_b; accept wins ties; otherwise run to 50 ('cap').  Never later than syndrome-zero.
Excess = false accept (accepted with errors, full 50-iteration run correct) + false abandon (abandoned, full run correct).
Tuning: for each rule, among its parameter grid pick the fewest mean iterations with excess_events <= rho * failing_frames.
If NO parameter meets the budget the rule is reported with feasible=False (never silently as if valid).
Person 3 owns this module for review; it is hashed at freeze time.
"""
import numpy as np
import calib_lib as cl
import p2_common as pc

T = cl.T
EA = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.2]
EB = [0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999]
RHOS = (0.05, 0.10, 0.20)


def first_t(mask):
    return np.where(mask.any(1), mask.argmax(1) + 1, T + 1)


def make_ctx(d, k, tmin, qn_cal, qf_cal):
    nf = len(d["frame_id"])
    tab = pc.table(d, (k,))
    return {"nf": nf, "snr": d["snr"], "tmin": tmin,
            "pos": d["syndrome_weight"] > 0, "wrong": d["info_err"] > 0, "ff": d["final_failure"].astype(bool),
            "tt": np.broadcast_to(np.arange(1, T + 1), (nf, T)).copy(),
            "qn": qn_cal.predict(tab).reshape(nf, T), "qf": qf_cal.predict(tab).reshape(nf, T),
            "raw": np.exp(np.maximum(d["log_r"], cl.LOG_R_FLOOR)),
            "min_abs": d["min_abs"], "mean_ent": d["mean_ent"]}


def sub(ctx, idx):
    out = {}
    for k, v in ctx.items():
        out[k] = v[idx] if isinstance(v, np.ndarray) and v.ndim >= 1 and len(v) == ctx["nf"] else v
    out["nf"] = len(idx)
    return out


def _never(c): return np.full(c["nf"], T + 1)
def _syn0(c): return first_t(~c["pos"])
def _acc(c, score, e, le=True):
    m = (score <= e) if le else (score >= e)
    return first_t(~c["pos"] | (c["pos"] & m))
def _ab(c, score, e):
    return first_t(c["pos"] & (c["tt"] >= c["tmin"]) & (score >= e))

RULES = {   # name -> (parameter grid, function(ctx, params) -> (t_acc, t_ab))
    "B1 fixed 50": ([{}], lambda c, p: (_never(c), _never(c))),
    "B2 syndrome-zero": ([{}], lambda c, p: (_syn0(c), _never(c))),
    "B4 ideal info-CRC (ceiling)": ([{}], lambda c, p: (first_t(~c["wrong"]), _never(c))),
    "B3 min|LLR| accept": ([{"tau": t} for t in (0.5, 1, 2, 3, 5, 8, 12, 16, 19)],
                           lambda c, p: (_acc(c, c["min_abs"], p["tau"], le=False), _never(c))),
    "B5 syndrome + cap": ([{"cap": x} for x in (6, 8, 10, 12, 15, 20, 25, 30, 35, 40, 45)],
                          lambda c, p: (_syn0(c), first_t(c["pos"] & (c["tt"] == p["cap"])))),
    "A1 mean-entropy accept": ([{"tau": t} for t in (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 3e-2, 0.1)],
                               lambda c, p: (_acc(c, c["mean_ent"], p["tau"]), _never(c))),
    "P-raw accept-only": ([{"eps_a": e} for e in EA], lambda c, p: (_acc(c, c["raw"], p["eps_a"]), _never(c))),
    "P-raw abandon-only": ([{"eps_b": e} for e in EB], lambda c, p: (_syn0(c), _ab(c, c["raw"], p["eps_b"]))),
    "P-raw both": ([{"eps_a": a, "eps_b": b} for a in EA for b in EB],
                   lambda c, p: (_acc(c, c["raw"], p["eps_a"]), _ab(c, c["raw"], p["eps_b"]))),
    "P-cal accept-only": ([{"eps_a": e} for e in EA], lambda c, p: (_acc(c, c["qn"], p["eps_a"]), _never(c))),
    "P-cal abandon-only": ([{"eps_b": e} for e in EB], lambda c, p: (_syn0(c), _ab(c, c["qf"], p["eps_b"]))),
    "P-cal both": ([{"eps_a": a, "eps_b": b} for a in EA for b in EB],
                   lambda c, p: (_acc(c, c["qn"], p["eps_a"]), _ab(c, c["qf"], p["eps_b"]))),
}


def per_frame(c, t_acc, t_ab):
    accepted = (t_acc <= t_ab) & (t_acc <= T)
    abandoned = (t_ab < t_acc) & (t_ab <= T)
    t_stop = np.where(accepted, t_acc, np.where(abandoned, t_ab, T))
    ia = np.clip(t_acc - 1, 0, T - 1)
    wrong_stop = accepted & c["wrong"][np.arange(c["nf"]), ia]
    fa = wrong_stop & ~c["ff"]
    fab = abandoned & ~c["ff"]
    rule_fail = wrong_stop | abandoned | (~accepted & ~abandoned & c["ff"])
    return {"t_stop": t_stop, "fa": fa, "fab": fab, "wrong_stop": wrong_stop, "rule_fail": rule_fail,
            "abandoned": abandoned}


def summarize(pf, ff):
    t = pf["t_stop"]
    return {"mean_it": float(t.mean()), "median_it": float(np.median(t)), "p95_it": float(np.percentile(t, 95)),
            "p99_it": float(np.percentile(t, 99)), "false_accept": int(pf["fa"].sum()), "false_abandon": int(pf["fab"].sum()),
            "excess_events": int(pf["fa"].sum() + pf["fab"].sum()),
            "undetected_per_1000": float(1000 * pf["wrong_stop"].mean()),
            "net_bler_change": float(pf["rule_fail"].mean() - ff.mean()), "abandon_share": float(pf["abandoned"].mean())}


def candidates(ctx):
    """All (rule, params, summary) for a context (one SNR)."""
    out = []
    for rule, (grid, fn) in RULES.items():
        for p in grid:
            ta, tb = fn(ctx, p)
            out.append((rule, p, summarize(per_frame(ctx, ta, tb), ctx["ff"])))
    return out


def tune(ctx, rhos=RHOS):
    """Returns {rho: {rule: {'params','feasible','budget_events', **summary}}} (selection on this context)."""
    cand = candidates(ctx)
    fails = int(ctx["ff"].sum())
    res = {}
    for rho in rhos:
        budget = rho * fails
        res[rho] = {}
        for rule in RULES:
            cs = [(p, s) for r, p, s in cand if r == rule]
            ok = [(p, s) for p, s in cs if s["excess_events"] <= budget]
            feasible = bool(ok)
            p, s = min(ok or [min(cs, key=lambda x: x[1]["excess_events"])], key=lambda x: x[1]["mean_it"] if feasible else x[1]["excess_events"])
            res[rho][rule] = {"params": p, "feasible": feasible, "budget_events": budget, **s}
    return res
