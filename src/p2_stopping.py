"""
Stopping rules, metrics, operation counts and threshold tuning (shared by preview, freeze and the final test).

Semantics (project plan): accept at the first t with syndrome = 0 or score <= eps_a; abandon at the first t >= t_min
with q_fail >= eps_b; accept wins ties (t_acc <= t_ab); otherwise run to 50 ('cap').  Never later than syndrome-zero.
Excess = false accept (accepted with errors, full 50-iteration run correct) + false abandon (abandoned, full run correct).
Abandonment is a NACK and counts as a failure in net BLER.

Tuning: for each rule pick the parameters with the fewest mean iterations subject to excess_events <= rho * failing_frames.
Deterministic tie-break: fewer mean iterations, then fewer excess events, then the parameter string.
Every grid contains a DISABLED trigger (accept: eps_a = -1; abandon: eps_b = 2; cap = 999; tau disabled), so every
rule can fall back to syndrome-zero. A rule is marked feasible=False only if even its best configuration exceeds the budget.

Features B3 (min|LLR|) and A1 (mean entropy) are computed over the 110 INFORMATION bits (same as the labels).

Operation counts: see the block comment above `prim_ops` (unit, BP/score/syndrome formulas, schedule, history).
"""
import numpy as np
import calib_lib as cl
import p2_common as pc

T = cl.T
EA = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.2, 0.3, 0.5]
EB = [0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999]
EA_OFF, EB_OFF, CAP_OFF, TAU_HI_OFF, TAU_LO_OFF = -1.0, 2.0, 999, 1e9, -1.0
RHOS = (0.05, 0.10, 0.20)
FRAGILE_BUDGET_EVENTS = 10          # budgets below this many excess events are flagged statistically fragile


def first_t(mask):
    return np.where(mask.any(1), mask.argmax(1) + 1, T + 1)


def make_ctx(d, k, tmin, qn_cal, qf_cal):
    nf = len(d["frame_id"])
    tab = pc.table(d, (k,))
    logr = np.clip(d["log_r"], cl.LOG_R_FLOOR, 0.0)          # r <= 1 by construction; guard against rounding
    return {"nf": nf, "snr": d["snr"], "tmin": tmin,
            "pos": d["syndrome_weight"] > 0, "wrong": d["info_err"] > 0, "ff": d["final_failure"].astype(bool),
            "tt": np.broadcast_to(np.arange(1, T + 1), (nf, T)).copy(),
            "qn": qn_cal.predict(tab).reshape(nf, T), "qf": qf_cal.predict(tab).reshape(nf, T),
            "raw": np.exp(logr), "min_abs": d["min_abs"], "mean_ent": d["mean_ent"]}


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

_EA = [EA_OFF] + EA
_EB = EB + [EB_OFF]
RULES = {   # name -> (parameter grid, function(ctx, params) -> (t_acc, t_ab))
    "B1 fixed 50": ([{}], lambda c, p: (_never(c), _never(c))),
    "B2 syndrome-zero": ([{}], lambda c, p: (_syn0(c), _never(c))),
    "B4 ideal info-CRC (ceiling)": ([{}], lambda c, p: (first_t(~c["wrong"]), _never(c))),
    "B3 min|LLR| accept": ([{"tau": t} for t in (0.5, 1, 2, 3, 5, 8, 12, 16, 19, TAU_HI_OFF)],
                           lambda c, p: (_acc(c, c["min_abs"], p["tau"], le=False), _never(c))),
    "B5 syndrome + cap": ([{"cap": x} for x in (6, 8, 10, 12, 15, 20, 25, 30, 35, 40, 45, 48, CAP_OFF)],
                          lambda c, p: (_syn0(c), first_t(c["pos"] & (c["tt"] == p["cap"])))),
    "A1 mean-entropy accept": ([{"tau": t} for t in (TAU_LO_OFF, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 3e-2, 0.1)],
                               lambda c, p: (_acc(c, c["mean_ent"], p["tau"]), _never(c))),
    "P-raw accept-only": ([{"eps_a": e} for e in _EA], lambda c, p: (_acc(c, c["raw"], p["eps_a"]), _never(c))),
    "P-raw abandon-only": ([{"eps_b": e} for e in _EB], lambda c, p: (_syn0(c), _ab(c, c["raw"], p["eps_b"]))),
    "P-raw both": ([{"eps_a": a, "eps_b": b} for a in _EA for b in _EB],
                   lambda c, p: (_acc(c, c["raw"], p["eps_a"]), _ab(c, c["raw"], p["eps_b"]))),
    "P-cal accept-only": ([{"eps_a": e} for e in _EA], lambda c, p: (_acc(c, c["qn"], p["eps_a"]), _never(c))),
    "P-cal abandon-only": ([{"eps_b": e} for e in _EB], lambda c, p: (_syn0(c), _ab(c, c["qf"], p["eps_b"]))),
    "P-cal both": ([{"eps_a": a, "eps_b": b} for a in _EA for b in _EB],
                   lambda c, p: (_acc(c, c["qn"], p["eps_a"]), _ab(c, c["qf"], p["eps_b"]))),
}


def disabled_note(rule, p):
    s = []
    if "eps_a" in p and p["eps_a"] < 0: s.append("accept trigger disabled")
    if "eps_b" in p and p["eps_b"] > 1: s.append("abandon trigger disabled")
    if "cap" in p and p["cap"] > T: s.append("cap disabled")
    if "tau" in p and (p["tau"] >= TAU_HI_OFF or (p["tau"] < 0 and rule.startswith("A1"))): s.append("accept trigger disabled")
    return "; ".join(s)


# ----------------------------------------------------------------------------------------------- per-frame outcomes
def per_frame(c, t_acc, t_ab):
    accepted = (t_acc <= t_ab) & (t_acc <= T)
    abandoned = (t_ab < t_acc) & (t_ab <= T)
    t_stop = np.where(accepted, t_acc, np.where(abandoned, t_ab, T))
    ia = np.clip(t_acc - 1, 0, T - 1)
    wrong_stop = accepted & c["wrong"][np.arange(c["nf"]), ia]
    fa = wrong_stop & ~c["ff"]
    fab = abandoned & ~c["ff"]
    capped = ~accepted & ~abandoned
    rule_fail = wrong_stop | abandoned | (capped & c["ff"])
    return {"t_stop": t_stop, "fa": fa, "fab": fab, "wrong_stop": wrong_stop, "rule_fail": rule_fail,
            "abandoned": abandoned, "accepted": accepted, "capped": capped}


def summarize(pf, ff):
    t = pf["t_stop"]; n_acc = int(pf["accepted"].sum())
    return {"mean_it": float(t.mean()), "median_it": float(np.median(t)), "p95_it": float(np.percentile(t, 95)),
            "p99_it": float(np.percentile(t, 99)), "false_accept": int(pf["fa"].sum()), "false_abandon": int(pf["fab"].sum()),
            "excess_events": int(pf["fa"].sum() + pf["fab"].sum()),
            "undetected_per_1000": float(1000 * pf["wrong_stop"].mean()),
            "undetected_given_accepted": float(pf["wrong_stop"].sum() / n_acc) if n_acc else float("nan"),
            "false_accept_given_accepted": float(pf["fa"].sum() / n_acc) if n_acc else float("nan"),
            "accept_share": float(pf["accepted"].mean()), "nack_rate": float(pf["abandoned"].mean()),
            "cap_share": float(pf["capped"].mean()),
            "net_bler_change": float(pf["rule_fail"].mean() - ff.mean())}


# ----------------------------------------------------------------------------------------------- operation counts
# Unit: scalar operations. add/sub/mul/compare/abs/clip-compare = 1; every transcendental (exp, log, log1p, expm1,
# tanh, atanh) and every division = cx (default 10). cx is varied CONSISTENTLY in BP and in the score (sensitivity 3/10/25).
#   BP per edge per iteration (flooding sum-product, tanh domain), TOTAL = VN + CN = 2 + (2cx + 7) = 2cx + 9:
#     variable node (VN_EDGE_OPS) = 2: accumulate + subtract own message;
#     check node (cn_edge_ops)    = (cx+1) + 3 + (cx+1) + 2 = 2cx + 7:
#         tanh(m/2) = 1 + cx; product of the other inputs by prefix/suffix products = 3; atanh + doubling = cx + 1; clips = 2.
#   cx = 3 (LUT/fast approximation) -> 15,  cx = 10 -> 29 (default),  cx = 25 (slow library call) -> 59.
#   Syndrome check per iteration = n_vn hard decisions + n_edges parity XORs + n_checks zero tests.
# Score schedule (sequential, accept has precedence), per iteration t, in this order:
#   1) syndrome check (every rule except B1/B4); syndrome 0 -> accept, nothing else is evaluated;
#   2) accept head (q_now / raw r / min|LLR| / entropy) at every syndrome-positive iteration;
#   3) abandon head only if no accept fired AND t >= t_min.
# History: the abandon head's change features need log r (and log(1+syndrome)) at iteration t-k; these are computed just in
# time at the k iterations t_min-k .. t_min-1 and charged there (nothing earlier). If the accept head already computes them
# they are free. Features are computed over the 110 information bits (stable_features implementation: |L| is used by the
# entropy formula softplus(-|L|) + |L| sigmoid(-|L|), hence mean_ent depends on abs).
# Logistic head = n_features multiplies + n_features adds (the last add is the bias) + sigmoid (cx + 1 + cx) + 1 compare.
# No run-time normalisation: LogisticCal stores weights in RAW feature units (standardisation folded in).
CX_DEFAULT = 10
CX_SENSITIVITY = (3, 10, 25)
VN_EDGE_OPS = 2


def cn_edge_ops(cx):
    return (cx + 1) + 3 + (cx + 1) + 2            # = 2cx + 7


def bp_edge_ops(cx):
    return VN_EDGE_OPS + cn_edge_ops(cx)          # = 2cx + 9  ->  15 / 29 / 59 for cx = 3 / 10 / 25


def prim_ops(cx):
    return {"abs": 110, "S": 110 * (2 * cx + 2), "raw_r": cx + 1, "log_r": 2 * cx + 3,    # expm1 + neg + log (+2 clip)
            "min_abs": 109, "mean_ent": 110 * (3 * cx + 5) + cx, "mean_abs": 111, "lsyn": cx + 1, "dlog": 1, "dlsyn": 1}
DEPS = {"S": ["abs"], "raw_r": ["S"], "log_r": ["S"], "min_abs": ["abs"], "mean_ent": ["abs"], "mean_abs": ["abs"],
        "dlog": ["log_r"], "dlsyn": ["lsyn"]}


def _closure(prims):
    out, stack = set(), list(prims)
    while stack:
        p = stack.pop()
        if p not in out:
            out.add(p); stack.extend(DEPS.get(p, []))
    return out


def _prim_cost(prims, cx):
    po = prim_ops(cx)
    return sum(po[p] for p in _closure(prims))


def _feature_prims(features):
    out = set()
    for f in features:
        if f in ("log_r", "S", "min_abs", "mean_abs", "lsyn"): out.add(f)
        elif f.startswith("dlog_k"): out.add("dlog")
        elif f.startswith("dlsyn_k"): out.add("dlsyn")
        elif f == "t": pass
        else: raise ValueError(f"no cost defined for feature {f}")
    return out


def _head(cal, cx):
    if isinstance(cal, cl.IdentityRaw):
        return {"raw_r"}, 0
    if isinstance(cal, cl.LogisticCal):
        return _feature_prims(cal.features), 2 * len(cal.features) + (2 * cx + 1)
    if isinstance(cal, cl.IsotonicCal):
        return {"log_r"}, 20
    raise ValueError(type(cal))


def make_cost_model(rm, n_edges, n_checks=150, n_vn=350, cx=CX_DEFAULT):
    pn, en = _head(rm.q_now_cal, cx); pf, ef = _head(rm.q_fail_cal, cx)
    hist = set()
    if isinstance(rm.q_fail_cal, cl.LogisticCal):
        if any(f.startswith("dlog_k") for f in rm.q_fail_cal.features): hist.add("log_r")
        if any(f.startswith("dlsyn_k") for f in rm.q_fail_cal.features): hist.add("lsyn")
    bp_edge = bp_edge_ops(cx)
    return {"n_edges": int(n_edges), "n_vn": int(n_vn), "n_checks": int(n_checks), "cx": cx, "bp_edge_ops": bp_edge,
            "bp_iter": n_edges * bp_edge, "syn": n_vn + n_edges + n_checks, "k": int(rm.k), "tmin": int(rm.t_min),
            "now": _prim_cost(pn, cx) + en + 1, "fail": _prim_cost(pf, cx) + ef + 1,
            "both": _prim_cost(pn | pf, cx) + en + ef + 2,
            "hist_alone": _prim_cost(hist, cx) if hist else 0,
            "hist_after_now": (_prim_cost(hist | pn, cx) - _prim_cost(pn, cx)) if hist else 0,
            "raw": _prim_cost({"raw_r"}, cx) + 1, "minabs": _prim_cost({"min_abs"}, cx) + 1,
            "ent": _prim_cost({"mean_ent"}, cx) + 1}


def _spec(rule, p, cm):
    """dict: uses_syn, c_before (t<t_min), c_after (t>=t_min, nothing fired), hist (per evaluated iteration in the history
    window), seq_save (cost not spent at the final iteration when the accept head fires at t>=t_min)."""
    z = {"uses_syn": True, "c_before": 0, "c_after": 0, "hist": 0, "seq_save": 0}
    if rule.startswith("B1") or rule.startswith("B4"):
        return {**z, "uses_syn": False}
    if rule.startswith("B2"):
        return z
    if rule.startswith("B3"):
        return z if p["tau"] >= TAU_HI_OFF else {**z, "c_before": cm["minabs"], "c_after": cm["minabs"]}
    if rule.startswith("A1"):
        return z if p["tau"] < 0 else {**z, "c_before": cm["ent"], "c_after": cm["ent"]}
    if rule.startswith("B5"):
        return z if p["cap"] > T else {**z, "c_before": 1, "c_after": 1}
    raw = rule.startswith("P-raw")
    a_on = ("eps_a" in p) and p["eps_a"] >= 0
    b_on = ("eps_b" in p) and p["eps_b"] <= 1
    ca = cm["raw"] if raw else cm["now"]
    cb = cm["raw"] if raw else cm["fail"]
    cboth = cm["raw"] + 1 if raw else cm["both"]
    if a_on and b_on:
        return {**z, "c_before": ca, "c_after": cboth, "hist": 0 if raw else cm["hist_after_now"], "seq_save": cboth - ca}
    if a_on:
        return {**z, "c_before": ca, "c_after": ca}
    if b_on:
        return {**z, "c_before": 0, "c_after": cb, "hist": 0 if raw else cm["hist_alone"]}
    return z


def ops_arrays(ctx, pf, rule, p, cm):
    """Per-frame operation counts of the ONLINE process (iterations executed until the stop)."""
    t = pf["t_stop"]; n = ctx["nf"]; idx = t - 1; ar = np.arange(n)
    sp = _spec(rule, p, cm)
    tmin = ctx["tmin"]
    after = ctx["pos"] & (ctx["tt"] >= tmin)
    window = ctx["pos"] & (ctx["tt"] >= tmin - cm["k"]) & (ctx["tt"] < tmin)
    n_pos = np.cumsum(ctx["pos"], 1)[ar, idx]; n_after = np.cumsum(after, 1)[ar, idx]
    n_win = np.cumsum(window, 1)[ar, idx]
    bp = t * cm["bp_iter"]
    syn = t * cm["syn"] if sp["uses_syn"] else np.zeros(n)
    score = (n_pos - n_after) * sp["c_before"] + n_after * sp["c_after"] + n_win * sp["hist"]
    by_score = pf["accepted"] & ctx["pos"][ar, idx] & (t >= tmin)          # accept head fired: abandon head skipped
    score = score - by_score * sp["seq_save"]
    return bp.astype(float), syn.astype(float), score.astype(float)


def ops_summary(ctx, pf, rule, p, cm):
    bp, syn, sc = ops_arrays(ctx, pf, rule, p, cm)
    return {"bp_ops": float(bp.mean()), "syndrome_ops": float(syn.mean()), "score_ops": float(sc.mean()),
            "total_ops": float((bp + syn + sc).mean())}


# ----------------------------------------------------------------------------------------------- tuning
def candidates(ctx):
    out = []
    for rule, (grid, fn) in RULES.items():
        for p in grid:
            ta, tb = fn(ctx, p)
            out.append((rule, p, summarize(per_frame(ctx, ta, tb), ctx["ff"])))
    return out


def tune(ctx, rhos=RHOS, cm=None):
    """{rho: {rule: {'params','feasible','budget_events','fragile_budget','disabled', **summary, **ops}}} on this context."""
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
            pool = ok if feasible else [min(cs, key=lambda x: (x[1]["excess_events"], str(sorted(x[0].items()))))]
            p, s = min(pool, key=lambda x: (x[1]["mean_it"], x[1]["excess_events"], str(sorted(x[0].items()))))
            row = {"params": p, "feasible": feasible, "budget_events": budget,
                   "fragile_budget": budget < FRAGILE_BUDGET_EVENTS, "disabled": disabled_note(rule, p), **s}
            if cm is not None:
                pf = per_frame(ctx, *RULES[rule][1](ctx, p))
                row.update(ops_summary(ctx, pf, rule, p, cm))
            res[rho][rule] = row
    return res
