"""python tests/test_stopping.py   -- properties of p2_stopping that Person 3 asked to be guaranteed."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import calib_lib as cl
import p2_stopping as ps

T = cl.T
class FakeRM:
    k = 2; t_min = 5
    q_now_cal = cl.LogisticCal(["log_r", "S", "lsyn", "min_abs"])
    q_fail_cal = cl.LogisticCal(["log_r", "dlog_k2", "t", "lsyn", "min_abs", "dlsyn_k2"])

def ctx(nf=400, seed=0):
    rng = np.random.default_rng(seed)
    ff = rng.random(nf) < 0.1
    conv = rng.integers(3, 15, nf)                                  # iteration at which a good frame converges
    tt = np.broadcast_to(np.arange(1, T + 1), (nf, T)).copy()
    pos = tt < conv[:, None]; pos[ff] = True                        # failing frames never reach syndrome 0
    wrong = pos.copy(); wrong[:, :] |= ff[:, None]
    qn = np.where(pos, rng.random((nf, T)), 0.0); qf = np.where(ff[:, None], 0.9, rng.random((nf, T)) * 0.5)
    return {"nf": nf, "snr": np.full(nf, 2.25), "tmin": 5, "pos": pos, "wrong": wrong, "ff": ff, "tt": tt, "qn": qn, "qf": qf,
            "raw": qn, "min_abs": rng.random((nf, T)) * 20, "mean_ent": rng.random((nf, T))}

def pf_for(c, rule, p):
    return ps.per_frame(c, *ps.RULES[rule][1](c, p))

def test_disabled_triggers_equal_b2():
    c = ctx(); b2 = pf_for(c, "B2 syndrome-zero", {})["t_stop"]
    assert (pf_for(c, "P-cal both", {"eps_a": ps.EA_OFF, "eps_b": ps.EB_OFF})["t_stop"] == b2).all()
    assert (pf_for(c, "B5 syndrome + cap", {"cap": ps.CAP_OFF})["t_stop"] == b2).all()
    assert (pf_for(c, "B3 min|LLR| accept", {"tau": ps.TAU_HI_OFF})["t_stop"] == b2).all()
    assert (pf_for(c, "A1 mean-entropy accept", {"tau": ps.TAU_LO_OFF})["t_stop"] == b2).all()

def test_accept_wins_ties():
    c = ctx(10)
    ta = np.full(10, 7); tb = np.full(10, 7)
    pf = ps.per_frame(c, ta, tb)
    assert pf["accepted"].all() and not pf["abandoned"].any() and (pf["t_stop"] == 7).all()
    pf = ps.per_frame(c, np.full(10, 8), np.full(10, 7))
    assert pf["abandoned"].all() and (pf["t_stop"] == 7).all()

def test_never_later_than_syndrome_zero():
    c = ctx(); b2 = pf_for(c, "B2 syndrome-zero", {})["t_stop"]
    for rule, p in (("P-cal accept-only", {"eps_a": 0.2}), ("P-cal both", {"eps_a": 0.1, "eps_b": 0.8}), ("P-raw accept-only", {"eps_a": 0.3})):
        assert (pf_for(c, rule, p)["t_stop"] <= b2).all()

def test_tuning_deterministic_and_feasibility_flag():
    c = ctx()
    r1, r2 = ps.tune(c), ps.tune(c)
    assert all(r1[rho][k]["params"] == r2[rho][k]["params"] for rho in r1 for k in r1[rho])
    for rho in r1:
        for rule, r in r1[rho].items():
            if r["feasible"]:
                assert r["excess_events"] <= r["budget_events"] + 1e-9
    assert all(r1[0.05][k]["fragile_budget"] for k in r1[0.05]) == (0.05 * c["ff"].sum() < ps.FRAGILE_BUDGET_EVENTS)

def test_shares_and_conditional_rates():
    c = ctx(); s = ps.summarize(pf_for(c, "P-cal both", {"eps_a": 0.2, "eps_b": 0.8}), c["ff"])
    assert abs(s["accept_share"] + s["nack_rate"] + s["cap_share"] - 1) < 1e-12
    assert s["undetected_given_accepted"] >= s["false_accept_given_accepted"] >= 0

def test_operation_counts():
    c = ctx(); cm = ps.make_cost_model(FakeRM, n_edges=700)
    def ops(rule, p):
        return ps.ops_summary(c, pf_for(c, rule, p), rule, p, cm)
    b1, b2 = ops("B1 fixed 50", {}), ops("B2 syndrome-zero", {})
    assert b1["bp_ops"] == 50 * cm["bp_iter"] and b1["syndrome_ops"] == 0 and b1["score_ops"] == 0
    assert b2["score_ops"] == 0 and b2["syndrome_ops"] > 0
    off = ops("P-cal both", {"eps_a": ps.EA_OFF, "eps_b": ps.EB_OFF})
    assert abs(off["total_ops"] - b2["total_ops"]) < 1e-9                    # disabled triggers cost nothing extra
    on = ops("P-cal both", {"eps_a": 0.2, "eps_b": 0.8})
    assert on["score_ops"] > 0 and on["bp_ops"] <= b2["bp_ops"]

def _one_frame(pos_until, qn_t, qf_t):
    """Single deterministic frame: syndrome>0 for t <= pos_until; qn/qf given as {t: value}."""
    c = ctx(1)
    c["pos"] = (c["tt"] <= pos_until); c["wrong"] = c["pos"].copy(); c["ff"] = np.array([False])
    c["qn"] = np.full((1, T), 0.9); c["qf"] = np.full((1, T), 0.1)
    for t, v in qn_t.items(): c["qn"][0, t - 1] = v
    for t, v in qf_t.items(): c["qf"][0, t - 1] = v
    return c

def test_sequential_schedule_both_heads():
    cm = ps.make_cost_model(FakeRM, n_edges=700)
    c = _one_frame(pos_until=8, qn_t={8: 0.05}, qf_t={})
    p = {"eps_a": 0.1, "eps_b": 0.9}
    pf = pf_for(c, "P-cal both", p)
    assert pf["t_stop"][0] == 8 and pf["accepted"][0]
    _, _, score = ps.ops_arrays(c, pf, "P-cal both", p, cm)
    # t=1..4: accept head only; t=5,6,7: both heads; t=8: accept fires -> abandon head NOT evaluated; history is free
    # because the accept head already computes log r and log(1+syndrome)
    assert cm["hist_after_now"] == 0
    assert score[0] == 5 * cm["now"] + 3 * cm["both"]

def test_abandon_only_pays_history_before_tmin():
    cm = ps.make_cost_model(FakeRM, n_edges=700)
    c = _one_frame(pos_until=30, qn_t={}, qf_t={6: 0.95})
    p = {"eps_b": 0.9}
    pf = pf_for(c, "P-cal abandon-only", p)
    assert pf["t_stop"][0] == 6 and pf["abandoned"][0]
    _, _, score = ps.ops_arrays(c, pf, "P-cal abandon-only", p, cm)
    assert cm["hist_alone"] > 0
    # history computed just in time at t = t_min-k .. t_min-1 = 3, 4 ; abandon head evaluated at t = 5, 6 only
    assert score[0] == 2 * cm["hist_alone"] + 2 * cm["fail"]

def test_bp_derivation_split():
    for cx, total in ((3, 15), (10, 29), (25, 59)):
        assert ps.VN_EDGE_OPS == 2 and ps.cn_edge_ops(cx) == 2 * cx + 7 and ps.bp_edge_ops(cx) == 2 * cx + 9 == total
    assert ps.make_cost_model(FakeRM, n_edges=1100, cx=10)["bp_iter"] == 31900          # matches the real preview run

def test_fragile_flag_matches_budget_definition():
    # validation failing frames per SNR (real): 305, 270, 209, 153 -> only 5% at 2.75 dB is below 10 events
    fails = {2.00: 305, 2.25: 270, 2.50: 209, 2.75: 153}
    flagged = [(s, r) for s, f in fails.items() for r in ps.RHOS if r * f < ps.FRAGILE_BUDGET_EVENTS]
    assert flagged == [(2.75, 0.05)]

def test_cost_model_consistent_sensitivity():
    cms = {x: ps.make_cost_model(FakeRM, n_edges=1100, cx=x) for x in ps.CX_SENSITIVITY}
    assert [cms[x]["bp_edge_ops"] for x in (3, 10, 25)] == [15, 29, 59]            # 2cx + 9
    assert cms[3]["now"] < cms[10]["now"] < cms[25]["now"]                         # the SCORE cost scales with cx as well
    assert cms[10]["syn"] == 350 + 1100 + 150

def test_logistic_head_cost_includes_bias_and_sigmoid():
    cm = ps.make_cost_model(FakeRM, n_edges=700, cx=10)
    prims_now = ps._prim_cost(ps._feature_prims(FakeRM.q_now_cal.features), 10)
    n = len(FakeRM.q_now_cal.features)
    assert cm["now"] == prims_now + 2 * n + 21 + 1      # dot product (n mul + n add incl. bias) + sigmoid(21) + threshold compare

if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"):
            f(); print("[PASS]", n)
