"""Run with:  python tests/test_calib_lib.py     (or pytest)"""
import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import calib_lib as cl  # noqa: E402


def test_log_domain_matches_naive_on_moderate_values():
    rng = np.random.default_rng(0)
    L = rng.normal(0, 4, size=(500, 110))
    r_naive = cl.naive_r(L)
    f = cl.stable_features(L)
    r_stable = np.exp(f["log_r"])
    assert np.allclose(r_naive, r_stable, rtol=1e-9, atol=1e-12)
    p1 = 1.0 / (1.0 + np.exp(L))
    S_naive = -np.log(np.maximum(p1, 1.0 - p1)).sum(axis=1)   # sum of logs: safe at moderate L
    assert np.allclose(f["S"], S_naive, rtol=1e-9)


def test_log_domain_does_not_saturate():
    L = np.full((1, 110), 25.0)           # float32 formula gives exactly 0 here
    f = cl.stable_features(L)
    assert np.isfinite(f["log_r"][0]) and f["log_r"][0] < -15
    assert abs(np.exp(f["log_r"][0]) - 110 * np.exp(-25.0)) / (110 * np.exp(-25.0)) < 1e-6
    # and the old float32 recipe really does collapse:
    p1 = 1 / (1 + np.exp(L.astype(np.float32)))
    r32 = -np.expm1(np.log(np.maximum(p1, 1 - p1)).sum(1))
    assert r32[0] == 0.0


def test_entropy_matches_definition():
    L = np.array([[0.0, 1.0, -3.0, 8.0]])
    p = 1 / (1 + np.exp(-np.abs(L)))
    H = -(p * np.log(p) + (1 - p) * np.log(1 - p))
    assert np.allclose(cl.stable_features(L)["mean_ent"], H.mean())


def test_dlog_definition():
    lr = np.tile(-np.arange(cl.T, dtype=float), (2, 1))   # log r_t = -(t-1)
    d = cl.dlog(lr, 3)
    assert np.allclose(d[:, 0:3], 0.0 + (-np.arange(3)))  # clipped to t=1 reference
    assert np.allclose(d[:, 10], -3.0)


def test_calibrator_json_roundtrip(tmp_path=Path("/tmp")):
    import pandas as pd
    rng = np.random.default_rng(1)
    x = rng.normal(-5, 3, 4000)
    y = (rng.random(4000) < 1 / (1 + np.exp(-(0.8 * x + 1)))).astype(int)
    df = pd.DataFrame({"log_r": x, "S": rng.random(4000)})
    for cal in (cl.LogisticCal(["log_r", "S"]).fit(df, y), cl.IsotonicCal().fit(df, y)):
        p = tmp_path / "cal_test.json"
        cl.save_calibrator(cal, p, Path("."), "train")
        cal2 = cl.load_calibrator(p)
        assert np.allclose(cal.predict(df), cal2.predict(df))


def test_trace_arrays_and_syndrome():
    import scipy.sparse as sp
    rng = np.random.default_rng(5)
    n_vn, m = 150, 40
    H = sp.csr_matrix((rng.random((m, n_vn)) < 0.05).astype(np.int8))
    u = rng.integers(0, 2, (7, cl.K))
    # fake decoder: LLR<0 means bit 1; frame 0 decodes perfectly, others have errors fading out
    xs = []
    for t in range(1, cl.T + 1):
        x = rng.normal(0, 1, (7, n_vn)) * 0.1
        sign = np.where(u == 1, -1.0, 1.0) * (1.0 + 0 * u)
        x[:, :cl.K] = sign * (5 + t)
        if t < 10:
            x[1:, :3] *= -1                       # 3 wrong bits early for frames 1..6
        xs.append(x)
    tr = cl.trace_arrays(xs, u, H)
    assert tr["info_err"].shape == (7, cl.T) and tr["syndrome_weight"].shape == (7, cl.T)
    assert (tr["info_err"][0] == 0).all() and (tr["info_err"][1:, :9] == 3).all() and (tr["info_err"][1:, 9:] == 0).all()
    assert tr["final_failure"].sum() == 0
    ref = cl.syndrome_weight(xs[-1], H)
    assert (tr["syndrome_weight"][:, -1] == ref).all()
    f = cl.stable_features(xs[3][:, :cl.K])
    assert np.allclose(tr["log_r"][:, 3], f["log_r"])


# ---------------- tree-code exact-marginals test ----------------
def _bp_tree(prior_llr, checks, iters=10):
    """Flooding sum-product on a tree. LLR = log P(0)/P(1). Returns posterior LLRs."""
    n = len(prior_llr)
    v2c = {(v, c): prior_llr[v] for c, vs in enumerate(checks) for v in vs}
    c2v = {}
    for _ in range(iters):
        for c, vs in enumerate(checks):
            for v in vs:
                prod = np.prod([np.tanh(v2c[(u, c)] / 2) for u in vs if u != v])
                c2v[(c, v)] = 2 * np.arctanh(np.clip(prod, -1 + 1e-15, 1 - 1e-15))
        for c, vs in enumerate(checks):
            for v in vs:
                v2c[(v, c)] = prior_llr[v] + sum(c2v[(c2, v)] for c2, vs2 in enumerate(checks)
                                                 if c2 != c and v in vs2)
    return np.array([prior_llr[v] + sum(c2v[(c, v)] for c, vs in enumerate(checks) if v in vs)
                     for v in range(n)])


def _exact_posterior(prior_llr, checks):
    n = len(prior_llr)
    words, logw = [], []
    for w in itertools.product([0, 1], repeat=n):
        if all(sum(w[v] for v in vs) % 2 == 0 for vs in checks):
            words.append(w)
            logw.append(sum((-0.5 * prior_llr[v]) if b == 1 else (0.5 * prior_llr[v]) for v, b in enumerate(w)))
    words = np.array(words); logw = np.array(logw)
    post = np.exp(logw - logw.max()); post /= post.sum()
    return words, post


def test_tree_code_bp_equals_exact_marginals():
    checks = [(0, 1, 2), (2, 3, 4)]      # two checks sharing bit 2 -> cycle-free (tree)
    rng = np.random.default_rng(3)
    worst_gap = 0.0
    for _ in range(30):
        prior = rng.normal(0, 2.5, 5)
        L_bp = _bp_tree(prior, checks)
        words, post = _exact_posterior(prior, checks)
        p1_exact = (post[:, None] * words).sum(0)
        L_exact = np.log((1 - p1_exact) / p1_exact)
        assert np.allclose(L_bp, L_exact, atol=1e-8), (L_bp, L_exact)
        # Mean-field r_t vs exact P(hard decision vector is wrong)
        x_hat = (L_bp < 0).astype(int)
        p_vec = post[(words == x_hat).all(1)].sum() if ((words == x_hat).all(1)).any() else 0.0
        r = np.exp(cl.stable_features(L_bp[None])["log_r"][0])
        worst_gap = max(worst_gap, abs(r - (1 - p_vec)))
    print(f"  tree test: BP == exact marginals. Max |r_meanfield - exact joint error| = {worst_gap:.3f}")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("[PASS]", name)
