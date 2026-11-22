"""Shared helpers for the scalar-trace pipeline (Person 2, stage 2).

Data layout (from Member 1's handoff), under --data-root:
    dev/train_snr_{snr:.2f}.npz, dev/validation_snr_{snr:.2f}.npz
    sealed_test/test_snr_{snr:.2f}.npz, sealed_test/test_snr_{snr:.2f}_delta{+d}.npz
    sealed_test/train_mismatch_snr_2.25_delta{+d}.npz      <- TRAINING data for refit, not test
Test files can only be loaded after unlock_test() has verified a FROZEN.json.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.linear_model import LogisticRegression

import calib_lib as cl

REPO = Path(__file__).resolve().parents[1]
SNRS_DEFAULT = [2.00, 2.25, 2.50, 2.75]
DELTAS = [-3, -2, -1, 1, 2, 3]
_TEST_UNLOCKED = False


def find_root(arg=None) -> Path:
    cands = [Path(arg)] if arg else [REPO / "data" / "scalar_traces", REPO.parent / "math_dataset",
                                     REPO.parent / "scalar_traces", Path("E:/math_dataset")]
    for c in cands:
        if (c / "dev").exists():
            return c
    raise FileNotFoundError(f"scalar-trace root not found; tried {[str(c) for c in cands]}. Use --data-root.")


def unlock_test(frozen_json: Path):
    """Only p2_final_test.py calls this, after checking the freeze file."""
    global _TEST_UNLOCKED
    if not Path(frozen_json).exists():
        raise PermissionError("FROZEN.json missing: freeze first (p2_freeze.py). Test data stays sealed.")
    _TEST_UNLOCKED = True


def _load(path: Path) -> dict:
    z = np.load(path)
    return {k: z[k] for k in z.files}


def _path(root: Path, kind: str, snr: float, delta=None) -> Path:
    if kind in ("train", "validation"):
        return root / "dev" / f"{kind}_snr_{snr:.2f}.npz"
    if kind == "test":
        if not _TEST_UNLOCKED:
            raise PermissionError("Test data is sealed. Run p2_freeze.py then p2_final_test.py.")
        suf = "" if delta in (None, 0) else f"_delta{delta:+d}"
        return root / "sealed_test" / f"test_snr_{snr:.2f}{suf}.npz"
    if kind == "refit":
        return root / "sealed_test" / f"train_mismatch_snr_{snr:.2f}_delta{delta:+d}.npz"
    raise ValueError(kind)


def load_group(root: Path, kind: str, snrs, delta=None) -> dict:
    parts = []
    for s in snrs:
        d = _load(_path(root, kind, s, delta))
        d["snr"] = np.full(len(d["frame_id"]), float(s))        # authoritative per-frame SNR
        parts.append(d)
    keys = [k for k in parts[0] if parts[0][k].ndim >= 1 and len(parts[0][k]) == len(parts[0]["frame_id"])]
    return {k: np.concatenate([p[k] for p in parts], axis=0) for k in keys}


def table(d: dict, ks=(1, 2, 3, 5, 10)) -> pd.DataFrame:
    """Flat feature table (rows = frame x iteration, frame-major). Only online-available features are used as inputs."""
    tab = cl.make_table(d, ks)                    # log_r(floored), S, r_stable, min_abs, mean_ent, y_now, y_fail, syn, dlog_k*
    lsyn = np.log1p(d["syndrome_weight"].astype(float))
    tab["lsyn"] = lsyn.ravel()
    tab["mean_abs"] = d["mean_abs"].ravel()
    for k in ks:
        idx = np.maximum(np.arange(cl.T) - k, 0)
        tab[f"dlsyn_k{k}"] = (lsyn - lsyn[:, idx]).ravel()
    return tab


def cal_slope_intercept(y, p):
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return np.nan, np.nan
    z = logit(np.clip(np.asarray(p, float), 1e-9, 1 - 1e-9))[:, None]
    m = LogisticRegression(C=1e6, max_iter=2000).fit(z, y)
    return float(m.coef_[0, 0]), float(m.intercept_[0])


def score2(y, p) -> dict:
    sc = cl.score(y, p)
    sc["cal_slope"], sc["cal_intercept"] = cal_slope_intercept(y, p)
    return sc


def boot_arrays(P: dict, Y: np.ndarray, M: np.ndarray, n_boot=100, seed=0, ece_bins=20):
    """Frame-level bootstrap. P[name]: (nf,T) probabilities; Y: (nf,T) labels; M: (nf,T) region mask.
    Returns {name: {'brier','logloss','ece': array(n_boot)}}."""
    nf = Y.shape[0]
    rng = np.random.default_rng(seed)
    boots = [rng.integers(0, nf, nf) for _ in range(n_boot)]
    cnt = M.sum(1)
    out = {}
    for name, p in P.items():
        pc = np.clip(p, 1e-15, 1 - 1e-15)
        b_f = (((p - Y) ** 2) * M).sum(1)
        l_f = (-(Y * np.log(pc) + (1 - Y) * np.log1p(-pc)) * M).sum(1)
        r = {"brier": [], "logloss": [], "ece": []}
        for ix in boots:
            n = cnt[ix].sum()
            r["brier"].append(b_f[ix].sum() / n)
            r["logloss"].append(l_f[ix].sum() / n)
            m = M[ix]
            r["ece"].append(cl.ece_equal_mass(Y[ix][m], p[ix][m], ece_bins))
        out[name] = {k: np.array(v) for k, v in r.items()}
    return out


def pick_simplest(order, logloss_boot, point_logloss):
    """order: names from simplest to most complex. Pick the best-logloss model, then the simplest one whose
    paired-bootstrap logloss difference vs best is NOT significantly worse (2.5th percentile of diff <= 0)."""
    best = min(order, key=lambda n: point_logloss[n])
    for n in order:
        diff = logloss_boot[n] - logloss_boot[best]
        if n == best or np.percentile(diff, 2.5) <= 0:
            return n, best
    return best, best


# ----------------------------------------------------------------------
# Regions, refit, and per-condition evaluation (used by dev, mismatch and final-test scripts)
# ----------------------------------------------------------------------
def eval_mask(tab: pd.DataFrame, head: str, t_min: int) -> np.ndarray:
    """Decision-relevant evaluation region: rows where the stopping rule is still undecided (syndrome > 0);
    the abandon head additionally needs t >= t_min."""
    m = tab["syn"].to_numpy() > 0
    return m & (tab["t"].to_numpy() >= t_min) if head == "q_fail" else m


def fit_mask(tab: pd.DataFrame, head: str, fit_region: str, t_min: int) -> np.ndarray:
    m = np.ones(len(tab), bool) if fit_region == "all" else tab["syn"].to_numpy() > 0
    return m & (tab["t"].to_numpy() >= t_min) if head == "q_fail" else m


def refit_like(cal, df, y):
    if isinstance(cal, cl.LogisticCal):
        return cl.LogisticCal(cal.features).fit(df, y)
    if isinstance(cal, cl.IsotonicCal):
        return cl.IsotonicCal(cal.feature).fit(df, y)
    return cl.LogisticCal(["log_r", "S"]).fit(df, y)


def head_rows(d: dict, k: int, t_min: int, models: dict, n_boot=100, extra=None) -> list:
    """models: {name: {'q_now': cal, 'q_fail': cal}}. One result row per (head, model) on region syn>0.
    ECE intervals are percentile frame-bootstrap intervals; ECE is biased upward on resamples, so the point
    estimate can sit near/below the lower end. Compare models with paired differences, not by eye."""
    tab = table(d, (k,))
    nf = len(d["frame_id"]); T = cl.T
    ff = d["final_failure"].astype(int)
    rows = []
    for head, ycol in (("q_now", "y_now"), ("q_fail", "y_fail")):
        y = tab[ycol].to_numpy(); mask = eval_mask(tab, head, t_min)
        P = {n: m[head].predict(tab) for n, m in models.items()}
        B = boot_arrays({n: p.reshape(nf, T) for n, p in P.items()}, y.reshape(nf, T), mask.reshape(nf, T), n_boot)
        for n, p in P.items():
            sc = score2(y[mask], p[mask])
            b = B[n]
            row = {"head": head, "model": n, "frames": nf, "failing_frames": int(ff.sum()), "bler": float(ff.mean()),
                   "region_rows": int(mask.sum()), "prevalence": sc["prevalence"],
                   "brier": sc["brier"], "brier_lo": np.percentile(b["brier"], 2.5), "brier_hi": np.percentile(b["brier"], 97.5),
                   "ece": sc["ece_eqmass20"], "ece_lo": np.percentile(b["ece"], 2.5), "ece_hi": np.percentile(b["ece"], 97.5),
                   "logloss": sc["logloss"], "cal_slope": sc["cal_slope"], "cal_intercept": sc["cal_intercept"],
                   "auc": sc["auc"], "pr_auc": sc["pr_auc"]}
            row.update(extra or {})
            rows.append(row)
    return rows


def find_n_edges(root: Path, arg=None) -> int:
    """nnz of the decoder's pruned parity-check matrix (150 x 350). From --n-edges or generation_config.json."""
    if arg:
        return int(arg)
    cfg = Path(root) / "generation_config.json"
    if cfg.exists():
        def walk(o):
            if isinstance(o, dict):
                for k, v in o.items():
                    if isinstance(v, (int, float)) and ("nnz" in k.lower() or "n_edges" in k.lower() or k.lower() == "edges"):
                        yield int(v)
                    yield from walk(v)
            elif isinstance(o, list):
                for v in o:
                    yield from walk(v)
        found = list(walk(json.load(open(cfg, encoding="utf-8"))))
        if found:
            return found[0]
    raise SystemExit("Need the number of edges of the decoder graph (nnz of decoder._pcm, shape 150x350). "
                     "Ask Person 1 to run: python -c \"print(dec._pcm.nnz)\" in the generation environment, "
                     "then pass --n-edges N.")
