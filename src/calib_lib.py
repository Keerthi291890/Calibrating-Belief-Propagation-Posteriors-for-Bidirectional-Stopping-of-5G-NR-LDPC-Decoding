"""
Person 2 library: stable posterior features, scalar-trace cache, metrics, calibrators.

Why this file exists
--------------------
1. The stored `residual_error` column was computed in float32. When every
   |LLR| >= ~20 it becomes exactly 0, so converged frames are indistinguishable.
   We recompute r_t from the stored LLRs in float64 using the log domain.
2. Reading 228-column CSVs again and again is slow. We extract SCALAR traces once
   (this is the "scalar traces only" contract of the project plan) and cache them.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

K = 110
T = 50
LOG_R_FLOOR = -60.0  # model-input guard only; documented in artifacts


# ----------------------------------------------------------------------
# Stable posterior features
# ----------------------------------------------------------------------
def stable_features(L: np.ndarray) -> dict:
    """
    L: (n, K) posterior LLRs, convention L = log P(b=0)/P(b=1).

    p_correct_i = max(p_i, 1-p_i) = sigmoid(|L_i|)
    -log p_correct_i = softplus(-|L_i|) = logaddexp(0, -|L_i|)      (stable)
    S   = -sum_i log p_correct_i          (>= 0)
    r   = 1 - prod p_correct = -expm1(-S)
    log r computed from r = -expm1(-S): exact for tiny S, no clipping needed.
    log(1 - r) = -S exactly.
    """
    L = np.asarray(L, dtype=np.float64)
    a = np.abs(L)
    sp = np.logaddexp(0.0, -a)
    S = sp.sum(axis=1)
    r = -np.expm1(-S)
    log_r = np.log(np.maximum(r, 1e-300))
    # binary entropy (nats) of the likely bit: H = softplus(-a) + a*sigmoid(-a)
    mean_ent = (sp + a * expit(-a)).mean(axis=1)
    return {
        "log_r": log_r,
        "S": S,
        "min_abs": a.min(axis=1),
        "mean_abs": a.mean(axis=1),
        "mean_ent": mean_ent,
    }


def naive_r(L: np.ndarray) -> np.ndarray:
    """Naive (float64) 1 - prod max(p,1-p). Only valid for moderate |L|; used in tests."""
    p1 = 1.0 / (1.0 + np.exp(L))
    return 1.0 - np.prod(np.maximum(p1, 1.0 - p1), axis=1)


def git_hash(repo_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


# ----------------------------------------------------------------------
# Generator-side helper: decoder callbacks -> scalar traces (our npz schema)
# ----------------------------------------------------------------------
def syndrome_weight(x_hat, H, neg_is_one=True) -> np.ndarray:
    """Per-frame number of violated parity checks. x_hat: (B, n_vn); H: scipy csr (m, n_vn)."""
    x = np.asarray(x_hat)
    hard = ((x < 0) if neg_is_one else (x > 0)).astype(np.int64)
    s = (H @ hard.T) % 2
    return np.asarray(s).sum(axis=0).astype(np.int32)


def trace_arrays(xhat_list, u, H=None, neg_is_one=True) -> dict:
    """
    xhat_list: list of T arrays (B, n_vn) for iterations 1..T (DROP the iteration-0 capture).
    u: (B, K) transmitted information bits.  H: pruned parity-check matrix matching n_vn (optional).
    Returns arrays (B, T) in the schema read by load_split().
    """
    u = np.asarray(u).astype(np.int64)
    B, nT = u.shape[0], len(xhat_list)
    keys = ["log_r", "S", "min_abs", "mean_abs", "mean_ent"]
    out = {k: np.zeros((B, nT)) for k in keys}
    info_err = np.zeros((B, nT), dtype=np.int32)
    syn = np.zeros((B, nT), dtype=np.int32) if H is not None else None
    for t, x in enumerate(xhat_list):
        x = np.asarray(x, dtype=np.float64)
        L = x[:, :K]
        f = stable_features(L)
        for k in keys:
            out[k][:, t] = f[k]
        bits = (L < 0) if neg_is_one else (L > 0)
        info_err[:, t] = (bits.astype(np.int64) != u).sum(axis=1)
        if H is not None:
            syn[:, t] = syndrome_weight(x, H, neg_is_one)
    out["info_err"] = info_err
    out["cur_err"] = (info_err > 0).astype(np.int8)
    out["stored_r"] = np.exp(out["log_r"])
    out["final_failure"] = (info_err[:, -1] > 0).astype(np.int8)
    if syn is not None:
        out["syndrome_weight"] = syn
    return out


# ----------------------------------------------------------------------
# CSV -> scalar trace cache
# ----------------------------------------------------------------------
def extract_csv(csv_path: Path, chunksize: int = 40_000) -> dict:
    """Stream one big CSV, keep scalar traces of shape (n_frames, T)."""
    csv_path = Path(csv_path)
    frame_ids = np.sort(pd.read_csv(csv_path, usecols=["frame_id"])["frame_id"].unique())
    nf = len(frame_ids)

    llr_cols = [f"llr_{i}" for i in range(K)]
    tb_cols = [f"true_bit_{i}" for i in range(K)]
    meta = ["frame_id", "iteration", "residual_error", "current_error", "final_failure"]

    out = {k: np.full((nf, T), np.nan) for k in
           ["log_r", "S", "min_abs", "mean_abs", "mean_ent", "stored_r", "info_err", "cur_err"]}
    final_failure = np.full(nf, -1, dtype=np.int8)
    filled = np.zeros((nf, T), dtype=np.int8)

    for chunk in pd.read_csv(csv_path, usecols=meta + llr_cols + tb_cols, chunksize=chunksize):
        fi = np.searchsorted(frame_ids, chunk["frame_id"].to_numpy())
        ti = chunk["iteration"].to_numpy().astype(int) - 1
        L = chunk[llr_cols].to_numpy(dtype=np.float64)
        tb = chunk[tb_cols].to_numpy(dtype=np.int8)
        feats = stable_features(L)
        for k, v in feats.items():
            out[k][fi, ti] = v
        out["stored_r"][fi, ti] = chunk["residual_error"].to_numpy()
        out["info_err"][fi, ti] = ((L < 0).astype(np.int8) != tb).sum(axis=1)
        out["cur_err"][fi, ti] = chunk["current_error"].to_numpy()
        final_failure[fi] = chunk["final_failure"].to_numpy().astype(np.int8)
        np.add.at(filled, (fi, ti), 1)

    if not (filled == 1).all():
        raise RuntimeError(f"{csv_path.name}: some (frame, iteration) cells missing or duplicated")
    if (final_failure < 0).any():
        raise RuntimeError(f"{csv_path.name}: missing final_failure")

    # consistency: label recomputed from LLR signs vs truth must equal stored label
    mism = int(((out["info_err"] > 0).astype(int) != out["cur_err"].astype(int)).sum())
    if mism:
        raise RuntimeError(f"{csv_path.name}: {mism} cells where recomputed current_error != stored")
    out["frame_id"] = frame_ids
    out["final_failure"] = final_failure
    return out


def cached_split(data_dir: Path, cache_dir: Path, split: str, snr: float, force=False) -> dict:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cpath = cache_dir / f"{split}_snr_{snr:.2f}.npz"
    if cpath.exists() and not force:
        z = np.load(cpath)
        return {k: z[k] for k in z.files}
    csv = Path(data_dir) / f"{split}_snr_{snr:.2f}dB.csv"
    print(f"  extracting {csv.name} (one-time, streams in chunks) ...")
    d = extract_csv(csv)
    np.savez_compressed(cpath, **d)
    return d


def load_split(data_dir, cache_dir, split, snrs, force=False) -> dict:
    parts = []
    for s in snrs:
        d = cached_split(data_dir, cache_dir, split, s, force)
        d["snr"] = np.full(len(d["frame_id"]), s)
        parts.append(d)
    return {k: np.concatenate([p[k] for p in parts], axis=0) for k in parts[0]}


# ----------------------------------------------------------------------
# Flat table with features
# ----------------------------------------------------------------------
def dlog(log_r: np.ndarray, k: int) -> np.ndarray:
    """Delta_k(t) = log r_t - log r_{max(t-k, 1)}  (defined for every t; =0 at t=1)."""
    idx = np.maximum(np.arange(T) - k, 0)
    return log_r - log_r[:, idx]


def make_table(d: dict, ks=(1, 2, 3, 5, 10)) -> pd.DataFrame:
    n = len(d["frame_id"])
    lr = np.maximum(d["log_r"], LOG_R_FLOOR)
    tab = {
        "frame": np.repeat(np.arange(n), T),
        "snr": np.repeat(d["snr"], T),
        "t": np.tile(np.arange(1, T + 1), n),
        "log_r": lr.ravel(),
        "S": d["S"].ravel(),
        "r_stable": np.exp(lr).ravel(),
        "stored_r": d["stored_r"].ravel(),
        "min_abs": d["min_abs"].ravel(),
        "mean_ent": d["mean_ent"].ravel(),
        "y_now": (d["info_err"] > 0).astype(np.int8).ravel(),
        "y_fail": np.repeat(d["final_failure"], T),
    }
    if "syndrome_weight" in d:
        tab["syn"] = d["syndrome_weight"].ravel()
    for k in ks:
        tab[f"dlog_k{k}"] = dlog(lr, k).ravel()
    return pd.DataFrame(tab)


# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------
def ece_equal_mass(y, p, n_bins=20) -> float:
    y = np.asarray(y, float); p = np.asarray(p, float)
    order = np.argsort(p, kind="stable")
    y, p = y[order], p[order]
    n = len(y)
    return float(sum(len(ix) / n * abs(y[ix].mean() - p[ix].mean())
                     for ix in np.array_split(np.arange(n), n_bins) if len(ix)))


def ece_fixed(y, p, n_bins=10) -> float:
    """Equal-width ECE exactly as in the first script (for comparability)."""
    y = np.asarray(y, float); p = np.asarray(p, float)
    b = np.minimum((p * n_bins).astype(int), n_bins - 1)
    n = len(y)
    return float(sum((b == i).sum() / n * abs(y[b == i].mean() - p[b == i].mean())
                     for i in range(n_bins) if (b == i).any()))


def logloss(y, p) -> float:
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log1p(-p)))


def score(y, p) -> dict:
    y = np.asarray(y); p = np.asarray(p, float)
    both = len(np.unique(y)) == 2
    return {
        "n": int(len(y)),
        "prevalence": float(y.mean()),
        "brier": float(np.mean((p - y) ** 2)),
        "logloss": logloss(y, p),
        "ece_eqmass20": ece_equal_mass(y, p),
        "ece_fixed10": ece_fixed(y, p),
        "auc": float(roc_auc_score(y, p)) if both else np.nan,
        "pr_auc": float(average_precision_score(y, p)) if both else np.nan,
    }


def reliability(y, p, n_bins=20) -> pd.DataFrame:
    y = np.asarray(y, float); p = np.asarray(p, float)
    order = np.argsort(p, kind="stable")
    rows = []
    for ix in np.array_split(order, n_bins):
        if len(ix):
            rows.append({"mean_pred": p[ix].mean(), "obs_rate": y[ix].mean(), "count": len(ix)})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Calibrators (JSON-serialisable; this is the artifact Person 3 consumes)
# ----------------------------------------------------------------------
class LogisticCal:
    """sigmoid(w . x + b), fitted unregularised on standardised inputs."""

    def __init__(self, features):
        self.features = list(features)

    def fit(self, df: pd.DataFrame, y):
        X = df[self.features].to_numpy(float)
        y = np.asarray(y)
        if len(np.unique(y)) < 2:                      # degenerate subset: constant probability
            self.w = np.zeros(X.shape[1]); pbar = float(np.clip(y.mean(), 1e-6, 1 - 1e-6))
            self.b = float(np.log(pbar / (1 - pbar)))
            return self
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-12
        m = LogisticRegression(C=1e6, max_iter=5000)
        m.fit((X - self.mu) / self.sd, y)
        self.w = m.coef_[0] / self.sd
        self.b = float(m.intercept_[0] - np.sum(m.coef_[0] * self.mu / self.sd))
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return expit(df[self.features].to_numpy(float) @ self.w + self.b)

    def to_dict(self, **meta):
        return {"method": "logistic", "features": self.features,
                "coefficients": [float(x) for x in self.w], "intercept": self.b,
                "log_r_floor": LOG_R_FLOOR, **meta}

    @staticmethod
    def from_dict(d):
        c = LogisticCal(d["features"])
        c.w = np.array(d["coefficients"]); c.b = d["intercept"]
        return c


class IsotonicCal:
    """Monotone map from log r to P(error). Ablation A2."""

    def __init__(self, feature="log_r"):
        self.feature = feature

    def fit(self, df, y):
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip")
        iso.fit(df[self.feature].to_numpy(float), y)
        self.x, self.y = iso.X_thresholds_, iso.y_thresholds_
        return self

    def predict(self, df):
        return np.interp(df[self.feature].to_numpy(float), self.x, self.y)

    def to_dict(self, **meta):
        return {"method": "isotonic", "feature": self.feature,
                "x_thresholds": self.x.tolist(), "y_thresholds": self.y.tolist(),
                "log_r_floor": LOG_R_FLOOR, **meta}

    @staticmethod
    def from_dict(d):
        c = IsotonicCal(d["feature"]); c.x = np.array(d["x_thresholds"]); c.y = np.array(d["y_thresholds"])
        return c


class IdentityRaw:
    """No calibration: q = raw r (used when raw is not beaten on validation)."""
    features = ["log_r"]

    def fit(self, df, y):
        return self

    def predict(self, df):
        return np.clip(df["r_stable"].to_numpy(float), 0.0, 1.0)

    def to_dict(self, **meta):
        return {"method": "identity_raw_r", "features": ["log_r"], **meta}

    @staticmethod
    def from_dict(d):
        return IdentityRaw()


def load_calibrator(path):
    d = json.load(open(path, encoding="utf-8"))
    cls = {"logistic": LogisticCal, "isotonic": IsotonicCal, "identity_raw_r": IdentityRaw}[d["method"]]
    return cls.from_dict(d)


def save_calibrator(cal, path, repo_root, fit_split, **meta):
    meta = {"fit_split": fit_split, "git_hash": git_hash(repo_root), **meta}
    json.dump(cal.to_dict(**meta), open(path, "w", encoding="utf-8"), indent=2)
