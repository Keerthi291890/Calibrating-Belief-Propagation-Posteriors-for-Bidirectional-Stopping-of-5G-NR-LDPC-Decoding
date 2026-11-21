"""FAKE scalar-trace dataset in Member 1's schema (code smoke tests only; numbers are meaningless)."""
import sys, json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import calib_lib as cl
K, T = cl.K, cl.T
root = Path(sys.argv[1]); (root / "dev").mkdir(parents=True, exist_ok=True); (root / "sealed_test").mkdir(exist_ok=True)
SIZES = {"train": 600, "validation": 250, "test": 300}
PFAIL = {2.0: .10, 2.25: .06, 2.5: .035, 2.75: .02}
CODE = {"train": 0, "validation": 1, "test": 2, "refit": 3}

def make(snr, n, code, first=0, scale=1.0, seed=0, delta=None):
    rng = np.random.default_rng(seed)
    tb = rng.integers(0, 2, (n, K)); bad = rng.random(n) < PFAIL[snr] * (1.5 if scale < 1 else 1.0) * (0.8 if scale > 1 else 1)
    tau = rng.exponential(6.0 / (snr - 1.4), (n, K)); stuck = (rng.random((n, K)) < .03) & bad[:, None]
    tau[stuck] = np.inf; tau[bad, 0] = np.inf
    u = rng.uniform(.5, 1.5, (n, K)); xs = []
    for t in range(1, T + 1):
        wrong = t < tau
        L = (1 - 2 * tb) * (1 - 2 * wrong) * np.minimum((0.3 + 0.7 * t * u) * scale, 20.0)
        x = np.zeros((n, 350)); x[:, :K] = L
        xs.append(x)
    tr = cl.trace_arrays(xs, tb, None)
    ie = tr["info_err"]
    und = rng.random(ie.shape) < 0.03
    tt = np.arange(1, T + 1)[None, :]
    leftover = (rng.random(ie.shape) < 0.35 * np.exp(-tt / 12.0)) * rng.integers(1, 4, ie.shape)   # info right, parity checks not yet
    tr["syndrome_weight"] = np.where(ie > 0, np.where(und, 0, np.minimum(1 + 2 * ie, 30)), leftover).astype(np.int32)
    ids = code * 10_000_000 + first + np.arange(n)
    tr.update(frame_id=ids, snr=np.full(n, snr), seed_key=ids // 1000)
    if delta is not None:
        tr.update(delta=np.array(delta), assumed_sigma2=np.array(0.5 * 10 ** (delta / 10)))
    return tr

rows = []
for snr in PFAIL:
    for split, n in SIZES.items():
        d = make(snr, n, CODE[split], seed=int(snr * 100) + CODE[split])
        sub = "dev" if split != "test" else "sealed_test"
        np.savez_compressed(root / sub / f"{split}_snr_{snr:.2f}.npz", **d)
        rows.append((split, snr, n, int(d["final_failure"].sum())))
    if snr != 2.75:
        n_mm = {2.0: 100, 2.25: 150, 2.5: 250}[snr]
        for dl in (-3, -2, -1, 1, 2, 3):
            d = make(snr, n_mm, 2, seed=int(snr * 100) + 2, scale=10 ** (-dl / 10), delta=dl)   # same first n frames' bits via same seed
            np.savez_compressed(root / "sealed_test" / f"test_snr_{snr:.2f}_delta{dl:+d}.npz", **d)
for dl in (-3, -2, -1, 1, 2, 3):
    d = make(2.25, 400, 3, seed=999, scale=10 ** (-dl / 10), delta=dl)
    np.savez_compressed(root / "sealed_test" / f"train_mismatch_snr_2.25_delta{dl:+d}.npz", **d)
print("mock npz written to", root, "files:", len(list(root.rglob('*.npz'))))
