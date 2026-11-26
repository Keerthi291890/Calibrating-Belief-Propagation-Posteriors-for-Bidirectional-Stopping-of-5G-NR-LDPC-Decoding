"""Person 3 data-layout adapter. Does not modify source NPZ files or Person 2 code.

Training/validation traces may be stored either in root/dev/ (handoff layout)
or directly in root/ (current local dataset layout). Sealed-test access is
handled separately by the leakage audit, which reads frame_id arrays only.
"""
from pathlib import Path
import numpy as np

def split_file(root: Path, split: str, filename: str) -> Path:
    root = Path(root)
    if split in ("train", "validation"):
        candidates = [root / "dev" / filename, root / filename]
    elif split in ("test", "refit"):
        candidates = [root / "sealed_test" / filename]
    else:
        raise ValueError(f"Unsupported split: {split}")
    existing = [p for p in candidates if p.is_file()]
    if not existing:
        raise FileNotFoundError(f"Could not find {filename}; checked: {[str(p) for p in candidates]}")
    # Prefer documented dev/ location if both layouts happen to exist.
    return existing[0]

def load_group(root: Path, kind: str, snrs, delta=None) -> dict:
    parts = []
    for s in snrs:
        if kind not in ("train", "validation"):
            raise PermissionError("Person 3 E1/E5 adapter only loads train/validation traces")
        name = f"{kind}_snr_{s:.2f}.npz"
        path = split_file(root, kind, name)
        with np.load(path, allow_pickle=False) as z:
            d = {k: z[k] for k in z.files}
        if "frame_id" not in d:
            raise KeyError(f"frame_id missing from {path}")
        d["snr"] = np.full(len(d["frame_id"]), float(s))
        parts.append(d)
    keys = [k for k in parts[0] if getattr(parts[0][k], "ndim", 0) >= 1 and len(parts[0][k]) == len(parts[0]["frame_id"])]
    return {k: np.concatenate([p[k] for p in parts], axis=0) for k in keys}
