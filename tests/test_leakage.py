
"""
Leakage test (Person 3).

Checks:
1. Train, validation and main-test frame IDs are unique.
2. Split codes are correct: 0=train, 1=validation, 2=test.
3. Train, validation and test IDs are disjoint.
4. Existing mismatch-test IDs are paired with main-test IDs.
5. Refit IDs use split code 3 and are consistent across delta files.
6. Refit IDs are disjoint from train, validation and test IDs.

Only frame_id arrays are read. No decoding outcomes are inspected.

Usage:
    python tests/test_leakage.py --data-root ..\math_dataset
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "src")
)

import p2_common as pc
from p3_data import split_file


# --------------------------------------------------
# Arguments and setup
# --------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--data-root", default=None)
args = parser.parse_args()

root = pc.find_root(args.data_root)
ok = True


def check(condition, message):
    global ok
    passed = bool(condition)
    ok = ok and passed
    print(("[PASS] " if passed else "[FAIL] ") + message)


def ids(path):
    """Read only frame IDs from an NPZ file."""
    with np.load(path) as data:
        return data["frame_id"]


# Store IDs for cross-split checks.
ALL = {}

# Track mismatch-file coverage.
mismatch_found = 0
mismatch_missing = 0


# --------------------------------------------------
# Main train / validation / test leakage checks
# --------------------------------------------------

for s in pc.SNRS_DEFAULT:

    tr = ids(
        split_file(root, "train", f"train_snr_{s:.2f}.npz")
    )

    va = ids(
        split_file(root, "validation", f"validation_snr_{s:.2f}.npz")
    )

    te = ids(
        split_file(root, "test", f"test_snr_{s:.2f}.npz")
    )

    # Uniqueness and split-code checks.
    for name, x, code in (
        ("train", tr, 0),
        ("validation", va, 1),
        ("test", te, 2),
    ):
        check(
            len(np.unique(x)) == len(x),
            f"{s:.2f} dB {name}: frame IDs unique"
        )

        check(
            bool((x // 10_000_000 == code).all()),
            f"{s:.2f} dB {name}: split code {code}"
        )

    # Cross-split disjointness.
    check(
        not (set(tr) & set(va))
        and not (set(tr) & set(te))
        and not (set(va) & set(te)),
        f"{s:.2f} dB: train/validation/test IDs disjoint"
    )

    ALL[s] = (set(tr), set(va), set(te))

    # --------------------------------------------------
    # Mismatch-test checks
    # --------------------------------------------------

    for dl in pc.DELTAS:

        filename = f"test_snr_{s:.2f}_delta{dl:+d}.npz"

        try:
            p = split_file(root, "test", filename)

        except FileNotFoundError:
            mismatch_missing += 1
            print(
                f"[INFO] {s:.2f} dB delta {dl:+d}: "
                "mismatch-test file not generated; skipped"
            )
            continue

        mismatch_found += 1
        mm = ids(p)

        check(
            set(mm) <= set(te),
            f"{s:.2f} dB delta {dl:+d}: "
            "mismatch-test IDs are a subset of main-test IDs "
            "(paired, not independent)"
        )

        check(
            not (set(mm) & (set(tr) | set(va))),
            f"{s:.2f} dB delta {dl:+d}: "
            "no overlap with train/validation"
        )


# --------------------------------------------------
# Refit leakage checks
# --------------------------------------------------

ref = []

for dl in pc.DELTAS:
    filename = f"train_mismatch_snr_2.25_delta{dl:+d}.npz"
    ref.append(ids(split_file(root, "refit", filename)))

check(
    all(bool((r // 10_000_000 == 3).all()) for r in ref),
    "refit files use split code 3"
)

check(
    all(np.array_equal(ref[0], r) for r in ref),
    "six refit files share identical frame IDs"
)

tr_all = set().union(*[v[0] for v in ALL.values()])
va_all = set().union(*[v[1] for v in ALL.values()])
te_all = set().union(*[v[2] for v in ALL.values()])

check(
    not (set(ref[0]) & (tr_all | va_all | te_all)),
    "refit IDs disjoint from train/validation/test"
)


# --------------------------------------------------
# Final report
# --------------------------------------------------

print("\nMismatch-test coverage:")
print(f"  Files checked: {mismatch_found}")
print(f"  Files not generated: {mismatch_missing}")

print(
    "\nNO LEAKAGE FOUND"
    if ok
    else "\nLEAKAGE / LAYOUT PROBLEM"
)

sys.exit(0 if ok else 1)
