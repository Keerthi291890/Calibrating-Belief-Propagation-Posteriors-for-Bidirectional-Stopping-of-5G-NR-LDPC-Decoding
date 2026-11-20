
from pathlib import Path
import csv
import json
import numpy as np

# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path("../data/scalar_traces")
DEV = ROOT / "dev"
SEALED = ROOT / "sealed_test"

SNRS = [2.00, 2.25, 2.50, 2.75]
DELTAS = [-3, -2, -1, 1, 2, 3]

EXPECTED_COUNTS = {
    "train": {
        2.00: 10_000,
        2.25: 15_000,
        2.50: 20_000,
        2.75: 30_000,
    },
    "validation": {
        2.00: 5_000,
        2.25: 8_000,
        2.50: 15_000,
        2.75: 25_000,
    },
    "test": {
        2.00: 10_000,
        2.25: 20_000,
        2.50: 40_000,
        2.75: 50_000,
    },
    "mismatch": {
        2.00: 5_000,
        2.25: 10_000,
        2.50: 20_000,
    },
}

TRACE_FIELDS = {
    "log_r", "S", "min_abs", "mean_abs", "mean_ent",
    "info_err", "cur_err", "stored_r", "syndrome_weight",
}

BASE_FIELDS = TRACE_FIELDS | {
    "frame_id", "snr", "seed_key", "final_failure"
}

TOTAL_FILES = 36

# ============================================================
# HELPERS
# ============================================================

passed_checks = 0
failed_checks = 0


def report(condition, message):
    global passed_checks, failed_checks

    if condition:
        passed_checks += 1
        print(f"[PASS] {message}")
    else:
        failed_checks += 1
        print(f"[FAIL] {message}")


def verify_npz(path, expected_count, expected_snr=None,
               expected_delta=None, expected_split=None):

    if not path.exists():
        report(False, f"Missing file: {path.name}")
        return None

    try:
        with np.load(path) as data:

            missing = BASE_FIELDS - set(data.files)

            if missing:
                report(False, f"{path.name}: missing fields {missing}")
                return None

            n = len(data["frame_id"])

            count_ok = n == expected_count
            report(count_ok, f"{path.name}: frame count {n}/{expected_count}")

            shapes_ok = all(
                data[key].shape == (n, 50)
                for key in TRACE_FIELDS
            )
            report(shapes_ok, f"{path.name}: trace shapes")

            metadata_ok = all(
                data[key].shape == (n,)
                for key in ["frame_id", "snr", "seed_key", "final_failure"]
            )
            report(metadata_ok, f"{path.name}: metadata shapes")

            unique_ids = len(np.unique(data["frame_id"])) == n
            report(unique_ids, f"{path.name}: unique frame IDs")

            if expected_snr is not None:
                snr_ok = np.allclose(data["snr"], expected_snr)
                report(snr_ok, f"{path.name}: SNR metadata")

            if expected_delta is not None:
                delta_ok = (
                    "delta" in data.files
                    and float(data["delta"].item()) == expected_delta
                )
                report(delta_ok, f"{path.name}: mismatch delta")

                sigma_ok = "assumed_sigma2" in data.files
                if sigma_ok:
                    sigma2_true = 1.0 / (
                        2 * 0.5 * (10 ** (expected_snr / 10))
                    )
                    expected_sigma2 = (
                        sigma2_true * 10 ** (expected_delta / 10)
                    )
                    sigma_ok = np.isclose(
                        float(data["assumed_sigma2"].item()),
                        expected_sigma2
                    )

                report(sigma_ok, f"{path.name}: assumed noise variance")

            if expected_split is not None:
                lower = expected_split * 10_000_000
                upper = (expected_split + 1) * 10_000_000

                split_ok = np.all(
                    (data["frame_id"] >= lower)
                    & (data["frame_id"] < upper)
                )
                report(split_ok, f"{path.name}: split-code frame IDs")

            return data["frame_id"].copy()

    except Exception as exc:
        report(False, f"{path.name}: could not read ({exc})")
        return None


# ============================================================
# 1. VERIFY ALL EXPECTED MAIN DATA FILES
# ============================================================

print("\n" + "=" * 70)
print("FINAL DATASET AUDIT")
print("=" * 70)

all_ids = {}
expected_files = []

print("\n1. DEVELOPMENT DATA")
print("-" * 70)

for split, split_code in [("train", 0), ("validation", 1)]:
    for snr, count in EXPECTED_COUNTS[split].items():

        filename = f"{split}_snr_{snr:.2f}.npz"
        path = DEV / filename
        expected_files.append(path)

        ids = verify_npz(
            path, count, expected_snr=snr,
            expected_split=split_code
        )

        if ids is not None:
            all_ids[(split, snr)] = ids

print("\n2. MAIN TEST DATA")
print("-" * 70)

for snr, count in EXPECTED_COUNTS["test"].items():

    filename = f"test_snr_{snr:.2f}.npz"
    path = SEALED / filename
    expected_files.append(path)

    ids = verify_npz(
        path, count, expected_snr=snr, expected_split=2
    )

    if ids is not None:
        all_ids[("test", snr)] = ids

print("\n3. MISMATCH TEST DATA")
print("-" * 70)

for snr, count in EXPECTED_COUNTS["mismatch"].items():
    main_ids = all_ids.get(("test", snr))

    for delta in DELTAS:

        filename = f"test_snr_{snr:.2f}_delta{delta:+d}.npz"
        path = SEALED / filename
        expected_files.append(path)

        ids = verify_npz(
            path, count,
            expected_snr=snr,
            expected_delta=delta
        )

        if ids is not None and main_ids is not None:
            alignment_ok = np.array_equal(ids, main_ids[:count])
            report(alignment_ok, f"{filename}: matches main test frame IDs")

print("\n4. MISMATCH REFIT DATA")
print("-" * 70)

for delta in DELTAS:

    filename = f"train_mismatch_snr_2.25_delta{delta:+d}.npz"
    path = SEALED / filename
    expected_files.append(path)

    verify_npz(
        path, 10_000,
        expected_snr=2.25,
        expected_delta=delta,
        expected_split=3
    )


# ============================================================
# 5. CHECK EXPECTED FILE INVENTORY
# ============================================================

print("\n5. FILE INVENTORY")
print("-" * 70)

existing_expected = sum(path.exists() for path in expected_files)

report(
    existing_expected == TOTAL_FILES,
    f"Expected dataset files present: {existing_expected}/{TOTAL_FILES}"
)

unexpected_npz = (
    set(DEV.glob("*.npz")) | set(SEALED.glob("*.npz"))
) - set(expected_files)

report(
    len(unexpected_npz) == 0,
    f"No unexpected NPZ files ({len(unexpected_npz)} found)"
)


# ============================================================
# 6. CHECK SPLIT FRAME-ID SEPARATION
# ============================================================

print("\n6. FRAME-ID SPLIT SEPARATION")
print("-" * 70)

for snr in SNRS:

    train = all_ids.get(("train", snr))
    validation = all_ids.get(("validation", snr))
    test = all_ids.get(("test", snr))

    if train is None or validation is None or test is None:
        report(False, f"SNR {snr:.2f}: unable to check all splits")
        continue

    train_val = len(np.intersect1d(train, validation)) == 0
    train_test = len(np.intersect1d(train, test)) == 0
    val_test = len(np.intersect1d(validation, test)) == 0

    report(
        train_val and train_test and val_test,
        f"SNR {snr:.2f}: train/validation/test IDs do not overlap"
    )


# ============================================================
# 7. CHECK GENERATION METADATA
# ============================================================

print("\n7. GENERATION METADATA")
print("-" * 70)

config_path = ROOT / "generation_config.json"

if config_path.exists():
    try:
        with open(config_path, "r") as f:
            config = json.load(f)

        report(True, "generation_config.json is valid JSON")

        report(
            config.get("sionna_version") == "2.2.0",
            f"Sionna version: {config.get('sionna_version')}"
        )

        report(
            config.get("base_seed") == 20261120,
            "Base seed is 20261120"
        )

        report(
            config.get("batch_size") == 1000,
            "Batch size is 1000"
        )

        report(
            config.get("decoder", {}).get("num_iterations") == 50,
            "Decoder iterations are 50"
        )

        print("Metadata keys:", list(config.keys()))

    except Exception as exc:
        report(False, f"Could not read generation metadata: {exc}")
else:
    report(False, "generation_config.json exists")


# ============================================================
# 8. CHECK BLER SUMMARY
# ============================================================

print("\n8. BLER SUMMARY")
print("-" * 70)

bler_path = ROOT / "bler_summary.csv"

if bler_path.exists():
    try:
        with open(bler_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        report(True, "bler_summary.csv is readable")

        print("CSV columns:", reader.fieldnames)
        print("CSV data rows:", len(rows))

        report(len(rows) > 0, "BLER summary contains data rows")

        print("\nFirst five BLER summary rows:")
        for row in rows[:5]:
            print(row)

    except Exception as exc:
        report(False, f"Could not read BLER summary: {exc}")
else:
    report(False, "bler_summary.csv exists")


# ============================================================
# 9. FINAL REPORT
# ============================================================

print("\n" + "=" * 70)
print("FINAL AUDIT RESULT")
print("=" * 70)

print(f"Expected NPZ files: {TOTAL_FILES}")
print(f"Present NPZ files:  {existing_expected}")
print(f"Checks passed:      {passed_checks}")
print(f"Checks failed:      {failed_checks}")

if failed_checks == 0:
    print("\nOVERALL RESULT: PASS")
    print("Dataset audit completed without detected issues.")
else:
    print("\nOVERALL RESULT: REVIEW REQUIRED")
    print("Inspect the failed checks before handing over the dataset.")

print("=" * 70)
