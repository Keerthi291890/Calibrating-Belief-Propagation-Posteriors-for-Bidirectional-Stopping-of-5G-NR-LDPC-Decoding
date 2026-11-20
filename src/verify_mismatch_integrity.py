
from pathlib import Path
import numpy as np

# --------------------------------------------------
# Paths and expected dataset configuration
# --------------------------------------------------

ROOT = Path("../data/scalar_traces")
SEALED = ROOT / "sealed_test"

DELTAS = [-3, -2, -1, 1, 2, 3]

EXPECTED_COUNTS = {
    2.00: 5_000,
    2.25: 10_000,
    2.50: 20_000,
}

TRACE_FIELDS = {
    "log_r",
    "S",
    "min_abs",
    "mean_abs",
    "mean_ent",
    "info_err",
    "cur_err",
    "stored_r",
    "syndrome_weight",
}

REQUIRED_FIELDS = {
    "frame_id",
    "snr",
    "seed_key",
    "final_failure",
    "delta",
    "assumed_sigma2",
} | TRACE_FIELDS


def check_file(snr, delta, expected_count):
    filename = f"test_snr_{snr:.2f}_delta{delta:+d}.npz"
    path = SEALED / filename

    if not path.exists():
        print(f"FAIL: {filename} — file missing")
        return False

    main_path = SEALED / f"test_snr_{snr:.2f}.npz"

    if not main_path.exists():
        print(f"FAIL: Main test file missing for {snr:.2f} dB")
        return False

    with np.load(main_path) as main:
        main_ids = main["frame_id"]

    with np.load(path) as data:
        missing = REQUIRED_FIELDS - set(data.files)

        if missing:
            print(f"FAIL: {filename} — missing fields: {missing}")
            return False

        n = len(data["frame_id"])

        # Check frame count
        count_ok = n == expected_count

        # Check all trace arrays have shape (frames, 50)
        shapes_ok = all(
            data[key].shape == (n, 50)
            for key in TRACE_FIELDS
        )

        # Check metadata array shapes
        metadata_shapes_ok = all(
            data[key].shape == (n,)
            for key in ["frame_id", "snr", "seed_key", "final_failure"]
        )

        # Check mismatch frames reuse the main test frame IDs
        ids_match = np.array_equal(
            data["frame_id"],
            main_ids[:expected_count]
        )

        # Check SNR and delta metadata
        snr_match = np.allclose(data["snr"], snr)
        delta_match = float(data["delta"].item()) == delta

        # Check assumed noise variance
        sigma2_true = 1.0 / (
            2 * 0.5 * (10 ** (snr / 10))
        )
        expected_sigma2 = sigma2_true * (10 ** (delta / 10))

        sigma2_match = np.isclose(
            float(data["assumed_sigma2"].item()),
            expected_sigma2
        )

        # Diagnostic: successful frames should have zero
        # syndrome weight at the final iteration.
        syndrome = data["syndrome_weight"][:, -1]
        success = data["final_failure"] == 0

        zero_syndrome_rate = (
            np.mean(syndrome[success] == 0)
            if np.any(success)
            else float("nan")
        )

        checks = [
            count_ok,
            shapes_ok,
            metadata_shapes_ok,
            ids_match,
            snr_match,
            delta_match,
            sigma2_match,
        ]

        passed = all(checks)

        print(
            f"{'PASS' if passed else 'FAIL'}: {filename}\n"
            f"  Frame count:       {n}/{expected_count}\n"
            f"  Trace shapes:      {shapes_ok}\n"
            f"  Metadata shapes:   {metadata_shapes_ok}\n"
            f"  Frame IDs match:   {ids_match}\n"
            f"  SNR matches:       {snr_match}\n"
            f"  Delta matches:     {delta_match}\n"
            f"  Sigma variance:    {sigma2_match}\n"
            f"  Zero syndrome among successful frames: "
            f"{zero_syndrome_rate:.3%}\n"
        )

        return passed


def main():
    print("=" * 65)
    print("MISMATCH DATA INTEGRITY CHECK")
    print("=" * 65)

    total_files = 0
    passed_files = 0

    for snr, expected_count in EXPECTED_COUNTS.items():
        print(f"\nChecking SNR = {snr:.2f} dB")
        print("-" * 65)

        for delta in DELTAS:
            total_files += 1

            if check_file(snr, delta, expected_count):
                passed_files += 1

    print("=" * 65)
    print(f"Files passed: {passed_files}/{total_files}")

    if passed_files == total_files:
        print("OVERALL RESULT: PASS")
    else:
        print("OVERALL RESULT: FAIL — inspect the files above")

    print("=" * 65)


if __name__ == "__main__":
    main()
