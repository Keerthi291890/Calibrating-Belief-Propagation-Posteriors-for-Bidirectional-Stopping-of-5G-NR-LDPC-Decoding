
from pathlib import Path
import numpy as np

ROOT = Path("../data/scalar_traces")
SEALED = ROOT / "sealed_test"

DELTAS = [-3, -2, -1, 1, 2, 3]
EXPECTED_FRAMES = 10_000
SNR = 2.25

TRACE_FIELDS = {
    "log_r", "S", "min_abs", "mean_abs", "mean_ent",
    "info_err", "cur_err", "stored_r", "syndrome_weight"
}

REQUIRED_FIELDS = TRACE_FIELDS | {
    "frame_id", "snr", "seed_key", "final_failure",
    "delta", "assumed_sigma2"
}


def verify_file(delta):
    filename = f"train_mismatch_snr_{SNR:.2f}_delta{delta:+d}.npz"
    path = SEALED / filename

    if not path.exists():
        print(f"FAIL: {filename} — missing")
        return False

    with np.load(path) as data:
        missing = REQUIRED_FIELDS - set(data.files)

        if missing:
            print(f"FAIL: {filename} — missing fields: {missing}")
            return False

        n = len(data["frame_id"])

        count_ok = n == EXPECTED_FRAMES

        trace_shapes_ok = all(
            data[key].shape == (n, 50)
            for key in TRACE_FIELDS
        )

        metadata_shapes_ok = all(
            data[key].shape == (n,)
            for key in ["frame_id", "snr", "seed_key", "final_failure"]
        )

        snr_ok = np.allclose(data["snr"], SNR)
        delta_ok = float(data["delta"].item()) == delta

        # Refit frames use split_code=3.
        ids = data["frame_id"]
        ids_ok = (
            np.all(ids >= 30_000_000)
            and np.all(ids < 40_000_000)
            and len(np.unique(ids)) == n
        )

        # Verify assumed noise variance.
        sigma2_true = 1.0 / (
            2 * 0.5 * (10 ** (SNR / 10))
        )
        expected_sigma2 = sigma2_true * (10 ** (delta / 10))

        sigma2_ok = np.isclose(
            float(data["assumed_sigma2"].item()),
            expected_sigma2
        )

        # Diagnostic: final syndrome among successful frames.
        syndrome = data["syndrome_weight"][:, -1]
        success = data["final_failure"] == 0

        zero_syndrome_rate = (
            np.mean(syndrome[success] == 0)
            if np.any(success) else float("nan")
        )

        checks = [
            count_ok,
            trace_shapes_ok,
            metadata_shapes_ok,
            snr_ok,
            delta_ok,
            ids_ok,
            sigma2_ok,
        ]

        passed = all(checks)

        print(
            f"{'PASS' if passed else 'FAIL'}: {filename}\n"
            f"  Frame count:       {n}/{EXPECTED_FRAMES}\n"
            f"  Trace shapes:      {trace_shapes_ok}\n"
            f"  Metadata shapes:   {metadata_shapes_ok}\n"
            f"  SNR matches:       {snr_ok}\n"
            f"  Delta matches:     {delta_ok}\n"
            f"  Split-3 unique IDs: {ids_ok}\n"
            f"  Sigma variance:    {sigma2_ok}\n"
            f"  Zero syndrome among successful frames: "
            f"{zero_syndrome_rate:.3%}\n"
        )

        return passed


def main():
    print("=" * 65)
    print("MISMATCH REFIT DATA INTEGRITY CHECK")
    print("=" * 65)

    passed_count = 0

    for delta in DELTAS:
        if verify_file(delta):
            passed_count += 1

    print("=" * 65)
    print(f"Files passed: {passed_count}/{len(DELTAS)}")

    if passed_count == len(DELTAS):
        print("OVERALL RESULT: PASS")
    else:
        print("OVERALL RESULT: FAIL — inspect the files above")

    print("=" * 65)


if __name__ == "__main__":
    main()
