
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd


# Repository root: .../Calibrating-.../
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "final_dataset"

SNRS = [2.25, 2.50, 2.75]
SPLITS = {
    "train": 7000,
    "validation": 1500,
    "test": 1500,
}

EXPECTED_FRAMES_PER_SNR = 10000
EXPECTED_ROWS_PER_FRAME = 50
EXPECTED_TOTAL_ROWS = 1_500_000
EXPECTED_LLR_COLUMNS = 110
EXPECTED_BIT_COLUMNS = 110

BASE_COLUMNS = {
    "frame_id",
    "split",
    "snr_db",
    "seed",
    "iteration",
    "residual_error",
    "current_error",
    "final_failure",
}


def fail(message):
    print(f"\n[FAIL] {message}")
    sys.exit(1)


def check(condition, message):
    if not condition:
        fail(message)
    print(f"[PASS] {message}")


def main():
    print("=" * 70)
    print("FINAL LDPC DATASET VALIDATION")
    print("=" * 70)
    print(f"Repository: {REPO_ROOT}")
    print(f"Dataset directory: {DATA_DIR}")

    check(DATA_DIR.is_dir(), "Dataset directory exists")

    summary_path = DATA_DIR / "snr_summary.csv"
    metadata_path = DATA_DIR / "experiment_metadata.json"
    manifest_path = DATA_DIR / "frame_split_manifest.csv"

    for path in [summary_path, metadata_path, manifest_path]:
        check(path.is_file(), f"Found {path.name}")

    summary = pd.read_csv(summary_path)
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    manifest = pd.read_csv(manifest_path)

    print("\n--- Experiment summary ---")
    print(summary.to_string(index=False))

    check(len(summary) == len(SNRS), "Summary contains exactly three SNR experiments")
    check(
        set(np.round(summary["snr_db"].astype(float), 2)) == set(SNRS),
        "Summary contains SNR values 2.25, 2.50 and 2.75 dB",
    )
    check(
        int(summary["frames"].sum()) == EXPECTED_FRAMES_PER_SNR * len(SNRS),
        "Summary reports 30,000 generated frames",
    )
    check(
        bool(summary["mapping_verified"].astype(bool).all()),
        "Decoder callback-to-output mapping was verified for every SNR",
    )

    total_rows = 0
    all_frame_keys = []

    for snr in SNRS:
        snr_tag = f"{snr:.2f}dB"
        print(f"\n{'=' * 70}")
        print(f"CHECKING SNR {snr:.2f} dB")
        print("=" * 70)

        snr_frames = 0

        for split, expected_frames in SPLITS.items():
            filename = f"{split}_snr_{snr_tag}.csv"
            path = DATA_DIR / filename

            check(path.is_file(), f"{filename} exists")

            df = pd.read_csv(path)
            rows = len(df)
            total_rows += rows

            check(
                BASE_COLUMNS.issubset(df.columns),
                f"{filename}: required metadata and label columns exist",
            )

            llr_columns = [c for c in df.columns if c.startswith("llr_")]
            bit_columns = [c for c in df.columns if c.startswith("true_bit_")]

            check(
                len(llr_columns) == EXPECTED_LLR_COLUMNS,
                f"{filename}: contains 110 LLR columns",
            )
            check(
                len(bit_columns) == EXPECTED_BIT_COLUMNS,
                f"{filename}: contains 110 transmitted-bit columns",
            )
            check(
                len(df.columns) == 228,
                f"{filename}: contains 228 columns in total",
            )

            check(
                df["snr_db"].nunique() == 1
                and np.isclose(float(df["snr_db"].iloc[0]), snr),
                f"{filename}: all rows have the correct SNR",
            )
            check(
                df["split"].nunique() == 1
                and df["split"].iloc[0] == split,
                f"{filename}: split column matches the file",
            )

            frame_counts = df.groupby("frame_id")["iteration"].nunique()
            check(
                len(frame_counts) == expected_frames,
                f"{filename}: contains {expected_frames} distinct frames",
            )
            check(
                (frame_counts == EXPECTED_ROWS_PER_FRAME).all(),
                f"{filename}: every frame has exactly 50 iterations",
            )

            check(
                df["iteration"].min() == 1 and df["iteration"].max() == 50,
                f"{filename}: iterations range from 1 through 50",
            )
            check(
                not df.duplicated(["frame_id", "iteration"]).any(),
                f"{filename}: no duplicate frame/iteration pairs",
            )

            check(
                df["residual_error"].between(0, 1).all(),
                f"{filename}: residual_error values are within [0, 1]",
            )

            for label in ["current_error", "final_failure"]:
                check(
                    set(df[label].dropna().unique()).issubset({0, 1}),
                    f"{filename}: {label} contains only 0 and 1",
                )

            check(
                np.isfinite(df[llr_columns].to_numpy(dtype=np.float64)).all(),
                f"{filename}: all LLR values are finite",
            )
            check(
                np.isfinite(df[bit_columns].to_numpy(dtype=np.float64)).all(),
                f"{filename}: all transmitted-bit values are finite",
            )
            check(
                set(np.unique(df[bit_columns].to_numpy())).issubset({0, 1}),
                f"{filename}: transmitted-bit values are binary",
            )

            # final_failure must be constant across all iterations of a frame.
            failure_counts = df.groupby("frame_id")["final_failure"].nunique()
            check(
                (failure_counts == 1).all(),
                f"{filename}: final_failure is constant within each frame",
            )

            # At iteration 50, current_error must equal final_failure.
            last_rows = df[df["iteration"] == 50]
            check(
                (last_rows["current_error"].to_numpy()
                 == last_rows["final_failure"].to_numpy()).all(),
                f"{filename}: iteration-50 current_error matches final_failure",
            )

            snr_frames += df["frame_id"].nunique()
            all_frame_keys.append(
                df[["frame_id"]].drop_duplicates().assign(snr_db=snr, split=split)
            )

            print(
                f"  {split:10s}: {rows:>7,} rows | "
                f"{df['frame_id'].nunique():,} frames"
            )

            del df

        check(
            snr_frames == EXPECTED_FRAMES_PER_SNR,
            f"SNR {snr:.2f} dB: exactly 10,000 frames across all splits",
        )

    check(
        total_rows == EXPECTED_TOTAL_ROWS,
        "All CSV files contain exactly 1,500,000 observations",
    )

    # Check frame split sizes and that a frame is not assigned to multiple splits.
    manifest_required = {"frame_id", "split", "snr_db"}
    check(
        manifest_required.issubset(manifest.columns),
        "Split manifest contains frame_id, split and snr_db",
    )

    for snr in SNRS:
        part = manifest[np.isclose(manifest["snr_db"].astype(float), snr)]
        check(
            len(part) == EXPECTED_FRAMES_PER_SNR,
            f"Manifest has 10,000 frame assignments for SNR {snr:.2f} dB",
        )
        counts = part["split"].value_counts().to_dict()
        check(
            counts == SPLITS,
            f"Manifest split counts for {snr:.2f} dB are 7000/1500/1500",
        )
        check(
            not part["frame_id"].duplicated().any(),
            f"Manifest has no duplicate frame assignments for {snr:.2f} dB",
        )

    print("\n" + "=" * 70)
    print("ALL VALIDATION CHECKS PASSED")
    print("=" * 70)
    print(f"Total frames: {EXPECTED_FRAMES_PER_SNR * len(SNRS):,}")
    print(f"Total observations: {total_rows:,}")
    print("The dataset is ready for the next project stage.")


if __name__ == "__main__":
    main()
