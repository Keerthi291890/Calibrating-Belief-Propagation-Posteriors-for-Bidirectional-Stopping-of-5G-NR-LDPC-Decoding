
from pathlib import Path

import pandas as pd



REPO_ROOT = Path(
    "/mnt/d/Calibrating-Belief-Propagation-Posteriors-for-Bidirectional-Stopping-of-5G-NR-LDPC-Decoding"
)

DATA_DIR = REPO_ROOT / "data" / "final_dataset"

SNRS = [2.25, 2.50, 2.75]
SPLITS = ["train", "validation", "test"]


def main():
    print("=" * 75)
    print("FINAL LDPC DATASET INSPECTION")
    print("=" * 75)
    print(f"Dataset directory: {DATA_DIR}")

    split_results = []
    iteration_results = []

    for snr in SNRS:
        for split in SPLITS:
            path = DATA_DIR / f"{split}_snr_{snr:.2f}dB.csv"

            if not path.exists():
                raise FileNotFoundError(path)

            df = pd.read_csv(path)

            # One row per frame is sufficient for final-failure statistics.
            frames = df.drop_duplicates("frame_id")

            failed_frames = int(frames["final_failure"].sum())
            total_frames = len(frames)

            split_results.append({
                "snr_db": snr,
                "split": split,
                "frames": total_frames,
                "failed_frames": failed_frames,
                "successful_frames": total_frames - failed_frames,
                "bler": failed_frames / total_frames if total_frames else float("nan"),
            })

            # Track current decoder errors and residual estimates by iteration.
            grouped = df.groupby("iteration").agg(
                frames=("frame_id", "nunique"),
                current_errors=("current_error", "sum"),
                mean_residual=("residual_error", "mean"),
                median_residual=("residual_error", "median"),
            ).reset_index()

            grouped["snr_db"] = snr
            grouped["split"] = split
            grouped["current_error_rate"] = (
                grouped["current_errors"] / grouped["frames"]
            )

            iteration_results.append(grouped)

    split_summary = pd.DataFrame(split_results)
    iteration_summary = pd.concat(iteration_results, ignore_index=True)

    print("\n--- Frame-level failure summary ---")
    print(
        split_summary.to_string(
            index=False,
            formatters={"bler": "{:.4%}".format},
        )
    )

    print("\n--- Iteration behaviour: validation split ---")
    validation = iteration_summary[
        iteration_summary["split"] == "validation"
    ]

    for snr in SNRS:
        part = validation[validation["snr_db"] == snr]
        print(f"\nSNR = {snr:.2f} dB")
        print(
            part[
                [
                    "iteration",
                    "current_error_rate",
                    "mean_residual",
                    "median_residual",
                ]
            ].iloc[[0, 4, 9, 19, 29, 39, 49]].to_string(
                index=False,
                formatters={
                    "current_error_rate": "{:.4%}".format,
                    "mean_residual": "{:.6f}".format,
                    "median_residual": "{:.6f}".format,
                },
            )
        )

    split_output = DATA_DIR / "split_failure_summary.csv"
    iteration_output = DATA_DIR / "iteration_statistics.csv"

    split_summary.to_csv(split_output, index=False)
    iteration_summary.to_csv(iteration_output, index=False)

    print("\nSaved inspection results:")
    print(split_output)
    print(iteration_output)

    print("\nInspection completed.")


if __name__ == "__main__":
    main()
