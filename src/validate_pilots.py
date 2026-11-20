
import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path("snr_refinement_results")

files = [
    DATA_DIR / "ldpc_pilot_2.25dB.csv",
    DATA_DIR / "ldpc_pilot_2.50dB.csv",
    DATA_DIR / "ldpc_pilot_2.75dB.csv"
]

for file in files:

    print("\n" + "=" * 60)
    print("Checking:", file.name)
    print("=" * 60)

    df = pd.read_csv(file)

    # 1. Basic dimensions
    print("Shape:", df.shape)

    # 2. Check expected columns
    required = [
        "frame_id", "iteration", "snr_db", "seed",
        "residual_error", "current_error", "final_failure"
    ]

    assert all(c in df.columns for c in required)

    # 3. Check 110 LLR and 110 true-bit columns
    llr_cols = [f"llr_{i}" for i in range(110)]
    bit_cols = [f"true_bit_{i}" for i in range(110)]

    assert all(c in df.columns for c in llr_cols)
    assert all(c in df.columns for c in bit_cols)

    # 4. Check missing and non-finite values
    assert not df.isnull().any().any()
    assert np.isfinite(df[llr_cols].to_numpy()).all()

    # 5. Check residual-error range
    assert df["residual_error"].between(0, 1).all()

    # 6. Check iteration range
    assert df["iteration"].min() == 1
    assert df["iteration"].max() == 50

    # 7. Check exactly 50 rows per frame
    counts = df.groupby("frame_id").size()
    assert (counts == 50).all()

    # 8. Check no duplicate frame/iteration pairs
    assert not df.duplicated(
        subset=["frame_id", "iteration"]
    ).any()

    # 9. Check final-failure label is constant per frame
    assert (
        df.groupby("frame_id")["final_failure"].nunique() == 1
    ).all()

    # 10. Compare final-failure with iteration-50 error
    final_rows = df[df["iteration"] == 50]

    comparison = final_rows.set_index("frame_id")[
        "current_error"
    ].sort_index()

    failure = final_rows.set_index("frame_id")[
        "final_failure"
    ].sort_index()

    assert comparison.equals(failure)

    failed_frames = int(failure.sum())
    total_frames = len(failure)

    print("Frames:", total_frames)
    print("Rows per frame: 50")
    print("LLR columns:", len(llr_cols))
    print("Original-bit columns:", len(bit_cols))
    print("Failed frames:", failed_frames)
    print("BLER:", failed_frames / total_frames)

    print("\nAll validation checks passed!")

print("\nAll pilot datasets validated successfully.")
