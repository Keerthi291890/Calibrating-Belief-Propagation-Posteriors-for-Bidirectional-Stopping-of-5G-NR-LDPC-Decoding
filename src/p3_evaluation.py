"""
Person 3 evaluation layer.

Reads Person 2's frozen result CSVs only. Does NOT rerun decoding, refit models,
retune thresholds, or modify Person 2 outputs.

Run from the repository root:
    python src/p3_evaluation.py
Outputs:
    results/person3/matched_tradeoff.csv
    results/person3/mismatch_stopping.csv
    results/person3/PERSON3_EVALUATION_SUMMARY.md
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
IN_DIR = ROOT / "results" / "person2_v3"
OUT_DIR = ROOT / "results" / "person3"
RHO = 0.10
TOL = 0.15  # percentage-point tolerance for recomputed vs recorded savings


def require_columns(df, required, filename):
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"{filename} is missing required columns: {missing}")


def matched_tradeoff():
    path = IN_DIR / "final_stopping_test.csv"
    df = pd.read_csv(path)
    require_columns(
        df,
        ["snr", "rho", "rule", "mean_it", "saving_vs_B2_pct",
         "excess_events", "budget_events", "budget_met_on_test",
         "total_ops_saving_vs_B2_pct", "total_saving_pct_cx10",
         "false_accept", "false_abandon", "accept_share", "nack_rate",
         "cap_share", "net_bler_change"],
        path.name,
    )

    # B2 is repeated across rho values; retain one baseline row per SNR.
    b2 = (df[df["rule"] == "B2 syndrome-zero"]
          .sort_values("rho").drop_duplicates("snr")
          [["snr", "mean_it"]]
          .rename(columns={"mean_it": "b2_mean_it"}))

    chosen = df[
        (df["rho"].sub(RHO).abs() < 1e-9)
        & df["rule"].isin(
            ["B2 syndrome-zero", "B5 syndrome + cap", "P-cal both"]
        )
    ].copy()

    chosen = chosen.merge(b2, on="snr", how="left", validate="many_to_one")
    chosen["recomputed_iteration_saving_pct"] = (
        100.0 * (chosen["b2_mean_it"] - chosen["mean_it"])
        / chosen["b2_mean_it"]
    )
    chosen["saving_check_abs_diff_pp"] = (
        chosen["recomputed_iteration_saving_pct"]
        - chosen["saving_vs_B2_pct"]
    ).abs()

    if chosen["b2_mean_it"].isna().any():
        raise ValueError("Could not match every selected row to its B2 baseline.")
    if (chosen["saving_check_abs_diff_pp"] > TOL).any():
        bad = chosen.loc[
            chosen["saving_check_abs_diff_pp"] > TOL,
            ["snr", "rule", "saving_check_abs_diff_pp"],
        ]
        raise ValueError(
            "Recorded iteration savings do not match recomputation:\n"
            + bad.to_string(index=False)
        )

    columns = [
        "snr", "rho", "rule", "params", "b2_mean_it", "mean_it",
        "recomputed_iteration_saving_pct", "excess_events", "budget_events",
        "budget_met_on_test", "false_accept", "false_abandon",
        "accept_share", "nack_rate", "cap_share", "net_bler_change",
        "total_ops_saving_vs_B2_pct", "total_saving_pct_cx10",
    ]
    return chosen[columns].sort_values(["snr", "rule"])


def mismatch_stopping():
    path = IN_DIR / "final_stopping_mismatch.csv"
    df = pd.read_csv(path)
    require_columns(
        df,
        ["snr", "delta", "calibrators", "rule", "mean_it",
         "saving_vs_B2_same_delta_pct", "excess_events", "excess_vs_budget",
         "budget_met", "false_accept", "false_abandon", "nack_rate",
         "accept_share", "net_bler_change",
         "total_ops_saving_vs_B2_same_delta_pct"],
        path.name,
    )

    # Retain B2 as the reference and the bidirectional calibrated method.
    chosen = df[
        df["rule"].isin(["B2 syndrome-zero", "P-cal both"])
        & (
            (df["rule"] == "B2 syndrome-zero")
            | (df["calibrators"].isin(["matched", "refit"]))
        )
    ].copy()

    # A unique B2 row exists for each SNR/delta; P-cal both may have
    # matched and refit variants at selected mismatch conditions.
    return chosen.sort_values(["snr", "delta", "rule", "calibrators"])


def write_summary(matched, mismatch):
    pcal = matched[matched["rule"] == "P-cal both"].copy()
    b5 = matched[matched["rule"] == "B5 syndrome + cap"].copy()

    pcal_budget_misses = int((~pcal["budget_met_on_test"].astype(bool)).sum())
    b5_budget_misses = int((~b5["budget_met_on_test"].astype(bool)).sum())

    # Mismatch budget failures are directly counted from the frozen table.
    mm_pcal = mismatch[mismatch["rule"] == "P-cal both"]
    mm_pcal_matched = mm_pcal[
        (mm_pcal["calibrators"] == "matched") & (mm_pcal["delta"] != 0)
    ]
    mm_misses = int((~mm_pcal_matched["budget_met"].astype(bool)).sum())

    lines = [
        "# Person 3: stopping and evaluation summary",
        "",
        "Generated from Person 2's frozen CSV outputs. No model fitting, threshold",
        "selection, decoding, or sealed-test rerun is performed by this script.",
        "",
        "## Matched test: 10% excess-event budget",
        "",
        "Iteration savings are recomputed from mean iterations and the B2 baseline.",
        "The recorded saving is checked against the recomputation.",
        "",
        "| SNR (dB) | Method | Mean iterations | Iteration saving vs B2 | Excess / budget | Test budget met | Total-op saving vs B2 (cx=10) |",
        "|---:|---|---:|---:|---:|:---:|---:|",
    ]
    for _, r in matched.iterrows():
        lines.append(
            f"| {r['snr']:.2f} | {r['rule']} | {r['mean_it']:.4f} | "
            f"{r['recomputed_iteration_saving_pct']:.2f}% | "
            f"{int(r['excess_events'])} / {r['budget_events']:.1f} | "
            f"{'Yes' if bool(r['budget_met_on_test']) else 'No'} | "
            f"{r['total_saving_pct_cx10']:.2f}% |"
        )

    lines += [
        "",
        f"- P-cal both test-budget misses: **{pcal_budget_misses}/{len(pcal)}**.",
        f"- B5 test-budget misses: **{b5_budget_misses}/{len(b5)}**.",
        "- Report realised excess events alongside savings; validation-tuned thresholds "
        "do not guarantee test-budget compliance.",
        "- Iteration savings and total-operation savings are different objectives. "
        "The thresholds were selected to minimise mean iterations, not total operations.",
        "",
        "## Mismatch stopping",
        "",
        "| SNR (dB) | Noise variance delta (dB) | Calibrators | Method | Iteration saving vs same-delta B2 | Excess / budget | Budget met | Total-op saving vs B2 |",
        "|---:|---:|---|---|---:|---:|:---:|---:|",
    ]
    for _, r in mm_pcal.iterrows():
        lines.append(
            f"| {r['snr']:.2f} | {int(r['delta']):+d} | {r['calibrators']} | "
            f"P-cal both | {r['saving_vs_B2_same_delta_pct']:.2f}% | "
            f"{r['excess_vs_budget']} | "
            f"{'Yes' if bool(r['budget_met']) else 'No'} | "
            f"{r['total_ops_saving_vs_B2_same_delta_pct']:.2f}% |"
        )

    lines += [
        "",
        f"- Matched-calibrator P-cal both mismatch budget misses: "
        f"**{mm_misses}/{len(mm_pcal_matched)}**.",
        "- This count excludes delta = 0 (the matched-no-mismatch reference) and "
        "excludes refit rows; refit results are a separate analysis.",
        "- Mismatch test frames are paired with the matched test frames; they are "
        "not independent samples.",
        "- Refitting results are reported only where the frozen CSV contains a refit row.",
        "",
        "## Interpretation constraints",
        "",
        "- Abandonment is a NACK and counts as a failure in net BLER.",
        "- False accept and false abandon are excess events only when the full 50-iteration "
        "decision would have been correct.",
        "- Operation counts are approximate scalar-operation estimates, not measured "
        "hardware instructions, latency, or energy.",
        "- The experiment covers one 5G NR BG2 code configuration, BPSK/AWGN, "
        "SNR 2.00–2.75 dB, and assumed noise-variance mismatch.",
        "- Reliability statements are empirical; no certified guarantee is established.",
        "",
        "## Source files",
        "",
        "- `results/person2_v3/final_stopping_test.csv`",
        "- `results/person2_v3/final_stopping_mismatch.csv`",
        "",
    ]
    (OUT_DIR / "PERSON3_EVALUATION_SUMMARY.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    matched = matched_tradeoff()
    mismatch = mismatch_stopping()

    matched.to_csv(OUT_DIR / "matched_tradeoff.csv", index=False)
    mismatch.to_csv(OUT_DIR / "mismatch_stopping.csv", index=False)
    write_summary(matched, mismatch)

    print("Person 3 evaluation outputs created:")
    print(f"  {OUT_DIR / 'matched_tradeoff.csv'}")
    print(f"  {OUT_DIR / 'mismatch_stopping.csv'}")
    print(f"  {OUT_DIR / 'PERSON3_EVALUATION_SUMMARY.md'}")
    print("\nMatched rows:", len(matched))
    print("Mismatch rows:", len(mismatch))
    print("Source frozen CSVs were read-only; no test or fitting script was run.")


if __name__ == "__main__":
    main()
