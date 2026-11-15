
import json
import time
import numpy as np
import pandas as pd
import torch

from pathlib import Path
from sionna.phy.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder


# ============================================================
# 1. CONFIGURATION
# ============================================================

K = 110
N = 220

CODE_RATE = K / N

NUM_ITER = 50
FRAMES_PER_SNR = 10_000

BATCH_SIZE = 200

SNR_VALUES = [2.25, 2.50, 2.75]

BASE_SEED = 2026

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

DEVICE = "cpu"

# ------------------------------------------------------------
# Locate repository root
# This script is located in repo/src/scripts/
# ------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = REPO_ROOT / "data"
OUTPUT_DIR = DATA_DIR / "final_dataset"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# Basic configuration checks
# ------------------------------------------------------------

assert abs(
    TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0
) < 1e-9

assert FRAMES_PER_SNR % 100 == 0

assert BATCH_SIZE > 0

torch.set_num_threads(4)

# ============================================================
# 2. CREATE ENCODER
# ============================================================

print("=" * 70)
print("FINAL LDPC DATASET GENERATION")
print("=" * 70)

print("\nRepository:", REPO_ROOT)
print("Output directory:", OUTPUT_DIR)

encoder = LDPC5GEncoder(K, N)

print("\nEncoder configuration:")
print("Information bits:", K)
print("Codeword length:", N)
print("Code rate:", CODE_RATE)
print("Base graph:", encoder._bg)
print("Lifting factor:", encoder._z)

# ============================================================
# 3. CREATE FRAME-LEVEL SPLITS
# ============================================================

def create_frame_splits(seed):

    rng = np.random.default_rng(seed)

    frame_ids = np.arange(FRAMES_PER_SNR)

    shuffled = rng.permutation(frame_ids)

    n_train = int(FRAMES_PER_SNR * TRAIN_RATIO)
    n_val = int(FRAMES_PER_SNR * VAL_RATIO)

    train_ids = shuffled[:n_train]

    val_ids = shuffled[
        n_train:n_train + n_val
    ]

    test_ids = shuffled[
        n_train + n_val:
    ]

    split_labels = np.empty(
        FRAMES_PER_SNR,
        dtype=object
    )

    split_labels[train_ids] = "train"
    split_labels[val_ids] = "validation"
    split_labels[test_ids] = "test"

    assert len(train_ids) == 7000
    assert len(val_ids) == 1500
    assert len(test_ids) == 1500

    return split_labels


# ============================================================
# 4. CREATE ONE EXPERIMENT
# ============================================================

def run_experiment(ebno_db):

    print("\n" + "=" * 70)
    print(f"STARTING SNR: {ebno_db:.2f} dB")
    print("=" * 70)

    start_time = time.time()

    # Each SNR receives a reproducible seed
    snr_code = int(round(ebno_db * 100))

    seed = BASE_SEED + snr_code

    torch.manual_seed(seed)

    # Generate frame-level split assignments
    frame_splits = create_frame_splits(seed)

    # --------------------------------------------------------
    # Output file paths
    # --------------------------------------------------------

    output_files = {
        split: OUTPUT_DIR /
        f"{split}_snr_{ebno_db:.2f}dB.csv"

        for split in ["train", "validation", "test"]
    }

    # Start fresh for these exact output files.
    # Existing pilot files are not touched.
    for path in output_files.values():
        if path.exists():
            path.unlink()

    file_initialized = {
        split: False
        for split in output_files
    }

    # --------------------------------------------------------
    # Decoder callback
    # Store only the 110 information-bit LLRs.
    # Do not retain the full internal decoder graph.
    # --------------------------------------------------------

    traces = []

    def capture(msg_vn, iteration, x_hat):

        info_llr = (
            x_hat[:, :K]
            .detach()
            .cpu()
            .clone()
        )

        traces.append((iteration, info_llr))

        return msg_vn

    # --------------------------------------------------------
    # Create decoder
    # --------------------------------------------------------

    decoder = LDPC5GDecoder(
        encoder,
        num_iter=NUM_ITER,
        return_infobits=True,
        hard_out=False,
        v2c_callbacks=[capture]
    )

    # --------------------------------------------------------
    # Calculate channel noise variance
    # --------------------------------------------------------

    ebno_linear = 10 ** (ebno_db / 10)

    noise_var = 1.0 / (
        2.0 * CODE_RATE * ebno_linear
    )

    noise_std = np.sqrt(noise_var)

    print("Noise variance:", noise_var)
    print("Noise standard deviation:", noise_std)

    # --------------------------------------------------------
    # Dataset counters
    # --------------------------------------------------------

    total_failed = 0
    total_successful = 0

    current_error_count = 0

    batch_count = 0

    # ========================================================
    # 5. GENERATE FRAMES IN BATCHES
    # ========================================================

    for start in range(
        0,
        FRAMES_PER_SNR,
        BATCH_SIZE
    ):

        end = min(
            start + BATCH_SIZE,
            FRAMES_PER_SNR
        )

        current_batch_size = end - start

        batch_count += 1

        frame_ids = np.arange(start, end)

        # Clear previous batch's callback captures
        traces.clear()

        # ----------------------------------------------------
        # Generate information bits
        # ----------------------------------------------------

        u = torch.randint(
            0,
            2,
            (current_batch_size, K)
        ).float()

        # ----------------------------------------------------
        # Encode
        # ----------------------------------------------------

        c = encoder(u)

        # ----------------------------------------------------
        # BPSK modulation
        # 0 -> +1
        # 1 -> -1
        # ----------------------------------------------------

        x = 1.0 - 2.0 * c

        # ----------------------------------------------------
        # AWGN channel
        # ----------------------------------------------------

        noise = (
            torch.randn_like(x)
            * noise_std
        )

        y = x + noise

        # ----------------------------------------------------
        # Sionna decoder input logits
        # ----------------------------------------------------

        logits = -2.0 * y / noise_var

        # ----------------------------------------------------
        # BP decoding
        # ----------------------------------------------------

        with torch.no_grad():
            u_soft = decoder(logits)

        # ----------------------------------------------------
        # Verify callback mapping
        # ----------------------------------------------------

        if len(traces) != NUM_ITER + 1:
            raise RuntimeError(
                f"Expected {NUM_ITER + 1} captures, "
                f"received {len(traces)}"
            )

        final_iteration, final_L = traces[-1]

        mapping_verified = torch.allclose(
            final_L,
            -u_soft,
            atol=1e-5
        )

        if (
            final_iteration != NUM_ITER
            or not mapping_verified
        ):
            raise RuntimeError(
                "Final decoder mapping verification failed."
            )

        # ----------------------------------------------------
        # Calculate final-failure labels
        # ----------------------------------------------------

        final_decoded = (final_L < 0).int()

        final_failure = (
            final_decoded != u.int()
        ).any(dim=1).int()

        total_failed += int(final_failure.sum())

        total_successful += (
            current_batch_size
            - int(final_failure.sum())
        )

        # ----------------------------------------------------
        # Move original bits to NumPy
        # ----------------------------------------------------

        true_bits = u.int().numpy()

        # ----------------------------------------------------
        # Generate each iteration's data
        # ----------------------------------------------------

        for iteration, L in traces:

            # Ignore channel-only initial state
            if iteration == 0:
                continue

            # L: [batch_size, 110]

            # Probability that each bit is 1
            p1 = torch.sigmoid(-L)

            # Probability that each bit is correct
            p_correct = torch.maximum(
                p1,
                1.0 - p1
            )

            # Frame-level probability estimate
            log_p_all_correct = torch.log(
                p_correct
            ).sum(dim=1)

            residual_error = -torch.expm1(
                log_p_all_correct
            )

            # Current hard decisions
            decoded = (L < 0).int()

            # Current-error labels
            current_error = (
                decoded != u.int()
            ).any(dim=1).int()

            # Convert arrays to NumPy
            llr_np = L.numpy()

            residual_np = residual_error.numpy()

            current_np = current_error.numpy()

            final_np = final_failure.numpy()

            current_error_count += int(
                current_error.sum()
            )

            # ------------------------------------------------
            # Construct iteration-level table
            # ------------------------------------------------

            data = {
                "frame_id": frame_ids,
                "split": frame_splits[frame_ids],
                "snr_db": np.full(
                    current_batch_size,
                    ebno_db
                ),
                "seed": np.full(
                    current_batch_size,
                    seed
                ),
                "iteration": np.full(
                    current_batch_size,
                    iteration
                ),
                "residual_error": residual_np,
                "current_error": current_np,
                "final_failure": final_np
            }

            # Add 110 posterior LLRs
            for bit in range(K):

                data[f"llr_{bit}"] = (
                    llr_np[:, bit]
                )

            # Add 110 original information bits
            for bit in range(K):

                data[f"true_bit_{bit}"] = (
                    true_bits[:, bit]
                )

            df = pd.DataFrame(data)

            # ------------------------------------------------
            # Write separate files for each split
            # ------------------------------------------------

            for split in [
                "train",
                "validation",
                "test"
            ]:

                split_df = df[
                    df["split"] == split
                ]

                if split_df.empty:
                    continue

                output_path = output_files[split]

                write_mode = (
                    "a"
                    if file_initialized[split]
                    else "w"
                )

                split_df.to_csv(
                    output_path,
                    mode=write_mode,
                    header=not file_initialized[split],
                    index=False,
                    float_format="%.8g"
                )

                file_initialized[split] = True

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (
            batch_count % 5 == 0
            or end == FRAMES_PER_SNR
        ):

            print(
                f"Frames processed: {end}/{FRAMES_PER_SNR}"
            )

    # ========================================================
    # 6. SNR SUMMARY
    # ========================================================

    elapsed = time.time() - start_time

    bler = total_failed / FRAMES_PER_SNR

    summary = {
        "snr_db": ebno_db,
        "frames": FRAMES_PER_SNR,
        "successful_frames": total_successful,
        "failed_frames": total_failed,
        "bler": bler,
        "runtime_seconds": elapsed,
        "seed": seed,
        "mapping_verified": True
    }

    print("\nSNR experiment completed.")

    print("Successful frames:", total_successful)
    print("Failed frames:", total_failed)

    print(
        f"Measured BLER: {bler:.6f} "
        f"({bler * 100:.3f}%)"
    )

    print(f"Runtime: {elapsed:.2f} seconds")

    print("\nSaved files:")

    for split, path in output_files.items():

        print(split, "->", path)

    return summary


# ============================================================
# 7. RUN ALL SNR EXPERIMENTS
# ============================================================

all_summaries = []

for snr in SNR_VALUES:

    result = run_experiment(snr)

    all_summaries.append(result)


# ============================================================
# 8. SAVE SUMMARY AND METADATA
# ============================================================

summary_df = pd.DataFrame(all_summaries)

summary_path = OUTPUT_DIR / "snr_summary.csv"

summary_df.to_csv(
    summary_path,
    index=False
)

# ------------------------------------------------------------
# Save frame-level split manifest
# ------------------------------------------------------------

manifest_rows = []

for snr in SNR_VALUES:

    seed = BASE_SEED + int(round(snr * 100))

    splits = create_frame_splits(seed)

    for frame_id, split in enumerate(splits):

        manifest_rows.append({
            "frame_id": frame_id,
            "snr_db": snr,
            "split": split,
            "seed": seed
        })

manifest_df = pd.DataFrame(manifest_rows)

manifest_path = (
    OUTPUT_DIR / "frame_split_manifest.csv"
)

manifest_df.to_csv(
    manifest_path,
    index=False
)

# ------------------------------------------------------------
# Save metadata
# ------------------------------------------------------------

metadata = {
    "project": (
        "Calibrating Belief-Propagation Posteriors "
        "for Bidirectional Stopping of 5G NR LDPC Decoding"
    ),
    "configuration": {
        "K": K,
        "N": N,
        "code_rate": CODE_RATE,
        "base_graph": "BG2",
        "lifting_factor": int(encoder._z),
        "num_iterations": NUM_ITER,
        "frames_per_snr": FRAMES_PER_SNR,
        "batch_size": BATCH_SIZE,
        "snr_values_db": SNR_VALUES,
        "modulation": "BPSK",
        "channel": "AWGN",
        "decoder": "Sionna LDPC5GDecoder",
        "device": DEVICE
    },
    "noise_model": {
        "ebno_definition": "Eb/N0",
        "noise_variance_formula": (
            "1 / (2 * code_rate * 10**(EbNo_dB/10))"
        ),
        "bpsk_mapping": {
            "bit_0": "+1",
            "bit_1": "-1"
        }
    },
    "posterior": {
        "internal_llr": "log(P(bit=0)/P(bit=1))",
        "probability_bit_1": "sigmoid(-internal_llr)",
        "residual_error": (
            "1 - product(max(p_i, 1-p_i))"
        )
    },
    "labels": {
        "current_error": (
            "1 if any information bit is incorrect "
            "at the current iteration"
        ),
        "final_failure": (
            "1 if any information bit is incorrect "
            "after 50 BP iterations"
        )
    },
    "splits": {
        "train": TRAIN_RATIO,
        "validation": VAL_RATIO,
        "test": TEST_RATIO,
        "split_unit": "frame"
    },
    "seeds": {
        "base_seed": BASE_SEED
    },
    "iteration_zero_excluded": True,
    "posterior_columns": K,
    "original_bit_columns": K
}

metadata_path = (
    OUTPUT_DIR / "experiment_metadata.json"
)

with open(metadata_path, "w") as f:

    json.dump(
        metadata,
        f,
        indent=4
    )


# ============================================================
# 9. FINAL REPORT
# ============================================================

print("\n" + "=" * 70)
print("FINAL DATASET GENERATION COMPLETED")
print("=" * 70)

print("\nSNR SUMMARY:")
print(summary_df.to_string(index=False))

print("\nTotal frames:", FRAMES_PER_SNR * len(SNR_VALUES))

print(
    "Total iteration observations:",
    FRAMES_PER_SNR * len(SNR_VALUES) * NUM_ITER
)

print("\nOutput directory:")
print(OUTPUT_DIR.resolve())

print("\nMetadata:", metadata_path)
print("Split manifest:", manifest_path)
print("Summary:", summary_path)

print("\nAll datasets generated.")
