
import torch
import pandas as pd
import numpy as np
import time
import json
from pathlib import Path

from sionna.phy.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder


# ============================================================
# 1. EXPERIMENT CONFIGURATION
# ============================================================

K = 110
N = 220

BATCH_SIZE = 1000
NUM_ITER = 50

SNR_VALUES = [2.25, 2.50, 2.75]

BASE_SEED = 42

OUTPUT_DIR = Path("snr_refinement_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cpu"

# ============================================================
# 2. CREATE ENCODER
# ============================================================

print("=" * 65)
print("LDPC SNR REFINEMENT EXPERIMENT")
print("=" * 65)

encoder = LDPC5GEncoder(K, N)

print("\nEncoder configuration:")
print("Information bits:", K)
print("Codeword length:", N)
print("Code rate:", K / N)
print("Base graph:", encoder._bg)
print("Lifting factor:", encoder._z)

# ============================================================
# 3. EXPERIMENT FUNCTION
# ============================================================

def run_experiment(ebno_db):

    print("\n" + "=" * 65)
    print(f"Running experiment at Eb/N0 = {ebno_db:.2f} dB")
    print("=" * 65)

    start_time = time.time()

    # Reproducible seed for each SNR
    seed = BASE_SEED + int(round(ebno_db * 100))
    torch.manual_seed(seed)
    np.random.seed(seed)

    # --------------------------------------------------------
    # Capture decoder estimates after every VN update
    # --------------------------------------------------------

    traces = []

    def capture(msg_vn, iteration, x_hat):
        traces.append(
            (iteration, x_hat[:, :K].detach().clone())
        )
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
    # Generate information bits
    # --------------------------------------------------------

    u = torch.randint(
        0, 2,
        (BATCH_SIZE, K)
    ).float()

    # --------------------------------------------------------
    # Encode
    # --------------------------------------------------------

    c = encoder(u)

    # --------------------------------------------------------
    # BPSK modulation
    # 0 -> +1
    # 1 -> -1
    # --------------------------------------------------------

    x = 1.0 - 2.0 * c

    # --------------------------------------------------------
    # AWGN channel
    # --------------------------------------------------------

    code_rate = K / N

    ebno_linear = 10 ** (ebno_db / 10)

    noise_var = 1.0 / (
        2.0 * code_rate * ebno_linear
    )

    noise = torch.randn_like(x) * np.sqrt(noise_var)

    y = x + noise

    # --------------------------------------------------------
    # Convert channel observations to Sionna input logits
    # --------------------------------------------------------

    logits = -2.0 * y / noise_var

    # --------------------------------------------------------
    # Decode
    # --------------------------------------------------------

    u_soft = decoder(logits)

    # --------------------------------------------------------
    # Verify final callback mapping
    # --------------------------------------------------------

    final_iteration, final_L = traces[-1]

    mapping_verified = torch.allclose(
        final_L,
        -u_soft,
        atol=1e-5
    )

    if final_iteration != NUM_ITER or not mapping_verified:
        raise RuntimeError(
            "Final callback verification failed. "
            "Check the decoder callback and output mapping."
        )

    # --------------------------------------------------------
    # Final-failure labels
    # --------------------------------------------------------

    final_decoded = (final_L < 0).int()

    final_failure = (
        final_decoded != u.int()
    ).any(dim=1).int()

    # --------------------------------------------------------
    # Generate iteration-wise dataset
    # --------------------------------------------------------

    rows = []

    for iteration, L in traces:

        # Exclude iteration 0 (channel-only state)
        if iteration == 0:
            continue

        # Posterior probability P(bit=1)
        p1 = torch.sigmoid(-L)

        # Probability that each bit is correct
        p_correct = torch.maximum(
            p1,
            1.0 - p1
        )

        # Frame-level residual-error estimate
        log_p_all_correct = torch.log(p_correct).sum(dim=1)

        residual_error = -torch.expm1(
            log_p_all_correct
        )

        # Current hard decisions
        decoded = (L < 0).int()

        # Current-error labels
        current_error = (
            decoded != u.int()
        ).any(dim=1).int()

        # Move current iteration data to CPU
        L_np = L.numpy()
        u_np = u.int().numpy()

        residual_np = residual_error.numpy()
        current_np = current_error.numpy()
        final_np = final_failure.numpy()

        # Construct one row per frame per iteration
        for frame in range(BATCH_SIZE):

            row = {
                "frame_id": frame,
                "iteration": iteration,
                "snr_db": ebno_db,
                "seed": seed,
                "residual_error": float(residual_np[frame]),
                "current_error": int(current_np[frame]),
                "final_failure": int(final_np[frame])
            }

            # Save posterior LLRs
            for bit in range(K):
                row[f"llr_{bit}"] = float(
                    L_np[frame, bit]
                )

            # Save original transmitted information bits
            for bit in range(K):
                row[f"true_bit_{bit}"] = int(
                    u_np[frame, bit]
                )

            rows.append(row)

    # --------------------------------------------------------
    # Create DataFrame
    # --------------------------------------------------------

    df = pd.DataFrame(rows)

    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------

    filename = (
        f"ldpc_pilot_{ebno_db:.2f}dB.csv"
    )

    output_path = OUTPUT_DIR / filename

    df.to_csv(output_path, index=False)

    # --------------------------------------------------------
    # Calculate frame-level statistics
    # --------------------------------------------------------

    frame_results = df.drop_duplicates(
        subset=["frame_id"]
    )

    failed_frames = int(
        frame_results["final_failure"].sum()
    )

    successful_frames = BATCH_SIZE - failed_frames

    bler = failed_frames / BATCH_SIZE

    elapsed = time.time() - start_time

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print("\nResults:")
    print("Dataset shape:", df.shape)
    print("Frames:", BATCH_SIZE)
    print("BP iterations:", NUM_ITER)
    print("Captured iterations:", len(traces))

    print("\nCurrent-error labels:")
    print(df["current_error"].value_counts())

    print("\nFinal-failure labels:")
    print(frame_results["final_failure"].value_counts())

    print("\nFrame-level results:")
    print("Successful frames:", successful_frames)
    print("Failed frames:", failed_frames)
    print(f"Measured BLER: {bler:.4f} ({bler * 100:.2f}%)")

    print(f"\nRuntime: {elapsed:.2f} seconds")
    print("Saved:", output_path)

    # --------------------------------------------------------
    # Return summary
    # --------------------------------------------------------

    return {
        "snr_db": ebno_db,
        "frames": BATCH_SIZE,
        "successful_frames": successful_frames,
        "failed_frames": failed_frames,
        "bler": bler,
        "runtime_seconds": elapsed,
        "mapping_verified": mapping_verified,
        "csv_file": str(output_path)
    }


# ============================================================
# 4. RUN ALL SNR EXPERIMENTS
# ============================================================

all_results = []

for snr in SNR_VALUES:

    result = run_experiment(snr)

    all_results.append(result)


# ============================================================
# 5. SAVE SUMMARY
# ============================================================

summary_df = pd.DataFrame(all_results)

summary_path = OUTPUT_DIR / "snr_summary.csv"

summary_df.to_csv(
    summary_path,
    index=False
)

metadata = {
    "K": K,
    "N": N,
    "code_rate": K / N,
    "base_graph": "BG2",
    "num_iterations": NUM_ITER,
    "batch_size_per_snr": BATCH_SIZE,
    "snr_values_db": SNR_VALUES,
    "base_seed": BASE_SEED,
    "modulation": "BPSK",
    "channel": "AWGN",
    "decoder": "Sionna LDPC5GDecoder",
    "posterior_convention": "P(bit=1) = sigmoid(-internal_LLR)",
    "iteration_zero_excluded": True
}

metadata_path = OUTPUT_DIR / "experiment_metadata.json"

with open(metadata_path, "w") as f:
    json.dump(metadata, f, indent=4)


# ============================================================
# 6. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("ALL EXPERIMENTS COMPLETED")
print("=" * 65)

print(summary_df.to_string(index=False))

print("\nSummary saved:", summary_path)
print("Metadata saved:", metadata_path)
print("\nOutput directory:", OUTPUT_DIR.resolve())
