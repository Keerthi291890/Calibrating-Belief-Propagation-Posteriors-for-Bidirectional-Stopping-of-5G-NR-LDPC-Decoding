#!/usr/bin/env python3
"""
Generate scalar posterior traces for:
"Calibrating Belief-Propagation Posteriors for Bidirectional Stopping
of 5G NR LDPC Decoding"

ACTIVE generator for Person 2's scalar-trace NPZ contract.

The previous CSV generator is retained separately as
generate_legacy_csv_dataset.py. Do not run that script for this dataset.

Default output:
    /mnt/e/5th_sem_projects/Math/scalar_traces/

Recommended sequence:
    python generate_scalar_traces.py --stage pilot
    # Check pilot BLER with Person 2.
    python generate_scalar_traces.py --stage trainval
    python generate_scalar_traces.py --stage test
    python generate_scalar_traces.py --stage mismatch
    # Optional, only if requested:
    python generate_scalar_traces.py --stage refit

To run the pilot and then all required stages automatically (the script
stops if pilot BLER is outside [3%, 12%]):
    python generate_scalar_traces.py --stage all

The pilot uses the first 5,000 frames of the deterministic 2.00 dB test
stream. The final test stage regenerates those same frames, so pilot frames
are not additional observations and are not double-counted.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import scipy.sparse as sp
import torch
from sionna.phy.fec.ldpc import LDPC5GDecoder, LDPC5GEncoder

# Make the sibling src/calib_lib.py importable when this script is run directly.
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from calib_lib import git_hash, trace_arrays
except ImportError as exc:
    raise ImportError(
        "Could not import calib_lib.py. Save Person 2's helper as "
        f"{SRC_DIR / 'calib_lib.py'} before running this generator."
    ) from exc


# ---------------------------------------------------------------------
# Fixed experiment specification — do not change without agreement
# ---------------------------------------------------------------------

K = 110
N = 220
CODE_RATE = K / N
NUM_ITER = 50
BATCH_SIZE = 1000
BASE_SEED = 20261120
SNR_VALUES = [2.00, 2.25, 2.50, 2.75]
SPLIT_CODES = {"train": 0, "validation": 1, "test": 2, "mismatch": 3}

SPLIT_SIZES = {
    2.00: {"train": 10_000, "validation": 5_000, "test": 10_000},
    2.25: {"train": 15_000, "validation": 8_000, "test": 20_000},
    2.50: {"train": 20_000, "validation": 15_000, "test": 40_000},
    2.75: {"train": 30_000, "validation": 25_000, "test": 50_000},
}

MISMATCH_TEST_SIZES = {2.00: 5_000, 2.25: 10_000, 2.50: 20_000}
MISMATCH_DELTAS_DB = [-3, -2, -1, 1, 2, 3]
REFIT_SNR = 2.25
REFIT_FRAMES_PER_DELTA = 10_000

DEFAULT_OUTPUT_DIR = Path("/mnt/d/Calibrating-Belief-Propagation-Posteriors-for-Bidirectional-Stopping-of-5G-NR-LDPC-Decoding/data/scalar_traces")
DEVICE = "cpu"
PILOT_FRAMES = 5_000
PILOT_MIN_BLER = 0.03
PILOT_MAX_BLER = 0.12

EXPECTED_TRACE_KEYS = {
    "log_r", "S", "min_abs", "mean_abs", "mean_ent",
    "info_err", "cur_err", "stored_r", "final_failure",
    "syndrome_weight",
}


def noise_variance(ebno_db: float) -> float:
    """BPSK/AWGN variance for the project's Eb/N0 convention."""
    return 1.0 / (2.0 * CODE_RATE * (10.0 ** (ebno_db / 10.0)))


def make_rng(ebno_db: float, split_code: int, batch_index: int):
    """The exact independent per-batch stream agreed with Person 2."""
    ss = np.random.SeedSequence(
        [BASE_SEED, int(round(ebno_db * 100)), split_code, batch_index]
    )
    return np.random.default_rng(ss)


def frame_ids_for(split_code: int, batch_index: int, batch_size: int):
    return (
        split_code * 10_000_000
        + batch_index * BATCH_SIZE
        + np.arange(batch_size, dtype=np.int64)
    )


def to_scipy_csr(matrix) -> sp.csr_matrix:
    """Convert Sionna's pruned PCM to the SciPy CSR format used by calib_lib."""
    if sp.issparse(matrix):
        return matrix.tocsr().astype(np.int8)

    if isinstance(matrix, torch.Tensor):
        # Covers COO and compressed sparse layouts (CSR/CSC), as well as dense.
        if matrix.layout != torch.strided:
            matrix = matrix.to_dense()
        matrix = matrix.detach().cpu().numpy()

    arr = np.asarray(matrix)
    return sp.csr_matrix(arr.astype(np.int8))


def find_llr_max(decoder):
    """Report the actual decoder LLR clipping setting without guessing."""
    candidates = ("llr_max", "_llr_max")
    for name in candidates:
        if hasattr(decoder, name):
            value = getattr(decoder, name)
            if isinstance(value, (int, float, np.number)):
                return float(value), name
            try:
                return float(value.item()), name
            except Exception:
                pass
    return None, "not exposed as llr_max or _llr_max"


def package_version(package_name: str) -> str:
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


class LDPCTraceRunner:
    """One encoder/decoder pair; callback captures all 350 VN LLRs."""

    def __init__(self):
        torch.set_num_threads(4)
        self.encoder = LDPC5GEncoder(K, N)
        self.captures = []

        def capture(msg_vn, iteration, x_hat):
            # Keep the complete decoder state: syndrome calculation requires
            # all 350 pruned variable-node values, not only the 110 info bits.
            self.captures.append(
                (int(iteration), x_hat.detach().cpu().numpy().copy())
            )
            return msg_vn

        self.decoder = LDPC5GDecoder(
            self.encoder,
            num_iter=NUM_ITER,
            return_infobits=True,
            hard_out=False,
            v2c_callbacks=[capture],
        )

        if not hasattr(self.decoder, "_pcm"):
            raise RuntimeError(
                "This Sionna decoder does not expose decoder._pcm; "
                "do not proceed until the syndrome matrix is identified."
            )

        self.H = to_scipy_csr(self.decoder._pcm)
        self.llr_max, self.llr_max_source = find_llr_max(self.decoder)

        if self.H.shape[1] != 350:
            raise RuntimeError(
                f"Expected pruned H with 350 columns; got {self.H.shape}"
            )

        print("Encoder:")
        print(f"  BG={self.encoder._bg}, lifting factor={self.encoder._z}")
        print(f"Decoder pruned PCM: shape={self.H.shape}, nnz={self.H.nnz}")
        print(f"Decoder LLR max: {self.llr_max} ({self.llr_max_source})")

    def decode_batch(
        self,
        u_np: np.ndarray,
        noise_np: np.ndarray,
        ebno_db: float,
        delta_db: float = 0.0,
    ):
        """Encode fixed bits, apply fixed standard-normal noise, then decode."""
        sigma2_true = noise_variance(ebno_db)
        sigma2_assumed = sigma2_true * (10.0 ** (delta_db / 10.0))

        u = torch.as_tensor(u_np, dtype=torch.float32, device=DEVICE)
        c = self.encoder(u)
        bpsk = 1.0 - 2.0 * c

        noise = torch.as_tensor(
            noise_np, dtype=torch.float32, device=DEVICE
        )
        y = bpsk + noise * np.sqrt(sigma2_true)
        logits = -2.0 * y / sigma2_assumed

        self.captures.clear()
        with torch.no_grad():
            u_soft = self.decoder(logits)

        if len(self.captures) != NUM_ITER + 1:
            raise RuntimeError(
                f"Expected {NUM_ITER + 1} callback captures; "
                f"received {len(self.captures)}"
            )

        iterations = [item[0] for item in self.captures]
        if iterations != list(range(NUM_ITER + 1)):
            raise RuntimeError(
                f"Unexpected callback iteration sequence: {iterations}"
            )

        # Verify Sionna's callback-to-return mapping on every batch.
        final_x = self.captures[-1][1]
        final_info_llr = torch.as_tensor(final_x[:, :K])
        mapping_ok = torch.allclose(
            final_info_llr,
            -u_soft.detach().cpu(),
            atol=1e-5,
            rtol=1e-5,
        )
        if not mapping_ok:
            raise RuntimeError("Final callback mapping verification failed.")

        # Drop iteration 0; helper expects iterations 1..50, each full width.
        xhat_list = [arr for iteration, arr in self.captures if iteration > 0]
        if len(xhat_list) != NUM_ITER:
            raise RuntimeError("Could not form exactly 50 iteration traces.")

        trace = trace_arrays(
            xhat_list, u_np, H=self.H, neg_is_one=True
        )

        # Cross-check final failure label against Sionna's returned soft bits.
        # Cross-check final failure label against Sionna's returned soft bits.
# Callback LLR = -u_soft, so bit 1 corresponds to u_soft > 0.
        final_hard = (u_soft.detach().cpu().numpy() > 0).astype(np.int64)

        final_failure_check = np.any(
             final_hard != u_np.astype(np.int64), axis=1
        ).astype(np.int8)

        if not np.array_equal(trace["final_failure"], final_failure_check):
            raise RuntimeError(
              "trace_arrays final_failure disagrees with decoder output."
            )

        missing = EXPECTED_TRACE_KEYS.difference(trace)
        if missing:
            raise RuntimeError(f"trace_arrays omitted required keys: {missing}")

        return trace, sigma2_true, sigma2_assumed


def make_batch(runner, ebno_db, split_code, batch_index, batch_size, delta_db=0.0):
    rng = make_rng(ebno_db, split_code, batch_index)
    u = rng.integers(0, 2, size=(batch_size, K), dtype=np.int8)
    noise = rng.standard_normal(size=(batch_size, N))
    trace, sigma2_true, sigma2_assumed = runner.decode_batch(
        u, noise, ebno_db, delta_db=delta_db
    )
    ids = frame_ids_for(split_code, batch_index, batch_size)
    return trace, ids, sigma2_true, sigma2_assumed


def concatenate_batches(batch_outputs):
    """Concatenate each scalar-trace field along its frame dimension."""
    if not batch_outputs:
        raise ValueError("No batches were generated.")
    keys = batch_outputs[0].keys()
    return {key: np.concatenate([item[key] for item in batch_outputs], axis=0)
            for key in keys}


def save_npz(path: Path, arrays: dict, extra: dict | None = None, overwrite=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"{path} already exists. Use --overwrite only if you intend "
            "to replace this output."
        )
    payload = dict(arrays)
    if extra:
        payload.update(extra)
    np.savez_compressed(path, **payload)


def run_split_file(runner, output_dir, split, ebno_db, frame_count, overwrite):
    split_code = SPLIT_CODES[split]
    outputs = []
    failed = 0
    started = time.time()
    n_batches = frame_count // BATCH_SIZE
    if frame_count % BATCH_SIZE:
        raise ValueError(f"{split} frame count must be divisible by {BATCH_SIZE}.")

    for batch_index in range(n_batches):
        trace, ids, _, _ = make_batch(
            runner, ebno_db, split_code, batch_index, BATCH_SIZE
        )
        outputs.append(trace)
        failed += int(trace["final_failure"].sum())
        done = (batch_index + 1) * BATCH_SIZE
        if (batch_index + 1) % 5 == 0 or done == frame_count:
            print(
                f"  {split} {ebno_db:.2f} dB: {done:,}/{frame_count:,} "
                f"frames | running BLER={failed/done:.5f}"
            )

    arrays = concatenate_batches(outputs)
    # IDs are generated by the same split stream; append per-batch IDs in order.
    arrays["frame_id"] = np.concatenate([
        frame_ids_for(split_code, b, BATCH_SIZE) for b in range(n_batches)
    ])
    arrays["snr"] = np.full(frame_count, ebno_db, dtype=np.float32)
    arrays["seed_key"] = np.tile(
        np.arange(n_batches, dtype=np.int32).repeat(BATCH_SIZE)[:frame_count],
        1,
    )

    path = output_dir / f"{split}_snr_{ebno_db:.2f}.npz"
    save_npz(path, arrays, overwrite=overwrite)
    elapsed = time.time() - started
    bler = failed / frame_count
    print(f"  Saved {path} | BLER={bler:.6f} | {elapsed:.1f}s")
    return {
        "file": str(path),
        "split": split,
        "snr_db": ebno_db,
        "frames": frame_count,
        "failed_frames": failed,
        "bler": bler,
        "runtime_seconds": elapsed,
    }


def run_pilot(runner, output_root, overwrite):
    print("\nPILOT: 5,000 frames at 2.00 dB using test stream (split_code=2)")
    failed = 0
    for batch_index in range(PILOT_FRAMES // BATCH_SIZE):
        trace, _, _, _ = make_batch(
            runner, 2.00, SPLIT_CODES["test"], batch_index, BATCH_SIZE
        )
        failed += int(trace["final_failure"].sum())
        done = (batch_index + 1) * BATCH_SIZE
        print(f"  Pilot: {done:,}/{PILOT_FRAMES:,} | running BLER={failed/done:.5f}")

    bler = failed / PILOT_FRAMES
    result = {
        "snr_db": 2.00,
        "frames": PILOT_FRAMES,
        "failed_frames": failed,
        "bler": bler,
        "test_stream": True,
        "note": (
            "These are the first 5,000 frames of the final 2.00 dB test "
            "stream; the final test stage deterministically regenerates them."
        ),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    path = output_root / "pilot_summary.json"
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; use --overwrite to replace it.")
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nPILOT BLER = {bler:.4%} ({failed}/{PILOT_FRAMES})")
    if bler < PILOT_MIN_BLER or bler > PILOT_MAX_BLER:
        print(
            "STOP: pilot BLER is outside the agreed 3%–12% range. "
            "Do not generate the full dataset until Person 2 confirms the SNR."
        )
        return False, result
    print("Pilot is within the agreed range. Full generation may proceed.")
    return True, result


def run_trainval(runner, root, overwrite):
    summaries = []
    for snr in SNR_VALUES:
        for split in ("train", "validation"):
            summaries.append(
                run_split_file(
                    runner, root / "dev", split, snr,
                    SPLIT_SIZES[snr][split], overwrite
                )
            )
    return summaries


def run_test(runner, root, overwrite):
    summaries = []
    for snr in SNR_VALUES:
        summaries.append(
            run_split_file(
                runner, root / "sealed_test", "test", snr,
                SPLIT_SIZES[snr]["test"], overwrite
            )
        )
    return summaries


def run_mismatch(runner, root, overwrite):
    summaries = []
    out_dir = root / "sealed_test"
    for snr, frame_count in MISMATCH_TEST_SIZES.items():
        n_batches = frame_count // BATCH_SIZE
        for delta in MISMATCH_DELTAS_DB:
            outputs = []
            failed = 0
            started = time.time()
            assumed_sigma2 = noise_variance(snr) * (10.0 ** (delta / 10.0))
            print(
                f"\nMismatch: SNR={snr:.2f} dB, delta={delta:+d} dB, "
                f"frames={frame_count:,}"
            )
            for batch_index in range(n_batches):
                # Reuse split_code=2 test RNG exactly: same u, same standard
                # normal noise, same frame IDs. Only assumed variance changes.
                trace, ids, sigma2_true, sigma2_assumed = make_batch(
                    runner, snr, SPLIT_CODES["test"], batch_index,
                    BATCH_SIZE, delta_db=delta
                )
                outputs.append(trace)
                failed += int(trace["final_failure"].sum())
                done = (batch_index + 1) * BATCH_SIZE
                if (batch_index + 1) % 5 == 0 or done == frame_count:
                    print(
                        f"  {done:,}/{frame_count:,} | "
                        f"running BLER={failed/done:.5f}"
                    )

            arrays = concatenate_batches(outputs)
            arrays["frame_id"] = np.concatenate([
                frame_ids_for(SPLIT_CODES["test"], b, BATCH_SIZE)
                for b in range(n_batches)
            ])
            arrays["snr"] = np.full(frame_count, snr, dtype=np.float32)
            arrays["seed_key"] = np.repeat(
                np.arange(n_batches, dtype=np.int32), BATCH_SIZE
            )
            path = out_dir / f"test_snr_{snr:.2f}_delta{delta:+d}.npz"
            save_npz(
                path,
                arrays,
                extra={
                    "delta": np.asarray(delta, dtype=np.int32),
                    "assumed_sigma2": np.asarray(assumed_sigma2, dtype=np.float64),
                },
                overwrite=overwrite,
            )
            bler = failed / frame_count
            elapsed = time.time() - started
            summaries.append({
                "file": str(path),
                "split": "mismatch",
                "snr_db": snr,
                "delta_db": delta,
                "frames": frame_count,
                "failed_frames": failed,
                "bler": bler,
                "assumed_sigma2": assumed_sigma2,
                "runtime_seconds": elapsed,
            })
            print(f"  Saved {path} | BLER={bler:.6f} | {elapsed:.1f}s")
    return summaries


def run_refit(runner, root, overwrite):
    """Optional mismatch-refit data: separate split_code=3 streams."""
    summaries = []
    out_dir = root / "sealed_test"
    snr = REFIT_SNR
    n_batches = REFIT_FRAMES_PER_DELTA // BATCH_SIZE
    for delta in MISMATCH_DELTAS_DB:
        outputs = []
        failed = 0
        started = time.time()
        assumed_sigma2 = noise_variance(snr) * (10.0 ** (delta / 10.0))
        for batch_index in range(n_batches):
            trace, _, _, _ = make_batch(
                runner, snr, SPLIT_CODES["mismatch"], batch_index,
                BATCH_SIZE, delta_db=delta
            )
            outputs.append(trace)
            failed += int(trace["final_failure"].sum())
        arrays = concatenate_batches(outputs)
        arrays["frame_id"] = np.concatenate([
            frame_ids_for(SPLIT_CODES["mismatch"], b, BATCH_SIZE)
            for b in range(n_batches)
        ])
        arrays["snr"] = np.full(REFIT_FRAMES_PER_DELTA, snr, dtype=np.float32)
        arrays["seed_key"] = np.repeat(
            np.arange(n_batches, dtype=np.int32), BATCH_SIZE
        )
        path = out_dir / f"train_mismatch_snr_{snr:.2f}_delta{delta:+d}.npz"
        save_npz(
            path,
            arrays,
            extra={
                "delta": np.asarray(delta, dtype=np.int32),
                "assumed_sigma2": np.asarray(assumed_sigma2, dtype=np.float64),
            },
            overwrite=overwrite,
        )
        bler = failed / REFIT_FRAMES_PER_DELTA
        summaries.append({
            "file": str(path),
            "split": "mismatch_refit",
            "snr_db": snr,
            "delta_db": delta,
            "frames": REFIT_FRAMES_PER_DELTA,
            "failed_frames": failed,
            "bler": bler,
            "assumed_sigma2": assumed_sigma2,
            "runtime_seconds": time.time() - started,
        })
        print(f"Saved {path} | BLER={bler:.6f}")
    return summaries


def write_generation_config(root, runner, summaries, pilot_result=None):
    root.mkdir(parents=True, exist_ok=True)
    config_path = root / "generation_config.json"
    existing = {}
    if config_path.exists():
        try:
            existing = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception:
            existing = {}

    all_summaries = existing.get("file_summaries", [])
    known_files = {row.get("file") for row in all_summaries}
    for row in summaries:
        if row["file"] not in known_files:
            all_summaries.append(row)
            known_files.add(row["file"])
        else:
            all_summaries = [
                row if old.get("file") == row["file"] else old
                for old in all_summaries
            ]

    if pilot_result is None:
        pilot_path = root / "pilot_summary.json"
        if pilot_path.exists():
            try:
                pilot_result = json.loads(pilot_path.read_text(encoding="utf-8"))
            except Exception:
                pilot_result = None

    config = {
        "project": (
            "Calibrating Belief-Propagation Posteriors for "
            "Bidirectional Stopping of 5G NR LDPC Decoding"
        ),
        "base_seed": BASE_SEED,
        "batch_size": BATCH_SIZE,
        "snr_values_db": SNR_VALUES,
        "split_sizes": {
            f"{snr:.2f}": SPLIT_SIZES[snr] for snr in SNR_VALUES
        },
        "split_codes": SPLIT_CODES,
        "frame_id_formula": "split_code*10000000 + batch_index*1000 + i",
        "seed_sequence": (
            "[BASE_SEED, round(snr_db*100), split_code, batch_index]"
        ),
        "code": {
            "K": K,
            "N": N,
            "code_rate": CODE_RATE,
            "base_graph": str(runner.encoder._bg),
            "lifting_factor": int(runner.encoder._z),
            "modulation": "BPSK: 0 -> +1, 1 -> -1",
            "channel": "AWGN",
        },
        "decoder": {
            "name": "Sionna LDPC5GDecoder",
            "num_iterations": NUM_ITER,
            "return_infobits": True,
            "hard_out": False,
            "device": DEVICE,
            "H_source": "decoder._pcm",
            "H_shape": list(runner.H.shape),
            "H_format": "SciPy CSR",
            "LLR_convention": "log(P(bit=0)/P(bit=1)); negative means bit 1",
            "llr_max": runner.llr_max,
            "llr_max_attribute": runner.llr_max_source,
        },
        "noise_model": {
            "ebno_definition": "Eb/N0",
            "variance_formula": "1 / (2 * code_rate * 10**(EbNo_dB/10))",
            "mismatch_assumed_variance": "sigma2_true * 10**(delta_db/10)",
            "decoder_logits": "-2*y/assumed_sigma2",
        },
        "trace_schema": {
            "per_iteration_shape": [BATCH_SIZE, NUM_ITER],
            "fields": [
                "frame_id", "snr", "seed_key", "final_failure",
                "log_r", "S", "min_abs", "mean_abs", "mean_ent",
                "info_err", "cur_err", "stored_r", "syndrome_weight",
            ],
            "iteration_indices": "1..50; iteration 0 excluded",
            "no_raw_llrs_or_true_bits_saved": True,
        },
        "mismatch": {
            "deltas_db": MISMATCH_DELTAS_DB,
            "test_frame_counts": MISMATCH_TEST_SIZES,
            "uses_same_test_bits_noise_and_frame_ids": True,
        },
        "provenance": {
            "git_hash": git_hash(SRC_DIR.parent),
            "sionna_version": package_version("sionna"),
            "torch_version": str(torch.__version__),
            "numpy_version": np.__version__,
            "scipy_version": package_version("scipy"),
            "generation_time_utc": datetime.now(timezone.utc).isoformat(),
            "generator": Path(__file__).name,
        },
        "pilot": pilot_result or existing.get("pilot"),
        "file_summaries": all_summaries,
    }
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    # One combined BLER report, covering every generated file.
    rows = []
    for item in all_summaries:
        rows.append({
            "file": Path(item["file"]).name,
            "split": item["split"],
            "snr_db": item["snr_db"],
            "delta_db": item.get("delta_db", 0),
            "frames": item["frames"],
            "failed_frames": item["failed_frames"],
            "bler": item["bler"],
            "runtime_seconds": item.get("runtime_seconds"),
        })
    import csv
    report_path = root / "bler_summary.csv"
    if rows:
        with report_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    print(f"\nUpdated metadata: {config_path}")
    if rows:
        print(f"Updated BLER report: {report_path}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("pilot", "trainval", "test", "mismatch", "refit", "all"),
        default="pilot",
        help="Which generation stage to run. Default: pilot.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output root (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing output files that already exist.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    root = args.output_dir
    root.mkdir(parents=True, exist_ok=True)
    (root / "dev").mkdir(exist_ok=True)
    (root / "sealed_test").mkdir(exist_ok=True)

    print("=" * 72)
    print("5G NR LDPC SCALAR-TRACE GENERATOR")
    print("=" * 72)
    print(f"Repository: {SRC_DIR.parent}")
    print(f"Output root: {root}")
    print(f"Stage: {args.stage}")
    print(f"Batch size: {BATCH_SIZE}; iterations: {NUM_ITER}")
    print(f"Base seed: {BASE_SEED}")

    runner = LDPCTraceRunner()
    summaries = []
    pilot_result = None

    if args.stage in ("pilot", "all"):
        passed, pilot_result = run_pilot(runner, root, args.overwrite)
        if args.stage == "pilot":
            return
        if not passed:
            print("\nFull generation was not started.")
            return

    if args.stage in ("trainval", "all"):
        summaries.extend(run_trainval(runner, root, args.overwrite))
    if args.stage in ("test", "all"):
        summaries.extend(run_test(runner, root, args.overwrite))
    if args.stage in ("mismatch", "all"):
        summaries.extend(run_mismatch(runner, root, args.overwrite))
    if args.stage == "refit":
        summaries.extend(run_refit(runner, root, args.overwrite))

    write_generation_config(root, runner, summaries, pilot_result=pilot_result)
    print("\nRequested generation stage completed.")


if __name__ == "__main__":
    main()
