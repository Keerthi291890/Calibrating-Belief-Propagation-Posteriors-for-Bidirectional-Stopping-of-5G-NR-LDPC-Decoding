# Calibrating Belief-Propagation Posteriors for Bidirectional Stopping of 5G NR LDPC Decoding

## 1. Project Overview

This project investigates whether the soft information produced by Belief Propagation (BP) decoding of 5G New Radio (NR) Low-Density Parity-Check (LDPC) codes can be calibrated to support more reliable early stopping decisions.

Conventional iterative LDPC decoders generally execute a predetermined number of iterations or use a stopping condition such as a zero syndrome. However, these conditions do not always establish whether the decoded information bits are actually correct.

The central idea is to use the decoder's posterior information to estimate the likelihood of decoding errors, calibrate that estimate using observed decoding outcomes, and investigate whether calibrated estimates can support two decisions:

- **Early acceptance:** Determine whether a decoded frame is sufficiently reliable to stop decoding before the maximum iteration count.
- **Early abandonment:** Determine whether a frame is unlikely to decode successfully and whether continuing iterations is worthwhile.

The project also investigates how the reliability of these estimates changes under channel-model mismatch.

## 2. Research Question

Can BP-posterior-derived residual-error estimates be calibrated to support reliable early acceptance and early abandonment of LDPC frames, and how does their reliability degrade under channel-model mismatch?

The project focuses on three aspects:

1. Measuring the relationship between BP posterior information and actual decoding errors.
2. Learning calibration models that convert raw residual-error estimates into more reliable error-risk estimates.
3. Evaluating whether calibrated estimates can support bidirectional stopping decisions without unacceptable degradation in decoding reliability.

## 3. Motivation

An LDPC decoder iteratively exchanges information between variable nodes and check nodes in a Tanner graph.

As iterations progress, the decoder updates its estimates of the transmitted bits. The magnitude of its log-likelihood ratios (LLRs) provides information about its confidence in individual bit decisions.

However, high confidence does not necessarily imply correctness. A decoder can become highly confident in an incorrect decision.

Similarly, a zero syndrome indicates that a decoded sequence satisfies the parity-check constraints, but it does not guarantee that the decoded information bits match the transmitted information bits.

Therefore, this project investigates whether decoder confidence can be converted into a more meaningful estimate of actual decoding risk.

## 4. Project Objectives

- Implement a valid 5G NR LDPC encoder, AWGN channel and BP decoder.
- Capture decoder posterior information across iterations.
- Construct a frame-level residual-error estimate from bit-level posterior probabilities.
- Generate ground-truth labels using the transmitted information bits.
- Generate scalar iteration traces for posterior calibration.
- Evaluate calibration models using held-out data.
- Investigate early acceptance and early abandonment rules.
- Compare calibrated stopping decisions with conventional decoding baselines.
- Study the effect of channel-model mismatch on calibration and stopping reliability.

## 5. System Model and Simulation Configuration

The simulation uses the following configuration.

| Parameter | Configuration |
|---|---|
| Standard | 5G NR LDPC |
| Base graph | BG2 |
| Information bits (K) | 110 |
| Transmitted codeword length (N) | 220 |
| Code rate | 0.5 |
| Lifting factor (Z) | 20 |
| Sionna encoder internal `k_ldpc` | 200 |
| Sionna encoder internal `n_ldpc` | 1040 |
| Filler bits | 90 |
| Encoder parity-check matrix | 840 × 1040 |
| Modulation | BPSK |
| Channel | AWGN |
| Decoder | Sionna LDPC5GDecoder |
| Maximum decoder iterations | 50 |
| Decoder output | Soft output |
| Execution device | CPU |
| Sionna version | 2.2.0 |
| PyTorch version | 2.15.0+cpu |

The configuration uses a valid Sionna 5G NR BG2 construction. It is not claimed to be an exact reproduction of a particular shortened-code configuration from a reference paper.

### 5.1 BPSK Mapping

The simulation uses:

- Bit 0 → +1
- Bit 1 → −1

The transmitted BPSK signal is passed through an Additive White Gaussian Noise (AWGN) channel.

### 5.2 Noise Variance

The noise variance is calculated using:

\[
\sigma^2 =
\frac{1}{2R\cdot 10^{E_b/N_0(\mathrm{dB})/10}}
\]

where:

- \(R\) is the code rate.
- \(E_b/N_0\) is the energy-per-bit to noise-power spectral-density ratio.

The corresponding noise standard deviation is:

\[
\sigma=\sqrt{\sigma^2}
\]

### 5.3 Channel LLR Convention

The channel logits supplied to the Sionna decoder use:

\[
\ell_{\mathrm{input}}=-\frac{2y}{\sigma^2}
\]

The internal callback LLRs use the convention:

\[
L_i=\log\frac{P(b_i=0)}{P(b_i=1)}
\]

Therefore:

\[
P(b_i=1)=\operatorname{sigmoid}(-L_i)
\]

A positive LLR indicates preference for bit 0, while a negative LLR indicates preference for bit 1.

This convention must be respected when calculating posterior probabilities and residual-error estimates.

## 6. Decoder Trace Collection

The BP decoder is configured to run for a maximum of 50 iterations.

The simulation captures internal decoder information at iteration 0 and after each subsequent variable-node update.

This produces 51 captures:

\[
t=0,1,2,\ldots,50
\]

Iteration 0 is excluded from the final scalar traces. The stored traces contain iterations 1 through 50.

Sionna's default graph pruning results in an internal callback vector of length 350 for this configuration. The first 110 positions correspond to the information-bit positions returned by the decoder.

The callback-to-final-output mapping was verified by checking that the final callback information-bit LLRs match the negative of the decoder's final soft information-bit output.

### 6.1 Syndrome Computation

The encoder's original parity-check matrix has dimensions 840 × 1040. However, the decoder operates on a pruned internal representation.

For syndrome-based scalar features, the validated decoder parity-check matrix is:

- Source: `decoder._pcm`
- Shape: 150 × 350
- Format: SciPy CSR

This matrix was experimentally checked against the decoder's hard decisions. In the syndrome probe, correctly decoded frames had zero syndrome, while failed frames had nonzero syndrome.

The syndrome weight is calculated from the hard decisions over the decoder's 350 internal variables.

The private Sionna attribute used for this purpose is recorded in the generation metadata.

## 7. Residual-Error Estimate

The project derives a frame-level residual-error estimate from the posterior probabilities of the 110 information bits.

For each information bit:

\[
p_i=P(b_i=1)
\]

The probability assigned to the decoder's most likely bit decision is:

\[
p_{\mathrm{correct},i}=\max(p_i,1-p_i)
\]

The raw frame-level residual-error estimate is:

\[
r_t=
1-\prod_{i=1}^{110}
p_{\mathrm{correct},i}(t)
\]

This is an independence-based approximation of the probability that at least one information bit is incorrect.

It is not assumed to be a perfectly calibrated probability of frame error.

For numerical stability, the implementation uses:

\[
S_t=\sum_{i=1}^{110}\operatorname{softplus}(-|L_i(t)|)
\]

\[
r_t=1-\exp(-S_t)
\]

The saved log residual is:

\[
\log r_t=\log(1-\exp(-S_t))
\]

The implementation uses numerically stable operations for these calculations.

The raw residual-error estimate is the main quantity intended for posterior calibration.

## 8. Ground-Truth Labels

The dataset contains two error targets.

### 8.1 Current-Iteration Error: `cur_err`

This indicates whether the decoded information bits at the current iteration contain at least one error.

\[
y_{\mathrm{now}}(t)=
\begin{cases}
1 & \text{if any decoded information bit is incorrect}\\
0 & \text{otherwise}
\end{cases}
\]

The decoded information bits are compared against the originally transmitted information bits.

The corresponding `info_err` array stores the number of incorrectly decoded information bits at each iteration.

### 8.2 Final Decoder Failure: `final_failure`

This indicates whether the frame is still incorrect after the maximum of 50 BP iterations.

\[
y_{\mathrm{fail}}=
\begin{cases}
1 & \text{if the final decoded information bits are incorrect}\\
0 & \text{otherwise}
\end{cases}
\]

This label is constant across all 50 observations belonging to a frame.

It represents failure after the configured 50-iteration decoder run. It does not claim that the frame would remain incorrect after an unlimited number of iterations.

### 8.3 Why Ground Truth Matters

A zero syndrome is not sufficient to establish that the decoded information bits are correct.

The simulation therefore uses the original transmitted information bits to generate error labels.

The transmitted bits are used during simulation and trace extraction, but the final scalar-trace files do not contain the original bit arrays.

Ground-truth labels are retained for calibration and evaluation.

## 9. Final Dataset Generation

The final dataset uses four SNR values:

- 2.00 dB
- 2.25 dB
- 2.50 dB
- 2.75 dB

The dataset is divided into training, validation and test partitions at the frame level.

### 9.1 Main Dataset Sizes

| SNR (dB) | Train | Validation | Test |
|---:|---:|---:|---:|
| 2.00 | 10,000 | 5,000 | 10,000 |
| 2.25 | 15,000 | 8,000 | 20,000 |
| 2.50 | 20,000 | 15,000 | 40,000 |
| 2.75 | 30,000 | 25,000 | 50,000 |
| **Total** | **75,000** | **53,000** | **120,000** |

The main matched-channel dataset therefore contains 248,000 frames across its three partitions.

Each frame contributes 50 iteration-level observations.

\[
248,000\times50=12,400,000
\]

Thus, the main matched-channel dataset represents 12.4 million iteration-level observations.

### 9.2 Measured BLER

The measured main-dataset BLER is:

| SNR (dB) | Train BLER | Validation BLER | Test BLER |
|---:|---:|---:|---:|
| 2.00 | 6.260% | 6.100% | 6.670% |
| 2.25 | 3.440% | 3.375% | 3.125% |
| 2.50 | 1.335% | 1.393% | 1.545% |
| 2.75 | 0.620% | 0.612% | 0.562% |

BLER is calculated as:

\[
\mathrm{BLER}=
\frac{\text{Number of failed frames}}
{\text{Total number of frames}}
\]

These are measured simulation results, not theoretical performance guarantees.

## 10. Channel-Model Mismatch Experiments

The project investigates how calibration behaves when the decoder's assumed noise variance differs from the true channel noise variance.

The mismatch is introduced through:

\[
\sigma^2_{\mathrm{assumed}}=
\sigma^2_{\mathrm{true}}\cdot10^{\Delta/10}
\]

where:

\[
\Delta\in\{-3,-2,-1,+1,+2,+3\}\text{ dB}
\]

A positive delta means the decoder assumes a higher noise variance than the true variance. A negative delta means it assumes a lower noise variance.

The transmitted bits and channel noise are reused from the corresponding main-test frames. Only the assumed noise variance used to calculate decoder LLRs changes.

### 10.1 Mismatch-Test Dataset

| Actual SNR (dB) | Frames per delta | Number of deltas | Total records |
|---:|---:|---:|---:|
| 2.00 | 5,000 | 6 | 30,000 |
| 2.25 | 10,000 | 6 | 60,000 |
| 2.50 | 20,000 | 6 | 120,000 |
| 2.75 | Not generated | 0 | 0 |
| **Total** | | | **210,000** |

The mismatch-test dataset reuses the corresponding main-test frame IDs.

It is intended for evaluating how calibration behaves under channel-model mismatch. It must not be used for model fitting, feature selection or threshold tuning.

### 10.2 Mismatch-Refit Dataset

An additional mismatch-refit dataset was generated at 2.25 dB.

It contains:

- 10,000 underlying frames per mismatch condition.
- Six nonzero mismatch deltas.
- 60,000 condition records in total.

All six mismatch-refit files use exactly the same frame IDs. This allows paired comparisons across mismatch conditions.

These files are training data for mismatch-aware calibration/refitting experiments, not held-out test data.

**Directory note:** The mismatch-refit files are currently stored inside `data/scalar_traces/sealed_test/` because of the generation pipeline's output organization. Their filenames begin with `train_mismatch_`, and their metadata identifies them as split-code 3 training data. Do not interpret every file under `sealed_test/` as held-out test data.

### 10.3 Mismatch BLER

Measured mismatch-test BLER:

| Actual SNR | −3 dB | −2 dB | −1 dB | +1 dB | +2 dB | +3 dB |
|---:|---:|---:|---:|---:|---:|---:|
| 2.00 dB | 11.66% | 9.48% | 7.80% | 6.34% | 9.02% | 26.52% |
| 2.25 dB | 6.26% | 5.14% | 4.10% | 3.18% | 3.95% | 11.52% |
| 2.50 dB | 3.10% | 2.485% | 1.84% | 1.34% | 1.585% | 4.135% |

No mismatch experiment was generated at 2.75 dB.

These results describe the simulated decoder behavior under the specified noise-variance mismatch conditions. They do not represent all possible channel impairments.

## 11. Scalar-Trace Dataset Schema

The final dataset is stored as compressed NumPy NPZ files rather than the earlier preliminary CSV format.

Each file contains arrays representing a batch of frames and their 50 decoder iterations.

### 11.1 Main NPZ Arrays

| Key | Shape | Description |
|---|---|---|
| `frame_id` | (B,) | Frame identifier |
| `snr` | (B,) | Actual channel SNR in dB |
| `seed_key` | (B,) | Batch-level seed identifier |
| `log_r` | (B, 50) | Log raw residual-error estimate |
| `S` | (B, 50) | Sum of softplus terms |
| `min_abs` | (B, 50) | Minimum absolute information-bit LLR |
| `mean_abs` | (B, 50) | Mean absolute information-bit LLR |
| `mean_ent` | (B, 50) | Mean bit-level entropy-like quantity |
| `info_err` | (B, 50) | Number of incorrect information bits |
| `cur_err` | (B, 50) | Current information-frame error label |
| `stored_r` | (B, 50) | Raw residual-error estimate |
| `syndrome_weight` | (B, 50) | Number of unsatisfied parity checks |
| `final_failure` | (B,) | Final 50-iteration failure label |

Here, B is the number of frames in the file.

The mismatch files additionally contain:

- `delta`: Assumed noise-variance mismatch in dB.
- `assumed_sigma2`: Assumed noise variance used by the decoder.

The original per-bit LLR arrays and transmitted information-bit arrays are not stored in the final scalar-trace dataset.

### 11.2 Loading a Dataset File

Example:

```python
import numpy as np

path = "../data/scalar_traces/dev/train_snr_2.00.npz"

data = np.load(path)

print(data.files)

for key in data.files:
    arr = data[key]
    print(key, arr.shape, arr.dtype)
```

Example feature access:

```python
log_r = data["log_r"]
stored_r = data["stored_r"]
cur_err = data["cur_err"]
final_failure = data["final_failure"]
syndrome_weight = data["syndrome_weight"]
```

Each row of a `(B, 50)` array corresponds to one frame. Its 50 columns correspond to iterations 1 through 50.

### 11.3 Important Feature and Label Separation

The following are ground-truth quantities and must not be used as predictive features:

- `info_err`
- `cur_err`
- `final_failure`

They are used as targets and for evaluation.

The following are candidate predictive features:

- `log_r`
- `stored_r`
- `S`
- `min_abs`
- `mean_abs`
- `mean_ent`
- `syndrome_weight`
- Iteration number, derived from the trace column position.
- Actual SNR, where appropriate.
- Trajectory-based features derived using only current and previous iterations.

The use of syndrome weight as a predictive feature should be evaluated separately from conventional syndrome-based stopping baselines.

## 12. Data Splitting and Reproducibility

The final dataset uses deterministic random-number generation.

### 12.1 Base Seed and Batch Size

| Parameter | Value |
|---|---:|
| Base seed | 20261120 |
| Batch size | 1000 |
| Main SNR values | 2.00, 2.25, 2.50, 2.75 dB |
| Decoder iterations | 50 |

The random-number streams are generated independently for each SNR, split and batch.

The frame ID layout is:

\[
\mathrm{frame\_id}
=
\mathrm{split\_code}\times10,000,000
+
\mathrm{batch\_index}\times1000+i
\]

where \(i\) is the frame's position within the batch.

The split codes are:

| Split | Code |
|---|---:|
| Train | 0 |
| Validation | 1 |
| Test | 2 |
| Mismatch refit | 3 |

`seed_key` stores the batch index.

### 12.2 Frame-Level Separation

The training, validation and main test datasets use separate frame-ID ranges and independent random streams.

All 50 observations belonging to a frame remain in the same partition.

Mismatch-test records intentionally reuse the corresponding main-test frame IDs. Mismatch-refit records use a separate split-code range.

Rows from the same frame are correlated across iterations. Statistical uncertainty should therefore be estimated using frame-level resampling rather than treating every iteration observation as an independent transmitted frame.

## 13. Repository Structure

The current dataset-related repository structure is:

```text
Calibrating-Belief-Propagation-Posteriors-for-Bidirectional-Stopping-of-5G-NR-LDPC-Decoding/
│
├── README.md
├── PERSON2_HANDOFF.md
│
├── src/
│   ├── generate_scalar_traces.py
│   ├── verify_mismatch_integrity.py
│   ├── verify_refit_integrity.py
│   └── final_dataset_audit.py
│
└── data/
    ├── scalar_traces/
    │   ├── dev/
    │   │   ├── train_snr_2.00.npz
    │   │   ├── train_snr_2.25.npz
    │   │   ├── train_snr_2.50.npz
    │   │   ├── train_snr_2.75.npz
    │   │   ├── validation_snr_2.00.npz
    │   │   ├── validation_snr_2.25.npz
    │   │   ├── validation_snr_2.50.npz
    │   │   └── validation_snr_2.75.npz
    │   │
    │   ├── sealed_test/
    │   │   ├── test_snr_*.npz
    │   │   ├── test_snr_*_delta*.npz
    │   │   └── train_mismatch_snr_*.npz
    │   │
    │   ├── generation_config.json
    │   └── bler_summary.csv
    │
    ├── final_dataset/
    │   └── Earlier preliminary CSV dataset
    │
    └── snr_refinement_results/
        └── Earlier pilot experiment outputs
```

The `final_dataset/` and `snr_refinement_results/` directories contain earlier experiments. They are preserved for reference and are not the authoritative final scalar-trace dataset.

The authoritative dataset for the current calibration stage is:

```text
data/scalar_traces/
```

## 14. Environment and Dependencies

The final simulation was run in WSL2 Ubuntu 24.04 using Python 3.12.

The environment uses:

- Sionna 2.2.0
- PyTorch 2.15.0+cpu
- NumPy
- SciPy
- Pandas
- Matplotlib
- Scikit-learn

The simulation was executed on CPU.

Activate the existing virtual environment:

```bash
source ~/venvs/ldpc-sionna/bin/activate
```

The repository is available in WSL at:

```bash
/mnt/d/Calibrating-Belief-Propagation-Posteriors-for-Bidirectional-Stopping-of-5G-NR-LDPC-Decoding
```

The generation scripts are located in `src/`.

## 15. Dataset Generation and Validation

The final scalar traces were generated in separate stages.

Run these commands from the `src/` directory with the project environment activated.

### 15.1 Generate training and validation data

```bash
python generate_scalar_traces.py --stage trainval
```

This generates the eight main training and validation NPZ files under:

```text
data/scalar_traces/dev/
```

### 15.2 Generate main test data

```bash
python generate_scalar_traces.py --stage test
```

This generates the four main test files under:

```text
data/scalar_traces/sealed_test/
```

### 15.3 Generate mismatch-test data

```bash
python generate_scalar_traces.py --stage mismatch
```

This generates the 18 mismatch-test files.

### 15.4 Generate mismatch-refit training data

```bash
python generate_scalar_traces.py --stage refit
```

This generates six additional mismatch-refit training files.

These are training data despite their current storage location under `sealed_test/`.

**Important:** These commands describe the completed generation stages. Do not rerun them against the existing final dataset unless regeneration is explicitly required and the existing data has been safely backed up.

### 15.5 Integrity Checks

Mismatch-test integrity:

```bash
python verify_mismatch_integrity.py
```

Mismatch-refit integrity:

```bash
python verify_refit_integrity.py
```

### 15.6 Final Dataset Audit

Run from `src/`:

```bash
python final_dataset_audit.py
```

The final audit checked:

- All 36 expected NPZ files are present.
- No unexpected NPZ files exist.
- Expected frame counts and trace shapes are correct.
- Metadata shapes and SNR values are correct.
- Frame IDs are unique within each file.
- Main train, validation and test frame IDs do not overlap.
- Mismatch-test frame IDs align with their corresponding main-test frames.
- Mismatch deltas and assumed noise variances are correct.
- Mismatch-refit files use the same frame IDs across all six delta conditions.
- Generation metadata is valid.
- Sionna version is recorded as 2.2.0.
- Decoder iteration count is recorded as 50.
- The BLER summary contains all 36 file records.

**Final audit result: 277 checks passed, 0 failed.**

This confirms structural and metadata consistency. It does not replace statistical evaluation of the calibration models or stopping rules.

## 16. Planned Calibration and Stopping Stage

The next stage investigates calibration models that map raw residual-error estimates to empirical error risks.

### 16.1 Current-Error Calibration

One candidate logistic model is:

\[
q_{\mathrm{now}}(t)=
\sigma(a\log r_t+b)
\]

This estimates the probability that the current decoded information frame contains at least one error.

### 16.2 Final-Failure Calibration

A candidate model for final decoder failure can incorporate residual-error history:

\[
q_{\mathrm{fail}}(t)=
\sigma(a\log r_t+b\Delta_k+c)
\]

where:

\[
\Delta_k=\log(r_t)-\log(r_{t-k})
\]

The iteration window \(k\) should be selected using validation data.

### 16.3 Evaluation

Calibration should be evaluated using:

- Brier score.
- Log loss.
- Expected Calibration Error (ECE).
- Reliability diagrams.
- Calibration intercept and slope.

Discrimination should also be reported using:

- ROC-AUC.
- Precision-recall AUC.
- Precision and recall at selected operating points.

Results should be reported separately by SNR, iteration and mismatch condition where appropriate.

The main test and mismatch-test datasets must remain untouched until model and preprocessing decisions are fixed.

### 16.4 Bidirectional Stopping

The stopping stage will investigate:

- Early acceptance when estimated current-error risk is sufficiently low.
- Early abandonment when estimated final-failure risk is sufficiently high.
- Fixed-iteration and syndrome-based baselines.
- False acceptance and false abandonment.
- Average decoding iterations.
- BLER and reliability trade-offs.

Thresholds should be selected using validation data and evaluated on held-out test data.

The project does not assume in advance that calibration will improve every metric or that early stopping will necessarily produce energy savings.

## 17. Current Project Status

### Completed

- 5G NR BG2 encoder configuration.
- BPSK and AWGN simulation.
- BP decoder integration.
- Iteration-wise decoder information extraction.
- Decoder callback-to-output mapping verification.
- Residual-error estimate calculation.
- Ground-truth error-label generation.
- Main dataset generation at four SNR values.
- Frame-level training, validation and test splitting.
- Mismatch-test generation at six nonzero mismatch deltas.
- Mismatch-refit training-data generation.
- Mismatch integrity verification.
- Mismatch-refit pairing verification.
- Final dataset audit.

### Remaining Research Work

- Posterior feature engineering.
- Current-error calibration model training.
- Final-failure calibration model training.
- Calibration quality evaluation.
- Matched-channel held-out testing.
- Channel-model mismatch evaluation.
- Mismatch-aware calibration/refitting analysis.
- Bidirectional stopping-rule implementation.
- Baseline comparisons.
- Final stopping-performance evaluation.
- Final research analysis and reporting.

## 18. Important Scope and Limitations

1. The main channel experiments use AWGN and BPSK.
2. The raw residual-error estimate uses an independence approximation across information bits.
3. `final_failure` refers to the output after 50 iterations, not unlimited decoder convergence.
4. Zero syndrome does not guarantee that the decoded information bits are correct.
5. The mismatch experiments vary the assumed noise variance; they do not represent every possible real-world channel impairment.
6. Mismatch-test records reuse main-test frames and must not be treated as independent additional test frames.
7. Mismatch-refit files are training data despite their current location in `sealed_test/`.
8. A good ROC-AUC does not automatically imply good probability calibration.
9. A calibrated probability estimate does not automatically prove that a stopping rule is safe or efficient.
10. The main test and mismatch-test datasets must not be used for model selection or threshold tuning.
11. Ground-truth labels must not leak into predictive features.
12. The configuration is a valid Sionna 5G NR BG2 setup and should not be described as an exact reproduction of a different shortened-code configuration.

## 19. Reproducibility Metadata

The authoritative generation metadata is stored at:

```text
data/scalar_traces/generation_config.json
```

It records:

- Base seed and random-number generation layout.
- Batch size.
- SNR values.
- Split sizes and split codes.
- Frame ID formula.
- LDPC code configuration.
- Decoder configuration.
- Parity-check matrix provenance.
- Noise model and LLR convention.
- Scalar trace schema.
- Mismatch experiment configuration.
- Sionna and PyTorch environment information.
- Per-file generation summaries.

The BLER summary is stored at:

```text
data/scalar_traces/bler_summary.csv
```

Preserve these metadata files alongside the generated dataset for reproducibility.

---

## Acknowledgement

This repository contains the simulation, scalar-trace generation, mismatch experiments and dataset-validation work for research on calibrated BP posterior estimates and bidirectional stopping of 5G NR LDPC decoding.

The calibration and stopping stages use these generated traces to investigate whether posterior-derived risk estimates can support more reliable decoding decisions.

## Calibration and stopping stage (Person 2) -- addendum to this README

### Status update
Completed in this stage: statistical audit of the scalar traces, posterior feature engineering, current-error and
final-failure calibration, calibration evaluation on validation (with frame-level bootstrap), matched-channel and mismatch
previews, tail calibration, stopping-rule module, operation-cost model, freeze/final-test protocol.
Final test run completed once (see FINAL_RESULTS_PERSON2.md). Pending: paper sections and the cross-run reproducibility check.

### Differences from the plan in section 16 above
- The candidate `q_now = sigmoid(a log r + b)` is significantly WORSE than raw r on validation (reported as a result).
  The calibrated models add terms (S = -log(1-r), syndrome weight, min|LLR|, iteration index) and are fitted on syndrome>0 rows,
  the only rows where a stopping decision is still open.
- The window-based `q_fail` is extended with the iteration index; the trajectory term alone adds little.

### Additional files
```text
src/
  calib_lib.py            stable features, metrics, calibrators (JSON), trace helper
  p2_common.py            data loading (test sealed), feature table, bootstrap, evaluation helpers
  p2_stopping.py          stopping rules, metrics, operation counts, threshold tuning
  risk_model.py           q_now / q_fail interface
  p2_audit.py             statistical audit of dev + refit data
  p2_fit_dev.py           fit on train, select on validation, save models
  p2_mismatch_dev.py      mismatch preview on the refit (training) files
  p2_stop_preview.py      tail calibration + in-sample stopping preview on validation
  p2_freeze.py            freeze models, code hashes, validation-tuned thresholds
  p2_final_test.py        one-shot final evaluation (calibration, stopping, mismatch)
  p2_paper_tables.py      paper tables/figures from the final CSVs (outputs only)
  p2_repro_check.py       cross-run reproducibility check of the frozen calibrators
tests/  test_calib_lib.py  test_stopping.py  make_mock_npz.py (fake data for code tests only)
results/person2_v3/       artifacts, tables, plots, FROZEN.json, TEST_ACCESS_LOG.txt
HANDOFF_PERSON2_TO_PERSON3.md   FINAL_RESULTS_PERSON2.md   RUN_ORDER.md   PERSON2_NOTES.md
```

### Reproduce
```bash
python tests/test_calib_lib.py && python tests/test_stopping.py
python src/p2_audit.py        --data-root <root>
python src/p2_fit_dev.py      --data-root <root>
python src/p2_mismatch_dev.py --data-root <root>
python src/p2_stop_preview.py --data-root <root> --n-edges <nnz of decoder._pcm>
python src/p2_freeze.py       --data-root <root> --n-edges <nnz> --i-have-reviewed-validation
python src/p2_final_test.py   --data-root <root>        # once
```
Requires one extra number from the generation environment: `dec._pcm.nnz` (for operation counts).
