# Calibrating Belief-Propagation Posteriors for Bidirectional Stopping of 5G NR LDPC Decoding

## 1. Project Overview

This project investigates whether the soft information produced by Belief Propagation (BP) decoding of 5G New Radio (NR) Low-Density Parity-Check (LDPC) codes can be calibrated to support more reliable early stopping decisions.

Conventional iterative LDPC decoders generally execute a predetermined number of iterations or use a stopping condition such as a zero syndrome. However, these conditions do not always indicate whether the decoded information bits are actually correct.

The central idea of this project is to use the decoder's posterior information to estimate the likelihood of decoding errors, calibrate that estimate using observed decoding outcomes, and use the calibrated estimate to support two decisions:

- **Early acceptance:** determine whether a decoded frame is sufficiently reliable to stop decoding before the maximum iteration count.
- **Early abandonment:** determine whether a frame is unlikely to decode successfully and whether continuing iterations is worthwhile.

The project also investigates how the reliability of these estimates changes under channel-model mismatch.

## 2. Research Question

Can BP-posterior-derived residual-error estimates be calibrated to support reliable early acceptance and early abandonment of LDPC frames, and how does their reliability degrade under channel-model mismatch?

The project focuses on three aspects:

1. Measuring the relationship between BP posterior information and actual decoding errors.
2. Learning a calibration model that converts raw residual-error estimates into more reliable error-risk estimates.
3. Evaluating whether calibrated estimates can support bidirectional stopping decisions without unacceptable degradation in decoding reliability.

## 3. Motivation

An LDPC decoder iteratively exchanges information between variable nodes and check nodes in a Tanner graph.

As iterations progress, the decoder updates its estimates of the transmitted bits. The magnitude of its log-likelihood ratios (LLRs) provides information about its confidence in individual bit decisions.

However, high confidence does not necessarily imply correctness. A decoder can become highly confident in an incorrect decision.

Similarly, a zero syndrome indicates that a decoded sequence satisfies the parity-check constraints, but it does not guarantee that the decoded information bits match the transmitted information bits.

Therefore, the project investigates whether decoder confidence can be converted into a more meaningful estimate of actual decoding risk.

## 4. Project Objectives

- Implement a valid 5G NR LDPC encoder, AWGN channel and BP decoder.
- Capture decoder posterior information across iterations.
- Construct a frame-level residual-error estimate from bit-level posterior probabilities.
- Generate ground-truth labels using the transmitted information bits.
- Build a dataset for studying posterior calibration.
- Evaluate calibration models using held-out data.
- Investigate early acceptance and early abandonment rules.
- Compare calibrated stopping decisions with conventional decoding baselines.
- Study the effect of channel-model mismatch on calibration and stopping reliability.

## 5. System Model and Simulation Configuration

The current simulation uses the following configuration.

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
| Parity-check matrix dimensions | 840 × 1040 |
| Modulation | BPSK |
| Channel | AWGN |
| Decoder | Belief Propagation |
| Maximum decoder iterations | 50 |
| Simulation device | CPU |

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
\frac{1}{2R \cdot 10^{E_b/N_0\text{(dB)}/10}}
\]

where:

- \(R\) is the code rate.
- \(E_b/N_0\) is the energy-per-bit to noise-power spectral-density ratio.

The corresponding noise standard deviation is:

\[
\sigma = \sqrt{\sigma^2}
\]

### 5.3 Channel LLR Convention

The channel logits supplied to the Sionna decoder use:

\[
\ell_{\text{input}} =
-\frac{2y}{\sigma^2}
\]

This follows Sionna's input-logit convention:

\[
\log\frac{P(b=1)}{P(b=0)}
\]

The internal callback LLRs use the opposite convention:

\[
L_i = \log\frac{P(b_i=0)}{P(b_i=1)}
\]

Therefore, the posterior probability of bit 1 is:

\[
p_i = P(b_i=1) = \frac{1}{1+e^{L_i}}
\]

Equivalently:

\[
p_i = \operatorname{sigmoid}(-L_i)
\]

The LLR convention must be respected when calculating posterior probabilities and residual-error estimates.

## 6. Decoder Trace Collection

The BP decoder is configured to run for a maximum of 50 iterations.

The simulation captures internal variable-node-to-check-node callback information at iteration 0 and after each subsequent variable-node update.

This produces 51 captures:

\[
t = 0,1,2,\ldots,50
\]

The dataset excludes iteration 0 and retains iterations 1 through 50.

For each iteration, the first 110 information-bit LLR values are extracted.

Sionna's default graph pruning results in an internal callback vector of length 350 for this configuration. This is expected. The first 110 positions correspond to the information-bit positions returned by the decoder.

The callback-to-final-output mapping was verified by checking that the final callback information-bit LLRs match the negative of the decoder's final soft information-bit output.

The mapping verification passed for all three final SNR experiments.

## 7. Residual-Error Estimate

The project derives a frame-level residual-error estimate from the bit-level posterior probabilities.

For each information bit:

\[
p_i = P(b_i=1)
\]

The probability assigned to the decoder's most likely bit decision is:

\[
p_{\text{correct},i} = \max(p_i,1-p_i)
\]

The raw frame-level residual-error estimate is:

\[
r_t =
1 -
\prod_{i=1}^{110}
\max(p_i(t),1-p_i(t))
\]

This is an independence-based approximation of the probability that at least one information bit is incorrect.

It is not assumed to be a perfectly calibrated probability of frame error.

For numerical stability, the implementation uses:

\[
r_t =
-\operatorname{expm1}
\left(
\sum_{i=1}^{110}
\log(p_{\text{correct},i}(t))
\right)
\]

This avoids numerical precision problems when multiplying many probabilities close to one.

The raw residual-error estimate is the main quantity intended for posterior calibration.

## 8. Ground-Truth Labels

The dataset contains two error labels.

### 8.1 `current_error`

This label indicates whether the decoded information bits at the current iteration contain at least one error.

\[
\text{current\_error}_t =
\begin{cases}
1 & \text{if any decoded information bit is incorrect}\\
0 & \text{otherwise}
\end{cases}
\]

The decoded information bits are compared against the originally transmitted information bits.

### 8.2 `final_failure`

This label indicates whether the frame is still incorrect after the maximum of 50 BP iterations.

\[
\text{final\_failure} =
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

The `true_bit_*` columns are retained for ground-truth comparison and evaluation. They must not be used as model input features.

## 9. Dataset Generation

The final dataset contains three SNR experiments.

| SNR (dB) | Frames | Successful frames | Failed frames | Measured BLER |
|---:|---:|---:|---:|---:|
| 2.25 | 10,000 | 9,667 | 333 | 3.330% |
| 2.50 | 10,000 | 9,854 | 146 | 1.460% |
| 2.75 | 10,000 | 9,945 | 55 | 0.550% |
| **Total** | **30,000** | **29,466** | **534** | — |

Each frame contributes 50 iteration-level observations.

Therefore:

\[
30,000 \times 50 = 1,500,000
\]

The final dataset contains 1.5 million observations.

### 9.1 Dataset Splitting

Frames are divided into training, validation and test sets separately for each SNR.

| Split | Frames per SNR | Percentage |
|---|---:|---:|
| Training | 7,000 | 70% |
| Validation | 1,500 | 15% |
| Test | 1,500 | 15% |

The split is performed at the frame level, not at the individual iteration-row level.

All 50 observations belonging to a frame remain in the same split. This prevents observations from the same transmitted frame from appearing in both training and test data.

## 10. Dataset Schema

Each CSV contains 228 columns.

| Column group | Number of columns | Description |
|---|---:|---|
| Metadata and labels | 8 | Frame identity, split, SNR, seed, iteration and labels |
| `llr_*` | 110 | Information-bit posterior LLRs |
| `true_bit_*` | 110 | Original transmitted information bits |
| **Total** | **228** | |

The eight metadata and label columns are:

| Column | Meaning |
|---|---|
| `frame_id` | Frame identifier within an SNR experiment |
| `split` | Training, validation or test |
| `snr_db` | Simulation SNR in dB |
| `seed` | Random seed associated with the SNR experiment |
| `iteration` | BP iteration number, from 1 to 50 |
| `residual_error` | Raw frame-level residual-error estimate |
| `current_error` | Actual error status at the current iteration |
| `final_failure` | Error status after 50 iterations |

The `llr_*` columns contain the information-bit LLR values captured at each iteration.

The `true_bit_*` columns contain the original transmitted information bits.

## 11. Repository Structure

```text
Calibrating-Belief-Propagation-Posteriors-for-Bidirectional-Stopping-of-5G-NR-LDPC-Decoding/
│
├── README.md
├── PERSON2_HANDOFF.md
│
├── src/
│   └── scripts/
│       ├── generate_final_dataset.py
│       ├── validate_final_dataset.py
│       └── inspect_final_dataset.py
│
└── data/
    ├── final_dataset/
    │   ├── train_snr_2.25dB.csv
    │   ├── validation_snr_2.25dB.csv
    │   ├── test_snr_2.25dB.csv
    │   ├── train_snr_2.50dB.csv
    │   ├── validation_snr_2.50dB.csv
    │   ├── test_snr_2.50dB.csv
    │   ├── train_snr_2.75dB.csv
    │   ├── validation_snr_2.75dB.csv
    │   ├── test_snr_2.75dB.csv
    │   ├── snr_summary.csv
    │   ├── experiment_metadata.json
    │   ├── frame_split_manifest.csv
    │   ├── split_failure_summary.csv
    │   └── iteration_statistics.csv
    │
    └── snr_refinement_results/
        └── Earlier pilot experiment outputs
```

## 12. Environment and Dependencies

The final simulation was run in WSL2 Ubuntu using Python 3.12.

The main environment components were:

- Sionna 2.2.0
- PyTorch 2.15.0+cpu
- NumPy
- SciPy
- Pandas
- Matplotlib
- Scikit-learn
- JupyterLab

The simulation was executed on CPU.

The Python environment used for the project was:

```bash
source ~/venvs/ldpc-sionna/bin/activate
```

## 13. Dataset Validation

The generated dataset was checked using `validate_final_dataset.py`.

The validation checks covered:

- Existence of all nine CSV files.
- Correct SNR values and split labels.
- Expected number of frames per split.
- Exactly 50 iterations per frame.
- Exactly 110 LLR and 110 transmitted-bit columns.
- Expected 228 columns per CSV.
- No duplicate frame/iteration pairs.
- Finite LLR and transmitted-bit values.
- Binary transmitted bits and error labels.
- Residual-error values within [0, 1].
- Consistency of `final_failure` across all iterations of a frame.
- Equality between `current_error` and `final_failure` at iteration 50.
- Correct frame-level split assignments.
- Decoder callback-to-output mapping verification.

The validation output reported that all checks passed for the nine CSV files and the split manifest.

The inspection script also generated:

- `split_failure_summary.csv`
- `iteration_statistics.csv`

These provide frame-level failure statistics and iteration-wise decoder behaviour.

## 14. Preliminary Observations

The measured BLER decreases as SNR increases:

- 3.330% at 2.25 dB.
- 1.460% at 2.50 dB.
- 0.550% at 2.75 dB.

The validation-set inspection also shows that the current decoding error rate decreases substantially during the initial BP iterations.

However, the median raw residual-error estimate becomes zero or numerically close to zero at several later iterations, while some frames still contain actual decoding errors.

This illustrates why raw posterior-derived estimates should not automatically be treated as calibrated frame-error probabilities.

These observations motivate the calibration stage; they do not, by themselves, establish that calibration or early stopping improves performance.

## 15. Planned Calibration and Stopping Stage

The next stage investigates calibration models that map raw residual-error estimates to empirical error risks.

One candidate model for current-iteration error risk is:

\[
q_{\mathrm{now}}(t)
=
\sigma(a\log r_t+b)
\]

A candidate model for final decoder failure risk may also incorporate the change in residual error over recent iterations:

\[
q_{\mathrm{fail}}(t)
=
\sigma(a\log r_t+b\Delta_k+c)
\]

where:

- \(\sigma(\cdot)\) is the logistic sigmoid function.
- \(r_t\) is the raw residual-error estimate.
- \(\Delta_k\) is a change in the log residual estimate over a selected iteration window.
- Model parameters are learned from training data and assessed using validation data.

The exact feature definition, clipping of residual estimates, calibration method and hyperparameters must be documented by the calibration stage.

The stopping stage will investigate:

- Early acceptance when estimated current error risk is sufficiently low.
- Early abandonment when estimated final failure risk is sufficiently high.
- Fixed-iteration and syndrome-based baselines.
- False acceptance and false abandonment.
- Decoding iteration savings.
- BLER and reliability trade-offs.

Thresholds should be selected using validation data. The test set should remain untouched until final evaluation.

## 16. Current Project Status

### Completed

- 5G NR BG2 encoder configuration.
- BPSK and AWGN simulation.
- BP decoder integration.
- Iteration-wise posterior LLR extraction.
- Decoder mapping verification.
- Residual-error estimate calculation.
- Ground-truth error-label generation.
- Three-SNR final dataset generation.
- Frame-level training, validation and test splitting.
- Dataset validation.
- Dataset inspection and summary generation.

### Remaining research work

- Posterior feature engineering.
- Calibration model training.
- Calibration quality evaluation.
- Channel-model mismatch experiments.
- Bidirectional stopping-rule implementation.
- Baseline comparisons.
- Final test-set evaluation.
- Final research analysis and reporting.

## 17. Important Scope and Limitations

1. The current final dataset represents matched AWGN simulation experiments at 2.25, 2.50 and 2.75 dB.
2. Channel-model mismatch experiments have not yet been included in this final dataset.
3. The residual-error estimate uses an independence approximation across information bits.
4. `final_failure` refers to the output after 50 iterations, not unlimited decoder convergence.
5. Zero syndrome does not guarantee that the decoded information bits are correct.
6. The current results establish the simulation and dataset pipeline; they do not yet demonstrate the effectiveness of calibrated bidirectional stopping.
7. The configuration is a valid Sionna 5G NR BG2 setup and should not be described as an exact reproduction of a different shortened-code configuration.

## 18. Reproducibility

Activate the project environment:

```bash
source ~/venvs/ldpc-sionna/bin/activate
```

From the repository root, generate the final dataset with:

```bash
python src/scripts/generate_final_dataset.py
```

Validate it with:

```bash
python src/scripts/validate_final_dataset.py
```

Inspect the generated results with:

```bash
python src/scripts/inspect_final_dataset.py
```

The generation script uses configured random seeds and records experiment metadata. Preserve the metadata and split manifest alongside the generated data.

---

## Acknowledgement

This repository contains the simulation, dataset-generation and validation work for the project on calibrated BP posterior estimates and bidirectional stopping of 5G NR LDPC decoding.