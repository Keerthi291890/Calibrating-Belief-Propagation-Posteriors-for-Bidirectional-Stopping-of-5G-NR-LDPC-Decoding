# Person 2 Handoff: Posterior Feature Engineering and Calibration

## 1. Your Responsibility

The LDPC simulation, decoder-trace extraction, dataset generation, and structural validation stages have been completed by Person 1.

Your responsibility is to investigate whether the raw BP-posterior-derived residual-error estimate can be calibrated to better represent actual decoding error risk.

The overall project investigates:

**Calibrating Belief-Propagation Posteriors for Bidirectional Stopping of 5G NR LDPC Decoding.**

The project responsibilities are:

- **Person 1:** LDPC simulation, encoder, channel, BP decoder, iteration traces, ground-truth labels, dataset generation and validation.
- **Person 2 (you):** Posterior feature engineering, calibration models and calibration evaluation.
- **Person 3:** Bidirectional stopping rules, baselines and stopping-performance evaluation.

Your main objective is to produce reliable calibrated risk estimates that Person 3 can use when designing and evaluating stopping decisions.

There are two distinct prediction targets:

1. Probability that the current decoded information frame contains an error.
2. Probability that the frame will still be incorrect after the full 50-iteration decoder run.

Keep these prediction problems separate throughout your implementation and evaluation.

---

## 2. What You Have Received

The final scalar-trace dataset contains four categories of data.

| Dataset category | Frames / condition records |
|---|---:|
| Training | 75,000 |
| Validation | 53,000 |
| Main test | 120,000 |
| Mismatch test | 210,000 |
| Mismatch refit | 60,000 |
| **Total condition records** | **518,000** |

Important: These are condition records, not 518,000 independent transmitted frames.

- The mismatch-test records reuse the corresponding main-test frames.
- The six mismatch-refit conditions use the same 10,000 underlying frames.
- The mismatch-refit dataset therefore contains 60,000 condition records representing six conditions over 10,000 paired frames.

### SNR values

The main experiments use four SNR values:

- 2.00 dB
- 2.25 dB
- 2.50 dB
- 2.75 dB

### Main train/validation/test frame counts

| SNR (dB) | Train | Validation | Test |
|---:|---:|---:|---:|
| 2.00 | 10,000 | 5,000 | 10,000 |
| 2.25 | 15,000 | 8,000 | 20,000 |
| 2.50 | 20,000 | 15,000 | 40,000 |
| 2.75 | 30,000 | 25,000 | 50,000 |
| **Total** | **75,000** | **53,000** | **120,000** |

Each frame contains 50 BP iteration observations, corresponding to iterations 1 through 50.

The dataset was generated using a valid 5G NR BG2 LDPC configuration.

| Parameter | Value |
|---|---|
| Information bits (K) | 110 |
| Transmitted codeword length (N) | 220 |
| Code rate | 0.5 |
| Base graph | BG2 |
| Lifting factor | 20 |
| Channel | AWGN |
| Modulation | BPSK |
| Decoder | Sionna LDPC5GDecoder |
| Maximum iterations | 50 |
| Decoder output | Soft output |
| Execution device | CPU |
| Sionna version | 2.2.0 |
| Parity-check matrix used for syndrome | 150 × 350 |

The decoder's LLR convention is:

\[
L_i=\log\frac{P(b_i=0)}{P(b_i=1)}
\]

Therefore:

\[
P(b_i=1)=\operatorname{sigmoid}(-L_i)
\]

A negative LLR indicates preference for bit 1, while a positive LLR indicates preference for bit 0.

---

## 3. Where the Dataset Is Located

The dataset is stored in:

```text
data/scalar_traces/
```

The directory structure is:

```text
data/scalar_traces/
│
├── dev/
│   ├── train_snr_2.00.npz
│   ├── train_snr_2.25.npz
│   ├── train_snr_2.50.npz
│   ├── train_snr_2.75.npz
│   │
│   ├── validation_snr_2.00.npz
│   ├── validation_snr_2.25.npz
│   ├── validation_snr_2.50.npz
│   └── validation_snr_2.75.npz
│
├── sealed_test/
│   ├── test_snr_2.00.npz
│   ├── test_snr_2.25.npz
│   ├── test_snr_2.50.npz
│   ├── test_snr_2.75.npz
│   │
│   ├── test_snr_2.00_delta-3.npz
│   ├── test_snr_2.00_delta-2.npz
│   ├── test_snr_2.00_delta-1.npz
│   ├── test_snr_2.00_delta+1.npz
│   ├── test_snr_2.00_delta+2.npz
│   ├── test_snr_2.00_delta+3.npz
│   │
│   ├── test_snr_2.25_delta-3.npz
│   ├── test_snr_2.25_delta-2.npz
│   ├── test_snr_2.25_delta-1.npz
│   ├── test_snr_2.25_delta+1.npz
│   ├── test_snr_2.25_delta+2.npz
│   ├── test_snr_2.25_delta+3.npz
│   │
│   ├── test_snr_2.50_delta-3.npz
│   ├── test_snr_2.50_delta-2.npz
│   ├── test_snr_2.50_delta-1.npz
│   ├── test_snr_2.50_delta+1.npz
│   ├── test_snr_2.50_delta+2.npz
│   ├── test_snr_2.50_delta+3.npz
│   │
│   ├── train_mismatch_snr_2.25_delta-3.npz
│   ├── train_mismatch_snr_2.25_delta-2.npz
│   ├── train_mismatch_snr_2.25_delta-1.npz
│   ├── train_mismatch_snr_2.25_delta+1.npz
│   ├── train_mismatch_snr_2.25_delta+2.npz
│   └── train_mismatch_snr_2.25_delta+3.npz
│
├── generation_config.json
└── bler_summary.csv
```

There are 36 NPZ files in total:

- 8 development files.
- 4 main test files.
- 18 mismatch-test files.
- 6 mismatch-refit files.

**Important directory clarification:** The six files beginning with `train_mismatch_` are training data for mismatch calibration/refitting, even though they currently reside in `sealed_test/`. Their filenames and split-code metadata identify their role. Do not treat them as held-out test data. Do not move them without coordinating with Person 1.

The `generation_config.json` file contains the generation configuration, decoder metadata, data schema, split sizes, seeds, mismatch specifications and per-file summaries.

The `bler_summary.csv` file contains the frame counts, failed-frame counts, BLER and runtime for each generated file.

---

## 4. NPZ File Structure and Feature Schema

Unlike the previous preliminary CSV dataset, the final dataset stores precomputed scalar traces.

Each NPZ file contains arrays representing frames and their 50 decoder iterations.

The expected keys are:

| Key | Shape | Description |
|---|---|---|
| `frame_id` | (B,) | Unique frame identifier |
| `snr` | (B,) | Actual channel SNR in dB |
| `seed_key` | (B,) | Batch-level seed identifier |
| `log_r` | (B, 50) | Log of raw residual-error estimate |
| `S` | (B, 50) | Sum of bit-level softplus terms |
| `min_abs` | (B, 50) | Minimum absolute information-bit LLR |
| `mean_abs` | (B, 50) | Mean absolute information-bit LLR |
| `mean_ent` | (B, 50) | Mean bit-level entropy-like quantity |
| `info_err` | (B, 50) | Number of incorrectly decoded information bits |
| `cur_err` | (B, 50) | Whether the current information frame contains an error |
| `stored_r` | (B, 50) | Raw residual-error estimate |
| `syndrome_weight` | (B, 50) | Number of unsatisfied parity checks |
| `final_failure` | (B,) | Whether the final 50-iteration output contains an error |

Here, B is the number of frames in that particular file.

The mismatch files additionally contain:

- `delta`: Assumed noise-variance mismatch in dB.
- `assumed_sigma2`: Assumed noise variance used by the decoder.

The main dataset does not store the original 110 LLR values or the 110 transmitted information bits. Their scalar-derived features and ground-truth labels have already been extracted.

**Do not expect the old `llr_*` or `true_bit_*` columns. They are not part of this final scalar-trace schema.**

### Loading a file

You can inspect a file using:

```python
import numpy as np

path = "../data/scalar_traces/dev/train_snr_2.00.npz"

data = np.load(path)

print(data.files)

for key in data.files:
    arr = data[key]
    print(key, arr.shape, arr.dtype)
```

For example:

```python
log_r = data["log_r"]
stored_r = data["stored_r"]
cur_err = data["cur_err"]
final_failure = data["final_failure"]
```

Each row of a `(B, 50)` array corresponds to one frame, and its columns correspond to iterations 1 through 50.

---

## 5. Understanding the Raw Residual-Error Estimate

The raw frame-level residual-error estimate is derived from the posterior probabilities of the 110 information bits.

For each bit, let:

\[
p_i=P(b_i=1)
\]

The probability assigned to the most likely bit decision is:

\[
p_{\mathrm{correct},i}=\max(p_i,1-p_i)
\]

The raw frame-level residual-error estimate is:

\[
r_t =
1-\prod_{i=1}^{110}
p_{\mathrm{correct},i}(t)
\]

This approximates the probability that at least one information bit is incorrect, under an independence approximation between bit decisions.

It is a raw estimate, not a guaranteed calibrated probability of frame error.

The stable implementation used during dataset generation is:

\[
S_t=\sum_{i=1}^{110}\operatorname{softplus}(-|L_i(t)|)
\]

\[
r_t=1-\exp(-S_t)
\]

The saved arrays include:

- `S`: The sum of softplus terms.
- `log_r`: The logarithm of the raw residual-error estimate.
- `stored_r`: The raw residual-error estimate itself.

You do not need to recompute this quantity unless you want to independently verify the feature extraction.

### Numerical handling

The saved `log_r` is calculated using numerically stable operations and a very small numerical guard for the logarithm.

A separate model-input guard is defined as:

```python
LOG_R_FLOOR = -60.0
```

This floor is applied later during feature-table construction, not during the saved trace generation.

When constructing model inputs, use a consistent numerical treatment. Do not overwrite the original `log_r` or `stored_r` values.

---

## 6. Ground-Truth Targets

There are two separate prediction problems.

### 6.1 Current-iteration error prediction

The target is:

```text
cur_err
```

It indicates whether the currently decoded information bits contain at least one error.

Mathematically:

\[
y_{\mathrm{now}}(t)=
\begin{cases}
1 & \text{if at least one information bit is incorrect}\\
0 & \text{otherwise}
\end{cases}
\]

The calibration model should estimate:

\[
q_{\mathrm{now}}(t)
\approx
P(\text{current error}=1\mid
\text{decoder information at iteration }t)
\]

This answers:

**How likely is the current decoded frame to contain an error?**

### 6.2 Final-decoder failure prediction

The target is:

```text
final_failure
```

It indicates whether the frame is incorrect after the full 50 BP iterations.

Mathematically:

\[
y_{\mathrm{fail}}=
\begin{cases}
1 & \text{if the final information frame is incorrect}\\
0 & \text{otherwise}
\end{cases}
\]

The calibration model should estimate:

\[
q_{\mathrm{fail}}(t)
\approx
P(\text{final failure}=1\mid
\text{decoder information at iteration }t)
\]

This answers:

**Given the decoder state now, how likely is this frame to remain incorrect after the full 50-iteration run?**

This estimate is particularly relevant to early abandonment.

Important: `final_failure` is defined using the decoder output at iteration 50. It does not represent failure after unlimited decoding.

### Label availability

`cur_err` has shape `(B, 50)`, so it provides a different target for each iteration.

`final_failure` has shape `(B,)`, so the same final outcome can be associated with each of the 50 observations belonging to that frame.

Do not confuse these targets or use one as an input to predict the other.

---

## 7. Candidate Calibration Models

The following are candidate approaches to investigate, not claims that these are the only valid calibration methods.

### 7.1 Current-error calibration

A simple logistic calibration model can be investigated:

\[
q_{\mathrm{now}}(t)=
\sigma(a\log r_t+b)
\]

where:

- \(r_t\) is the raw residual-error estimate.
- \(a,b\) are learned parameters.
- \(\sigma\) is the logistic sigmoid function.

The objective is to learn a mapping between the raw residual estimate and the observed probability of current decoding error.

### 7.2 Final-failure calibration

For predicting final decoder failure, investigate whether the current residual estimate and its trajectory provide additional predictive information.

One candidate model is:

\[
q_{\mathrm{fail}}(t)=
\sigma(a\log r_t+b\Delta_k+c)
\]

where:

\[
\Delta_k=\log(r_t)-\log(r_{t-k})
\]

Here, \(k\) is a selected iteration window.

Two frames can have similar residual-error estimates at the current iteration but different trajectories. One may be improving rapidly, while another may have stopped improving.

The trajectory may therefore provide additional information about eventual decoding success.

Use validation data to select the iteration window and compare models.

### 7.3 Additional candidate features

The dataset already contains several useful scalar features:

| Feature | Meaning |
|---|---|
| `log_r` | Log raw residual-error estimate |
| `stored_r` | Raw residual-error estimate |
| `S` | Sum of softplus terms |
| `min_abs` | Minimum absolute information-bit LLR |
| `mean_abs` | Mean absolute information-bit LLR |
| `mean_ent` | Mean bit-level entropy-like quantity |
| `syndrome_weight` | Number of unsatisfied parity checks |
| `iteration` | Current iteration, derived from the column position |
| `snr` | Actual channel SNR |

You can also derive:

- Change in `log_r` over a selected iteration window.
- Residual-error trend.
- Change in `mean_abs`.
- Change in `syndrome_weight`.
- Other trajectory-based features that can be computed using information available up to the current iteration.

Start with simple models and add features only when they demonstrate useful improvements.

**Important:** `info_err` is the number of incorrectly decoded information bits, and `cur_err` is derived from it. These are ground-truth quantities. Do not use `info_err` or `cur_err` as predictive features when evaluating realistic online calibration. Doing so would leak the answer into the model.

Likewise, do not use `final_failure` as a predictive feature.

---

## 8. Train, Validation and Test Protocol

The main dataset is already separated by frame.

| SNR (dB) | Train | Validation | Test |
|---:|---:|---:|---:|
| 2.00 | 10,000 | 5,000 | 10,000 |
| 2.25 | 15,000 | 8,000 | 20,000 |
| 2.50 | 20,000 | 15,000 | 40,000 |
| 2.75 | 30,000 | 25,000 | 50,000 |

### Training data

Use:

```text
data/scalar_traces/dev/train_snr_*.npz
```

Use these files to:

- Construct feature tables.
- Fit calibration models.
- Compare feature combinations.
- Learn model parameters.

### Validation data

Use:

```text
data/scalar_traces/dev/validation_snr_*.npz
```

Use these files to:

- Select model variants.
- Select iteration windows.
- Tune regularisation and hyperparameters.
- Compare calibration methods.
- Select any thresholds needed by Person 3.

### Main test data

The main test files are:

```text
data/scalar_traces/sealed_test/test_snr_*.npz
```

Keep these untouched until the model and preprocessing decisions are fixed.

Use them only for final evaluation.

**Do not use the main test data for model fitting, feature selection, hyperparameter tuning or threshold selection.**

### Mismatch-refit training data

The six files beginning with:

```text
train_mismatch_snr_2.25_delta
```

are additional training data for mismatch calibration/refitting.

They contain six mismatch conditions:

\[
\Delta\in\{-3,-2,-1,+1,+2,+3\}\text{ dB}
\]

Each condition contains 10,000 frames.

All six files use exactly the same frame IDs. This was explicitly verified.

These records are intended for mismatch-refit experiments, not final held-out mismatch evaluation.

The files are currently stored in `sealed_test/` for organizational convenience. Their actual role is training data. Keep their use separate from the main held-out test files.

### Frame-level leakage prevention

All 50 observations belonging to a frame must remain in the same partition.

The frame IDs use different split-code ranges:

| Split | Split code |
|---|---:|
| Train | 0 |
| Validation | 1 |
| Test | 2 |
| Mismatch refit | 3 |

The main train, validation and test frame IDs were checked for separation at every SNR.

Mismatch-test frame IDs intentionally match their corresponding main-test frame IDs. This allows paired comparison of matched and mismatched decoding conditions.

Do not treat mismatch-test records as independent test frames or add them to the training set.

Rows from the same frame are correlated across iterations. Do not treat 50 observations from one frame as 50 independent transmitted frames when estimating uncertainty. Where confidence intervals are required, use frame-level resampling.

---

## 9. Channel-Model Mismatch Experiments

The project also investigates how calibration behaves when the decoder's assumed noise variance differs from the true channel noise variance.

The mismatch is applied to the assumed noise variance:

\[
\sigma^2_{\mathrm{assumed}}
=
\sigma^2_{\mathrm{true}}\cdot10^{\Delta/10}
\]

where:

\[
\Delta\in\{-3,-2,-1,+1,+2,+3\}\text{ dB}
\]

A positive delta means the decoder assumes a higher noise variance than the true variance. A negative delta means it assumes a lower noise variance.

The transmitted bits and channel noise are reused from the corresponding main-test frames. Only the assumed noise variance used to calculate decoder LLRs changes.

### Mismatch test coverage

| Actual SNR | Frames per delta | Number of deltas |
|---:|---:|---:|
| 2.00 dB | 5,000 | 6 |
| 2.25 dB | 10,000 | 6 |
| 2.50 dB | 20,000 | 6 |
| 2.75 dB | Not generated | 0 |

Total mismatch-test records: 210,000.

The mismatch-test files are intended for evaluating how the calibration model behaves under channel-model mismatch.

They must remain held out from model fitting and threshold tuning.

### Mismatch-refit coverage

The optional mismatch-refit dataset contains:

- Actual SNR: 2.25 dB.
- 10,000 underlying frames per delta.
- Six nonzero mismatch deltas.
- 60,000 condition records in total.

Use this dataset only for experiments involving mismatch-aware calibration or model refitting. Keep its results separate from the held-out mismatch-test evaluation.

The existence of mismatch-refit training data does not mean that the model has already been shown to generalize to unseen mismatch conditions.

---

## 10. Evaluation Metrics

Evaluate both predictive performance and probability calibration.

### 10.1 Calibration metrics

Recommended metrics:

- Brier score.
- Log loss.
- Expected Calibration Error (ECE).
- Reliability diagrams.
- Calibration intercept and slope.

These help determine whether predicted probabilities agree with observed error frequencies.

### 10.2 Discrimination metrics

Also report:

- ROC-AUC.
- Precision-recall AUC.
- Precision and recall at selected operating points.

These measure how well the model distinguishes erroneous and correct frames. They do not, by themselves, establish good probability calibration.

### 10.3 Stratified evaluation

Report main matched-channel results separately for:

- 2.00 dB.
- 2.25 dB.
- 2.50 dB.
- 2.75 dB.

Also consider reporting results by iteration or iteration range.

Report mismatch results separately by:

- Actual SNR.
- Assumed-noise mismatch delta.
- Iteration or iteration range.

The higher-SNR datasets may contain fewer failed frames. Interpret metrics with the number of positive examples and their uncertainty in mind.

For final reporting, clearly distinguish:

1. Matched-channel validation.
2. Matched-channel held-out test.
3. Mismatch-refit training experiments.
4. Held-out mismatch testing.

---

## 11. Expected Investigation

The central motivation is that the raw residual-error estimate is based on an independence approximation and may not correspond accurately to the actual frame-error probability.

In particular, a small or numerically negligible raw estimate does not necessarily guarantee that the decoded frame is correct.

Your investigation should establish:

1. How well does the raw residual-error estimate predict current decoding errors?
2. Does a calibration mapping improve probability reliability?
3. Can iteration history improve prediction of final decoder failure?
4. Does calibration behavior change across SNR values?
5. How does calibration degrade when the assumed noise variance is mismatched?
6. Can mismatch-aware refitting improve calibration under mismatch conditions?
7. Does the calibrated model remain reliable on held-out data?

Do not assume calibration will improve every metric or every SNR. Report the observed results, including cases where it does not improve.

---

## 12. Expected Outputs from Person 2

Please produce:

1. A feature-engineering script or notebook for loading the NPZ files.
2. A current-error calibration model.
3. A final-failure calibration model.
4. A comparison of raw versus calibrated estimates.
5. Calibration plots and reliability diagrams.
6. Evaluation metrics on training and validation data.
7. Final matched-channel test metrics after the model is fixed.
8. Held-out mismatch evaluation across the available SNR and delta conditions.
9. A separate analysis of mismatch-aware refitting, if performed.
10. A saved model and preprocessing configuration.
11. A short report explaining the selected models, results and limitations.

The report must clearly distinguish the model predicting current-iteration error from the model predicting final decoder failure.

The saved model interface should expose:

- `q_now`: Estimated probability of a current decoding error.
- `q_fail`: Estimated probability of final decoder failure.

The model should also document the required input features, preprocessing steps, iteration handling and supported operating conditions.

---

## 13. How Your Results Connect to Person 3

Person 3 will use the calibrated outputs to investigate bidirectional stopping decisions.

The intended interface is:

- `q_now`: Estimated probability that the current decoded information frame contains an error.
- `q_fail`: Estimated probability that the frame will remain incorrect after 50 iterations.

These estimates can support two separate decisions.

### Early acceptance

Accept a decoded frame when the estimated current-error risk is sufficiently low, subject to the project's selected reliability constraints and evaluation.

### Early abandonment

Abandon further decoding when the estimated probability of final decoder failure is sufficiently high, subject to the project's selected constraints.

Person 3 is responsible for implementing and evaluating the stopping rules.

Your responsibility is to produce and validate the calibrated risk estimates.

A well-calibrated model does not automatically prove that a stopping rule is safe, efficient or beneficial. Those claims must be evaluated separately by Person 3.

---

## 14. Important Limitations and Notes

1. The main channel experiments use AWGN and BPSK.
2. The final dataset stores scalar iteration traces rather than the original per-bit LLR arrays and transmitted information bits.
3. The raw residual-error estimate uses an independence approximation between bit decisions.
4. The final-failure target is defined using the 50-iteration decoder output.
5. The mismatch experiments vary the assumed noise variance; they do not represent every possible real-world channel impairment.
6. Mismatch-test records reuse main-test frames and must not be treated as independent additional test frames.
7. Mismatch-refit files are training data, despite their current location in the `sealed_test/` directory.
8. A good ROC-AUC does not automatically imply good probability calibration.
9. A calibrated probability estimate does not automatically prove that a stopping rule is safe or efficient.
10. The main test and mismatch test datasets must not be used for model selection or threshold tuning.
11. `info_err`, `cur_err` and `final_failure` are ground-truth information and must not leak into predictive features.
12. Preserve frame-level grouping when splitting, resampling or calculating confidence intervals.

---

## 15. Dataset Validation Status

Person 1 completed a final structural audit of the generated dataset.

The audit confirmed:

- All 36 expected NPZ files are present.
- No unexpected NPZ files were detected.
- All expected frame counts and trace shapes are correct.
- Metadata and SNR values are consistent.
- Main train, validation and test frame IDs are separated.
- Mismatch-test frame IDs align with their corresponding main-test frames.
- Mismatch-refit files use the same frame IDs across all six delta conditions.
- Mismatch deltas and assumed noise variances are correct.
- Decoder metadata specifies 50 iterations.
- Sionna version is recorded as 2.2.0.
- The BLER summary contains all 36 generated-file records.

**Final audit result: 277 checks passed, 0 failed.**

The dataset has passed its structural and metadata validation. This does not replace Person 2's statistical validation of the calibration models.

---

## 16. Handoff Summary

The simulation, scalar-trace generation, mismatch generation and dataset-validation stages are complete.

Your work begins with the precomputed scalar traces in:

```text
data/scalar_traces/
```

Start by loading the NPZ files and understanding the available arrays before implementing feature engineering.

The central research question for your stage is:

**Can we transform the raw BP-posterior-derived residual-error estimate into a more reliable estimate of current decoding error and eventual 50-iteration decoder failure, and how robust are these estimates under channel-model mismatch?**

Your results will provide the calibrated risk estimates needed for the bidirectional stopping stage.
