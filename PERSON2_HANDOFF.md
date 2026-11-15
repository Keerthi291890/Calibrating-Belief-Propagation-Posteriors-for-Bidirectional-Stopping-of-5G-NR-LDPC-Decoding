# Person 2 Handoff: Posterior Feature Engineering and Calibration

## 1. Your Responsibility

The simulation and dataset-generation stage has been completed.

Your responsibility is to investigate whether the raw BP-posterior-derived residual-error estimate can be calibrated to better represent actual decoding error risk.

The overall project investigates calibrated posterior estimates for bidirectional stopping of 5G NR LDPC decoding.

Your work forms the calibration stage between:

- **Person 1:** LDPC simulation, encoder, channel, BP decoder, iteration traces and dataset generation.
- **Person 2 (you):** Posterior feature engineering, calibration models and calibration evaluation.
- **Person 3:** Bidirectional stopping rules, baselines and stopping-performance evaluation.

Your main objective is to produce reliable calibrated estimates that Person 3 can use when designing stopping decisions.

## 2. What You Have Received

The final dataset contains:

- 30,000 simulated frames.
- 1,500,000 iteration-level observations.
- Three SNR values: 2.25, 2.50 and 2.75 dB.
- 50 BP iterations per frame.
- 110 information-bit LLR values per observation.
- Ground-truth current-error and final-failure labels.

The data was generated using a valid 5G NR BG2 LDPC configuration with:

| Parameter | Value |
|---|---|
| Information bits | 110 |
| Codeword length | 220 |
| Code rate | 0.5 |
| Lifting factor | 20 |
| Channel | AWGN |
| Modulation | BPSK |
| Decoder | Belief Propagation |
| Maximum iterations | 50 |

## 3. Where the Dataset Is Located

The dataset is inside the repository:

```text
data/final_dataset/
```

The nine main files are:

```text
data/final_dataset/
│
├── train_snr_2.25dB.csv
├── validation_snr_2.25dB.csv
├── test_snr_2.25dB.csv
│
├── train_snr_2.50dB.csv
├── validation_snr_2.50dB.csv
├── test_snr_2.50dB.csv
│
├── train_snr_2.75dB.csv
├── validation_snr_2.75dB.csv
├── test_snr_2.75dB.csv
│
├── snr_summary.csv
├── experiment_metadata.json
├── frame_split_manifest.csv
├── split_failure_summary.csv
└── iteration_statistics.csv
```

Each CSV contains 228 columns and 50 observations per frame.

## 4. Dataset Column Explanation

Every row represents one frame at one BP iteration.

### 4.1 Metadata and labels

| Column | Description | Use |
|---|---|---|
| `frame_id` | Frame identifier within an SNR experiment | Grouping and leakage prevention |
| `split` | Train, validation or test | Dataset partition |
| `snr_db` | SNR value in dB | Analysis or optional model feature |
| `seed` | Experiment random seed | Reproducibility metadata |
| `iteration` | BP iteration number, 1–50 | Model feature |
| `residual_error` | Raw frame-level posterior-derived error estimate | Main calibration input |
| `current_error` | Whether the current decoded information bits contain any error | Current-error target |
| `final_failure` | Whether the frame is incorrect after 50 iterations | Final-failure target |

### 4.2 LLR columns

The dataset contains 110 columns with names beginning with:

```text
llr_
```

These are the information-bit posterior LLR values captured from the decoder at the corresponding iteration.

The callback convention is:

\[
L_i = \log\frac{P(b_i=0)}{P(b_i=1)}
\]

Therefore:

\[
P(b_i=1)=\operatorname{sigmoid}(-L_i)
\]

A positive LLR indicates preference for bit 0, while a negative LLR indicates preference for bit 1.

### 4.3 Ground-truth bit columns

The dataset contains 110 columns beginning with:

```text
true_bit_
```

These contain the originally transmitted information bits.

They are retained for evaluating decoded results and constructing ground-truth labels.

**Do not use these columns as calibration model inputs.** Doing so would introduce direct information about the correct transmitted message into the model.

## 5. The Raw Residual-Error Estimate

The raw frame-level residual-error estimate was calculated from the posterior probabilities of the 110 information bits.

First, for each bit:

\[
p_i=P(b_i=1)
\]

The probability assigned to the most likely bit decision is:

\[
p_{\mathrm{correct},i}=\max(p_i,1-p_i)
\]

The frame-level residual-error estimate is:

\[
r_t =
1-\prod_{i=1}^{110}
p_{\mathrm{correct},i}(t)
\]

This approximates the probability that at least one information bit is incorrect, assuming independence between bit decisions.

It is a raw estimate, not a guaranteed calibrated probability of frame error.

The stable implementation used during data generation is equivalent to:

\[
r_t =
-\operatorname{expm1}
\left(
\sum_i \log(p_{\mathrm{correct},i}(t))
\right)
\]

You do not need to recompute this quantity unless you want to independently verify the feature.

The `residual_error` column already contains it.

## 6. Ground-Truth Targets

There are two different prediction problems in this dataset. Keep them separate.

### 6.1 Current-iteration error prediction

Target:

```text
current_error
```

This indicates whether the decoded information bits contain any error at the current iteration.

\[
y_{\mathrm{now}}(t)=\text{current\_error}_t
\]

The calibration model should estimate:

\[
q_{\mathrm{now}}(t)
\approx
P(\text{current\_error}_t=1\mid \text{available decoder information})
\]

This answers:

**How likely is the current decoded frame to contain an error?**

### 6.2 Final-decoder failure prediction

Target:

```text
final_failure
```

This indicates whether the frame is incorrect after 50 BP iterations.

\[
y_{\mathrm{fail}}=\text{final\_failure}
\]

The calibration model should estimate:

\[
q_{\mathrm{fail}}(t)
\approx
P(\text{final\_failure}=1\mid \text{information available at iteration }t)
\]

This answers:

**Given the decoder's state now, how likely is this frame to remain incorrect after the full 50-iteration run?**

This second estimate is particularly relevant to early abandonment.

Important: `final_failure` is defined using the 50-iteration output. It does not represent failure after unlimited decoding.

## 7. Candidate Calibration Models

The following are candidate models to investigate. They are proposed modelling choices for this project, not claims that these are the only valid calibration methods.

### 7.1 Current-error calibration

A simple logistic model can be investigated:

\[
q_{\mathrm{now}}(t)
=
\sigma(a\log r_t+b)
\]

where:

- \(r_t\) is the raw residual-error estimate.
- \(a,b\) are learned parameters.
- \(\sigma\) is the logistic sigmoid function.

This learns a mapping between the raw residual estimate and the observed probability of a current decoding error.

### 7.2 Final-failure calibration

For predicting final decoder failure, the model can also use the change in residual error over recent iterations:

\[
q_{\mathrm{fail}}(t)
=
\sigma(a\log r_t+b\Delta_k+c)
\]

where \(\Delta_k\) represents the change in log residual error over a selected iteration window.

One possible definition is:

\[
\Delta_k =
\log(r_t)-\log(r_{t-k})
\]

The value of \(k\) should be selected using validation data.

The idea is that two frames can have similar residual-error estimates at the current iteration but different trajectories. One may be improving rapidly, while another may have stopped improving.

The trajectory may therefore provide additional information about eventual decoding success.

### 7.3 Numerical handling

Some residual-error estimates are zero or extremely close to zero.

Before taking logarithms, clip the values:

\[
r_t^{\mathrm{clip}}
=
\max(r_t,\epsilon)
\]

where \(\epsilon\) is a small positive numerical constant.

Document the chosen value of \(\epsilon\) and use the same preprocessing consistently during training, validation and testing.

## 8. Suggested Feature Engineering

Start with simple features before introducing complex models.

| Feature | Meaning |
|---|---|
| `residual_error` | Raw residual-error estimate |
| `log_residual` | Log of clipped residual-error estimate |
| `iteration` | Current decoder iteration |
| `snr_db` | Channel SNR |
| `delta_log_residual` | Change in log residual over a selected window |
| `residual_trend` | Whether residual error is decreasing or increasing |
| `mean_abs_llr` | Mean absolute information-bit LLR |
| `min_abs_llr` | Minimum absolute information-bit LLR |
| `std_abs_llr` | Standard deviation of absolute information-bit LLRs |

The LLR-derived features can be computed from the 110 `llr_*` columns.

These are candidate features. Evaluate whether each adds useful information rather than assuming that more features always improve calibration.

Do not include:

- `true_bit_*` columns.
- `current_error` as an input when predicting `current_error`.
- `final_failure` as an input when predicting `final_failure`.
- Frame identifiers or random seeds as predictive features.

## 9. Train, Validation and Test Protocol

The dataset is already split by frame.

For each SNR:

| Split | Number of frames |
|---|---:|
| Train | 7,000 |
| Validation | 1,500 |
| Test | 1,500 |

Use the splits as follows:

### Training set

Use this to:

- Engineer candidate features.
- Fit logistic calibration models.
- Learn model parameters.
- Compare candidate feature combinations.

### Validation set

Use this to:

- Select model variants.
- Select the iteration window \(k\).
- Tune regularisation and other hyperparameters.
- Choose any calibration or risk thresholds required by the next stage.

### Test set

Keep this untouched until the final model and preprocessing decisions are fixed.

Use it only for final reporting.

**Important:** All 50 rows belonging to a frame are in the same split. Preserve this property in any additional experiments or resampling.

Rows from the same frame are correlated across iterations. Do not treat 50 observations from one frame as 50 independent transmitted frames when estimating uncertainty. Where confidence intervals are needed, use frame-level resampling.

## 10. Evaluation Metrics

Evaluate both predictive performance and calibration quality.

### 10.1 Calibration metrics

Recommended metrics include:

- Brier score.
- Log loss.
- Expected Calibration Error (ECE).
- Reliability diagrams.
- Calibration intercept and slope.

These help assess whether predicted probabilities agree with observed error frequencies.

### 10.2 Discrimination metrics

Also report:

- ROC-AUC.
- Precision-recall AUC.
- Precision and recall at selected operating points.

These measure how well the model distinguishes erroneous from correct frames. They do not, by themselves, establish good probability calibration.

### 10.3 Stratified evaluation

Report results separately for:

- 2.25 dB.
- 2.50 dB.
- 2.75 dB.

Also consider reporting by iteration or iteration range.

The 2.75 dB experiment has fewer failures than the lower-SNR experiments. In particular, its validation and test sets contain relatively few failed frames. Interpret metrics at that SNR with this limited sample size in mind.

## 11. Important Observation from the Initial Inspection

The raw residual-error estimate becomes zero or numerically close to zero for many frames at later iterations.

However, some of these frames still have actual decoding errors.

For example, the validation-set inspection showed that at 2.25 dB:

| Iteration | Current error rate | Mean residual estimate | Median residual estimate |
|---:|---:|---:|---:|
| 10 | 14.13% | 0.151009 | approximately 0 |
| 20 | 5.47% | 0.051737 | approximately 0 |
| 30 | 4.13% | 0.040960 | approximately 0 |
| 50 | 3.53% | 0.034138 | approximately 0 |

This is an important motivation for calibration.

The median being close to zero does not mean all frames have zero residual error. A smaller number of frames with larger residual estimates can raise the mean.

The goal is to investigate how well the raw estimate corresponds to empirical error risk and whether a learned mapping improves that correspondence.

## 12. Expected Outputs from Person 2

Please produce the following:

1. A feature-engineering script or notebook.
2. A current-error calibration model.
3. A final-failure calibration model.
4. A comparison of raw versus calibrated estimates.
5. Calibration plots and reliability diagrams.
6. Evaluation metrics on training and validation data.
7. Final test-set metrics after the model is fixed.
8. A saved model and preprocessing configuration.
9. A short report explaining the selected model and its limitations.

The report should clearly state which model predicts current-iteration error and which model predicts final decoder failure.

## 13. How Your Results Connect to Person 3

Person 3 will use the calibrated outputs to investigate stopping decisions.

The intended interface is:

- `q_now`: estimated probability of a current decoding error.
- `q_fail`: estimated probability of final decoder failure.

These can support two separate decisions:

### Early acceptance

Accept a decoded frame when the estimated current-error risk is sufficiently low, subject to the project's selected reliability constraints.

### Early abandonment

Abandon further decoding when the estimated probability of final failure is sufficiently high, subject to the project's selected constraints.

Person 3 is responsible for implementing and evaluating the stopping rules. Your responsibility is to produce and validate the calibrated risk estimates.

## 14. Important Limitations and Notes

1. The current dataset contains matched AWGN experiments only.
2. Channel-model mismatch experiments are not yet included in these final CSV files.
3. The raw residual-error estimate is based on an independence approximation.
4. The final-failure target is defined after 50 iterations.
5. A good ROC-AUC does not automatically imply good calibration.
6. A calibrated probability estimate does not automatically prove that a stopping rule is safe or efficient.
7. The test set must not be used for model selection or threshold tuning.
8. The transmitted-bit columns are for evaluation only and must not leak into model features.

## 15. Handoff Summary

The simulation and dataset-generation pipeline is complete and has passed its structural validation checks.

Your task begins with the existing `residual_error`, `current_error` and `final_failure` columns.

The central question for your stage is:

**Can we transform the raw residual-error estimate into a more reliable estimate of current decoding error and eventual 50-iteration decoder failure?**

Your results will provide the calibrated risk estimates needed for the bidirectional stopping stage.