# Person 3: stopping and evaluation summary

Generated from Person 2's frozen CSV outputs. No model fitting, threshold
selection, decoding, or sealed-test rerun is performed by this script.

## Matched test: 10% excess-event budget

Iteration savings are recomputed from mean iterations and the B2 baseline.
The recorded saving is checked against the recomputation.

| SNR (dB) | Method | Mean iterations | Iteration saving vs B2 | Excess / budget | Test budget met | Total-op saving vs B2 (cx=10) |
|---:|---|---:|---:|---:|:---:|---:|
| 2.00 | B2 syndrome-zero | 11.5842 | 0.00% | 0 / 66.7 | Yes | 0.00% |
| 2.00 | B5 syndrome + cap | 10.5333 | 9.07% | 80 / 66.7 | No | 9.07% |
| 2.00 | P-cal both | 10.1251 | 12.60% | 69 / 66.7 | No | 6.11% |
| 2.25 | B2 syndrome-zero | 9.1788 | 0.00% | 0 / 62.5 | Yes | 0.00% |
| 2.25 | B5 syndrome + cap | 8.8628 | 3.44% | 43 / 62.5 | Yes | 3.44% |
| 2.25 | P-cal both | 8.6428 | 5.84% | 35 / 62.5 | Yes | -0.99% |
| 2.50 | B2 syndrome-zero | 7.6744 | 0.00% | 0 / 61.8 | Yes | 0.00% |
| 2.50 | B5 syndrome + cap | 7.5953 | 1.03% | 35 / 61.8 | Yes | 1.03% |
| 2.50 | P-cal both | 7.4185 | 3.33% | 57 / 61.8 | Yes | -3.48% |
| 2.75 | B2 syndrome-zero | 6.5308 | 0.00% | 0 / 28.1 | Yes | 0.00% |
| 2.75 | B5 syndrome + cap | 6.4709 | 0.92% | 38 / 28.1 | No | 0.92% |
| 2.75 | P-cal both | 6.4194 | 1.71% | 10 / 28.1 | Yes | -5.10% |

- P-cal both test-budget misses: **1/4**.
- B5 test-budget misses: **2/4**.
- Report realised excess events alongside savings; validation-tuned thresholds do not guarantee test-budget compliance.
- Iteration savings and total-operation savings are different objectives. The thresholds were selected to minimise mean iterations, not total operations.

## Mismatch stopping

| SNR (dB) | Noise variance delta (dB) | Calibrators | Method | Iteration saving vs same-delta B2 | Excess / budget | Budget met | Total-op saving vs B2 |
|---:|---:|---|---|---:|---:|:---:|---:|
| 2.00 | -3 | matched | P-cal both | 19.17% | 86/58.3 | No | 13.01% |
| 2.00 | -2 | matched | P-cal both | 16.68% | 70/47.4 | No | 10.39% |
| 2.00 | -1 | matched | P-cal both | 14.52% | 43/39.0 | No | 8.13% |
| 2.00 | +0 | matched | P-cal both | 12.37% | 29/32.6 | Yes | 5.87% |
| 2.00 | +1 | matched | P-cal both | 10.06% | 10/31.7 | Yes | 3.38% |
| 2.00 | +2 | matched | P-cal both | 10.34% | 19/45.1 | Yes | 3.54% |
| 2.00 | +3 | matched | P-cal both | 18.37% | 78/132.6 | Yes | 11.94% |
| 2.25 | -3 | matched | P-cal both | 10.69% | 74/62.6 | No | 4.01% |
| 2.25 | -3 | refit | P-cal both | 9.24% | 31/62.6 | Yes | 2.49% |
| 2.25 | -2 | matched | P-cal both | 8.86% | 40/51.4 | Yes | 2.11% |
| 2.25 | -2 | refit | P-cal both | 7.01% | 15/51.4 | Yes | 0.14% |
| 2.25 | -1 | matched | P-cal both | 7.55% | 34/41.0 | Yes | 0.78% |
| 2.25 | -1 | refit | P-cal both | 6.62% | 24/41.0 | Yes | -0.22% |
| 2.25 | +0 | matched | P-cal both | 6.13% | 17/32.8 | Yes | -0.69% |
| 2.25 | +1 | matched | P-cal both | 5.00% | 10/31.8 | Yes | -1.90% |
| 2.25 | +1 | refit | P-cal both | 7.01% | 17/31.8 | Yes | 0.28% |
| 2.25 | +2 | matched | P-cal both | 4.10% | 3/39.5 | Yes | -2.98% |
| 2.25 | +2 | refit | P-cal both | 8.51% | 25/39.5 | Yes | 1.77% |
| 2.25 | +3 | matched | P-cal both | 7.56% | 25/115.2 | Yes | 0.43% |
| 2.25 | +3 | refit | P-cal both | 14.07% | 48/115.2 | Yes | 7.44% |
| 2.50 | -3 | matched | P-cal both | 6.84% | 84/62.0 | No | 0.08% |
| 2.50 | -2 | matched | P-cal both | 5.30% | 39/49.7 | Yes | -1.50% |
| 2.50 | -1 | matched | P-cal both | 4.17% | 54/36.8 | No | -2.65% |
| 2.50 | +0 | matched | P-cal both | 3.28% | 25/30.8 | Yes | -3.54% |
| 2.50 | +1 | matched | P-cal both | 2.44% | 16/26.8 | Yes | -4.43% |
| 2.50 | +2 | matched | P-cal both | 2.02% | 5/31.7 | Yes | -4.99% |
| 2.50 | +3 | matched | P-cal both | 3.76% | 24/82.7 | Yes | -3.44% |

- Matched-calibrator P-cal both mismatch budget misses: **6/18**.
- This count excludes delta = 0 (the matched-no-mismatch reference) and excludes refit rows; refit results are a separate analysis.
- Mismatch test frames are paired with the matched test frames; they are not independent samples.
- Refitting results are reported only where the frozen CSV contains a refit row.

## Interpretation constraints

- Abandonment is a NACK and counts as a failure in net BLER.
- False accept and false abandon are excess events only when the full 50-iteration decision would have been correct.
- Operation counts are approximate scalar-operation estimates, not measured hardware instructions, latency, or energy.
- The experiment covers one 5G NR BG2 code configuration, BPSK/AWGN, SNR 2.00–2.75 dB, and assumed noise-variance mismatch.
- Reliability statements are empirical; no certified guarantee is established.

## Source files

- `results/person2_v3/final_stopping_test.csv`
- `results/person2_v3/final_stopping_mismatch.csv`
