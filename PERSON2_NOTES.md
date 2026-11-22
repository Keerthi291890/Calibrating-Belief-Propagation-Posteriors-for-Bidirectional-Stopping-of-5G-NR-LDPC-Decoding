# Person 2 notes (updated 17 Nov 2026)

## 1. Definitions to freeze (do not change after test data exists)
- r_t = 1 - prod_i max(p_i, 1-p_i) over the 110 info bits, from LLRs in float64: r = -expm1(-S), S = sum log1p(exp(-|L_i|)).
  NOTE: the decoder clips LLRs at |L| = 20, so every converged frame has log r = -15.30. Stored float32 column
  gave identical metrics on this data.
- y_now(t) = any info-bit error at iteration t.  y_fail = any info-bit error at iteration 50.
- q_now (primary, spec): logistic on log r.  q_now (extended, FROZEN as the main calibrated method): logistic on [log r, S]
  where S = -log(1 - r). Isotonic on log r = ablation A2. Report spec form as the negative result it is.
- q_fail (FROZEN): logistic on [log r_t, log r_t - log r_{max(t-3,1)}, t], fitted on t >= 5.
  The "+t" term is a deviation from the written spec (log r + change only). Declare it in the paper.
- Decisions on model form were made on the PILOT set (data in final_dataset, 30k frames). The new data's test split
  must be touched once.

## 2. Findings on the pilot set (validation, exploratory)
- H1 (raw overconfident, esp. early / low SNR): NOT supported. Early iterations are UNDER-confident (raw r overstates risk;
  e.g. t<=4 'looks right' band: mean r 0.25 vs observed 0.07). Late (t>=20) raw is mildly over-confident
  (r<0.5 band: predicted 0.00037, observed 0.00061; r>=0.5 band: 0.964 vs 0.986) but tiny in absolute terms.
- H2 (calibration halves ECE): supported pooled over all t (0.0060 -> 0.0011 extended logistic, 0.0003 isotonic; bootstrap
  intervals exclude 0). For t >= 10 raw is already calibrated (ECE 0.0004) and calibrated versions are slightly worse.
  The spec logistic on log r alone is significantly WORSE than raw.
- q_fail: log r alone PR-AUC 0.63; + 3-step change 0.66 (not clearly better); + t 0.875 [0.84, 0.90]; t alone 0.019 (=chance).
- Abandon preview (validation, F2): eps_b=0.9 catches 87/87 failing frames, 5 false abandons, saves ~4% of ideal-stop iterations.
  Only 87 failing frames: every number here is a small-count estimate.
- G1 (headroom): weak pass. Failing frames use 20% / 10% / 4-8% of ideal-stop iterations at 2.25 / 2.5 / 2.75 dB.
- G2 (recalibration improves ECE on validation; abandon head well above chance): PASS on the pilot set.
- Open: H3 vs B2/B5 needs syndrome weight (not in the pilot data).

## 3. FINAL DECISION: one regeneration, main + mismatch together (details in MESSAGE_TO_MEMBER1.md)
Reason: syndrome weight is missing from the pilot (needed by B2, B5, the accept rule); the pilot test has 71 failing frames.
Pilot evidence that more data is needed: F2 abandon vs ORACLE iteration cap differ by 0.0-0.07 iterations/frame at equal
false-abandon counts of 3-14 events. Separating those needs dozens of events per cell.

| SNR | train | val | test | expected failing (train / val / test) |
|---|---|---|---|---|
| 2.00 | 10k | 5k | 10k | 700 / 350 / 700 |
| 2.25 | 15k | 8k | 20k | 500 / 260 / 660 |
| 2.50 | 20k | 15k | 40k | 300 / 225 / 600 |
| 2.75 | 30k | 25k | 50k | 165 / 140 / 275 |
Main total 248k frames. Mismatch: delta in {-3,-2,-1,+1,+2,+3}, SNR 2.0/2.25/2.5 with 5k/10k/20k frames
(the FIRST frames of the test stream, same keys; delta=0 is those same frames from the main run, free) = 210k frames.
Optional last: matched-refit reference, train stream, SNR 2.25 only, 10k frames x 6 delta = 60k.
Total 458k (518k with optional) ~ 3.0 h (3.4 h) at the measured 42 frames/s.
Budgets: rho = 10% and 20% primary; 5% reported with CI. 2.75 dB test has ~275 failures: say so in the paper.

## 4. Syndrome facts from the probe (Member 1's run)
encoder: k=110 n=220 n_ldpc=1040 k_ldpc=200 z=20; pcm (840,1040); decoder x_hat has 350 columns
(= 220 transmitted + 40 punctured + 90 shortened, consistent with a pruned decoder graph), so encoder.pcm cannot be used
directly. probe_syndrome.py (v2) tests candidate matrices empirically. 40 of the 110 info bits are punctured (input LLR 0):
document under "SNR and puncturing".
