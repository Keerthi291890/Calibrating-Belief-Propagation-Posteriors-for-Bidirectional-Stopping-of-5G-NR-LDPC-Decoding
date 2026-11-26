# Person 3 paper sections (draft; all statements are empirical)

Numbers come from `results/person2_v3/final_*.csv` (frozen thresholds, sealed test opened once) and are re-derived by
`src/p3_analysis.py` (9 number checks, all pass). Operation counts are approximate scalar-operation estimates, not hardware
instructions and not energy. Thresholds minimise MEAN ITERATIONS under the excess-event budget; operations are supplementary.

## A. Evaluation protocol (Person 3)
**Rules.** Accept at the first iteration with syndrome zero or q_now <= eps_a; abandon (NACK) at the first t >= t_min = 5 with
q_fail >= eps_b; accept wins ties; otherwise run to 50. The rule never stops later than syndrome-zero (property tested in
`tests/test_stopping.py`). Comparators: B1 fixed 50, B2 syndrome-zero, B3 min|LLR|, B4 ideal information-bit CRC, B5 syndrome plus a tuned iteration cap,
P-raw, P-cal, and ablations A1 (mean entropy, accept only) and accept-only / abandon-only / both.
**Excess loss** is measured against the full 50-iteration run: false accept (accepted with errors, full run correct) plus false
abandon (abandoned, full run correct). We also report the undetected-error rate and the net BLER change.
**Budget.** For rho in {5, 10, 20}% the budget is rho x (failing frames). Each rule's thresholds are the ones with the fewest mean
iterations whose validation excess is within budget (deterministic tie-break; every grid includes a disabled trigger so a rule can
fall back to syndrome-zero). Thresholds are tuned per SNR on validation, frozen, and evaluated once on the sealed test
(10k, 20k, 40k, 50k frames at 2.00 to 2.75 dB). Because thresholds sit at the edge of the validation budget, the realised test excess can exceed the budget;
we therefore report the realised excess beside every saving and do not call the comparison an unconditional equal-budget comparison.
**Uncertainty.** Frame-level bootstrap intervals (all 50 iterations of a frame stay together); mismatch-test frames are the same frames as the matched test (paired, not independent).

## B. Trade-off results (matched channel)
1. *Iterations.* At rho = 10%, P-cal both saves 12.6, 5.8, 3.3 and 1.7% of mean iterations relative to syndrome-zero at 2.00, 2.25, 2.50 and 2.75 dB;
   B5 saves 9.1, 3.4, 1.0 and 0.9%. The paired advantage of P-cal both over B5 is 0.05 to 0.62 iterations, all intervals excluding zero. Savings shrink as SNR rises.
2. *Where the saving comes from.* Abandonment: abandon-only gives 81 to 96% of the P-cal both saving at rho = 10% and is the larger head in 11 of 12 SNR x rho cells (exception: 2.75 dB, rho = 5%, where accept-only saves 1.06% and abandon-only 0.69%).
   B4 (information-CRC) is an accept-side bound only: at 2.00 dB it saves 7.4% while P-cal both saves 12.6%, because abandoning failing frames is not available to B4.
   The accept head alone saves 0.8 to 4.2% and costs more than it saves in operations.
3. *Total operations.* P-cal both does not beat B5 in total operations: at rho = 10% and cx = 10 it is +6.1, -1.0, -3.5, -5.1% against B5 at +9.1, +3.4, +1.0, +0.9%.
   Across 36 SNR x rho x cx cells P-cal both is ahead of B5 in only 2 (`p3_ops_sensitivity.csv`). The score cost (calibrated heads evaluated every iteration) is the reason.
4. *Reliability cost.* Early abandonment raises the NACK rate (about 7% at 2.00 dB down to 0.5% at 2.75 dB at rho = 10%) and the net BLER (+0.0068 at 2.00 dB). Early accept raises the undetected-error rate among accepted frames
   (0.7% at 2.00 dB, rho = 10%, against about 0.1% for syndrome-zero).
5. *Raw posterior.* P-raw cannot support abandonment inside any budget (best feasible threshold is never), so calibration is what makes the rule usable.
6. *Budget compliance on the test.* P-cal both exceeds the budget in 2 of 12 cells (2.00 dB at 5% and 10%; 69 against 66.7 events at 10%), B5 in 5 of 12.
Figures: `p3_fig_tradeoff.png`, `p3_fig_decomposition.png`; tables `p3_tradeoff_table.csv`, `p3_saving_decomposition.csv`.
E1 — Development headroom:The development analysis compared syndrome-zero stopping (B2) with the ideal information-bit CRC ceiling (B4). Across the four SNR points, the estimated iteration headroom ranged from 7.43% to 11.73% on training data and from 7.80% to 11.74% on validation data. This indicates that an accept-side stopping opportunity exists in the development data. B4 is an ideal reference bound, not an implementable stopping rule.

E5 — Threshold transfer:Threshold transfer was evaluated using validation data only. The experiment produced 32 results: 16 for P-cal both and 16 for B5, covering four threshold-tuning SNRs and four application SNRs. For both methods, all four same-SNR applications met the excess-event budget. Among the 12 off-diagonal transfers, five exceeded the budget for each method. All 16 transfers for each method achieved positive iteration savings relative to B2. However, iteration savings did not guarantee reliability-budget compliance. These results demonstrate SNR-dependent threshold transfer behaviour and do not establish generalisation to unseen channel conditions.

## C. Reliability under channel-model mismatch
Setting: assumed noise variance offset delta in {-3..+3} dB at 2.00, 2.25, 2.50 dB; frozen matched thresholds (rho = 10%) applied unchanged.
- *Calibration.* Matched calibrators keep q_now ECE <= 0.027 and q_fail ECE <= 0.089, against raw q_now up to 0.107 at delta = -3 (2.25 dB). ECE of the matched calibrators is largest at delta = +3 (not delta < 0), so H4 is not supported for calibration error.
- *Stopping.* P-cal both exceeds the excess budget in 6 of 18 delta != 0 cells, all at delta < 0 (excess/budget up to 1.48); none at delta > 0. B5 exceeds it in 6 of 18 (5 at delta < 0, 1 at delta = +3), so the asymmetry is a property of the setting, not only of calibration.
  P-raw both exceeds the budget at every delta < 0 (up to 14.5 times the budget at 2.25 dB, delta = -3) while saving almost nothing.
- *Refit.* Calibrators refit at 2.25 dB restore compliance in all six delta cells (e.g. delta = -3: 74 events against a budget of 62.6 before, 31 after). Refit is not uniformly better in stopping: at delta > 0 its excess/budget (0.42 to 0.63) is higher than the matched calibrators' (0.08 to 0.31), though still inside the budget. Single SNR only.
- *Wording.* "Mismatch calibration remains useful, but stopping reliability is not universally robust." Savings in over-budget cells must not be quoted as free savings.
Figures: `p3_fig_mismatch_budget_heatmap.png`; tables `p3_mismatch_budget_ratio.csv`, `p3_refit_vs_matched_2.25dB.csv`.

## D. Limitations
Single code and channel (5G NR BG2, K = 110, N = 220, AWGN/BPSK, 50 flooding BP iterations, 2.00 to 2.75 dB); mismatch covers only the assumed noise variance, at three SNRs and rho = 10%, with refit at one SNR.
Operation counts are approximate scalar estimates and depend on cx; thresholds minimise iterations, not operations. Validation-tuned thresholds can exceed the budget on the test. The 2.75 dB, rho = 5% budget is statistically fragile (about 7.7 validation events).
The accept head cannot be trusted below q about 0.01 (confident-but-wrong frames put a floor under any posterior score). The pooled q_fail intercept drifts with SNR (+0.41 at 2.00 dB, -0.59 at 2.75 dB); SNR-aware calibration was not tried.
The sealed test was used once, so no further tuning on it is allowed. No energy claim, no certified guarantee, and the idea of posterior-based early stopping is not claimed to be new.

## E. Hypothesis wording (replace the "equal budget" phrasing)
H3: "At thresholds tuned on validation to the same relative excess budget, P-cal both saves iterations relative to B2 and B5 (mostly through abandonment) but not total estimated operations; some frozen rules exceed the realised test budget, so realised excess is reported beside each saving."

## F. Related work to cite (verify every reference before use)
Syndrome / CRC / cross-entropy stopping for iterative decoders (Hagenauer, Offer and Papke, IEEE Trans. Inf. Theory, 1996, as I recall it), factor graphs and sum-product (Kschischang, Frey and Loeliger, 2001), 5G NR LDPC (3GPP TS 38.212),
probability calibration (Platt scaling; Zadrozny and Elkan, isotonic regression; Guo, Pleiss, Sun and Weinberger, ICML 2017), and HARQ / ML-based early termination papers. These are from memory and unchecked: confirm titles, years and venues, and add the project's selected reference paper.
