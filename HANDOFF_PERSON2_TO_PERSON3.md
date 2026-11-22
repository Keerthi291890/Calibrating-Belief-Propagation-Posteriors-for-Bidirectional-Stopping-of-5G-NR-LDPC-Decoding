# Handoff: Person 2 (calibration) -> Person 3 (stopping and evaluation)

## 1. What you receive
| Item | Where | Purpose |
|---|---|---|
| Frozen calibrators `q_now`, `q_fail` | `results/person2_v3/q_now.json`, `q_fail.json`, `model_card.json` | calibrated risk estimates |
| Python interface | `src/risk_model.py` (`RiskModel`) | `q_now`, `q_fail` arrays + validity masks |
| Stopping module | `src/p2_stopping.py` | rules B1-B5, P-raw, P-cal, A1; metrics; tuning; operation counts |
| Freeze record | `results/person2_v3/FROZEN.json` | hashes of models and code, validation-tuned thresholds, cost-model parameters |
| One-shot test script | `src/p2_final_test.py` | calibration + stopping + mismatch evaluation, run ONCE |
| Validation evidence | `results/person2_v3/*.csv, *.png` | calibration tables, bootstrap CIs, tail calibration, in-sample stopping preview |
| Tests | `tests/test_calib_lib.py`, `tests/test_stopping.py` | numerics, tree-code check, stopping properties |

## 2. Using the models
```python
from risk_model import RiskModel
import numpy as np
rm  = RiskModel("results/person2_v3")
d   = dict(np.load("data/scalar_traces/dev/validation_snr_2.25.npz"))
out = rm.predict(d)          # q_now, q_fail: (frames, 50); valid_now, valid_fail: (frames, 50) bool
```
Inputs used: `log_r`, `S`, `min_abs`, `syndrome_weight`, iteration index. Never `info_err`, `cur_err`, `final_failure`.
Both heads are only valid where the stopping rule is still undecided: `valid_now` = syndrome > 0,
`valid_fail` = syndrome > 0 and iteration >= t_min (5). Outside, the rules accept at syndrome zero anyway.

## 3. Frozen definitions
- r_t = 1 - prod_i max(p_i, 1-p_i) over the 110 information bits, computed from LLRs in the log domain (`-expm1(-S)`).
  The decoder clips LLRs at |L| = 20, so converged frames sit at log r = -15.2995.
- y_now(t) = any information-bit error at iteration t. y_fail = any error at iteration 50.
- q_now: logistic on [log r, S, log(1+syndrome), min|LLR|], fitted on syndrome>0 rows (selected on validation).
- q_fail: logistic on [log r, change in log r over k iterations, t, log(1+syndrome), min|LLR|, change in log(1+syndrome)],
  fitted on syndrome>0 and t>=5 rows (k selected on validation).
- Selection rule (declared before looking): best validation log loss, then the simplest model not significantly worse
  (paired frame-level bootstrap, 95%).
- Deviations from the written spec (must appear in the paper): the spec's logistic on log r alone is significantly worse than
  raw r; the added terms (S, syndrome, min|LLR|, iteration t) and the syndrome>0 fitting region are ours.

## 4. What validation showed (exploratory; test not yet opened)
- q_now, region syndrome>0: raw ECE 0.0134 -> 0.0017 with the simple extended logistic (paired CIs exclude 0); the selected
  model has better log loss (0.139 vs 0.182) and AUC (0.967 vs 0.939).
- Tail: raw r under-states error risk by 10-200x for r <= 0.01. The selected q_now is reasonable for 0.01-0.1 but still
  under-states risk below 0.01 (observed 3.9% vs predicted 0.6%). A few percent of syndrome>0 frames are confident but wrong,
  so accept thresholds below about 0.01 cannot be trusted.
- q_fail is calibrated across all bins >= 0.5 (observed/predicted 0.96-1.08); raw r cannot support abandonment at all.
- Stopping preview (in-sample): P-cal saves about 1.8-12% (10% budget) and 2.5-17% (20% budget) of iterations vs syndrome-zero,
  roughly 1.2-2.5x the iteration-cap baseline B5, mostly through abandonment. Absolute savings are small.
- Mismatch preview (2.25 dB): raw r ECE rises up to 8x; the matched calibrators degrade far less; refitting helps modestly.
  For delta < 0 q_now under-states error risk (unsafe direction); for delta > 0 it over-states (conservative).

## 5. Review comments applied (your review of p2_stopping.py)
Applied: explicit deterministic tie-break; extended grids (accept up to 0.5) and a disabled trigger in every grid (so every
rule can fall back to syndrome-zero); conditional undetected-error rate among accepted frames, NACK rate and cap share
reported separately; fragile-budget flag (budget < 10 events) kept for the 5% budget instead of dropping it; operation counts
(below); log_r clipped to [floor, 0] before exponentiation; B3/A1 documented as information-bit features.
Correction recorded: at 2.75 dB the 153 failures are in 25,000 validation frames (50,000 is the test split); 5% of 153 = 7.7 events.

## 6. Operation counts (final model, after your second review)
Reported separately: BP operations, syndrome-check operations, score operations, total. ONLINE process only (iterations actually
executed until the stop); offline full-run trace generation excluded; calibration training time is in `selection_summary.json`.
Graph size comes from the pruned decoder graph: `n_edges = dec._pcm.nnz`, `n_vn = 350`, `n_checks = 150` (shape of `dec._pcm`);
all three are stored in FROZEN.json and must be confirmed by Person 1 in `generation_config.json`.
Unit: scalar operations. add/sub/mul/compare/abs = 1; each transcendental (exp, log, log1p, expm1, tanh, atanh) and division = cx.
- BP per edge per iteration = variable node (2: accumulate, subtract) + check node (2cx+7: tanh(m/2) = 1+cx, 3 multiplications for the
  product of the other inputs, atanh+doubling = cx+1, 2 clips) = 2cx+9. So 15 (cx=3, LUT/fast approximation), 29 (cx=10, default),
  59 (cx=25, slow library call). `2cx+7` is the check-node part only; the per-edge total is `2cx+9` (code: `bp_edge_ops`). The sensitivity varies cx in BOTH BP and the score.
- Syndrome per iteration = n_vn + n_edges + n_checks. Logistic head = n features multiplies + n adds (last add = bias) + sigmoid
  (2cx+1) + 1 threshold compare. No run-time normalisation (weights are stored in raw feature units).
- Schedule (sequential, accept precedence): syndrome -> accept head at every syndrome-positive iteration -> abandon head only if
  accept did not fire and t >= t_min. Both heads share features (computed once). When the accept head fires at t >= t_min the
  abandon head is not evaluated at that iteration.
- History for the abandon head's change features is computed just in time at iterations t_min-k .. t_min-1 and charged there
  (free if the accept head already computes log r and log(1+syndrome)).
- `mean_ent` depends on `abs` because the implemented stable formula is softplus(-|L|) + |L| sigmoid(-|L|).
Tests: `tests/test_stopping.py` checks exact counts for the sequential schedule, the history window and the cx scaling.
Open for you: sign off these prices; optionally decide whether thresholds should ALSO be tuned for total operations (not done:
thresholds are tuned for mean iterations, as approved).

## 6b. Final test results
The frozen models and thresholds were evaluated once on the sealed test files. Results, hypothesis verdicts and the cautions to state are in
`FINAL_RESULTS_PERSON2.md`; tables/figures via `python src/p2_paper_tables.py`. Your two required statements (approximate scalar-operation
estimates; thresholds minimise mean iterations, operations are supplementary) are quoted at the top of that file.

## 7. Test protocol
`p2_freeze.py` tunes thresholds on validation only and stores them; `p2_final_test.py` refuses to run unless the frozen hashes
match, and refuses a second run (override is logged in `TEST_ACCESS_LOG.txt`). Test files cannot be opened before the freeze
(`p2_common.load_group` raises). Mismatch-test frames are the same frames as the main test (paired); they are not independent.
`train_mismatch_*` files are training data (refit), despite living in `sealed_test/`.

## 8. What you still own
1. Sign off `src/p2_stopping.py` (including the operation-cost model) BEFORE the freeze; it is hashed.
2. Interpret and write the trade-off, mismatch-stopping and limitation sections from the final CSVs.
3. State clearly: abandonment drives most savings; savings are modest. Fragility is flagged by `budget < 10 excess events` computed on the
   frames the thresholds were tuned on: on validation only (2.75 dB, 5%) = 7.7 events is flagged; 2.75 dB at 10% = 15.3 events is not flagged but is
   still small, so draw no strong conclusion from it alone. The test table reports both `fragile_budget_validation` and `fragile_budget_test`.
4. Wording: contribution = calibration, bidirectional decisions, mismatch reliability. Not "posterior early stopping", not energy,
   no certified guarantees. Say "empirical" for every reliability statement.
