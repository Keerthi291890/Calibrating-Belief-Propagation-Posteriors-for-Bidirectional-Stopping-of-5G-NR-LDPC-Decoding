# Person 2 run order (scalar-trace data)

Copy the contents of this patch into the repo (src/, tests/, RUN_ORDER.md). Then from the repo root
(data root = the folder that contains dev/ and sealed_test/, e.g. E:/math_dataset):

1. python tests/test_calib_lib.py                                  # unit tests
2. python src/p2_audit.py --data-root E:/math_dataset              # statistical audit (dev + refit files only)
3. python src/p2_fit_dev.py --data-root E:/math_dataset            # fit on train, select on validation, save models
4. python src/p2_mismatch_dev.py --data-root E:/math_dataset       # mismatch PREVIEW on refit files (training data)
4b. python src/p2_stop_preview.py   --data-root E:/math_dataset     # tail calibration + validation-only stopping preview
   -> send me the printed output of steps 4 and 4b (2 and 3 already received).

Only after we both agree the validation results are final:
5. python src/p2_freeze.py --data-root E:/math_dataset --i-have-reviewed-validation   # hashes models + code, tunes thresholds on VALIDATION
6. python src/p2_final_test.py --data-root E:/math_dataset         # opens the sealed test files ONCE: calibration + stopping + mismatch

Outputs go to results/person2_v3/. Person 3 uses:  from risk_model import RiskModel  (see docstring in src/risk_model.py).
Rules: never edit src files after step 5; do not run step 6 until Person 3's code is also frozen (G3).

Before step 5: Person 3 reviews src/p2_stopping.py (rule semantics, excess definition, grids, budgets). It is hashed at freeze.
