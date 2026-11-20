Hi! Person 2 here. We analysed your dataset: it is good and our pipeline reads it fine (labels re-derived from your
LLRs match your current_error on every cell, great). We need ONE more generation run, because the stopping-rule
baselines need syndrome weight and the pilot test split has only 71 failing frames. Please do everything below in one go,
so nobody has to regenerate again. Decoder, code, SNR conversion and 50 iterations stay exactly as they are.

STEP 0 (1 minute) - run `python src/probe_syndrome.py` (attached, v2) and paste me the whole output.
 Your earlier output showed x_hat has 350 columns, not 1040 (pcm), so we must find the right parity-check matrix.
 The probe tests candidates and checks "syndrome = 0 on correctly decoded frames".

STEP 1 - pilot: 5,000 frames at 2.00 dB. Send me the BLER (I expect ~7%). If it is below 3% or above 12%, tell me
 before continuing and we move the SNR.

STEP 2 - generate (SCALAR traces only, no LLR CSVs). Use calib_lib.trace_arrays(xhat_list, u, H, neg_is_one) per batch
 (drop the iteration-0 capture, keep iterations 1..50). It returns the exact arrays we read. Save one .npz per file with
 these keys: frame_id, snr, seed_key, final_failure, log_r, S, min_abs, mean_abs, mean_ent, info_err, cur_err, stored_r,
 syndrome_weight. Arrays are (frames, 50), per-frame arrays are (frames,).
 Sizes (frames):
   SNR   train   validation   test
   2.00  10,000   5,000       10,000
   2.25  15,000   8,000       20,000
   2.50  20,000  15,000       40,000
   2.75  30,000  25,000       50,000      (total 248,000)
 File names: {split}_snr_{snr:.2f}.npz with split in train / validation / test.
 Put the TEST files in a folder called sealed_test/ and do not send them to me until I say code freeze.

KEYED RANDOM NUMBERS (important, needed for mismatch to be comparable):
   B = 1000 frames per batch, fixed forever (changing it shifts every frame).
   ss  = np.random.SeedSequence([BASE_SEED, round(snr*100), split_code, batch_index])
   rng = np.random.default_rng(ss);  u = rng.integers(0,2,(B,110));  noise = rng.standard_normal((B,220))
   split_code: train=0, validation=1, test=2, train_mismatch=3.   frame_id = batch_index*1000 + i.
   Splits are separate streams, so they cannot overlap. Save BASE_SEED and the git hash in a config json.

STEP 3 - mismatch (same keys, same bits, same noise as the test stream; ONLY the assumed noise variance changes):
   y = (1-2c) + noise*sqrt(sigma2_true);   llr = -2*y / (sigma2_true * 10**(delta/10))      [keep your sign convention]
   delta in {-3, -2, -1, +1, +2, +3} dB. delta = 0 is NOT regenerated: it is the first frames of the main test run.
   Frames per (SNR, delta): 2.00 dB: first 5,000 test frames (batches 0-4); 2.25 dB: first 10,000 (batches 0-9);
   2.50 dB: first 20,000 (batches 0-19). That is 210,000 frames in total.
   File names: test_snr_{snr:.2f}_delta{delta:+d}.npz (e.g. test_snr_2.25_delta-3.npz), also in sealed_test/.
   Add a "delta" and "assumed_sigma2" entry to the config json.

STEP 4 (OPTIONAL, only if everything above is done): matched-refit reference. Stream split_code=3, SNR 2.25 only,
   10,000 frames for each delta in the six values: train_mismatch_snr_2.25_delta{delta:+d}.npz.

CHECKS before you send anything (please paste the results):
 1. Determinism: regenerate the first 1,000 test frames at 2.25 dB, the arrays must be identical.
 2. delta = 0 consistency: the first 10,000 frames of main test at 2.25 dB must equal what the mismatch code gives
    when you set delta = 0 (this proves the mismatch path changes nothing else).
 3. On correctly decoded frames, syndrome_weight at iteration 50 is 0 (>= 99%).
 4. No frame_id overlap between train / validation / test for any SNR.
 5. Report BLER per (SNR, split) and per (SNR, delta) in one csv.
Timing at ~42 frames/s: main ~1.6 h, mismatch ~1.4 h, optional ~0.4 h. Order: pilot, train+validation, test, mismatch.
Please do not store LLRs or true bits (only scalar traces). Thank you!
