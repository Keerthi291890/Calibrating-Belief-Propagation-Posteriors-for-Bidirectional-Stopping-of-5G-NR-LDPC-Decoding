"""
PROBE v2 (run where Sionna works).  Finds a parity-check matrix that matches the decoder's x_hat
(350 columns in your setup, not the 1040 of encoder.pcm) and checks it EMPIRICALLY:
a candidate is valid if frames that decoded correctly have syndrome 0 at iteration 50.
Also detects the LLR sign convention.  Paste the whole output back.

    python src/probe_syndrome.py
"""
import numpy as np
import scipy.sparse as sp
import torch
from sionna.phy.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder

K, N, NUM_ITER, B = 110, 220, 50, 512
EBNO_DB = 2.0                      # low SNR on purpose so that some frames fail
torch.manual_seed(0)

enc = LDPC5GEncoder(K, N)
cap = []
def capture(msg_vn, it, x_hat):
    cap.append((int(it), x_hat.detach().cpu().numpy().copy()))
    return msg_vn
dec = LDPC5GDecoder(enc, num_iter=NUM_ITER, return_infobits=True, hard_out=False, v2c_callbacks=[capture])

nv = 1.0 / (2.0 * (K / N) * 10 ** (EBNO_DB / 10))
u = torch.randint(0, 2, (B, K)).float()
c = enc(u)
y = (1.0 - 2.0 * c) + torch.randn_like(c) * np.sqrt(nv)
with torch.no_grad():
    dec(-2.0 * y / nv)
u = u.numpy().astype(int)
x_last = [x for it, x in cap if it == NUM_ITER][-1]
n_vn = x_last.shape[1]
print(f"captures={len(cap)}  x_hat shape={x_last.shape}  encoder.pcm shape={enc.pcm.shape}")

# ---- sign convention ----
ok_neg = (((x_last[:, :K] < 0).astype(int)) == u).all(1)
ok_pos = (((x_last[:, :K] > 0).astype(int)) == u).all(1)
neg_is_one = ok_neg.sum() >= ok_pos.sum()
fin_ok = ok_neg if neg_is_one else ok_pos
print(f"sign check: correct frames if (L<0 => bit1): {ok_neg.sum()}/{B};  if (L>0 => bit1): {ok_pos.sum()}/{B}")
print(f"=> convention: {'L<0 means bit 1' if neg_is_one else 'L>0 means bit 1'};  failed frames: {(~fin_ok).sum()}")

# ---- candidate matrices ----
def to_csr(obj):
    try:
        if sp.issparse(obj):
            return obj.tocsr()
        if hasattr(obj, "detach"):
            obj = obj.detach().cpu().numpy()
        a = np.asarray(obj)
        return sp.csr_matrix(a) if a.ndim == 2 else None
    except Exception:
        return None

cands = {}
def scan(obj, name, depth=0):
    if depth > 2 or obj is None:
        return
    for k, v in list(vars(obj).items()) if hasattr(obj, "__dict__") else []:
        m = to_csr(v) if ("pcm" in k.lower() or k.lower() in ("_h", "h")) else None
        if m is not None and m.shape[1] == n_vn:
            cands[f"{name}.{k}"] = m
        if "decoder" in k.lower() and depth < 2:
            scan(v, f"{name}.{k}", depth + 1)
scan(dec, "dec")
H = enc.pcm.tocsr()
rows_inside = np.where(H[:, n_vn:].getnnz(axis=1) == 0)[0]
cands["H_full[rows inside first n_vn cols, :n_vn]"] = H[rows_inside][:, :n_vn]
cands["H_full[:, :n_vn] (all rows)"] = H[:, :n_vn]

print("\ncandidate matrices:")
best = None
for name, M in cands.items():
    hard = ((x_last < 0) if neg_is_one else (x_last > 0)).astype(np.int64)
    sw = np.asarray(((M @ hard.T) % 2).sum(axis=0)).ravel()
    zero_ok = (sw[fin_ok] == 0).mean() if fin_ok.any() else np.nan
    zero_bad = (sw[~fin_ok] == 0).mean() if (~fin_ok).any() else np.nan
    print(f"  {name:48s} shape={M.shape}  syndrome==0 among correct frames: {zero_ok:6.1%}"
          f" | among failed frames: {zero_bad:6.1%}")
    if zero_ok >= 0.99 and (best is None):
        best = name
print()
if best:
    print(f"RESULT: use '{best}' as H.  neg_is_one={neg_is_one}.  (calib_lib.trace_arrays(..., H, neg_is_one))")
    print("Failed frames with syndrome==0 are undetected errors (expected to be rare).")
else:
    print("RESULT: no candidate gave syndrome 0 on correct frames -> send this output, mapping still unknown.")
