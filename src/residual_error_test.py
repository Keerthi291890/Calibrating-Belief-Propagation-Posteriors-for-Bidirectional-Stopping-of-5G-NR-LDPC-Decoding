
import torch
import pandas as pd

from sionna.phy.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder

# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

K = 110
N = 220
BATCH_SIZE = 500
NUM_ITER = 50
EBNO_DB = 3.0

device = "cpu"
SEED = 42 + int(EBNO_DB * 100)
torch.manual_seed(SEED)     

# --------------------------------------------------
# 2. Create encoder and decoder
# --------------------------------------------------

encoder = LDPC5GEncoder(K, N)

traces = []

def capture(msg_vn, iteration, x_hat):
    traces.append((iteration, x_hat.detach().clone()))
    return msg_vn

decoder = LDPC5GDecoder(
    encoder,
    num_iter=NUM_ITER,
    return_infobits=True,
    hard_out=False,
    v2c_callbacks=[capture]
)

# --------------------------------------------------
# 3. Generate information bits and encode
# --------------------------------------------------

u = torch.randint(0, 2, (BATCH_SIZE, K)).float()

c = encoder(u)

# --------------------------------------------------
# 4. BPSK modulation and AWGN channel
# --------------------------------------------------

rate = K / N
ebno = 10 ** (EBNO_DB / 10)

noise_var = 1.0 / (2.0 * rate * ebno)

# BPSK: 0 -> +1, 1 -> -1
x = 1.0 - 2.0 * c

noise = torch.randn_like(x) * (noise_var ** 0.5)
y = x + noise

# Sionna decoder input convention
logits = -2.0 * y / noise_var

# --------------------------------------------------
# 5. Decode and capture iteration estimates
# --------------------------------------------------

u_soft = decoder(logits)


# --------------------------------------------------
# 6. Calculate labels and save iteration-wise data
# --------------------------------------------------

# Final-failure label: compare final decoded bits
# against the original transmitted information bits.

final_iteration, final_x_hat = traces[-1]

final_decoded = (final_x_hat[:, :K] < 0).int()

final_failure = (
    final_decoded != u.int()
).any(dim=1).int()

rows = []

for iteration, x_hat in traces:

    # Iteration 0 is the initial channel-only state.
    # Keep only actual BP iterations for the research dataset.
    if iteration == 0:
        continue

    L = x_hat[:, :K]

    p1 = torch.sigmoid(-L)
    p_correct = torch.maximum(p1, 1.0 - p1)

    log_p_all_correct = torch.log(p_correct).sum(dim=1)
    r = -torch.expm1(log_p_all_correct)

    decoded = (L < 0).int()
    y_now = (decoded != u.int()).any(dim=1).int()

    for j in range(BATCH_SIZE):

        row = {
            "frame_id": j,
            "iteration": iteration,
            "snr_db": EBNO_DB,
            "residual_error": r[j].item(),
            "current_error": y_now[j].item(),
            "final_failure": final_failure[j].item()
        }

        # Save all 110 information-bit posterior LLRs
        for bit in range(K):
            row[f"llr_{bit}"] = L[j, bit].item()

        rows.append(row)

df = pd.DataFrame(rows)

# --------------------------------------------------
# 7. Save dataset and summary
# --------------------------------------------------

df.to_csv(f"ldpc_pilot_{EBNO_DB}dB.csv", index=False)

print("Dataset shape:", df.shape)
print("Number of frames:", BATCH_SIZE)
print("Number of BP iterations:", NUM_ITER)

print("\nCurrent-error labels:")
print(df["current_error"].value_counts())

print("\nFinal-failure labels:")
print(
    df.drop_duplicates("frame_id")["final_failure"].value_counts()
)

print("\nSaved: ldpc_pilot_dataset.csv")
