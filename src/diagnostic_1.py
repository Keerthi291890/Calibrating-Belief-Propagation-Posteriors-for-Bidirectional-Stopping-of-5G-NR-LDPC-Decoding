import torch
from sionna.phy.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder

K, N = 110, 220
NUM_ITER = 5

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

u = torch.randint(0, 2, (4, K)).float()
c = encoder(u)

rate = K / N
ebno_db = 3.0
ebno = 10 ** (ebno_db / 10)
noise_var = 1.0 / (2.0 * rate * ebno)

x = 1.0 - 2.0 * c
y = x + torch.randn_like(x) * noise_var**0.5

# Input logits: log(P(bit=1)/P(bit=0))
logits = -2.0 * y / noise_var

u_soft = decoder(logits)

print("Decoder internal variable nodes:", decoder.num_vns)
print("Pruned graph length:", decoder._n_pruned)
print("Information bits:", K)
print("Number of captures:", len(traces))

# Compare final callback estimates with final decoder output
last_iteration, last_estimate = traces[-1]

print("\nLast callback iteration:", last_iteration)
print("Callback information-bit shape:",
      last_estimate[:, :K].shape)

print("Final decoder output shape:", u_soft.shape)

# Callback contains LLR = log(P0/P1).
# Final soft output contains logits = log(P1/P0).
print("Mapping verified:",
      torch.allclose(last_estimate[:, :K], -u_soft, atol=1e-5))

# Inspect initial punctured positions
print("\nInitial first 40 positions:")
print(traces[0][1][0, :40])

# Convert callback LLRs to probabilities
p1 = torch.sigmoid(-last_estimate[:, :K])

print("\nInformation-bit probability shape:", p1.shape)
print("First 10 probabilities:", p1[0, :10])