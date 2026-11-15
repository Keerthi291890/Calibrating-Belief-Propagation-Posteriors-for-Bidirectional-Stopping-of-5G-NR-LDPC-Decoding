import torch
from sionna.phy.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder

K = 110
N = 220
BATCH_SIZE = 4
NUM_ITER = 5

encoder = LDPC5GEncoder(K, N)

# Store intermediate decoder estimates
traces = []

def capture_estimates(msg_vn, iteration, x_hat):
    traces.append({
        "iteration": iteration,
        "shape": tuple(x_hat.shape),
        "estimate": x_hat.detach().clone()
    })
    return msg_vn

decoder = LDPC5GDecoder(
    encoder,
    num_iter=NUM_ITER,
    return_infobits=True,
    hard_out=False,
    v2c_callbacks=[capture_estimates]
)

# Generate information bits
u = torch.randint(0, 2, (BATCH_SIZE, K)).float()

# Encode
c = encoder(u)

# BPSK modulation
x = 1.0 - 2.0 * c

# AWGN channel
ebno_db = 3.0
rate = K / N
ebno = 10 ** (ebno_db / 10)
noise_variance = 1.0 / (2.0 * rate * ebno)

noise = torch.randn_like(x) * noise_variance**0.5
y = x + noise

# Sionna decoder input logits
logits = -2.0 * y / noise_variance

# Decode
u_soft = decoder(logits)

print("Final soft output shape:", tuple(u_soft.shape))
print("Number of callback captures:", len(traces))

print("\nCaptured iterations:")
for item in traces:
    print(
        "Iteration:", item["iteration"],
        "| Shape:", item["shape"]
    )

print("\nFirst captured estimate (sample):")
print(traces[0]["estimate"][0, :10])

print("\nFinal decoder output (sample):")
print(u_soft[0, :10])