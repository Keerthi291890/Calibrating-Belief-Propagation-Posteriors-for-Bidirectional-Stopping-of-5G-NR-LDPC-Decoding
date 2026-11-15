import torch
from sionna.phy.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder

K = 110
N = 220

encoder = LDPC5GEncoder(K, N)

decoder = LDPC5GDecoder(
    encoder,
    num_iter=20,
    return_infobits=True,
    hard_out=True
)

# Generate information bits
u = torch.randint(0, 2, (4, K)).float()

# Encode
c = encoder(u)

# Sionna-compatible perfect logits
llr = 20.0 * (2.0 * c - 1.0)

# Decode
u_hat = decoder(llr)

errors = torch.sum(u != u_hat).item()

print("Original shape:", u.shape)
print("Encoded shape:", c.shape)
print("Decoded shape:", u_hat.shape)
print("Information bit errors:", errors)
print("Round-trip successful:", errors == 0)