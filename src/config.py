# LDPC configuration

BASE_GRAPH = 2

N = 220
K = 110
RATE = K / N

MAX_ITERATIONS = 50

# BPSK and AWGN configuration
MODULATION = "BPSK"
CHANNEL = "AWGN"

# Reproducibility
RANDOM_SEED = 42