import random
import time

# ------------------------------------------------------------------
# 1) Define DES S-box (S-box 1) and a helper to get its 4-bit output.
# ------------------------------------------------------------------
S_BOX = [
    [14,  4, 13,  1,  2, 15, 11,  8,  3, 10,  6, 12,  5,  9,  0,  7],
    [ 0, 15,  7,  4, 14,  2, 13,  1, 10,  6, 12, 11,  9,  5,  3,  8],
    [ 4,  1, 14,  8, 13,  6,  2, 11, 15, 12,  9,  7,  3, 10,  5,  0],
    [15, 12,  8,  2,  4,  9,  1,  7,  5, 11,  3, 14, 10,  0,  6, 13]
]

def apply_sbox_6to4(x_6bit):
    row = ((x_6bit & 0b100000) >> 4) | (x_6bit & 0b000001)
    col = (x_6bit & 0b011110) >> 1
    return S_BOX[row][col]

def popcount6(x):
    """Return number of bits set in x, x up to 6 bits."""
    return bin(x).count('1')

# Precompute real S-box outputs for x in [0..63].
REAL_SBOX = [apply_sbox_6to4(x) for x in range(64)]

# ------------------------------------------------------------------
# 2) Represent A, B in a single integer "combo":
#    - B is 6 bits (lowest bits)
#    - A is 24 bits (highest bits), which is 4 columns of 6 bits each.
#
#    So total = 30 bits.
# ------------------------------------------------------------------
def mismatch_count(combo):
    """Compute mismatch of the given (A,B) vs. real S-box on all 64 inputs."""
    B  =  combo        & 0x3F
    A24 = (combo >>  6) & 0xFFFFFF

    # Extract 4 columns from A24 (each 6 bits).
    A_col0 =  (A24 >>  0) & 0x3F
    A_col1 =  (A24 >>  6) & 0x3F
    A_col2 = (A24 >> 12) & 0x3F
    A_col3 = (A24 >> 18) & 0x3F

    mismatches = 0
    for x in range(64):
        real_val = REAL_SBOX[x]
        xprime   = x ^ B

        out0 = popcount6(A_col0 & xprime) & 1
        out1 = popcount6(A_col1 & xprime) & 1
        out2 = popcount6(A_col2 & xprime) & 1
        out3 = popcount6(A_col3 & xprime) & 1
        approx_val = (out3 << 3) | (out2 << 2) | (out1 << 1) | out0

        if approx_val != real_val:
            mismatches += 1
    return mismatches

def flip_random_bits(value, num_bits_to_flip=1):
    """
    Return 'value' with 'num_bits_to_flip' random bits flipped
    in the range [0..29] (since total is 30 bits).
    """
    for _ in range(num_bits_to_flip):
        bitpos = random.randint(0, 29)
        mask = 1 << bitpos
        value ^= mask
    return value

# ------------------------------------------------------------------
# 3) A simple iterative "learning"/hill-climbing approach:
#    - Start from a random combo
#    - On each iteration, propose flipping 1 or 2 bits
#    - If it improves mismatch, keep it; otherwise revert
# ------------------------------------------------------------------
def hill_climb(
    max_iterations=1_000_000, 
    bits_to_flip_options=[1,1,1,2], 
    report_interval=100_000
):
    """
    Attempt to find a good (A,B) that approximates the S-box, 
    by flipping bits that reduce mismatch.
    'bits_to_flip_options' is a small list or distribution from which
    we pick how many bits to flip at each iteration (usually 1 or 2).
    """
    # Start random
    current = random.getrandbits(30)  # random 30-bit int
    current_mis = mismatch_count(current)

    best = current
    best_mis = current_mis

    start = time.time()

    for i in range(max_iterations):
        # Decide how many bits we try to flip this iteration
        k = random.choice(bits_to_flip_options)

        candidate = flip_random_bits(current, k)
        candidate_mis = mismatch_count(candidate)

        if candidate_mis < current_mis:
            # improvement => accept
            current = candidate
            current_mis = candidate_mis
            if current_mis < best_mis:
                best = current
                best_mis = current_mis

        # (If you want simulated annealing or so, you might accept worse solutions sometimes.)

        if (i+1) % report_interval == 0:
            elapsed = time.time() - start
            speed = (i+1) / elapsed
            print(
                f"Iter={i+1}, best mismatch={best_mis}, current mismatch={current_mis}, "
                f"speed={speed:,.0f} it/s"
            )
            # If best_mis=0 (perfect match is basically impossible for a DES S-box), we'd break.

    return best, best_mis

# ------------------------------------------------------------------
# 4) Run the learning approach, then decode the best solution
# ------------------------------------------------------------------
if __name__ == "__main__":
    random.seed(0)  # reproducibility
    MAX_ITERS = 500_000  # Increase if you want more thorough search

    print(f"Starting hill climb for up to {MAX_ITERS} iterations...")
    final_combo, final_mis = hill_climb(max_iterations=MAX_ITERS, report_interval=50_000)
    
    # decode
    B6_final  =  final_combo        & 0x3F
    A24_final = (final_combo >>  6) & 0xFFFFFF

    A_col0 =  (A24_final >>  0) & 0x3F
    A_col1 =  (A24_final >>  6) & 0x3F
    A_col2 = (A24_final >> 12) & 0x3F
    A_col3 = (A24_final >> 18) & 0x3F

    print("\nRESULTS:")
    print(f"  best mismatch found = {final_mis} out of 64")
    print(f"  B = {B6_final:06b} (decimal {B6_final})")
    print("  A columns (6 bits each):")
    print(f"    col0 = {A_col0:06b}")
    print(f"    col1 = {A_col1:06b}")
    print(f"    col2 = {A_col2:06b}")
    print(f"    col3 = {A_col3:06b}")
