import torch
import torch.nn as nn
import torch.optim as optim

# ---------------------------------------------------------
# 1) Define the DES S-box (S-box 1) and data set
# ---------------------------------------------------------
S_BOX = [
    [14,  4, 13,  1,  2, 15, 11,  8,  3, 10,  6, 12,  5,  9,  0,  7],
    [ 0, 15,  7,  4, 14,  2, 13,  1, 10,  6, 12, 11,  9,  5,  3,  8],
    [ 4,  1, 14,  8, 13,  6,  2, 11, 15, 12,  9,  7,  3, 10,  5,  0],
    [15, 12,  8,  2,  4,  9,  1,  7,  5, 11,  3, 14, 10,  0,  6, 13]
]

def apply_sbox_6to4(x_6bit):
    """
    x_6bit: integer in [0..63], representing 6 bits.
    Returns an integer in [0..15], representing 4 bits (the S-box output).
    """
    row = ((x_6bit & 0b100000) >> 4) | (x_6bit & 0b000001)  # bits (x5, x0)
    col = (x_6bit & 0b011110) >> 1
    return S_BOX[row][col]

def int_to_6bits(x):
    """Convert 0..63 into a [6]-dim tensor of 0/1 floats."""
    return [float((x >> i) & 1) for i in range(6)]

def int_to_4bits(x):
    """Convert 0..15 into a [4]-dim tensor of 0/1 floats."""
    return [float((x >> i) & 1) for i in range(4)]

# Build a dataset of all 64 possible 6-bit inputs
# X_data: shape [64, 6], Y_data: shape [64, 4]
X_data = []
Y_data = []
for x in range(64):
    X_data.append(int_to_6bits(x))
    s_val = apply_sbox_6to4(x)
    Y_data.append(int_to_4bits(s_val))

X_data = torch.tensor(X_data)  # shape [64,6], dtype=float
Y_data = torch.tensor(Y_data)  # shape [64,4], dtype=float

# ---------------------------------------------------------
# 2) Define a simple model: Linear -> Sigmoid
#    This yields 4 outputs in (0,1).
# ---------------------------------------------------------
model = nn.Sequential(
    nn.Linear(6, 4),  # from 6 -> 4 real-valued
    nn.Sigmoid()      # output each bit's probability
)

# We'll use Binary Cross Entropy over the 4 output bits
criterion = nn.BCELoss()

# Simple optimizer
optimizer = optim.SGD(model.parameters(), lr=0.1)

# ---------------------------------------------------------
# 3) Train in a loop (batch gradient descent, all 64 at once)
# ---------------------------------------------------------
NUM_EPOCHS = 2000

for epoch in range(NUM_EPOCHS):
    optimizer.zero_grad()
    output = model(X_data)          # shape [64,4]
    loss = criterion(output, Y_data)
    loss.backward()
    optimizer.step()

    if (epoch+1) % 200 == 0:
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, loss={loss.item():.6f}")

# ---------------------------------------------------------
# 4) Evaluate the final mismatch
# ---------------------------------------------------------
with torch.no_grad():
    output = model(X_data)  # shape [64,4], each in (0,1)
    # We'll threshold at 0.5 to get final bits
    predicted_bits = (output >= 0.5).int()  # shape [64,4], each 0 or 1

# Reconstruct 4-bit int from predicted bits
def bits4_to_int(bits4):
    return (bits4[0] << 0) | (bits4[1] << 1) | (bits4[2] << 2) | (bits4[3] << 3)

mismatch_count = 0
for i in range(64):
    pred_4 = predicted_bits[i].tolist()  # [0..1, 0..1, 0..1, 0..1]
    pred_val = bits4_to_int(pred_4)
    real_val = apply_sbox_6to4(i)
    if pred_val != real_val:
        mismatch_count += 1

print(f"\nFinal mismatch = {mismatch_count} out of 64 inputs")
print("Some final predictions:")
for i in range(5):
    print(f" Input={i:02d} (binary={i:06b}), S-Box={apply_sbox_6to4(i):04b}, "
          f"Approx={bits4_to_int(predicted_bits[i].tolist()):04b}")
