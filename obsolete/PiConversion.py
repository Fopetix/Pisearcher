import numpy as np

#read file and replace non numbers
with open("../pi_input/pi_hex.txt", "r") as f:
    hex_str = f.read().strip().replace("\n", "").replace(" ", "").replace(".", "")

#converting hex to bin
binary_str = "".join([f"{int(c, 16):04b}" for c in hex_str])

#packing bits
if len(binary_str) % 32 != 0:
    binary_str += "0" * (32 - len(binary_str) % 32)  # Pad to 32-bit alignment
pi_packed = np.array([int(binary_str[i:i+32], 2) for i in range(0, len(binary_str), 32)], dtype=np.uint32)
pi_packed.tofile("pi_input/pi_packed_binary.bin")