import time
import numpy as np


def ascii_digit_to_packed_binary(input_path, output_path):
    # Create lookup table: ASCII '0'-'9' -> 0/1
    lookup = np.zeros(256, dtype=np.uint8)
    for i in range(10):
        lookup[48 + i] = i % 2  # 48 is ASCII '0'

    buffer = np.empty(0, dtype=np.uint8)

    # Use Latin-1 encoding to read ALL bytes safely
    with open(input_path, 'r', encoding='latin-1') as infile, \
            open(output_path, 'wb') as outfile:

        while True:
            # Read text in chunks
            chunk = infile.read(1024 * 1024*2)  # 1MB chunks
            if not chunk:
                break

            # Convert characters to bytes
            raw_bytes = chunk.encode('latin-1')
            converted = lookup[np.frombuffer(raw_bytes, dtype=np.uint8)]
            buffer = np.concatenate([buffer, converted])

            # Process full 32-bit groups
            full_groups = len(buffer) // 32
            if full_groups > 0:
                bits = buffer[:full_groups * 32].reshape(-1, 32)

                # Pack to 4 bytes with big-endian bit order
                packed = np.packbits(bits, axis=1, bitorder='big')

                # Convert to uint32 with matching endianness
                uint32s = packed.view('>u4').astype(np.uint32)  # Big -> native
                outfile.write(uint32s.tobytes())

                buffer = buffer[full_groups * 32:]

        # Process remaining bits with zero-padding
        if len(buffer) > 0:
            pad_length = 32 - (len(buffer) % 32)
            padded = np.concatenate([buffer, np.zeros(pad_length, dtype=np.uint8)])
            packed = np.packbits(padded.reshape(-1, 32), axis=1, bitorder='big')
            uint32s = packed.view('>u4').astype(np.uint32)
            outfile.write(uint32s.tobytes())


if __name__ == "__main__":
    start = time.time()
    ascii_digit_to_packed_binary("pi_input/pi_dec_10b.txt", "pi_input/pi_packed.bin")
    print(f"Processed in {time.time() - start:.2f} seconds")