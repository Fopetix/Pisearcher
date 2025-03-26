import time

def fast_even_odd_convert(input_path, output_path):
    lookup = [b'0'] * 256
    for i in range(10):
        lookup[48 + i] = b'1' if i % 2 == 0 else b'0'

    with open(input_path, 'rb') as infile, open(output_path, 'wb') as outfile:
        chunk_size = 1024 * 1024 * 8
        while True:
            chunk = infile.read(chunk_size)
            if not chunk:
                break
            converted = b''.join(lookup[b] for b in chunk)
            outfile.write(converted)

if __name__ == "__main__":
    start = time.time()
    fast_even_odd_convert("pi_big.txt", "output.txt")
    print(f"Processed in {time.time() - start:.4f} seconds")