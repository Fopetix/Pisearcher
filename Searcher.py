import os

os.environ["PYTHONUTF8"] = "1"
import sys

sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
from time import time

# Configuration
PI_BITS_FILE = "pi_input/pi_packed.bin"
FRAMES_FILE = "bad_apple_frames.npy"
OUTPUT_FILE = "best_digits.npy"
PI_LENGTH = 1_000_000_000
FRAME_SIZE = 108

# Load data with validation
pi_data = np.fromfile(PI_BITS_FILE, dtype=np.uint32)
actual_pi_bits = len(pi_data) * 32
assert PI_LENGTH <= actual_pi_bits, \
    f"PI_LENGTH ({PI_LENGTH}) exceeds loaded bits ({actual_pi_bits})"

frames = np.load(FRAMES_FILE).astype(np.uint8)
num_frames = len(frames)

# CUDA kernel with corrected memory handling
cuda_code = """
__global__ void find_best_matches(
    const uint32_t* __restrict__ pi_bits,
    const uint8_t* __restrict__ frames,
    int* __restrict__ best_dists,
    long long* __restrict__ best_pos,
    int num_frames,
    long long pi_length
) {
    extern __shared__ char s_buff[];
    int* s_dist = (int*)s_buff;
    long long* s_pos = (long long*)(s_buff + blockDim.x * sizeof(int));

    int frame_idx = blockIdx.x;
    int tid = threadIdx.x;
    int local_best_dist = 108;
    long long local_best_pos = -1;

    // Load target frame
    __shared__ uint8_t target[108];
    if(tid < 108) {
        target[tid] = frames[frame_idx * 108 + tid];
    }
    __syncthreads();

    // Main search loop
    for(long long pos = tid; pos < pi_length - 108; pos += blockDim.x) {
        int dist = 0;
        int chunk_idx = pos >> 5;
        int bit_offset = 31 - (pos % 32);

        for(int i=0; i<108; i++) {
            uint32_t chunk = __ldg(&pi_bits[chunk_idx]);
            int pi_bit = (chunk >> bit_offset) & 1;
            dist += abs(pi_bit - target[i]);

            if(--bit_offset < 0) {
                chunk_idx++;
                bit_offset = 31;
            }
        }

        if(dist < local_best_dist) {
            local_best_dist = dist;
            local_best_pos = pos;
        }
    }

    // Store in shared memory
    s_dist[tid] = local_best_dist;
    s_pos[tid] = local_best_pos;
    __syncthreads();

    // Parallel reduction
    for(int s = blockDim.x/2; s > 0; s >>= 1) {
        if(tid < s) {
            if(s_dist[tid + s] < s_dist[tid]) {
                s_dist[tid] = s_dist[tid + s];
                s_pos[tid] = s_pos[tid + s];
            }
        }
        __syncthreads();
    }

    // Write result
    if(tid == 0) {
        best_dists[frame_idx] = s_dist[0];
        best_pos[frame_idx] = s_pos[0];
    }
}
"""

# Compile CUDA kernel
mod = SourceModule(cuda_code)
search_func = mod.get_function("find_best_matches")

# GPU memory allocation (correct types)
best_dists = np.full(num_frames, 108, dtype=np.int32)
best_pos = np.full(num_frames, -1, dtype=np.int64)

pi_gpu = cuda.mem_alloc(pi_data.nbytes)
frames_gpu = cuda.mem_alloc(frames.nbytes)
best_dists_gpu = cuda.mem_alloc(best_dists.nbytes)
best_pos_gpu = cuda.mem_alloc(best_pos.nbytes)

# Transfer data to GPU
cuda.memcpy_htod(pi_gpu, pi_data)
cuda.memcpy_htod(frames_gpu, frames)
cuda.memcpy_htod(best_dists_gpu, best_dists)
cuda.memcpy_htod(best_pos_gpu, best_pos)

# Kernel configuration
block_size = 512
grid_size = num_frames
shared_mem = block_size * (4 + 8)  # int + long long per thread

# Execute search
print(f"Searching {num_frames} frames in {PI_LENGTH:,} pi bits...")
start_time = time()

search_func(
    pi_gpu,
    frames_gpu,
    best_dists_gpu,
    best_pos_gpu,
    np.int32(num_frames),
    np.int64(PI_LENGTH),
    block=(block_size, 1, 1),
    grid=(grid_size, 1),
    shared=shared_mem
)

# Retrieve results
cuda.memcpy_dtoh(best_dists, best_dists_gpu)
cuda.memcpy_dtoh(best_pos, best_pos_gpu)

# Validate positions
valid = (best_pos >= 0) & (best_pos <= PI_LENGTH - FRAME_SIZE)
valid_indices = np.where(valid)[0]
valid_pos = best_pos[valid]

# Prepare final results
final_results = np.zeros(num_frames, dtype=[
    ('distance', np.int32),
    ('position', np.int64),
    ('sequence', np.uint8, (FRAME_SIZE,))
])

final_results['distance'] = best_dists
final_results['position'] = best_pos

# Bit extraction for valid positions
if len(valid_pos) > 0:
    bit_positions = valid_pos[:, np.newaxis] + np.arange(FRAME_SIZE)
    bit_positions = np.clip(bit_positions, 0, PI_LENGTH - 1)

    chunk_indices = bit_positions // 32
    bits_in_chunk = 31 - (bit_positions % 32)

    chunks = pi_data[chunk_indices]
    bits = (chunks >> bits_in_chunk) & 1
    final_results['sequence'][valid_indices] = bits.astype(np.uint8)

# Save results
np.save(OUTPUT_FILE, final_results)

# Analysis
print(f"Completed in {time() - start_time:.2f} seconds")
print(f"Valid positions found: {len(valid_pos)}/{num_frames}")

if len(valid_pos) > 0:
    hamming_distances = final_results['distance'][valid_indices]
    avg_h = 1 - (np.mean(hamming_distances) / 108)
    best_h = 1 - (np.min(hamming_distances) / 108)
    print(f"Best match similarity: {best_h:.2%}")
    print(f"Average similarity: {avg_h:.2%}")
else:
    print("No valid positions found!")

# Cleanup
del pi_gpu, frames_gpu, best_dists_gpu, best_pos_gpu
cuda.Context.synchronize()