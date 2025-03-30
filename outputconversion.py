import numpy as np
import cv2
import os

# Load the saved matches
results = np.load("best_digits.npy")

# Video parameters
FRAME_WIDTH = 12
FRAME_HEIGHT = 9
frame_scale = 45
fps = 29.97
video_size = (FRAME_WIDTH * frame_scale, FRAME_HEIGHT * frame_scale)

# Reshape to original dimensions
reconstructed_frames = results['sequence'].reshape(-1, FRAME_HEIGHT, FRAME_WIDTH)

# Create video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # More reliably available
video = cv2.VideoWriter('output.mp4', fourcc, fps, video_size, isColor=False)

for idx, frame in enumerate(reconstructed_frames):
    # Convert to grayscale
    grayscale_frame = np.where(frame > 0, 255, 0).astype(np.uint8)

    # Scale up with correct aspect ratio
    scaled_frame = cv2.resize(grayscale_frame,
                              video_size,
                              interpolation=cv2.INTER_NEAREST)

    video.write(scaled_frame)

video.release()
print("Video generation complete")