import cv2
import numpy as np
import os

# Configuration
img_size = (8, 6)  # Target resolution (width, height)
output_dir = "frames/scaled_down/"  # Changed to binary frames directory
os.makedirs(output_dir, exist_ok=True)
framename = "scaleddown_"

# Initialize video capture and frame storage
vidcap = cv2.VideoCapture('bad_apple.mp4')
frame_bits = []
count = 0

while True:
    success, image = vidcap.read()
    if not success:
        break

    resized_img = cv2.resize(image, img_size, interpolation=cv2.INTER_AREA)

    # Step 2: Convert to grayscale and binarize (0 or 255)
    gray = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)  # Now outputs 0 or 255

    scaled_down = cv2.resize(binary, (480,360), interpolation=cv2.INTER_AREA)
    # Save purely black and white image (0 or 255 values only)
    cv2.imwrite(f"{output_dir}{framename}{count}.png", scaled_down)

    # Step 3: Convert to 0/1 for binary array
    binary_01 = (binary / 255).astype(np.uint8)  # Convert 255->1, 0->0
    frame_bits.append(binary_01.flatten())

    # Progress update
    if count % 100 == 0:
        print(f'Processed frame {count}')

    count += 1

vidcap.release()

# Save all binary frames as single numpy array
frame_array = np.vstack(frame_bits)
np.save("bad_apple_frames.npy", frame_array)

print(f"Finished! Processed {count} frames.")
print(f"Final array shape: {frame_array.shape}")  # Should be (5400, 192)
