from controller import Robot, Display
from Basic_Pixel_Processing import gray_scale, gaussian_blur, edge_detection, hysteresis, normalize
from Blob import blobize
from optical_flow import optical_flow
import numpy as np
from PIL import Image
import time


# Initialize Robot
robot = Robot()
timestep = int(robot.getBasicTimeStep())

# Setup Camera
camera = robot.getDevice("camera")
camera.enable(timestep)
width = camera.getWidth()
height = camera.getHeight()

# Setup Display
display = robot.getDevice("display")

print("Vision system started...")

k1 = 1
k2 = 1
k3 = 1
MOVING_FLOW_THRESHOLD = 0.4

# --- Setup ---
# Wait for the first simulation step to get camera data
robot.step(timestep)

t = time.time()
# --- Capture ---
raw_image = camera.getImage()

if raw_image:
    # Convert Webots raw bytes (BGRA) to NumPy (H, W, 4)
    img_arr = np.frombuffer(raw_image, dtype=np.uint8).reshape((height, width, 4))
    # Drop Alpha, Convert BGR to RGB
    prev_frame_arr = img_arr[:, :, :3][:, :, ::-1]

# --- Processing Frame ---
gray_prev_frame = gray_scale(prev_frame_arr, method='luminosity') # Convert to Grayscale
blurred_prev_frame = gaussian_blur(gray_prev_frame) # Apply Gaussian Blur
edges_prev_frame = edge_detection(blurred_prev_frame) # Perform Edge Detection
normalized_prev_frame = normalize(edges_prev_frame) # Normalize edges to range 0-255 for hysteresis
hysteresis_prev_frame = hysteresis(normalized_prev_frame, weak=30, strong=100) # Apply Hysteresis Thresholding

prev_frame_blobs = blobize(prev_frame_arr,hysteresis_prev_frame)

# Maintain ~30 FPS
if time.time() - t < 1/30:
    time.sleep((1/30) - (time.time() - t))

def contains_pixels(blob, array):
    if not blob.pixels:
        return False
    
    # Count how many blob pixels are in the thresholded array
    count = 0
    for x in range(array.shape[1]):
        for y in range(array.shape[0]):
            if array[y, x] != 0 and (x, y) in blob.pixels:
                count += 1
    
    # Calculate percentage
    percentage = (count / len(blob.pixels)) * 100
    return percentage > 1

# --- Main Loop ---
while robot.step(timestep) != -1:
    # --- Capture ---
    raw_image = camera.getImage()
    
    if raw_image:
        img_arr = np.frombuffer(raw_image, dtype=np.uint8).reshape((height, width, 4))

        # Drop Alpha, Convert BGR to RGB
        current_frame_arr = img_arr[:, :, :3][:, :, ::-1]
        
    # --- Processing Frame ---
    gray_current_frame = gray_scale(current_frame_arr, method='luminosity') # Convert to Grayscale
    blurred_current_frame = gaussian_blur(gray_current_frame) # Apply Gaussian Blur
    edges_current_frame = edge_detection(blurred_current_frame) # Perform Edge Detection
    normalized_current_frame = normalize(edges_current_frame) # Normalize edges to range 0-255 for hysteresis
    hysteresis_current_frame = hysteresis(normalized_current_frame, weak=30, strong=100) # Apply Hysteresis Thresholding
    current_frame_blobs = blobize(current_frame_arr,hysteresis_current_frame)
    

    # --- Optical Flow ---
    flow = optical_flow(gray_prev_frame, gray_current_frame, window_size=[3,5], max_flow=5)

    processed_img = current_frame_arr.copy()

    for blob in current_frame_blobs:
        blob.update_flow_from_field(flow)
        if np.hypot(blob.avg_u, blob.avg_v) > MOVING_FLOW_THRESHOLD:
            for x, y in blob.pixels:
                if 0 <= x < width and 0 <= y < height:
                    processed_img[y, x] = [255, 0, 0]
    

    # --- Update Previous Frame ---
    gray_prev_frame = gray_current_frame
    blurred_prev_frame = blurred_current_frame
    edges_prev_frame = edges_current_frame
    normalized_prev_frame = normalized_current_frame
    hysteresis_prev_frame = hysteresis_current_frame
    prev_frame_blobs = current_frame_blobs

    # --- Display Output in Webots ---
    # Convert the processed NumPy array back to bytes
    # Ensure the array is type uint8 and contiguous before converting
    img_data = processed_img.astype(np.uint8).tobytes()
    
    # Create a Webots image reference
    # Format must match your array (Display.RGB for 3 channels)
    ir = display.imageNew(img_data, Display.RGB, width, height)
    
    # Paste the image onto the Display device
    display.imagePaste(ir, 0, 0, False)
    
    # CRITICAL: Delete the image reference to free memory
    display.imageDelete(ir)

    # Maintain ~30 FPS
    if time.time() - t < 1/30:
        time.sleep((1/30) - (time.time() - t))
   