from controller import Robot, Display
from Basic_Pixel_Processing import gray_scale, gaussian_blur, edge_detection, hysteresis, normalize
from Blob import blobize, histogram_distance
import numpy as np
from PIL import Image
import time

# Load in the goal
goal_image = Image.open("goal.jpg") # Open the image file
goal_array = np.array(goal_image)

# Find the blob of the goal
gray_goal = gray_scale(goal_array, method='luminosity') # Convert to Grayscale
blurred_goal = gaussian_blur(gray_goal) # Apply Gaussian Blur
edges_goal = edge_detection(blurred_goal) # Perform Edge Detection
normalized_goal = normalize(edges_goal) # Normalize edges to range 0-255 for hysteresis
hysteresis_goal = hysteresis(normalized_goal, weak=30, strong=100) # Apply Hysteresis Thresholding

goal_blob = blobize(goal_array,hysteresis_goal)[1]

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
    return percentage > 25

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
    
    # --- Compare Blobs ---
    Blobs = []
    for i, blob in enumerate(prev_frame_blobs):
        best_distance = float('inf')
        best_match = -1
        for j, other_blob in enumerate(current_frame_blobs):       
            # The normalization now happens automatically inside the distance function
            hc_distance = histogram_distance(blob.color_histogram, other_blob.color_histogram)
            hog_distance = histogram_distance(blob.hog_descriptor, other_blob.hog_descriptor)
            center_distance = np.sqrt((blob.center[0]-other_blob.center[0])**2 + (blob.center[1]-other_blob.center[1])**2)
            total_distance = k1*hc_distance + k2*hog_distance + k3*center_distance
            if total_distance < best_distance:
                best_distance = total_distance
                best_match = j
        # print(f"Blob {i+1} in frame 1 is Blob {best_match+1} in frame 2")
        Blobs.append((blob, current_frame_blobs[best_match]))

    # --- Compare Frames ---
    diff_array = np.abs(blurred_current_frame - blurred_prev_frame) # Background Subtraction
    diff_array = normalize(diff_array) # Normalize to range 0-255
    thresholded_diff_array = hysteresis(diff_array, weak=30, strong=100)

    # --- Highlight Moving Blobs and Goal ---
    processed_img = current_frame_arr.copy()
    for i, blob_pair in enumerate(Blobs):
        hc_distance = histogram_distance(blob_pair[1].color_histogram, goal_blob.color_histogram)
        hog_distance = histogram_distance(blob_pair[1].hog_descriptor, goal_blob.hog_descriptor)
        if hc_distance + hog_distance < 1:
            # print(f"Blob {i+1} is goal")
            for x, y in blob_pair[1].pixels:
                # Highlight in red
                processed_img[y, x] = [0, 255, 0]  # RGB Green
        elif contains_pixels(blob_pair[0],thresholded_diff_array) and contains_pixels(blob_pair[1],thresholded_diff_array):
            for x, y in blob_pair[1].pixels:
                # Highlight in red
                processed_img[y, x] = [255, 0, 0]  # RGB Red

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
   