from controller import Robot, Display, Keyboard
from Basic_Pixel_Processing import gray_scale, gaussian_blur, edge_detection, hysteresis, normalize
from Blob import blobize
from optical_flow import optical_flow
import numpy as np
from PIL import Image
import time
import math


# Initialize Robot
robot = Robot()
timestep = int(robot.getBasicTimeStep())
wheel_radius = 0.033

# Initialize motors
left_motor = robot.getDevice('left wheel motor')
right_motor = robot.getDevice('right wheel motor')
left_motor.setPosition(float('inf'))
right_motor.setPosition(float('inf'))
left_motor.setVelocity(0.0)
right_motor.setVelocity(0.0)

# Initialize lidar
lidar = robot.getDevice("SickLms291") 
lidar.enable(timestep)
lidar.enablePointCloud()

# Enable Keyboard
keyboard = robot.getKeyboard()
keyboard.enable(timestep)

# Setup Position Sensor
left_ps = robot.getDevice("left wheel sensor")
right_ps = robot.getDevice("right wheel sensor")
left_ps.enable(timestep)
right_ps.enable(timestep)
left_ps_last = 0.0
right_ps_last = 0.0
first_step = True

# Setup imu
imu = robot.getDevice("inertial unit")
imu.enable(timestep)

# Setup gyro
gyro = robot.getDevice("gyro")
gyro.enable(timestep)

# Setup Camera
camera = robot.getDevice("camera")
camera.enable(timestep)
width = camera.getWidth()
height = camera.getHeight()
fov = camera.getFov()

f_x = width / (2.0 * math.tan(fov / 2.0))
f_y = height / (2.0 * math.tan(fov / 2.0))


# Setup Display
display = robot.getDevice("display")

print("Vision system started...")

MOVING_FLOW_THRESHOLD = 0.4
MIN_DEPTH_METERS = 0.10
MAX_COMPENSATED_FLOW = 5.0
MAX_WHEEL_SPEED = 1.0
TURN_WHEEL_SPEED = 3.14/8


def keyboard_drive_command(keyboard_device):
    left_speed = 0.0
    right_speed = 0.0

    key = keyboard_device.getKey()
    while key != -1:
        if key in (ord('W'), ord('w'), Keyboard.UP):
            left_speed = MAX_WHEEL_SPEED
            right_speed = MAX_WHEEL_SPEED
        elif key in (ord('S'), ord('s'), Keyboard.DOWN):
            left_speed = -MAX_WHEEL_SPEED
            right_speed = -MAX_WHEEL_SPEED
        elif key in (ord('A'), ord('a'), Keyboard.LEFT):
            left_speed = -TURN_WHEEL_SPEED
            right_speed = TURN_WHEEL_SPEED
        elif key in (ord('D'), ord('d'), Keyboard.RIGHT):
            left_speed = TURN_WHEEL_SPEED
            right_speed = -TURN_WHEEL_SPEED
        elif key == ord(' '):
            left_speed = 0.0
            right_speed = 0.0

        key = keyboard_device.getKey()

    return left_speed, right_speed


def estimate_forward_depth(lidar_device):
    ranges = np.asarray(lidar_device.getRangeImage(), dtype=np.float32)
    if ranges.size == 0:
        return 1.0

    max_range = float(lidar_device.getMaxRange())

    center_start = ranges.size // 3
    center_end = (2 * ranges.size) // 3
    center_ranges = ranges[center_start:center_end]

    valid_center = (
        np.isfinite(center_ranges)
        & (center_ranges > MIN_DEPTH_METERS)
        & (center_ranges < max_range)
    )
    if np.any(valid_center):
        return float(np.median(center_ranges[valid_center]))

    valid_all = np.isfinite(ranges) & (ranges > MIN_DEPTH_METERS) & (ranges < max_range)
    if np.any(valid_all):
        return float(np.median(ranges[valid_all]))

    return 1.0


def estimate_ego_flow(vx, wz, depth_z, dt, u_grid, v_grid, fx, fy):
    depth_z = max(float(depth_z), MIN_DEPTH_METERS)

    u_dot = (-(vx * fx) / depth_z) + (wz * v_grid)

    translation_v = np.zeros_like(v_grid, dtype=np.float32)
    valid_u = np.abs(u_grid) >= 1.0
    translation_v[valid_u] = (
        -(vx * fy * v_grid[valid_u]) / (depth_z * u_grid[valid_u])
    )
    v_dot = translation_v - (wz * u_grid)

    ego_flow = np.zeros((height, width, 2), dtype=np.float32)
    ego_flow[:, :, 0] = u_dot * dt
    ego_flow[:, :, 1] = v_dot * dt
    return ego_flow


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
hysteresis_prev_frame = hysteresis(normalized_prev_frame, weak=60, strong=100) # Apply Hysteresis Thresholding

prev_frame_blobs = blobize(prev_frame_arr,hysteresis_prev_frame)

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

# get initial simulation time
prev_time = robot.getTime()

u_axis = np.arange(width, dtype=np.float32) - ((width - 1) / 2.0)
v_axis = np.arange(height, dtype=np.float32) - ((height - 1) / 2.0)
u_grid, v_grid = np.meshgrid(u_axis, v_axis)

# --- Main Loop ---
while robot.step(timestep) != -1:

    # calculate time delta
    current_time = robot.getTime()
    dt = current_time - prev_time
    prev_time = current_time
    if dt <= 0:
        continue

    # Obtain initial wheel encoder values
    if first_step:
        left_ps_last = left_ps.getValue()
        right_ps_last = right_ps.getValue()
        first_step = False
        continue

    left_ps_current = left_ps.getValue()
    right_ps_current = right_ps.getValue()
    
    dl = (left_ps_current - left_ps_last)
    dr = (right_ps_current - right_ps_last)

    vx = (dl + dr) * wheel_radius / (2 * dt)
    left_ps_last = left_ps_current
    right_ps_last = right_ps_current

    # obtain wz from gyro
    current_rz_velocity = gyro.getValues()[2]
    depth_z = estimate_forward_depth(lidar)

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
    flow = optical_flow(gray_prev_frame, gray_current_frame, window_size=[7,9,12], max_flow=5)
    ego_flow = estimate_ego_flow(vx, current_rz_velocity, depth_z, dt, u_grid, v_grid, f_x, f_y)
    compensated_flow = flow - ego_flow
    np.clip(compensated_flow, -MAX_COMPENSATED_FLOW, MAX_COMPENSATED_FLOW, out=compensated_flow)

    processed_img = current_frame_arr.copy()

    for blob in current_frame_blobs:
        blob.update_flow_from_field(compensated_flow)
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

    l_speed, r_speed = keyboard_drive_command(keyboard)

    left_motor.setVelocity(l_speed)
    right_motor.setVelocity(r_speed)


    # Maintain ~30 FPS
    if time.time() - t < 1/30:
        time.sleep((1/30) - (time.time() - t))
   