"""
Integrated Visual-GraphSLAM Controller for Webots
Features:
- Graph-based SLAM (Nodes & Edges) for infinite map expansion.
- Optical Flow + Blob detection for dynamic object masking.
- Lidar-to-Camera projection to reject moving objects as landmarks.
- Real-time Matplotlib SLAM Visualization & Webots CV Display.
"""

from controller import Supervisor, Display, Keyboard
from Basic_Pixel_Processing import gray_scale, gaussian_blur, edge_detection, hysteresis, normalize
from Blob import blobize
from optical_flow import optical_flow
import numpy as np
import time
import math
import matplotlib.pyplot as plt

# --- 1. Graph Data Structures ---
class PoseNode:
    def __init__(self, id, x, y, theta):
        self.id = id
        self.x = x
        self.y = y
        self.theta = theta

class LandmarkNode:
    def __init__(self, id, x, y):
        self.id = id
        self.x = x
        self.y = y

class OdometryEdge:
    def __init__(self, from_id, to_id, dx, dy, dtheta):
        self.from_id = from_id
        self.to_id = to_id
        self.dx = dx
        self.dy = dy
        self.dtheta = dtheta

class MeasurementEdge:
    def __init__(self, pose_id, landmark_id, local_x, local_y):
        self.pose_id = pose_id
        self.landmark_id = landmark_id
        self.local_x = local_x 
        self.local_y = local_y

# --- 2. Matplotlib Visualization Setup ---
plt.ion() 
fig, ax = plt.subplots(figsize=(8, 8))
fig.canvas.manager.set_window_title('Dynamic-Object Aware GraphSLAM')

def draw_graph(poses, landmarks, odom_edges, meas_edges, true_traj):
    ax.clear()
    ax.set_title("Live GraphSLAM (Dynamic Objects Ignored)")
    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")
    ax.grid(True, linestyle='--', alpha=0.6)

    if true_traj:
        tx, ty = [p[0] for p in true_traj], [p[1] for p in true_traj]
        ax.plot(tx, ty, color='orange', linestyle='--', linewidth=3, alpha=0.8, label="Ground Truth")

    for edge in odom_edges:
        p1, p2 = poses[edge.from_id], poses[edge.to_id]
        ax.plot([p1.x, p2.x], [p1.y, p2.y], color='blue', linestyle='-', alpha=0.4)

    for edge in meas_edges:
        p, lm = poses[edge.pose_id], landmarks[edge.landmark_id]
        ax.plot([p.x, lm.x], [p.y, lm.y], color='green', linestyle=':', alpha=0.4)

    if poses:
        px, py = [p.x for p in poses], [p.y for p in poses]
        ax.scatter(px, py, color='blue', s=10, label="Estimated Path")
        ax.scatter(poses[-1].x, poses[-1].y, color='black', s=50, marker='o')

    if landmarks:
        lx, ly = [lm.x for lm in landmarks], [lm.y for lm in landmarks]
        ax.scatter(lx, ly, color='red', s=100, marker='*', label="Static Landmarks")

    ax.legend(loc='upper right')
    ax.axis('equal') 
    fig.canvas.draw()
    fig.canvas.flush_events()

# --- 3. SLAM Optimization ---
def optimize_graph(poses, landmarks, o_edges, m_edges, iterations=10):
    for _ in range(iterations):
        for edge in o_edges:
            p_from, p_to = poses[edge.from_id], poses[edge.to_id]
            exp_x = p_from.x + (edge.dx * math.cos(p_from.theta))
            exp_y = p_from.y + (edge.dx * math.sin(p_from.theta))
            exp_theta = p_from.theta + edge.dtheta 
            
            err_x = exp_x - p_to.x
            err_y = exp_y - p_to.y
            err_theta = math.atan2(math.sin(exp_theta - p_to.theta), math.cos(exp_theta - p_to.theta))
            
            if p_from.id != 0:
                p_from.x -= err_x * 0.05
                p_from.y -= err_y * 0.05
                p_from.theta -= err_theta * 0.05 
                
            p_to.x += err_x * 0.05
            p_to.y += err_y * 0.05
            p_to.theta += err_theta * 0.05       

        for edge in m_edges:
            pose, lm = poses[edge.pose_id], landmarks[edge.landmark_id]
            exp_lm_x = pose.x + (edge.local_x * math.cos(pose.theta) - edge.local_y * math.sin(pose.theta))
            exp_lm_y = pose.y + (edge.local_x * math.sin(pose.theta) + edge.local_y * math.cos(pose.theta))
            
            err_x, err_y = exp_lm_x - lm.x, exp_lm_y - lm.y
            
            if pose.id != 0:
                pose.x -= err_x * 0.1
                pose.y -= err_y * 0.1
            lm.x += err_x * 0.1
            lm.y += err_y * 0.1

# --- CV Helper Functions ---
def estimate_forward_depth(point_cloud):
    # Extracts median forward depth from lidar point cloud
    forward_pts = [p.x for p in point_cloud if not math.isinf(p.x) and abs(p.y) < 0.2 and p.x > 0.1]
    if forward_pts:
        return float(np.median(forward_pts))
    return 1.0

def estimate_column_depth_from_lidar(lidar_device, u_axis, fx, min_depth=0.10):
    ranges = np.asarray(lidar_device.getRangeImage(), dtype=np.float32)
    if ranges.size == 0:
        return np.full(u_axis.shape, 1.0, dtype=np.float32)

    max_range = float(lidar_device.getMaxRange())
    valid = np.isfinite(ranges) & (ranges > min_depth) & (ranges < max_range)
    if not np.any(valid):
        return np.full(u_axis.shape, 1.0, dtype=np.float32)

    idx = np.arange(ranges.size, dtype=np.float32)
    if np.count_nonzero(valid) == 1:
        range_filled = np.full(ranges.shape, float(ranges[valid][0]), dtype=np.float32)
    else:
        range_filled = np.interp(idx, idx[valid], ranges[valid]).astype(np.float32)

    lidar_fov = float(lidar_device.getFov())
    lidar_angles = np.linspace(-lidar_fov / 2.0, lidar_fov / 2.0, ranges.size, dtype=np.float32)

    # Camera horizontal ray angle alpha follows y/x = -u/fx in robot frame.
    pixel_angles = -np.arctan2(u_axis, float(fx)).astype(np.float32)
    clipped_angles = np.clip(pixel_angles, lidar_angles[0], lidar_angles[-1])
    sampled_ranges = np.interp(clipped_angles, lidar_angles, range_filled).astype(np.float32)

    depth_x_cols = sampled_ranges * np.cos(pixel_angles)
    return np.maximum(depth_x_cols, min_depth)


def estimate_blob_depth_x(blob, depth_x_cols, depth_percentile=15.0):
    blob.finalize_pixels()
    if blob.pixels_np is None or blob.pixels_np.size == 0:
        return float(np.percentile(depth_x_cols, depth_percentile))

    xs = blob.pixels_np[:, 0]
    valid = (xs >= 0) & (xs < depth_x_cols.shape[0])
    if not np.any(valid):
        return float(np.percentile(depth_x_cols, depth_percentile))

    return float(np.percentile(depth_x_cols[xs[valid]], depth_percentile))


def estimate_ego_flow(vx, wz, depth_x_cols, dt, u_grid, v_grid, fx, fy, height, width):
    depth_x = np.maximum(depth_x_cols[np.newaxis, :], 0.10)

    # Yaw-flow model for a forward-facing pinhole camera (x-forward in robot frame).
    fx_safe = max(float(fx), 1e-6)
    u_dot = (vx / depth_x) * u_grid + wz * (fx_safe + (u_grid * u_grid) / fx_safe)
    v_dot = (vx / depth_x) * v_grid + wz * ((u_grid * v_grid) / fx_safe)

    ego_flow = np.zeros((height, width, 2), dtype=np.float32)
    ego_flow[:, :, 0] = u_dot * dt
    ego_flow[:, :, 1] = v_dot * dt
    return ego_flow


def estimate_ego_scale(observed_flow, predicted_flow, min_pred_mag=0.10, scale_min=0.01, scale_max=0.05):
    pred_u = predicted_flow[:, :, 0]
    pred_v = predicted_flow[:, :, 1]
    pred_mag = np.hypot(pred_u, pred_v)
    valid = pred_mag > min_pred_mag
    if not np.any(valid):
        return 1.0

    obs_u = observed_flow[:, :, 0]
    obs_v = observed_flow[:, :, 1]
    dot = (obs_u * pred_u) + (obs_v * pred_v)
    denom = (pred_u * pred_u) + (pred_v * pred_v) + 1e-6

    ratios = dot[valid] / denom[valid]
    ratios = ratios[np.isfinite(ratios)]
    if ratios.size == 0:
        return 1.0

    low, high = np.percentile(ratios, [10.0, 90.0])
    inliers = ratios[(ratios >= low) & (ratios <= high)]
    if inliers.size == 0:
        inliers = ratios

    return float(np.clip(np.median(inliers), scale_min, scale_max))


def compensation_cost(observed_flow, predicted_flow, min_pred_mag=0.10):
    residual = observed_flow - predicted_flow
    residual_mag = np.hypot(residual[:, :, 0], residual[:, :, 1])

    pred_mag = np.hypot(predicted_flow[:, :, 0], predicted_flow[:, :, 1])
    valid = pred_mag > min_pred_mag
    if np.any(valid):
        return float(np.percentile(residual_mag[valid], 45.0))
    return float(np.percentile(residual_mag, 45.0))


def estimate_best_ego_flow(observed_flow, vx, wz, depth_x_cols, dt, u_grid, v_grid, fx, fy, height, width):
    ego_plus = estimate_ego_flow(vx, wz, depth_x_cols, dt, u_grid, v_grid, fx, fy, height, width)
    ego_minus = estimate_ego_flow(vx, -wz, depth_x_cols, dt, u_grid, v_grid, fx, fy, height, width)

    scale_plus = estimate_ego_scale(observed_flow, ego_plus)
    scale_minus = estimate_ego_scale(observed_flow, ego_minus)
    pred_plus = ego_plus * scale_plus
    pred_minus = ego_minus * scale_minus

    cost_plus = compensation_cost(observed_flow, pred_plus)
    cost_minus = compensation_cost(observed_flow, pred_minus)

    if cost_plus <= cost_minus:
        return pred_plus, 1.0, scale_plus
    return pred_minus, -1.0, scale_minus

# --- 4. Initialization ---
robot = Supervisor()
TIME_STEP = int(robot.getBasicTimeStep())
WHEEL_RADIUS = 0.033
MATCH_THRESHOLD = 0.35 
MOVING_FLOW_THRESHOLD = 0.25
MAX_COMPENSATED_FLOW = 100.0
DISTANCE_REF_DEPTH = 2.0
DISTANCE_SPEED_MAX_GAIN = 1.1

keyboard = robot.getKeyboard()
keyboard.enable(TIME_STEP)

robot_node = robot.getSelf()

left_motor = robot.getDevice('left wheel motor')
right_motor = robot.getDevice('right wheel motor')
left_motor.setPosition(float('inf'))
right_motor.setPosition(float('inf'))

left_ps = robot.getDevice('left wheel sensor')
right_ps = robot.getDevice('right wheel sensor')
left_ps.enable(TIME_STEP)
right_ps.enable(TIME_STEP)

inertial_unit = robot.getDevice('inertial unit')
gyro = robot.getDevice('gyro')
inertial_unit.enable(TIME_STEP)
gyro.enable(TIME_STEP)

lidar = robot.getDevice('LDS-01')
if not lidar:
    lidar = robot.getDevice('SickLms291') # Fallback
lidar.enable(TIME_STEP)
lidar.enablePointCloud() 

# front camera
# camera = robot.getDevice("camera")
# camera.enable(TIME_STEP)
# width = camera.getWidth()
# height = camera.getHeight()
# fov = camera.getFov()
# f_x = width / (2.0 * math.tan(fov / 2.0))
# f_y = height / (2.0 * math.tan(fov / 2.0))

# initialize all cameras
camname = ["camera", "back camera", "left camera", "right camera"]
cameras = {}
moving_pixel_sets = {name: set() for name in camname}
for name in camname:
    cam = robot.getDevice(name)
    cam.enable(TIME_STEP)
    cameras[name] = cam
width = cameras["camera"].getWidth()
height = cameras["camera"].getHeight()
f_x = width / (2.0 * math.tan(cameras["camera"].getFov() / 2.0))
f_y = height / (2.0 * math.tan(cameras["camera"].getFov() / 2.0))

display = robot.getDevice("display")
edge_display = robot.getDevice("edge")
edge_width = width
edge_height = height
if edge_display:
    edge_width = edge_display.getWidth()
    edge_height = edge_display.getHeight()

# Graph Variables
pose_nodes = [PoseNode(0, 0.0, 0.0, 0.0)]
landmark_nodes = []
odom_edges = []
meas_edges = []
true_trajectory = []
pose_id_counter = 1
landmark_id_counter = 0

# Odometry Variables
left_ps_last, right_ps_last = 0.0, 0.0
accumulated_dist = 0.0
world_x, world_y = 0.0, 0.0

# CV Grid setup
u_axis = np.arange(width, dtype=np.float32) - ((width - 1) / 2.0)
v_axis = np.arange(height, dtype=np.float32) - ((height - 1) / 2.0)
u_grid, v_grid = np.meshgrid(u_axis, v_axis)

# # Initialize CV First Frame
# robot.step(TIME_STEP)
# raw_image = camera.getImage()
# img_arr = np.frombuffer(raw_image, dtype=np.uint8).reshape((height, width, 4))
# prev_frame_arr = img_arr[:, :, :3][:, :, ::-1]
# gray_prev_frame = gray_scale(prev_frame_arr, method='luminosity')
# prev_time = robot.getTime()
# first_step = True

# initialize CV First Frame for all cameras
prev_gray_frames = {name: None for name in camname}
robot.step(TIME_STEP)
for name in camname:
    raw_img = cameras[name].getImage()
    img_arr = np.frombuffer(raw_img, dtype=np.uint8).reshape((height, width, 4))
    frame_arr = img_arr[:, :, :3][:, :, ::-1]
    prev_gray_frames[name] = gray_scale(frame_arr, method='luminosity')
prev_time = robot.getTime()
first_step = True
frame_index = 0

print("=== Visual-GraphSLAM Started ===")

# --- 5. Main Loop ---
while robot.step(TIME_STEP) != -1:
    frame_index +=1
    current_time = robot.getTime()
    dt = current_time - prev_time
    prev_time = current_time
    if dt <= 0: continue

    if first_step:
        left_ps_last, right_ps_last = left_ps.getValue(), right_ps.getValue()
        first_step = False
        continue

    # Ground Truth Tracking
    if robot_node:
        actual_pos = robot_node.getPosition()
        true_trajectory.append((actual_pos[0], actual_pos[1]))

    # Odometry & Kinematics
    left_ps_curr, right_ps_curr = left_ps.getValue(), right_ps.getValue()
    dl = (left_ps_curr - left_ps_last)
    dr = (right_ps_curr - right_ps_last)
    step_dist = (dl + dr) * WHEEL_RADIUS / 2.0
    vx = step_dist / dt

    left_ps_last, right_ps_last = left_ps_curr, right_ps_curr
    
    rpy = inertial_unit.getRollPitchYaw()
    angle_rad = rpy[2]
    current_rz_velocity = gyro.getValues()[2]
    
    world_x += step_dist * math.cos(angle_rad)
    world_y += step_dist * math.sin(angle_rad)
    accumulated_dist += step_dist

    # # Camera Capture & CV Pipeline
    # raw_image = camera.getImage()
    # current_frame_arr = np.frombuffer(raw_image, dtype=np.uint8).reshape((height, width, 4))[:, :, :3][:, :, ::-1]
    # gray_current_frame = gray_scale(current_frame_arr, method='luminosity')
    
    # blurred_current = gaussian_blur(gray_current_frame)
    # edges_current = edge_detection(blurred_current)
    # hysteresis_current = hysteresis(normalize(edges_current), weak=30, strong=100)
    # current_frame_blobs = blobize(current_frame_arr, hysteresis_current)

    # point_cloud = lidar.getPointCloud()
    # depth_z = estimate_forward_depth(point_cloud)

    # flow = optical_flow(gray_prev_frame, gray_current_frame, window_size=[7,9,12], max_flow=5)
    # ego_flow = estimate_ego_flow(vx, current_rz_velocity, depth_z, dt, u_grid, v_grid, f_x, f_y, height, width)
    # compensated_flow = flow - ego_flow
    # np.clip(compensated_flow, -MAX_COMPENSATED_FLOW, MAX_COMPENSATED_FLOW, out=compensated_flow)

    # processed_img = current_frame_arr.copy()
    # moving_pixels = set()

    # for blob in current_frame_blobs:
    #     blob.update_flow_from_field(compensated_flow)
    #     if np.hypot(blob.avg_u, blob.avg_v) > MOVING_FLOW_THRESHOLD:
    #         for x, y in blob.pixels:
    #             moving_pixels.add((x, y))
    #             if 0 <= x < width and 0 <= y < height:
    #                 processed_img[y, x] = [255, 0, 0] # Highlight moving pixels in Red

    # # Update Webots Display
    # img_data = processed_img.astype(np.uint8).tobytes()
    # ir = display.imageNew(img_data, Display.RGB, width, height)
    # display.imagePaste(ir, 0, 0, False)
    # display.imageDelete(ir)

    # Camera capture & CV pipeline for all cameras
    for name in camname:
        raw_image = cameras[name].getImage()
        current_frame_arr = np.frombuffer(raw_image, dtype=np.uint8).reshape((height, width, 4))[:, :, :3][:, :, ::-1]
        gray_current_frame = gray_scale(current_frame_arr, method='luminosity')
        
        blurred = gaussian_blur(gray_current_frame)
        edges = edge_detection(blurred)
        hyst_curr = hysteresis(normalize(edges), weak=30, strong=100)
        blobs = blobize(current_frame_arr, hyst_curr)

        point_cloud = lidar.getPointCloud()
        # depth_z = estimate_forward_depth(point_cloud)
        depth_x_cols = estimate_column_depth_from_lidar(lidar, u_axis, f_x)

        flow = optical_flow(prev_gray_frames[name], gray_current_frame, window_size=[7,9,12], max_flow=5)
        ego_flow, yaw_sign, ego_scale = estimate_best_ego_flow(
        flow,
        vx,
        current_rz_velocity,
        depth_x_cols,
        dt,
        u_grid,
        v_grid,
        f_x,
        f_y,
        height,
        width,)        
        compensated_flow = flow - ego_flow
        np.clip(compensated_flow, -MAX_COMPENSATED_FLOW, MAX_COMPENSATED_FLOW, out=compensated_flow)
        processed_img = current_frame_arr.copy()
        moving_pixels = set()
        blob_stats = []

        # # 4. Identify moving pixels for THIS specific camera
        # moving_pixel_sets[name].clear() # Clear from previous step
        # for blob in blobs:
        #     blob.update_flow_from_field(compensated_flow)
        #     if np.hypot(blob.avg_u, blob.avg_v) > MOVING_FLOW_THRESHOLD:
        #         for px, py in blob.pixels:
        #             moving_pixel_sets[name].add((px, py))
        moving_pixel_sets[name].clear() # Clear from previous step
        for blob_idx, blob in enumerate(blobs):
            blob.update_flow_from_field(compensated_flow)
            blob_speed = float(np.hypot(blob.avg_u, blob.avg_v))
            blob_depth_x = estimate_blob_depth_x(blob, depth_x_cols)
            depth_gain = np.clip(
                math.sqrt(blob_depth_x / DISTANCE_REF_DEPTH),
                1.0,
                DISTANCE_SPEED_MAX_GAIN,
            )
            boosted_speed = blob_speed * float(depth_gain)
            blob_stats.append((blob_idx, boosted_speed, blob_depth_x))
            if boosted_speed > MOVING_FLOW_THRESHOLD:
                for x, y in blob.pixels:
                    moving_pixel_sets[name].add((x, y))
                    if 0 <= x < width and 0 <= y < height:
                        processed_img[y, x] = [255, 0, 0] # Highlight moving pixels in Red

        fast_blobs = [b for b in blob_stats if b[1] > MOVING_FLOW_THRESHOLD]
        if fast_blobs:
            speeds_text = ", ".join(
                f"blob_{idx}: speed={speed:.3f}, depth={depth:.3f}"
                for idx, speed, depth in fast_blobs
            )
        elif blob_stats:
            fastest_idx, fastest_speed, fastest_depth = max(blob_stats, key=lambda b: b[1])
            speeds_text = (
                f"fastest blob_{fastest_idx}: speed={fastest_speed:.3f}, depth={fastest_depth:.3f}"
            )
        else:
            speeds_text = "none"
        print(
            f"[Frame {frame_index}] blobs: {len(blobs)} | "
            f"yaw_sign: {yaw_sign:+.0f} | ego_scale: {ego_scale:.2f} | "
            f"avg speed (px/frame): {speeds_text}"
        )

        # # 5. Prepare for next iteration
        prev_gray_frames[name] = gray_current_frame

        # Optional: Update Webots Display with the FRONT camera only
        if name == "camera":
            processed_img = current_frame_arr.copy()
            for (px, py) in moving_pixel_sets[name]:
                processed_img[py, px] = [255, 0, 0]
            img_data = processed_img.astype(np.uint8).tobytes()
            ir = display.imageNew(img_data, Display.RGB, width, height)
            display.imagePaste(ir, 0, 0, False)
            display.imageDelete(ir)

    # gray_prev_frames = gray_current_frame

    # --- SLAM GRAPH UPDATE ---
    curr_pose = pose_nodes[-1]
    dtheta_from_last_node = angle_rad - curr_pose.theta
    graph_updated = False 

    if abs(accumulated_dist) > 0.05 or abs(dtheta_from_last_node) > 0.05:
        new_pose = PoseNode(pose_id_counter, world_x, world_y, angle_rad)
        pose_nodes.append(new_pose)
        odom_edges.append(OdometryEdge(pose_id_counter - 1, pose_id_counter, accumulated_dist, 0.0, dtheta_from_last_node))
        pose_id_counter += 1
        accumulated_dist = 0.0
        graph_updated = True

        # Landmark Extraction with Dynamic Object Masking
        if abs(dtheta_from_last_node) < 0.08:
            points = [p for p in point_cloud if not (math.isinf(p.x) or math.isinf(p.y))]
            best_feature = None
            min_dist = float('inf')
            
            step = 3
            for i in range(step, len(points) - step):
                # p_curr = points[i]
                # d_curr = math.hypot(p_curr.x, p_curr.y)
                
                # if not (0.12 < d_curr < 1.5): continue
                    
                # d_left = math.hypot(points[i-step].x, points[i-step].y)
                # d_right = math.hypot(points[i+step].x, points[i+step].y)
                
                # jump_left, jump_right = abs(d_curr - d_left), abs(d_curr - d_right)
                
                # if jump_left > 0.20 or jump_right > 0.20:
                #     # PROJECTION CHECK: Is this Lidar spike inside a moving object?
                #     is_moving = False
                #     if p_curr.x > 0.01:
                #         # Project Lidar 2D local coordinate to Camera pixel 'u'
                #         u = int(width / 2.0 - (p_curr.y / p_curr.x) * f_x)
                #         if 0 <= u < width:
                #             # Check a vertical swath (since Lidar is flat but objects span height)
                #             v_center = height // 2
                #             for v in range(max(0, v_center - 30), min(height, v_center + 30)):
                #                 if (u, v) in moving_pixels:
                #                     is_moving = True
                #                     break
                    
                #     if is_moving:
                #         # Skip this landmark! It is a moving object.
                #         continue

                p_curr = points[i]
                d_curr = math.hypot(p_curr.x, p_curr.y)
                if not (0.12 < d_curr < 1.5): continue
                d_left = math.hypot(points[i-step].x, points[i-step].y)
                d_right = math.hypot(points[i+step].x, points[i+step].y)
                jump_left, jump_right = abs(d_curr - d_left), abs(d_curr - d_right)

                local_angle = math.degrees(math.atan2(p_curr.y, p_curr.x))
                target_cam = None
                if -45 <= local_angle <= 45:
                    target_cam = "camera"
                    u = int(width/2.0 - (p_curr.y / p_curr.x) * f_x)
                elif 45 < local_angle <= 135:
                    target_cam = "right camera"
                    u = int(width/2.0 - (-p_curr.x / p_curr.y) * f_x)
                elif local_angle > 135 or local_angle <= -135:
                    target_cam = "back camera"
                    u = int(width/2.0 - (p_curr.y / p_curr.x) * f_x)
                elif -135 < local_angle < -45:
                    target_cam = "left camera"
                    u = int(width/2.0 - (p_curr.x / -p_curr.y) * f_x)

                # REJECTION LOGIC
                if jump_left > 0.2 or jump_right>0.2:
                    is_moving=False
                    if p_curr.x>0.1:
                        if target_cam and 0 <= u < width:
                            is_moving = False
                            v_center = height // 2
                            # Check the moving_pixels specific to THIS camera
                            for v in range(v_center - 30, v_center + 30):
                                if (u, v) in moving_pixel_sets[target_cam]:
                                    is_moving = True
                                    break
                        
                        if is_moving:
                            continue # Dynamic object detected - do not add as landmark

##################################################

                    if d_curr < min_dist:
                        min_dist = d_curr
                        best_feature = p_curr
                        
            if best_feature:
                current_p = pose_nodes[-1]
                gl_x = current_p.x + (best_feature.x * math.cos(current_p.theta) - best_feature.y * math.sin(current_p.theta))
                gl_y = current_p.y + (best_feature.x * math.sin(current_p.theta) + best_feature.y * math.cos(current_p.theta))
                
                matched_id = -1
                for lm in landmark_nodes:
                    if math.hypot(gl_x - lm.x, gl_y - lm.y) < MATCH_THRESHOLD:
                        matched_id = lm.id
                        break 
                
                if matched_id != -1:
                    meas_edges.append(MeasurementEdge(current_p.id, matched_id, best_feature.x, best_feature.y))
                    optimize_graph(pose_nodes, landmark_nodes, odom_edges, meas_edges)
                    world_x, world_y = pose_nodes[-1].x, pose_nodes[-1].y
                else:
                    new_lm = LandmarkNode(landmark_id_counter, gl_x, gl_y)
                    landmark_nodes.append(new_lm)
                    meas_edges.append(MeasurementEdge(current_p.id, landmark_id_counter, best_feature.x, best_feature.y))
                    landmark_id_counter += 1

    # --- Draw Matplotlib Graph ---
    if graph_updated or (robot.getTime() * 1000) % 500 < TIME_STEP:
        draw_graph(pose_nodes, landmark_nodes, odom_edges, meas_edges, true_trajectory)

    # --- Input Handling ---
    key = keyboard.getKey()
    vL, vR = 0.0, 0.0
    while key != -1:
        if key in [ord('W'), ord('w'), Keyboard.UP]: vL, vR = 1, 1
        elif key in [ord('S'), ord('s'), Keyboard.DOWN]: vL, vR = -1.0, -1.0
        elif key in [ord('A'), ord('a'), Keyboard.LEFT]: vL, vR = -0.2, 0.2
        elif key in [ord('D'), ord('d'), Keyboard.RIGHT]: vL, vR = 0.2, -0.2
        key = keyboard.getKey()

    left_motor.setVelocity(vL)
    right_motor.setVelocity(vR)