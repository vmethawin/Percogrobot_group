"""
Autonomous Visualized GraphSLAM Controller for Webots (TurtleBot3 LDS-01)
Features:
- WASD manual driving
- IMU + Encoder Odometry with continuous frame accumulation
- Supervisor Ground Truth Tracking (Orange Dashed Line)
- Automated Landmark Extraction & Iterative Graph Optimization
- Real-time Matplotlib Visualization
"""

from controller import Supervisor, Keyboard
import math
import matplotlib.pyplot as plt
from optical_flow import optical_flow

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
fig.canvas.manager.set_window_title('GraphSLAM Live Map vs Ground Truth')

def draw_graph(poses, landmarks, odom_edges, meas_edges, true_traj):
    """Draws the SLAM graph alongside the Supervisor Ground Truth."""
    ax.clear()
    ax.set_title("Live GraphSLAM vs Actual Position")
    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")
    ax.grid(True, linestyle='--', alpha=0.6)

    # Draw Ground Truth Trajectory (Supervisor)
    if true_traj:
        tx = [p[0] for p in true_traj]
        ty = [p[1] for p in true_traj]
        ax.plot(tx, ty, color='orange', linestyle='--', linewidth=3, alpha=0.8, label="Actual Path (Ground Truth)")

    # Draw Odometry Edges (SLAM Estimated Path)
    for edge in odom_edges:
        p1 = poses[edge.from_id]
        p2 = poses[edge.to_id]
        ax.plot([p1.x, p2.x], [p1.y, p2.y], color='blue', linestyle='-', alpha=0.4)

    # Draw Measurement Edges (Landmark Sightings)
    for edge in meas_edges:
        p = poses[edge.pose_id]
        lm = landmarks[edge.landmark_id]
        ax.plot([p.x, lm.x], [p.y, lm.y], color='green', linestyle=':', alpha=0.4)

    # Draw SLAM Pose Nodes
    if poses:
        px = [p.x for p in poses]
        py = [p.y for p in poses]
        ax.scatter(px, py, color='blue', s=10, label="SLAM Estimated Nodes")
        ax.scatter(poses[-1].x, poses[-1].y, color='black', s=50, marker='o', label="Estimated Robot Pose")

    # Draw Landmark Nodes
    if landmarks:
        lx = [lm.x for lm in landmarks]
        ly = [lm.y for lm in landmarks]
        ax.scatter(lx, ly, color='red', s=100, marker='*', label="Mapped Landmarks")

    ax.legend(loc='upper right')
    ax.axis('equal') 
    fig.canvas.draw()
    fig.canvas.flush_events()

# --- 3. Robot & Supervisor Initialization ---
TIME_STEP = 64
WHEEL_RADIUS = 0.033
WHEEL_BASE = 0.16
MATCH_THRESHOLD = 0.5 

LINEAR_CORRECTION = 1.00  
ANGULAR_CORRECTION = 1.00 

# Change: Use Supervisor instead of Robot
robot = Supervisor()
keyboard = robot.getKeyboard()
keyboard.enable(TIME_STEP)

# Change: Get God-Mode reference to self
robot_node = robot.getSelf()
if robot_node is None:
    print("CRITICAL ERROR: 'supervisor' field in the Webots Scene Tree is set to FALSE!")
    print("Please set it to TRUE, save, and reload the world.")

left_motor = robot.getDevice('left wheel motor')
right_motor = robot.getDevice('right wheel motor')
left_motor.setPosition(float('inf'))
right_motor.setPosition(float('inf'))
left_motor.setVelocity(0.0)
right_motor.setVelocity(0.0)

left_ps = robot.getDevice('left wheel sensor')
right_ps = robot.getDevice('right wheel sensor')
left_ps.enable(TIME_STEP)
right_ps.enable(TIME_STEP)

inertial_unit = robot.getDevice('inertial unit')
if inertial_unit: 
    inertial_unit.enable(TIME_STEP)

lidar = robot.getDevice('LDS-01')
if lidar:
    lidar.enable(TIME_STEP)
    lidar.enablePointCloud() 

# Graph & Tracking State Initialization
pose_nodes = [PoseNode(0, 0.0, 0.0, 0.0)]
landmark_nodes = []
odom_edges = []
meas_edges = []
true_trajectory = [] # New: Array to hold ground truth coordinates

pose_id_counter = 1
landmark_id_counter = 0
first_step = True

left_ps_last, right_ps_last = 0.0, 0.0
accumulated_dist = 0.0
world_x, world_y = 0.0, 0.0

# --- 4. Graph Optimization ---
def optimize_graph(poses, landmarks, o_edges, m_edges, iterations=10):
    for _ in range(iterations):
        # Odometry Optimization
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

        # Measurement Optimization
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

print("=== Autonomous Visualized SLAM Started ===")
draw_graph(pose_nodes, landmark_nodes, odom_edges, meas_edges, true_trajectory)

# --- 5. Main Loop ---
while robot.step(TIME_STEP) != -1:
    if first_step:
        left_ps_last, right_ps_last = left_ps.getValue(), right_ps.getValue()
        first_step = False
        continue

    # --- Fetch Ground Truth ---
    if robot_node:
        actual_pos = robot_node.getPosition()
        # Webots floor is usually the X-Y plane
        true_trajectory.append((actual_pos[0], actual_pos[1]))

    # --- INPUT HANDLING ---
    key = keyboard.getKey()
    vL, vR = 0.0, 0.0
    while key != -1:
        if key in [ord('W'), ord('w')]: vL, vR = 4.0, 4.0
        elif key in [ord('S'), ord('s')]: vL, vR = -4.0, -4.0
        elif key in [ord('A'), ord('a')]: vL, vR = -2.0, 2.0
        elif key in [ord('D'), ord('d')]: vL, vR = 2.0, -2.0
        key = keyboard.getKey()
        
    left_motor.setVelocity(vL)
    right_motor.setVelocity(vR)

    # --- Continuous Odometry Calculation ---
    left_ps_curr, right_ps_curr = left_ps.getValue(), right_ps.getValue()
    rpy = inertial_unit.getRollPitchYaw()
    
    raw_step_dist = ((left_ps_curr - left_ps_last) + (right_ps_curr - right_ps_last)) * WHEEL_RADIUS / 2.0
    left_ps_last, right_ps_last = left_ps_curr, right_ps_curr
    
    step_dist = raw_step_dist * LINEAR_CORRECTION
    angle_rad = rpy[2] * ANGULAR_CORRECTION
    
    world_x += step_dist * math.cos(angle_rad)
    world_y += step_dist * math.sin(angle_rad)
    
    accumulated_dist += step_dist
    curr_pose = pose_nodes[-1]
    dtheta_from_last_node = angle_rad - curr_pose.theta
    
    graph_updated = False 

    # Update Graph Path 
    if abs(accumulated_dist) > 0.05 or abs(dtheta_from_last_node) > 0.05:
        new_pose = PoseNode(pose_id_counter, world_x, world_y, angle_rad)
        pose_nodes.append(new_pose)
        odom_edges.append(OdometryEdge(pose_id_counter - 1, pose_id_counter, accumulated_dist, 0.0, dtheta_from_last_node))
        pose_id_counter += 1
        accumulated_dist = 0.0
        graph_updated = True

        # DATA ASSOCIATION & LANDMARK EXTRACTION
        # 2. DATA ASSOCIATION & LANDMARK EXTRACTION
        if abs(dtheta_from_last_node) < 0.08:
            if lidar:
                point_cloud = lidar.getPointCloud()
                best_feature = None
                min_dist = float('inf')
                
                # Filter out infinite points first
                points = [p for p in point_cloud if not (math.isinf(p.x) or math.isinf(p.y))]
                
                # Look for 'spikes' by comparing a point to its neighbors 
                # (using a step of 3 to skip over minor Lidar noise)
                step = 3
                for i in range(step, len(points) - step):
                    p_curr = points[i]
                    d_curr = math.hypot(p_curr.x, p_curr.y)
                    
                    if not (0.12 < d_curr < 1.5):
                        continue
                        
                    # Get distance of points a few steps to the left and right
                    d_left = math.hypot(points[i-step].x, points[i-step].y)
                    d_right = math.hypot(points[i+step].x, points[i+step].y)
                    
                    # Calculate how sharply the depth changes
                    jump_left = abs(d_curr - d_left)
                    jump_right = abs(d_curr - d_right)
                    # SPIKE THRESHOLD: If depth changes by more than 20cm, it's an edge/object!
                    if jump_left > 0.20 or jump_right > 0.20:
                        if d_curr < min_dist:
                            min_dist = d_curr
                            best_feature = p_curr
                            
                # Rename back to closest_pt so the rest of your code works seamlessly
                closest_pt = best_feature 
                        
                if closest_pt:
                    current_p = pose_nodes[-1]
                    
                    gl_x = current_p.x + (closest_pt.x * math.cos(current_p.theta) - closest_pt.y * math.sin(current_p.theta))
                    gl_y = current_p.y + (closest_pt.x * math.sin(current_p.theta) + closest_pt.y * math.cos(current_p.theta))
                    matched_id = -1
                    for lm in landmark_nodes:
                        dist_to_lm = math.hypot(gl_x - lm.x, gl_y - lm.y)
                        if dist_to_lm < MATCH_THRESHOLD:
                            matched_id = lm.id
                            break 
                    
                    if matched_id != -1:
                        meas_edges.append(MeasurementEdge(current_p.id, matched_id, closest_pt.x, closest_pt.y))
                        print(f"🔄 MATCH! Recognized Landmark #{matched_id}. Optimizing Graph...")
                        optimize_graph(pose_nodes, landmark_nodes, odom_edges, meas_edges)
                        world_x, world_y = pose_nodes[-1].x, pose_nodes[-1].y
                        graph_updated = True
                    else:
                        new_lm = LandmarkNode(landmark_id_counter, gl_x, gl_y)
                        landmark_nodes.append(new_lm)
                        meas_edges.append(MeasurementEdge(current_p.id, landmark_id_counter, closest_pt.x, closest_pt.y))
                        landmark_id_counter += 1
                        graph_updated = True

    # --- Render Visualization ---
    # We update the plot every 10 steps so it catches the ground truth, or immediately on a graph update
    if graph_updated or (robot.getTime() * 1000) % 500 < TIME_STEP:
        draw_graph(pose_nodes, landmark_nodes, odom_edges, meas_edges, true_trajectory)