"""
Autonomous Visualized GraphSLAM Controller for Webots (TurtleBot3 LDS-01)
Features:
- WASD manual driving
- IMU + Encoder Odometry with continuous frame accumulation
- Calibration multipliers for real-world tuning
- Automated Landmark Extraction (Spike logic) & Data Association
- Iterative Graph Optimization (Pose X, Y, Theta + Landmark X, Y)
- Real-time Matplotlib Visualization of Nodes, Edges, and Landmarks
"""

from controller import Robot, Keyboard
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
plt.ion() # Enable interactive mode for real-time updating
fig, ax = plt.subplots(figsize=(8, 8))
fig.canvas.manager.set_window_title('GraphSLAM Live Map')

def draw_graph(poses, landmarks, odom_edges, meas_edges):
    """Draws the current state of the SLAM graph."""
    ax.clear()
    ax.set_title("Live GraphSLAM Visualization")
    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")
    ax.grid(True, linestyle='--', alpha=0.6)

    # Draw Odometry Edges (Robot Path)
    for edge in odom_edges:
        p1 = poses[edge.from_id]
        p2 = poses[edge.to_id]
        ax.plot([p1.x, p2.x], [p1.y, p2.y], color='blue', linestyle='-', alpha=0.4)

    # Draw Measurement Edges (Landmark Sightings)
    for edge in meas_edges:
        p = poses[edge.pose_id]
        lm = landmarks[edge.landmark_id]
        ax.plot([p.x, lm.x], [p.y, lm.y], color='green', linestyle=':', alpha=0.4)

    # Draw Pose Nodes
    if poses:
        px = [p.x for p in poses]
        py = [p.y for p in poses]
        ax.scatter(px, py, color='blue', s=10, label="Trajectory Nodes")
        # Highlight current position
        ax.scatter(poses[-1].x, poses[-1].y, color='black', s=50, marker='o', label="Current Robot Pose")

    # Draw Landmark Nodes
    if landmarks:
        lx = [lm.x for lm in landmarks]
        ly = [lm.y for lm in landmarks]
        ax.scatter(lx, ly, color='red', s=100, marker='*', label="Extracted Landmarks")

    ax.legend(loc='upper right')
    ax.axis('equal') # Forces 1:1 aspect ratio
    fig.canvas.draw()
    fig.canvas.flush_events()

# --- 3. Robot Initialization ---
TIME_STEP = 64
WHEEL_RADIUS = 0.033
WHEEL_BASE = 0.16
MATCH_THRESHOLD = 0.5 

# Kinematic Calibration Factors (Tune these based on physical drift)
LINEAR_CORRECTION = 1.00  
ANGULAR_CORRECTION = 1.00 

robot = Robot()
keyboard = robot.getKeyboard()
keyboard.enable(TIME_STEP)

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
    lidar.enablePointCloud() # Required to grab coordinates without throwing Webots errors

# Graph State Initialization
pose_nodes = [PoseNode(0, 0.0, 0.0, 0.0)]
landmark_nodes = []
odom_edges = []
meas_edges = []

pose_id_counter = 1
landmark_id_counter = 0
first_step = True

# Trackers for Odometry Integration
left_ps_last, right_ps_last = 0.0, 0.0
accumulated_dist = 0.0
world_x, world_y = 0.0, 0.0

# --- 4. Graph Optimization ---
def optimize_graph(poses, landmarks, o_edges, m_edges, iterations=10):
    """
    Refines node positions using a numerical method (Iterative Relaxation).
    Adjusts X, Y, and Theta to minimize graph tension.
    """
    for _ in range(iterations):
        # --- Odometry Optimization ---
        for edge in o_edges:
            p_from, p_to = poses[edge.from_id], poses[edge.to_id]
            
            exp_x = p_from.x + (edge.dx * math.cos(p_from.theta))
            exp_y = p_from.y + (edge.dx * math.sin(p_from.theta))
            exp_theta = p_from.theta + edge.dtheta  # Expected heading
            
            err_x = exp_x - p_to.x
            err_y = exp_y - p_to.y
            # Normalize angular error to be between -PI and PI
            err_theta = math.atan2(math.sin(exp_theta - p_to.theta), math.cos(exp_theta - p_to.theta))
            
            # Pin the origin (Node 0) to avoid map drift
            if p_from.id != 0:
                p_from.x -= err_x * 0.05
                p_from.y -= err_y * 0.05
                p_from.theta -= err_theta * 0.05 # Adjust from-node angle
                
            p_to.x += err_x * 0.05
            p_to.y += err_y * 0.05
            p_to.theta += err_theta * 0.05       # Adjust to-node angle

        # --- Measurement Optimization ---
        for edge in m_edges:
            pose, lm = poses[edge.pose_id], landmarks[edge.landmark_id]
            
            exp_lm_x = pose.x + (edge.local_x * math.cos(pose.theta) - edge.local_y * math.sin(pose.theta))
            exp_lm_y = pose.y + (edge.local_x * math.sin(pose.theta) + edge.local_y * math.cos(pose.theta))
            
            err_x, err_y = exp_lm_x - lm.x, exp_lm_y - lm.y
            
            # Pin the origin (Node 0) to avoid map drift
            if pose.id != 0:
                pose.x -= err_x * 0.1
                pose.y -= err_y * 0.1
            lm.x += err_x * 0.1
            lm.y += err_y * 0.1

print("=== Autonomous Visualized SLAM Started ===")
print("Drive with WASD. The map will draw in the Matplotlib window.")
draw_graph(pose_nodes, landmark_nodes, odom_edges, meas_edges)

# --- 5. Main Loop ---
while robot.step(TIME_STEP) != -1:
    if first_step:
        left_ps_last, right_ps_last = left_ps.getValue(), right_ps.getValue()
        first_step = False
        continue

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
    
    # Calculate raw step and apply calibration multiplier
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

    # 1. Update Graph Path 
    if abs(accumulated_dist) > 0.05 or abs(dtheta_from_last_node) > 0.05:
        new_pose = PoseNode(pose_id_counter, world_x, world_y, angle_rad)
        pose_nodes.append(new_pose)
        odom_edges.append(OdometryEdge(pose_id_counter - 1, pose_id_counter, accumulated_dist, 0.0, dtheta_from_last_node))
        pose_id_counter += 1
        accumulated_dist = 0.0
        graph_updated = True

        # 2. DATA ASSOCIATION & LANDMARK EXTRACTION
        if abs(dtheta_from_last_node) < 0.08:
            if lidar:
                point_cloud = lidar.getPointCloud()
                closest_pt = None
                min_dist = float('inf')
                
                for point in point_cloud:
                    if math.isinf(point.x) or math.isinf(point.y): continue
                    
                    pt_dist = math.hypot(point.x, point.y)
                    if 0.12 < pt_dist < 1.5 and pt_dist < min_dist:
                        min_dist = pt_dist
                        closest_pt = point
                        
                if closest_pt:
                    current_p = pose_nodes[-1]
                    
                    gl_x = current_p.x + (closest_pt.x * math.cos(current_p.theta) - closest_pt.y * math.sin(current_p.theta))
                    gl_y = current_p.y + (closest_pt.x * math.sin(current_p.theta) + closest_pt.y * math.cos(current_p.theta))
                    
                    # Nearest-Neighbor Data Association
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
                        
                        # Sync physical tracker to optimized graph node
                        world_x, world_y = pose_nodes[-1].x, pose_nodes[-1].y
                        graph_updated = True
                    else:
                        new_lm = LandmarkNode(landmark_id_counter, gl_x, gl_y)
                        landmark_nodes.append(new_lm)
                        meas_edges.append(MeasurementEdge(current_p.id, landmark_id_counter, closest_pt.x, closest_pt.y))
                        print(f"📍 Mapped NEW Landmark #{landmark_id_counter} at Global X:{gl_x:.2f}, Y:{gl_y:.2f}")
                        landmark_id_counter += 1
                        graph_updated = True

    # --- 3. Render Visualization ---
    if graph_updated:
        draw_graph(pose_nodes, landmark_nodes, odom_edges, meas_edges)