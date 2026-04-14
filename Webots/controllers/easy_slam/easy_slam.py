from collections import deque

class GridMap:
    def __init__(self, length, height, resolution):
        self.length = length
        self.height = height
        self.resolution = resolution
        self.cols = int(length / resolution)
        self.rows = int(height / resolution)
        # Initialize map with -1 (Unknown)
        self.grid = np.full((self.rows, self.cols), -1.0)

    def world_to_grid(self, x, y):
        col = int((x + self.length / 2) / self.resolution)
        row = int((y + self.height / 2) / self.resolution)
        row = self.rows - 1 - row  # Invert row so +Y is up
        return row, col

    def grid_to_world(self, row, col):
        # Useful for pathfinding back to world coordinates
        x = (col * self.resolution) - (self.length / 2)
        y = ((self.rows - 1 - row) * self.resolution) - (self.height / 2)
        return x, y

    def update_map(self, robot_x, robot_y, obs_x, obs_y):
        r_row, r_col = self.world_to_grid(robot_x, robot_y)
        o_row, o_col = self.world_to_grid(obs_x, obs_y)
        
        # Simple Bresenham-like raycast to mark free space (0)
        # For brevity, a simple interpolation is used here
        steps = max(abs(o_row - r_row), abs(o_col - r_col))
        for i in range(steps):
            t = i / float(steps)
            curr_row = int(r_row * (1 - t) + o_row * t)
            curr_col = int(r_col * (1 - t) + o_col * t)
            if 0 <= curr_row < self.rows and 0 <= curr_col < self.cols:
                # Only overwrite unknown space with free space, preserve existing obstacles
                if self.grid[curr_row, curr_col] == -1:
                    self.grid[curr_row, curr_col] = 0.0 
        
        # Mark obstacle (1)
        if 0 <= o_row < self.rows and 0 <= o_col < self.cols:
            self.grid[o_row, o_col] = 1.0


class BFSExplorer:
    def __init__(self, grid_map):
        self.map = grid_map

    def find_nearest_frontier(self, robot_x, robot_y):
        """
        Uses BFS to find the closest 'Free' cell (0) that is adjacent to an 'Unknown' cell (-1).
        """
        start_row, start_col = self.map.world_to_grid(robot_x, robot_y)
        
        # If robot is out of bounds, return None
        if not (0 <= start_row < self.map.rows and 0 <= start_col < self.map.cols):
            return None

        queue = deque([(start_row, start_col)])
        visited = set()
        visited.add((start_row, start_col))

        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)] # Up, Down, Left, Right

        while queue:
            r, c = queue.popleft()

            # Check if this cell is a frontier (is free space AND touches unknown space)
            if self.map.grid[r, c] == 0.0:
                for dr, dc in directions:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < self.map.rows and 0 <= nc < self.map.cols:
                        if self.map.grid[nr, nc] == -1.0:
                            # Frontier found! Return the world coordinates of this free cell
                            return self.map.grid_to_world(r, c)

            # Continue BFS expansion
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.map.rows and 0 <= nc < self.map.cols:
                    if (nr, nc) not in visited:
                        # Only traverse through Free space (0) or Unknown (-1), do not cross obstacles (1)
                        if self.map.grid[nr, nc] <= 0.0: 
                            visited.add((nr, nc))
                            queue.append((nr, nc))
        
        return None # Map is fully explored
    
class GraphSLAM:
    def __init__(self):
        # State vector components
        self.poses = []      # List of [x, y, theta] representing robot history
        self.landmarks = []  # List of [x, y] representing detected obstacles/objects
        
        # Edges
        self.odometry_edges = []   # Links between consecutive poses
        self.measurement_edges = [] # Links between a pose and a landmark

    def add_pose(self, pose):
        """Add a new robot pose to the graph."""
        self.poses.append(np.array(pose))
        return len(self.poses) - 1 # Return pose index

    def add_landmark(self, landmark):
        """Add a new landmark to the graph."""
        self.landmarks.append(np.array(landmark))
        return len(self.landmarks) - 1 # Return landmark index

    def add_odometry_edge(self, pose_idx_1, pose_idx_2, delta_pose, information_matrix_R):
        """
        pose_idx_1 -> pose_idx_2: The predicted movement (odometry).
        information_matrix_R: Confidence in odometry (higher = more confident).
        """
        self.odometry_edges.append({
            'from': pose_idx_1,
            'to': pose_idx_2,
            'delta': np.array(delta_pose),
            'info': information_matrix_R
        })

    def add_measurement_edge(self, pose_idx, landmark_idx, measurement, information_matrix_Q):
        """
        pose_idx -> landmark_idx: The sensor reading.
        information_matrix_Q: Confidence in sensor (higher = more confident).
        """
        self.measurement_edges.append({
            'pose_id': pose_idx,
            'land_id': landmark_idx,
            'measurement': np.array(measurement),
            'info': information_matrix_Q
        })

    def optimize_graph_2d(self):
        """
        A simplified least-squares solver demonstrating Omega * Delta_Mu = -Xi
        (Based on the Dr. Garcia lecture slides)
        NOTE: This simplifies to X/Y translation optimization. 
        Full SLAM requires linearizing the rotation matrix Jacobians.
        """
        num_poses = len(self.poses)
        num_landmarks = len(self.landmarks)
        total_nodes = num_poses + num_landmarks
        
        # Create Omega (Information Matrix) and Xi (Information Vector)
        # Dimensions are *2 because we are solving for x and y.
        Omega = np.zeros((total_nodes * 2, total_nodes * 2))
        Xi = np.zeros((total_nodes * 2, 1))

        # Anchor the first pose (x0) so the graph doesn't float away
        Omega[0:2, 0:2] = np.eye(2) * 1000.0 # High confidence anchor

        # 1. Add Odometry Constraints (R)
        for edge in self.odometry_edges:
            i = edge['from'] * 2
            j = edge['to'] * 2
            R_inv = edge['info'] # Assuming 2x2 matrix
            delta = edge['delta'][0:2] # just x, y
            
            # Error = (x_j - x_i) - delta
            # Add to Information Matrix
            Omega[i:i+2, i:i+2] += R_inv
            Omega[j:j+2, j:j+2] += R_inv
            Omega[i:i+2, j:j+2] -= R_inv
            Omega[j:j+2, i:i+2] -= R_inv
            
            # Add to Information Vector
            Xi[i:i+2, 0] -= R_inv @ delta
            Xi[j:j+2, 0] += R_inv @ delta

        # 2. Add Measurement Constraints (Q)
        for edge in self.measurement_edges:
            i = edge['pose_id'] * 2
            # Landmarks come after poses in the global matrix
            j = (num_poses + edge['land_id']) * 2 
            Q_inv = edge['info']
            meas = edge['measurement'][0:2] # distance x, y relative to robot
            
            Omega[i:i+2, i:i+2] += Q_inv
            Omega[j:j+2, j:j+2] += Q_inv
            Omega[i:i+2, j:j+2] -= Q_inv
            Omega[j:j+2, i:i+2] -= Q_inv
            
            Xi[i:i+2, 0] -= Q_inv @ meas
            Xi[j:j+2, 0] += Q_inv @ meas

        # 3. Solve the system: Omega * Delta_Mu = -Xi -> Mu = Omega^-1 * Xi
        # In a real scenario you solve for Delta and update iteratively, 
        # but for a linear least squares this solves directly.
        try:
            Mu_optimized = np.linalg.solve(Omega, Xi)
            
            # Update Poses
            for p in range(num_poses):
                self.poses[p][0] = Mu_optimized[p*2, 0]
                self.poses[p][1] = Mu_optimized[p*2+1, 0]
                
            # Update Landmarks
            for l in range(num_landmarks):
                self.landmarks[l][0] = Mu_optimized[(num_poses + l)*2, 0]
                self.landmarks[l][1] = Mu_optimized[(num_poses + l)*2+1, 0]
                
            print("Graph Optimized Successfully.")
        except np.linalg.LinAlgError:
            print("Singular Matrix: Graph not constrained enough yet.")

from controller import Supervisor
import numpy as np
import math
import random
import matplotlib.pyplot as plt
from collections import deque

# Helper functions
def get_coordinate(supervisor):
    # NOTE: In real SLAM, you replace this with wheel odometry math!
    # For now, we will use it as a simulated "odometry" estimate to feed the graph.
    robot_node = supervisor.getSelf()
    position = robot_node.getPosition()
    R = robot_node.getOrientation()
    return position[0], position[1], math.atan2(R[3], R[0])
    
def get_point_cloud(lidar_device):
    """
    Extract point cloud from LiDAR device.
    For 2D LiDAR (LDS-01), returns points in [x, y] format.
    """
    points = []
    point_cloud = lidar_device.getPointCloud()
    
    if point_cloud:
        for p in point_cloud:
            # 3. FLOOR FILTER: Comment this out for a 2D LiDAR!
            # if p.z < 0.1: 
            #     continue
            points.append([p.x, p.y])
    
    return points

if __name__ == "__main__":
    supervisor = Supervisor()
    timestep = int(supervisor.getBasicTimeStep())
    
    # Initialize Map, BFS, and SLAM
    my_map = GridMap(length=20.0, height=20.0, resolution=0.1)
    explorer = BFSExplorer(my_map)
    slam = GraphSLAM()
    
    # Initialize LiDAR and Motors
    lidar = supervisor.getDevice('LDS-01')
    lidar.enable(timestep)
    lidar.enablePointCloud()
    
    left_motor = supervisor.getDevice('left wheel motor')
    right_motor = supervisor.getDevice('right wheel motor')
    left_motor.setPosition(float('inf'))
    right_motor.setPosition(float('inf'))
    left_motor.setVelocity(0.0)
    right_motor.setVelocity(0.0)
    
    # Setup Matplotlib
    plt.ion()
    fig, ax = plt.subplots()
    ax.set_title("Live 2D Occupancy Grid")
    img_plot = ax.imshow(my_map.grid, cmap='viridis', origin='upper') # changed to viridis to see -1, 0, 1
    plt.show()
    
    loop_counter = 0
    
    # Keep track of previous pose for odometry edges
    prev_pose_idx = None
    
    while supervisor.step(timestep) != -1:
        loop_counter += 1
        
        # 1. Get "Odometry" (Estimated Position)
        x, y, theta = get_coordinate(supervisor)
        
        # Add pose to GraphSLAM
        current_pose_idx = slam.add_pose([x, y, theta])
        
        # Link to previous pose with odometry edge (if not the first pose)
        if prev_pose_idx is not None:
            # Fake Odometry confidence (R_inv). High value = high confidence.
            R_inv = np.eye(2) * 10.0 
            delta_x = x - slam.poses[prev_pose_idx][0]
            delta_y = y - slam.poses[prev_pose_idx][1]
            slam.add_odometry_edge(prev_pose_idx, current_pose_idx, [delta_x, delta_y], R_inv)
        
        prev_pose_idx = current_pose_idx

        # 2. Get LiDAR Data & Update Map
        points = get_point_cloud(lidar)
        for p in points:
            p_x, p_y = p[0], p[1]
            m = math.hypot(p_x, p_y)
            n = math.atan2(p_y, p_x)
            
            obs_x = x + m * math.cos(theta + n)
            obs_y = y + m * math.sin(theta + n)
            
            # Update the GridMap (Raycasting frees space, marks obstacle)
            my_map.update_map(x, y, obs_x, obs_y)
            
            # (Optional for later) Add Measurement Edges to GraphSLAM here
            # if object_is_recognized(obs_x, obs_y):
            #     land_idx = slam.add_landmark([obs_x, obs_y])
            #     slam.add_measurement_edge(...)

        # 3. Graph Optimization Trigger
        # Optimize every 100 steps so we don't freeze the simulation
        if loop_counter % 100 == 0:
            print("Optimizing Graph...")
            slam.optimize_graph_2d()

        # 4. BFS Frontier Exploration
        frontier = explorer.find_nearest_frontier(x, y)
        if frontier:
            target_x, target_y = frontier
            # TODO: Add your path planning logic here to drive towards (target_x, target_y)
            # For now, we will just spin randomly to gather data
            left_motor.setVelocity(2.0)
            right_motor.setVelocity(-2.0)
        else:
            print("Map fully explored or robot trapped!")
            left_motor.setVelocity(0)
            right_motor.setVelocity(0)

        # 5. Update Visualizer
        if loop_counter % 20 == 0:
            img_plot.set_data(my_map.grid)
            # Adjust color limits so -1 (Unknown), 0 (Free), 1 (Obstacle) show up clearly
            img_plot.set_clim(-1, 1) 
            plt.pause(0.001)