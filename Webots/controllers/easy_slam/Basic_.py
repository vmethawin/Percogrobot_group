import math
import numpy as np
import collections

# ==========================================
# --- ROBOT CONSTANTS & MATH ---
# ==========================================
WHEEL_RADIUS = 0.033
WHEEL_BASE = 0.16
Magnitude = 10
CELL_SIZE = 0.5 / Magnitude         # 0.05m per cell
MAP_CELLS = 20 * Magnitude            
OFFSET = MAP_CELLS // 2   

# Vertex States for the Navigational Grid
UNEXPLORED = 0
EXPLORED = 1
BLOCKED = 2

def update_odometry(left_current, right_current, left_last, right_last, rpy, current_angle):
    dl = (left_current - left_last) * WHEEL_RADIUS
    dr = (right_current - right_last) * WHEEL_RADIUS
    dist = (dl + dr) / 2.0
    
    if rpy is not None:
        new_angle = rpy[2] 
    else:
        d_theta = (dr - dl) / WHEEL_BASE
        new_angle = current_angle + d_theta

    new_angle = (new_angle + math.pi) % (2 * math.pi) - math.pi
    return dist, new_angle

def get_grid_coords(x, y):
    mx = int(round(x / CELL_SIZE)) + OFFSET
    my = OFFSET - int(round(y / CELL_SIZE))
    return mx, my

def bfs_suggest_path(grid, start_x, start_y):
    """Runs BFS to find the nearest UNEXPLORED frontier, avoiding BLOCKED walls."""
    if not (0 <= start_x < MAP_CELLS and 0 <= start_y < MAP_CELLS):
        return []
        
    if grid[start_y, start_x] == BLOCKED:
        return []
        
    queue = collections.deque([(start_x, start_y, [])])
    visited = set([(start_x, start_y)])
    
    while queue:
        cx, cy, path = queue.popleft()
        
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < MAP_CELLS and 0 <= ny < MAP_CELLS:
                if (nx, ny) in visited:
                    continue
                if grid[ny, nx] == BLOCKED:
                    continue 
                if grid[ny, nx] == UNEXPLORED:
                    return path + [(nx, ny)]  
                if grid[ny, nx] == EXPLORED:
                    visited.add((nx, ny))
                    queue.append((nx, ny, path + [(nx, ny)]))
    return []

# ==========================================
# --- FEATURE EXTRACTOR ---
# ==========================================

def extract_landmarks(range_image, max_range):
    """ Groups raw LiDAR hits into distinct objects. """
    if not range_image:
        return []

    num_points = len(range_image)
    clusters = []
    current_cluster = []

    DISTANCE_THRESHOLD = 0.2 

    for i in range(num_points):
        distance = range_image[i]
        
        if not (0.12 < distance < max_range) or math.isinf(distance):
            if current_cluster:
                clusters.append(current_cluster)
                current_cluster = []
            continue

        rel_beam_angle = - (i * 2 * math.pi / num_points) + math.pi
        
        lx = distance * math.cos(rel_beam_angle)
        ly = distance * math.sin(rel_beam_angle)
        point = (lx, ly)

        if not current_cluster:
            current_cluster.append(point)
        else:
            last_point = current_cluster[-1]
            dist_between_points = math.hypot(lx - last_point[0], ly - last_point[1])
            
            if dist_between_points < DISTANCE_THRESHOLD:
                current_cluster.append(point)
            else:
                clusters.append(current_cluster)
                current_cluster = [point]

    if current_cluster:
        clusters.append(current_cluster)

    landmarks = []
    MIN_POINTS_PER_CLUSTER = 3 

    for cluster in clusters:
        if len(cluster) >= MIN_POINTS_PER_CLUSTER:
            avg_x = sum(p[0] for p in cluster) / len(cluster)
            avg_y = sum(p[1] for p in cluster) / len(cluster)
            landmarks.append((avg_x, avg_y))

    return landmarks

# ==========================================
# --- GRAPHSLAM DATA STRUCTURE ---
# ==========================================

class SLAMGraph:
    def __init__(self):
        self.poses = {0: (0.0, 0.0, 0.0)}  
        self.landmarks = {}                
        self.odometry_edges = []           
        self.measurement_edges = []        
        self.current_time = 0
        self.landmark_counter = 0

    def add_odometry(self, world_x, world_y, theta):
        self.current_time += 1
        self.poses[self.current_time] = (world_x, world_y, theta)
        
        last_x, last_y, last_theta = self.poses[self.current_time - 1]
        dx = world_x - last_x
        dy = world_y - last_y
        dtheta = theta - last_theta
        
        self.odometry_edges.append((self.current_time - 1, self.current_time, dx, dy, dtheta))

    def associate_and_add_landmark(self, rel_x, rel_y, robot_world_x, robot_world_y, robot_theta):
        abs_x = robot_world_x + (rel_x * math.cos(robot_theta) - rel_y * math.sin(robot_theta))
        abs_y = robot_world_y + (rel_x * math.sin(robot_theta) + rel_y * math.cos(robot_theta))

        ASSOCIATION_THRESHOLD = 0.5  
        
        best_match_id = None
        min_dist = float('inf')

        for l_id, (lx, ly) in self.landmarks.items():
            dist = math.hypot(abs_x - lx, abs_y - ly)
            if dist < min_dist and dist < ASSOCIATION_THRESHOLD:
                min_dist = dist
                best_match_id = l_id

        if best_match_id is not None:
            self.measurement_edges.append((self.current_time, best_match_id, rel_x, rel_y))
            return best_match_id, "RECOGNIZED"
        else:
            new_id = self.landmark_counter
            self.landmarks[new_id] = (abs_x, abs_y)
            self.measurement_edges.append((self.current_time, new_id, rel_x, rel_y))
            self.landmark_counter += 1
            return new_id, "NEW"