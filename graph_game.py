from controller import Robot
import numpy as np
import math
import matplotlib.pyplot as plt
import collections
from matplotlib.colors import ListedColormap
import Basic_  
import explore

print("[INIT] Starting WASD Manual Exploration...")

robot = Robot()
timestep = int(robot.getBasicTimeStep())

# --- Device Initialization ---
left_motor = robot.getDevice('left wheel motor')
right_motor = robot.getDevice('right wheel motor')
left_motor.setPosition(float('inf'))
right_motor.setPosition(float('inf'))

left_ps = robot.getDevice('left wheel sensor')
left_ps.enable(timestep)
right_ps = robot.getDevice('right wheel sensor')
right_ps.enable(timestep)

lidar = robot.getDevice("SickLms291") 
if lidar is None:
    lidar = robot.getDevice("LDS-01")
lidar.enable(timestep)
lidar.enablePointCloud()

inertial_unit = robot.getDevice("inertial unit")
inertial_unit.enable(timestep)

# --- Predetermined Grid Initialization ---
world_x, world_y, angle_rad = 0.0, 0.0, 0.0 
left_ps_last, right_ps_last = 0.0, 0.0
first_step = True

# Logical grid representing our predetermined 0.5m vertices
# All start as UNEXPLORED (0)
grid_graph = np.zeros((Basic_.MAP_CELLS, Basic_.MAP_CELLS), dtype=np.uint8)

# ==========================================
# --- LIVE MATPLOTLIB SETUP ---
# ==========================================
print("[INIT] Building Predetermined Grid Visualization...")
plt.ion() 
fig, ax = plt.subplots(figsize=(8, 8))

# Define colors for our Vertices: 0=Gray (Unexplored), 1=Cyan (Explored), 2=Black (Blocked)
cmap = ListedColormap(['gray', 'cyan', 'black'])

# Visualize the vertices as a grid of points
img_plot = ax.imshow(grid_graph, cmap=cmap, origin='upper', vmin=0, vmax=2)

# Create a grid appearance for the "Edges"
ax.set_xticks(np.arange(-0.5, Basic_.MAP_CELLS, 1), minor=True)
ax.set_yticks(np.arange(-0.5, Basic_.MAP_CELLS, 1), minor=True)
ax.grid(which="minor", color="white", linestyle='-', linewidth=0.5, alpha=0.3)
ax.tick_params(which="minor", size=0)

robot_plot, = ax.plot([], [], 'go', markersize=10, label='Robot Location')
bfs_path_plot, = ax.plot([], [], 'y-', linewidth=2.5, label='BFS Suggested Path')

plt.title("WASD Control + Predetermined Grid (0.5m Edges)")
plt.legend(loc="upper right")
plt.tight_layout()
plt.show(block=False) 

loop_counter = 0

def bfs_suggest_path(grid, start_x, start_y):
    """Runs BFS through EXPLORED edges to find the nearest UNEXPLORED vertex."""
    if not (0 <= start_x < Basic_.MAP_CELLS and 0 <= start_y < Basic_.MAP_CELLS):
        return []
        
    queue = collections.deque([(start_x, start_y, [])])
    visited = set([(start_x, start_y)])
    
    while queue:
        cx, cy, path = queue.popleft()
        
        # Traverse along edges (up, down, left, right)
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < Basic_.MAP_CELLS and 0 <= ny < Basic_.MAP_CELLS:
                if grid[ny, nx] == Basic_.UNEXPLORED:
                    return path + [(nx, ny)]  # Found frontier!
                elif grid[ny, nx] == Basic_.EXPLORED and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    queue.append((nx, ny, path + [(nx, ny)]))
    return []

# --- Main Loop ---
while robot.step(timestep) != -1:
    if first_step:
        left_ps_last, right_ps_last = left_ps.getValue(), right_ps.getValue()
        first_step = False
        continue

    # 1. Sensors & Odometry
    left_ps_current, right_ps_current = left_ps.getValue(), right_ps.getValue()
    rpy = inertial_unit.getRollPitchYaw()
    range_image = lidar.getRangeImage()

    dist, angle_rad = Basic_.update_odometry(left_ps_current, right_ps_current, left_ps_last, right_ps_last, rpy, angle_rad)
    left_ps_last, right_ps_last = left_ps_current, right_ps_current

    # update global coordinate
    world_x += dist * math.cos(angle_rad)
    world_y += dist * math.sin(angle_rad)
    
    # Current grid position (convert world to grid)
    rx, ry = Basic_.get_grid_coords(world_x, world_y)

    # 2. Update Grid Vertices (Mark Lidar pass-throughs as EXPLORED)
    grid_graph = Basic_.process_lidar_grid(
        grid_graph, range_image, lidar.getMaxRange(), world_x, world_y, angle_rad
    )
    
    # 3. Path Planning & Control
    suggested_path = bfs_suggest_path(grid_graph, rx, ry)

    l_speed, r_speed = 5.0, 5.0 # Default to stop
    
    if suggested_path:
        look_ahead = min(len(suggested_path) - 1, 3)
        target_node = suggested_path[0]
        
        l_speed, r_speed = explore.move_to_goal(target_node[0], target_node[1], world_x, world_y, angle_rad)

    left_motor.setVelocity(l_speed)
    right_motor.setVelocity(r_speed)

    # 4. Live Graph Update
    loop_counter += 1
    if loop_counter % 15 == 0:
        # Update colors on the grid
        img_plot.set_data(grid_graph)
        
        # Draw Robot
        robot_plot.set_data([rx], [ry])
        
        # Run BFS to draw the path to the next logical goal
        if suggested_path:
            path_x = [rx] + [p[0] for p in suggested_path]
            path_y = [ry] + [p[1] for p in suggested_path]
            bfs_path_plot.set_data(path_x, path_y)
        else:
            bfs_path_plot.set_data([], [])
            
        fig.canvas.draw()
        fig.canvas.flush_events()