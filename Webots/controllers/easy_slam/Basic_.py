import math
import numpy as np

# --- 0.5m Fixed Grid Constants ---
WHEEL_RADIUS = 0.033
WHEEL_BASE = 0.16
Magnitude = 10
CELL_SIZE = 0.5 * Magnitude         # Distance between vertices (0.1m)
MAP_CELLS = 20 * Magnitude            # 40 cells * 0.1m = 20m x 20m map
OFFSET = MAP_CELLS // 2   # Center at index (20, 20)

# Vertex States
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
    """Convert world coordinates to our 0.5m grid indices."""
    mx = int(round(x / CELL_SIZE)) + OFFSET
    my = OFFSET - int(round(y / CELL_SIZE))
    return mx, my

def bresenham(x0, y0, x1, y1):
    """Yield integer coordinates on the line from (x0, y0) to (x1, y1)."""
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    x, y = x0, y0
    sx = -1 if x0 > x1 else 1
    sy = -1 if y0 > y1 else 1
    if dx > dy:
        err = dx / 2.0
        while x != x1:
            yield x, y
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
    else:
        err = dy / 2.0
        while y != y1:
            yield x, y
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy
    yield x, y

def process_lidar_grid(grid, range_image, max_range, world_x, world_y, angle_rad):
    """Maps LiDAR data to the 0.5m vertices."""
    if range_image:
        num_points = len(range_image)
        rx, ry = get_grid_coords(world_x, world_y)
        
        for i, distance in enumerate(range_image):
            if 0.12 < distance < max_range and not math.isinf(distance):
                beam_angle = angle_rad - (i * 2 * math.pi / num_points) + math.pi
                
                wx = world_x + distance * math.cos(beam_angle)
                wy = world_y + distance * math.sin(beam_angle)
                
                mx, my = get_grid_coords(wx, wy)
                
                if 0 <= mx < MAP_CELLS and 0 <= my < MAP_CELLS and 0 <= rx < MAP_CELLS and 0 <= ry < MAP_CELLS:
                    # Mark all intermediate vertices as EXPLORED
                    for px, py in bresenham(rx, ry, mx, my):
                        if 0 <= px < MAP_CELLS and 0 <= py < MAP_CELLS:
                            if grid[py, px] != BLOCKED: 
                                grid[py, px] = EXPLORED
                                
                    # --- OBSTACLE ASSIGNMENT (Currently disabled per your request) ---
                    # To enable assigning blocked nodes, uncomment the line below:
                    # grid[my, mx] = BLOCKED 
                    
    return grid