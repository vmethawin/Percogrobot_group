import math
import Basic_
import numpy as np

def get_frontiers(grid): #Frontier is unexplored node adjacent to explored space.
    frontiers = []
    for y in range(Basic_.MAP_CELLS):
        for x in range(Basic_.MAP_CELLS):
            if grid[y, x] != Basic_.UNEXPLORED:
                continue
            for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                nx, ny = x+dx, y+dy
                if 0 <= nx < Basic_.MAP_CELLS and 0 <= ny < Basic_.MAP_CELLS:
                    if grid[ny, nx] == Basic_.EXPLORED:
                        frontiers.append((x, y))
                        break
    return frontiers

def frontier_utility(robot_x, robot_y, frontier):
    fx, fy = frontier

    manhattan = abs(fx - robot_x) + abs(fy - robot_y)
    euclidean = math.sqrt((fx - robot_x)**2 + (fy - robot_y)**2)

    utility = 0.6 * manhattan + 0.4 * euclidean
    return utility

def choose_best_frontier(grid, rx, ry):
    frontiers = get_frontiers(grid)

    if not frontiers:
        return None

    best = min(frontiers, key=lambda f: frontier_utility(rx, ry, f))
    return best

def grid_to_world(mx, my):
    wx = (mx - Basic_.OFFSET) * Basic_.CELL_SIZE
    wy = (Basic_.OFFSET - my) * Basic_.CELL_SIZE
    return wx, wy

def move_to_goal(target_mx, target_my, current_wx, current_wy, current_angle):
    # 1. Get world coords of the center of the target cell
    target_wx, target_wy = grid_to_world(target_mx, target_my)

    # 2. Angle to target
    dx = target_wx - current_wx
    dy = target_wy - current_wy
    target_angle = math.atan2(dy, dx)

    # 3. Compute error
    angle_error = target_angle - current_angle
    while angle_error > math.pi: angle_error -= 2.0 * math.pi
    while angle_error < -math.pi: angle_error += 2.0 * math.pi

    # 4. Controller Params
    MAX_SPEED = 3

    if abs(angle_error) > 1:  # Pivot in place if heading is off
        print(f"angle_error = {angle_error}")
        left_speed = MAX_SPEED*0.5 
        right_speed = -left_speed
    else:  # Drive forward with slight steering corrections
        left_speed = MAX_SPEED * 0.7 - (angle_error * 2.0)
        right_speed = MAX_SPEED * 0.7 + (angle_error * 2.0)

    return np.clip(left_speed, -MAX_SPEED, MAX_SPEED), np.clip(right_speed, -MAX_SPEED, MAX_SPEED)
