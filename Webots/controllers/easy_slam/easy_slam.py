from controller import Robot, Keyboard
import math
import numpy as np

# TurtleBot3 Burger in Webots parameters:
wheel_radius = 0.033
wheel_base = 0.16

robot = Robot()

timestep = int(robot.getBasicTimeStep())

kb = Keyboard()
kb.enable(timestep)

# Initialize motors
left_motor = robot.getDevice('left wheel motor')
right_motor = robot.getDevice('right wheel motor')
left_motor.setPosition(float('inf'))
right_motor.setPosition(float('inf'))

# Initialize Position Sensors
left_ps = robot.getDevice('left wheel sensor')
left_ps.enable(timestep)
right_ps = robot.getDevice('right wheel sensor')
right_ps.enable(timestep)

# Initialize LiDAR
lidar = robot.getDevice("LDS-01")
lidar.enable(timestep)
lidar.enablePointCloud()

# Initialize IMU
inertial_unit = robot.getDevice("inertial unit")
inertial_unit.enable(timestep)

# Initialize Odometry State
world_x = 0.0
world_y = 0.0
# The IMU provides absolute orientation, so 'angle_rad' becomes the current absolute heading.
# If we want odometry relative to start, we'd enable that, but here we can just use the IMU directly.
angle_rad = 0.0 
left_ps_last = 0.0
right_ps_last = 0.0
first_step = True

# Setup Display
display = robot.getDevice("display")

# Setup Map
map_grid = np.zeros((250,250,3),dtype=np.uint8)

# Mapping Parameters
scale = 100
offset = 250 // 2

while robot.step(timestep) != -1:
    if first_step:
        left_ps_last = left_ps.getValue()
        right_ps_last = right_ps.getValue()
        first_step = False
        continue

    # --- Odometry (Forward Kinematics) ---
    left_ps_current = left_ps.getValue()
    right_ps_current = right_ps.getValue()

    dl = (left_ps_current - left_ps_last) * wheel_radius
    dr = (right_ps_current - right_ps_last) * wheel_radius

    left_ps_last = left_ps_current
    right_ps_last = right_ps_current

    dist = (dl + dr) / 2.0
    
    # Use IMU for orientation to fix drift
    rpy = inertial_unit.getRollPitchYaw()
    # If the IMU is valid, use it. Otherwise, fallback to encoder integration (d_theta)
    if rpy is not None:
        angle_rad = rpy[2] # Yaw
    else:
         d_theta = (dr - dl) / wheel_base
         angle_rad += d_theta

    # Update Position using the corrected heading
    world_x += dist * math.cos(angle_rad)
    world_y += dist * math.sin(angle_rad)
    
    # Normalize Angle
    angle_rad = (angle_rad + math.pi) % (2 * math.pi) - math.pi
    
    print(f"POS: X={world_x:.2f} Y={world_y:.2f} | HEADING: {math.degrees(angle_rad):.1f}°")
    
    # --- Mapping ---
    px = int(world_x * scale) + offset
    py = offset - int(world_y * scale)
    
    # Mark Robot
    if 1 <= px < 250-1 and 1 <= py < 250-1:
        map_grid[py-1:py+1, px-1:px+1] = [255, 0, 0]
        
    # Process LiDAR
    range_image = lidar.getRangeImage()
    if range_image:
        num_points = len(range_image)
        for i, distance in enumerate(range_image):
            if 0.1 < distance < lidar.getMaxRange():
                # Adjusted angle calculation based on robot heading
                # beam_angle matches chat log logic
                beam_angle = angle_rad - (i * 2 * math.pi / num_points) + math.pi
                
                wx = world_x + distance * math.cos(beam_angle)
                wy = world_y + distance * math.sin(beam_angle)
                
                mx = int(wx * scale) + offset
                my = offset - int(wy * scale)
                
                if 0 <= mx < 250 and 0 <= my < 250:
                    map_grid[my, mx] = [255, 255, 255] # White walls
                    
    # Display
    map_data = map_grid.astype(np.uint8).tobytes()
    ir = display.imageNew(map_data, display.RGB, 250, 250)
    display.imagePaste(ir, 0, 0, False)
    display.imageDelete(ir)

    # Movement
    key = kb.getKey()
    if key == Keyboard.UP:
        left_motor.setVelocity(5.0)
        right_motor.setVelocity(5.0)
    elif key == Keyboard.DOWN:
        left_motor.setVelocity(-5.0)
        right_motor.setVelocity(-5.0)
    elif key == Keyboard.LEFT:
        left_motor.setVelocity(-5.0)
        right_motor.setVelocity(5.0)
    elif key == Keyboard.RIGHT:
        left_motor.setVelocity(5.0)
        right_motor.setVelocity(-5.0)
    else:
        left_motor.setVelocity(0)
        right_motor.setVelocity(0)