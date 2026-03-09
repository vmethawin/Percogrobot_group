from controller import Supervisor, Keyboard
import math
import numpy as np

robot = Supervisor()

timestep = int(robot.getBasicTimeStep())

kb = Keyboard()
kb.enable(timestep)

# Get the robot's own node and its rotation field
robot_node = robot.getSelf()
position_field = robot_node.getField("translation")
rotation_field = robot_node.getField("rotation")

# Initialize motors
left_motor = robot.getDevice('left wheel motor')
right_motor = robot.getDevice('right wheel motor')
left_motor.setPosition(float('inf'))
right_motor.setPosition(float('inf'))

# Setup Display
display = robot.getDevice("display")

# Setup Map
map = np.zeros((250,250,3),dtype=np.uint8)

# Mapping Parameters
scale = 100
offset = 250 // 2

while robot.step(timestep) != -1:
    # Get Position
    pos = position_field.getSFVec3f()
    world_x, world_y = pos[0], pos[1]
    
    # Get Rotation
    rot = rotation_field.getSFRotation()
    
    # Calculate Heading (Yaw)
    angle_rad = rot[3]
    if rot[1] < 0:  # Adjusting for inverted axis direction
        angle_rad = -angle_rad
        
    heading_deg = math.degrees(angle_rad) % 360

    print(f"POS: X={world_x:.2f} Y={world_y:.2f} | HEADING: {heading_deg:.1f}°")
    
    # Mapping   
    # Convert to Array Indices
    px = int(world_x * scale) + offset
    py = int(-world_y * scale) + offset
    
    if 2 <= px < 250-2 and 2 <= py < 250-2:
        map[py-2:py+3, px-2:px+3] = [255, 0, 0]
        
    # Display
    map_data = map.astype(np.uint8).tobytes()
    ir = display.imageNew(map_data, display.RGB, 250, 250)
    display.imagePaste(ir, 0, 0, False)
    display.imageDelete(ir) # Free memory from the temporary image


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