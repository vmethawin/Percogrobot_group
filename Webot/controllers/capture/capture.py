from controller import Robot

# Initialize the Robot instance
robot = Robot()

# Get the time step of the current world
timestep = int(robot.getBasicTimeStep())

# Initialize the camera device
# "camera" should match the 'name' field of the Camera node in your .wbt file
cam = robot.getDevice("camera")
cam.enable(timestep)

# Main simulation loop
while robot.step(timestep) != -1:
    # Capture and save the image
    # The second argument is the quality (1-100)
    status = cam.saveImage("captured_frame.jpg", 80)
    
    if status == 0:
        print("Image saved successfully!")
    else:
        print("Failed to save image.")
        
    # Break after one capture if you only want a single photo
    break