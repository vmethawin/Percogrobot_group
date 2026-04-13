import numpy as np

def solve_lucas_kanade(Ix, Iy, It, x, y, window_size=3):
    """
    Solves for velocity (u, v) at pixel (x, y) by breaking down the matrix operations.

    Parameters:
    Ix (ndarray): Gradient of the first image in the x direction.
    Iy (ndarray): Gradient of the first image in the y direction.
    It (ndarray): Temporal gradient between the first and second images.
    x (int): x-coordinate of the pixel.
    y (int): y-coordinate of the pixel.
    window_size (int): Size of the window to consider around the pixel.

    Returns:
    u, v (ndarray): Optical flow components in the x and y directions at the specified pixel.
    """
    # Calculate the gradients within the window
    window_radius = window_size // 2
    Ix_window = Ix[y - window_radius:y + window_radius + 1, x - window_radius:x + window_radius + 1].flatten()
    Iy_window = Iy[y - window_radius:y + window_radius + 1, x - window_radius:x + window_radius + 1].flatten()
    It_window = It[y - window_radius:y + window_radius + 1, x - window_radius:x + window_radius + 1].flatten()

    if len(Ix_window) == 0: 
        return 0, 0
    
    # Construct the A matrix
    A = np.vstack((Ix_window, Iy_window)).T

    # Construct the b vector
    b = -It_window.reshape(-1, 1)

    # Compute A^T * A
    AtA = A.T @ A

    # Compute A^T * b
    Atb = A.T @ b
    
    # Compute the eigenvalues of AtA (structure tensor)
    eigenvalues = np.linalg.eigvals(AtA)

    l1 = eigenvalues.max()
    l2 = eigenvalues.min()
    
    if l2 < 1e-2:
        return 0, 0
    
    if l1 / l2 > 20:
        return 0, 0
    
    # Compute the inverse of A^T * A
    AtA_inv = np.linalg.pinv(AtA)

    # Compute the velocity vector [u, v]
    velocity = AtA_inv @ Atb
    u, v = velocity.flatten()
    return u, v


def get_derivatives(img1, img2):
    # Calculate gradients
    Iy, Ix = np.gradient(img1)
    It = img2 - img1
    return Ix, Iy, It

def optical_flow(frame1, frame2, window_size=[3], max_flow=5):
    """
    Computes the optical flow between two frames using the Lucas-Kanade method.

    Parameters:
    frame1 (ndarray): First image frame (grayscale).
    frame2 (ndarray): Second image frame (grayscale).
    window_size (list): Set of sizes of the window to consider around each pixel.
    max_flow (float): Maximum allowed flow magnitude.
    Returns:
    flow (ndarray): Optical flow vectors for each pixel.
    """
    height, width = frame1.shape
    flows = []

    # Get derivatives
    Ix, Iy, It = get_derivatives(frame1, frame2)

    # Compute optical flow for each pixel
    for size in window_size:
        flow = np.zeros((height, width, 2))  # Initialize flow array
        for y in range(size // 2, height - size // 2):
            for x in range(size // 2, width - size // 2):
                u, v = solve_lucas_kanade(Ix, Iy, It, x, y, size)
                flow[y, x] = [u, v]
        # Apply flow magnitude constraint
        flow_magnitude = np.linalg.norm(flow, axis=2)
        flow[flow_magnitude > max_flow] = 0
        flows.append(flow.copy())

    # compute the average flow across different window sizes
    average_flow = np.mean(flows, axis=0)


    return average_flow