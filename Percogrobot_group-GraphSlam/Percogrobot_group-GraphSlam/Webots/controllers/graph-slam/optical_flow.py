import numpy as np

def _box_sum(image: np.ndarray, size: int) -> np.ndarray:
    """Compute sum over a square window for every pixel using integral images."""
    radius = size // 2
    padded = np.pad(image, ((radius, radius), (radius, radius)), mode='edge')
    integral = np.pad(padded, ((1, 0), (1, 0)), mode='constant').cumsum(axis=0).cumsum(axis=1)
    return (
        integral[size:, size:]
        - integral[:-size, size:]
        - integral[size:, :-size]
        + integral[:-size, :-size]
    )

def get_derivatives(img1, img2):
    img1 = img1.astype(np.float32, copy=False)
    img2 = img2.astype(np.float32, copy=False)
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
    if isinstance(window_size, int):
        window_sizes = [window_size]
    else:
        window_sizes = [int(size) for size in window_size if int(size) > 1 and int(size) % 2 == 1]

    if not window_sizes:
        window_sizes = [3]

    height, width = frame1.shape

    Ix, Iy, It = get_derivatives(frame1, frame2)
    Ixx = Ix * Ix
    Ixy = Ix * Iy
    Iyy = Iy * Iy
    Ixt = Ix * It
    Iyt = Iy * It

    flow_sum = np.zeros((height, width, 2), dtype=np.float32)
    valid_size_count = 0
    eps = 1e-6

    for size in window_sizes:
        Sxx = _box_sum(Ixx, size)
        Sxy = _box_sum(Ixy, size)
        Syy = _box_sum(Iyy, size)
        Sxt = _box_sum(Ixt, size)
        Syt = _box_sum(Iyt, size)

        b0 = -Sxt
        b1 = -Syt

        det = (Sxx * Syy) - (Sxy * Sxy)
        trace = Sxx + Syy
        discriminant = np.maximum((trace * trace) - (4.0 * det), 0.0)
        sqrt_disc = np.sqrt(discriminant)

        l1 = 0.5 * (trace + sqrt_disc)
        l2 = 0.5 * (trace - sqrt_disc)
        condition = l1 / (l2 + eps)

        valid = (det > eps) & (l2 >= 1e-2) & (condition <= 20.0)

        radius = size // 2
        if radius > 0:
            valid[:radius, :] = False
            valid[-radius:, :] = False
            valid[:, :radius] = False
            valid[:, -radius:] = False

        u = np.zeros((height, width), dtype=np.float32)
        v = np.zeros((height, width), dtype=np.float32)

        u[valid] = (Syy[valid] * b0[valid] - Sxy[valid] * b1[valid]) / det[valid]
        v[valid] = (-Sxy[valid] * b0[valid] + Sxx[valid] * b1[valid]) / det[valid]

        flow_magnitude = np.sqrt((u * u) + (v * v))
        keep = flow_magnitude <= max_flow
        u[~keep] = 0.0
        v[~keep] = 0.0

        flow_sum[:, :, 0] += u
        flow_sum[:, :, 1] += v
        valid_size_count += 1

    if valid_size_count == 0:
        return np.zeros((height, width, 2), dtype=np.float32)

    return flow_sum / float(valid_size_count)