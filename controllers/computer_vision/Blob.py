from Basic_Pixel_Processing import pipeline1
import numpy as np
from PIL import Image

class Blob:
    def __init__(self):
        self.pixels = []
        self.center: tuple[int, int] | None = None
        self.color_histogram = np.zeros([8,8,8], dtype=int)  # 8 bins for R, G, B
    
    def add_pixel(self, x, y):
        self.pixels.append((x, y))

    def _compute_hog(self, img_array):
        """
        Computes the HOG descriptor for the blob.
        1. Extracts ROI.
        2. Resizes to standard 64x64.
        3. Pads image so edge pixels are centered in cells.
        4. Computes Gradients and Histograms.
        """
        if not self.pixels:
            return

        # 1. Determine Bounding Box
        xs = [p[0] for p in self.pixels]
        ys = [p[1] for p in self.pixels]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        # Extract ROI
        roi = img_array[min_y:max_y+1, min_x:max_x+1]

        # 2. HOG Calculation with Padding
        self.hog_descriptor = hog_compute(roi, resize_shape=(64, 64))

def hog_compute(img_roi: np.ndarray, resize_shape=(64,64), cell_size=(8,8), block_size=(2,2), nbins=4):
    #resizing image
    img = Image.fromarray(img_roi.astype('uint8'))
    img = img.resize(resize_shape, Image.Resampling.BILINEAR)
    img = np.array(img)

    # Convert to grayscale
    if len(img.shape) == 3:
        # Standard RGB to Grayscale weights
        gray = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])
    else:
        gray = img

    # Padding
    py = cell_size[1] // 2
    px = cell_size[0] // 2
    
    # Use 'reflect' to avoid artificial edge creation at borders
    gray_padded = np.pad(gray, ((py, py), (px, px)), mode='reflect')

    # Compute Gradient
    gx = np.gradient(gray_padded, axis=1)
    gy = np.gradient(gray_padded, axis=0)

    magnitude = np.sqrt(gx**2+gy**2)
    orientation = np.arctan2(gy,gx) * (180/np.pi)
    orientation = orientation % 180

    # Divide image into Cells
    h, w = gray_padded.shape
    cx, cy = cell_size
    n_cells_x = w // cx
    n_cells_y = h // cy

    # Grid of histograms
    histograms = np.zeros((n_cells_y, n_cells_x, nbins))

    bin_width = 180 / nbins

    for y in range(n_cells_y):
        for x in range(n_cells_x):
            cell_mag= magnitude[y*cy:(y+1)*cy, x*cx:(x+1)*cx]
            cell_ang = orientation[y*cy:(y+1)*cy, x*cx:(x+1)*cx]

            # Vectorized Bin Voting
            bin_indices = cell_ang / bin_width
            
            bin_1 = np.floor(bin_indices).astype(int) % nbins
            bin_2 = (bin_1 + 1) % nbins
            
            weight_2 = bin_indices - np.floor(bin_indices)
            weight_1 = 1.0 - weight_2
            
            flat_bin_1 = bin_1.ravel()
            flat_bin_2 = bin_2.ravel()
            flat_mag = cell_mag.ravel()
            
            hist_cell = np.zeros(nbins)
            np.add.at(hist_cell, flat_bin_1, flat_mag * weight_1.ravel())
            np.add.at(hist_cell, flat_bin_2, flat_mag * weight_2.ravel())
            
            histograms[y, x, :] = hist_cell

    # Normalize over Blocks
    bx, by = block_size
    n_blocks_x = n_cells_x - bx + 1
    n_blocks_y = n_cells_y - by + 1
    
    normalized_blocks = []
    eps = 1e-5

    for y in range(n_blocks_y):
        for x in range(n_blocks_x):
            block = histograms[y:y+by, x:x+bx, :]
            block_vector = block.flatten()
            k = np.sqrt(np.sum(block_vector**2) + eps**2)
            normalized_blocks.append(block_vector / k)

    if normalized_blocks:
        return np.concatenate(normalized_blocks)
    return np.array([])
    
def blobize(img_array: np.ndarray, edge_array: np.ndarray | None = None) -> list:
    height, width = img_array.shape[:2]
    
    if edge_array is None:
        edge_array = pipeline1(img_array)

    # Track which pixels have been assigned to blobs
    assigned = np.zeros((height, width), dtype=bool)
    blobs = []
    
    def is_edge(x, y):
        """Check if pixel is an edge (white pixel in binary image)"""
        if len(edge_array.shape) == 2:
            return edge_array[y, x] == 255
    
    def assign_to_blob(start_x, start_y):
        """Assign a pixel to a new blob using iterative flood fill"""
        # If already assigned to a blob, return
        if assigned[start_y, start_x]:
            return
        
        # If is an edge, return
        if is_edge(start_x, start_y):
            return
        
        # Create new blob
        blob = Blob()
        
        # Use stack for iterative flood fill instead of recursion
        stack = [(start_x, start_y)]
        
        while stack:
            x, y = stack.pop()
            
            # Check bounds
            if x < 0 or x >= width or y < 0 or y >= height:
                continue
            
            # If already assigned, skip
            if assigned[y, x]:
                continue
            
            # If is an edge, skip
            if is_edge(x, y):
                continue
            
            # Mark as assigned and add to blob
            assigned[y, x] = True
            blob.add_pixel(x, y)
            chanel_bins = [0, 0, 0]  
            for c in range(3):  # Assuming RGB
                pixel_value = img_array[y, x, c]
                chanel_bins[c] = pixel_value // 32 # 8 bins (0-7)
            blob.color_histogram[chanel_bins[0], chanel_bins[1], chanel_bins[2]] += 1
            
            # Add neighboring pixels to stack
            stack.append((x - 1, y))  # left
            stack.append((x + 1, y))  # right
            stack.append((x, y - 1))  # up
            stack.append((x, y + 1))  # down
        
        # Only add blob if it has pixels
        if len(blob.pixels) > height * width * 0.001:  # Minimum size threshold (0.1% of image)
            blob.center = (
                sum(p[0] for p in blob.pixels) // len(blob.pixels),
                sum(p[1] for p in blob.pixels) // len(blob.pixels)
            )
            blob.color_histogram = blob.color_histogram
            blob._compute_hog(img_array)
            blobs.append(blob)
    
    # For each pixel in frame
    for y in range(height):
        for x in range(width):
            assign_to_blob(x, y)
    
    return blobs

def histogram_distance(hist1, hist2):
    """Calculate the Bhattacharyya distance between two histograms."""
    
    # Check 1: Ensure inputs are numpy arrays
    h1 = np.array(hist1, dtype=float)
    h2 = np.array(hist2, dtype=float)
    
    # Normalize histograms to sum to 1 (convert to probability distributions)
    sum1 = np.sum(h1)
    sum2 = np.sum(h2)
    
    # Avoid division by zero
    if sum1 > 0:
        h1 = h1 / sum1
    if sum2 > 0:
        h2 = h2 / sum2
    
    # Check 2: Calculate the Bhattacharyya Coefficient (sum of sqrt of product)
    # This replaces the triple for-loop
    bc = np.sum(np.sqrt(h1 * h2))
    
    # Clamp bc to [0, 1] to avoid numerical issues
    bc = np.clip(bc, 0, 1)
    
    # Check 3: Add a tiny epsilon to avoid log(0) if histograms don't overlap
    return -np.log(bc + 1e-10)

def main():
    pass

if __name__ == "__main__":
    main()