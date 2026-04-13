from Basic_Pixel_Processing import pipeline1
import numpy as np
from PIL import Image

class Blob:
    def __init__(self):
        self.pixels = []
        self.flow_vectors = []
        self.avg_u = 0.0
        self.avg_v = 0.0
    
    def add_pixel(self, x, y):
        self.pixels.append((x, y))

    def add_flow(self, u, v):
        self.flow_vectors.append((float(u), float(v)))

    def update_flow_from_field(self, flow_field: np.ndarray):
        self.flow_vectors = []
        self.avg_u = 0.0
        self.avg_v = 0.0

        if flow_field is None or flow_field.ndim != 3 or flow_field.shape[2] < 2:
            return

        height, width = flow_field.shape[:2]

        for x, y in self.pixels:
            if 0 <= x < width and 0 <= y < height:
                u, v = flow_field[y, x]
                if u != 0 or v != 0:
                    self.add_flow(u, v)

        if self.flow_vectors:
            flow_array = np.array(self.flow_vectors)
            self.avg_u = float(np.mean(flow_array[:, 0]))
            self.avg_v = float(np.mean(flow_array[:, 1]))

    
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
            
            # Add neighboring pixels to stack
            stack.append((x - 1, y))  # left
            stack.append((x + 1, y))  # right
            stack.append((x, y - 1))  # up
            stack.append((x, y + 1))  # down
        
        # Only add blob if it has pixels
        if len(blob.pixels) > height * width * 0.001:  # Minimum size threshold (0.1% of image)
            blobs.append(blob)
    
    # For each pixel in frame
    for y in range(height):
        for x in range(width):
            assign_to_blob(x, y)
    
    return blobs

def main():
    pass

if __name__ == "__main__":
    main()