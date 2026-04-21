from Basic_Pixel_Processing import pipeline1
import numpy as np

class Blob:
    def __init__(self):
        self.pixels = []
        self.pixels_np = None
        self.avg_u = 0.0
        self.avg_v = 0.0
    
    def add_pixel(self, x, y):
        self.pixels.append((x, y))

    def finalize_pixels(self):
        if self.pixels_np is None:
            if self.pixels:
                self.pixels_np = np.asarray(self.pixels, dtype=np.int32)
            else:
                self.pixels_np = np.empty((0, 2), dtype=np.int32)

    def update_flow_from_field(self, flow_field: np.ndarray):
        self.avg_u = 0.0
        self.avg_v = 0.0

        if flow_field is None or flow_field.ndim != 3 or flow_field.shape[2] < 2:
            return

        self.finalize_pixels()
        if self.pixels_np is None or self.pixels_np.size == 0:
            return

        height, width = flow_field.shape[:2]
        coords = self.pixels_np
        valid = (
            (coords[:, 0] >= 0)
            & (coords[:, 0] < width)
            & (coords[:, 1] >= 0)
            & (coords[:, 1] < height)
        )
        if not np.any(valid):
            return

        coords = coords[valid]
        sampled = flow_field[coords[:, 1], coords[:, 0], :2]
        non_zero = (sampled[:, 0] != 0) | (sampled[:, 1] != 0)
        if not np.any(non_zero):
            return

        sampled = sampled[non_zero]
        self.avg_u = float(np.mean(sampled[:, 0]))
        self.avg_v = float(np.mean(sampled[:, 1]))

    
def blobize(img_array: np.ndarray, edge_array: np.ndarray | None = None) -> list:
    height, width = img_array.shape[:2]
    
    if edge_array is None:
        edge_array = pipeline1(img_array)

    if edge_array.ndim == 2:
        non_edge = edge_array != 255
    else:
        non_edge = np.any(edge_array != 255, axis=2)

    assigned = np.zeros((height, width), dtype=np.bool_)
    min_blob_pixels = max(1, int(height * width * 0.001))
    blobs = []

    for y in range(height):
        for x in range(width):
            if assigned[y, x] or not non_edge[y, x]:
                continue

            stack = [(x, y)]
            pixels = []

            while stack:
                cx, cy = stack.pop()
                if cx < 0 or cx >= width or cy < 0 or cy >= height:
                    continue
                if assigned[cy, cx] or not non_edge[cy, cx]:
                    continue

                assigned[cy, cx] = True
                pixels.append((cx, cy))

                stack.append((cx - 1, cy))
                stack.append((cx + 1, cy))
                stack.append((cx, cy - 1))
                stack.append((cx, cy + 1))

            if len(pixels) > min_blob_pixels:
                blob = Blob()
                blob.pixels = pixels
                blob.finalize_pixels()
                blobs.append(blob)
    
    return blobs

def main():
    pass

if __name__ == "__main__":
    main()