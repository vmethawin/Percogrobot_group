import numpy as np
from PIL import Image
import os

def convolution(image, kernel):
    k_height, k_width = kernel.shape
    i_height, i_width = image.shape
    pad_height = k_height // 2
    pad_width = k_width // 2
    padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='edge')
    # Use float64 to avoid overflow/underflow during calculations
    convolved_image = np.zeros((i_height, i_width), dtype=np.float64)
    for i in range(i_height):
        for j in range(i_width):
            region = padded_image[i:i + k_height, j:j + k_width]
            convolved_image[i, j] = np.sum(region * kernel)
    return convolved_image

def gaussian_blur(image):
    gaussian_kernel = np.array([
        [1, 4, 6, 4, 1],
        [4, 16, 24, 16, 4],
        [6, 24, 36, 24, 6],
        [4, 16, 24, 16, 4],
        [1, 4, 6, 4, 1]
    ]) / 256
    return convolution(image, gaussian_kernel)

def gray_scale(image, method='luminosity'):
    if len(image.shape) == 3 and image.shape[2] == 3:
        if method == 'saturation':
            # 1. Convert to float to avoid issues with subtraction (e.g. 10 - 20 becoming 246)
            img_float = image.astype(np.float32)

            # 2. Calculate the range of color in each pixel.
            #    - White/Gray pixels have R≈G≈B, so max - min ≈ 0 (Black)
            #    - Yellow pixels have High R/G and Low B, so max - min ≈ High (White)
            max_val = np.max(img_float, axis=2)
            min_val = np.min(img_float, axis=2)
            saturation = max_val - min_val
            return saturation.astype(np.uint8)
        elif method == 'luminosity':
            return np.dot(image[..., :3], [0.299, 0.587, 0.114])
    else:
        return image

def edge_detection(image):
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]])
    grad_x = convolution(image, sobel_x)
    grad_y = convolution(image, sobel_y)
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    return magnitude

def hysteresis(img, weak, strong=150):
    M, N = img.shape
    
    final_output = np.zeros((M, N), dtype=np.uint8)
    
    # 1. FIND STRONG EDGES
    strong_mask = img >= strong
    final_output[strong_mask] = 255
    
    # 2. START TRACKING (Using a Stack)
    strong_i, strong_j = np.where(strong_mask)
    stack = list(zip(strong_i, strong_j))
    
    while stack:
        x, y = stack.pop()
        
        # 3. CHECK NEIGHBORS
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0: continue # Skip the pixel itself
                
                nx, ny = x + dx, y + dy
                
                # Check if neighbor is inside image boundaries
                if 0 <= nx < M and 0 <= ny < N:
                    
                    # THE CRITICAL CHECK:
                    # Is the neighbor a "Weak" pixel (between weak and strong)?
                    # AND has it not been processed yet?
                    if weak <= img[nx, ny] < strong and final_output[nx, ny] == 0:
                        
                        # Promote it! (Mark as confirmed in final output)
                        final_output[nx, ny] = 255 
                        
                        # Add it to the stack so we can check ITS neighbors later
                        stack.append((nx, ny))

    return final_output

def normalize(img_array: np.ndarray):
    max_val = np.max(img_array)
    if max_val > 0:
        return (img_array / max_val) * 255
    else:
        return img_array

def pipeline1(image):
    image_array = np.array(image) # Convert image to numpy array
    grayscale_image = gray_scale(image_array) # Convert to grayscale
    blurred_image = gaussian_blur(grayscale_image) # Apply Gaussian blur
    edge_image = edge_detection(blurred_image) # Perform edge detection
    return edge_image

def main():
    #This image processing will pollute the image because it uses the same buffer for each stage causing data pollution.
    folder_name = "image_processing"
    image_file = "Shape.jpg"
    image_path = os.path.join(folder_name, image_file)
    image = Image.open(image_path) # Open the image file
    procesed_image_array = pipeline1(image) # Process the image through the pipeline1
    threshold = 150
    result_array = np.where(procesed_image_array > threshold, 255, 0).astype(np.uint8) # Apply thresholding
    result_image = Image.fromarray(result_array) # Convert back to image
    result_image.show() # Display the result
    result_image.save("edge_output.jpg") # Uncomment to save

if __name__ == "__main__":
    print("Running Basic Pixel Processing Module as Main")
    main()