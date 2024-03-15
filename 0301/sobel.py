
import cv2
import numpy as np

def sobel_edge_detection(image_path):
    # Load the image in grayscale
    image = cv2.imread(image_path, 0)

    # Apply Gaussian Blur to smooth the image and reduce noise
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)

    # Apply Sobel edge detection in both horizontal and vertical directions
    sobelx = cv2.Sobel(blurred_image, cv2.CV_64F, 1, 0, ksize=3)  # x-direction
    sobely = cv2.Sobel(blurred_image, cv2.CV_64F, 0, 1, ksize=3)  # y-direction

    # Calculate the magnitude of the gradients
    magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)
    
    # Normalize the magnitude to range 0 to 255
    magnitude = np.uint8(255 * magnitude / np.max(magnitude))

    # Display the original and edge-detected images
    cv2.imshow('Original Image', image)
    cv2.imshow('Sobel Edge Detection', magnitude)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    image_path = "/Users/liushiwen/Desktop/大四下/機器學習系統/0301/EX1_file/horses_copy.jpg"  # Update this to your image path
    sobel_edge_detection(image_path)
