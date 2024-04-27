#paste your code below
import cv2
import numpy as np
import matplotlib.pyplot as plt 

def increase_contrast(img, alpha=1.5, beta=0):
    img_float32 = np.float32(img)
    contrast_img = np.clip(alpha * img_float32 + beta, 0, 255)
    contrast_img = np.uint8(contrast_img)
    return contrast_img

def get_cat_outline(img):
    gray = img[:,:,2]
    blur = cv2.GaussianBlur(gray, (7, 7), 20)
    contrast = increase_contrast(blur, 2)
    blur = cv2.GaussianBlur(contrast, (7, 7), 0)
    edges = cv2.Canny(blur, 50, 150)
    return edges

if __name__ == "__main__":
    ori_img = cv2.imread('input/cat.jpg')
    cat_outline = get_cat_outline(ori_img)
    cv2.imwrite('output/cat.jpg', cat_outline)

    plt.imshow(cat_outline, cmap="gray")
    plt.show()