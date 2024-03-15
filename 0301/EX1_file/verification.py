# In[]: 匯入package函式庫 執行後不會有結果產生
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
from PIL import Image 
import numpy as np
import cv2 as cv

# In[]: 驗證matplotlib是否可正確執行 執行後可以看到狗的照片與照片維度(576,768,3)印出
img = mpimg.imread('dog.jpg') 
plt.imshow(img)
plt.show()
print(img.shape)

# In[]: 驗證pillow是否可正確執行 執行過後可以看到一個額外視窗顯示老鷹的照片
im = Image.open('eagle.jpg')
im.show()

# In[]: 驗證numpy是否可正確執行 執行之後可以看到兩欄矩陣和一行維度(4,2,2)印出
arr = np.arange(15).reshape(3,5)
print(arr)
print(arr.T)
arr = np.arange(16).reshape(2,2,4)
print(arr.transpose(2,1,0).shape)

# In[]: 驗證opencv是否可正確執行 執行後可看到資料夾中多一個horses_copy.jpg的圖片，與horses.jpg一模一樣
img = cv.imread('horses.jpg')
cv.imwrite('horses_copy.jpg',img)

# In[]: 綜合驗證 執行後可以看到老鷹與轉置的老應
im_array = np.array(im)
plt.imshow(im_array)
plt.show()
im_array = im_array.transpose(1,0,2)
plt.imshow(im_array)
plt.show()
print('劉世文 B093040051')

# %%
import cv2

def crop_and_resize_image(image_path, output_size=1024):
    """
    Reads an image, crops the largest square from the center, and resizes it to the specified size.

    Args:
    - image_path (str): The path to the image file.
    - output_size (int, optional): The size of the output image's width and height in pixels. Defaults to 1024.

    Returns:
    - The cropped and resized image as a numpy array.
    """
    # Load the image
    image = cv2.imread(image_path)

    # Determine the shortest side to crop the image to a square
    height, width = image.shape[:2]
    new_size = min(height, width)

    # Calculate the cropping coordinates
    startx = width // 2 - new_size // 2
    starty = height // 2 - new_size // 2

    # Crop the image to a square
    cropped_image = image[starty:starty+new_size, startx:startx+new_size]

    # Resize the image
    resized_image = cv2.resize(cropped_image, (output_size, output_size))

    return resized_image

# Example usage
image_path = "/Users/liushiwen/Downloads/powerman.png"  # Update this with the actual path to your image
resized_image = crop_and_resize_image(image_path)
cv2.imwrite("BIG_powerman.png", resized_image)
# Display the result
cv2.imshow('Cropped and Resized Image', resized_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# %%
