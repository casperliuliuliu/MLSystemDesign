# paste your code below
import cv2
import ImageProcess
import numpy as np
import matplotlib.pyplot as plt

def main(img_path, ii):
    ori_img = cv2.imread(img_path) 
    plt.imshow(ori_img, cmap="gray")

    img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2GRAY)
    img=ImageProcess.Image_Filter(img,'GaussianBlur',show_image=False,size=5)
    post_img=ImageProcess.Edge_Detection(img,'Sobel',gray=False,show_image=False)
    _, th1 = cv2.threshold(post_img,85,255,cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(th1, cv2.MORPH_CLOSE, kernel)
    closed = cv2.dilate(closed, None, iterations=2)
    closed = cv2.erode(closed, None, iterations=2)

    (cnts, _) = cv2.findContours(closed,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    possible_img = ori_img.copy()

    for c in sorted(cnts, key=cv2.contourArea, reverse=True):
        rect = cv2.minAreaRect(c)
        Box = np.intp(cv2.boxPoints(rect))
        Box=ImageProcess.order_points_new(Box) # return  左上/右上/右下/左下 (x,y)

        if  20<Box[2][1]-Box[1][1]<80  and 60<Box[1][0]-Box[0][0]<200  and -3<Box[0][1]-Box[1][1]<10  :
            possible_img = cv2.drawContours(possible_img, [Box], -1, (255, 0, 255), 3)
            plt.imshow(possible_img[:,:,::-1], cmap="gray")
            plt.show()
            break
    # cv2.imwrite(f'output/0{ii}.jpg', possible_img)
    
if __name__ == "__main__":
    image_file = ["image/01.jpg", "image/02.jpg", "image/03.jpg"]
    for ii in range(3):
        main(image_file[ii], ii)
