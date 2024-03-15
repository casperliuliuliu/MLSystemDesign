# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 02:18:06 2021

@author: jacky
"""

# In[1]: p4
    
import cv2 
import matplotlib.pyplot as plt 

img = cv2.imread("ori_image/lenna.jpg")
print ("lenna.jpg shape:",img.shape)
print ("lenna.jpg dtype:",img.dtype)
print ("//-------------------------------------------")
print ("original RGB lenna image")
plt.imshow(img[:,:,::-1])
plt.show()
print ("img shape:",img.shape)
print ("img dtype:",img.dtype)
print ("//-------------------------------------------")
print ("grayscale lenna image")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.imshow(img_gray, cmap="gray")
plt.show()
print ("img_gray shape:",img_gray.shape)
print ("img_gray dtype:",img_gray.dtype)
print ("//-------------------------------------------")  
#opencv save image   
cv2.imwrite('ouutput/lenna_img_gray.jpg', img_gray) 

# In[2]: p6
'''
1. Sobel
2. Scharr
3. Laplacian
4. Canny
'''    
# In[3]: p6
print ("//-------------------------------------------")
print ("Edge Detection- Sobel")
# CV_16S : signed 16-bit
x = cv2.Sobel(img_gray, cv2.CV_16S, 1, 0, ksize=3) # 計算水平
y = cv2.Sobel(img_gray, cv2.CV_16S, 0, 1, ksize=3) # 計算垂直

# 轉回uint8
absX = cv2.convertScaleAbs(x)
absY = cv2.convertScaleAbs(y)

#show with plt
print("sobel img absX")
plt.imshow(absX, cmap="gray")
plt.show()
print ("//-------------------------------------------")
print("sobel img absY")
plt.imshow(absY, cmap="gray")
plt.show()
print ("//-------------------------------------------")
print("sobel img (absY+absY)*0.5")
sobel_img = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
plt.imshow(sobel_img, cmap="gray")
plt.show()
print ("//-------------------------------------------")
# In[4]: p8
#show with opencv
print ('跳出opencv視窗 請勿直接關閉!!!') 
print ('在opencv視窗中 按任意鍵即可跳出')
cv2.imshow("img", img)
cv2.imshow("img_gray", img_gray)
cv2.imshow("absY", absY)
cv2.imshow("absX", absX)  
cv2.imshow("sobel_img", sobel_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
    
#opencv save image
cv2.imwrite('ouutput/lenna_sobel_absY.jpg', absY) 
cv2.imwrite('ouutput/lenna_sobel_absX.jpg', absX) 
cv2.imwrite('ouutput/lenna_sobel.jpg', sobel_img) 
print ("//-------------------------------------------")
# In[5]: p9
    
def Edge_Detection(img,method,gray=True,XY=(1,1),Weight=(0.5,0.5,0),th=(30,150),size=3,save_image=True,show_image=True):
    # RGB to grayscale
    if gray:
        try:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        except:
            print ('Error!!! check your [img] shape/dtype')
            return
       
    # Edge_Detection
    if method=='Sobel':
        img_x = cv2.Sobel(img, cv2.CV_16S, XY[0], 0, ksize=size) # 計算水平
        img_y = cv2.Sobel(img, cv2.CV_16S, 0, XY[1], ksize=size) # 計算垂直

    elif method=='Scharr':
        img_x = cv2.Scharr(img,cv2.CV_64F, XY[0], 0)
        img_y = cv2.Scharr(img,cv2.CV_64F, 0, XY[1])
    
    elif method=='Laplacian':
        post_img = cv2.Laplacian(img, cv2.CV_16S, ksize=size)
        post_img = cv2.convertScaleAbs(post_img)
        
    elif method=='Canny':
        if gray:
            post_img = cv2.Canny(img, th[0], th[1], apertureSize=size)
        else:
            print ('Error!!! Canny must be converted to grayscale, set gray=True')
            return
    else:
        print ('method have to be Sobel, Scharr, Laplacian or Canny')
        
    
    # addWeight for Sobel and Scharr
    if method=='Sobel' or method=='Scharr':
        absX = cv2.convertScaleAbs(img_x)
        absY = cv2.convertScaleAbs(img_y)
        post_img=cv2.addWeighted(absX, Weight[0], absY, Weight[1], Weight[2])
       
    # save image or not    
    if save_image: cv2.imwrite('ouutput/lenna_'+method+'.jpg', post_img) 
    
    print ('edge detect method:',method)
    print ('post_img.shape:',post_img.shape)
    print ('post_img.dtype',post_img.dtype)
    print ("//-------------------------------------------")
    
    # show image or not 
    if show_image:
        cv2.imshow(str(method)+"_post_img", post_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    return post_img
        
# In[6]: p13
'''
Edge_Detection(img,method='Sobel,gray=True',XY=(1,1),Weight=(0.5,0.5,0),th=None,size=3,save_image=True)  
XY must not contain 0 (i.e. xy=(1,0), xy=(0,1) will be error) 

Try yourself 
'''
#sobel 
Edge_Detection(img,'Sobel') 
Edge_Detection(img,'Sobel',XY=(2,1),size=5) 
 
#Scharr 
Edge_Detection(img,'Scharr')  

#Laplacian 
Edge_Detection(img,'Laplacian',gray=False) 
Edge_Detection(img,'Laplacian') 
#Canny 
post_img=Edge_Detection(img,'Canny')      
        
# In[7]: p14           
'''
Binarization without filter
'''
post_img=Edge_Detection(img,'Sobel',XY=(2,2),size=5,show_image=False)  
ret, th1 = cv2.threshold(post_img,127,255,cv2.THRESH_BINARY)
cv2.imshow('th1_post_img', th1)
cv2.imshow('post_img', post_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

     
# In[8]: p15 
'''
pre-filter or post-filter
'''
''' set pre-filter '''

#pre_img = cv2.blur(img, (3,3))
pre_img = cv2.GaussianBlur(img, (5, 5), 0)
#pre_img = cv2.medianBlur(img, 3)
#pre_img = cv2.bilateralFilter(img,3,75,75)

post_img=Edge_Detection(pre_img,'Sobel',XY=(2,2),size=5,show_image=False)  

''' set post-filter '''
#th1 = cv2.blur(th1, (3,3))
#th1 = cv2.GaussianBlur(th1, (3, 3), 0)
#th1 = cv2.medianBlur(th1, 3)
#th1 = cv2.bilateralFilter(th1,3,75,75)

ret, th1 = cv2.threshold(post_img,127,255,cv2.THRESH_BINARY)


cv2.imshow('th1_post_img', th1)
cv2.imshow('post_img', post_img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[]: p17

def Image_Filter (img,method,size=3,show_image=False): 
    if method=='blur':
        img = cv2.blur(img, (size,size))
    elif method=='GaussianBlur':
        img = cv2.GaussianBlur(img, (size, size), 0)
    elif method=='medianBlur':
        img = cv2.medianBlur(img, size)
    elif method=='bilateralFilter':
        img = cv2.bilateralFilter(img,size,75,75)
    else:
        print ('method have to be blur, GaussianBlur, medianBlur or bilateralFilter')
    if show_image:
        cv2.imshow("Image_Filter", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return img

# In[]: p18
''' 上課練習-- 試著調整配置 輸出最佳邊緣 '''
img = cv2.imread("ori_image/lenna.jpg")
# turn to grayscale first
pre_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
# pre-filter
pre_img = Image_Filter(pre_img,'GaussianBlur',size=3,show_image=True)
# edge detection
post_img=Edge_Detection(pre_img,'Sobel',gray=False,show_image=True)
# post-filter
post_img=Image_Filter(post_img,'GaussianBlur',show_image=True)
# Binarization
ret, th1 = cv2.threshold(post_img,127,255,cv2.THRESH_BINARY)



cv2.imshow('post_img', post_img)
cv2.imshow('th1_post_img', th1)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('ouutput/lenna_final.jpg', th1)


# In[]: p19
''' 回家作業1-- 試著調整配置 輸出最佳邊緣 '''  



