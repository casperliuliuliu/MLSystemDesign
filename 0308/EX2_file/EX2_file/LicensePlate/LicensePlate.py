# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 00:17:18 2021

@author: jacky
"""
# In[]
import cv2
import ImageProcess
import numpy as np 
# In[]:
ori_img = cv2.imread("image/02.jpg")

cv2.imshow('img', ori_img)
cv2.waitKey(0)
img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2GRAY)
img=ImageProcess.Image_Filter(img,'GaussianBlur',show_image=True,size=5)
post_img=ImageProcess.Edge_Detection(img,'Sobel',gray=False,show_image=True)
#post_img=ImageProcess.Image_Filter(post_img,'GaussianBlur',show_image=True,size=7)
ret, th1 = cv2.threshold(post_img,80,255,cv2.THRESH_BINARY)


cv2.imshow('post_img', post_img)
cv2.imshow('th1_post_img', th1)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('ouutput/lenna_final.jpg', th1)

# In[]:
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
#執行影象形態學
closed = cv2.morphologyEx(th1, cv2.MORPH_CLOSE, kernel)
cv2.imshow('erode dilate', closed)
cv2.waitKey(0)
cv2.destroyAllWindows() 
# 腐蝕4次，去掉細節
closed = cv2.erode(closed, None, iterations=4)
cv2.imshow('erode dilate', closed)
cv2.waitKey(0)
cv2.destroyAllWindows() 
# 膨脹4次，讓輪廓突出
closed = cv2.dilate(closed, None, iterations=4)
cv2.imshow('erode dilate', closed)
cv2.waitKey(0)
cv2.destroyAllWindows() 

# (_, cnts, _) = cv2.findContours(closed,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
(cnts, _) = cv2.findContours(closed,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

 
'''
# 參考
opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
cv2.imshow('dilate erode', opened)
cv2.waitKey(0)
cv2.destroyAllWindows()  
# 膨脹10次，讓輪廓突出
opened = cv2.dilate(opened, None, iterations=10)
cv2.imshow('dilate erode', opened)
cv2.waitKey(0)
cv2.destroyAllWindows()  
# 腐蝕4次，去掉細節
opened = cv2.erode(opened, None, iterations=4)
cv2.imshow('dilate erode', opened)
cv2.waitKey(0)
cv2.destroyAllWindows()  

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (18, 7))
opened = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
cv2.imshow('morphologyEx2', opened)
cv2.waitKey(0)
cv2.destroyAllWindows()  


(_, cnts, _) = cv2.findContours(opened,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
'''


# In[]: determine by opencv

possible_img = ori_img.copy()
c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
rect = cv2.minAreaRect(c)
Box = np.int0(cv2.boxPoints(rect))
Final_img = cv2.drawContours(possible_img, [Box], -1, (0, 0, 255), 3)
cv2.imshow('Final_img', Final_img)
cv2.waitKey(0) 
cv2.destroyAllWindows()


# In[]: determine by 工人智慧
possible_img = ori_img.copy()

for c in sorted(cnts, key=cv2.contourArea, reverse=True):
    #print (c)
      
    rect = cv2.minAreaRect(c)
    #print ('rectt', rect)
    Box = np.int0(cv2.boxPoints(rect))
    #print ('Box', Box)    
    Box=ImageProcess.order_points_new(Box) # return  左上/右上/右下/左下 (x,y)
    #print ('Box2',Box)
    
    possible_img = cv2.drawContours(possible_img, [Box], -1, (0, 0, 255), 3)
    cv2.imshow('possible_img', possible_img)
    
#cv2.imshow('possible_img', possible_img) 
cv2.waitKey(0)
cv2.destroyAllWindows() 


# In[]: determine by 工人智慧
# '''
possible_img = ori_img.copy()

for c in sorted(cnts, key=cv2.contourArea, reverse=True):
    #print (c)
      
    rect = cv2.minAreaRect(c)
    #print ('rectt', rect)
    Box = np.int0(cv2.boxPoints(rect))
    #print ('Box', Box)    
    Box=ImageProcess.order_points_new(Box) # return  左上/右上/右下/左下 (x,y)
    #print ('Box2',Box)
    
    # determine by 工人智慧
    
    if ((Box[1][0]-Box[0][0])>(Box[3][1]-Box[0][1])) and abs(Box[0][1]-Box[1][1])<30:
        possible_img = cv2.drawContours(possible_img, [Box], -1, (0, 0, 255), 3)
        cv2.imshow('possible_img', possible_img)
        break
    
#cv2.imshow('possible_img', possible_img) 
cv2.waitKey(0)
cv2.destroyAllWindows() 
# '''


# In[]:  
''' 回家作業2-- 試著調整配置、參數 嘗試以經驗法則/工人智慧 輸出最佳車牌定位結果 ''' 




