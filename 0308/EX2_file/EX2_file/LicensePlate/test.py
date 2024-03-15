# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 12:41:13 2021

@author: jacky
"""
import cv2
import numpy as np
# In[]: p25

MORPH_CROSS = cv2.getStructuringElement(cv2.MORPH_CROSS, (10, 5))
print('\n MORPH_CROSS:\n',MORPH_CROSS)
MORPH_ELLIPSE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
print('\n MORPH_ELLIPSE:\n',MORPH_ELLIPSE)
MORPH_RECT = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
print('\n MORPH_RECT:\n',MORPH_RECT)


# In[]: p25
kernel1 = np.ones((3, 3), np.uint8)
kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
print('\nkernel1')
print(kernel1)
print('\nkernel2')
print(kernel2)
print('\nResult')
print(kernel1 == kernel2)

