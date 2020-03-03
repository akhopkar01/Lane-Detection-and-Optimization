#!/usr/bin/env python
# coding: utf-8

# In[37]:


import cv2
import numpy as np
import matplotlib.pyplot as plt

cap = cv2.VideoCapture('/home/akhopkar/Documents/Night Drive - 2689.mp4')

if cap.isOpened() == False:
    print('Error Loading the video!')


def equalizeFrameHSV(frame):
    #img = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    img = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    h = img[:,:,0]
    s = img[:,:,1]
    v =  img[:,:,2]
    HSV = [h,s,v]
    h = cv2.equalizeHist(HSV[0])
    s = cv2.equalizeHist(HSV[1])
    v = cv2.equalizeHist(HSV[2])
    img = cv2.merge((h,s,v))
    img = cv2.cvtColor(img,cv2.COLOR_HSV2BGR)
    return img

# def plotHist(img):
#     color = ('b','g','r')
#     for i,c in enumerate(color):
#         hist = cv2.calcHist([img],[i],None,[256],[0,256])
#         plt.plot(hist,color=c)
#         plt.xlim([0,256])
#     plt.show()

def equalize_Value(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    img[:,:,2] = cv2.equalizeHist(img[:,:,2])
    img = cv2.cvtColor(img,cv2.COLOR_HSV2BGR)
    return img

def gamma_correction(img,gamma):
    #Output = Input^(1/gamma)
    #Scale the input from (0 to 256) to (0 to 1)
    #Apply gamma correction
    #Scale back to original values
    gamma = 1/gamma
    lT =[]
    for i in np.arange(0,256).astype(np.uint8):
        lT.append(np.uint8(((i/255)**gamma)*255))
    lookup = np.array(lT)
    #Creating the lookup table, cv can find the gamma corrected value of each pixel value
    corrected = cv2.LUT(img,lookup)
    return corrected

def imageCorrection(img,a,b):
    Beta = np.full([img.shape[0],img.shape[1],img.shape[2]],b)
    Out = np.zeros((img.shape),dtype=np.uint8)
    Out = Out + a*img + Beta
    return Out
    

Frame = 0
while cap.isOpened():
    ret, frame = cap.read()
    if ret == False:
        break
    img = cv2.resize(frame, (0,0),fx=0.5,fy=0.5)
    cv2.imshow('Input',img)
#     eq1 = equalizeFrameHSV(img)
    #print(img.shape)
    eq3 = gamma_correction(img,2)
#     #cv2.imshow('Equalized by HSV', eq1)
#     #cv2.imshow('Equalized by VALUE',eq1)
#     cv2.imshow('Gamma Correction',eq3)
#     eq2 = equalize_Value(eq3)
#     cv2.imshow('Check',eq2)
    #eq4 = imageCorrection(img,0.5,0)
    eq5 = cv2.addWeighted(img,1,img,0,10)
    cv2.imshow('Custom',eq5)
    eq6 = cv2.convertScaleAbs(eq3,-1,0.8,3)
    cv2.imshow('Cust2',eq6)

    
    Frame+=1
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()


# In[25]:





# In[ ]:




