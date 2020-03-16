#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import matplotlib.pyplot as plt

def preprocess(img,k,p):
    h,w = img.shape[:2]

    ## UNDISTORTING THE IMAGE AND SELECTING ROI ##
    dst = cv2.undistort(img,k,p, None)
    #dst = dst[h//3:h,0:w-100]

    ## DENOISING THE IMAGE ##
    #blur = cv2.blur(dst,(3,3))
    #blur = cv2.bilateralFilter(dst,9,75,75)
    
    return dst

# Function to warp the lane
def lane_process(img):
    hls_img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS).astype(np.float)
    
    #Seperate yellow
    lower_yellow = np.array([20,0,100],dtype=np.uint8)
    upper_yellow = np.array([40,220,255],dtype=np.uint8)
    yellow_mask = cv2.inRange(hls_img,lower_yellow,upper_yellow)
    
    #Seperate White
    lower_white = np.array([0,200,0],dtype=np.uint8)
    upper_white = np.array([255,255,255],dtype = np.uint8)
    white_mask = cv2.inRange(hls_img,lower_white,upper_white)
    
    #Create thesholded image based on yellow OR white color
    combined_frames = np.zeros((img.shape[0], img.shape[1]))
    for row in range(0, hls_img.shape[0]):
        for col in range(0, hls_img.shape[1]):
            if(yellow_mask[row, col] > 200 or white_mask[row, col] > 200):
                combined_frames[row,col] = 255
                
    #lanes = cv2.bitwise_or(yellow_detect,white_detect)
#     lanes = cv2.cvtColor(lanes,cv2.COLOR_BGR2GRAY)
#     ret,lanes = cv2.threshold(lanes,120,255,cv2.THRESH_BINARY)
    
    return combined_frames

def image_warp(img,src_pts):
    dest_pts = np.float32([[0,0],[300,0],[0,300],[300,300]])
    H = cv2.getPerspectiveTransform(src_pts,dest_pts)
    out = cv2.warpPerspective(img,H,(300,300))
    #out = cv2.warpPerspective(img,H,(img.shape[1],img.shape[0]))
    return out

def lane_line_fit(lanes):
    
    nwindows = 10
    window_height = np.int(lanes.shape[0]/nwindows)
    #window_width = np.int(lanes.shape[1]/nwindows)
    margin= 50
    #Using histogram to get pixels with max value
    
    hist = np.sum(lanes[lanes.shape[0]//nwindows:,:],axis=0) #Calculating histogram
    
    plt.plot(hist)
    plt.show()
    
    channel_out = np.dstack((lanes,lanes,lanes))*255
    midpoint = int(hist.shape[0]/2)
    image_center = int(lanes.shape[1]/2)
    #getting the max values from Left and right lane 
    left_lane_ix = np.argmax(hist[:midpoint]) #gives index of the maximum pixel in hist(0 to midpoint) : (white pixel)
    right_lane_ix = np.argmax(hist[midpoint:])+midpoint #Adding midpoint for bias from 0

    nwindows = 10
    window_height = np.int(lanes.shape[0]/nwindows)
    #window_width = np.int(lanes.shape[1]/nwindows)
    margin= 30
    nonzero_pts = lanes.nonzero()
    nonzero_x = np.array(nonzero_pts[1])
    #print(nonzero_x)
    nonzero_y = np.array(nonzero_pts[0])

    #cv2_imshow('Lanes',lanes)
    # print('Non zero pixel points',nonzero_pts)
    #print('X nonzero',nonzero_x)
    #print('Y nonzero',nonzero_y)
    
    left_shift = left_lane_ix
    right_shift = right_lane_ix
    leftLane_px = []
    rightLane_px = []

    better_left_fit = np.zeros(3)
    better_right_fit = np.zeros(3)

    for window in range(nwindows):
        window_left_border_LL = left_shift - margin
        window_right_border_LL = left_shift + margin
        window_left_border_RL = right_shift - margin
        window_right_border_RL = right_shift + margin
        window_top_border = lanes.shape[0]-  window*window_height
        window_bottom_border = lanes.shape[0] -(window+1)*window_height

#         cv2.rectangle(channel_out,(window_left_border_LL, window_top_border),(window_right_border_LL, window_bottom_border),(0,255,0),2)
#         cv2.rectangle(channel_out,(window_left_border_RL, window_top_border),(window_right_border_RL, window_bottom_border),(0,0,255),2)

        desired_pixels_leftLane = ((nonzero_y >= window_bottom_border) & 
                                   (nonzero_y < window_top_border) & 
                                   (nonzero_x >= window_left_border_LL) & 
                                   (nonzero_x < window_right_border_LL)).nonzero()[0]
        desired_pixels_rightLane = ((nonzero_y >= window_bottom_border) & 
                                    (nonzero_y < window_top_border) & 
                                    (nonzero_x >= window_left_border_RL) & 
                                    (nonzero_x < window_right_border_RL)).nonzero()[0]
        #desired_pixels_leftLane = lanes[window_top_border:window_bottom_border,window_left_border_LL:window_right_border_LL].nonzero()[0]
        #desired_pixels_rightLane = lanes[window_top_border:window_bottom_border,window_left_border_RL:window_right_border_RL].nonzero()[0]
        leftLane_px.append(desired_pixels_leftLane)
        rightLane_px.append(desired_pixels_rightLane)
        #print('White px indices left lane',leftLane_px)
        #print('White px indices right lane',desired_pixels_rightLane)
        if len(desired_pixels_leftLane) > 5:
            left_shift = int(np.mean(nonzero_x[desired_pixels_leftLane]))
        #print(left_shift)
        if len(desired_pixels_rightLane) > 5:
            right_shift = int(np.mean(nonzero_x[desired_pixels_rightLane]))

    #cv2_imshow('Left Lane',channel_out)
    #print('Left lane',leftLane_px)
    leftLane_px = np.concatenate(leftLane_px)
    rightLane_px = np.concatenate(rightLane_px)

    #Left lane pixels are given as
    Leftx = nonzero_x[leftLane_px]
    Lefty = nonzero_y[leftLane_px]
    
    #Right lane pixels are
    Rightx = nonzero_x[rightLane_px]
    Righty = nonzero_y[rightLane_px]
    print('RIGHT X-Y Length: ',len(Rightx),len(Righty))
    xm = 3.65/270
    ym = 30/300
    
    if len(Rightx)>5:
        right_fit = np.polyfit(Righty,Rightx,2)
        #print('checkpoint')
        right_fit_meters = np.polyfit(Righty*ym,Rightx*xm,2)
        prev_fit_right.append(right_fit)
        prev_fit_right_m.append(right_fit_meters)
    else:
        right_fit = prev_fit_right[-1].copy()
        right_fit_meters = prev_fit_right_m[-1].copy()
    
    if len(Leftx)>5:
        left_fit = np.polyfit(Lefty,Leftx,2)
        left_fit_meters = np.polyfit(Lefty*ym,Leftx*xm,2)
        prev_fit_left.append(left_fit)
        prev_fit_left_m.append(left_fit_meters)
    else:
        left_fit = prev_fit_left[-1].copy()
        left_fit_meters = prev_fit_left_m[-1].copy()

    left_line = []
    right_line = []
    x_pts = []


    for i in range(lanes.shape[0]):
      #y1 = better_left_fit[0]*i**2 + better_left_fit[1]*i + better_left_fit[2]
        y1 = left_fit[0]*i**2 + left_fit[1]*i + left_fit[2]
        x_pts.append(i)
        left_line.append(y1)
          #left_line_y.append(i)
        y2 = right_fit[0]*i**2 + right_fit[1]*i + right_fit[2]    
        right_line.append(y2) 

    left_pts = np.array([np.transpose(np.vstack([left_line, x_pts]))])
    right_pts = np.array([np.flipud(np.transpose(np.vstack([right_line, x_pts])))])

    #print('right line pts', right_pts)
    # print('left line pts', left_pts)
    pts = np.hstack((left_pts, right_pts))
    #Fill poly must come here on channel_out
    cv2.fillPoly( channel_out, np.int_([pts]), (255,0, 0))
    cv2.polylines(channel_out, np.int32([left_pts]), isClosed=False, color=(0,0,255), thickness=10)
    cv2.polylines(channel_out, np.int32([right_pts]), isClosed=False, color=(0,0,255), thickness=10)
    #Fitting the curve on these points
    # Pixel to meters
    
    #right_fit_meters = np.polyfit(Righty*ym,Rightx*xm,2)
    #left_fit_meters = np.polyfit(Lefty*ym,Leftx*xm,2)
    # Curvature of left and right lanes
    left_radius = ((1 + (2*left_fit_meters[0]*ym*300 + left_fit_meters[1])**2)**1.5) / np.absolute(2*left_fit_meters[0])
    right_radius = ((1 + (2*right_fit_meters[0]*ym*300 +right_fit_meters[1])**2)**1.5) / np.absolute(2*right_fit_meters[0])
    roc = left_radius+right_radius/2
    return right_fit,left_fit,channel_out, roc

def dewarp(img, src_pts):
    dest_pts = np.float32([[252,250], [403,250], [105,343], [569,343]])
    H1= cv2.getPerspectiveTransform(src_pts,dest_pts)
    out= cv2.warpPerspective(img, H1, (640,360))
  #out = cv2.warpPerspective(img,H1,(img.shape[1],img.shape[0]))
    return out


cam_mtx = np.array([[  1.15422732e+03 ,  0.00000000e+00 ,  6.71627794e+02],
 [  0.00000000e+00 ,  1.14818221e+03 ,  3.86046312e+02],
 [  0.00000000e+00 ,  0.00000000e+00 ,  1.00000000e+00]],dtype=np.int32)

dist_mtx = np.array([[ -2.42565104e-01 , -4.77893070e-02 , -1.31388084e-03 , -8.79107779e-05,
    2.20573263e-02]],dtype=float)
dest_pts = np.float32([[0,0],[300,0],[0,300],[300,300]])
#src = np.float32([[230,250], [430,250], [10,350], [600,350]])
src = np.float32([[252,250], [403,250], [105,343], [569,343]])

font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (30,70)
fontScale              = 0.7
fontColor              = (0,0,255)
lineType               = 2
#prev_fit_right_a, prev_fit_right_b, prev_fit_right_c =[],[],[]
#prev_fit_left_a,prev_fit_left_b,prev_fit_left_c=[],[],[]
prev_fit_right=[]
prev_fit_left=[]
prev_fit_right_m = []
prev_fit_left_m = []

cap = cv2.VideoCapture('/home/akhopkar/Documents/challenge_video.mp4')
out = cv2.VideoWriter('challenge_test.avi',cv2.VideoWriter_fourcc(*'XVID'), 30, (640,360))
if cap.isOpened == False:
    print('Error Loading!')
Frame = 0
while cap.isOpened():
    ret,img = cap.read()
    if ret == False:
        break
    img = cv2.resize(img,(0,0),fx=0.5,fy=0.5)
    print(img.shape)
    #cv2.imshow('input',img)
    print('Frame #',Frame)
    pp_img = preprocess(img,cam_mtx,dist_mtx)
    #cv2.imshow('pre-process',pp_img)
    img_process = lane_process(pp_img)
    lanes_warp = image_warp(pp_img,src)
    lanes = image_warp(img_process,src)
    #cv2.imshow('Lanes',lanes)
    #cv2_imshow(lanes_warp)
    # if cv2.waitKey(0) == 27:
    #   cv2.destroyAllWindows
    Frame+=1
    try:
    	right_fit, left_fit, out_img, roc = lane_line_fit(lanes)
    	print('Right fit info:',len(right_fit))
    	print('Left fit info:',len(left_fit))

    	lane_dewarp = dewarp(out_img, dest_pts)
    #cv2.imshow('lane dewarp',lane_dewarp)

    	final_frame = cv2.addWeighted(np.uint8(img), 1, np.uint8(lane_dewarp), 0.5, 0)
    #cv2_imshow(final_frame)
    #out.write(np.uint8(final_frame))

    	cv2.putText(final_frame,'Radius of Curvature: ' + str(int(roc/1000))+' Km', (30,70), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
(0,0,255), 2)
    	if (int(roc/1000)) > 10:
           pass
    	else:
           cv2.putText(final_frame,'Turn right -- >', (30,100), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
(255,0,0), 2)
           print('Turn right -- >')
	#out.write(np.uint8(final_frame))
    	cv2_imshow(final_frame)
    
    	print('Rendered frame no:', Frame)
    	print('ROC is:', roc/1000, 'Km')
    #Frame+=1
    except:
    #print('Error!')
    	out.write(np.uint8(img))
    
    if cv2.waitKey(5) & 0xFF == 27:
        break

out.release()
cap.release()
cv2.destroyAllWindows()


