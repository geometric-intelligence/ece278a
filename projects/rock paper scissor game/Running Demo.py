from unittest import result
import cv2
import numpy as np
import copy
import random
import math
# import win32api
# import win32con
 
cap_region_x_begin = 0.5
cap_region_y_end = 0.8
threshold = 60
blurValue = 41  # gaussian
bgSubThreshold = 50
learningRate = 0
airps = cv2.imread('rock.jpg')

isBgCaptured = 0 
triggerSwitch = False 
 
def printThreshold(thr):
    print("! Changed threshold to " + str(thr))
 
def removeBG(frame):
    fgmask = bgModel.apply(frame, learningRate=learningRate)
    kernel = np.ones((3, 3), np.uint8)
    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    res = cv2.bitwise_and(frame, frame, mask=fgmask) 
    return res
 

camera = cv2.VideoCapture(0)
camera.set(10, 200) 
cv2.namedWindow('trackbar') 
cv2.resizeWindow("trackbar", 640, 200) 
cv2.createTrackbar('threshold', 'trackbar', threshold, 100, printThreshold)
cnt = 0


while camera.isOpened():
    ret, frame = camera.read()
    threshold = cv2.getTrackbarPos('threshold', 'trackbar')
    # frame = cv2.cvtColor(frame,cv2.COLOR_RGB2YCrCb)
    frame = cv2.bilateralFilter(frame, 5, 50, 100) 
    frame = cv2.flip(frame, 1) 
    cv2.rectangle(frame, (int(cap_region_x_begin * frame.shape[1]), 0),(frame.shape[1], int(cap_region_y_end * frame.shape[0])), (0, 0, 255), 2)
    # draw the detect area
    cv2.imshow('original', frame) 
 

    if isBgCaptured == 1:
        img = removeBG(frame)  
        img = img[0:int(cap_region_y_end * frame.shape[0]),int(cap_region_x_begin * frame.shape[1]):frame.shape[1]]  
        cv2.imshow('mask', img)
 
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
        blur = cv2.GaussianBlur(gray, (blurValue, blurValue), 0)  # Apply gaussian filter
        cv2.imshow('blur', blur)
        ret, thresh = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY) 
        cv2.imshow('binary', thresh)
 
        # get the coutours
        thresh1 = copy.deepcopy(thresh)
        contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        length = len(contours)
        maxArea = -1
        if length > 0:
            for i in range(length):  # find the largest area
                temp = contours[i]
                area = cv2.contourArea(temp) 
                if area > maxArea:
                    maxArea = area
                    ci = i
 
            res = contours[ci]  
            hull = cv2.convexHull(res)  
            drawing = np.zeros(img.shape, np.uint8)
            cv2.drawContours(drawing, [res], 0, (0, 255, 0), 2)   # draw the contour
            cv2.drawContours(drawing, [hull], 0, (0, 0, 255), 3)  # draw the convex
 
            moments = cv2.moments(res)  
            center = (int(moments['m10'] / moments['m00']), int(moments['m01'] / moments['m00']))
            cv2.circle(drawing, center, 8, (0,0,255), -1)   # draw the center (do not use the lower convex)
 
            fingerRes = []  
            max = 0; count = 0; notice = 0; cnt = 0
            for i in range(len(res)):
                temp = res[i]
                dist = (temp[0][0] -center[0])*(temp[0][0] -center[0]) + (temp[0][1] -center[1])*(temp[0][1] -center[1])
                if dist > max:
                    max = dist
                    notice = i
                if dist != max:
                    count = count + 1
                    if count > 40:
                        count = 0
                        max = 0
                        flag = False 
                        if center[1] < res[notice][0][1]: # the direction of hand
                            continue
                        for j in range(len(fingerRes)):  # if the two points are too close, get rid of it
                            if abs(res[notice][0][0]-fingerRes[j][0]) < 40 :
                                flag = True
                                break
                        if flag :
                            continue
                        fingerRes.append(res[notice][0])
                        cv2.circle(drawing, tuple(res[notice][0]), 8 , (255, 0, 0), -1) 
                        cv2.line(drawing, center, tuple(res[notice][0]), (255, 0, 0), 2)
                        cnt = cnt + 1
 
            cv2.imshow('output', drawing)
            print(cnt)
 
    k = cv2.waitKey(10)
    if k == 27:  # ESC to quit
        break
    elif k == ord('b'):  # capture the background
        bgModel = cv2.createBackgroundSubtractorMOG2(0, bgSubThreshold)
        isBgCaptured = 1
        print('!!!Background Captured!!!')
    elif k == ord('r'):  
        bgModel = None
        triggerSwitch = False
        isBgCaptured = 0
        print('!!!Reset BackGround!!!')
    elif k == ord('s'):
        ai = 2
        player = frame
        ais = ' '
        ps = ' '
        result = ' '
        if ai == 0:
            ais = ' rock'
        elif ai == 1:
            ais = ' paper'
        elif ai == 2:
            ais = ' sessor'

        player = cv2.putText(player,'AI choses'+ais,(20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
        player = cv2.putText(player,'Your choice is',(20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

        if cnt == 1:
            ps='rock'
            if ai == 0:
                result ='Draw'
            elif ai == 1:
                result = 'You Lose'
            elif ai == 2:
                result = 'You win'
        elif cnt == 2 or cnt == 3:
            ps='sessor'
            if ai == 2:
                result ='Draw'
            elif ai == 0:
                result = 'You Lose'
            elif ai == 1:
                result = 'You win'
        elif cnt >= 4:
            ps='papper'
            if ai == 1:
                result ='Draw'
            elif ai == 2:
                result = 'You Lose'
            elif ai == 0:
                result = 'You win'
        player = cv2.putText(player,ps,(20, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
        player = cv2.putText(player,result,(20, 170), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
        cv2.imshow('Player', frame)

    elif k == ord('n'):
        triggerSwitch = True
        print('!!!Trigger On!!!')
