import numpy as np
import cv2 as cv
import os 

path = "gesture_data"

"""extracts mask for skin color region of image
    NOTE: sensitive to lighting conditions, works better under semi-cool tone lighting"""
def processAndSegment(input_img:np.array)->np.array:
    # denoise/smooth image 
    input_img = cv.GaussianBlur(input_img, (9,9), 0)
    
    # assuming input image is in BGR format
    blue = input_img[:,:,0]
    green = input_img[:,:,1]
    red = input_img[:,:,2]

    # extract hsv and ycrcb representation of img 
    hsv_img = cv.cvtColor(input_img, cv.COLOR_BGR2HSV_FULL)
    ycrcb = cv.cvtColor(input_img, cv.COLOR_BGR2YCrCb)
    y,cr,cb = np.dsplit(ycrcb, 3)
    h,s,_ = np.dsplit(hsv_img, 3)
    s = s/255.0 # normalize saturation value

    # using thresholding segment image into skin color and non-skin color region
    # threshold values found through reference [1]
    # NOTE: h val is halved because of openCV's hsv implmentation h range being 180 and not 360 
    hs_mask = (0<=h) & (h<=25) & (0.23<=s) & (s<=0.68)
    rgb_mask = (red>95) & (green>40) & (blue>20) & (red>green) & (red>blue) & (abs(red-green)>15)
    ycrcb_mask = (cr>135) & (cr<=(1.5862*cb)+20) & (cr>=(0.3448*cb)+76.2069) & (cr>=(-4.5652*cb)+234.5652) & (cr<=(-1.15*cb)+301.75) & (cr<=(-2.2857*cb)+432.85) & (y>80)
    overall_mask = (np.squeeze(hs_mask) & rgb_mask) | (rgb_mask & np.squeeze(ycrcb_mask))
    return overall_mask

# Isolate the palm from the forearm if applicable 
def isolateWrist(mask:np.array):
    contours,_ = cv.findContours(mask.astype("uint8"),2,1)

    # hand area should be largest portion of image
    region_of_interest = max(contours, key=cv.contourArea)

    x,y,width,height = cv.boundingRect(region_of_interest)
    
    # if ratio of bounding box is greater than threshold, image likely includes forearm
    # this isolates the upper palm and fingers from rest of arm 
    if(width/height > 1.40):
        max_height = 0
        max_height_x = x
        # person can be right or left handed, so must keep track of second widest point 
        second_max_height_x = x
        for i in range(x, x+width):
            points = np.where((region_of_interest[:,0,0] == i))[0]
            if(len(points)>0):
                min_y = np.min(region_of_interest[points, 0,1])
                max_y = np.max(region_of_interest[points, 0,1])
                height = max_y-min_y
                if(height>max_height):
                    max_height = height
                    second_max_height_x = max_height_x
                    max_height_x = i
        # extract palm region before widest point 
        hand_countor = []
        if max_height_x > second_max_height_x:
            for point in region_of_interest:
                if (point[0][0] >= max_height_x-(0.10*width)):
                    hand_countor.append(point)
        else:
            for point in region_of_interest:
                if (point[0][0] <= max_height_x+(0.10*width)):
                    hand_countor.append(point)
        hand_countor = np.array(hand_countor)
        return hand_countor
    elif(height/width > 1.40):
        # Iterate through each horizontal segment of the bounding rectangle
        max_width = 0
        max_width_y = y
        for i in range(y, y + height):
            # Find the width at this horizontal segment
            points = np.where((region_of_interest[:,0,1] == i))[0]
            if (len(points) > 0):
                min_x = np.min(region_of_interest[points, 0, 0])
                max_x = np.max(region_of_interest[points, 0, 0])
                width = max_x - min_x
                if (width > max_width):
                    max_width = width
                    max_width_y = i
        # Extract palm region above widest point 
        hand_countor = []
        for point in region_of_interest:
            if (point[0][1] <= max_width_y+(0.10*height)):
                hand_countor.append(point)
        hand_countor = np.array(hand_countor)
        return hand_countor
    else:
        return region_of_interest

"""Iterates through folders in gesture_data, constructs Hu moments for each example image,
    and then saves Hu moment arrays into corresponding folder"""
def buildHuMomentData():
    for sub_directory, directories, files in os.walk(path):
        for directory_name in directories:
            if(directory_name != 'test_images'):
                sub_path = os.path.join(sub_directory, directory_name)
                files = os.listdir(sub_path)
                hu_moments = np.empty((len(files), 7))
                for ind,file in enumerate(files):
                    file_path = os.path.join(sub_path, file)
                    img = cv.imread(file_path)
                    skin_mask = processAndSegment(img)
                    contour = isolateWrist(skin_mask)
                    # hand region should be largest contour in example image
                    moments = cv.moments(contour)
                    curr_hu = cv.HuMoments(moments)
                    hu_moments[ind] = np.squeeze(curr_hu)
                # save hu moments into corresponding files 
                hu_dest = os.path.join(sub_directory, directory_name+"_hu_moments")
                np.save(hu_dest, hu_moments, allow_pickle=False)         

if __name__ == "__main__":
    buildHuMomentData()