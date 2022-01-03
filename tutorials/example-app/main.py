# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 20:37:29 2020

@author: aniket wattamwar
"""

import streamlit as st
from PIL import Image
import cv2 
import numpy as np



def main():

    selected_box = st.sidebar.selectbox(
    'Choose one of the following',
    ('Welcome','Image Processing', 'Video', 'Face Detection', 'Feature Detection', 'Object Detection')
    )
    
    if selected_box == 'Welcome':
        welcome() 
    if selected_box == 'Image Processing':
        photo()
    if selected_box == 'Video':
        video()
    if selected_box == 'Face Detection':
        face_detection()
    if selected_box == 'Feature Detection':
        feature_detection()
    if selected_box == 'Object Detection':
        object_detection() 
 

def welcome():
    
    st.title('Image Processing using Streamlit')
    
    st.subheader('A simple app that shows different image processing algorithms. You can choose the options'
             + ' from the left. I have implemented only a few to show how it works on Streamlit. ' + 
             'You are free to add stuff to this app.')
    
    st.image('hackershrine.jpg',use_column_width=True)


def load_image(filename):
    image = cv2.imread(filename)
    return image
 
def photo():

    st.header("Thresholding, Edge Detection and Contours")
    
    if st.button('See Original Image of Tom'):
        
        original = Image.open('tom.jpg')
        st.image(original, use_column_width=True)
        
    image = cv2.imread('tom.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    x = st.slider('Change Threshold value',min_value = 50,max_value = 255)  

    ret,thresh1 = cv2.threshold(image,x,255,cv2.THRESH_BINARY)
    thresh1 = thresh1.astype(np.float64)
    st.image(thresh1, use_column_width=True,clamp = True)
    
    st.text("Bar Chart of the image")
    histr = cv2.calcHist([image],[0],None,[256],[0,256])
    st.bar_chart(histr)
    
    st.text("Press the button below to view Canny Edge Detection Technique")
    if st.button('Canny Edge Detector'):
        image = load_image("jerry.jpg")
        edges = cv2.Canny(image,50,300)
        cv2.imwrite('edges.jpg',edges)
        st.image(edges,use_column_width=True,clamp=True)
      
    y = st.slider('Change Value to increase or decrease contours',min_value = 50,max_value = 255)     
    
    if st.button('Contours'):
        im = load_image("jerry1.jpg")
          
        imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        ret,thresh = cv2.threshold(imgray,y,255,0)
        image, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        
        img = cv2.drawContours(im, contours, -1, (0,255,0), 3)
 
        
        st.image(thresh, use_column_width=True, clamp = True)
        st.image(img, use_column_width=True, clamp = True)
         

    
def video():
    uploaded_file = st.file_uploader("Choose a video file to play")
    if uploaded_file is not None:
         bytes_data = uploaded_file.read()
 
         st.video(bytes_data)
         
    video_file = open('typing.mp4', 'rb')
         
 
    video_bytes = video_file.read()
    st.video(video_bytes)
 

def face_detection():
    
    st.header("Face Detection using haarcascade")
    
    if st.button('See Original Image'):
        
        original = Image.open('friends.jpeg')
        st.image(original, use_column_width=True)
    
    
    image2 = cv2.imread("friends.jpeg")

    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(image2)
    print(f"{len(faces)} faces detected in the image.")
    for x, y, width, height in faces:
        cv2.rectangle(image2, (x, y), (x + width, y + height), color=(255, 0, 0), thickness=2)
    
    cv2.imwrite("faces.jpg", image2)
    
    st.image(image2, use_column_width=True,clamp = True)
 

def feature_detection():
    st.subheader('Feature Detection in images')
    st.write("SIFT")
    image = load_image("tom1.jpg")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()    
    keypoints = sift.detect(gray, None)
     
    st.write("Number of keypoints Detected: ",len(keypoints))
    image = cv2.drawKeypoints(image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    st.image(image, use_column_width=True,clamp = True)
    
    
    st.write("FAST")
    image_fast = load_image("tom1.jpg")
    gray = cv2.cvtColor(image_fast, cv2.COLOR_BGR2GRAY)
    fast = cv2.FastFeatureDetector_create()
    keypoints = fast.detect(gray, None)
    st.write("Number of keypoints Detected: ",len(keypoints))
    image_  = cv2.drawKeypoints(image_fast, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    st.image(image_, use_column_width=True,clamp = True)

    
    
def object_detection():
    
    st.header('Object Detection')
    st.subheader("Object Detection is done using different haarcascade files.")
    img = load_image("clock.jpg")
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
    
    clock = cv2.CascadeClassifier('haarcascade_wallclock.xml')  
    found = clock.detectMultiScale(img_gray,  
                                   minSize =(20, 20)) 
    amount_found = len(found)
    st.text("Detecting a clock from an image")
    if amount_found != 0:  
        for (x, y, width, height) in found:
     
            cv2.rectangle(img_rgb, (x, y),  
                          (x + height, y + width),  
                          (0, 255, 0), 5) 
    st.image(img_rgb, use_column_width=True,clamp = True)
    
    
    st.text("Detecting eyes from an image")
    
    image = load_image("eyes.jpg")
    img_gray_ = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
    img_rgb_ = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
        
    eye = cv2.CascadeClassifier('haarcascade_eye.xml')  
    found = eye.detectMultiScale(img_gray_,  
                                       minSize =(20, 20)) 
    amount_found_ = len(found)
        
    if amount_found_ != 0:  
        for (x, y, width, height) in found:
         
            cv2.rectangle(img_rgb_, (x, y),  
                              (x + height, y + width),  
                              (0, 255, 0), 5) 
        st.image(img_rgb_, use_column_width=True,clamp = True)
    
    
    
    
if __name__ == "__main__":
    main()
