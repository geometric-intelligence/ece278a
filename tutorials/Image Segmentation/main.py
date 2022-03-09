"""-------------------------------
ECE 278A Image Processing
Web App: Image Segmentation

Created By
Tainan Song
Roger Lin

This is the main function of the app.
lib/webapp.py contains section specific functions
lib/imgproc.py contains image processing functions
---------------------------------"""
import numpy as np
import scipy
import streamlit as st
import matplotlib.pyplot as plt

from lib.webapp import *
from lib.imgproc import *


def main():
    st.title('Image Segmentation')

    selected_box = st.sidebar.selectbox(
        'Image Segmentation Algorithms',
        ('Introduction', 'Thresholding', 'Region Based', 'Clustering')
    )
    if selected_box == 'Introduction':
        st.header('Introduction')
        intro()

    if selected_box == 'Thresholding':
        st.header('Thresholding')
        threshold()
    if selected_box == 'Region Based':
        st.header('Region Based')
        region()
    if selected_box == 'Clustering':
        st.header('Clustering')
        cluster()


if __name__ == "__main__":
    main()









