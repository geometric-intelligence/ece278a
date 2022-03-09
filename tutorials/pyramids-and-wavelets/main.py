# installl dependencies necessary
import streamlit as st
import numpy as np
from Pyramids.pyramids import display_pyramids
from Wavelets.wavelets import display_wavelets

# Title
st.title("Pyramids and Wavelets")

st.subheader("by Brycen Westgarth & Jackie Burd")

add_selectbox = st.sidebar.selectbox(
    "Page Select",
    ("Pyramids", "Wavelets")
)

if add_selectbox == 'Pyramids':
    display_pyramids()
else:
    display_wavelets()

