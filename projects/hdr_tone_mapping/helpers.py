import cv2
import glob
import matplotlib.pyplot as plt
from ipywidgets import *
import tone_mapping_methods as tm
import warnings
warnings.filterwarnings('ignore')

def display_single_image(image, title):
    
    plt.figure(title, figsize=(9, 6))

    plt.imshow(image) 
    plt.axis('off') 

    plt.tight_layout()
    plt.show()

def display_images(images, num_rows, num_cols, title):

    num_images = len(images)
    
    fig = plt.figure(title, figsize=(9, 6))

    for i in range(num_images):
        fig.add_subplot(num_rows, num_cols, i+1) 
        plt.imshow(images[i]) 
        plt.axis('off') 
        plt.title('[' + str(i+1) + ']') 

    plt.tight_layout()
    plt.show()

def display_images_link(images, num_rows, num_cols, title):

    num_images = len(images)
    
    fig = plt.figure(title, figsize=(9, 6))

    ax = None
    for i in range(num_images):
        if i == 0:
            ax = fig.add_subplot(num_rows, num_cols, i+1) 
        else:
            fig.add_subplot(num_rows, num_cols, i+1, sharex=ax, sharey=ax) 
        plt.imshow(images[i]) 
        plt.axis('off') 
        plt.title('[' + str(i+1) + ']') 

    plt.tight_layout()
    plt.show()

def read_images(img_path):
    
    # Get list of image files
    img_files = glob.glob(img_path)
    
    images = []
    for file in img_files:
        img = cv2.imread(file, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)  # Read the image as grayscale
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if img is not None:
            images.append(img)
        else:
            print(f"Failed to read {file}")
    return images

def enhanced_local_tone_mapping_live_tuning(images):
    fig1 = plt.figure('Live Tuning', figsize=(9, 6))

    img_titles = ["desk", "nancy church", "snow", "design center", "bridge", "warwick", "belgium", "bottles", "mountain", "memorial", "cathedral", "office", "oxford church", "seymour park", "kitchen"]

    def update(img_idx=5, p=0.04, cmin=0.1, cmax=0.9, lambda_coarse=0.02, eta_coarse=1.5, lambda_fine=1, eta_fine=1, s=1):
        global img
        img = images[img_idx]
        plt.imshow(tm.enhanced_local_tone_mapping(img, p, cmin, cmax, lambda_coarse, eta_coarse, lambda_fine, eta_fine, s), cmap='gray')
        plt.axis('off')
        fig1.canvas.draw_idle()

    # Create the dropdown widget for image selection
    img_dropdown = widgets.Dropdown(options=[(img_titles[i], i) for i in range(len(images))], value=0, description='Select Image:', style={'description_width': 'initial'}, layout=widgets.Layout(width='75%'))

    interactive_ui = interactive(update,
        img_idx=img_dropdown,
        p=widgets.FloatSlider(value=0.04, min=0.00001, max=1, step=0.00001, description='Brightness ', style={'description_width': 'initial'}, layout=widgets.Layout(width='75%'), readout_format='.5f'),
        cmin=widgets.FloatSlider(value=0.1, min=0, max=0.4, step=0.001, description='Global Contrast Trade-Off ', style={'description_width': 'initial'}, layout=widgets.Layout(width='75%'), readout_format='.5f'),
        cmax=widgets.FloatSlider(value=0.9, min=0.6, max=2, step=0.001, description='Local Contrast Trade-Off ', style={'description_width': 'initial'}, layout=widgets.Layout(width='75%'), readout_format='.5f'),
        lambda_coarse=widgets.FloatSlider(value=0.02, min=0, max=1, step=0.01, description='Local Contrast Detail Limit ', style={'description_width': 'initial'}, layout=widgets.Layout(width='75%'), readout_format='.5f'),
        eta_coarse=widgets.FloatSlider(value=1.5, min=0, max=3, step=0.01, description='Local Contrast Gain ', style={'description_width': 'initial'}, layout=widgets.Layout(width='75%'), readout_format='.5f'),
        lambda_fine=widgets.FloatSlider(value=1, min=0, max=1, step=0.01, description='Micro Contrast Detail Limit ', style={'description_width': 'initial'}, layout=widgets.Layout(width='75%'), readout_format='.5f'),
        eta_fine=widgets.FloatSlider(value=1, min=0, max=3, step=0.01, description='Micro Contrast Gain', style={'description_width': 'initial'}, layout=widgets.Layout(width='75%'), readout_format='.5f'),
        s=widgets.FloatSlider(value=1, min=0, max=2, step=0.01, description='Saturation ', style={'description_width': 'initial'}, layout=widgets.Layout(width='75%'), readout_format='.5f')
    )

    # Display the interactive UI
    display(interactive_ui)