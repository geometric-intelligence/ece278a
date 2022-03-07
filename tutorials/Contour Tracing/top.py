import streamlit as st
from webcam import webcam
import level_set_kerr_2 as lsk2
import numpy as np
from skimage.color import rgba2rgb
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage import color, io
from matplotlib.colors import Normalize
import matplotlib.cm as cmx
from PIL import Image
import cv2

from skimage.color import rgb2gray
from skimage import data
from skimage.filters import gaussian
from skimage.segmentation import active_contour

def load_image(filename):
    image = io.imread(filename)
    return image

def main():

    selected_box = st.sidebar.selectbox(
    'Choose one of the following',
    ('Introduction', 'Contours Detection I: Snakes Method', 'Contours Detection II: Level Set Method Demo', 'Level Set Camera')
    )
    if selected_box == 'Introduction':
        introduction()
    if selected_box == 'Contours Detection I: Snakes Method':
        snake_method()
    if selected_box == 'Contours Detection II: Level Set Method Demo':
        level_set_demo()
    if selected_box == 'Level Set Camera':
        camera()

def level_set_demo():
    st.title('Contours Detection II: Level Set Method Demo')

    r'''Contours can be implicitly represented using level sets. Basically, level sets method intersects a plane on the surface and gives us a contour. Different from the limitation of Snakes method, level sets method has the advantages that No extra care is needed for topology change, like merging and splitting. Merging and splitting are handled naturally by the surface motion. In addition, by using the level set, we only calculate energy in a small neighborhood of the object contour.'''
    
    st.image('source/03_level_set_ex2.png', caption='Illustration In Lecture.')
    
    r'''In lecture, we have learned that the evolution equation for the embedding function is: $\newline$
    $$ F\|\nabla \phi\|+\phi_{t}=0 $$ $\newline$
    By solving the PDE, according to Agustinus Kristiadi's Blog, the update function is: $\newline$
    $$ \phi^{\prime}=\phi+\Delta t F\|\nabla \phi\|$$ $\newline$
    $F$ is intuitively a force that drive curve propagation. In other words, we could think of $F$ as a velocity field, i.e. $F$ is a vector field where at every point it tells us the direction and magnitude of movement of our surface $\phi$. $\newline$
    As $F$ is a velocity field and consider the Level Set PDE above, we want $F$ to be high at all region that are not the border of the object we want to segment, and low otherwise. Intuitively, we want the curve to propagate quickly in the background of the image, and we want the curve to slowly propagate or even stop the propagation at the border of the object.
    One way to do it is obviously derive our $F$ from edge detector. Simplest way to do edge detection is to take the gradients of the image: $\newline$
    $$
    g(I)=\frac{1}{1+\|\nabla I\|^{2}}
    $$
    '''

    uploaded_file = st.file_uploader("Choose a video file to play")
    if uploaded_file is not None:

        img = load_image(uploaded_file)
        r'''
        uploaded image:
        '''
        st.image(img, use_column_width=True)
    
        n_iter = st.slider('Change number of iteration',min_value = 50,max_value = 555)
        d3 = True

        ea = st.slider('Change number of Elevation angle', min_value = -90, max_value = 90)
        az = st.slider('Change number of Azimuth ', min_value = -90, max_value = 90)

        if st.button('Perform Level Set'):
            l = lsk2.levelSetSolver(1, 1, n_iter, d3)
            l.run(img)
            l.scan_contour()
            l.expand_contour()
            l.write_contour_to_image()

            r'''
            The result image $\phi > Threshold(0.5)$
            '''
            st.image(255*(l.phi > 0.5), use_column_width=True)

            r'''
            The velocity field F is shown below:
            '''
            st.image(l.F, use_column_width=True)
            r'''
            The image with contour shows below:
            '''

            st.image(l.img_with_contour, use_column_width=True)

            
                
            t = (l.phis > 0.5)
            x = []
            y = []
            z = []
            skip = 0
            for i in range(t.shape[0]):
                for j in range(t.shape[1]):
                    for k in range(t.shape[2]):
                        if t[i][j][k] == False:
                            x.append(j)
                            y.append(k)
                            z.append(i)
                            
            x = np.asarray(x)
            y = np.asarray(y)
            z = np.asarray(z)

            # fig = plt.figure()
            # ax = fig.add_subplot(111, projection='3d')
            # ax.view_init(elev=ea, azim=az)
            # ax.plot3D(x, y, -z, zdir='z')
            # st.pyplot(fig)

            cm = plt.get_cmap('jet')
            cNorm = Normalize(vmin=min(z), vmax=max(z))
            scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
            fig = plt.figure()
            ax = Axes3D(fig)
            ax.view_init(elev=ea, azim=az)
            ax.scatter(x, y, z, c=scalarMap.to_rgba(z), s=3)
            scalarMap.set_array(z)
            fig.colorbar(scalarMap)
            r'''
            Then the 3d visualization of updating contour is shown below.
            '''
            st.pyplot(fig, use_column_width=True)




def camera():
    st.title('Contours Detection II: Level Set Method Camera')
    captured_image = webcam()
    if captured_image is None:
        st.write("Waiting for capture...")
    else:
        st.write("Got an image from the webcam:")
        st.image(captured_image, use_column_width=True)
        # # print(type(captured_image))
        # with open('test.npy', 'wb') as f:
        #     np.save(f, np.array(captured_image))

    n_iter = st.slider('Change number of iteration',min_value = 50,max_value = 555)
    d3 = False

    if st.button('Perform Level Set'):
        if captured_image is None:
            st.write("Capture before perform level set")
        else:
            l=lsk2.levelSetSolver(1, 1, n_iter, d3)
            l.run(rgba2rgb(np.array(captured_image)))
            l.scan_contour()
            l.expand_contour()
            l.write_contour_to_image()
            st.image(l.img_with_contour, use_column_width=True)

            st.image(255*(l.phi > 0.5), use_column_width=True)


def find_contour_cv2(img):
    # convert to grayscale
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(img_gray, 150, 255, cv2.THRESH_BINARY)
    # find contours
    contours, hierarchy = cv2.findContours(
        image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
    # draw contours
    cv2.drawContours(image=image_copy, contours=contours, contourIdx=-1,
                     color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
    return img


def circle(center, radius, noise=None):
    """
    Generate a circle with given center and radius.
    """
    n = 200
    t = np.linspace(0, 2 * np.pi, n)
    x = center[0] + radius * np.cos(t)
    y = center[1] + radius * np.sin(t)
    if noise is not None:
        x += noise * np.random.randn(n)
        y += noise * np.random.randn(n)
    return x, y


def internal_energy_continuity(v):
    return np.sum(np.abs(np.diff(v)))


def internal_energy_curvature(v):
    return np.sum(np.abs(np.diff(v, 2)))


def introduction():
    st.title('Contours Detection')
    st.subheader('Edge Detection and Contours Detection?')
    st.image('source/sample_image.png', caption='Illustration by Stefano.')
    st.image('source/sample_image_edge_detection.png',
             caption='Illustration by Stefano.')
    st.image('source/sample_image_controus.png',
             caption='Illustration by Stefano.')

    st.subheader('How to impelement Contours Detection?')
    r'''
    We are going to implement 2 method to find the contours of an image. $\newline$
    The first method is the Snakes Method.$\newline$
    The second method is the Level Set method.
    '''
    # st.text('We are going to implement 2 method to find the contours of an image.')
    # st.text('The first method is the Snakes Method.')
    # st.text('The second method is the Level Set method.')


def snake_method_part1():
    st.subheader('Continuity and Curvature')
    st.text('The continuity of the contour is the distance between the endpoints of the curve.')
    st.latex(
        r'''\int_0^1E_{cont}(v(s))ds =  \int_0^1\left|\frac{dv}{ds}\right|^2ds ''')
    # draw circle with center (0,0) and radius 1, and calculate the internal_energy_continuity
    radius = st.slider('Radius:', 0.0, 2.0, 1.0, key='radius')
    x, y = circle(center=(0, 0), radius=radius)
    v = (x, y)
    fig, ax = plt.subplots()
    ax.plot(x, y, 'r-')
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_aspect('equal')
    st.pyplot(fig)

    st.text('The internal energy of continuity is: {:.2f}'.format(
        internal_energy_continuity(v)))
    st.text(
        'The curvature of the contour is the distance between the endpoints of the curve.')
    st.latex(
        r''' \int_0^1E_{curv}(v(s))ds =  \int_0^1\left|\frac{d^2v}{ds^2}\right|^2ds ''')

    noise = st.slider('Noise:', 0.0, 0.10, 0.0, step=0.01, key='noise')
    x, y = circle(center=(0, 0), radius=radius, noise=noise)
    v = (x, y)
    fig, ax = plt.subplots()
    ax.plot(x, y, 'r-')
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_aspect('equal')
    st.pyplot(fig)
    st.text('The internal energy of curvature is: {:.2f}'.format(
        internal_energy_curvature(v)))


def snake_method_part2():
    st.subheader('External Energy: Line Functional')
    st.markdown('The line functional attracts the snake to dark or bright lines, ' +
                'and writes simply as the image intensity, as:')
    st.latex(r'''E_{image}(v(s)) = I(v(s))''')
    st.subheader('External Energy: Edge Functional')
    st.markdown('The edge functional attracts the snake to the edges of the image, ' +
                'and writes simply as the image gradient, as:')
    st.latex(r'''E_{image}(v(s)) = - \int_0^1 |\nabla I(v(s))|ds''')


def snake_method_part3():
    st.subheader('Minimum Energy')
    st.markdown(
        'Minimum the ennergy is equivelent to get the contour of the image.')
    st.latex(r'''E_{snake}(v(s)) = E_{Internal}(v(s)) + E_{External}(v(s))''')
    st.latex(r'''E_{snake} = \alpha \int_0^1E_{cont}(v(s))ds + \beta \int_0^1 E_{curv}(v(s))ds + \gamma \int_0^1 E_{image}(v(s))ds''')
    st.markdown('by gradient descent,')
    st.latex(r'''v_i \leftarrow v_i - \mu \frac{\partial}{\partial v_i} \sum_i \left(
    \alpha |v_i - v_{i-1}|^2 + \beta |v_{i+1} -2 v_i + v_{i-1}|^2 + \gamma (I(v_i) + \nabla I(v_i))\right)''')
    st.markdown(
        'Where $\mu$ is the step size, others are the weights of each energy.')


def shake_sample_setup(img, center, radius):
    img = rgb2gray(img)

    smooth_img = gaussian(img, 2)
    s = np.linspace(0, 2*np.pi, 400)
    v_init = np.array([center[0] + radius*100*np.sin(s), center[1] + radius*100*np.cos(s)]).T

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(img, cmap=plt.cm.gray)
    ax.plot(v_init[:, 1], v_init[:, 0], '--r', lw=3)
    return fig, v_init


def shake_sample_train(img, v_init, alpha, beta, gamma):
    img = rgb2gray(img)
    smooth_img = gaussian(img, 2)
    snake = active_contour(
    smooth_img, v_init, alpha=alpha, beta=beta, gamma=gamma)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(img, cmap=plt.cm.gray)
    ax.plot(v_init[:, 1], v_init[:, 0], '--r', lw=3)
    ax.plot(snake[:, 1], snake[:, 0], '-b', lw=3)
    # x.set_xticks([]), ax.set_yticks([]), ax.axis([0, img.shape[1], img.shape[0], 0])
    return fig, snake

def shake_sample():
    # load image from sample_image.png
    st.subheader('Example:')
    img = plt.imread('source/sample_image.png')
    x = st.slider('x:', 0, img.shape[0], 215, key='x')
    y  = st.slider('y:', 0, img.shape[1], 385, key='y')
    center = (x, y)
    radius = st.slider('radius:', 0., 2., 1.5, key='radius_example')
    fig, v_init = shake_sample_setup(img, center, radius)
    st.pyplot(fig)

    alpha = st.slider('alpha:', 0.0, 1.0, 0.3, step=0.01, key='alpha')
    beta = st.slider('beta:', 10.0, 30.0, 20.0, key='beta')
    gamma = st.slider('gamma:', 0.001, 0.1, 0.001, step=0.001, key='gamma')

    fig, snake = shake_sample_train(img, v_init, alpha, beta, gamma)
    st.pyplot(fig)

def snake_method():
    st.title('Contours Detection I: Snakes Method')
    st.image('source/contour_sample.png',
             caption='Illustration In Lecture.', use_column_width=True)
    st.markdown('Intuitively, the Loss Function of contours should consist of two parts. '
                + 'First, we want the area of the closed curve to be as small as possible, '
                + 'and at the same time, we want the closed curve to be as close as possible to the edge of the target object.')
    snake_method_part1()
    snake_method_part2()
    snake_method_part3()
    shake_sample()

if __name__ == "__main__":
    main()
