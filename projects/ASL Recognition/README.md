# Classifying ASL Hand Signs on Synthetic Backgrounds using Image Segmentation and a Transformer Encoder Model

Shawson was responsible for the image blending technique. He wrote a program that can blend an image of an ASL hand sign onto a synthetic background.

Brian was responsible for the image segmentation technique. He compared four different methods of extracting the segmentation of a hand sign from Shawson's synthetic images.

Tyler was repsonsible for the transformer encoder model. He realized a way to consolidate three ASL datasets, tokenize segmented images, and achieve 97.9% recognition (on average) for 10 ASL hand signs.

The "results" folder illustrates the softmax outputs for each of these 10 ASL hand signs on the images taken from each stage of the project. We show the model's recognition performance on (1) test images from the dataset, (2) images of hand signs screenshotted from google, (3) images of Shawson's hand before applying the blending algorithm, (4) images from the simple thresholding segmentation technique, (5) images from otsu's thresholding segmentation technique, and (6) images from the adaptive contouring segmentation technique.

The other folders contain the slides used in the presentation on Wednesday 6/5. 
