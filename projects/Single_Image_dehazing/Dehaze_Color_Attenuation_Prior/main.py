import cv2
import numpy as np
import scipy
import scipy.ndimage
import skimage
from GuidedFilter import GuidedFilter
from skimage.future import graph
from scipy import ndimage
import os
from skimage.metrics import structural_similarity as ssim

# segmentation
def _weight_mean_color(graph, src, dst, n):

    diff = graph.nodes[dst]['mean color'] - graph.nodes[n]['mean color']
    diff = np.linalg.norm(diff)
    return {'weight': diff}

def merge_mean_color(graph, src, dst):

    graph.nodes[dst]['total color'] += graph.nodes[src]['total color']
    graph.nodes[dst]['pixel count'] += graph.nodes[src]['pixel count']
    graph.nodes[dst]['mean color'] = (graph.nodes[dst]['total color'] /
                                      graph.nodes[dst]['pixel count'])
def segmentation_merge(img):
    labels = skimage.segmentation.slic(img, compactness=30, n_segments=600, start_label=1)
    g = graph.rag_mean_color(img, labels)
    labels1 = graph.merge_hierarchical(labels, g, thresh=15, rag_copy=False, in_place_merge=True, merge_func=merge_mean_color, weight_func=_weight_mean_color)
    return labels1

def calDepthMap_nei(I, r):
    # Calculate the depth of picture (neighborhood)

    hsvI = cv2.cvtColor(I, cv2.COLOR_BGR2HSV)
    s = hsvI[:, :, 1] / 255.0  # saturation component
    v = hsvI[:, :, 2] / 255.0  # brightness component

    sigma = 0.041337
    sigmaMat = np.random.normal(0, sigma, (I.shape[0], I.shape[1]))  # random error of the model

    output = 0.121779 + 0.959710 * v - 0.780245 * s + sigmaMat  # eq.8
    outputPixel = output
    output = scipy.ndimage.minimum_filter(output, (r, r))  # eq.21
    outputRegion = output

    return outputRegion, outputPixel

def calDepthMap_seg(I):
    # Calculate the depth of picture (segmentation)

    labels = segmentation_merge(I*255)
    hsvI = cv2.cvtColor(I, cv2.COLOR_BGR2HSV)
    s = hsvI[:, :, 1] / 255.0  # saturation component
    v = hsvI[:, :, 2] / 255.0  # brightness component

    sigma = 0.041337
    sigmaMat = np.random.normal(0, sigma, (I.shape[0], I.shape[1]))  # random error of the model

    output = 0.121779 + 0.959710 * v - 0.780245 * s + sigmaMat  # eq.8
    outputPixel = output
    # eq.21
    minimum = ndimage.minimum(output, labels=labels, index=np.arange(np.min(labels), np.max(labels)+1))
    for i in range(output.shape[0]-1):
        for j in range(output.shape[1]-1):
            label_pixel = labels[i, j]
            output[i, j] = minimum[label_pixel]
    outputRegion = output

    return outputRegion, outputPixel

def calfinalimage(I, t, a):
    # Calculate eq.23
    if I.dtype == np.uint8:
        I = np.float32(I) / 255

    h, w, c = I.shape
    J = np.zeros((h, w, c), dtype=np.float32)
    t0, t1 = 0.05, 1
    t = t.clip(t0, t1)  # Let t(x) between 0.05 to 1

    for i in range(3):
        J[:, :, i] = I[:, :, i] - a[0, i]
        J[:, :, i] = J[:, :, i] / t
        J[:, :, i] = J[:, :, i] + a[0, i]

    return J

def estA(I, Jdark):
    # estimate atmospheric light A

    h, w, c = I.shape
    if I.dtype == np.uint8:
        I = np.float32(I) / 255

    # Compute number for 0.1% brightest pixels
    n_bright = int(np.ceil(0.001 * h * w))
    #  Loc contains the location of the sorted pixels
    reshaped_Jdark = Jdark.reshape(1, -1)
    Loc = np.argsort(reshaped_Jdark)

    # column-stacked version of I
    Ics = I.reshape(1, h * w, 3)
    ix = I.copy()

    # init matrix to store candidate airlight pixels
    Acand = np.zeros((1, n_bright, 3), dtype=np.float32)
    # init matrix to store largest norm arilight
    Amag = np.zeros((1, n_bright, 1), dtype=np.float32)

    # Compute magnitudes of RGB vectors of A
    for i in list(range(n_bright)):
        x = Loc[0, h * w - 1 - i]
        ix[int(x / w), int(x % w), 0] = 0
        ix[int(x / w), int(x % w), 1] = 0
        ix[int(x / w), int(x % w), 2] = 1

        Acand[0, i, :] = Ics[0, Loc[0, h * w - 1 - i], :]
        Amag[0, i] = np.linalg.norm(Acand[0, i, :])

    # Sort A magnitudes
    reshaped_Amag = Amag.reshape(1, -1)
    Y2 = np.sort(reshaped_Amag)
    Loc2 = np.argsort(reshaped_Amag)
    # A now stores the best estimate of the airlight
    if len(Y2) > 20:
        A = Acand[0, Loc2[0, n_bright - 19:n_bright], :]
    else:
        A = Acand[0, Loc2[0, n_bright - len(Y2):n_bright], :]

    return A



if __name__ == "__main__":

    # the setting of parameters
    r = 15  # from eq.21
    beta = 1  # from eq.23
    gimfiltR = 60  # the radius parameters for guided image filtering (Figure 8(d))
    eps = 10 ** -3  # the epsilon parameters for guided image filtering (Figure 8(d))
    MSE = np.empty((3, 0), float)
    SSIM = np.empty((3, 0), float)


    # Indoor
    haze_path = "SOTS/indoor/hazy/"
    GT_path = "SOTS/indoor/gt/"
    output_path = "SOTS/indoor/output/"
    list_img = os.listdir(haze_path)
    list_img = [s for s in list_img if "_10" in s]
    haze_dir = sorted(list_img)

    # Outdoor
    # haze_path = "SOTS/outdoor/hazy/"
    # GT_path = "SOTS/outdoor/gt/"
    # output_path = "SOTS/outdoor/output/"

    # Synthetic
    # haze_path = "SOTS/synthetic/hazy/"
    # GT_path = "SOTS/synthetic/gt/"
    # output_path = "SOTS/synthetic/output/"


    # haze_dir = sorted(os.listdir(haze_path))
    GT_dir = sorted(os.listdir(GT_path))

    num_files = 50  # Number of images to run


    for i in range(num_files):


        input_image = haze_dir[i]
        inputImagePath = haze_path + input_image

        GT_image = GT_dir[i]
        GTImagePath = GT_path + GT_image

        I = cv2.imread(inputImagePath)
        GT = cv2.imread(GTImagePath)

        # Calculate d(x)
        dR_nei, dP_nei = calDepthMap_nei(I, r)
        dR_seg, dP_seg = calDepthMap_seg(I)

        # use guided image filtering to smooth image
        guided_filter = GuidedFilter(I, gimfiltR, eps)
        refineDR_nei = guided_filter.filter(dR_nei)
        refineDR_seg = guided_filter.filter(dR_seg)


        tR_nei = np.exp(-beta * refineDR_nei)
        tP_nei = np.exp(-beta * dP_nei)
        tR_seg = np.exp(-beta * refineDR_seg)
        tP_seg = np.exp(-beta * dP_seg)


        A_nei = estA(I, dR_nei)
        A_seg = estA(I, dR_seg)

        J_seg = calfinalimage(I, tR_seg, A_seg)
        J_nei = calfinalimage(I, tR_nei, A_nei)
        cv2.imwrite(output_path + "neighborhood/" + GT_image, J_nei * 255)
        cv2.imwrite(output_path + "segmentation/" + GT_image, J_seg * 255)



        nei = cv2.imread(output_path + "neighborhood/" + GT_image)
        seg = cv2.imread(output_path + "segmentation/" + GT_image)

        # Calculate MSE
        mse = np.zeros((3, 1))
        mse[0] = np.sqrt(np.mean(np.square(np.subtract(I, GT))))
        mse[1] = np.sqrt(np.mean(np.square(np.subtract(nei, GT))))
        mse[2] = np.sqrt(np.mean(np.square(np.subtract(seg, GT))))
        print('mean square error on haze image ', mse[0])
        print('mean square error on neighborhood dehaze image ', mse[1])
        print('mean square error on segmentation dehaze image ', mse[2])
        MSE = np.append(MSE, mse, axis=1)


        # convert BGR format to gray format
        gray_ori = cv2.cvtColor(nei, cv2.COLOR_BGR2GRAY)
        gray_seg = cv2.cvtColor(seg, cv2.COLOR_BGR2GRAY)
        gray_I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
        gray_GT = cv2.cvtColor(GT, cv2.COLOR_BGR2GRAY)

        # Calculate SSIM
        ssim_gray = np.zeros((3, 1))
        ssim_gray[0] = ssim(gray_I, gray_GT)
        ssim_gray[1] = ssim(gray_ori, gray_GT)
        ssim_gray[2] = ssim(gray_seg, gray_GT)
        print('SSIM on haze image ', ssim_gray[0])
        print('SSIM on neighborhood dehaze image ', ssim_gray[1])
        print('SSIM on segmentation dehaze image ', ssim_gray[2])
        SSIM = np.append(SSIM,  ssim_gray, axis=1)



    # Calculate average of MSE and SSIM
    mse_average = np.zeros((3, 1))
    for i in range(3):
        mse_average[i] = np.average(MSE[i,:])
    MSE = np.append(MSE, mse_average, axis=1)
    ssim_average = np.zeros((3, 1))
    for i in range(3):
        ssim_average[i] = np.average(SSIM[i,:])
    SSIM = np.append(SSIM, ssim_average, axis=1)

    print('Average mean square error on haze image ', mse_average[0])
    print('Average mean square error on neighborhood dehaze image ', mse_average[1])
    print('Average mean square error on segmentation dehaze image ', mse_average[2])

    print('Average SSIM on haze image ', ssim_average[0])
    print('Average SSIM on neighborhood dehaze image ', ssim_average[1])
    print('Average SSIM on segmentation dehaze image ', ssim_average[2])

    # Save to csv file
    np.savetxt(output_path + "MSE.csv", MSE, delimiter=',', fmt='%f')
    np.savetxt(output_path + "SSIM.csv", SSIM, delimiter=',', fmt='%f')

