# Author: Brycen

import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

def gaussian_pyr(img, levels):
    output = np.zeros((img.shape[0], img.shape[1] + int(img.shape[1]/2), img.shape[2]))
    output[0:img.shape[0], 0:img.shape[1], :] = img
    ch, cw = img.shape[0:2]
    row_index = 0
    prev_level = img
    for i in range(levels):
        cw = int(cw / 2)
        ch = int(ch / 2)
        decimated = blur(prev_level)[::2, ::2, :]
        output[row_index:(row_index+ch), img.shape[1]:(img.shape[1]+cw), :] = decimated
        prev_level = decimated
        row_index += ch

    return output

def laplacian_pyr(img, levels):
    output = np.zeros((img.shape[0], img.shape[1] + int(img.shape[1]/2), img.shape[2]), dtype=np.float32)
    ch, cw = img.shape[0:2]
    row_index = 0
    prev_level = img
    for i in range(levels):
        decimated = blur(prev_level)[::2, ::2, :]
        reconstructed = cv2.resize(decimated, (ch,cw), interpolation=cv2.INTER_CUBIC)
        # reconstructed = bicubic_rescale(decimated, 2)
        if i > 0:
            output[row_index:(row_index+ch), img.shape[1]:(img.shape[1]+cw), :] = prev_level - reconstructed
            row_index += ch
        else:
            output[row_index:(row_index+ch), 0:(0+cw), :] = prev_level - reconstructed
        cw = int(cw / 2)
        ch = int(ch / 2)
        prev_level = decimated

    output[row_index:(row_index+ch), img.shape[1]:(img.shape[1]+cw), :] = decimated

    return output

# bluring with binomial kernel
def blur(img):
    binomial_kernel = np.array([0.0625, 0.25, 0.375, 0.25, 0.0625])
    partial = cv2.filter2D(img, -1, binomial_kernel)
    return cv2.filter2D(partial, -1, binomial_kernel.reshape((5, 1)))


def laplacian_reconstruct(pyr, level):
    size = pyr.shape[0]
    order = int(math.log2(size))
    start_ind = (2**order) - (2**(order - level + 1))
    curr_size = int(size / (2**level))
    base = pyr[start_ind:start_ind+curr_size, size:size+curr_size, :]
    for i in range(level):
        curr_size *= 2
        start_ind = start_ind - curr_size
        base = cv2.resize(base, (curr_size, curr_size), interpolation=cv2.INTER_CUBIC)
        if i<level-1:
            base = base + pyr[start_ind:start_ind+curr_size, size:size+curr_size, :]
        else:
            base = base + pyr[0:curr_size, 0:curr_size, :]

    return base

def laplacian_display(pyr, level):
    size = pyr.shape[0]
    order = int(math.log2(size))
    start_ind = (2**order) - (2**(order - level + 1))
    curr_size = int(size / (2**level))
    output = pyr + 0.5
    output[start_ind:start_ind+curr_size, size:size+curr_size, :] -= 0.5
    return output



