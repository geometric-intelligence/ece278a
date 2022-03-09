'''-------------------------------------------
Created By:
Roger Lin

Segment images
--------------------------------------------'''
import numpy as np
from skimage import img_as_float
from skimage.segmentation import *
from skimage.color import label2rgb
from skimage.future import graph

# def img_seg(image, segment_number_aprx, weight_on_prox):
#     segments = slic(image, compactness=weight_on_prox, n_segments=segment_number_aprx, start_label=1)
#     g = graph.rag_mean_color(image, segments)
#
#     labels = graph.merge_hierarchical(segments, g, thresh=35, rag_copy=False,
#                                       in_place_merge=True,
#                                       merge_func=merge_mean_color,
#                                       weight_func=_weight_mean_color)
#
#     segmented_img = label2rgb(labels, image, kind='avg', bg_label=0)
#     segmented_img = mark_boundaries(segmented_img, labels, (0, 0, 0))
#     return segmented_img

from skimage import data, segmentation, filters, color
from skimage.future import graph
from matplotlib import pyplot as plt


def weight_boundary(graph, src, dst, n):
    """
    Handle merging of nodes of a region boundary region adjacency graph.

    This function computes the `"weight"` and the count `"count"`
    attributes of the edge between `n` and the node formed after
    merging `src` and `dst`.


    Parameters
    ----------
    graph : RAG
        The graph under consideration.
    src, dst : int
        The vertices in `graph` to be merged.
    n : int
        A neighbor of `src` or `dst` or both.

    Returns
    -------
    data : dict
        A dictionary with the "weight" and "count" attributes to be
        assigned for the merged node.

    """
    default = {'weight': 0.0, 'count': 0}

    count_src = graph[src].get(n, default)['count']
    count_dst = graph[dst].get(n, default)['count']

    weight_src = graph[src].get(n, default)['weight']
    weight_dst = graph[dst].get(n, default)['weight']

    count = count_src + count_dst
    return {
        'count': count,
        'weight': (count_src * weight_src + count_dst * weight_dst) / count
    }


def merge_boundary(graph, src, dst):
    """Call back called before merging 2 nodes.

    In this case we don't need to do any computation here.
    """
    pass

# felzenszwalb
# def img_seg(img, thres):
#     edges = filters.sobel(color.rgb2gray(img))
#     labels = segmentation.felzenszwalb(img, scale=50, sigma=1, min_size=200)
#
#     g = graph.rag_boundary(labels, edges)
#
#     labels2 = graph.merge_hierarchical(labels, g, thresh=thres, rag_copy=False,
#                                        in_place_merge=True,
#                                        merge_func=merge_boundary,
#                                        weight_func=weight_boundary)
#
#     out = color.label2rgb(labels2, img, kind='avg', bg_label=0)
#     segmented_img = mark_boundaries(out, labels2, (0, 0, 0))
#     return segmented_img

def img_seg(img, thres):
    edges = filters.sobel(color.rgb2gray(img))
    labels = segmentation.quickshift(img, )

    g = graph.rag_boundary(labels, edges)

    labels2 = graph.merge_hierarchical(labels, g, thresh=thres, rag_copy=False,
                                       in_place_merge=True,
                                       merge_func=merge_boundary,
                                       weight_func=weight_boundary)

    out = color.label2rgb(labels2, img, kind='avg', bg_label=0)
    segmented_img = mark_boundaries(out, labels2, (0, 0, 0))
    return segmented_img