#!/usr/bin/env python3
"""
@project: object_detection
@file: bounding_box
@time: 2021/3/4
 
@function:
"""
import numpy as np


def convert_coordinates(tensor, start_index, conversion, border_pixels='half'):
    """
    Convert coordinate format.

    There are three supported coordinate format:
    1) (xmin, xman, ymin, ymax) - the "minmax" format
    2) (xmin, ymin, xmax, ymax) - the "corners" format
    3) (cx, cy, w, h) - the "centroids" format

    :param tensor: A NumPy nD array containing the four consecutive coordinates
        to be converted in the last axis.
    :param start_index: The index of the first coordinate in the last axis of tensor.
    :param conversion: Can be minmax2centroids, centroids2minmax, corners2centroids,
        centroids2corners, minmax2corners, corners2minmax.
    :param border_pixels: How to treat the border pixels of the bounding box.
        Can be "include", "exclude", or "half".
    :return:
    """
    if border_pixels == 'half':
        d = 0
    elif border_pixels == 'include':
        d = 1
    elif border_pixels == 'exclude':
        d = -1

    idx = start_index
    tensor_copy = np.copy(tensor).astype(np.float)
    if conversion == 'minmax2centroids':
        tensor_copy[..., idx] = (tensor[..., idx] + tensor[..., idx + 1]) / 2  # Set cx
        tensor_copy[..., idx + 1] = (tensor[..., idx + 2] + tensor[..., idx + 3]) / 2  # Set cy
        tensor_copy[..., idx + 2] = (tensor[..., idx + 1] - tensor[..., idx]) + d  # Set w
        tensor_copy[..., idx + 3] = (tensor[..., idx + 3] - tensor[..., idx + 2]) + d  # Set h
    elif conversion == 'corners2centroids':
        tensor_copy[..., idx] = (tensor[..., idx] + tensor[..., idx + 2]) / 2  # Set cx
        tensor_copy[..., idx + 1] = (tensor[..., idx + 1] + tensor[..., idx + 3]) / 2  # Set cy
        tensor_copy[..., idx + 2] = (tensor[..., idx + 2] - tensor[..., idx]) + d  # Set w
        tensor_copy[..., idx + 3] = (tensor[..., idx + 3] - tensor[..., idx + 1]) + d  # Set h
    elif conversion == 'centroids2corners':
        tensor_copy[..., idx] = tensor[..., idx] - tensor[..., idx + 2] / 2  # Set xmin
        tensor_copy[..., idx + 1] = tensor[..., idx + 1] - tensor[..., idx + 3] / 2  # Set ymin
        tensor_copy[..., idx + 2] = tensor[..., idx] + tensor[..., idx + 2] / 2  # Set xmax
        tensor_copy[..., idx + 3] = tensor[..., idx + 1] + tensor[..., idx + 3] / 2

    else:
        raise ValueError('Unexpected conversion value.'
                         'Supported values are "minmax2centroids", "centroids2minmax", '
                         '"corners2centroids", "centroids2corners", "minmax2corners", '
                         '"corners2minmax".')
    return tensor_copy


def iou(boxes1, boxes2, coords='centroids', mode='outer_product', border_pixels='half'):
    pass
