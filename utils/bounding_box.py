from __future__ import division
import numpy as np


def convert_coordinates(tensor, start_index, conversion, border_pixels='half'):
    """
    Convert coordinate format.

    There are three supported coordinate format:
    1) (xmin, xmax, ymin, ymax) - the "minmax" format
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
        tensor_copy[..., idx + 3] = tensor[..., idx + 1] + tensor[..., idx + 3] / 2  # Set ymax
    elif conversion == 'centroids2minmax':
        tensor_copy[..., idx] = tensor[..., idx] - tensor[..., idx + 2] / 2  # Set xmin
        tensor_copy[..., idx + 1] = tensor[..., idx] + tensor[..., idx + 2] / 2  # Set xmax
        tensor_copy[..., idx + 2] = tensor[..., idx + 1] - tensor[..., idx + 3] / 2  # Set ymin
        tensor_copy[..., idx + 3] = tensor[..., idx + 1] + tensor[..., idx + 3] / 2  # Set ymax
    elif conversion == 'minmax2corners' or conversion == 'corners2minmax':
        tensor_copy[..., idx + 1] = tensor[..., idx + 2]
        tensor_copy[..., idx + 2] = tensor[..., idx + 1]

    else:
        raise ValueError('Unexpected conversion value.'
                         'Supported values are "minmax2centroids", "centroids2minmax", '
                         '"corners2centroids", "centroids2corners", "minmax2corners", '
                         '"corners2minmax".')
    return tensor_copy


def get_intersection_area(boxes1, boxes2, coords='corners', mode='outer_product', border_pixels='half'):
    """
    To get the intersection ares (for internal use, without the safety check).
    """
    m = boxes1.shape[0]
    n = boxes2.shape[0]

    # Set the correct coordinate indices for the respective formats.
    if coords == 'corners':
        xmin = 0
        ymin = 1
        xmax = 2
        ymax = 3
    elif coords == 'minmax':
        xmin = 0
        xmax = 1
        ymin = 2
        ymax = 3

    if border_pixels == 'half':
        d = 0
    elif border_pixels == 'include':
        d = 1
    elif border_pixels == 'exclude':
        d = -1

    if mode == 'outer_product':
        # For all possible box combinations, get the greater xmin and ymin values.
        # This is a tensor of shape (m,n,2).
        min_xy = np.maximum(np.tile(np.expand_dims(boxes1[:, [xmin, ymin]], axis=1), reps=(1, n, 1)),
                            np.tile(np.expand_dims(boxes2[:, [xmin, ymin]], axis=0), reps=(m, 1, 1)))
        # For all possible box combinations, get the smaller xmax and ymax values.
        # This is a tensor of shape (m,n,2).
        max_xy = np.minimum(np.tile(np.expand_dims(boxes1[:, [xmax, ymax]], axis=1), reps=(1, n, 1)),
                            np.tile(np.expand_dims(boxes2[:, [xmax, ymax]], axis=0), reps=(m, 1, 1)))

        # Compute the side length of the intersection rectangles.
        side_lengths = np.maximum(0, max_xy - min_xy + d)

        return side_lengths[:, :, 0] * side_lengths[:, :, 1]

    elif mode == 'element_wise':
        min_xy = np.maximum(boxes1[:, [xmin, ymin]], boxes2[:, [xmin, ymin]])
        max_xy = np.maximum(boxes1[:, [xmax, ymax]], boxes2[:, [xmax, ymax]])
        side_lengths = np.maximum(0, max_xy - min_xy + d)
        return side_lengths[:, 0] * side_lengths[:, 1]


def iou(boxes1, boxes2, coords='centroids', mode='outer_product', border_pixels='half'):
    '''
    Computes the intersection-over-union similarity (also known as Jaccard similarity)
    of two sets of axis-aligned 2D rectangular boxes.

    Let `boxes1` and `boxes2` contain `m` and `n` boxes, respectively.

    In 'outer_product' mode, returns an `(m,n)` matrix with the IoUs for all possible
    combinations of the boxes in `boxes1` and `boxes2`.

    In 'element-wise' mode, `m` and `n` must be broadcast-compatible. Refer to the explanation
    of the `mode` argument for details.

    Arguments:
        boxes1 (array): Either a 1D Numpy array of shape `(4, )` containing the coordinates for one box in the
            format specified by `coords` or a 2D Numpy array of shape `(m, 4)` containing the coordinates for `m` boxes.
            If `mode` is set to 'element_wise', the shape must be broadcast-compatible with `boxes2`.
        boxes2 (array): Either a 1D Numpy array of shape `(4, )` containing the coordinates for one box in the
            format specified by `coords` or a 2D Numpy array of shape `(n, 4)` containing the coordinates for `n` boxes.
            If `mode` is set to 'element_wise', the shape must be broadcast-compatible with `boxes1`.
        coords (str, optional): The coordinate format in the input arrays. Can be either 'centroids' for the format
            `(cx, cy, w, h)`, 'minmax' for the format `(xmin, xmax, ymin, ymax)`, or 'corners' for the format
            `(xmin, ymin, xmax, ymax)`.
        mode (str, optional): Can be one of 'outer_product' and 'element-wise'. In 'outer_product' mode, returns an
            `(m,n)` matrix with the IoU overlaps for all possible combinations of the `m` boxes in `boxes1` with the
            `n` boxes in `boxes2`. In 'element-wise' mode, returns a 1D array and the shapes of `boxes1` and `boxes2`
            must be boadcast-compatible. If both `boxes1` and `boxes2` have `m` boxes, then this returns an array of
            length `m` where the i-th position contains the IoU overlap of `boxes1[i]` with `boxes2[i]`.
        border_pixels (str, optional): How to treat the border pixels of the bounding boxes.
            Can be 'include', 'exclude', or 'half'. If 'include', the border pixels belong
            to the boxes. If 'exclude', the border pixels do not belong to the boxes.
            If 'half', then one of each of the two horizontal and vertical borders belong
            to the boxex, but not the other.

    Returns:
        A 1D or 2D Numpy array (refer to the `mode` argument for details) of dtype float containing values in [0,1],
        the Jaccard similarity of the boxes in `boxes1` and `boxes2`. 0 means there is no overlap between two given
        boxes, 1 means their coordinates are identical.
    '''

    # Make sure the boxes have the right shapes.
    if boxes1.ndim > 2:
        raise ValueError(f"boxes1 must have rank either 1 or 2, but has rank {boxes1.ndim}.")
    if boxes2.ndim > 2:
        raise ValueError(f"boxes2 must have rank either 1 or 2, but has rank {boxes2.ndim}.")

    if boxes1.ndim == 1:
        boxes1 = np.expand_dims(boxes1, axis=0)
    if boxes2.ndim == 1:
        boxes2 = np.expand_dims(boxes2, axis=0)

    if not (boxes1.shape[1] == boxes2.shape[1] == 4):
        raise ValueError("All boxes must consist of 4 coordinates, "
                         f"but the boxes in `boxes1` and `boxes2` have {boxes1.shape[1]} "
                         f"and {boxes2.shape[1]} coordinates, respectively.")
    if mode not in {'outer_product', 'element_wise'}:
        raise ValueError(f"`mode` must be one of 'outer_product' and 'element-wise', but got '{mode}'.")

    # Convert the coordinates if necessary.
    if coords == 'centroids':
        boxes1 = convert_coordinates(boxes1, start_index=0, conversion='centroids2corners')
        boxes2 = convert_coordinates(boxes2, start_index=0, conversion='centroids2corners')
        coords = 'corners'
    elif coords not in {'minmax', 'corners'}:
        raise ValueError("Unexpected value for `coords`. Supported values are 'minmax', 'corners' and 'centroids'.")

    # Compute the IoU.

    # Compute the intersection areas.

    intersection_areas = get_intersection_area(boxes1, boxes2, coords=coords, mode=mode)

    m = boxes1.shape[0]  # The number of boxes in `boxes1`
    n = boxes2.shape[0]  # The number of boxes in `boxes2`

    # Compute the union areas.

    # Set the correct coordinate indices for the respective formats.
    if coords == 'corners':
        xmin = 0
        ymin = 1
        xmax = 2
        ymax = 3
    elif coords == 'minmax':
        xmin = 0
        xmax = 1
        ymin = 2
        ymax = 3

    if border_pixels == 'half':
        d = 0
    elif border_pixels == 'include':
        d = 1
    elif border_pixels == 'exclude':
        d = -1

    if mode == 'outer_product':
        boxes1_areas = np.tile(
            np.expand_dims((boxes1[:, xmax] - boxes1[:, xmin] + d) * (boxes1[:, ymax] - boxes1[:, ymin] + d), axis=1),
            reps=(1, n))
        boxes2_areas = np.tile(
            np.expand_dims((boxes2[:, xmax] - boxes2[:, xmin] + d) * (boxes2[:, ymax] - boxes2[:, ymin] + d), axis=0),
            reps=(m, 1))
    elif mode == 'element_wise':
        boxes1_areas = (boxes1[:, xmax] - boxes1[:, xmin] + d) * (boxes1[:, ymax] - boxes1[:, ymin] + d)
        boxes2_areas = (boxes2[:, xmax] - boxes2[:, xmin] + d) * (boxes2[:, ymax] - boxes2[:, ymin] + d)

    union_areas = boxes1_areas + boxes2_areas - intersection_areas

    return intersection_areas / union_areas
