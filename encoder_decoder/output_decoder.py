from __future__ import division

import numpy as np
from utils.bounding_box import iou, convert_coordinates


def decode_detections(y_pred,
                      confidence_thresh=0.01,
                      iou_threshold=0.45,
                      top_k=200,
                      input_coords='centroids',
                      normalize_coords=True,
                      img_height=None,
                      img_width=None,
                      border_pixels='half'):
    """
    Convert model prediction output back to a format that contains only the positive
    box predictions (i.e., the same format that `SSDInputEncoder` takes as input).

    After the decoding, two steps of prediction filtering are performed for each loss
    individually: First confidence thresholding, then greedy non-maximum suppression.
    The filtering results for all classes are concatenated and the `top_k` overall
    highest confidence results constitute the final predictions for a given batch item.

    Arguments:
        y_pred: The prediction output of the SSD model, expected to be a NumPY array of
            shape `(batch_size, #boxes, #classes + 4 + 4 + 4)`, where `#boxes` is the
            total number of boxes predicted by the model per image and the last axis
            contains `[one-hot vector for the classes, 4 predicted coordinate offsets,
            4 anchor box coordinate, 4 variances]`.
        confidence_thresh: A float in [0, 1), the minimum classification  confidence in
            a specific positive class in order to be considered for the non-maximum
            suppression stage for the respective class. A lower value will result in a
             larger part of the selection process being done by the non-maximum
             suppression stage, while a larger value will result in a larger part of
             the selection process happening in the confidence thresholding stage.
        iou_threshold: A float in [0, 1]. All boxes with a IoU similarity of greater
            than `iou_threshold` with a locally maximal box will be removed from the
            set of predictions for a given class, where `maximal` refers to the box
            score.
        top_k: The number of highest scoring predictions to be kept for each batch item
            after the non-maximum suppression stage.
        input_coords: The box coordinate format that the model outputs. Can be either
            'centroids' for the format `(cx, cy, w, h)`, 'minmax' for the format
            `(xmin, xmax, ymin, ymax)`, or 'corners' for the format
            `(xmin, ymin, xmax, ymax)`.
        normalize_coords: Set to `True` if the model outputs relative coordinates
            (i.e., coordinates in [0,1]) and you wish to transform these relative
            coordinates back to absolute coordinates. If the model outputs relative
            coordinates, but you do not want to convert them back to absolute coordinates,
            set this to `False`. Do not set this is `True` if the model already outputs
            absolute coordinates, as that would result in incorrect coordinates. Requires
            `img_height` and `img_width` if set to `True`.
        img_height: The height of the input image. Only needed if `normalized_coords` is `True`.
        img_width: The width of the input image. Only needed if `normalized_coords` is `True`.
        border_pixels: How to treat the border pixels of the bounding boxes. Can be 'include',
            'exclude' and 'half'. If 'include', the border pixels belong to the boxes.
            If 'exclude', the border pixels do not belong to the boxes. If 'half', then one
            of each of the two horizontal and vertical borders belong to the boxes, but not
            the other.

    Returns:
        A python list of length `batch_size` where each list element represent the predicted
        boxes for one image and contains NumPy array of shape `(boxes, 6)` where each row is
        a box prediction for a non-background class for the respective image in the format
        `[class_id, confidence, xmin, ymin, xmax, ymax]`.
    """
    if normalize_coords and (img_height is None or img_width is None):
        raise ValueError('If relative box coordinates are supposed to be converted to absolute'
                         'coordinates, the decoder needs the image size in order to decode the '
                         f'prediction, but `img_height` == {img_height} and `img_width` == '
                         f'{img_width}')

    # Slice out the classes and the four offsets, throw away the anchor coordinates and variance,
    # resulting in a tensor of shape `(batch, n_boxes, n_classes + 4 coordinates]`.
    y_pred_decoded_raw = np.copy(y_pred[:, :, :-8])

    if input_coords == 'centroids':
        # exp(ln(w(pred)/w(anchor)) / w_variance * w_variance) == w(pred) / w(anchor)
        # exp(ln(h(pred)/h(anchor)) / h_variance * h_variance) == h(pred) / h(anchor)
        y_pred_decoded_raw[:, :, [-2, -1]] = np.exp(y_pred_decoded_raw[:, :, [-2, -1]] * y_pred[:, :, [-2, -1]])
        # (w(pred) / w(anchor)) * w(anchor) == w(pred)
        # (h(pred) / h(anchor)) * h(anchor) == h(pred)
        y_pred_decoded_raw[:, :, [-2, -1]] *= y_pred[:, :, [-6, -5]]
        # (delta_cx(pred) / w(anchor) / cx_variance) * cx_variance * w(anchor) == delta_cx(pred)
        # (delta_cy(pred) / h(anchor) / cy_variance) * cy_variance * h(anchor) == delta_cy(pred)
        y_pred_decoded_raw[:, :, [-4, -3]] *= y_pred[:, :, [-4, -3]] * y_pred[:, :, [-6, -5]]
        # delta_cx(pred) + cx(anchor) == cx(pred)
        # delta_cy(pred) + cy(anchor) == cy(pred)
        y_pred_decoded_raw[:, :, [-4, -3]] += y_pred[:, :, [-8, -7]]
        y_pred_decoded_raw = convert_coordinates(y_pred_decoded_raw, start_index=-4, conversion='centroids2corners')
    elif input_coords == 'minmax':
        # delta(pred) / size(anchor) / variance * variance == delta(pred) / size(anchor)
        # for all four coordinates, where 'size' refers to w or h, respectively
        y_pred_decoded_raw[:, :, -4:] *= y_pred[:, :, -4:]
        # delta_xmin(pred) / w(anchor) * w(anchor) == delta_xmin(pred)
        # delta_xmax(pred) / w(anchor) * w(anchor) == delta_xmax(pred)
        y_pred_decoded_raw[:, :, [-4, -3]] *= np.expand_dims(y_pred[:, :, -7] - y_pred[:, :, -8], axis=-1)
        # delta_ymin(pred) / h(anchor) * h(anchor) == delta_ymin(pred)
        # delta_ymax(pred) / h(anchor) * h(anchor) == delta_ymax(pred)
        y_pred_decoded_raw[:, :, [-2, -1]] *= np.expand_dims(y_pred[:, :, -5] - y_pred[:, :, -6], axis=-1)
        # delta(pred) + anchor == pred for all four coordinates
        y_pred_decoded_raw[:, :, -4:] += y_pred[:, :, -8:-4]
        y_pred_decoded_raw = convert_coordinates(y_pred_decoded_raw, start_index=-4, conversion='minmax2corners')
    elif input_coords == 'corners':
        # delta(pred) / size(anchor) / variance * variance == delta(pred) / size(anchor)
        # for all four coordinates, where 'size' refers to w or h, respectively
        y_pred_decoded_raw[:, :, -4:] *= y_pred[:, :, -4:]
        # delta_xmin(pred) / w(anchor) * w(anchor) == delta_xmin(pred)
        # delta_xmax(pred) / w(anchor) * w(anchor) == delta_xmax(pred)
        y_pred_decoded_raw[:, :, [-4, -2]] *= np.expand_dims(y_pred[:, :, -6] - y_pred[:, :, -8], axis=-1)
        # delta_ymin(pred) / h(anchor) * h(anchor) == delta_ymin(pred)
        # delta_ymax(pred) / h(anchor) * h(anchor) == delta_ymax(pred)
        y_pred_decoded_raw[:, :, [-3, -1]] *= np.expand_dims(y_pred[:, :, -5] - y_pred[:, :, -7], axis=-1)
        # delta(pred) + anchor == pred for all four coordinates
        y_pred_decoded_raw[:, :, -4:] += y_pred[:, :, -8:-4]
    else:
        raise ValueError(
            "Unexpected value for `input_coords`. "
            "Supported input coordinate formats are 'minmax', 'corners' and 'centroids'.")

    if normalize_coords:
        y_pred_decoded_raw[:, :, [-4, -2]] *= img_width
        y_pred_decoded_raw[:, :, [-3, -1]] *= img_height

    # Apply confidence thresholding and non-maximum suppression per class.
    n_classes = y_pred_decoded_raw.shape[-1] - 4

    y_pred_decode = []  # Store the final predictions in this list.
    for batch_item in y_pred_decoded_raw:  # Batch item has the shape `[n_boxes, n_classes + 4 coords]`.
        pred = []  # Store the final predictions for this batch item.
        for class_id in range(1, n_classes):  # For each class except the background class.
            # Keep only the confidences for that class,
            # making this array of shape `[n_boxes, confidence + 4 coords]`
            single_class = batch_item[:, [class_id, -4, -3, -2, -1]]
            # Keep only those boxes with a confidence above the threshold.
            threshold_met = single_class[single_class[:, 0] > confidence_thresh]

            # If any boxes pass the threshold.
            if threshold_met.shape[0] > 0:
                # Perform non-maximum suppression.
                maxima = _greedy_nms(threshold_met, iou_threshold=iou_threshold,
                                     coords='corners',
                                     border_pixels=border_pixels)
                # Expand the last dimension by one element to have room for the class ID.
                # This is now an array of shape `[n_boxes, confidence + 4 coords + class_id]`.
                maxima_output = np.zeros((maxima.shape[0], maxima.shape[1] + 1))
                # Write the class ID to the first column
                maxima_output[:, 0] = class_id
                # Write the maxima to the other columns
                maxima_output[:, 1:] = maxima
                pred.append(maxima_output)

        # If there are any predictions left.
        if pred:
            pred = np.concatenate(pred, axis=0)
            # If we have more than `top_k` results left at this point,
            # Otherwise there is nothing to filter.
            if top_k != 'all' and pred.shape[0] > top_k:
                # Get the indices of the `top_k` highest score maximum.
                top_k_indices = np.argpartition(pred[:, 1], kth=pred.shape[0] - top_k, axis=0)[pred.shape[0] - top_k:]
                pred = pred[top_k_indices]
            else:
                pred = np.array(pred)
            y_pred_decode.append(pred)
        return y_pred_decode


def _greedy_nms(predictions, iou_threshold=0.45, coords='corners', border_pixels='half'):
    """
    Perform greedy non-maximum suppression on the input boxes.

    Greedy NMS works by selecting the box with the highest score and removing all boxes
    around it that are close to it measured by IoU similarity. Out of the boxes that are
    left over, once again the one with the highest score is selected and so on, until
    no boxes with to much overlap are left.
    """
    boxes_left = np.copy(predictions)
    # Store the boxes that make it through the non-maximum suppression.
    maxima = []
    while boxes_left.shape[0] > 0:
        # Get the index of the box with the highest confidence.
        maximum_index = np.argmax(boxes_left[:, 0])
        maximum_box = np.copy(boxes_left[maximum_index])
        maxima.append(maximum_box)
        boxes_left = np.delete(boxes_left, maximum_index, axis=0)
        if boxes_left.shape[0] == 0:
            break
        # Compare (IoU) the other left over boxes to the maximum box.
        similarities = iou(boxes_left[:, 1:],
                           maximum_box[1:],
                           coords=coords,
                           mode='element_wise',
                           border_pixels=border_pixels)
        # Remove the ones that overlap too much with the maximum box.
        boxes_left = boxes_left[similarities <= iou_threshold]
    return np.array(maxima)
