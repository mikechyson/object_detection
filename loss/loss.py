#!/usr/bin/env python3
"""
@project: object_detection
@file: loss
@author: mike
@time: 2021/3/8
 
@function:
"""
from __future__ import division
import tensorflow as tf


class SSDLoss:
    def __init__(self, neg_pos_ratio=3, n_neg_min=0, alpha=1.0):
        self.neg_pos_ratio = neg_pos_ratio
        self.n_neg_min = n_neg_min
        self.alpha = alpha

    def smooth_L1_loss(self, y_true, y_pred):
        """
        Compute the L1 loss.

        :param y_true: nD tensor containing the ground truth data.
            In this context, the expected tensor has shape (batch, #boxes, 4) and
            contains the ground truth bounding box coordinates, where the last dimension
            contains (xmin, xmax, ymin, ymax).
        :param y_pred: nD tensor containing the predicted data.
        :return: The smooth L1 loss, a nD-1 tensor. In this context, a 2D tensor of the shape
            (batch, n_boxes_total)
        """
        absolute_loss = tf.abs(y_pred - y_true)
        square_loss = 0.5 * (y_pred - y_true) ** 2
        l1_loss = tf.where(tf.less(absolute_loss, 1.0), square_loss, absolute_loss - 0.5)
        return tf.reduce_sum(l1_loss, axis=-1)

    def log_loss(self, y_true, y_pred):
        # Make sure that y_pred doesn't contain any zeros (which would break the log function)
        y_pred = tf.maximum(y_pred, 1e-15)
        log_loss = -tf.reduce_sum(y_true * tf.log(y_pred), axis=-1)
        return log_loss

    def compute_loss(self, y_true, y_pred):
        """
        Compute the loss of the SSD model prediction against the ground truth.

        :param y_true: A NumPy array of shape (batch, #boxes, #classes + 12),
            where #boxes is the total number of boxes that the model predicts per image.
            The last axis contains [classes one-hot encoded, 4 ground truth box coordinate offsets,
            8 arbitrary entries] in this order, including the background class. The last 8 entries
            of the last axis are not used by this function, they only exist so that `y_true` has the
            same shape as `y_pred` where the last four entries of the last axis contain the anchor box
            coordinates, which are needed during inference.
        :param y_pred: The model prediction. The shape is identical
            to that of `y_true`, i.e. `(batch_size, #boxes, #classes + 12)`.
            The last axis must contain entries in the format
            `[classes one-hot encoded, 4 predicted box coordinate offsets, 8 arbitrary entries]`.
        :return: A scalar, the total multitask loss for classification and localization.
        """
        self.neg_pos_ratio = tf.constant(self.neg_pos_ratio)
        self.n_neg_min = tf.constant(self.n_neg_min)
        self.alpha = tf.constant(self.alpha)

        batch_size = tf.shape(y_pred)[0]  # tf.int32
        n_boxes = tf.shape(y_pred)[1]  # tf.int32

        # Compute the losses for class and box predictions for every box

        # Output shape: (batch_size, n_boxes)
        classification_loss = tf.to_float(self.log_loss(y_true[:, :, : -12], y_pred[:, :, :-12]))
        # Output shape: (batch_size, n_boxes)
        localization_loss = tf.to_float(self.smooth_L1_loss(y_true[:, :, -12:-8], y_pred[:, :, -12:-8]))

        # Compute the classification loss for the positive and negative targets.

        # Create masks for the positive and negative ground truth classes.
        # Output shape: (batch_size, n_boxes)
        negatives = y_true[:, :, 0]
        # Output shape: (batch_size, n_boxes)
        positives = tf.to_float(tf.reduce_max(y_true[:, :, 1:-12], axis=-1))

        # Count the number of positive boxes (classes 1 to n) in y_true across the whole batch.
        n_positive = tf.reduce_sum(positives)

        # Mask all negative boxes and sum up the losses for the positive box per batch item.
        # Keras loss functions must output one scalar loss value per batch item, rather than just
        # one scalar for the entire batch, that's why we're not summing across all axes.
        pos_class_loss = tf.reduce_sum(classification_loss * positives, axis=-1)

        # Compute the classification loss for the negative default boxes (if there are any).

        # First, compute the classification loss for all negative boxes.
        neg_class_loss_all = classification_loss * negatives
        n_neg_losses = tf.count_nonzero(neg_class_loss_all, dtype=tf.int32)
        # What's the point of `n_neg_losses`?
        # For the next step, which will be to compute which negative boxes enter the classification
        # loss, we don't just want to know how many negative ground truth boxes there are, but for
        # how many of those there actually is a positive (i.e. non-zero) loss. This is necessary
        # because `tf.nn.top-k()` in the function below will pick the top k boxes with the highest
        # losses no matter what, even if it receives a vector where all losses are zero. In the
        # unlikely event that all negative classification losses ARE actually zero though, this
        # behavior might lead to `tf.nn.top-k()` returning the indices of positive boxes, leading
        # to an incorrect negative classification loss computation, and hence an incorrect overall
        # loss computation. We therefore need to make sure that `n_negative_keep`, which assumes
        # the role of the `k` argument in `tf.nn.top-k()`, is at most the number of negative boxes
        # for which there is a positive classification loss.

        # Compute the number of negative examples we want to account for in the loss.
        # We'll keep at most `self.neg_pos_ratio` times the number of positives in `y_true`,
        # but at least `self.n_neg_min` (unless `n_neg_loses` is smaller).
        n_negative_keep = tf.minimum(tf.maximum(self.neg_pos_ratio * tf.to_int32(n_positive), self.n_neg_min),
                                     n_neg_losses)

        # In the unlikely case when either (1) there are no negative ground truth boxes at all
        # or (2) the classification loss for all negative boxes is zero, return zero as the `neg_class_loss`.
        def f1():
            return tf.zeros([batch_size])

        # Otherwise, compute the negative loss
        def f2():
            # Now, we identify the top-k (where k == `n_negative_keep`) boxes with the highest confidence loss
            # that belong to background class in the ground truth data. Note that this doesn't necessarily mean
            # that the model predict the wrong class for those boxes, it just mean that the loss for those boxes
            # is the highest.

            # To do this, we reshape `neg_class_loss_all` to 1D
            # Shape: (batch_size * n_boxes,)
            neg_class_loss_all_1D = tf.reshape(neg_class_loss_all, [-1])
            # Then we get the indices for the `n_negative_keep` boxes with the highest loss
            values, indices = tf.nn.top_k(neg_class_loss_all_1D, k=n_negative_keep, sorted=False)
            # With this indices, we create a mask
            # Shape: (batch_size * n_boxes,)
            negatives_keep = tf.scatter_nd(indices=tf.expand_dims(indices, axis=1),
                                           updates=tf.ones_like(indices, dtype=tf.int32),
                                           shape=tf.shape(neg_class_loss_all_1D))
            # Shape: (batch_size, n_boxes)
            negatives_keep = tf.to_float(tf.reshape(negatives_keep, [batch_size, n_boxes]))
            # Ust it to keep only those boxes and mask all other classification losses
            # Shape: (batch,)
            neg_class_loss = tf.reduce_sum(classification_loss * negatives_keep, axis=-1)
            return neg_class_loss

        neg_class_loss = tf.cond(tf.equal(n_neg_losses, tf.constant(0)), f1, f2)

        class_loss = pos_class_loss + neg_class_loss

        # Compute the localization loss for the positive targets.
        # We don't compute a localization loss for negative predicted boxes.
        # (obviously: there are no ground truth boxes they would correspond to)
        loc_loss = tf.reduce_sum(localization_loss * positives, axis=-1)

        # Compute the total loss
        total_loss = (class_loss + self.alpha * loc_loss) / tf.maximum(1.0, n_positive)  # In case `n_positive == 0`
        # Keras divides the loss by the batch size.
        # So in order to revert Keras' averaging over the batch size, we'll have to multiply by it.
        total_loss = total_loss * tf.to_float(batch_size)

        return total_loss
