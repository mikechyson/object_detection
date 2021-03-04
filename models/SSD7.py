#!/usr/bin/env python3
"""
@project: object_detection
@file: SSD7
@author: mike
@time: 2021/3/3
 
@function:
"""
import numpy as np
from keras import backend as K
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, BatchNormalization, ELU, Reshape, Concatenate, Activation
from keras.models import Model
from keras.regularizers import l2

from layers.AnchorBoxes import AnchorBoxes


def SSD7(image_size,
         n_classes,
         mode='training',
         l2_reg=0.0,
         min_scale=0.1,
         max_scale=0.9,
         scales=None,
         aspect_ratios_global=[0.5, 1.0, 2.0],
         aspect_ratios_per_layer=None,
         two_boxes_for_ar1=True,
         steps=None,
         offsets=None,
         clip_boxes=False,
         variances=[1.0, 1.0, 1.0, 1.0],
         coords='centroids',
         normalize_coords=False,
         subtract_mean=None,
         divide_by_stddev=None,
         swap_channels=False,
         confidence_thresh=0.01,
         iou_threshold=0.45,
         top_k=200,
         nms_max_output_size=400,
         return_predictor_sizes=False
         ):
    """
    Build a SSD model with Keras.

    The model consists of convolutional feature layers and
    a number of convolutional predictor layers that take their input
    from different feature layers. The model is fully convolutional.
    This implementation has 7 convolutional layers and 4 convolutional predictor
    layers that take their input from layers 4, 5, 6, and 7, respectively.


    :param image_size:
    :param n_classes:
    :param mode:
    :param l2_reg:
    :param min_scale:
    :param max_scale:
    :param scales:
    :param aspect_ratios_global:
    :param aspect_ratios_per_layer:
    :param two_boxes_for_ar1:
    :param steps:
    :param offsets:
    :param clip_boxes:
    :param variances:
    :param coords:
    :param normalize_coords:
    :param subtract_mean:
    :param divide_by_stddev:
    :param swap_channels:
    :param confidence_thresh:
    :param iou_threshold:
    :param top_k:
    :param nms_max_output_size:
    :param return_predictor_sizes:
    :return: Keras SSD model.
    :return: predictor_sizes (optional): A Numpy array containing the `(height, width)` portion
                of the output tensor shape for each convolutional predictor layer. During
                training, the generator function needs this in order to transform
                the ground truth labels into tensors of identical structure as the
                output tensors of the model, which is in turn needed for the cost
                function.
    """
    n_predictor_layers = 4  # The number of predictor conv layers
    n_classes += 1  # Account for the background class
    img_height, img_width, img_channels = image_size

    #################
    # Some exceptions
    #################
    if aspect_ratios_global is None and aspect_ratios_per_layer is None:
        raise ValueError('"aspect_ratio_global" and "aspect_ratios_per_layer" cannot both be None.'
                         'At least one needs to be specified.')
    if aspect_ratios_per_layer and len(aspect_ratios_per_layer) != n_predictor_layers:
        raise ValueError('It must be either "aspect_ratio_per_layer" is None or '
                         f'len(aspect_ratio_per_layer == {n_predictor_layers}, '
                         f'but len(aspect_ratio_per_layer) == {len(aspect_ratios_per_layer)}.')
    if (min_scale is None or max_scale is None) and scales is None:
        raise ValueError('Either "min_scale" and "max_scale" or "scales" need to be specified.')
    if scales:
        if len(scales) != n_predictor_layers + 1:
            raise ValueError(f'It must be either "scales" is None or len(scales) =={n_predictor_layers + 1} '
                             f'but len(scales) == {len(scales)}')
    # If no explicit list of scaling factors is passed,
    # compute the list of scaling from min_scale and max_scale
    else:
        scales = np.linspace(min_scale, max_scale, n_predictor_layers + 1)
    if len(variances) != 4:
        raise ValueError(f'4 variance values must be passed, but {len(variances)} values were received.')
    variances = np.array(variances)
    if np.any(variances <= 0):
        raise ValueError(f'All variances must be > 0, but the variances given are {variances}')
    if steps is not None and len(steps) != n_predictor_layers:
        raise ValueError('You must provide at least one step value per predictor layer.')
    if offsets is not None and len(offsets) != n_predictor_layers:
        raise ValueError('You must provide at least one offset value per predictor layer.')

    ###################################
    # Compute the anchor box parameters
    ###################################
    # Set the aspect ratio for each predictor layer.
    # There are only need for the anchor box layers.
    if aspect_ratios_per_layer:
        aspect_ratios = aspect_ratios_per_layer
    else:
        aspect_ratios = [aspect_ratios_global] * n_predictor_layers

    # Compute the number of boxes to be predicted per cell for each predictor layer.
    # We need this so that we know how many channels the predictor layer needs to have.
    if aspect_ratios_per_layer:
        n_boxes = []
        for ar in aspect_ratios_per_layer:
            if (1 in ar) & two_boxes_for_ar1:
                n_boxes.append(len(ar) + 1)  # +1 for the second box for the aspect ratio 1
            else:
                n_boxes.append(len(ar))
    # If only a global aspect ratio list was passed,
    # then the number of boxes is the same for each predictor layer
    else:
        if (1 in aspect_ratios_global) & two_boxes_for_ar1:
            n_box = len(aspect_ratios_global) + 1
        else:
            n_box = len(aspect_ratios_global)
        n_boxes = [n_box] * n_predictor_layers

    if steps is None:
        steps = [None] * n_predictor_layers
    if offsets is None:
        offsets = [None] * n_predictor_layers

    ##############################################
    # Define functions for the lambda layers below
    ##############################################
    def identity_layer(tensor):
        return tensor

    def input_mean_normalization(tensor):
        return tensor - np.array(subtract_mean)

    def input_stddev_normalization(tensor):
        return tensor / np.array(divide_by_stddev)

    def input_channel_swap(tensor):
        if len(swap_channels) == 3:
            return K.stack(tensor[..., swap_channels[0]],
                           tensor[..., swap_channels[1]],
                           tensor[..., swap_channels[2]],
                           axis=-1)
        elif len(swap_channels) == 4:
            return K.stack(tensor[..., swap_channels[0]],
                           tensor[..., swap_channels[1]],
                           tensor[..., swap_channels[2]],
                           tensor[..., swap_channels[3]],
                           axis=-1)

    ###################
    # Build the network
    ###################
    x = Input(shape=(img_height, img_width, img_channels))

    # The following identity layer is only needed so that the subsequent lambda layer can be optional.
    x1 = Lambda(identity_layer, output_shape=(img_height, img_width, img_channels), name='identity_layer')(x)

    if subtract_mean is not None:
        x1 = Lambda(input_mean_normalization, output_shape=(img_height, img_width, img_channels),
                    name='input_mean_normalization')
    if divide_by_stddev is not None:
        x1 = Lambda(input_stddev_normalization, output_shape=(img_height, img_width, img_channels),
                    name='input_stddev_normalization')
    if swap_channels:
        x1 = Lambda(input_channel_swap, output_shape=(img_height, img_width, img_channels),
                    name='input_channel_swap')

    # Layer 1
    conv1 = Conv2D(filters=32,
                   kernel_size=(5, 5),
                   strides=(1, 1),
                   padding='same',
                   kernel_initializer='he_normal',
                   kernel_regularizer=l2(l2_reg),
                   name='conv1')(x1)
    # Tensorflow uses filter format [filter_height, filter_width, in_channels, out_channels], hence axis = 3
    conv1 = BatchNormalization(axis=3, momentum=0.99, name='bn1')(conv1)  # todo axis
    conv1 = ELU(name='elu1')(conv1)
    conv1 = MaxPooling2D(pool_size=(2, 2), name='pool1')(conv1)

    # Layer 2
    conv2 = Conv2D(filters=48,
                   kernel_size=(3, 3),
                   strides=(1, 1),
                   padding='same',
                   kernel_initializer='he_normal',
                   kernel_regularizer=l2(l2_reg),
                   name='conv2')(conv1)
    conv2 = BatchNormalization(axis=3, momentum=0.99, name='bn2')(conv2)
    conv2 = ELU(name='elu2')(conv2)
    conv2 = MaxPooling2D(pool_size=(2, 2), name='pool2')(conv2)

    # Layer 3
    conv3 = Conv2D(filters=64,
                   kernel_size=(3, 3),
                   strides=(1, 1),
                   padding='same',
                   kernel_initializer='he_normal',
                   kernel_regularizer=l2(l2_reg),
                   name='conv3')(conv2)
    conv3 = BatchNormalization(axis=3, momentum=0.99, name='bn3')(conv3)
    conv3 = ELU(name='elu3')(conv3)
    conv3 = MaxPooling2D(pool_size=(2, 2), name='pool3')(conv3)

    # Layer 4
    conv4 = Conv2D(filters=64,
                   kernel_size=(3, 3),
                   strides=(1, 1),
                   padding='same',
                   kernel_initializer='he_normal',
                   kernel_regularizer=l2(l2_reg),
                   name='conv4')(conv3)
    conv4 = BatchNormalization(axis=3, momentum=0.99, name='bn4')(conv4)
    conv4 = ELU(name='elu4')(conv4)
    conv4 = MaxPooling2D(pool_size=(2, 2), name='pool4')(conv4)

    # Layer 5
    conv5 = Conv2D(filters=48,
                   kernel_size=(3, 3),
                   strides=(1, 1),
                   padding='same',
                   kernel_initializer='he_normal',
                   kernel_regularizer=l2(l2_reg),
                   name='conv5')(conv4)
    conv5 = BatchNormalization(axis=3, momentum=0.99, name='bn5')(conv5)
    conv5 = ELU(name='elu5')(conv5)
    conv5 = MaxPooling2D(pool_size=(2, 2), name='pool5')(conv5)

    # Layer 6
    conv6 = Conv2D(filters=48,
                   kernel_size=(3, 3),
                   strides=(1, 1),
                   padding='same',
                   kernel_initializer='he_normal',
                   kernel_regularizer=l2(l2_reg),
                   name='conv6')(conv5)
    conv6 = BatchNormalization(axis=3, momentum=0.99, name='bn6')(conv6)
    conv6 = ELU(name='elu6')(conv6)
    conv6 = MaxPooling2D(pool_size=(2, 2), name='pool6')(conv6)

    # Layer 7
    conv7 = Conv2D(filters=32,
                   kernel_size=(3, 3),
                   strides=(1, 1),
                   padding='same',
                   kernel_initializer='he_normal',
                   kernel_regularizer=l2(l2_reg),
                   name='conv7')(conv6)
    conv7 = BatchNormalization(axis=3, momentum=0.99, name='bn7')(conv7)
    conv7 = ELU(name='elu7')(conv7)

    # The next part is to add the convolutional predictor layers.
    # Build the convolutional predictor layers on the top of conv layers 4, 5, 6 and 7.
    # We build two predictor layers on top of each of these layers:
    # 1. One for class prediction (classification).
    # 2. One for box coordinate prediction (localization).
    # We predict n_classes confidence values for each box,
    # hence the classes predictors have depth n_boxes * n_classes.
    # We predict 4 coordinates for each box,
    # hence the boxes predictors have depth n_boxes * 4.
    # Output shape of class predictors: (batch, height, width, n_boxes * n_classes)
    classes4 = Conv2D(filters=n_boxes[0] * n_classes,
                      kernel_size=(3, 3),
                      strides=(1, 1),
                      padding='same',
                      kernel_initializer='he_normal',
                      kernel_regularizer=l2(l2_reg),
                      name='classes4')(conv4)
    classes5 = Conv2D(filters=n_boxes[1] * n_classes,
                      kernel_size=(3, 3),
                      strides=(1, 1),
                      padding='same',
                      kernel_initializer='he_normal',
                      kernel_regularizer=l2(l2_reg),
                      name='classes5')(conv5)
    classes6 = Conv2D(filters=n_boxes[2] * n_classes,
                      kernel_size=(3, 3),
                      strides=(1, 1),
                      padding='same',
                      kernel_initializer='he_normal',
                      kernel_regularizer=l2(l2_reg),
                      name='classes6')(conv6)
    classes7 = Conv2D(filters=n_boxes[3] * n_classes,
                      kernel_size=(3, 3),
                      strides=(1, 1),
                      padding='same',
                      kernel_initializer='he_normal',
                      kernel_regularizer=l2(l2_reg),
                      name='classes7')(conv7)

    # Output shape of boxes predictor: (batch, height, width, n_boxes * 4)
    boxes4 = Conv2D(filters=n_boxes[0] * 4,
                    kernel_size=(3, 3),
                    strides=(1, 1),
                    kernel_initializer='he_normal',
                    kernel_regularizer=l2(l2_reg),
                    name='boxes4')(conv4)
    boxes5 = Conv2D(filters=n_boxes[1] * 4,
                    kernel_size=(3, 3),
                    strides=(1, 1),
                    kernel_initializer='he_normal',
                    kernel_regularizer=l2(l2_reg),
                    name='boxes5')(conv5)
    boxes6 = Conv2D(filters=n_boxes[2] * 4,
                    kernel_size=(3, 3),
                    strides=(1, 1),
                    kernel_initializer='he_normal',
                    kernel_regularizer=l2(l2_reg),
                    name='boxes6')(conv6)
    boxes7 = Conv2D(filters=n_boxes[3] * 4,
                    kernel_size=(3, 3),
                    strides=(1, 1),
                    kernel_initializer='he_normal',
                    kernel_regularizer=l2(l2_reg),
                    name='boxes7')(conv7)

    # Generate the anchor boxes
    # Output shape of anchors: (batch, height, width, n_boxes, 8) # todo 8
    anchors4 = AnchorBoxes(img_height,
                           img_width,
                           this_scale=scales[0],
                           next_scale=scales[1],
                           aspect_ratio=aspect_ratios[0],
                           two_boxes_for_ar1=two_boxes_for_ar1,
                           this_steps=steps[0],
                           this_offsets=offsets[0],
                           clip_boxes=clip_boxes,
                           variances=variances,
                           coords=coords,
                           normalize_coords=normalize_coords,
                           name='anchors4')(boxes4)
    anchors5 = AnchorBoxes(img_height,
                           img_width,
                           this_scale=scales[1],
                           next_scale=scales[2],
                           aspect_ratio=aspect_ratios[1],
                           two_boxes_for_ar1=two_boxes_for_ar1,
                           this_steps=steps[1],
                           this_offsets=offsets[1],
                           clip_boxes=clip_boxes,
                           variances=variances,
                           coords=coords,
                           normalize_coords=normalize_coords,
                           name='anchors5')(boxes5)
    anchors6 = AnchorBoxes(img_height,
                           img_width,
                           this_scale=scales[2],
                           next_scale=scales[3],
                           aspect_ratio=aspect_ratios[2],
                           two_boxes_for_ar1=two_boxes_for_ar1,
                           this_steps=steps[2],
                           this_offsets=offsets[2],
                           clip_boxes=clip_boxes,
                           variances=variances,
                           coords=coords,
                           normalize_coords=normalize_coords,
                           name='anchors6')(boxes6)
    anchors7 = AnchorBoxes(img_height,
                           img_width,
                           this_scale=scales[3],
                           next_scale=scales[4],
                           aspect_ratio=aspect_ratios[3],
                           two_boxes_for_ar1=two_boxes_for_ar1,
                           this_steps=steps[3],
                           this_offsets=offsets[3],
                           clip_boxes=clip_boxes,
                           variances=variances,
                           coords=coords,
                           normalize_coords=normalize_coords,
                           name='anchors7')(boxes7)

    # Reshape the class predictions, yielding 3D tensors of shape
    # (batch, height * width * n_boxes, n_classes)
    # We want the class isolated in the last axis to perform softmax on them
    classes4_reshaped = Reshape((-1, n_classes), name='classes4_reshaped')(classes4)
    classes5_reshaped = Reshape((-1, n_classes), name='classes5_reshaped')(classes5)
    classes6_reshaped = Reshape((-1, n_classes), name='classes6_reshaped')(classes6)
    classes7_reshaped = Reshape((-1, n_classes), name='classes7_reshaped')(classes7)

    # Reshape the box coordinate prediction, yielding 3D tensors of shape
    # (batch, height * width * n_boxes, 4)
    # We want the four box coordinates isolated in the last axis to compute the smooth L1 loss
    boxes4_reshaped = Reshape((-1, 4), name='boxes4_reshaped')(boxes4)
    boxes5_reshaped = Reshape((-1, 4), name='boxes5_reshaped')(boxes5)
    boxes6_reshaped = Reshape((-1, 4), name='boxes6_reshaped')(boxes6)
    boxes7_reshaped = Reshape((-1, 4), name='boxes7_reshaped')(boxes7)

    # Reshape the anchor box tensors, yielding 3D tensor of shape
    # (batch, height * width * n_boxes, 8)
    anchors4_reshaped = Reshape((-1, 8), name='anchors4_reshaped')(anchors4)
    anchors5_reshaped = Reshape((-1, 8), name='anchors5_reshaped')(anchors5)
    anchors6_reshaped = Reshape((-1, 8), name='anchors6_reshaped')(anchors6)
    anchors7_reshaped = Reshape((-1, 8), name='anchors7_reshaped')(anchors7)

    # Concatenate the predictions from the different layers and
    # the associated anchor box tensors.
    # Output shape classes_concat: (batch, n_boxes_total, n_classes)
    classes_concat = Concatenate(axis=1, name='classes_concat')([classes4_reshaped,
                                                                 classes5_reshaped,
                                                                 classes6_reshaped,
                                                                 classes7_reshaped])
    # Output shape of boxes_concat: (batch, n_boxes_total, 4)
    boxes_concat = Concatenate(axis=1, name='boxes_concat')([boxes4_reshaped,
                                                             boxes5_reshaped,
                                                             boxes6_reshaped,
                                                             boxes7_reshaped])
    # Output shape of anchors_concat: (batch, n_boxes_total, 8)
    anchors_concat = Concatenate(axis=1, name='anchors_concat')([anchors4_reshaped,
                                                                 anchors5_reshaped,
                                                                 anchors6_reshaped,
                                                                 anchors7_reshaped])

    # The box coordinate predictions will go into the loss functions just the way they are,
    # but for the class predictions, we will apply a softmax activation layer first.
    classes_softmax = Activation('softmax', name='classes_softmax')(classes_concat)

    # Concatenate the class and box coordinate predictions and the anchors to
    # one large predictions tensor.
    # Output shape of predictions: (batch, n_boxes_total, n_classes + 4 + 8)
    predictions = Concatenate(axis=2, name='predictions')([classes_softmax,
                                                           boxes_concat,
                                                           anchors_concat])

    if mode == 'training':
        model = Model(inputs=x, outputs=predictions)
    elif mode == 'inference':
        pass  # todo
    else:
        raise ValueError('"mode" must be one of "training" or "inference,'
                         f'but received {mode}')

    if return_predictor_sizes:
        # todo classes4._keras_shape[1:3]
        predictor_sizes = np.array([classes4._keras_shape[1:3],
                                    classes5._keras_shape[1:3],
                                    classes6._keras_shape[1:3],
                                    classes7._keras_shape[1:3]])
        return model, predictor_sizes
    else:
        return model
