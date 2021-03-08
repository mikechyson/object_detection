#!/usr/bin/env python3
"""
@project: object_detection
@file: train
@author: mike
@time: 2021/3/4
 
@function:
"""
import keras.backend as K
from models.SSD7 import SSD7
from keras.optimizers import Adam
from loss.loss import SSDLoss

# Set the configs
img_height = 300
img_width = 480
img_channels = 3
# Set this to your preference (maybe `None`).
# The current settings transform the input pixel values to the interval `[-1,1]`.
intensity_mean = 127.5
intensity_range = 127.5
n_classes = 5  # Number of positive classes
# An explicit list of anchor box scaling factors.
# If this is passed, it will override `min_scale` and `max_scale`.
scales = [0.08, 0.16, 0.32, 0.64, 0.96]
aspect_ratios = [0.5, 1.0, 2.0]  # The list of aspect ratios for the anchor boxes
two_boxes_for_ar1 = True  # Whether or not you want to generate two anchor boxes for aspect ratio 1
steps = None  # In case you'd like to set the step sizes for the anchor box grids manually; not recommended
offsets = None  # In case you'd like to set the offsets for the anchor box grids manually; not recommended
clip_boxes = False  # Whether or not to clip the anchor boxes to lie entirely within the image boundaries
variances = [1.0, 1.0, 1.0, 1.0]  # The list of variances by which the encoded target coordinates are scaled
normalize_coords = True  # Whether or not the model is supposed to use coordinates relative to the image size

K.clear_session()  # Clear previous models from memory.

# Create the model
model = SSD7(image_size=(img_height, img_width, img_channels),
             n_classes=n_classes,
             mode='training',
             l2_reg=0.0005,
             scales=scales,
             aspect_ratios_global=aspect_ratios,
             aspect_ratios_per_layer=None,
             two_boxes_for_ar1=two_boxes_for_ar1,
             steps=steps,
             offsets=offsets,
             clip_boxes=clip_boxes,
             variances=variances,
             normalize_coords=normalize_coords,
             subtract_mean=intensity_mean,
             divide_by_stddev=intensity_range)
print(model.summary())

# Compile the model
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)
model.compile(optimizer=adam, loss=ssd_loss.compute_loss)
