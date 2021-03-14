#!/usr/bin/env python3
"""
@project: object_detection
@file: ssd_predict
@author: mike
@time: 2021/3/14
 
@function:
"""
import matplotlib.pyplot as plt
from generator.data_generator import DataGenerator
import numpy as np
from encoder_decoder.output_decoder import decode_detections
from keras import backend as K
from models.SSD7 import SSD7

img_height = 300  # Height of the input images
img_width = 480  # Width of the input images
img_channels = 3  # Number of color channels of the input images
# Set this to your preference (maybe `None`).
# The current settings transform the input pixel values to the interval `[-1,1]`.
intensity_mean = 127.5
# Set this to your preference (maybe `None`).
# The current settings transform the input pixel values to the interval `[-1,1]`.
intensity_range = 127.5
n_classes = 5  # Number of positive classes
# An explicit list of anchor box scaling factors. If this is passed, it will override `min_scale` and `max_scale`.
scales = [0.08, 0.16, 0.32, 0.64, 0.96]
aspect_ratios = [0.5, 1.0, 2.0]  # The list of aspect ratios for the anchor boxes
two_boxes_for_ar1 = True  # Whether or not you want to generate two anchor boxes for aspect ratio 1
steps = None  # In case you'd like to set the step sizes for the anchor box grids manually; not recommended
offsets = None  # In case you'd like to set the offsets for the anchor box grids manually; not recommended
clip_boxes = False  # Whether or not to clip the anchor boxes to lie entirely within the image boundaries
variances = [1.0, 1.0, 1.0, 1.0]  # The list of variances by which the encoded target coordinates are scaled
normalize_coords = True  # Whether or not the model is supposed to use coordinates relative to the image size

# 1: Build the Keras model

K.clear_session()  # Clear previous models from memory.

model = SSD7(image_size=(img_height, img_width, img_channels),
             n_classes=n_classes,
             mode='training',
             l2_regularization=0.0005,
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

model.load_weights('checkpoints/ssd7_epoch-07_loss-5.1630_val_loss-6.1335.h5')

val_dataset = DataGenerator(load_images_into_memory=True,
                            hdf5_dataset_path='dataset/dataset_udacity_traffic_val.h5')
########################################################################################################################
# Predictions
predict_generator = val_dataset.generate(batch_size=1,
                                         shuffle=True,
                                         transformations=[],
                                         label_encoder=None,
                                         returns={'processed_images',
                                                  'processed_labels',
                                                  'filenames'},
                                         keep_images_without_gt=False)
batch_images, batch_labels, batch_filenames = next(predict_generator)
i = 0  # Which batch item to look at
# print('Image: ', batch_filenames[i])
# print()
print('Ground truth boxes:\n')
print(batch_labels[i])

y_pred = model.predict(batch_images)
y_pred_decoded = decode_detections(y_pred,
                                   confidence_thresh=0.5,
                                   iou_threshold=0.45,
                                   top_k=200,
                                   normalize_coords=normalize_coords,
                                   img_height=img_height,
                                   img_width=img_width)
# These options determine the way floating point numbers, arrays and
# other NumPy objects are displayed.
np.set_printoptions(precision=2, suppress=True, linewidth=90)
print('Predicted boxes:\n')
print('   class   conf   xmin   ymin   xmax   ymax')
print(y_pred_decoded[i])

########################################################################################################################
# Draw boxes onto the image.

plt.figure(figsize=(20, 12))
plt.imshow(batch_images[i])
current_axis = plt.gca()
# Set colors for the bounding boxes.
colors = plt.cm.hsv(np.linspace(0, 1, n_classes + 1)).tolist()
classes = ['background', 'car', 'truck', 'pedestrian', 'bicyclist', 'light']

# Draw the ground truth boxes in green.
for box in batch_labels[i]:
    xmin = box[1]
    ymin = box[2]
    xmax = box[3]
    ymax = box[4]
    label = f'{classes[int(box[0])]}'
    current_axis.add_patch(
        plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, color='green', fill=False, linewidth=2)
    )
# Draw the predicted boxes in blue.
for box in y_pred_decoded[i]:
    xmin = box[-4]
    ymin = box[-3]
    xmax = box[-2]
    ymax = box[-1]
    color = colors[int(box[0])]
    label = f'{classes[int(box[0])]}: {box[1]:.2f}'
    current_axis.add_patch(
        plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, color=color, fill=False, linewidth=2)
    )
    current_axis.text(xmin, ymin, label, size='x-large', color='white', bbox={'facecolor': color, 'alpha': 1.0})
plt.show()