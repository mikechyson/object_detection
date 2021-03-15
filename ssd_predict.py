import cv2
import matplotlib.pyplot as plt
from generator.data_generator import DataGenerator
import numpy as np
from encoder_decoder.output_decoder import decode_detections
from keras import backend as K
from models.ssd7 import build_ssd7
from loss.loss import SSDLoss
from keras.models import load_model
from layers.L2Normalization import L2Normalization
from layers.DecodeDetections import DecodeDetections
from layers.AnchorBoxes import AnchorBoxes
from imageio import imread
from keras.preprocessing import image

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

K.clear_session()  # Clear previous models from memory.
model = build_ssd7(image_size=(img_height, img_width, img_channels),
                   n_classes=n_classes,
                   mode='inference',
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

model_path = 'checkpoints/ssd7_epoch-12_loss-2.3330_val_loss-1.9446.h5'
ssd_loss = SSDLoss(neg_pos_ratio=3, n_neg_min=0, alpha=1.0)
K.clear_session()
model = load_model(model_path, custom_objects={'AnchorBoxes': AnchorBoxes,
                                               'L2Normalization': L2Normalization,
                                               'DecodeDetections': DecodeDetections,
                                               'compute_loss': ssd_loss.compute_loss})

orig_images = []  # Store the images here.
input_images = []  # Store resized versions of the images here.

img_path = '/Users/mike/Downloads/udacity_driving_datasets/1478019960680764792.jpg'

orig_images.append(imread(img_path))
img = image.load_img(img_path, target_size=(img_height, img_width))
img = image.img_to_array(img)
input_images.append(img)
input_images = np.array(input_images)

y_pred = model.predict(input_images)

confidence_threshold = 0.5

y_pred_thresh = [y_pred[k][y_pred[k, :, 1] > confidence_threshold] for k in range(y_pred.shape[0])]

np.set_printoptions(precision=2, suppress=True, linewidth=90)
print("Predicted boxes:\n")
print('   class   conf xmin   ymin   xmax   ymax')
print(y_pred_thresh[0])

# Set the colors for the bounding boxes
colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
classes = ['background', 'car', 'truck', 'pedestrian', 'bicyclist', 'light']

plt.figure(figsize=(20, 12))
plt.imshow(orig_images[0])

current_axis = plt.gca()

for box in y_pred_thresh[0]:
    # Transform the predicted bounding boxes for the 512x512 image to the original image dimensions.
    xmin = box[-4] * orig_images[0].shape[1] / img_width
    ymin = box[-3] * orig_images[0].shape[0] / img_height
    xmax = box[-2] * orig_images[0].shape[1] / img_width
    ymax = box[-1] * orig_images[0].shape[0] / img_height
    color = colors[int(box[0])]
    label = '{}: {:.2f}'.format(classes[int(box[0])], box[1])
    current_axis.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, color=color, fill=False, linewidth=2))
    current_axis.text(xmin, ymin, label, size='x-large', color='white', bbox={'facecolor': color, 'alpha': 1.0})

plt.show()
