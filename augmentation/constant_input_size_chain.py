from __future__ import division
import numpy as np
from utils.boxes_validation import BoxFilter, ImageValidator


class DataAugmentationConstantInputSize:
    """
    Applies a chain of photometric and geometric image transformations.
    For documentation, please refer to the documentation of the individual transformation involved.

    Note: This augmentation chain is suitable for constant size images only.
    """

    def __init__(self,
                 random_brightness=(-48, 48, 0.5),
                 random_contrast=(0.5, 1.8, 0.5),
                 random_saturation=(0.5, 1.8, 0.5),
                 random_hue=(18, 0.5),
                 random_flip=0.5,
                 random_translate=((0.03, 0.5), (0.03, 0.5), 0.5),
                 random_scale=(0.5, 2.0, 0.5),
                 n_trials_max=3,
                 clip_boxes=True,
                 overlap_criterion='area',
                 bounds_box_filter=(0.3, 1.0),
                 bounds_validator=(0.5, 1.0),
                 n_boxes_min=1,
                 background=(0, 0, 0),
                 labels_format={'class_id': 0, 'xmin': 1, 'ymin': 2, 'xmax': 3, 'ymax': 4}
                 ):
        # The prerequisite check.
        if random_scale[0] >= 1 or random_scale[1] <= 1:
            raise ValueError("This sequence of transformations only makes sense if the minimum scaling "
                             "factor <1 and the maximum scaling factor >1.")

        self.n_trials_max = n_trials_max
        self.clip_boxes = clip_boxes
        self.overlap_criterion = overlap_criterion
        self.bounds_box_filter = bounds_box_filter
        self.bounds_validator = bounds_validator
        self.n_boxes_min = n_boxes_min
        self.background = background
        self.labels_format = labels_format

        # Determines which boxes are kept in an image after the transformations have been applied.
        self.box_filter = BoxFilter(check_overlap=True,
                                    check_min_area=True,
                                    check_degenerate=True,
                                    overlap_criterion=self.overlap_criterion,
                                    overlap_bounds=self.bounds_box_filter,
                                    min_area=16,
                                    labels_format=self.labels_format)

        # Determines whether the result of the transformations is a valid training image.
        self.image_validator = ImageValidator(overlap_criterion=self.overlap_criterion,
                                              bounds=self.bounds_validator,
                                              n_boxes_min=self.n_boxes_min,
                                              labels_format=self.labels_format)

        # Utility distortions.
