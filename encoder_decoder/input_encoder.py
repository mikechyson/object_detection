#!/usr/bin/env python3
"""
@project: object_detection
@file: input_encoder
@author: mike
@time: 2021/3/10
 
@function:
"""
import numpy as np


class SSDInputEncoder:
    """
    Transforms ground truth for object detection in images
    (2D bounding box coordinates and class labels) to the format required
    for training an SSD model.

    In the process of encoding the ground truth labels, a template of anchor
    boxes is being built, which are subsequently matched to the ground truth
    boxes via an intersection-over-union threshold criterion.
    """

    def __init__(self,
                 img_height,
                 img_width,
                 n_classes,
                 predictor_sizes,
                 min_scale=0.1,
                 max_scale=0.9,
                 scales=None,
                 aspect_ratios_global=[0.5, 1.0, 2.0],
                 aspect_ratios_per_layer=None,
                 two_boxes_for_ar1=True,
                 steps=None,
                 offsets=None,
                 clip_boxes=False,
                 variances=[0.1, 0.1, 0.2, 0.2],
                 matching_type='multi',
                 pos_iou_threshold=0.5,
                 neg_iou_limit=0.3,
                 border_pixels='half',
                 coords='centroids',
                 normalize_coords=True,
                 background_id=0):
        """
        Arguments:
            img_height: The height of the input images.
            img_width: The width of the input images.
            n_classes: The number of positive classes.
            predictor_sizes: A list of int-tuples of the format `(height, width)` containing
                the output heights and widths of the convolutional predictor layers.
            min_scale: The smallest scaling factor for the size of the anchor boxes as a fraction of
                the shorter side of the input images. Note that you should set the scaling factors
                such that the resulting anchor box sizes correspond to the sizes of the objects you
                are trying to detect. Must be >0.
            max_scale: The largest scaling factor for the size of the anchor boxes as a fraction of
                the shorter size of the input images. All scaling factors between the smallest and
                the largest factors will be linearly interpolated. Note that the second to last of
                the linearly interpolated scaling factors will actually be the scaling factor for
                the last predictor layer, while the last scaling factor is used for the second box
                for aspect ratio 1 in the last predictor layer if `two_boxes_for_ar1` is `True`.
                Note that you should set the scaling factors such that the resulting anchor box
                sizes corresponding to the sizes of the objects you are trying to detect. Must be
                greater than or equal to `min_scale`.
            scales: A list of floats >0 containing scaling factors per convolutional predictor layer.
                This list must be one element longer than the number of predictor layers. The first
                `k` elements are the scaling factors for the `k` predictor layers, while the last
                element is used for the second box for aspect ratio 1 in the last predictor layer if
                `two_boxes_for_ar1` is `True`. This additional last scaling must be passed either way,
                even if it is not being used. If a list is passed, this argument overrides `min_scale`
                and `max_scale`. All scaling factors must be greater than zero. Note that you should
                set the scaling factors such that the resulting anchor box sizes correspond to
                the sizes of the objects you are trying to detect.
            aspect_ratios_global: The list of aspect ratios for which anchor boxes are to be generated.
                This list is valid for all prediction layers. Note that you should set the aspect ratios
                such that the resulting anchor box shapes roughly correspond to the shapes of the objects
                you are trying to detect.
            aspect_ratios_per_layer: A list containing one aspect ratio list for each prediction layer.
                If a list is passed, it overrides `aspect_ratios_global`. Note that you should set the
                aspect ratios such that the resulting anchor box shapes very roughly correspond to the
                shapes of the objects you are trying to detect.
            two_boxes_for_ar1: Only relevant for aspect ratios lists that contain 1. Will be ignored otherwise.
                If `True`, two anchor boxes will be generated for aspect ratio 1. The first will be generated
                using the scaling factor for the respective layer, the second one will be generated using
                geometric mean of said scaling factor and next bigger scaling factor.
            steps: `None` or a list with as many elements as there are predictor layers. The elements can be
                either ints/floats or tuples of two ints/floats. These numbers represent for each predictor layer
                how many pixels apart the anchor box center points should be vertically and horizontally along
                the spatial grid over the image. If the list contains ints/floats, then that value will be used
                for both spatial dimensions. If the list contains tuples of two ints/floats, then they represent
                `(step_height, step_width)`. If no steps are provided, then they will be computed such that the
                anchor box center points will form an equidistant grid within the image dimensions.
            offsets: `None` or a list with as many elements as there are predictor layers. The elements can be
                either floats or tuples of two floats. These numbers represent for each predictor layer how many
                pixels from the top and left boarders of the image the top-most and left-most anchor box center
                points should be as a fraction of `steps`. The last bit is important: The offsets are not absolute
                pixel values, but fractions of the step size specified in the `steps` argument. If the list
                contains floats, then that value will be used for both spatial dimensions. If the list contains
                tuples of two floats, then they represent `(vertical_offset, horizontal_offset)`. If no offsets
                are provided, then they will default to 0.5 of the step size.
            clip_boxes: If `True`, limits the anchor box coordinates to stay within image boundaries.
            variances: A list of 4 floats >0. The anchor box offset for each coordinate will be divided by
                its respective variance value.
            matching_type: Can be either 'multi' or 'bipartite'. In 'bipartite' mode, each ground truth box will
                be matched only to the one anchor box with the highest IoU overlap. In 'multi' mode, in addition
                to the aforementioned bipartite matching, all anchor boxes with an IoU overlap greater than or
                equal to the `pos_iou_threshold` will be matched to a given ground truth box.
            pos_iou_threshold: The intersection-over-union similarity threshold that must be
                met in order to match a given ground truth box to a given anchor box.
            neg_iou_limit: The maximum allowed intersection-over-union similarity of an
                anchor box with any ground truth box to be labeled a negative (i.e. background) box. If an
                anchor box is neither a positive, nor a negative box, it will be ignored during training.
            border_pixels: How to treat the border pixels of the bounding boxes.
                Can be 'include', 'exclude', or 'half'. If 'include', the border pixels belong
                to the boxes. If 'exclude', the border pixels do not belong to the boxes.
                If 'half', then one of each of the two horizontal and vertical borders belong
                to the boxex, but not the other.
            coords: The box coordinate format to be used internally by the model (i.e. this is not the input format
                of the ground truth labels). Can be either 'centroids' for the format `(cx, cy, w, h)`,
                'minmax' for the format `(xmin, xmax, ymin, ymax)`, or 'corners' for the format
                `(xmin, ymin, xmax, ymax)`.
            normalize_coords: If `True`, the encoder uses relative instead of absolute coordinates.
                This means instead of using absolute target coordinates, the encoder will scale all coordinates
                to be within [0,1]. This way learning becomes independent of the input image size.
            background_id: Determines which class ID is for the background class.
        """
        predictor_sizes = np.array(predictor_sizes)
        if predictor_sizes.ndim == 1:
            predictor_sizes = np.expand_dims(predictor_sizes, axis=0)

        ################################################################################
        # Handle Exceptions.
        ################################################################################
        if (min_scale is None or max_scale is None) and scales is None:
            raise ValueError("Either `min_scale` and `max_scale` or `scales` need to be specified.")

        if scales:
            if len(scales) != predictor_sizes.shape[0] + 1:
                raise ValueError(f"It must be either scales is None or len(scales) == len(predictor_sizes) + 1, "
                                 f"but len(scales) == {len(scales)} and len(predictor_size) + 1 =="
                                 f"{len(predictor_sizes) + 1}")
            scales = np.array(scales)
            if np.any(scales <= 0):
                raise ValueError("All value in `scales` must be greater than 0. but the passed list of "
                                 f"scales is {scales}")
        # # If no list of scales was passed, we need to make sure that `min_scale` and `max_scale` are valid values.
        else:
            if not 0 < min_scale <= max_scale:
                raise ValueError(f"It must be 0 < min_scale <= max_scale, but it is min_scale = {min()} and "
                                 f"max_scale = {max_scale}.")

        if aspect_ratios_per_layer is not None:
            if len(aspect_ratios_per_layer) != predictor_sizes.shape[0]:
                raise ValueError("It must be either aspect_ratios_per_layer is None or len(aspect_ratios_per_layer) "
                                 f"== len(predictor_sizes), but len(aspect_ratios_per_layer) = "
                                 f"{len(aspect_ratios_per_layer)} and len(predictor_sizes) = {len(predictor_sizes)}.")
            for aspect_ratios in aspect_ratios_per_layer:
                if np.any(np.array(aspect_ratios) <= 0):
                    raise ValueError("All aspect ratios must be greater than 0.")
        else:
            if aspect_ratios_global is None:
                raise ValueError("At least one of `aspect_ratios_global` and `aspect_ratios_per_layer` must not "
                                 "be `None`.")
            if np.any(np.array(aspect_ratios_global) <= 0):
                raise ValueError("All aspect ratios must be greater than 0.")

        if len(variances) != 4:
            raise ValueError(f"4 variance values must be passed, but {len(variances)} values were received.")
        variances = np.array(variances)
        if np.any(variances <= 0):
            raise ValueError(f"All variances must be >0, but the variances given are {variances}")

        if coords not in {'minmax', 'centroids', 'corners'}:
            raise ValueError(f"Supported values are 'minmax', 'corners' and 'centroids', but received {coords}")

        if steps is not None and len(steps) != predictor_sizes.shape[0]:
            raise ValueError('You must provide at least one step value per predictor layer.')

        if offsets is not None and len(offsets) != predictor_sizes.shape[0]:
            raise ValueError("You must provide at least one offset value per predictor layer.")

        ####################################################################
        # Set or compute members
        ####################################################################
        self.img_height = img_height
        self.img_width = img_width
        self.n_classes = n_classes + 1
        self.predictor_sizes = predictor_sizes
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.two_boxes_for_ar1 = two_boxes_for_ar1
        self.clip_boxes = clip_boxes
        self.variances = variances
        self.matching_type = matching_type
        self.pos_iou_threshold = pos_iou_threshold
        self.neg_iou_limit = neg_iou_limit
        self.border_pixels = border_pixels
        self.coords = coords
        self.normalize_coords = normalize_coords
        self.background_id = background_id
        # If `scales` is None, compute the scaling factors by linearly interpolating between
        # `min_scale` and `max_scale`.
        if scales is None:
            self.scales = np.linspace(self.min_scale, self.max_scale, len(self.predictor_sizes) + 1)
        else:
            self.scales = scales
        # If `aspect_ratios_per_layer` is None, then we use the same list of aspect ratios
        # `aspect_ratios_global` for all predictor layers.
        if aspect_ratios_per_layer is None:
            self.aspect_ratios_per_layer = [aspect_ratios_global] * predictor_sizes.shape[0]
        else:
            self.aspect_ratios_per_layer = aspect_ratios_per_layer
        if steps is not None:
            self.steps = steps
        else:
            self.steps = [None] * predictor_sizes.shape[0]
        if offsets is not None:
            self.offsets = offsets
        else:
            self.offsets = [None] * predictor_sizes.shape[0]

        # Compute the number of boxes per spatial location for each predictor layer.
        # For example, if a predictor layer has three different aspect ratios, [1.0, 0.5, 2.0],
        # and is supposed to predict two boxes of slightly different size for aspect ratio 1.0,
        # then that predictor layer predicts a total of four boxes at every spatial location
        # across the feature map.
        if aspect_ratios_per_layer is not None:
            self.n_boxes = []
            for aspect_ratios in aspect_ratios_per_layer:
                if 1 in aspect_ratios and two_boxes_for_ar1:
                    self.n_boxes.append(len(aspect_ratios) + 1)
                else:
                    self.n_boxes.append(len(aspect_ratios))
        else:
            if 1 in aspect_ratios_global and two_boxes_for_ar1:
                self.n_boxes = len(aspect_ratios_global) + 1
            else:
                self.n_boxes = len(aspect_ratios_global)

        ##################################################################################
        # Compute the anchor boxes for each predictor layer.
        ##################################################################################
        # Compute the anchor boxes for each predictor layer. We only have to do this once
        # since the anchor boxes depend only on the model configuration, not on the input data.
        # For each predictor layer (i.e. for each scaling factor) the tensors for that layer's
        # anchor boxes will have the shape `(feature_map_height, feature_map_width, n_boxes, 4)`.

        self.boxes_list = []  # Store the anchor boxes for each predictor layer.

        # The following  lists just store diagnostic information.
        self.wh_list_diag = []  # Box widths and heights for each predictor layer.
        self.steps_diag = []  # Horizontal and vertical distances between any two boxes for each predictor layer.
        self.offsets_diag = []  # Offsets for each predictor layer.
        self.centers_diag = []  # Anchor box center points as `(cy, cx)` for each predictor layer.

        # Iterate over all predictor layers and compute the anchor boxes for each one.
        for i in range(len(self.predictor_sizes)):
            boxes, center, wh, step, offset = self.generate_anchor_boxes_for_layer()  # todo implementation
            self.boxes_list.append(boxes)
            self.wh_list_diag.append(wh)
            self.steps_diag.append(step)
            self.offsets_diag.append(offset)
            self.centers_diag.append(center)
