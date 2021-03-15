from __future__ import division
import numpy as np
from utils.bounding_box import convert_coordinates
from utils.matching import match_bipartite_greedy
from utils.matching import match_multi
from utils.bounding_box import iou


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
                raise ValueError(f"It must be 0 < min_scale <= max_scale, but it is min_scale = {min_scale} and "
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
        if scales is not None:
            self.scales = scales
        else:
            self.scales = np.linspace(self.min_scale, self.max_scale, len(self.predictor_sizes) + 1)
        # If `aspect_ratios_per_layer` is None, then we use the same list of aspect ratios
        # `aspect_ratios_global` for all predictor layers.
        if aspect_ratios_per_layer is None:
            self.aspect_ratios = [aspect_ratios_global] * predictor_sizes.shape[0]
        else:
            self.aspect_ratios = aspect_ratios_per_layer
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
            boxes, center, wh, step, offset = self.generate_anchor_boxes_for_layer(
                feature_map_size=self.predictor_sizes[i],
                aspect_ratios=self.aspect_ratios[i],
                this_scale=self.scales[i],
                next_scale=self.scales[i + 1],
                this_steps=self.steps[i],
                this_offsets=self.offsets[i],
                diagnostics=True)
            self.boxes_list.append(boxes)
            self.wh_list_diag.append(wh)
            self.steps_diag.append(step)
            self.offsets_diag.append(offset)
            self.centers_diag.append(center)

    def __call__(self, ground_truth_labels, diagnostics=False):
        """
        Convert ground truth bounding box data into a format to train SSD model.

        Arguments:
            ground_truth_labels: A python list of length `batch_size` that contains one 2D Numpy array
                for each batch image. Each such array has `k` rows for the `k` ground truth bounding boxes belonging
                to the respective image, and the data for each ground truth bounding box has the format
                `(class_id, xmin, ymin, xmax, ymax)` (i.e. the 'corners' coordinate format), and `class_id` must be
                an integer greater than 0 for all boxes as class ID 0 is reserved for the background class.
            diagnostics: If `True`, not only the encoded ground truth tensor will be returned,
                but also a copy of it with anchor box coordinates in place of the ground truth coordinates.
                This can be very useful if you want to visualize which anchor boxes got matched to which ground truth
                boxes.
        Returns:
            `y_encoded`, a 3D NumPy array of the shape `(batch_size, #boxes, #classes + 4 + 4 + 4)` that serves as
            the ground truth label tensor for training, where `#boxes` is the total number of boxes predicted by the
            model per image, and the classes are one-hot-encoded. The four elements after the class vectors in the
            last axis are the box coordinates, the next for elements after that are just dummy elements, and the last
            four elements are the variances.
        """
        # Mapping to define which indices represent which coordinates in the ground truth.
        class_id = 0
        xmin = 1
        ymin = 2
        xmax = 3
        ymax = 4

        batch_size = len(ground_truth_labels)

        # Generate the template for y_encdoed.
        y_encoded = self.generate_encoding_template(batch_size=batch_size, diagnostics=False)

        # Match ground truth boxes to anchor boxes.
        # Every anchor box that does not have a ground truth match and for which the maximal IoU overlap
        # with any ground truth box is less than or equal to `neg_iou_limit` will be a negative (background)
        # box.
        y_encoded[:, :, self.background_id] = 1  # All boxes are background boxes be default.
        n_boxes = y_encoded.shape[1]  # The total number of boxes that the model predicts per batch item.
        class_vectors = np.eye(self.n_classes)  # An identity matrix that we'll use an one-hot class vectors.

        for i in range(batch_size):
            # If there is no ground truth for this batch item, there is nothing to match.
            if ground_truth_labels[i].size == 0:
                continue
            labels = ground_truth_labels[i].astype(np.float)
            # Check for degenerate ground truth bounding boxes before attempting any computations.
            if np.any(labels[:, [xmax]] - labels[:, [xmin]] <= 0) or np.any(labels[:, [ymax]] - labels[:, [ymin]] <= 0):
                raise DegenerateBoxError("SSDInputEncoder detected degenerate ground truth bounding boxes for "
                                         f"batch item {i} with bounding boxes {labels}, i.e. bounding boxes where "
                                         f"xmax <= xmin and/or ymax <= ymin. Degenerate ground truth bounding boxes "
                                         f"will lead to NaN errors during the training.")
            # Maybe normalize the box coordinates.
            if self.normalize_coords:
                labels[:, [ymin, ymax]] /= self.img_height
                labels[:, [xmin, xmax]] /= self.img_width

            # Maybe convert the box coordinate format.
            if self.coords == 'centroids':
                labels = convert_coordinates(labels, start_index=xmin,
                                             conversion='corners2centroids',
                                             border_pixels=self.border_pixels)
            elif self.coords == 'minmax':
                labels = convert_coordinates(labels, start_index=xmin, conversion='corners2minmax')

            classes_one_hot = class_vectors[labels[:, class_id].astype(np.int)]
            # The one-hot version of the labels for this batch item.
            labels_one_hot = np.concatenate([classes_one_hot, labels[:, [xmin, ymin, xmax, ymin]]], axis=-1)

            # Compute the IoU similarity between all anchor boxes and all ground truth boxes for this batch item.
            # This is a matrix of shape `(num_ground_truth_boxes, num_anchor_boxes)`.
            similarities = iou(labels[:, [xmin, ymin, xmax, ymax]],
                               y_encoded[i, :, -12:-8],
                               coords=self.coords,
                               mode='outer_product',
                               border_pixels=self.border_pixels)

            # First:
            # Do bipartite matching, i.e. match each ground truth box to the one anchor box with the highest IoU.
            # This ensures that each ground truth box will have at least one good match.
            bipartite_matches = match_bipartite_greedy(weight_matrix=similarities)
            # Write the ground truth data to the matched anchor boxes.
            y_encoded[i, bipartite_matches, :-8] = labels_one_hot
            # Set the columns of the matched anchor boxes to zero to indicate that they were matched.
            similarities[:, bipartite_matches] = 0

            # Second:
            # Maybe do 'multi' matching, where each remaining anchor box will be matched to its most
            # similar ground truth with an IoU of at least `pos_iou_threshold`, or not matched if there
            # is no such ground truth box.
            if self.matching_type == 'multi':
                # Get all matches that specify the IoU threshold.
                matches = match_multi(weight_matrix=similarities, threshold=self.pos_iou_threshold)
                # Write the ground truth data to the matched anchor box.
                y_encoded[i, matches, :-8] = labels_one_hot[matches[0]]
                # Set the columns of the matched anchor boxes to zero to indicate that they were matched.
                similarities[:, matches[1]] = 0

            # Third:
            # After the matching is done, all negative (background) anchor boxes that have an IoU of `neg_iou_limit`
            # or more with any ground truth box will be set to neutral, i.e. they will no longer be background boxes.
            # These anchors are "too close" to a ground truth box to be valid background boxes.
            max_background_similarities = np.amax(similarities, axis=0)
            neutral_boxes = np.nonzero(max_background_similarities >= self.neg_iou_limit)[0]
            y_encoded[i, neutral_boxes, self.background_id] = 0

        # Convert box coordinates to anchor box offsets.
        # [-12, -9]: ground truth
        # [-8, -5]: anchor
        # [-4, -1]: variance
        if self.coords == 'centroids':
            # cx(gt) - cx(anchor), cy(gt) - cy(anchor)
            y_encoded[:, :, [-12, -11]] -= y_encoded[:, :, [-8, -7]]
            # (cx(gt) - cx(anchor)) / w(anchor) / cx_variance,
            # (cy(gt) - cy(anchor)) / w(anchor) / cy_variance
            y_encoded[:, :, [-12, -11]] /= y_encoded[:, :, [-6, -5]] * y_encoded[:, :, [-4, -3]]
            # w(gt) / w(anchor), h(gt) / h(anchor)
            y_encoded[:, :, [-10, -9]] /= y_encoded[:, :, [-6, -5]]
            # ln(w(gt) / w(anchor)) / w_variance, ln(h(gt) / h(anchor)) / h_variance
            y_encoded[:, :, [-10, -9]] = np.log(y_encoded[:, :, [-10, -9]]) / y_encoded[:, :, [-2, -1]]
        elif self.coords == 'corners':
            # gt - anchor for all four coordinates
            y_encoded[:, :, -12:-8] -= y_encoded[:, :, -8:-4]
            # (xmin(gt) - xmin(anchor)) / w(anchor), (xmax(gt) - xmax(anchor)) / w(anchor)
            y_encoded[:, :, [-12, -10]] /= np.expand_dims(y_encoded[:, :, -6] - y_encoded[:, :, -8], axis=-1)
            # (ymin(gt) - ymin(anchor)) / h(anchor), (ymax(gt) - ymax(anchor)) / h(anchor)
            y_encoded[:, :, [-11, -9]] /= np.expand_dims(y_encoded[:, :, -5] - y_encoded[:, :, -7], axis=-1)
            # (gt - anchor) / size(anchor) / variance for all for coordinates,
            # where size refers to w and h respectively.
            y_encoded[:, :, -12:-8] /= y_encoded[:, :, -4:]
        elif self.coords == 'minmax':
            # gt - anchor for all four coordinates
            y_encoded[:, :, -12:-8] -= y_encoded[:, :, -8, -4]
            # (xmin(gt) - xmin(anchor)) / w(anchor), (xmax(gt) - xmax(anchor)) / w(anchor)
            y_encoded[:, :, [-12, -11]] /= np.expand_dims(y_encoded[:, :, -7] - y_encoded[:, :, -8], axis=-1)
            # (ymin(gt) - ymin(anchor)) / h(anchor), (ymax(gt) - ymax(anchor)) / h(anchor)
            y_encoded[:, :, [-10, -9]] /= np.expand_dims(y_encoded[:, :, -5] - y_encoded[:, :, -6], axis=-1)
            # (gt - anchor) / size(anchor) / variance for all for coordinates,
            # where size refers to w and h respectively.
            y_encoded[:, :, -12:-8] /= y_encoded[:, :, -4:]

        if diagnostics:
            # We save the matched anchor boxes (i.e. anchor boxes that were matched  to a ground truth box,
            # but keeping the anchor box coordinates).
            y_matched_anchors = np.copy(y_encoded)
            # Keeping the anchor box coordinates means setting the offsets to zero.
            y_matched_anchors[:, :, -12:-8] = 0
            return y_encoded, y_matched_anchors
        else:
            return y_encoded

    def generate_anchor_boxes_for_layer(self,
                                        feature_map_size,
                                        aspect_ratios,
                                        this_scale,
                                        next_scale,
                                        this_steps=None,
                                        this_offsets=None,
                                        diagnostics=False):
        """
        Compute an array of the spatial positions and sizes of the anchor boxes for one predictor layer
        of size `feature_map_size == [feature_map_height, feature_map_width]`.

        Arguments:
            feature_map_size: A list or tuple `[feature_map_height, feature_map_width]` with the spatial
                dimensions of the feature map for which to generate the anchor boxes.
            aspect_ratios: A list of floats, the aspect ratios for which anchor boxes are to be generated.
                All list element must be unique.
            this_scale: A float in [0, 1], the scaling factor for the size of the generate anchor boxes
                as a fraction of the shorter side the input image.
            next_scale: A float in [0, 1], the next larger scaling factor. Only relevant if
                `self.two_boxes_for_ar1 == True`.
            diagnostics: If `True`, the following additional outputs will be returned:
                1) A list of the center point `x` and `y` coordinates for each spatial location.
                2) A list containing `(width, height)` for each box aspect ratio.
                3) A tuple containing `(step_height, step_width)`.
                4) A tuple containing `(offset_height, offset_width)`.
                This information can be useful to understand in just a few numbers what generated grid of
                anchor boxes actually looks like, i.e. how large the different boxes are and how dense their
                spatial distribution is, in order to determine whether the box grid covers the input images
                appropriately and whether the box sizes are appropriate to fit the sizes of the objects to
                be detected.
        Returns:
            A 4D NumPy tensor of shape `(feature_map_height, feature_map_width, n_boxes_per_cell, 4)` where
            the last dimension contain `(xmin, xmax, ymin, ymax)` for each anchor box in each cell of the
            feature map.
        """
        # Compute box width and height for each aspect ratio.

        # The shorter side of the image will be used to compute `w` and `h` using `scale` and `aspect_ratios`.
        size = min(self.img_width, self.img_height)
        # Compute the box widths and heights for all aspect ratios.
        wh_list = []
        for ar in aspect_ratios:
            if ar == 1:
                # Compute the regular anchor box for aspect ratio 1.
                box_height = box_width = this_scale * size
                wh_list.append((box_width, box_height))
                if self.two_boxes_for_ar1:
                    box_height = box_width = np.sqrt(this_scale * next_scale) * size
                    wh_list.append((box_width, box_height))
            else:
                box_width = this_scale * size * np.sqrt(ar)
                box_height = this_scale * size / np.sqrt(ar)
                wh_list.append((box_width, box_height))
        wh_list = np.array(wh_list)

        n_boxes = len(wh_list)

        # Compute the grid of box center points.
        # They are identical for all aspect ratios.

        # Compute the step sizes, i.e. how far apart the anchor box center points will be vertically and horizontally.
        if this_steps is None:
            step_height = self.img_height / feature_map_size[0]
            step_width = self.img_width / feature_map_size[1]
        else:
            if isinstance(this_steps, (list, tuple)) and len(this_steps) == 2:
                step_height = this_steps[0]
                step_width = this_steps[1]
            elif isinstance(this_steps, (int, float)):
                step_height = this_steps
                step_width = this_steps

        # Compute the offsets, i.e. at what pixel values the first anchor box center point will be
        # from the top and from the left of the image.
        if this_offsets is None:
            offset_height = 0.5
            offset_width = 0.5
        else:
            if isinstance(this_offsets, (list, tuple)) and len(this_offsets) == 2:
                offset_height = this_offsets[0]
                offset_width = this_offsets[1]
            elif isinstance(this_offsets, (int, float)):
                offset_height = this_offsets
                offset_width = this_offsets

        # Now we have the offsets and step sizes, compute the grid of anchor box center points.
        cy = np.linspace(offset_height * step_height,
                         (offset_height + feature_map_size[0] - 1) * step_height,
                         feature_map_size[0])
        cx = np.linspace(offset_width * step_width,
                         (offset_width + feature_map_size[1] - 1) * step_width,
                         feature_map_size[1])
        cx_grid, cy_grid = np.meshgrid(cx, cy)
        # This is necessary for np.tile() to do what we want future down.
        cx_grid = np.expand_dims(cx_grid, -1)
        cy_grid = np.expand_dims(cy_grid, -1)

        # Create a 4D tensor template of shape `(feature_map_height, feature_map_width, n_boxes, 4)`
        # where the last dimension will contain `(cx, cy, w, h)`.
        boxes_tensor = np.zeros((feature_map_size[0], feature_map_size[1], n_boxes, 4))
        boxes_tensor[:, :, :, 0] = np.tile(cx_grid, (1, 1, n_boxes))  # Set cx
        boxes_tensor[:, :, :, 1] = np.tile(cy_grid, (1, 1, n_boxes))  # Set cy
        boxes_tensor[:, :, :, 2] = wh_list[:, 0]  # Set w
        boxes_tensor[:, :, :, 3] = wh_list[:, 1]  # Set h

        # Convert `(cx, cy, w, h)` to `(xmin, ymin, xmax, ymax)`.
        boxes_tensor = convert_coordinates(boxes_tensor, start_index=0, conversion='centroids2corners')

        # If `clip_boxes` is enabled, clip the coordinates to lie within the image boundaries.
        if self.clip_boxes:
            x_coords = boxes_tensor[:, :, :, [0, 2]]
            x_coords[x_coords >= self.img_width] = self.img_width - 1
            x_coords[x_coords < 0] = 0
            boxes_tensor[:, :, :, [0, 2]] = x_coords
            y_coords = boxes_tensor[:, :, :, [1, 3]]
            y_coords[y_coords >= self.img_height] = self.img_height - 1
            y_coords[y_coords < 0] = 0
            boxes_tensor[:, :, :, [1, 3]] = y_coords

        # `normalize_coords` is enabled, normalize the coordinates to be within [0, 1].
        if self.normalize_coords:
            boxes_tensor[:, :, :, [0, 2]] /= self.img_width
            boxes_tensor[:, :, :, [1, 3]] /= self.img_height

        # todo directly limit for (cx, cy, w, h)
        if self.coords == 'centroids':
            boxes_tensor = convert_coordinates(boxes_tensor, start_index=0, conversion='corners2centroids',
                                               border_pixels='half')
        elif self.coords == 'minmax':
            boxes_tensor = convert_coordinates(boxes_tensor, start_index=0, conversion='corners2minmax',
                                               border_pixels=self.border_pixels)

        if diagnostics:
            return boxes_tensor, (cy, cx), wh_list, (step_height, step_width), (offset_height, offset_width)
        else:
            return boxes_tensor

    def generate_encoding_template(self, batch_size, diagnostics=False):
        """
        Product an encoding template for the ground truth label tensor for a given batch.

        Note that all tensor creation, reshaping and concatenation operations performed in this function
        and the sub-functions it calls are identical to those performed inside the SSD model.

        In other words, the boxes in `y_encoded` must have a specific order in order correspond to the right spatial
        positions and scales of the boxes predicted by the model. The sequence of operations here ensures that
        `y_encoded` has this specific form.

        Returns:
            A NumPy array of shape `(batch_size, #boxes, #classes + 12)`, the template into which to encode
            the background truth labels for training. The last axis has length `#classes + 12` because the
            model output contains not only the 4 predicted box coordinate offsets, but also the 4 coordinates
            for the anchor and the 4 variance values.
        """
        # Tile the anchor boxes for each predictor layer across all batch items.
        boxes_batch = []
        for boxes in self.boxes_list:
            # Prepend one dimension to `self.boxes_list` to account for the batch size and tile it along.
            # The result will be a 5D tensor of shape `(batch_size, feature_map_height, feature_map_width, n_boxes, 4)`
            boxes = np.expand_dims(boxes, axis=0)
            boxes = np.tile(boxes, (batch_size, 1, 1, 1, 1))

            # Reshape the 5D tensor above into a 3D tensor of shape
            # `(batch, feature_map_height * feature_map_width * n_boxes, 4)`. The resulting order of the
            # tensor content will be identical to the order obtained from the reshaping operation in our
            # Keras model (we are using the TensorFlow backend, and tf.reshape() and np.reshape() use the
            # same default index order.)
            boxes = np.reshape(boxes, (batch_size, -1, 4))
            boxes_batch.append(boxes)

        # Concatenate the anchor tensors from the individual layers to one.
        boxes_tensor = np.concatenate(boxes_batch, axis=1)

        # Create a template tensor to hold the one-hot class encoding of shape `(batch, #boxes, #classes)`.
        # It will contain all zeros for now, the classes will be set in the matching process that follows.
        classes_tensor = np.zeros((batch_size, boxes_tensor.shape[1], self.n_classes))

        # Create a tensor to contain the variances. This tensor has the same shape as `boxes_tensor` and
        # simply contains the same 4 variance value for every position in the last axis.
        variances_tensor = np.zeros_like(boxes_tensor)
        variances_tensor += self.variances

        # Concatenate the classes, boxes and variances tensors to get the final template for y_encoded.
        # We also need anchor tensor of the shape `boxes_tensor` as a space filter so that
        # `y_encoded_template` has the same shape as the SSD model output tensor. The content of this
        # tensor is irrelevant, we'll just use `boxes_tensor` a second time.
        y_encoded_template = np.concatenate((classes_tensor, boxes_tensor, boxes_tensor, variances_tensor), axis=2)

        if diagnostics:
            return y_encoded_template, self.centers_diag, self.wh_list_diag, self.steps_diag, self.offsets_diag
        else:
            return y_encoded_template


class DegenerateBoxError(Exception):
    """
    An exception class to be raised if degenerate boxes are being detected.
    """
    pass
