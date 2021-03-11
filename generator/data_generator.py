#!/usr/bin/env python3
"""
@project: object_detection
@file: data_generator
@time: 2021/3/9
 
@function:
"""
import pickle
import os
import numpy as np
from tqdm import tqdm, trange
import sys
from PIL import Image
import h5py
import csv
import cv2
import warnings
from encoder_decoder.input_encoder import SSDInputEncoder
import sklearn.utils
from utils.boxes_validation import BoxFilter
from copy import deepcopy
import inspect

PROCESSED_IMAGES = 'processed_images'
ENCODED_LABELS = 'encoded_labels'
MATCHED_ANCHORS = 'matched_anchors'
PROCESSED_LABELS = 'processed_labels'
FILENAMES = 'filenames'
IMAGE_IDS = 'image_ids'
EVALUATION_NEUTRAL = 'evaluation_neutral'
INVERSE_TRANSFORM = 'inverse_transform'
ORIGINAL_IMAGES = 'original_images'
ORIGINAL_LABELS = 'original_labels'


class DatasetError(Exception):
    """
    An exception class to be raised if anything is wrong with the dataset.
    """
    pass


class DegenerateBatchError(Exception):
    """
    An exception class to be raised if a generated batch ends up being degenerate,
    e.g. if a generated batch is empty.
    """


class DataGenerator:
    """
    A generator to generate batches of samples and corresponding labels.
    """

    def __init__(self,
                 load_images_into_memory=False,
                 hdf5_dataset_path=None,
                 filenames=None,
                 filenames_type='text',
                 images_dir=None,
                 labels=None,
                 image_ids=None,
                 eval_neutral=None,
                 labels_output_format=('class_id', 'xmin', 'ymin', 'xmax', 'ymax'),
                 verbose=True):
        """
        Arguments:
            load_images_into_memory (bool): If `True`, the entire dataset will be loaded into memory.
                This enables noticeably faster data generation than loading batches of images into memory.
            hdf5_dataset_path (string or list or None): The full file path of an HDF5 file that contains a dataset
                in the format that the `create_hdf5_dataset()` method produces. If you load an HDF5 dataset,
                you don't need to use any of the parser methods anymore, the HDF5 dataset already contains
                all relevant data.
            filenames (string or list): `None` or either a Python list/tuple or a string representing a
                filepath. If a list/tuple is passed, it must contain the file names (full paths) of the
                images to be used. If a filepath string is passed, it must point either to
                (1) a pickled file containing a list/tuple as described above. In this case the
                `filenames_type` argument must be set to `pickle`.
                (2) a text file. Each line of the text file contains the file name (basename of the file
                only, not the full directory path) to one image and nothing else. In this case the
                `filenames_type` argument must be set to `text` and you must pass the path to the
                directory that contains the images in `images_dir`.
            filenames_type (string): In case a string is passed for `filenames`, this indicates what type
                of file `filenames` is. It can be either `pickle` for a pickled file or `text` for a
                plain text file.
            images_dir (string): In case a text file is passed for `filenames`, the full paths to images
                will be composed from `images_dir` and the names in the text file. If `filenames_type`
                is not `text`, then this argument is irrelevant.
            labels (string or list): `None` or either a Python list/tuple or a string representing the
                path to a pickled file containing a list/tuple. The list/tuple must contain NumPy arrays
                that represent the labels of the dataset.
            image_ids (string or list): `None` or either a Python list/tuple or a string representing
                the path to a pickled file containing a list/tuple. The list/tuple must contain the image
                IDs of the images in the dataset.
            eval_neutral (string or list): `None` or either a Python list/tuple or a string representing
                the path to a pickled file containing a list/tuple. The list/tuple must contain for each
                image a list that indicates for each ground truth object in the image whether that object
                is supposed to be treated as neutral during an evaluation.
            labels_output_format (list): A list of five strings representing the desired order of the
                five items class_id, xmin, ymin, xmax, ymax in the generated ground truth data (if any).
            verbose (bool): If `True`, prints out the progress for some constructor operations that may
                take a bit longer.
        """
        self.labels_output_format = labels_output_format
        # For internal use
        self.labels_format = {
            'class_id': labels_output_format.index('class_id'),
            'xmin': labels_output_format.index('xmin'),
            'ymin': labels_output_format.index('ymin'),
            'xmax': labels_output_format.index('xmax'),
            'ymax': labels_output_format.index('ymax')
        }
        # If we haven't loaded anything yet, the dataset size is zero.
        self.dataset_size = 0
        self.load_images_into_memory = load_images_into_memory
        # The only way that this list will not stay `None` is
        # if `load_images_into_memory` is `True`.
        self.images = None

        # `self.filenames` is a list containing all filenames of the images (full path).
        # This list is one of the outputs of the parser methods.
        # In case you are loading an HDF5 dataset, this list will be `None`.
        if filenames is not None:
            if isinstance(filenames, (list, tuple)):
                self.filenames = filenames
            elif isinstance(filenames, str):
                if filenames_type == 'pickle':
                    with open(filenames, 'rb') as fh:
                        self.filenames = pickle.load(fh)
                elif filenames_type == 'text':
                    with open(filenames) as fh:
                        self.filenames = [os.path.join(images_dir, line.strip()) for line in fh]
                else:
                    raise ValueError("'filenames_type' can be either 'text' or 'pickle'.")
            else:
                raise ValueError("'filenames' must be either a Python list/tuple or "
                                 "a string representing a filepath (to a pickled or text file). "
                                 f"The value you passed is {filenames}.")

            self.dataset_size = len(self.filenames)
            self.dataset_indices = np.arange(self.dataset_size, dtype=np.int32)

            # Cash loaded memory into `self.images`.
            if load_images_into_memory:
                self.images = []
                if verbose:
                    # Show verbose process information
                    it = tqdm(self.filenames, desc='Loading images into memory', file=sys.stdout)
                else:
                    it = self.filenames
                for filename in it:
                    with Image.open(filename) as image:
                        self.images.append(np.array(image, dtype=np.uint8))
        else:
            self.filenames = None

        # In case ground truth is available, `self.labels` is a list containing for each image
        # a list (or NumPy array) of the ground truth bounding boxes for that images.
        if labels is not None:
            if isinstance(labels, (list, tuple)):
                self.labels = labels
            elif isinstance(labels, str):
                with open(labels, 'rb') as fh:
                    self.labels = pickle.load(fh)
            else:
                raise ValueError("'labels' must be either a Python list/tuple or "
                                 "a string representing the path to a pickled file "
                                 "containing a list/tuple. The value you passed is "
                                 f"{labels}.")
        else:
            self.labels = None

        if image_ids is not None:
            if isinstance(image_ids, (list, tuple)):
                self.image_ids = image_ids
            elif isinstance(image_ids, str):
                with open(image_ids, 'rb') as fh:
                    self.image_ids = pickle.load(fh)
            else:
                raise ValueError("'image_ids' must be either a Python list/tuple or "
                                 "a string representing the path to a pickled file "
                                 "containing a list/tuple. The value you passed is "
                                 f"{image_ids}.")
        else:
            self.image_ids = None

        if eval_neutral is not None:
            if isinstance(eval_neutral, (list, tuple)):
                self.eval_neutral = eval_neutral
            elif isinstance(eval_neutral, str):
                with open(eval_neutral, 'rb') as fh:
                    self.eval_neutral = pickle.load(fh)
            else:
                raise ValueError("'eval_neutral' must be either a Python list/tuple or "
                                 "a string representing the path to a pickled tile "
                                 "containing a list/tuple. The value you passed is "
                                 f"{eval_neutral}.")
        else:
            self.eval_neutral = None

        if hdf5_dataset_path is not None:
            self.hdf5_dataset_path = hdf5_dataset_path
            self.load_hdf5_dataset(verbose=verbose)
        else:
            self.hdf5_dataset = None

    def load_hdf5_dataset(self, verbose=True):
        """
         Loads an HDF5 dataset that is in the format that the `create_hdf5_dataset()` method
        produces.

        :param verbose:
        :return:
        """
        self.hdf5_dataset = h5py.File(self.hdf5_dataset_path, 'r')
        self.dataset_size = len(self.hdf5_dataset['images'])
        # Instead of shuffling the HDF5 dataset or images in memory, we will shuffle this index list.
        self.dataset_indices = np.arange(self.dataset_size, dtype=np.int32)

        if self.load_images_into_memory:
            self.images = []
            if verbose:
                tr = trange(self.dataset_size, desc='Loading images into memory', file=sys.stdout)
            else:
                tr = range(self.dataset_size)

            for i in tr:
                self.images.append(self.hdf5_dataset['images'][i].reshape(self.hdf5_dataset['image_shapes'][i]))

        if self.hdf5_dataset.attrs['has_labels']:
            self.labels = []
            labels = self.hdf5_dataset['labels']
            label_shapes = self.hdf5_dataset['label_shapes']
            if verbose:
                tr = trange(self.dataset_size, desc='Loading labels', file=sys.stdout)
            else:
                tr = range(self.dataset_size)
            for i in tr:
                self.labels.append(labels[i].reshape(label_shapes[i]))

        if self.hdf5_dataset.attrs['has_image_ids']:
            self.image_ids = []
            image_ids = self.hdf5_dataset['image_ids']
            if verbose:
                tr = trange(self.dataset_size, desc='Loading image IDs', file=sys.stdout)
            else:
                tr = range(self.dataset_size)
            for i in tr:
                self.image_ids.append(image_ids[i])

        if self.hdf5_dataset.attrs['has_eval_neutral']:
            self.eval_neutral = []
            eval_neutral = self.hdf5_dataset['eval_neutral']
            if verbose:
                tr = trange(self.dataset_size, desc='Loading evaluation neutrality annotations', file=sys.stdout)
            else:
                tr = range(self.dataset_size)
            for i in tr:
                self.eval_neutral.append(eval_neutral[i])

    def parse_csv(self,
                  images_dir,
                  labels_filename,
                  input_format,
                  include_classes='all',
                  random_sample=False,
                  ret=False,
                  verbose=True):
        """
        Arguments:
            images_dir (str): The path to the directory that contains the images.
            labels_filename (str): The filepath to a CSV file that contains one ground truth
                bounding box per line and each line contains the following six items:
                image file name, class ID, xmin, xmax, ymin, ymax. The six items do not have
                to be in a specific order, but they must be the first six columns of each
                line. The order of these items in the CSV file must be specified in `input_format`.
                The class ID is an integer greater than zero. Class ID 0 is reserved for the
                background class. `xmin` and `xmax` are the left-most and right-most absolute
                horizontal coordinates of the box, `ymin` and `ymax` are the top-most and
                bottom-most absolute vertical coordinates of the box. The image name is expected
                to be just the name of the image file without the directory path at which the
                image is located.
            input_format (list): A list of six strings representing the order of the six items:
                image file name, class ID, xmin, xmax, ymin, ymax in the input CSV file. The
                expected strings are 'image_name', 'xmin', 'xmax, 'ymin', 'ymax', 'class_id'.
            include_classes (list, str): Either 'all' or a list of integers containing the class IDs
                that are to be included in the dataset. If 'all', all ground truth boxes will be
                included in the dataset.
            random_sample (float): Either `False` or a float in [0, 1]. If this is `False`, the
                full dataset will be used by the generator. If this is a float, a randomly
                sampled fraction of the dataset will be used, where `random_sample` is the
                fraction of the dataset to be used. The fraction refers to the number of images,
                not to the number of boxes.
            ret (bool): Whether or not to return the outputs of the parser.
            verbose (bool): If `True`, prints out the progress for operations that may take
                a bit longer.
        """
        # Set class members.
        self.images_dir = images_dir
        self.labels_filename = labels_filename
        self.input_format = input_format
        self.include_classes = include_classes

        # Prerequisites
        if self.labels_filename is None:
            raise ValueError("'labels_filename' have not been set yet.")
        if self.input_format is None:
            raise ValueError("'input_format' have not been set yet.")

        # Erase data that might have been parsed before.
        self.filenames = []
        self.image_ids = []
        self.labels = []

        data = []

        with open(self.labels_filename, newline='') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            next(csv_reader)  # Skip the header row.
            # For every line (i.e. for every bounding box) in the CSV file
            for row in csv_reader:
                if (self.include_classes == 'all' or
                        int(row[self.input_format.index('class_id')].strip()) in self.include_classes):
                    # Store the box class and coordinates
                    box = []
                    # Select the image name column in the input format and append its content to `box`
                    box.append(row[self.input_format.index('image_name')].strip())
                    # The elements are class_id, xmin, xmax, ymin, ymax
                    for element in self.labels_output_format:
                        # Select the respective column in the input format and append it to `box`.
                        box.append(int(row[self.input_format.index(element)].strip()))
                    data.append(box)
        # The data needs to be sorted,
        # otherwise the next step won't give the correct result.
        data = sorted(data)

        current_file = data[0][0]
        # The image ID will be the portion of the image name before the first dot.
        current_image_id = data[0][0].split('.')[0]
        current_labels = []
        add_to_dataset = False

        for i, box in enumerate(data):
            # If this box (i.e. this line in the csv file) belongs to the current image file
            if box[0] == current_file:
                current_labels.append(box[1:])
                # If this is the last line of the CSV file.
                if i == len(data) - 1:
                    if random_sample:
                        # Draw samples from a uniform distribution.
                        p = np.random.uniform(0, 1)
                        if p >= (1 - random_sample):
                            self.labels.append(np.stack(current_labels, axis=0))
                            self.filenames.append(os.path.join(self.images_dir, current_file))
                            self.image_ids.append(current_image_id)
                    else:
                        self.labels.append(np.stack(current_labels, axis=0))
                        self.filenames.append(os.path.join(self.images_dir, current_file))
                        self.image_ids.append(current_image_id)
            else:  # If this box belongs to a new image file.
                if random_sample:
                    p = np.random.uniform(0, 1)
                    if p >= 1 - random_sample:
                        self.labels.append(np.stack(current_labels, axis=0))
                        self.filenames.append(os.path.join(self.images_dir, current_file))
                        self.image_ids.append(current_image_id)
                else:
                    self.labels.append(np.stack(current_labels, axis=0))
                    self.filenames.append(os.path.join(self.images_dir, current_file))
                    self.image_ids.append(current_image_id)

                # Reset the labels list because this is a new file.
                current_labels = []
                current_file = box[0]
                current_image_id = box[0].split('.')[0]
                current_labels.append(box[1:])

                if i == len(data) - 1:
                    if random_sample:
                        # Draw samples from a uniform distribution.
                        p = np.random.uniform(0, 1)
                        if p >= (1 - random_sample):
                            self.labels.append(np.stack(current_labels, axis=0))
                            self.filenames.append(os.path.join(self.images_dir, current_file))
                            self.image_ids.append(current_image_id)
                    else:
                        self.labels.append(np.stack(current_labels, axis=0))
                        self.filenames.append(os.path.join(self.images_dir, current_file))
                        self.image_ids.append(current_image_id)

        self.dataset_size = len(self.filenames)
        self.dataset_indices = np.arange(self.dataset_size, dtype=np.int32)

        # Cache into memory
        if self.load_images_into_memory:
            self.images = []
            if verbose:
                it = tqdm(self.filenames, desc='Loading images into memory', file=sys.stdout)
            else:
                it = self.filenames
            for filename in it:
                with Image.open(filename) as image:
                    self.images.append(np.array(image, dtype=np.uint8))

        if ret:
            return self.images, self.filenames, self.labels, self.image_ids

    def create_hdf5_dataset(self,
                            file_path='dataset.h5',
                            resize=False,
                            variable_image_size=True,
                            verbose=True):
        """
        Converts the currently loaded dataset into a HDF5 file.

        This HDF5 file contains all images as uncompressed arrays in a contiguous block of memory,
        which allows for them to be loaded faster. Such an uncompressed dataset, however, may take
        up considerably more space on your hard drive than the sum of the source images in a
        compressed format such as JPG or PNG.

        It is recommended that you always convert the dataset into an HDF5 dataset if you
        have enough hard drive space since loading from an HDF5 dataset accelerates the data
        generation noticeably.

        Note that you must load a dataset (e.g. via one of the parse methods) before creating an
        HDF5 dataset from it.

        Arguments:
            file_path (str): The full file path under which to store the HDF5 dataset.
            resize (tuple): `False` or a 2-tuple `(height, width)` that represents the target
                size for the image. All images in the dataset will be resized to this target
                size before they will be written to the HDF5 file. If `False`, no resizing
                will be performed.
            variable_image_size (bool): The only purpose of this argument is that its value
                will be stored in the HDF5 dataset in order to be able to quickly find out
                whether the images in the dataset all have the same size or not.
            verbose (bool):
        """
        self.hdf5_dataset_path = file_path
        dataset_size = len(self.filenames)

        # Create the HDF5 file.
        hdf5_dataset = h5py.File(file_path, 'w')

        # Create a few attributes that tell us what this dataset contains.
        # The dataset will obviously always contain images, but maybe it will
        # also contain labels, image IDs, etc.
        hdf5_dataset.attrs.create(name='has_labels', data=False, shape=None, dtype=np.bool_)
        hdf5_dataset.attrs.create(name='has_image_ids', data=False, shape=None, dtype=np.bool_)
        hdf5_dataset.attrs.create(name='has_eval_neutral', data=False, shape=None, dtype=np.bool_)

        # It's useful to be able to quickly check whether the images in a dataset all
        # have the same size or not, so add a boolean attribute for that.
        if variable_image_size and not resize:
            hdf5_dataset.attrs.create(name='variable_image_size', data=True, shape=None, dtype=np.bool_)
        else:
            hdf5_dataset.attrs.create(name='variable_image_size', data=False, shape=None, dtype=np.bool_)

        # Create the dataset in which the images will be stored as flattened arrays.
        # This allows us to store images of variable size.
        hdf5_images = hdf5_dataset.create_dataset(name='images',
                                                  shape=(dataset_size,),
                                                  maxshape=(None),
                                                  dtype=h5py.special_dtype(vlen=np.uint8))

        # Create the dataset that will hold the image heights, widths and channels that
        # we need in order to reconstruct the images from the flattened arrays later.
        hdf5_image_shapes = hdf5_dataset.create_dataset(name='image_shapes',
                                                        shape=(dataset_size, 3),
                                                        maxshape=(None, 3),
                                                        dtype=np.int32)

        if self.labels is not None:
            # Create the dataset in which the labels will be stored as flattened arrays.
            hdf5_labels = hdf5_dataset.create_dataset(name='labels',
                                                      shape=(dataset_size,),
                                                      maxshape=(None),
                                                      dtype=h5py.special_dtype(vlen=np.int32))
            # Create the dataset that will hold the dimension of the labels arrays for
            # each image so that we can restore the labels from the flattened arrays later.
            hdf5_label_shapes = hdf5_dataset.create_dataset(name='label_shape',
                                                            shape=(dataset_size, 2),
                                                            maxshape=(None, 2),
                                                            dtype=np.int32)
            hdf5_dataset.attrs.modify(name='has_labels', value=True)

        if self.image_ids is not None:
            hdf5_image_ids = hdf5_dataset.create_dataset(name='image_ids',
                                                         shape=(dataset_size,),
                                                         maxshape=(None),
                                                         dtype=h5py.special_dtype(vlen=str))
            hdf5_dataset.attrs.modify(name='has_image_ids', value=True)

        if self.eval_neutral is not None:
            hdf5_eval_neutral = hdf5_dataset.create_dataset(name='eval_neutral',
                                                            shape=(dataset_size,),
                                                            maxshape=(None),
                                                            dtype=h5py.special_dtype(vlen=np.bool_))
            hdf5_dataset.attrs.modify(name='has_eval_neutral', value=True)

        if verbose:
            tr = trange(dataset_size, desc='Creating HDF5 dataset', file=sys.stdout)
        else:
            tr = range(dataset_size)

        for i in tr:
            # Store the image.
            with Image.open(self.filenames[i]) as image:
                image = np.array(image, dtype=np.uint8)
                # Make sure all images end up having three channels.
                if image.ndim == 2:  # white black
                    image = np.stack([image] * 3, axis=-1)
                elif image.ndim == 3:
                    if image.shape[2] == 1:  # white black
                        image = np.concatenate([image] * 3, axis=-1)
                    elif image.shape[2] == 4:  # r,g,b,alpha
                        image = image[:, :, :3]

                if resize:
                    image = cv2.resize(image, dsize=(resize[1], resize[0]))

                # Flatten the image array and write it to the images dataset.
                hdf5_images[i] = image.reshape(-1)
                # Write the image's shape to the image shapes dataset.
                hdf5_image_shapes[i] = image.shape

            # Store the ground truth if we have any.
            if self.labels is not None:
                labels = np.asarray(self.labels[i])
                hdf5_labels[i] = labels.reshape(-1)
                hdf5_label_shapes[i] = labels.shape

            # Store the image ID if we have one.
            if self.image_ids is not None:
                hdf5_image_ids[i] = self.image_ids[i]

            # Store the evaluation-neutrality annotations if we have any.
            if self.eval_neutral is not None:
                hdf5_eval_neutral[i] = self.eval_neutral[i]

        hdf5_dataset.close()

        # For immediately use.
        self.hdf5_dataset = h5py.File(file_path, 'r')
        self.hdf5_dataset_path = file_path
        self.dataset_size = len(self.hdf5_dataset['images'])
        # Instead of shuffling the HDF5 dataset, we will shuffle this index list.
        self.dataset_indices = np.arange(self.dataset_size, dtype=np.int32)

    def generate(self,
                 batch_size=32,
                 shuffle=True,
                 transformations=[],
                 label_encoder=None,
                 returns={'processed_images', 'encoded_labels'},
                 keep_images_without_gt=False,
                 degenerate_box_handling='remove'
                 ):
        """
        Arguments:
            batch_size: The size of the batches to be generated.
            shuffle: Whether or not to shuffle the dataset before each pass.
                This option should always be `True` during training, but it can be useful to turn shuffling off
                for debugging or if you're using the generator for prediction.
            transformations (list): A list of transformations that will be applied to the images and labels
                in the given order. Each transformation is a callable that takes as input an image (as a Numpy array)
                and optionally labels (also as a Numpy array) and returns an image and optionally labels in the same
                format.
            label_encoder (callable): Only relevant if labels are given. A callable that takes as input the
                labels of a batch (as a list of Numpy arrays) and returns some structure that represents those labels.
                The general use case for this is to convert labels from their input format to a format that a given
                object detection model needs as its training targets.
            returns (set): A set of strings that determines what outputs the generator yields. The generator's output
                is always a tuple that contains the outputs specified in this set and only those. If an output is not
                available, it will be `None`. The output tuple can contain the following outputs according to the
                specified keyword strings:
                * 'processed_images': An array containing the processed images. Will always be in the outputs, so
                    it doesn't matter whether or not you include this keyword in the set.
                * "encoded_labels': The encoded labels tensor. Will always be in the outputs if a label encoder is
                    given, so it doesn't matter whether or not you include this keyword in the set if you pass a
                    label encoder.
                * 'matched_anchors': Only available if `labels_encoder` is an `SSDInputEncoder` object. The same as
                    `encoded_labels` but containing anchor box coordinates for all matched anchor boxes instead of
                    ground truth coordinates. This can be useful to visualize what anchor boxes are being matched to
                    each ground truth box. Only available in training mode.
                * 'precessed_labels': The processed, but not yet encoded labels. This is a list that contains for
                    each batch image a NumPy array with all ground truth boxes for that image. Only available if ground
                    truth is available.
                * 'filenames': A list containing the filenames (full path) of the images in the batch.
                * 'image_ids': A list containing the integer IDs of the images in the batch. Only available if there
                    are image IDs available.
                * 'evaluation_neutral': A nested list of lists of booleans. Each list contains `True` or `False` for
                    every ground truth bounding box of the respective image depending on whether that bounding box is
                    supposed to be evaluation_neutral (`True`) or not (`False`). May return `None` if there exists no
                    such concept for a given dataset.
                * 'inverse_transform': A nested list that contains a list of "inverter" functions for each item in the
                    batch. These inverter functions take (predicted) labels for an image as input and apply the inverse
                    of the transformations that were applied to the original image to them. This makes it possible to
                    make the model predictions on a transformed image then convert these predictions back to the
                    original image. This is mostly relevant for evaluation: If you want to evaluate your model on a
                    dataset with varying image sizes, then you are forced to transform the images somehow (e.g. by
                    resizing or cropping) to make them all the same size. Your model will then predict boxes for those
                    transformed images, but for the evaluation you will need predictions with respect to the original
                    image, not with the transformed images. This means you will have to transform the predicted box
                    coordinates back to the original image sizes. Note that for each image, the inverter functions for
                    that image need to be applied in the order in which they are given in the respective list for that
                    image.
                * 'original_images': A list containing the original images in the batch before any processing.
                * 'original_labels': A list containing the original ground truth for the images in this batch before
                    any processing. Only available if ground truth is available.
                The order of the outputs in the tuple is the order of the list of above. If `returns` contains a
                keyword for an output that is unavailable, the output omitted in the yielded tuples and a warining will
                be raised.
            keep_images_without_gt: If `False`, images for which there aren't any ground truth boxes before any
                transformations have be applied will be removed from the batch. If `True`, such images will be kept
                in the batch.
            degenerate_box_handling: How to handle degenerate boxes, which are boxes that have `xmax <= xmin` and/or
                `ymax <= ymin`. Degenerate boxes are sometimes be in the dataset, or non-degenerate boxes can become
                degenerate after they were processed by transformations. The generator checks for degenerate boxes
                after all transformations have been applied (if any), but before the labels were passed to the
                `label_encoder` (if one was given). Can be one of 'warn' or 'remove'. If 'warn', the generator will
                merely print a warning to let you know that there aer degenerate boxes in a batch. If 'remove',
                the generator will remove degenerate boxes form the batch silently.
        Yields:
            The next batch as a tuple of items as defined by the `returns` argument.
        """
        #######################
        # Exception and Warning
        #######################
        if self.dataset_size == 0:
            raise DatasetError('Cannot generate batches because you did dot load a dataset.')

        if self.labels is None:
            if any([ret in returns for ret in
                    [ORIGINAL_LABELS, PROCESSED_LABELS, ENCODED_LABELS, MATCHED_ANCHORS, EVALUATION_NEUTRAL]]):
                warnings.warn(f"Since no labels were given, none of '{ORIGINAL_LABELS}', '{PROCESSED_LABELS}' "
                              f"'{EVALUATION_NEUTRAL}', '{ENCODED_LABELS}', and '{MATCHED_ANCHORS}' "
                              f"are possible returns, but you set 'returns = {returns}'. The impossible returns "
                              f"will be 'None'")
            elif label_encoder in None:
                if any([ret in returns for ret in [ENCODED_LABELS, MATCHED_ANCHORS]]):
                    warnings.warn(f"Since no label encoder was given, '{ENCODED_LABELS}' and '{MATCHED_ANCHORS}' "
                                  f"aren't possible returns, but you set 'returns = {returns}'. The impossible returns"
                                  f"will be 'None'.")
            elif not isinstance(label_encoder, SSDInputEncoder):  # todo implementation
                if MATCHED_ANCHORS in returns:
                    warnings.warn(f"'label_encoder' is not an 'SSDInputEncoder' object, therefore '{MATCHED_ANCHORS}' "
                                  f"is not a possible return, but you set 'returns = {returns}'. The impossible "
                                  f"returns will be 'None'.")

        ################
        # Shuffle or not
        ################
        if shuffle:
            objects_to_shuffle = [self.dataset_indices]
            if self.filenames is not None:
                objects_to_shuffle.append(self.filenames)
            if self.labels is not None:
                objects_to_shuffle.append(self.labels)
            if self.image_ids is not None:
                objects_to_shuffle.append(self.image_ids)
            if self.eval_neutral is not None:
                objects_to_shuffle.append(self.eval_neutral)
            shuffle_objects = sklearn.utils.shuffle(*objects_to_shuffle)
            for i in range(len(objects_to_shuffle)):
                objects_to_shuffle[i][:] = shuffle_objects[i]

        #############################
        # Handing of degenerate boxes
        #############################
        if degenerate_box_handling == 'remove':
            box_filter = BoxFilter(check_overlap=False,
                                   check_min_area=False,
                                   check_degenerate=True,
                                   labels_format=self.labels_format)

        # Override the labels formats of all the transformation to make sure they are set correctly.
        if self.labels is not None:
            for transform in transformations:
                transform.label_format = self.labels_format

        #######################
        # Generate mini batches
        #######################
        current = 0
        while True:
            batch_x, batch_y = [], []

            if current >= self.dataset_size:
                current = 0

                # Maybe shuffle the dataset if a FULL pass over the dataset has finished.
                if shuffle:
                    objects_to_shuffle = [self.dataset_indices]
                    if self.filenames is not None:
                        objects_to_shuffle.append(self.filenames)
                    if self.labels is not None:
                        objects_to_shuffle.append(self.labels)
                    if self.image_ids is not None:
                        objects_to_shuffle.append(self.image_ids)
                    if self.eval_neutral is not None:
                        objects_to_shuffle.append(self.eval_neutral)
                    shuffle_objects = sklearn.utils.shuffle(*objects_to_shuffle)
                    for i in range(len(objects_to_shuffle)):
                        objects_to_shuffle[i][:] = shuffle_objects[i]
                #########################################################################
                # Get the images, (maybe) image IDs, (maybe) labels, etc. for this batch.
                #########################################################################
                # We prioritize our options in the following order:
                # 1) If we have the images already loaded in memory, get them from there.
                # 2) Else, if we have an HDF5 dataset, get the images from there.
                # 3) Else, if we have neither of the above, we'll have to load the individual image
                #     files from disk.
                batch_indices = self.dataset_indices[current:current + batch_size]
                # From memory
                if self.images is not None:
                    for i in batch_indices:
                        batch_x.append(self.images[i])
                    if self.filenames is not None:
                        batch_filenames = self.filenames[current:current + batch_size]
                    else:
                        batch_filenames = None
                # From hdf5
                elif self.hdf5_dataset is not None:
                    for i in batch_indices:
                        batch_x.append(self.hdf5_dataset['images'][i].reshape(self.hdf5_dataset['image_shapes'][i]))
                    if self.filenames is not None:
                        batch_filenames = self.filenames[current:current + batch_size]
                    else:
                        batch_filenames = None
                # From disk
                else:
                    # filenames cannot be None if loaded from disk
                    batch_filenames = self.filenames[current:current + batch_size]
                    for filename in batch_filenames:
                        with Image.open(filename) as image:
                            batch_x.append(np.array(image, dtype=np.uint8))

                # Get the labels for this batch (if there are any).
                if self.labels is not None:
                    # May do do some transformation later
                    batch_y = deepcopy(self.labels[current:current + batch_size])
                else:
                    batch_y = None

                if self.eval_neutral is not None:
                    batch_eval_neutral = self.eval_neutral[current:current + batch_size]
                else:
                    batch_eval_neutral = None

                if self.image_ids is not None:
                    batch_image_ids = self.image_ids[current:current + batch_size]
                else:
                    batch_image_ids = None

                # Keep a copy the original data
                if ORIGINAL_IMAGES in returns:
                    batch_original_images = deepcopy(batch_x)
                if ORIGINAL_LABELS in returns:
                    batch_original_labels = deepcopy(batch_y)

                # Jump to the next batch
                current += batch_size

                ######################################
                # Maybe perform image transformations.
                ######################################
                # In case we need to remove any images from the batch,
                # store their indices in this list.
                batch_items_to_remove = []
                batch_inverse_transforms = []

                for i in range(len(batch_x)):
                    if self.labels is not None:
                        # Convert to NumPy array if they aren't
                        batch_y[i] = np.array(batch_y[i])
                        # If this image has no ground truth boxes, maybe we don't want to keep it in the batch
                        if batch_y[i].size == 0 and not keep_images_without_gt:
                            batch_items_to_remove.append(i)
                            batch_inverse_transforms.append([])
                            continue

                    # Apply any image transformations we may have received.
                    if transformations:
                        inverse_transforms = []
                        for transform in transformations:
                            # Transform images and labels at the same time.
                            if self.labels is not None:
                                if (INVERSE_TRANSFORM in returns and
                                        'return_inverter' in inspect.signature(transform).parameters):
                                    batch_x[i], batch_y[i], inverse_transform = transform(batch_x[i], batch_y[i],
                                                                                          return_inverter=True)
                                    inverse_transforms.append(inverse_transform)
                                else:
                                    batch_x[i], batch_y[i] = transform(batch_x[i], batch_y[i])

                                # In case the transform failed to produce an output image,
                                # which is possible for some random transforms.
                                if batch_x[i] is None:
                                    batch_items_to_remove.append(i)
                                    batch_inverse_transforms.append([])
                                    continue
                            # Only transform the images.
                            else:
                                if (INVERSE_TRANSFORM in returns and
                                        'return_inverter' in inspect.signature(transform).parameters):
                                    batch_x[i], inverse_transform = transform(batch_x[i], return_inverter=True)
                                    inverse_transforms.append(inverse_transform)
                                else:
                                    batch_x[i] = transform(batch_x[i])
                        # Append the reversed inverse transforms
                        batch_inverse_transforms.append(inverse_transforms[::-1])

                    ################################################
                    # Check for degenerate boxes in this batch item.
                    ################################################
                    if self.labels is not None:
                        xmin = self.labels_format['xmin']
                        ymin = self.labels_format['ymin']
                        xmax = self.labels_format['xmax']
                        ymax = self.labels_format['ymax']

                        # If there exists degenerate box.
                        if (np.any(batch_y[i][:, xmax] - batch_y[i][:, xmin] <= 0) or
                                np.any(batch_y[i][:, ymax] - batch_y[i][ymin] <= 0)):
                            if degenerate_box_handling == 'warn':
                                warnings.warn(f"Detected degenerate ground truth bounding boxes for batch item "
                                              f"{i} with bounding boxes {batch_y[i]}, i.e. bounding boxes where"
                                              f"xmax <= xmin and/or ymax <= ymin. This could mean that your dataset "
                                              f"contains degenerate ground truth boxes, or that any image "
                                              f"transformations you apply might result in degenerate ground truth "
                                              f"boxes, or that you are parsing the ground truth in the wrong "
                                              f"coordinates format. Degenerate ground truth bounding boxes may leads "
                                              f"to NaN errors during the training.")
                            elif degenerate_box_handling == 'remove':
                                batch_y[i] = box_filter(batch_y[i])
                                if batch_y[i].size == 0 and not keep_images_without_gt:
                                    batch_items_to_remove.append(i)

                ############################################################
                # Remove any items we might not want to keep from the batch.
                ############################################################
                if batch_items_to_remove:
                    for i in sorted(batch_items_to_remove, reverse=True):
                        batch_x.pop(i)
                        batch_filenames.pop(i)

                        if self.labels is not None:
                            batch_y.pop(i)
                        if self.image_ids is not None:
                            batch_image_ids.pop(i)
                        if self.eval_neutral is not None:
                            batch_eval_neutral.pop(i)
                        if ORIGINAL_IMAGES in returns:
                            batch_original_images.pop(i)
                        if ORIGINAL_LABELS in returns and self.labels is not None:
                            batch_original_labels.pop(i)

                # CAUTION: Converting `batch_x` into an array will result in an empty batch if the images habve
                #          varying sizes or varying numbers of channels. At this point, all images must have the
                #          same size and the same number of channels.
                batch_x = np.array(batch_x)
                if batch_x.size == 0:
                    raise DegenerateBatchError("You produced an empty batch. This might be because the images "
                                               "in the batch vary in their size and/or number of channels. Note "
                                               "that after all transformations (if any were given) have been "
                                               "applied to all images in the batch, all images must be homogenous "
                                               "in size along all axes.")

                ################################################
                # If we have a label encoder, encode our labels.
                ################################################
                if label_encoder is not None or self.labels is None:
                    if MATCHED_ANCHORS in returns and isinstance(label_encoder, SSDInputEncoder):
                        batch_y_encoded, batch_matched_anchors = label_encoder(batch_y, diagnostics=True)
                    else:
                        batch_y_encoded = label_encoder(batch_y, diagnostics=False)
                        batch_matched_anchors = None
                else:
                    batch_y_encoded = None
                    batch_matched_anchors = None

                #####################
                # Compose the output.
                #####################
                ret = []
                if PROCESSED_IMAGES in returns:
                    ret.append(batch_x)
                if ENCODED_LABELS in returns:
                    ret.append(batch_y_encoded)
                if MATCHED_ANCHORS in returns:
                    ret.append(batch_matched_anchors)
                if PROCESSED_LABELS in returns:
                    ret.append(batch_y)
                if FILENAMES in returns:
                    ret.append(batch_filenames)
                if IMAGE_IDS in returns:
                    ret.append(batch_image_ids)
                if EVALUATION_NEUTRAL in returns:
                    ret.append(batch_eval_neutral)
                if INVERSE_TRANSFORM in returns:
                    ret.append(batch_inverse_transforms)
                if ORIGINAL_IMAGES in returns:
                    ret.append(batch_original_images)
                if ORIGINAL_LABELS in returns:
                    ret.append(batch_original_labels)

                yield ret

    def get_dataset_size(self):
        return self.dataset_size
