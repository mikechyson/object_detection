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
                 labels_output_format=('class_id', 'xmin', 'ymin', 'ymin', 'ymax'),
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
        pass  # todo

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
        pass  # todo

        def get_dataset_size(self):
            return self.dataset_size
