from generator.data_generator import DataGenerator

# 1: Instantiate two `DataGenerator` objects: One for training, one for validation.

# Optional: If you have enough memory, consider loading the images into memory for the reasons explained above.

train_dataset = DataGenerator()
val_dataset = DataGenerator()

# 2: Parse the image and label lists for the training and validation datasets.


# Images
images_dir = '/Users/mike/Downloads/udacity_driving_datasets'

# Ground truth
train_labels_filename = '/Users/mike/Downloads/udacity_driving_datasets/labels_train.csv'
val_labels_filename = '/Users/mike/Downloads/udacity_driving_datasets/labels_val.csv'

train_dataset.parse_csv(images_dir=images_dir,
                        labels_filename=train_labels_filename,
                        input_format=['image_name', 'xmin', 'xmax', 'ymin', 'ymax', 'class_id'],
                        include_classes='all')

val_dataset.parse_csv(images_dir=images_dir,
                      labels_filename=val_labels_filename,
                      input_format=['image_name', 'xmin', 'xmax', 'ymin', 'ymax', 'class_id'],
                      include_classes='all')

train_dataset.create_hdf5_dataset(file_path='dataset/dataset_udacity_traffic_train.h5',
                                  resize=False,
                                  variable_image_size=True,
                                  verbose=True)

val_dataset.create_hdf5_dataset(file_path='dataset/dataset_udacity_traffic_val.h5',
                                resize=False,
                                variable_image_size=True,
                                verbose=True)
