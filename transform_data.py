import os
import numpy
from PIL import Image
from scipy.ndimage import zoom
from itertools import izip


def get_images_as_numpy_arrays(image_files):
    return (numpy.array(Image.open(image)) for image in image_files)


def get_sex_labels_as_binary(filenames):
    return map(lambda filename: 0 if filename.split("_")[-1] == "M" else 1,
               map(lambda filename: filename.split(".")[0],
                   filenames))


def get_all_image_files_from_path(path):
    return (os.path.join(path, filename) for filename
            in os.listdir(path)
            if os.path.isfile(os.path.join(path, filename)) and
            filename.split(".")[1] == "jpg")


def resize_images_to_uniform_size(images):
    return (zoom(image,
                 (1,
                  200.0 / image.shape[1],
                  200.0 / image.shape[2]),
                 order=0)
            for image in images)


def transpose_images_to_channel_row_column(images):
    return (image.transpose(2, 0, 1) for image in images)


def transform_image_values_to_proper_range(images):
    return ((image.astype(numpy.float32) / 255.0) for image in images)


def get_input_images_and_ouput_labels(path_to_images="data/"):
    labelFiles = get_all_image_files_from_path(path_to_images)
    imageFiles = get_all_image_files_from_path(path_to_images)
    labels = get_sex_labels_as_binary(labelFiles)
    image_arrays = transform_image_values_to_proper_range(
        resize_images_to_uniform_size(
            transpose_images_to_channel_row_column(
                get_images_as_numpy_arrays(imageFiles))))
    return izip(image_arrays, labels)

train_data = get_input_images_and_ouput_labels("data_old/")
for X, Y in train_data:
    print X.shape, Y
