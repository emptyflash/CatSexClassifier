import os
from math import ceil, floor
from numpy import pad, array, float32
from PIL import Image
from scipy.ndimage import zoom
from itertools import izip


def get_images_as_numpy_arrays(image_files):
    # return (array(Image.open(image)) for image in image_files)
    for image in image_files:
        try:
            yield array(Image.open(image))
        except IOError as ex:
            print "IOError reading image: " + str(ex) + ": " + image
            continue


def get_sex_labels_as_binary(filenames):
    return map(lambda filename: 0 if filename.split("_")[-1] == "M" else 1,
               map(lambda filename: filename.split(".")[0],
                   filenames))


def get_number_of_image_files_in_path(path="data/"):
    return sum(1 for filename in get_all_image_files_from_path(path))


def get_all_image_files_from_path(path):
    return (os.path.join(path, filename) for filename
            in os.listdir(path)
            if os.path.isfile(os.path.join(path, filename)) and
            filename.split(".")[1] == "jpg")


def resize_images_to_uniform_size_on_one_axis(images):
    return (zoom(image,
                 (1,
                  96.0 / max(image.shape[1], image.shape[2]),
                  96.0 / max(image.shape[1], image.shape[2])),
                 order=0)
            for image in images)


def pad_images_to_be_unifrom_size(images):
    return (pad(image,
                ((0, 0),
                 (ceil((96 - image.shape[1]) / 2.0),
                  floor((96 - image.shape[1]) / 2.0)),
                 (ceil((96 - image.shape[2]) / 2.0),
                  floor((96 - image.shape[2]) / 2.0))),
                mode="edge") for image in images)


def transpose_images_to_channel_row_column(images):
    # return (image.transpose(2, 0, 1) for image in images)
    for image in images:
        try:
            yield image.transpose(2, 0, 1)
        except ValueError as ex:
            print "ValueError transposing image: " + str(ex) + ": " + str(image.shape)
            continue


def transform_image_values_to_proper_range(images):
    return ((image.astype(float32) / 255.0) for image in images)


def get_input_images_and_ouput_labels(path_to_images="data/"):
    labelFiles = get_all_image_files_from_path(path_to_images)
    imageFiles = get_all_image_files_from_path(path_to_images)
    labels = get_sex_labels_as_binary(labelFiles)
    image_arrays = transform_image_values_to_proper_range(
        pad_images_to_be_unifrom_size(
            resize_images_to_uniform_size_on_one_axis(
                transpose_images_to_channel_row_column(
                    get_images_as_numpy_arrays(imageFiles)))))
    return izip(image_arrays, labels)
