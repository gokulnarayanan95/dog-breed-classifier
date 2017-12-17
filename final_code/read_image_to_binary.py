# Copyright 2017 Raghavan Renganathan, Gokul Anandhanarayanan.
# All rights reserved.
#
# Licensed under the MIT License;
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://opensource.org/licenses/MIT
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

"""Convert all the image files from jpg to binary.
This file reads all the image files from the data dict containing all the
images' filename and their respective classes.

The images are read and are cropped based on the co-ordinates specified in
the annotation data.

The cropped images are stored along with their class name in the binary format.

First Byte - label value (One Hot encoded)
Second Byte till the end - Image value in the following order
    R values of all the pixels
    G values of all the pixels
    B values of all the pixels
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from PIL import Image
from os.path import join
import pandas as pd
from tqdm import tqdm
import xml.etree.ElementTree as ElementTree
from array import *

ONE_HOT_DICT = join("meta-data", "one-hot-dict.csv")
DATA_DICT = join("meta-data", "data_dict.csv")
IMAGES_DIR = join("data", "Images")
ANNOTATION_DIR = join("data", "Annotation")
BIN_DIR = join("binary_data")
TRAIN_BIN_DIR = join(BIN_DIR, "train")
TEST_BIN_DIR = join(BIN_DIR, "test")
IMAGE_SIZE = 224
IMAGE_EXTN = ".jpg"
IMAGES_PER_BIN = 256

NUM_CLASSES = 120


def _convert_images_to_binary():
    """Reads the images from the directory and converts them into CIFAR
    binary format and stores them in the directory specified.
    """
    print("Converting images into binary -> %s" % BIN_DIR)
    data_dict = pd.read_csv(DATA_DICT)

    selected_breed_list = list(
        data_dict.groupby('breed_name').count()
            .sort_values(by='file_name', ascending=False)
            .head(NUM_CLASSES).index)

    meta_data = data_dict[(data_dict["breed_name"].isin(
        selected_breed_list))].copy()

    # Covert labels into one-hot values
    one_hot_labels_dict = pd.DataFrame(columns=["class", "one_hot"])
    labels = meta_data["breed_name"]
    unique_labels = labels.unique()
    unique_labels.sort()
    for i in range(unique_labels.size):
        one_hot_labels_dict.append([unique_labels[i], i])
        meta_data.loc[meta_data["breed_name"] == unique_labels[i],
                      "one_hot_class"] = i
    one_hot_labels_dict.to_csv(ONE_HOT_DICT)

    image_data = array('B')

    train_data = meta_data[meta_data["type"] == "train"].copy()
    test_data = meta_data[meta_data["type"] == "test"].copy()

    with tqdm(total=train_data.shape[0],
              desc="Reading Train Images -> %s" % TRAIN_BIN_DIR) as pbar:
        i = 0
        batch_number = 0
        for row in train_data.itertuples():
            if (i % IMAGES_PER_BIN == 0 or i == train_data.shape[0] - 1) \
                    and i != 0:
                output_file = open(join(TRAIN_BIN_DIR, "data_batch_%d.bin" %
                                        batch_number),
                                   "wb")
                image_data.tofile(output_file)
                output_file.close()
                image_data = array('B')
                batch_number += 1

            folder_name = "%s-%s" % (
                row.file_name.split('_')[0], row.breed_name)

            # Reading the image
            image_file_path = join(IMAGES_DIR, folder_name, row.file_name +
                                   IMAGE_EXTN)
            image = Image.open(image_file_path)

            # Reading the annotation for getting the bounding box
            annotation_file_path = join(ANNOTATION_DIR, folder_name,
                                        row.file_name)
            annotation_object = ElementTree.parse(annotation_file_path)
            boundary_box = annotation_object.find("object").find("bndbox")
            xmin = int(boundary_box.find("xmin").text)
            xmax = int(boundary_box.find("xmax").text)
            ymin = int(boundary_box.find("ymin").text)
            ymax = int(boundary_box.find("ymax").text)

            # Cropping the image and resizing it to the standard size
            cropped_image = image.crop((xmin, ymin, xmax, ymax))
            resized_image = cropped_image.resize(
                (IMAGE_SIZE, IMAGE_SIZE)).load()

            # Append data to the binary array
            image_data.append(int(row.one_hot_class))

            for channel in range(0, 3):
                for x in range(0, IMAGE_SIZE):
                    for y in range(0, IMAGE_SIZE):
                        image_data.append(resized_image[x, y][channel])

            pbar.update(1)
            i += 1

    with tqdm(total=test_data.shape[0],
              desc="Reading Test Images -> %s" % TEST_BIN_DIR) as pbar:
        i = 0
        for row in test_data.itertuples():
            folder_name = "%s-%s" % (row.file_name.split('_')[0],
                                     row.breed_name)

            # Reading the image
            image_file_path = join(IMAGES_DIR, folder_name, row.file_name +
                                   IMAGE_EXTN)
            image = Image.open(image_file_path)

            # Reading the annotation for getting the bounding box
            annotation_file_path = join(ANNOTATION_DIR, folder_name,
                                        row.file_name)
            annotation_object = ElementTree.parse(annotation_file_path)
            boundary_box = annotation_object.find("object").find("bndbox")
            xmin = int(boundary_box.find("xmin").text)
            xmax = int(boundary_box.find("xmax").text)
            ymin = int(boundary_box.find("ymin").text)
            ymax = int(boundary_box.find("ymax").text)

            # Cropping the image and resizing it to the standard size
            cropped_image = image.crop((xmin, ymin, xmax, ymax))
            resized_image = cropped_image.resize(
                (IMAGE_SIZE, IMAGE_SIZE)).load()

            # Append data to the binary array
            image_data.append(int(row.one_hot_class))

            for channel in range(0, 3):
                for x in range(0, IMAGE_SIZE):
                    for y in range(0, IMAGE_SIZE):
                        image_data.append(resized_image[x, y][channel])

            pbar.update(1)
            i += 1

        output_file = open(join(TEST_BIN_DIR, "data_batch.bin"), "wb")
        image_data.tofile(output_file)
        output_file.close()


def _create_dirs(delete_if_exists=False):
    """Create the directories for storing the generated binaries if they do
    not exist
    """
    print("Creating dirs...")
    if tf.gfile.Exists(BIN_DIR):
        if delete_if_exists:
            tf.gfile.DeleteRecursively(BIN_DIR)
    tf.gfile.MakeDirs(BIN_DIR)
    tf.gfile.MakeDirs(TRAIN_BIN_DIR)
    tf.gfile.MakeDirs(TEST_BIN_DIR)


def _check_for_data_files():
    """Check for the data files to be present in the mentioned directories
    """
    if not tf.gfile.Exists(DATA_DICT):
        raise ValueError("Data dictionary file is not located in %s" %
                         DATA_DICT)
    if not tf.gfile.Exists(IMAGES_DIR):
        raise  ValueError("Images not found in %s" % IMAGES_DIR)
    if not tf.gfile.Exists(ANNOTATION_DIR):
        raise ValueError("Annotation not found in %s" % ANNOTATION_DIR)


def check_for_binary_data(force_create=False):
    """Check for the binary files to be present in the directory specified.
    If it is not present, this function will read the images and generate
    the binary files for the model.

    Args:
          force_create: A boolean to specify whether to delete all the old
          binaries and to create new ones
    """
    _check_for_data_files()
    if force_create:
        _create_dirs(delete_if_exists=True)
        _convert_images_to_binary()
    else:
        if not tf.gfile.Exists(BIN_DIR):
            _create_dirs()
            _convert_images_to_binary()
        else:
            if not (tf.gfile.Exists(TRAIN_BIN_DIR) or
                    tf.gfile.Exists(TEST_BIN_DIR)):
                _create_dirs(delete_if_exists=True)
