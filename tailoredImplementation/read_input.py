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

"""Binary for reading the images and cropping them based on the boundary
boxes"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from typing import List, Tuple

import tensorflow as tf
import pandas as pd
import xml.etree.ElementTree as ElementTree

StringArray = List[str]
IntArray = List[int]
TensorTuple = Tuple[tf.Tensor, tf.Tensor]
# Defining constants
IMAGE_SIZE = 224
NUM_CLASSES = 120
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000
DATA_DIR = os.path.join("..", "data")
IMAGE_DIR = os.path.join("..", "data", "Images")
ANNOTATION_DIR = os.path.join("..", "data", "Annotation")
IMAGE_EXTN = ".jpg"


class DogsRecordObject(object):
    def __init__(self, images=None, label=None, encoded_label=None):
        self.images = images
        self.label = label
        self.encoded_label = encoded_label


def _string_labels_to_one_hot(labels: StringArray) -> IntArray:
    label_dict = pd.DataFrame()
    label_unique = pd.Series(labels).unique().sort_values()
    for i in range(len(label_unique)):
        label_dict[label_unique[i]] = i
    encoded_labels = []
    for label in labels:
        encoded_labels.append(int(label_dict[label]))

    return encoded_labels


def _read_images(filename_queue: StringArray,
                 label_array: StringArray) -> DogsRecordObject:
    result = DogsRecordObject()
    image_reader = tf.WholeFileReader()
    key, value = image_reader.read(filename_queue)
    image_values = tf.image.decode_image(value)

    result.images = image_values
    result.label = label_array
    result.encoded_labels = _string_labels_to_one_hot(label_array)

    return result


def _crop_images(images: tf.Tensor, meta_data_path: str) -> tf.Tensor:
    meta_data = pd.read_csv(meta_data_path)
    bounding_box = {}
    bounding_box.offset_height = []
    bounding_box.offset_width = []
    bounding_box.target_height = []
    bounding_box.target_width = []
    for row in meta_data.itertuples():
        folder_name = "%s-%s" % (row.file_name.split('_')[0], row.breed_name)
        filename = os.path.join(ANNOTATION_DIR, folder_name, row.file_name)
        annotation_object = ElementTree.parse(filename)
        boundary_box = annotation_object.find("object").find("bndbox")
        xmin = int(boundary_box.find("xmin").text)
        xmax = int(boundary_box.find("xmax").text)
        ymin = int(boundary_box.find("ymin").text)
        ymax = int(boundary_box.find("ymax").text)

        bounding_box.offset_height.append(ymin)
        bounding_box.offset_width.append(xmin)
        bounding_box.target_height.append(ymax - ymin)
        bounding_box.target_width.append(xmax - xmin)

    cropped_image = tf.image.crop_to_bounding_box(images,
                                                  bounding_box.offset_height,
                                                  bounding_box.offset_width,
                                                  bounding_box.target_height,
                                                  bounding_box.target_width)
    resized_image = tf.image.resize_image_with_crop_or_pad(cropped_image,
                                                           IMAGE_SIZE,
                                                           IMAGE_SIZE)
    return resized_image


def _generate_image_and_label_batch(image: tf.Tensor,
                                    label: tf.Tensor,
                                    min_queue_examples: int,
                                    batch_size: int) -> TensorTuple:
    num_preprocess_threads = 16
    images, label_batch = tf.train.batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size
    )
    tf.summary.image("Images", images)

    return images, tf.reshape(label_batch, [batch_size])


def inputs(eval_data: bool, meta_data_path: str,
           batch_size: int) -> TensorTuple:
    filenames = []
    meta_data = pd.read_csv(meta_data_path)

    if not eval_data:
        meta_data = meta_data[meta_data["type"] == "train"][
            ["file_name", "breed_name"]].copy()
    else:
        meta_data = meta_data[meta_data["type"] == "test"][
            ["file_name", "breed_name"]].copy()

    for row in meta_data.itertuples():
        folder_name = "%s-%s" % (row.file_name.split('_')[0], row.breed_name)
        filename = os.path.join(IMAGE_DIR, folder_name, row.file_name +
                                IMAGE_EXTN)
        filenames.append(filename)

    for f in filenames:
        if not tf.gfile.Exists(f):
            raise FileNotFoundError("File \"%s\" not found" % f)

    filename_queue = tf.train.string_input_producer(filenames)
    read_input = _read_images(filename_queue, meta_data["breed_name"])
    cropped_resized_images = _crop_images(read_input.images,
                                          DATA_DIR + "data_dict.csv")

    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int((NUM_EXAMPLES_PER_EPOCH_FOR_EVAL if eval_data else
                              NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN) *
                             min_fraction_of_examples_in_queue)

    return _generate_image_and_label_batch(cropped_resized_images,
                                           read_input.encoded_label,
                                           min_queue_examples, batch_size)
