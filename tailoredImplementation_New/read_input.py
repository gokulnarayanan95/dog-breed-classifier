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
from os.path import join
import gc

import tensorflow as tf
from tensorflow.contrib import keras
import pandas as pd
import xml.etree.ElementTree as ElementTree
from tqdm import tqdm
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)

# Defining constants
IMAGE_SIZE = 224
NUM_CLASSES = 6
BATCH_SIZE = 64
DATA_DIR = join("..", "data")
MODEL_DATA_DIR = join("..", "tmp")
IMAGE_DIR = join("..", "data", "Images")
ANNOTATION_DIR = join("..", "data", "Annotation")
IMAGE_EXTN = ".jpg"


class PreProcessImages(object):
    def __init__(self, meta_data=join(DATA_DIR, "data_dict.csv"),
                 data_dir=DATA_DIR, images_dir=IMAGE_DIR,
                 image_size=IMAGE_SIZE, model_data_dir = MODEL_DATA_DIR,
                 annotations_dir=ANNOTATION_DIR, image_extn=IMAGE_EXTN,
                 batch_size=BATCH_SIZE):
        self.META_DATA = meta_data
        self.DATA_DIR = data_dir
        self.IMAGES_DIR = images_dir
        self.IMAGE_SIZE = image_size
        self.MODEL_DATA_DIR = model_data_dir
        self.ANNOTATIONS_DIR = annotations_dir
        self.IMAGE_EXTN = image_extn
        self.BATCH_SIZE = batch_size

    @staticmethod
    def _crop_image(image_tensor, bounding_box):
        cropped_image_tensor = tf.image.crop_to_bounding_box(
            image=image_tensor,
            offset_height=bounding_box["ymin"],
            offset_width=bounding_box["xmin"],
            target_height=(bounding_box["ymax"] - bounding_box["ymin"]),
            target_width=(bounding_box["xmax"] - bounding_box["xmin"])
        )
        return cropped_image_tensor

    def _read_image(self, image_meta, is_to_be_cropped):
        image_tools = keras.preprocessing.image
        image = image_tools.load_img(path=image_meta["path"],
                                     grayscale=False,
                                     target_size=None)
        image_binary = image_tools.img_to_array(image)
        image_tensor = tf.convert_to_tensor(image_binary, tf.float32)

        if is_to_be_cropped:
            image_tensor = self._crop_image(image_tensor,
                                            image_meta["bounding_box"])

        resized_image = tf.image.resize_image_with_crop_or_pad(
            image=image_tensor,
            target_height=self.IMAGE_SIZE,
            target_width=self.IMAGE_SIZE
        )
        return resized_image.eval()

    def _get_images_meta_with_annotations(self):
        meta_data = pd.read_csv(self.META_DATA)
        images_meta = []

        for row in meta_data.itertuples():
            folder_name = "%s-%s" % (
                row.file_name.split('_')[0], row.breed_name)
            filename = join(self.ANNOTATIONS_DIR, folder_name,
                            row.file_name)
            annotation_object = ElementTree.parse(filename)
            boundary_box = annotation_object.find("object").find("bndbox")

            images_meta.append({
                "file_name": row.file_name + self.IMAGE_EXTN,
                "path": join(self.IMAGES_DIR, folder_name,
                             row.file_name + self.IMAGE_EXTN),
                "class": row.breed_name,
                "bounding_box": {
                    "xmin": int(boundary_box.find("xmin").text),
                    "xmax": int(boundary_box.find("xmax").text),
                    "ymin": int(boundary_box.find("ymin").text),
                    "ymax": int(boundary_box.find("ymax").text)
                }
            })

        return images_meta

    def inputs(self):
        images_meta_data = self._get_images_meta_with_annotations()
        inputs = np.zeros((self.BATCH_SIZE, self.IMAGE_SIZE,
                           self.IMAGE_SIZE, 3),
                          np.float32)
        bin_files = []
        batch_number = 0

        with tqdm(total=len(images_meta_data),
                  desc="Reading train data") as pbar:
            for i, image_meta in enumerate(images_meta_data):
                if i % self.BATCH_SIZE == 0 or \
                        i == (len(images_meta_data) - 1):
                    np.save(join(self.MODEL_DATA_DIR,
                                 "image-data-batch-%s" % batch_number), inputs)
                    bin_files.append(join(self.MODEL_DATA_DIR,
                                          "image-data-batch-%s" %
                                          batch_number))
                    inputs = np.zeros((self.BATCH_SIZE, self.IMAGE_SIZE,
                                       self.IMAGE_SIZE, 3),
                                      np.float32)
                    batch_number += 1
                    gc.collect()

                image_data = self._read_image(image_meta=image_meta,
                                              is_to_be_cropped=True)
                inputs[(i % self.BATCH_SIZE) - 1] = image_data
                pbar.update(1)
        return bin_files


def main(argv=None):
    data_dict = pd.read_csv(join(DATA_DIR, "data_dict.csv"))
    selected_breed_list = list(
        data_dict.groupby('breed_name').count()
            .sort_values(by='file_name', ascending=False)
            .head(NUM_CLASSES).index)
    selected_data_dict = data_dict[
        (data_dict["breed_name"].isin(selected_breed_list))]
    selected_data_dict.to_csv(join(DATA_DIR, "selected_data_dict.csv"))
    process_input = PreProcessImages(meta_data=join(DATA_DIR,
                                                    "selected_data_dict.csv"))
    return process_input.inputs()


if __name__ == '__main__':
    sess = tf.InteractiveSession()
    tf.app.run()
