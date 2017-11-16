from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join
from image_reader import read_image
from tqdm import tqdm

import deep_neural_network as network
import pandas as pd
import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)
READ_DATA_FROM_FILE = True


def main(unused_argv):
    data_dict = pd.read_csv('./data/data_dict.csv')
    # Loading the data
    train_data = data_dict[data_dict["type"] == "train"][["file_name", "breed_name"]].copy()
    test_data = data_dict[data_dict["type"] == "test"][["file_name", "breed_name"]].copy()
    labels = data_dict["breed_name"].unique()
    label_mapping = {}

    count = 0
    for label in labels:
        label_mapping[label] = count
        count += 1
    del count

    # Due to CPU and memory constraint using less number of training data
    train_data = train_data.sample(2000)
    test_data = test_data.sample(500)

    x_train = x_test = None
    y_train = y_test = []

    if READ_DATA_FROM_FILE:
        # Loading data from file
        x_train = np.load("data/train_data_x.npy")
        y_train = np.load("data/train_data_y.npy")
        x_test = np.load("data/test_data_x.npy")
        y_test = np.load("data/test_data_y.npy")
    else:
        with tqdm(total=train_data.shape[0], desc="Reading train data") as pbar:
            for row in train_data.itertuples():
                folder_name = row.file_name.split("_")[0] + "-" + row.breed_name
                img_data = read_image(join("./data", "Images", folder_name, "%s.jpg" % row.file_name), (100, 100))
                if x_train is None:
                    x_train = [img_data]
                else:
                    x_train = np.append(x_train, [img_data], axis=0)
                y_train = np.append(y_train, label_mapping[row.breed_name])
                pbar.update(1)

        with tqdm(total=test_data.shape[0], desc="Reading test data") as pbar:
            for row in test_data.itertuples():
                folder_name = row.file_name.split("_")[0] + "-" + row.breed_name
                img_data = read_image(join("./data", "Images", folder_name, "%s.jpg" % row.file_name), (100, 100))
                if x_test is None:
                    x_test = [img_data]
                else:
                    x_test = np.append(x_test, [img_data], axis=0)

                y_test = np.append(y_test, label_mapping[row.breed_name])
                pbar.update(1)

        # Writing the train and test data to a file
        np.save("data/train_data_x.npy", x_train)
        np.save("data/train_data_y.npy", y_train)
        np.save("data/test_data_x.npy", x_test)
        np.save("data/test_data_y.npy", y_test)

    # Creating estimator
    classifier = tf.estimator.Estimator(
        model_fn=network.create_cnn_model,
        model_dir="/tmp/model_data"
    )

    # Setting up logging for predictions
    tensors_to_log = {
        "probabilities": "softmax_tensor"
    }
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log,
        every_n_iter=100
    )

    # Training the model
    print("Training the model...")
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": x_train},
        y=y_train,
        batch_size=100,
        num_epochs=None,
        shuffle=True)
    classifier.train(
        input_fn=train_input_fn,
        steps=2000,
        hooks=[logging_hook])

    # Evaluate the model and print results
    print("Evaluating the model...")
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": x_test},
        y=y_test,
        num_epochs=1,
        shuffle=False)
    eval_results = classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)


if __name__ == "__main__":
    tf.app.run()
