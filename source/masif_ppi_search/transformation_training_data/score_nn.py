import os
import tensorflow as tf
import numpy as np
from tensorflow import keras

os.environ["TF_USE_LEGACY_KERAS"] = "1"
tf.compat.v1.disable_v2_behavior()


"""
score_nn.py: Class to score protein complex alignments based on a pre-trained neural network (used for MaSIF-search's second stage protocol).
Freyr Sverrisson and Pablo Gainza - LPDI STI EPFL 2019
Released under an Apache License 2.0
"""


class ScoreNN:
    def __init__(self):
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        _ = tf.compat.v1.Session(config=config)

        np.random.seed(42)
        tf.compat.v1.random.set_random_seed(42)

        reg = tf.compat.v1.keras.regularizers.l2(l=0.0)
        model = tf.compat.v1.keras.models.Sequential()

        model.add(tf.compat.v1.layers.conv1d(filters=8, kernel_size=1, strides=1))
        model.add(tf.compat.v1.layers.batch_normalizatio)
        model.add(tf.compat.v1.layers.ReLU())
        model.add(
            tf.compat.v1.layers.conv1d(
                filters=16, kernel_size=1, strides=1, input_shape=(200, 3)
            )
        )
        model.add(tf.compat.v1.layers.batch_normalization())
        model.add(tf.compat.v1.layers.ReLU())
        model.add(tf.compat.v1.layers.conv1d(filters=32, kernel_size=1, strides=1))
        model.add(tf.compat.v1.layers.batch_normalization())
        model.add(tf.compat.v1.layers.ReLU())
        model.add(tf.compat.v1.layers.conv1d(filters=64, kernel_size=1, strides=1))
        model.add(tf.compat.v1.layers.batch_normalization())
        model.add(tf.compat.v1.layers.ReLU())
        model.add(tf.compat.v1.layers.conv1d(filters=128, kernel_size=1, strides=1))
        model.add(tf.compat.v1.layers.batch_normalization())
        model.add(tf.compat.v1.layers.ReLU())
        model.add(tf.compat.v1.layers.conv1d(filters=256, kernel_size=1, strides=1))
        model.add(tf.compat.v1.layers.batch_normalization())
        model.add(tf.compat.v1.layers.ReLU())
        model.add(tf.compat.v1.layers.global_average_pooling1d())
        model.add(
            tf.compat.v1.layers.dense(128, activation=tf.nn.relu, kernel_regularizer=reg)
        )
        model.add(tf.compat.v1.layers.dense(64, activation=tf.nn.relu, kernel_regularizer=reg))
        model.add(tf.compat.v1.layers.dense(32, activation=tf.nn.relu, kernel_regularizer=reg))
        model.add(tf.compat.v1.layers.dense(16, activation=tf.nn.relu, kernel_regularizer=reg))
        model.add(tf.compat.v1.layers.dense(8, activation=tf.nn.relu, kernel_regularizer=reg))
        model.add(tf.compat.v1.layers.dense(4, activation=tf.nn.relu, kernel_regularizer=reg))
        model.add(tf.compat.v1.layers.dense(2, activation="softmax"))

        opt = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-4)
        model.compile(
            optimizer=opt, loss="sparse_categorical_crossentropy", metrics=["accuracy"]
        )

        self.model = model
        self.restore_model()

    def restore_model(self):
        self.model.load_weights("models/nn_score/trained_model.hdf5")

    def train_model(self, features, labels, n_negatives, n_positives):
        callbacks = [
            tf.compat.v1.keras.callbacks.ModelCheckpoint(
            filepath="models/nn_score/{}.hdf5".format("trained_model"),
            save_best_only=True,
            monitor="val_loss",
            save_weights_only=True,
            ),
            tf.compat.v1.keras.callbacks.TensorBoard(
            log_dir="./logs/nn_score", write_graph=False, write_images=True
            ),
        ]
        _ = self.model.fit(
            features,
            labels,
            batch_size=32,
            epochs=50,
            validation_split=0.1,
            shuffle=True,
            class_weight={0: 1.0 / n_negatives, 1: 1.0 / n_positives},
            callbacks=callbacks,
        )

    def eval(self, features):
        y_test_pred = self.model.predict(features)
        return y_test_pred
