import itertools

import numpy as np
import tensorflow as tf

from random import randint


class DataGenerator2D(tf.keras.utils.Sequence):
    """
    Generates data for Keras.
    """

    def __init__(self, data_path, instances_ids, labels, batch_size=32, dim=(224, 224), n_channels=3,
                 n_classes=7, shuffle=True):
        self.data_path = data_path
        self.instances_ids = instances_ids
        self.labels = labels
        self.batch_size = batch_size
        self.dim = dim
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle

        # Generate initial instances_indexes
        self.on_epoch_end()

    def on_epoch_end(self):
        """
        Updates indexes after each epoch.
        """

        self.instances_indexes = np.arange(len(self.instances_ids))
        if self.shuffle == True:
            np.random.shuffle(self.instances_indexes)

    def __len__(self):
        """
        Denotes the number of batches per epoch.
        """

        return int(np.ceil(len(self.instances_ids) / self.batch_size))

    def __getitem__(self, index):
        """
        Generate one batch of data.
        """

        # Generate indexes of the batch
        indexes = self.instances_indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        batch_ids = [self.instances_ids[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(batch_ids)

        return X, y

    def __data_generation(self, batch_ids):
        """
        Generates data containing batch_size File Names.
        """

        # Initialization
        cur_batch_size = len(batch_ids)
        X = np.empty((cur_batch_size, *self.dim, self.n_channels))
        y = np.empty(cur_batch_size, dtype=int)

        # Generate data
        for i, instance_id in enumerate(batch_ids):
            X[i] = np.load('{}/{}.npz'.format(self.data_path, instance_id))['frames']

            # Store correct class
            y[i] = self.labels[instance_id]

        return X, tf.keras.utils.to_categorical(y, num_classes=self.n_classes)


class DataGenerator3D(tf.keras.utils.Sequence):
    """
    Generates data for Keras.
    """

    def __init__(self, data_path, instances_ids, labels, batch_size=32, dim=(32, 32, 32), n_channels=3,
                 n_classes=3, shuffle=True, crop_strategy='center'):
        self.data_path = data_path
        self.instances_ids = instances_ids
        self.labels = labels
        self.batch_size = batch_size
        self.dim = dim
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.crop_strategy = crop_strategy

        # Generate initial instances_indexes
        self.on_epoch_end()

    def on_epoch_end(self):
        """
        Updates indexes after each epoch.
        """

        self.instances_indexes = np.arange(len(self.instances_ids))
        if self.shuffle == True:
            np.random.shuffle(self.instances_indexes)

    def __len__(self):
        """
        Denotes the number of batches per epoch.
        """

        return int(np.ceil(len(self.instances_ids) / self.batch_size))

    def __getitem__(self, index):
        """
        Generate one batch of data.
        """

        # Generate indexes of the batch
        indexes = self.instances_indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        batch_ids = [self.instances_ids[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(batch_ids)

        return X, y

    def __data_generation(self, batch_ids):
        """
        Generates data containing batch_size samples.
        """

        # Initialization
        cur_batch_size = len(batch_ids)
        X = np.empty((cur_batch_size, *self.dim, self.n_channels))
        y = np.empty(cur_batch_size, dtype=int)

        # Generate data
        for i, instance_id in enumerate(batch_ids):
            x = np.load('{}/{}.npz'.format(self.data_path, instance_id))['frames']

            X[i, ] = x

            # Store correct class
            y[i] = self.labels[instance_id]

        return X, tf.keras.utils.to_categorical(y, num_classes=self.n_classes)
