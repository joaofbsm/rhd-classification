"""Classification of echocardiogram viewpoints using machine learning models"""

import argparse
import gc
import itertools
import json
import os
import random
import sys
from datetime import datetime
from math import log, ceil
from random import randint
from time import time, ctime

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn import metrics, model_selection
from sklearn.utils import class_weight
from tensorflow.keras import applications
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Activation, BatchNormalization, Dense, Dropout, Flatten, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from data_generator import DataGenerator2D


# Random generators initialization
random.seed(123456)
np.random.seed(123456)
tf.set_random_seed(123456)

# Explicitly choose matplotlib backend
matplotlib.use('Agg')

# Floating point precision
pd.set_option('precision', 4)
np.set_printoptions(precision=4)


def set_backend_session():

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    tf.keras.backend.set_session(sess)


def get_params(args):

    params = {}

    params['nn_type'] = args.nn_type

    # Path variables
    params['data_path'] = args.data_path
    params['models_path'] = args.models_path
    params['results_path'] = args.results_path
    params['splits_file_path'] = args.splits_file_path

    # Hyperparameters
    params['batch_size'] = args.batch_size
    params['num_epochs'] = args.num_epochs
    params['learning_rate'] = args.learning_rate
    params['lr_decay'] = args.lr_decay

     # Miscellaneous variables
    params['dim'] = (224, 224)
    params['n_channels'] = 3
    params['n_classes'] = 7
    params['shuffle'] = True
    params['num_splits'] = args.num_splits

    return params


def get_splits(df, num_splits):
    splits = []
    for i in range(num_splits):
        splits.append({
            'train': list(df[df['split{}'.format(i)] == 'train']['File Name'].values),
            'val': list(df[df['split{}'.format(i)] == 'val']['File Name'].values),
            'test': list(df[df['split{}'.format(i)] == 'test']['File Name'].values),
        })

    return splits


def build_generators(params, partitions):

    generators = {}

    generators['train'] = DataGenerator2D(
        params['data_path'],
        partitions['train'],
        params['labels'],
        batch_size=params['batch_size'],
        dim=params['dim'],
        n_channels=params['n_channels'],
        n_classes=params['n_classes'],
        shuffle=params['shuffle']
    )
    generators['val'] = DataGenerator2D(
        params['data_path'],
        partitions['val'],
        params['labels'],
        batch_size=params['batch_size'],
        dim=params['dim'],
        n_channels=params['n_channels'],
        n_classes=params['n_classes'],
        shuffle=params['shuffle']
    )
    generators['test'] = DataGenerator2D(
        params['data_path'],
        partitions['test'],
        params['labels'],
        batch_size=params['batch_size'],
        dim=params['dim'],
        n_channels=params['n_channels'],
        n_classes=params['n_classes'],
        shuffle=params['shuffle']
    )

    return generators


def get_model_name(params):

    creation_datetime = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Name to be used for the model
    model_name = "{}_bs{}_ne{}_lr{}_ld{}_ts{}".format(
        params['nn_type'],
        params['batch_size'],
        params['num_epochs'],
        params['learning_rate'],
        params['lr_decay'],
        creation_datetime
    )

    return model_name


def build_model(params):

    if params['nn_type'] == 'vgg16':
        model_base = applications.VGG16(weights='imagenet', include_top=False, input_tensor=Input(shape=(224, 224, 3)))
        for layer in model_base.layers:
            layer.trainable = False

        model_head = model_base.output
        model_head = Flatten(name='flatten')(model_head)
        model_head = Dense(512, activation='relu')(model_head)
        model_head = Dropout(0.5)(model_head)
        model_head = Dense(params['n_classes'], activation='softmax')(model_head)

        model = Model(inputs=model_base.input, outputs=model_head)

        optimizer = Adam(lr=params['learning_rate'])

        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    else:
        sys.exit('Unknown neural network type.')

    return model


def train_model(params, model, model_name, training_generator, validation_generator, classes_weights, save_model=False):

    print(
        '[{}] Training with batch_size={} and classes_weights={}'.format(
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            params['batch_size'],
            classes_weights
        )
    )

    # Callbacks
    early_stop_cb = EarlyStopping(monitor='val_loss', patience=10)
    callbacks = [early_stop_cb]
    
    # Train model
    model.fit_generator(generator=training_generator,
                        validation_data=validation_generator,
                        epochs=params['num_epochs'],
                        class_weight=classes_weights,
                        callbacks=callbacks,
                        max_queue_size=100,
                        workers=16,
                        use_multiprocessing=True)

    # Save model if needed
    if save_model:
        path_to_model = '{}/{}/trained_models'.format(params['models_path'], params['nn_type'])
        os.makedirs(path_to_model, exist_ok=True)
        model.save('{}/{}.h5'.format(path_to_model, model_name))
        print(
            '[{}] Model saved to: {}/{}.h5'.format(
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                path_to_model,
                model_name
            )
        )

    print('[{}] Done training!'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))


def predict_on_batch_and_store(params, model, generator, partition_name, model_name, instances_info, split_index):

    # In case of an empty (test) generator when traning the final classifier 
    if not generator.instances_ids:
        return

    batch_size = params['batch_size']
    dims = params['dim']
    n_channels = params['n_channels']
    n_classes = params['n_classes']

    partition_ids = [generator.instances_ids[k] for k in generator.instances_indexes]

    num_batches = np.ceil(len(partition_ids) / params['batch_size']).astype(int)

    results_path = params['results_path']

    # Debugging prints
    print('\n[{}] Predicting and storing data using the trained model using the "{}" data (size={})'.format(
        datetime.now().strftime('%Y-%m-%d %H:%M:%S'), partition_name, len(partition_ids)))

    # Iterate over number of batches predicting each batch
    predictions = np.empty((len(partition_ids), n_classes))
    for idx in range(num_batches):
        batch_start = idx * batch_size
        batch_end = min((idx + 1) * batch_size, len(partition_ids))
        current_partition_ids = partition_ids[batch_start:batch_end]

        X = np.empty((len(current_partition_ids), *dims, n_channels))

        for i, ID in enumerate(current_partition_ids):
            x = np.load('{}/{}.npz'.format(generator.data_path, ID))['frames']
            X[i,] = x

        predictions[batch_start:batch_end] = model.predict_on_batch(X)

    partition_info = pd.DataFrame({'File Name': partition_ids}).merge(instances_info, on='File Name')
    partition_info['pred_proba'] = np.array(predictions).tolist()
    partition_info['pred'] = partition_info['pred_proba'].apply(lambda x: np.argmax(x))

    os.makedirs('{}/{}/pred/'.format(results_path, model_name), exist_ok=True)
    partition_info.to_csv(
        '{}/{}/pred/{}_split{}_info.csv'.format(results_path, model_name, partition_name, split_index), index=False
    )

    print('[{}] Done!'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))


def try_params(params, instances_data):

    model_name = get_model_name(params)

    splits = get_splits(instances_data, params['num_splits'])

    for split_index in range(params['num_splits']):
        partitions = splits[split_index]
        print('\nProcessing split {}/{}'.format(split_index + 1, len(splits)))
        print('{}\n'.format('=' * (18 + len(str(split_index + 1)) + len(str(len(splits))))))

        model = build_model(params)
        save_model = True if split_index == 0 else False

        # Calculate weights for each class in training data
        viewpoint_classes = instances_data[instances_data['File Name'].isin(partitions['train'])]['Viewpoint Class']
        classes_weights = dict(zip(np.unique(viewpoint_classes),
                                   class_weight.compute_class_weight('balanced', np.unique(viewpoint_classes), viewpoint_classes)))

        generators = build_generators(params, partitions)

        train_model(params, model, model_name, generators['train'], generators['val'], classes_weights,
                            save_model)

        # Predict
        for partition_name, generator in generators.items():
            # Prevent ID shuffling after prediction
            generator.shuffle = False

            predict_on_batch_and_store(
                params,
                model,
                generator,
                partition_name,
                model_name,
                instances_data,
                split_index
            )

        # Clear memory as much as possible between executions
        tf.keras.backend.clear_session()
        del model
        del generators
        gc.collect()

        # Setup new TensorFlow/Keras session
        set_backend_session()

    print('\n[{}] Results saved to {}/{}/'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), params['results_path'],
                                                  model_name))


def main(args):
    # Set Keras backend session with correct configurations
    set_backend_session()

    # Get parameters
    params = get_params(args)

    # Load data
    instances_data = pd.read_csv(params['splits_file_path'])
    params['labels'] = pd.Series(instances_data['Viewpoint Class'].values, index=instances_data['File Name']).to_dict()

    try_params(params, instances_data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train and predict echocardiogram viewpoint.')

    parser.add_argument('-nn', '--nn-type', type=str, choices=['vgg16'], dest='nn_type',
                        help='Neural network type that images should serve as input.')
    parser.add_argument('-dp', '--data-path', type=str, dest='data_path',
                        help='Path where the preprocessed dataset can be found.')
    parser.add_argument('-mp', '--models-path', type=str, dest='models_path',
                        help='Path where models can be saved or loaded from.')
    parser.add_argument('-rp', '--results-path', type=str, dest='results_path',
                        help='Path where results can be saved or loaded from.')
    parser.add_argument('-sf', '--splits-file-path', type=str, dest='splits_file_path',
                        help='Path to the file that contains classification information for each instance and experimental splits.')
    parser.add_argument('-lr', '--learning-rate', type=float, dest='learning_rate',
                        help='Learning rate for the machine learning model.')
    parser.add_argument('-ld', '--lr-decay', type=float, dest='lr_decay', default=0.0,
                        help='Decay rate for the learning rate. Occurs at each epoch.')
    parser.add_argument('-ep', '--num-epochs', type=int, dest='num_epochs',
                        help='Number of epochs to train the machine learning model.')
    parser.add_argument('-bs', '--batch-size', type=int, default='8', dest='batch_size',
                        help='Size of the batch to be feed to the network.')
    parser.add_argument('-ns', '--num-splits', type=int, default='10', dest='num_splits',
                        help='Number of data splits to train and test on.')

    main(parser.parse_args())
