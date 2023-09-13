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
from tensorflow.keras.models import load_model, Model
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

    # Path variables
    params['data_path'] = args.data_path
    params['model_path'] = args.model_path
    params['results_path'] = args.results_path
    params['info_file_path'] = args.info_file_path

     # Miscellaneous variables
    params['batch_size'] = 8
    params['dim'] = (224, 224)
    params['n_channels'] = 3
    params['n_classes'] = 7
    params['shuffle'] = True

    return params


def build_generator(params, instance_ids):

    generator = DataGenerator2D(
        params['data_path'],
        instance_ids,
        params['labels'],
        batch_size=params['batch_size'],
        dim=params['dim'],
        n_channels=params['n_channels'],
        n_classes=params['n_classes'],
        shuffle=params['shuffle']
    )

    return generator


def predict_on_batch_and_store(params, model, generator, instances_info):

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
    print('\n[{}] Predicting and storing data using the trained model (size={})'.format(
        datetime.now().strftime('%Y-%m-%d %H:%M:%S'), len(partition_ids)))

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

    os.makedirs('{}'.format(results_path), exist_ok=True)
    partition_info.to_csv('{}/viewpoint_prediction_info.csv'.format(results_path), index=False)

    print('[{}] Done!'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))


def main(args):
    # Set Keras backend session with correct configurations
    set_backend_session()

    # Get parameters
    params = get_params(args)

    # Load model
    model = load_model(params['model_path'])

    # Load data
    instances_data = pd.read_csv(params['info_file_path'])
    # The Diagnosis column is not really used, but as this should be executed with preprocessed_rhd_frames_info
    # the referenced column is needed to create the params['labels']
    params['labels'] = pd.Series(instances_data['Diagnosis'].values, index=instances_data['File Name']).to_dict()

    instances_ids = list(instances_data['File Name'].values)
    generator = build_generator(params, instances_ids)

    predict_on_batch_and_store(params, model, generator, instances_data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict echocardiogram viewpoints.')

    parser.add_argument('-dp', '--data-path', type=str, dest='data_path',
                        help='Path where the preprocessed dataset can be found.')
    parser.add_argument('-mp', '--model-path', type=str, dest='model_path',
                        help='Path to the file that consists of the trained model.')
    parser.add_argument('-rp', '--results-path', type=str, dest='results_path',
                        help='Path where results can be saved to.')
    parser.add_argument('-if', '--info-file-path', type=str, dest='info_file_path',
                        help='Path to the file that contains classification information for each instance that should be predicted.')

    main(parser.parse_args())
