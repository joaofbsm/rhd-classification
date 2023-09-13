"""Classification of Rheumatic Heart Disease using machine learning models"""

import argparse
import gc
import json
import os
import random
from datetime import datetime
from math import log, ceil
from time import time, ctime

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn import metrics, model_selection
from sklearn.utils import class_weight
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import multi_gpu_model

import models
import optimizers
import utils
from data_filter_types import UndersamplingFiltering, DopplerFiltering
from data_generator import DataGenerator


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
    """
    Configure Keras TensorFlow backend to allocate GPU memory efficiently in shared environments.
    """

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    tf.keras.backend.set_session(sess)


def get_data_filtering_options(args):
    """
    Create a chain with all filters to be applied.
    """

    doppler_filter_dict = {
        'only_with_doppler': DopplerFiltering.ONLY_WITH_DOPPLER,
        'only_without_doppler': DopplerFiltering.ONLY_WITHOUT_DOPPLER
    }
    undersampling_filter_dict = {
        'normal_definite': (UndersamplingFiltering.NORMAL_DEFINITE, ['Normal', 'Definite RHD']),
        'normal_borderline': (UndersamplingFiltering.NORMAL_BORDERLINE, ['Normal', 'Borderline RHD']),
        'normal_definite_borderline': (UndersamplingFiltering.NORMAL_DEFINITE_BORDERLINE,
                                       ['Normal', 'Borderline RHD', 'Definite RHD']),
        'normal_diseasecarrier': (UndersamplingFiltering.NORMAL_DISEASECARRIER, ['Normal', 'Disease Carrier']),
    }

    doppler_filtering = doppler_filter_dict.get(args.doppler_filtering, DopplerFiltering.NONE)
    undersampling_filtering, possible_classes = undersampling_filter_dict.get(args.undersampling_filtering,
                                                                              (UndersamplingFiltering.NONE,
                                                                               ['Normal',
                                                                                'Borderline RHD',
                                                                                'Definite RHD']))

    return doppler_filtering, undersampling_filtering, possible_classes


def get_params(args):
    """
    Receive the input arguments and generate the set of parameters.
    """

    params = {}

    params['nn_type'] = args.nn_type

    # Path variables
    params['data_path'] = args.data_path
    params['models_path'] = args.models_path
    params['results_path'] = args.results_path
    params['info_file_path'] = args.info_file_path

    # Instance filtering variables
    doppler_filtering, undersampling_filtering, possible_classes = get_data_filtering_options(args)
    params['doppler_filtering'] = doppler_filtering
    params['undersampling_filtering'] = undersampling_filtering
    params['possible_classes'] = possible_classes
    params['bound_undersampling_to'] = args.bound_undersampling_to
    params['bound_undersampling_to_value'] = args.bound_undersampling_to_value
    params['filter_exam_size'] = args.filter_exam_size
    params['aug_percentage'] = args.aug_percentage
    params['label_map'] = {
        'Normal': 0,
        'Borderline RHD': 1,
        'Definite RHD': 2 if undersampling_filtering != UndersamplingFiltering.NORMAL_DISEASECARRIER else 1
    }

    # Miscellaneous variables
    params['dim'] = (16, 112, 112)
    params['n_channels'] = 3
    params['n_classes'] = len(possible_classes)
    params['shuffle'] = True
    params['cross_validation'] = args.cross_validation
    params['random_search'] = args.random_search
    params['time_logs'] = args.time_logs
    params['multi_gpu'] = args.multi_gpu

    # Hyperparameters
    params['batch_size'] = args.batch_size
    params['num_epochs'] = args.num_epochs
    params['learning_rate'] = args.learning_rate
    params['lr_decay'] = args.lr_decay
    params['learn_visual_features'] = args.nn_type == 'c3d'
    params['crop_strategy'] = args.crop_strategy
    params['batch_normalization'] = args.batch_normalization

    return params


def filter_instances(params, instances_data):
    """
    Filter instances to be used according to filtering parameters.
    """

    # Filter out instances
    if params['filter_exam_size']:
        instances_data_gb = instances_data.groupby('Exam').count()
        condition = (10 <= instances_data_gb['Sample']) & (instances_data_gb['Sample'] <= 20)
        instances_data = instances_data[instances_data['Exam'].isin(instances_data_gb[condition].index)]

    # Filter augmented instances according to flag
    size_without_aug = len(instances_data[instances_data['Augmentation'].isnull()])
    size_aug_partition = int(size_without_aug * params['aug_percentage'])
    if size_aug_partition > 0:
        instances_data = pd.concat([instances_data[instances_data['Augmentation'].isnull()],
                                    instances_data[instances_data['Augmentation'].notnull()].sample(
                                        size_aug_partition,
                                        random_state=123456)])
    else:
        instances_data = instances_data[instances_data['Augmentation'].isnull()]

    # Convert string to class ID
    instances_data['Diagnosis'] = instances_data['Diagnosis'].map(params['label_map'])

    filtered_instances_data = instances_data
    chosen_rhd_exams = None

    # Doppler filtering
    if params['doppler_filtering'] == DopplerFiltering.ONLY_WITHOUT_DOPPLER:
        filtered_instances_data = filtered_instances_data.loc[filtered_instances_data['With_Doppler'] == False]
        chosen_rhd_exams = filtered_instances_data['Exam'].unique()
    elif params['doppler_filtering'] == DopplerFiltering.ONLY_WITH_DOPPLER:
        filtered_instances_data = filtered_instances_data.loc[filtered_instances_data['With_Doppler'] == True]
        chosen_rhd_exams = filtered_instances_data['Exam'].unique()

    # Undersampling filtering
    if params['undersampling_filtering'] == UndersamplingFiltering.NORMAL_DISEASECARRIER:
        negative_rhd_exams = filtered_instances_data[filtered_instances_data['Diagnosis'] == 0]['Exam'].unique()
        positive_rhd_exams = filtered_instances_data[filtered_instances_data['Diagnosis'] == 1]['Exam'].unique()

        bound_to_size = min(len(negative_rhd_exams), len(positive_rhd_exams))

        if params['bound_undersampling_to'] == 'normal':
            bound_to_size = len(negative_rhd_exams)
        elif (params['bound_undersampling_to'] == 'personalized'
              and params['bound_undersampling_to_value'] < bound_to_size):
            bound_to_size = params['bound_undersampling_to_value']

        # Selecting exams randomly to compose data
        chosen_negative_rhd_exams = np.random.choice(negative_rhd_exams, bound_to_size, replace=False)
        chosen_positive_rhd_exams = np.random.choice(positive_rhd_exams,
                                                     min(len(positive_rhd_exams), bound_to_size),
                                                     replace=False)
        chosen_rhd_exams = np.hstack([chosen_negative_rhd_exams, chosen_positive_rhd_exams])
    elif params['undersampling_filtering'] == UndersamplingFiltering.NONE:
        # If NONE, choose all exams from Doppler filtering
        chosen_rhd_exams = filtered_instances_data['Exam'].unique()
    else:
        exams_per_class = []

        # Normal
        exams_per_class.append(filtered_instances_data[filtered_instances_data['Diagnosis'] == 0]['Exam'].unique())
        # Borderline
        exams_per_class.append(filtered_instances_data[filtered_instances_data['Diagnosis'] == 1]['Exam'].unique())
        # Definite
        exams_per_class.append(filtered_instances_data[filtered_instances_data['Diagnosis'] == 2]['Exam'].unique())

        classes_to_use = {
            UndersamplingFiltering.NORMAL_DEFINITE_BORDERLINE: [0, 1, 2],
            UndersamplingFiltering.NORMAL_BORDERLINE: [0, 1],
            UndersamplingFiltering.NORMAL_DEFINITE: [0, 2]
        }
        usable_indexes = classes_to_use[params['undersampling_filtering']]

        if params['bound_undersampling_to'] == 'normal':
            bound_to_size = len(exams_per_class[0])
        elif params['bound_undersampling_to'] == 'borderline':
            bound_to_size = len(exams_per_class[1])
        elif params['bound_undersampling_to'] == 'definite':
            bound_to_size = len(exams_per_class[2])
        elif params['bound_undersampling_to'] == 'personalized':
            bound_to_size = params['bound_undersampling_to_value']
        else:
            bound_to_size = np.array([len(exams_per_class[i]) for i in usable_indexes]).min

        chosen_exams_per_class = []
        chosen_exams_per_class.append(np.random.choice(exams_per_class[0],
                                                       min(len(exams_per_class[0]), bound_to_size),
                                                       replace=False))
        chosen_exams_per_class.append(np.random.choice(exams_per_class[1],
                                                       min(len(exams_per_class[1]), bound_to_size),
                                                       replace=False))
        chosen_exams_per_class.append(np.random.choice(exams_per_class[2],
                                                       min(len(exams_per_class[2]), bound_to_size),
                                                       replace=False))

        chosen_rhd_exams = np.hstack([chosen_exams_per_class[i] for i in usable_indexes])

    # Select instances according to the chosen exams and the Doppler filtering
    filtered_instances_data = filtered_instances_data[filtered_instances_data['Exam'].isin(chosen_rhd_exams)]

    # Create final labels
    labels = pd.Series(filtered_instances_data['Diagnosis'].values, index=filtered_instances_data['Sample']).to_dict()

    return filtered_instances_data, labels


def get_splits(df):
    splits = []
    for i in range(10):
        splits.append({
            'train': list(df[df['split{}'.format(i)] == 'train']['Sample'].values),
            'val': list(df[df['split{}'.format(i)] == 'val']['Sample'].values),
            'test': list(df[df['split{}'.format(i)] == 'test']['Sample'].values),
        })

    return splits

def get_model_name(params):
    """
    Return the model name according to the hyperparameters and creation time.
    """

    creation_time = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Name to be used for the model
    model_name = "{}_nc{}_bs{}_ep{}_lr{}_ld{}_df{}_uf{}_bt{}_cr{}_bn{}_ts{}".format(
        params['nn_type'],
        params['n_classes'],
        params['batch_size'],
        params['num_epochs'],
        params['learning_rate'],
        params['lr_decay'],
        params['doppler_filtering'],
        params['undersampling_filtering'],
        params['bound_undersampling_to'],
        params['crop_strategy'],
        params['batch_normalization'],
        creation_time
    )

    return model_name


def build_c3d_model(params):
    """
    Build C3D models according to the parameters.
    """

    if params['nn_type'] == 'c3d_test':
        model = models.c3d_test()
    else:
        model = models.c3d(summary=False, learn_visual_features=params['learn_visual_features'])
        model.load_weights('{}/C3D/sports1M_weights_tf.h5'.format(params['models_path']))
        model = models.c3d_int_model(
            model=model,
            layer='fc8',
            batch_normalization=params['batch_normalization'],
            learn_visual_features=params['learn_visual_features'],
            backend='tf'
        )

    # Change the "head" of the network
    if params['batch_normalization']:
        model.pop()
    model.pop()
    model.pop()
    model.add(Dense(params['n_classes'], name='fc8'))
    if params['batch_normalization']:
        model.add(BatchNormalization())
    model.add(Activation('softmax'))

    optimizer = SGD(lr=params['learning_rate'], decay=params['lr_decay'])
    # optimizer = optimizers.RAdam(lr=params['learning_rate'], decay=params['lr_decay'])

    if params['multi_gpu'] >= 2:
        model = multi_gpu_model(model, gpus=params['multi_gpu'])

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model


def build_generators(params, partitions):
    """
    Create data generators to train and evaluate the model.
    """

    generators = {}

    generators['train'] = DataGenerator(
        params['data_path'],
        partitions['train'],
        params['labels'],
        batch_size=params['batch_size'],
        dim=params['dim'],
        n_channels=params['n_channels'],
        n_classes=params['n_classes'],
        shuffle=params['shuffle'],
        crop_strategy=params['crop_strategy']
    )
    generators['val'] = DataGenerator(
        params['data_path'],
        partitions['val'],
        params['labels'],
        batch_size=params['batch_size'],
        dim=params['dim'],
        n_channels=params['n_channels'],
        n_classes=params['n_classes'],
        shuffle=params['shuffle'],
        crop_strategy=params['crop_strategy']
    )
    generators['test'] = DataGenerator(
        params['data_path'],
        partitions['test'],
        params['labels'],
        batch_size=params['batch_size'],
        dim=params['dim'],
        n_channels=params['n_channels'],
        n_classes=params['n_classes'],
        shuffle=params['shuffle'],
        crop_strategy=params['crop_strategy']
    )

    return generators


def train_model(params, model, model_name, training_generator, validation_generator, classes_weights, save_model=False):
    """
    Train neural network model.
    """

    print(
        '[{}] Training with batch_size={} and classes_weights={}'.format(
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            params['batch_size'],
            classes_weights
        )
    )

    # Callbacks
    tensorboard_log_dir = '{}/{}/tb_logs'.format(params['results_path'], model_name)
    tensorboard_cb = TensorBoard(log_dir=tensorboard_log_dir,
                                 histogram_freq=0)
    early_stop_cb = EarlyStopping(monitor='val_loss', patience=5)
    callbacks = [early_stop_cb, tensorboard_cb]

    # Create time logs callbacks
    if params['time_logs']:

        path_to_logs = '{}/{}/time_logs'.format(params['results_path'], model_name)
        os.makedirs(path_to_logs, exist_ok=True)
        callbacks.append(utils.TimeLog(path_to_logs))

        print(
            '[{}] Time logs per batch per epoch will be saved to {}'.format(
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                path_to_logs
            )
        )

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


def predict_generator_and_store(params, model, generator, partition_name, model_name, instances_info, split_index):
    """
    Use the trained model to predict the labels of instances in a given partition (training, validation or test) and
    data split. The results are stored to be evaluated across all splits at once afterwards.
    """

    print(
        '\n[{}] Predicting and storing data using the trained model using the "{}" data (size={})'.format(
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            partition_name,
            len(generator.instances_ids)
        )
    )

    predictions = model.predict_generator(generator)

    partition_ids = [generator.instances_ids[k] for k in generator.instances_indexes]
    partition_info = pd.DataFrame({'Sample': partition_ids}).merge(instances_info, on='Sample')

    results_path = params['results_path']
    os.makedirs('{}/{}/pred/'.format(results_path, model_name), exist_ok=True)
    np.save(
        '{}/{}/pred/{}_split{}_pred.npy'.format(results_path, model_name, partition_name, split_index),
        np.array(predictions)
    )
    partition_info.to_csv(
        '{}/{}/pred/{}_split{}_info.csv'.format(results_path, model_name, partition_name, split_index),
        index=False
    )

    print('[{}] Done!'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))


def predict_on_batch_and_store(params, model, generator, partition_name, model_name, instances_info, split_index):
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

        # WARNING: If __data_generator is changed, you need to also change this function
        for i, ID in enumerate(current_partition_ids):
            x = np.load('{}/{}.npz'.format(generator.data_path, ID))['frames']
            if generator.crop_strategy == 'none':
                X[i,] = x
            else:
                X[i,] = x[:, 8:120, 20:132, :]  # Center crop

        predictions[batch_start:batch_end] = model.predict_on_batch(X)

    partition_info = pd.DataFrame({'Sample': partition_ids}).merge(instances_info, on='Sample')

    os.makedirs('{}/{}/pred/'.format(results_path, model_name), exist_ok=True)
    np.save('{}/{}/pred/{}_split{}_pred.npy'.format(results_path, model_name, partition_name, split_index),
            np.array(predictions))
    partition_info.to_csv(
        '{}/{}/pred/{}_split{}_info.csv'.format(results_path, model_name, partition_name, split_index), index=False)

    # Debugging prints
    print('[{}] Done!'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))


def get_confusion_matrix(y_true, y_pred):
    """
    Get default and normalized confusion matrices in the format of numpy arrays.
    """

    confusion_matrix = metrics.confusion_matrix(y_true, y_pred)
    normalized_confusion_matrix = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]

    return confusion_matrix, normalized_confusion_matrix


def get_confusion_matrix_mean_std(confusion_matrix, possible_classes):
    """
    Calculate the mean and standard deviation from all the confusion matrix in a cross-validation procedure.
    """

    str_cm = np.array([['', ''], ['', '']], dtype=object)
    mean_cm = np.mean(confusion_matrix, axis=0)
    std_cm = np.std(confusion_matrix, axis=0)

    for i in range(str_cm.shape[0]):
        for j in range(str_cm.shape[1]):
            str_cm[i][j] = '{:0.4g} ({:0.4g})'.format(mean_cm[i][j], std_cm[i][j])

    df = pd.DataFrame(mean_cm, index=possible_classes, columns=possible_classes)
    annot_df = pd.DataFrame(str_cm, index=possible_classes, columns=possible_classes)

    return df, annot_df


def save_confusion_matrix(confusion_matrix, annot_cm, results_path, model_name, partition_name, prediction_strategy,
                          normalized=False):
    """
    Save the given confusion matrix as a matplotlib plot with the correct information.
    """

    # Disable iteractive mode when there is no $DISPLAY to show (in processing-only machines)
    plt.ioff()

    plt.figure(figsize=(7, 5))
    sns.heatmap(confusion_matrix, annot=annot_cm, square=True, cmap='rocket_r', fmt='')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.xlabel('Predicted Classes', labelpad=15)
    plt.ylabel('Actual Classes', labelpad=15)
    plt.title('{} {}'.format(partition_name, prediction_strategy))

    if normalized:
        plt.savefig(
            '{}/{}/cm/{}_{}_normalized_cm.png'.format(results_path, model_name, partition_name, prediction_strategy),
            bbox_inches='tight',
            dpi=166
        )
    else:
        plt.savefig(
            '{}/{}/cm/{}_{}_cm.png'.format(results_path, model_name, partition_name, prediction_strategy),
            bbox_inches='tight',
            dpi=166
        )

    plt.close()


def evaluate_model(params, model_name, partition_name, num_splits):
    """
    Evaluate the model by partition type across all the data splits trained on. This calculates the accuracy in three
    different ways: by individual videos, majority vote of predictions across videos from a single exam and probability
    mean of predictions across videos from a single exam.

    Because we execute a 10-fold cross-validation, the results for both training and validation are not a mean, as all
    videos are evaluated exactly once.
    """

    print('\n[{}] Evaluating the trained model using the "{}" data'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                                                           partition_name))

    results_path = params['results_path']
    possible_classes = params['possible_classes']

    accuracies_per_class_per_split = {
        'split': [],
        'partition': [],
        'ind_acc': [],
        'maj_acc': [],
        'prob_acc': [],
        'ind_rhd_negative_acc': [],
        'ind_rhd_positive_acc': [],
        'ind_rhd_negative_acc': [],
        'maj_rhd_positive_acc': [],
        'maj_rhd_negative_acc': [],
        'prob_rhd_positive_acc': [],
        'prob_rhd_negative_acc': []
    }

    cm_inds = []
    normalized_cm_inds = []
    cm_majs = []
    normalized_cm_majs = []
    cm_probs = []
    normalized_cm_probs = []

    # Get predictions and infos
    for split_index in range(num_splits):

        predictions_arr = np.load(
            '{}/{}/pred/{}_split{}_pred.npy'.format(results_path, model_name, partition_name, split_index)
        )
        infos_df = pd.read_csv(
            '{}/{}/pred/{}_split{}_info.csv'.format(results_path, model_name, partition_name, split_index)
        )

        partition_exams = infos_df['Exam'].unique()
        inverted_partition_ids = {v: i for i, v in infos_df['Sample'].iteritems()}

        # Individual Video Predictions
        y_true_ind = infos_df['Diagnosis'].values
        y_pred_ind = np.argmax(predictions_arr, axis=1)
        acc_pred_ind = metrics.accuracy_score(y_true_ind, y_pred_ind, normalize=True)

        # Majority Vote Predictions
        y_true_maj = []
        y_pred_maj = []
        for e in partition_exams:
            current_ids = infos_df[infos_df['Exam'] == e]['Sample'].values
            exam_preds = []
            for c in current_ids:
                exam_preds.append(predictions_arr[inverted_partition_ids[c]])
            exam_preds = np.array(exam_preds)

            # Prioritize disease classes in case of a draw
            votes = np.bincount(np.argmax(exam_preds, axis=1))
            reversed_votes = votes[::-1]
            final_vote = len(reversed_votes) - np.argmax(reversed_votes) - 1

            y_true_maj.append(infos_df[infos_df['Exam'] == e]['Diagnosis'].unique()[0])
            y_pred_maj.append(final_vote)
        acc_pred_maj = metrics.accuracy_score(y_true_maj, y_pred_maj, normalize=True)

        # Probability Mean Predictions
        y_true_mean = []
        y_pred_mean = []
        for e in partition_exams:
            current_ids = infos_df[infos_df['Exam'] == e]['Sample'].values
            exam_preds = []
            for c in current_ids:
                exam_preds.append(predictions_arr[inverted_partition_ids[c]])
            exam_preds = np.array(exam_preds)

            y_true_mean.append(infos_df[infos_df['Exam'] == e]['Diagnosis'].unique()[0])
            y_pred_mean.append(np.argmax(exam_preds.mean(axis=0)))
        acc_pred_mean = metrics.accuracy_score(y_true_mean, y_pred_mean, normalize=True)

        cm_ind, normalized_cm_ind = get_confusion_matrix(y_true_ind, y_pred_ind)
        cm_maj, normalized_cm_maj = get_confusion_matrix(y_true_maj, y_pred_maj)
        cm_prob, normalized_cm_prob = get_confusion_matrix(y_true_mean, y_pred_mean)

        cm_inds.append(cm_ind)
        normalized_cm_inds.append(normalized_cm_ind)
        cm_majs.append(cm_maj)
        normalized_cm_majs.append(normalized_cm_maj)
        cm_probs.append(cm_prob)
        normalized_cm_probs.append(normalized_cm_prob)

        accuracies_per_class_per_split['split'].append(split_index)
        accuracies_per_class_per_split['partition'].append(partition_name)
        accuracies_per_class_per_split['ind_acc'].append(acc_pred_ind)
        accuracies_per_class_per_split['maj_acc'].append(acc_pred_maj)
        accuracies_per_class_per_split['prob_acc'].append(acc_pred_mean)
        accuracies_per_class_per_split['ind_rhd_negative_acc'].append(normalized_cm_ind[0][0])
        accuracies_per_class_per_split['ind_rhd_positive_acc'].append(normalized_cm_ind[1][1])
        accuracies_per_class_per_split['maj_rhd_positive_acc'].append(normalized_cm_maj[0][0])
        accuracies_per_class_per_split['maj_rhd_negative_acc'].append(normalized_cm_maj[1][1])
        accuracies_per_class_per_split['prob_rhd_positive_acc'].append(normalized_cm_prob[0][0])
        accuracies_per_class_per_split['prob_rhd_negative_acc'].append(normalized_cm_prob[1][1])

    # Create and save confusion matrix
    os.makedirs('{}/{}/cm/'.format(results_path, model_name), exist_ok=True)

    df_ind, annot_ind = get_confusion_matrix_mean_std(cm_inds, possible_classes)
    normalized_df_ind, normalized_annot_ind = get_confusion_matrix_mean_std(normalized_cm_inds, possible_classes)
    save_confusion_matrix(df_ind, annot_ind, results_path, model_name, partition_name, 'individual')
    save_confusion_matrix(normalized_df_ind, normalized_annot_ind, results_path, model_name, partition_name,
                          'individual', normalized=True)

    df_maj, annot_maj = get_confusion_matrix_mean_std(cm_majs, possible_classes)
    normalized_df_maj, normalized_annot_maj = get_confusion_matrix_mean_std(normalized_cm_majs, possible_classes)
    save_confusion_matrix(df_maj, annot_maj, results_path, model_name, partition_name, 'majority_vote')
    save_confusion_matrix(normalized_df_maj, normalized_annot_maj, results_path, model_name, partition_name,
                          'majority_vote', normalized=True)

    df_prob, annot_prob = get_confusion_matrix_mean_std(cm_probs, possible_classes)
    normalized_df_prob, normalized_annot_prob = get_confusion_matrix_mean_std(normalized_cm_probs, possible_classes)
    save_confusion_matrix(df_prob, annot_prob, results_path, model_name, partition_name, 'probability_mean')
    save_confusion_matrix(normalized_df_prob, normalized_annot_prob, results_path, model_name, partition_name,
                          'probability_mean', normalized=True)

    print('[{}] Done!'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))

    return accuracies_per_class_per_split


def save_results_report(params, model_name, accuracies):
    train = pd.DataFrame(accuracies['train'])
    val = pd.DataFrame(accuracies['val'])
    test = pd.DataFrame(accuracies['test'])

    column_order = ['split', 'partition', 'ind_acc', 'maj_acc', 'prob_acc', 'ind_rhd_negative_acc',
                    'ind_rhd_positive_acc', 'maj_rhd_positive_acc', 'maj_rhd_negative_acc', 'prob_rhd_positive_acc',
                    'prob_rhd_negative_acc']

    final_df = pd.concat([train, val, test]).sort_values(by=['split', 'partition']).reset_index(drop=True)
    final_df = final_df[column_order]
    final_df = final_df.round(4)
    final_df.to_csv('{}/{}/report.csv'.format(params['results_path'], model_name), index=False)


def calculate_best_results(accuracies_report):
    best_results = {}

    for partition_name in accuracies_report.keys():
        best_results[partition_name] = max(
            accuracies_report[partition_name]['ind_acc'],
            accuracies_report[partition_name]['maj_acc'],
            accuracies_report[partition_name]['prob_acc']
        )[0]

    return best_results


def save_model_hyperparameters(params, model_name):

    hyperparam_names = ['n_classes', 'batch_size', 'num_epochs', 'learning_rate', 'lr_decay', 'doppler_filtering',
                        'undersampling_filtering', 'bound_undersampling_to', 'crop_strategy', 'batch_normalization']
    hyperparams = {}
    for n in hyperparam_names:
        hyperparams[n] = params[n]

    with open('{}/{}/hyperparameters.json'.format(params['results_path'], model_name), 'w') as f:
        f.write(json.dumps(hyperparams))


def try_params(params, instances_data):
    """
    Train the model and evaluate it, given a set of arguments.
    """

    splits = get_splits(instances_data)

    model_name = get_model_name(params)

    for split_index, partitions in enumerate(splits):
        print('\nProcessing split {}/{}'.format(split_index + 1, len(splits)))
        print('{}\n'.format('=' * (18 + len(str(split_index + 1)) + len(str(len(splits))))))

        model = build_c3d_model(params)
        save_model = True if split_index == 0 else False

        # Calculate weights for each class in training data
        diagnosis = instances_data[instances_data['Sample'].isin(partitions['train'])]['Diagnosis']
        classes_weights = dict(zip(np.unique(diagnosis),
                                   class_weight.compute_class_weight('balanced', np.unique(diagnosis), diagnosis)))

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

    # Evaluate model
    accuracies_report = {}
    accuracies_report['train'] = evaluate_model(params, model_name, 'train', len(splits))
    accuracies_report['val'] = evaluate_model(params, model_name, 'val', len(splits))
    accuracies_report['test'] = evaluate_model(params, model_name, 'test', len(splits))

    save_results_report(params, model_name, accuracies_report)

    best_results = calculate_best_results(accuracies_report)

    print('\n[{}] Results saved to {}/{}/'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), params['results_path'],
                                                  model_name))
    print(
        'Train, validation and test accuracies were {:.3f}%, {:.3f}% and {:.3f}% respectively'.format(
            best_results['train'] * 100,
            best_results['val'] * 100,
            best_results['test'] * 100
        )
    )

    return model_name, best_results


def main(args):
    # Set Keras backend session with correct configurations
    set_backend_session()

    # Get parameters
    params = get_params(args)

    # Load data
    instances_data = pd.read_csv(params['info_file_path'])
    labels = pd.Series(instances_data['Diagnosis'].values, index=instances_data['Sample']).to_dict()

    params['labels'] = labels

    try_params(params, instances_data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess a given dataset.')

    parser.add_argument('-dp', '--data-path', type=str, dest='data_path',
                        help='Path where the preprocessed dataset can be found.')
    parser.add_argument('-mp', '--models-path', type=str, dest='models_path',
                        help='Path where models can be saved or loaded from.')
    parser.add_argument('-rp', '--results-path', type=str, dest='results_path',
                        help='Path where results can be saved or loaded from.')
    parser.add_argument('-nn', '--nn-type', type=str, choices=['c3d', 'c3d_frozen', 'c3d_test'], dest='nn_type',
                        help='Neural network type that images should serve as input.')
    parser.add_argument('-if', '--info-file-path', type=str, dest='info_file_path',
                        help='Path to the file that contains classification information for each instance.')
    parser.add_argument('-lr', '--learning-rate', type=float, dest='learning_rate',
                        help='Learning rate for the machine learning model.')
    parser.add_argument('-ld', '--lr-decay', type=float, dest='lr_decay', default=0.0,
                        help='Decay rate for the learning rate. Occurs at each epoch.')
    parser.add_argument('-ep', '--num-epochs', type=int, dest='num_epochs',
                        help='Number of epochs to train the machine learning model.')
    parser.add_argument('-bs', '--batch-size', type=int, default='8', dest='batch_size',
                        help='Size of the batch to be feed to the network.')
    parser.add_argument('-df', '--doppler-filtering', type=str, dest='doppler_filtering',
                        choices=['none', 'only_with_doppler', 'only_without_doppler'], default='none',
                        help=('Doppler filter applied to data before training (may reduce the number of training/valida'
                              'tion/test samples).'))
    parser.add_argument('-uf', '--undersampling-filtering', type=str, dest='undersampling_filtering',
                        choices=['none', 'normal_definite', 'normal_borderline',
                                 'normal_definite_borderline', 'normal_diseasecarrier'], default='none',
                        help=('Undersampling filter applied to data before training (may reduce the number of training/'
                              'validation/test samples).'))
    parser.add_argument('-bt', '--bound-undersampling-to', type=str, dest='bound_undersampling_to',
                        choices=['minimum', 'normal', 'borderline', 'definite', 'personalized'], default='minimum',
                        help='Reduce number of exams to the number of <YOUR CHOICE> exams when undersampling.')
    parser.add_argument('-btv', '--bound-undersampling-to-value', type=int, dest='bound_undersampling_to_value',
                        default=0, help='Reduce number of exams to the number of <YOUR CHOICE> when undersampling.')
    parser.add_argument('-cs', '--crop-strategy', type=str, choices=['center', 'random', 'none'], dest='crop_strategy',
                        default='center', help='Cropping strategy to be used in training.')
    parser.add_argument('-bn', '--batch-normalization', action='store_true', default=False, dest='batch_normalization',
                        help='Add batch normalization layers before each activation.')
    parser.add_argument('-ap', '--aug-percentage', type=float, dest='aug_percentage', default=0.0,
                        help='Percentage of augmented data to use.')
    parser.add_argument('-fes', '--filter-exam-size', action='store_true', default=False, dest='filter_exam_size',
                        help='Filter out exams with size less than 10 and greater than 20.')
    parser.add_argument('-cv', '--cross-validation', action='store_true', default=False, dest='cross_validation',
                        help='Whether to use 10-fold cross-validation. The default is False (no cross-validation).')
    parser.add_argument('-tl', '--time-logs', action='store_true', default=False, dest='time_logs',
                        help='Flag to save time logs per batch.')
    parser.add_argument('-mg', '--multi-gpu', type=int, default=1, dest='multi_gpu',
                        help='Number of GPUs to train the model on.')

    main(parser.parse_args())
