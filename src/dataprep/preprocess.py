"""
Useful functions for preprocessing medical videos
"""

import argparse
import math
import os
import random
from joblib import Parallel, delayed
from typing import Dict, List, Tuple

import cv2
import multiprocessing
import numpy as np
import pandas as pd
import skimage
from tqdm import tqdm

random.seed(123456)


def create_mask() -> np.ndarray:

    img_original_shape = (320, 240, 3)

    mask = np.full(img_original_shape, 255, dtype="uint8")
    left_triangle = np.array([(0, 0), (0, 200), (112, 0)])
    right_triangle = np.array([(240, 0), (240, 198), (126, 0)])

    cv2.drawContours(mask, [left_triangle], 0, (0, 0, 0), -1)
    cv2.drawContours(mask, [right_triangle], 0, (0, 0, 0), -1)

    return mask


def resize_img(img: np.ndarray, resize_dims: Tuple) -> np.ndarray:

    return skimage.transform.resize(img,
                                    output_shape=resize_dims,
                                    preserve_range=True,
                                    anti_aliasing=True).astype('uint8')


def preprocess_frame(
        frame: np.ndarray,
        mask: np.ndarray,
        turn_gray: bool,
        rotate_frames: bool,
        resize_dims: Tuple
    ) -> np.ndarray:
    
    # Apply mask
    frame = cv2.bitwise_and(frame, mask)

    if turn_gray:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        assert len(resize_dims) == 2, 'If using conversion to gray, the new image shape must have only 2 dimensions.'

    if rotate_frames:  # Rotate frame 90 degrees clockwise
        frame = cv2.rotate(frame, 0)

    frame = resize_img(frame, resize_dims)

    return frame


def sample_frames(frames: List, frame_sample_style: str, frame_sample_size: int) -> List:

    frame_cnt_diff = len(frames) - frame_sample_size

    if frame_sample_style == 'continuous':
        if frame_cnt_diff < 0:
            first_frame_qty = math.ceil(abs(frame_cnt_diff) / 2)
            last_frame_qty = math.floor(abs(frame_cnt_diff) / 2)
            frames = ([frames[0]] * first_frame_qty) + frames + ([frames[-1]] * last_frame_qty)
        else:
            frames = frames[0:frame_sample_size]
    elif frame_sample_style == 'random':
        if frame_cnt_diff < 0:
            # random.sample is faster than np.random.choice
            frames = random.sample(frames, len(frames))
        else:
            frames = random.sample(frames, frame_sample_size)
    elif frame_sample_style == 'random_ignore_short':
        if frame_cnt_diff < 0:
            frames = []
        else:
            frames = random.sample(frames, frame_sample_size)

    return frames


def is_doppler_video(video_path: str) -> bool:
    colorbar_roi_threshold = 15

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()

    # Could not read video frames
    if not ret:
        print('Could not read video from {}'.format(video_path))
        return None

    # Verify if sample was generated with Doppler
    colorbar_ROI = frame[:150, :30]  # Top-left corner 150x30 from the last read frame
    with_doppler = np.mean(colorbar_ROI) > colorbar_roi_threshold  # If colorbar is not present, the ROI is black
    return with_doppler


def preprocess_video_2D(
        input_path: str,
        output_path: str,
        mean_matrix: np.ndarray,
        mask: np.ndarray,
        video_file_name: str,
        params: Dict,
    ) -> Tuple:
    
    video_path = '{}/{}'.format(input_path, video_file_name)

    frames = []
    cap = cv2.VideoCapture(video_path)
    while True:
        ret, frame = cap.read()

        # End of the video
        if not ret:
            break

        frame = preprocess_frame(
            frame,
            mask,
            params['turn_gray'],
            params['rotate_frames'],
            params['resize_dims']
        )

        if mean_matrix is not None:
            frame = np.subtract(frame, mean_matrix)

        frames.append(frame)
    frames = np.stack(frames)

    crop_dims = params['crop_dims']
    if crop_dims:
        frames = frames[crop_dims[0]:crop_dims[1], crop_dims[2]:crop_dims[3], :]

    # Normalize frames
    frames = frames / 255

    video_name = video_file_name[:-4]
    with_doppler = is_doppler_video(video_path)

    files_info = []
    for frame_id in range(len(frames)):
        frame_file_name = '{}_{}'.format(video_name, frame_id)
        files_info.append((video_name, frame_id, frame_file_name, with_doppler))
        np.savez_compressed('{}/{}.npz'.format(output_path, frame_file_name), frames=frames[frame_id])

    return files_info


def preprocess_video_3D(
        input_path: str,
        output_path: str,
        mean_cube: np.ndarray,
        mask: np.ndarray,
        video_file_name: str,
        params: Dict,
    ) -> Tuple:
    
    video_path = '{}/{}'.format(input_path, video_file_name)

    frames = []
    cap = cv2.VideoCapture(video_path)
    while True:
        ret, frame = cap.read()

        # End of the video
        if not ret:
            break

        frames.append(
            preprocess_frame(
                frame,
                mask,
                params['turn_gray'],
                params['rotate_frames'],
                params['resize_dims']
            )
        )
    frames = sample_frames(frames, params['frame_sample_style'], params['frame_sample_size'])
    frames = np.stack(frames)

    if mean_cube is not None:
        frames = np.subtract(frames, mean_cube)

    crop_dims = params['crop_dims']
    if crop_dims:
        frames = frames[:, crop_dims[0]:crop_dims[1], crop_dims[2]:crop_dims[3], :]

    # Normalize frames
    # frames = frames / 255

    video_name = video_file_name[:-4]
    with_doppler = is_doppler_video(video_path)
    np.savez_compressed('{}/{}.npz'.format(output_path, video_name), frames=frames)
    file_info = (video_name, video_name, with_doppler)

    return file_info


def preprocess_dataset_2D(
        input_path: str,
        output_path: str,
        cleansing_info_df: pd.core.frame.DataFrame,
        mean_matrix: np.ndarray,
        params: Dict
    ):

    # Creates folder to save generated inputs
    os.makedirs(output_path, exist_ok=True)

    # Creation of specific deidentification pixel mask
    mask = create_mask()

    if mean_matrix is not None:
        mean_matrix = np.transpose(mean_matrix, (1, 2, 0))

    # Uses only 80% of available cores so that the machine remains functional
    num_cores = int(0.8 * multiprocessing.cpu_count())

    # List of info attributed to videos that will eventually become a CSV file
    preprocessed_info = []

    video_file_names = [v for v in os.listdir(input_path) if v.endswith('.MP4')]

    # Parallel preprocessing of videos
    preprocessed_info.extend(
        Parallel(n_jobs=num_cores)(
            delayed(preprocess_video_2D)(
                input_path,
                output_path,
                mean_matrix,
                mask,
                video_file_name,
                params
            ) for video_file_name in video_file_names
        )
    )

    # Flatten the files_info lists
    preprocessed_info = [file_info for video_list in preprocessed_info for file_info in video_list]
    preprocessed_info_df = pd.DataFrame(preprocessed_info, columns=['Video ID', 'Frame ID', 'File Name', 'Doppler'])
    preprocessed_info_df['Augmentation'] = None  # No videos are augmented in this stage
    preprocessed_info_df = preprocessed_info_df.merge(cleansing_info_df, on='Video ID', how='left')
    preprocessed_info_df = preprocessed_info_df[['Exam ID', 'Video ID', 'Frame ID', 'File Name', 'Diagnosis', 'Doppler', 'Augmentation']]
    preprocessed_info_df.columns = ['Exam ID', 'Video ID', 'Frame ID', 'File Name', 'Diagnosis', 'Doppler', 'Augmentation']
    preprocessed_info_df.to_csv('{}/../preprocessed_rhd_frames_info.csv'.format(output_path), index=False)


def preprocess_dataset_3D(
        input_path: str,
        output_path: str,
        cleansing_info_df: pd.core.frame.DataFrame,
        mean_cube: np.ndarray,
        params: Dict
    ):

    # Creates folder to save generated inputs
    os.makedirs(output_path, exist_ok=True)

    # Creation of specific deidentification pixel mask
    mask = create_mask()

    if mean_cube is not None:
        mean_cube = np.transpose(mean_cube, (1, 2, 3, 0))

    # Uses only 80% of available cores so that the machine remains functional
    num_cores = int(0.8 * multiprocessing.cpu_count())

    # List of info attributed to videos that will eventually become a CSV file
    preprocessed_info = []

    video_file_names = [v for v in os.listdir(input_path) if v.endswith('.MP4')]

    # Parallel preprocessing of videos
    preprocessed_info.extend(
        Parallel(n_jobs=num_cores)(
            delayed(preprocess_video_3D)(
                input_path,
                output_path,
                mean_cube,
                mask,
                video_file_name,
                params
            ) for video_file_name in video_file_names
        )
    )

    preprocessed_info_df = pd.DataFrame(preprocessed_info, columns=['Video ID', 'File Name', 'Doppler'])
    preprocessed_info_df['Augmentation'] = None
    preprocessed_info_df = preprocessed_info_df.merge(cleansing_info_df, on='Video ID', how='left')
    preprocessed_info_df = preprocessed_info_df[['Exam ID', 'Video ID', 'File Name', 'Diagnosis', 'Doppler', 'Augmentation']]
    preprocessed_info_df.to_csv('{}/../preprocessed_rhd_videos_info.csv'.format(output_path), index=False)


def main(args):
    """
    Preprocessing parameters
    -------------------------

    resize_dims: Shape of images after preprocesing. Only used if no neural network type is passed.
    turn_gray: Flag to get grayscale images. Only used if no neural network type is passed.
    frame_sample_style: Style of frame sampling from videos. Only used if no neural network type is passed.
        - values: ['all', 'continuous', 'random', 'random_ignore_short']
            - all: Do nothing
            - continuous: Fill in a balanced manner the beginning and end with copies of the borderline frames
            - random: Use video even if sample size is bigger than number of frame
            - random_ignore_short: If sample size is bigger than number of frames, don't use video

    frame_sample_size: Style of frame sampling from videos. Only used if no neural network type is passed.
    rotate_frames: Rotate frames 90 degrees before resizing for a better aspect ratio compatibility (320x240 -> 128x171).
    """

    nn_map = {
        'c3d': {
            'preprocess_function': preprocess_dataset_3D,
            'params': {
                'turn_gray': False,
                'rotate_frames': True,
                'resize_dims': (128, 171, 3),
                'crop_dims': (8, 120, 20, 132),
                'frame_sample_style': 'continuous',
                'frame_sample_size': 16
            }
        },
        'vgg': {
            'preprocess_function': preprocess_dataset_2D,
            'params': {
                'turn_gray': False,
                'rotate_frames': False,
                'resize_dims': (224, 224, 3),
                'crop_dims': None,
                'frame_sample_style': None,
                'frame_sample_size': None,
            }
        },
        'wacv': {
            'preprocess_function': preprocess_dataset_3D,
            'params': {
                'turn_gray': False,
                'rotate_frames': False,
                'resize_dims': (224, 224, 3),
                'crop_dims': None,
                'frame_sample_style': 'continuous',
                'frame_sample_size': 16,
            }
        }
    }

    cleansing_info_df = pd.read_csv(args.info_file_path)
    mean_array = np.load(args.mean_array_path) if args.mean_array_path is not None else None

    nn_map[args.nn_type]['preprocess_function'](
        args.input_path,
        args.output_path,
        cleansing_info_df,
        mean_array,
        nn_map[args.nn_type]['params']
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess a given dataset of medical videos.')

    parser.add_argument('-nn', '--nn-type', type=str, choices=['c3d', 'vgg', 'wacv'], default='c3d' , dest='nn_type',
                        help='Neural network type that images should serve as input.')
    parser.add_argument('-i', '--input-path', type=str, dest='input_path',
                        help='Input path where the dataset can be found.')
    parser.add_argument('-o', '--output-path', type=str, dest='output_path',
                        help='Output path to which the preprocessed dataset must be saved.')
    parser.add_argument('-if', '--info-file-path', type=str, dest='info_file_path',
                        help='Path to the file that contains information obtaining during cleansing for each video.')
    parser.add_argument('-ma', '--mean-array-path', type=str, default=None, dest='mean_array_path',
                        help='Path to the mean matrix or cube of a pre-trained model. It is used for zero centering the data before training.')

    main(parser.parse_args())