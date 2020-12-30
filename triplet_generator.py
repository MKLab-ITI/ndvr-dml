# Copyright 2018 Giorgos Kordopatis-Zilos. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
Implementation of the triplet generation process.
"""

from __future__ import division
from __future__ import print_function

import os
import argparse
import numpy as np

from tqdm import tqdm
from multiprocessing import Pool
from future.utils import viewitems, lrange
from scipy.spatial.distance import cdist, euclidean
from utils import load_dataset, load_feature_files, global_vector

# we found that there are distractor videos that share segments with videos in the core dataset
excluded_negatives = [6171, 97284]


def dataset_global_features(dataset, feature_files, cores):
    """
      Function that extracts the global feature vectors
      for each video in the given dataset.

      Args:
        dataset: dataset object that contains the video ids
        feature_files: a dictionary that contains the feature path of the given dataset videos
        cores: CPU cores for the parallel extraction
      Returns:
        global_features: global features of the videos in the given dataset
    """
    print('Number of videos: ', len(feature_files))
    print('CPU cores: ', cores)

    print('\nGlobal Vectors Extraction')
    print('=========================')
    progress_bar = tqdm(dataset['index'], unit='videos')

    # extract features in parallel
    pool = Pool(cores)
    future = []
    for video_id in dataset['index']:
        future += [pool.apply_async(global_vector,
                                    args=[feature_files.get(video_id)],
                                    callback=(lambda *a: progress_bar.update()))]
    pool.close()
    pool.join()

    # find feature dimension
    dim = 0
    for f in future:
        if f.get().size > 0:
            dim = f.get().shape[1]
            break

    # collect global features
    global_features = np.zeros((len(future), dim))
    for i, f in enumerate(future):
        if f.get().size > 0:
            global_features[i] = f.get()

    progress_bar.close()
    pool.terminate()

    return global_features


def triplet_generator_vcdb(dataset, vcdb_features, threshold):
    """
      Function that generates video triplets from VCDB.

      Args:
        dataset: dataset object that contains the VCDB video pairs
        vcdb_features: global features of the videos in VCDB
        threshold: overlap threshold
      Returns:
        triplets: the list of triplets with video indexes
    """
    # split VCBD dataset
    core_dataset = vcdb_features[:528]
    distractors = vcdb_features[528:]

    print('\nVCDB Triplet Generation')
    print('=======================')
    triplets = []
    for video_pair in tqdm(dataset['video_pairs']):
        if video_pair['overlap'][0] > threshold and video_pair['overlap'][1] > threshold:
            video1 = core_dataset[video_pair['videos'][0]]
            video2 = core_dataset[video_pair['videos'][1]]

            # calculate distances
            pair_distance = euclidean(video1, video2)
            negative_distances = cdist(np.array([video1, video2]), distractors, metric='euclidean')

            hard_negatives = np.where(negative_distances[0] < pair_distance)[0] + 528
            triplets += [[video_pair['videos'][0], video_pair['videos'][1], negative]
                         for negative in hard_negatives if negative not in excluded_negatives]

            hard_negatives = np.where(negative_distances[1] < pair_distance)[0] + 528
            triplets += [[video_pair['videos'][1], video_pair['videos'][0], negative]
                         for negative in hard_negatives if negative not in excluded_negatives]
    return triplets


def triplet_generator_cc(dataset, cc_web_features):
    """
      Function that generates video triplets from CC_WEB_VIDEO.

      Args:
        dataset: dataset object that contains the VCDB video pairs
        vcdb_features: global features of the videos in CC_WEB_VIDEO
      Returns:
        triplets: the list of triplets with video indexes
    """
    print('\nCC_WEB_VIDEO Triplet Generation')
    print('===============================')
    triplets = []

    # generate triplets from each query set
    for i, ground_truth in enumerate(dataset['ground_truth']):
        pos = [k for k, v in viewitems(ground_truth) if v in ['E', 'L', 'V', 'S', 'M']]
        neg = [k for k, v in viewitems(ground_truth) if v in ['X', '-1']]
        for q in tqdm(lrange(len(pos)), desc='Query {}'.format(i)):
            for p in lrange(q + 1, len(pos)):
                video1 = cc_web_features[pos[q]]
                video2 = cc_web_features[pos[p]]

                # calculate distances
                pair_distance = euclidean(video1, video2)
                if pair_distance > 0.1:
                    negative_distances = cdist(np.array([video1, video2]), cc_web_features[neg], metric='euclidean')

                    hard_negatives = np.where(negative_distances[0] < pair_distance)[0]
                    triplets += [[pos[q], pos[p], neg[e]] for e in hard_negatives]

                    hard_negatives = np.where(negative_distances[1] < pair_distance)[0]
                    triplets += [[pos[p], pos[q], neg[e]] for e in hard_negatives]
    return triplets


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--feature_files', type=str, required=True,
                        help='Path to the .npy files that contains the feature vectors '
                             'of the videos in the provided dataset. Each line of the '
                             'file have to contain the video id (name of the video file) '
                             'and the full path to the corresponding .npy file, separated '
                             'by a tab character (\\t)')
    parser.add_argument('-o', '--output_dir', type=str, required=True,
                        help='Output directory where the generated files will be stored')
    parser.add_argument('-d', '--dataset', type=str, required=True,
                        help='Processed dataset. Options: VCDB and CC_WEB_VIDEO')
    parser.add_argument('-t', '--overlap_threshold', type=float, default=0.8,
                        help='Overlap threshold over which the video pairs in VCDB dataset'
                             'are considered positives. Default: 0.8')
    parser.add_argument('-c', '--cores', type=int, default=8,
                        help='CPU cores to be used for the parallel load of video '
                             'feature vectors. Default: 8')
    args = vars(parser.parse_args())

    args['dataset'] = args['dataset'].lower()
    if args['dataset'] not in ['vcdb', 'cc_web_video']:
        raise Exception('--dataset is invalid. Only VCDB and CC_WEB_VIDEO datasets are supported')
    dataset = load_dataset(args['dataset'])

    print('Processed dataset: ', args['dataset'])
    print('Storage directory: ', args['output_dir'])
    features = dataset_global_features(dataset, load_feature_files(args['feature_files']), args['cores'])
    np.save(os.path.join(args['output_dir'], '{}_features'.format(args['dataset'])), features)
    if 'vcdb' in args['dataset'].lower():
        triplets = triplet_generator_vcdb(dataset, features, args['overlap_threshold'])
    elif 'cc_web_video' in args['dataset'].lower():
        triplets = triplet_generator_cc(dataset, features)
    np.save(os.path.join(args['output_dir'], '{}_triplets'.format(args['dataset'])), triplets)
