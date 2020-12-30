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
Implementation of the evaluation process based on CC_WEB_VIDEO dataset.
"""

from __future__ import division
from __future__ import print_function

import argparse

from utils import *
from model import DNN
from tqdm import tqdm
from scipy.spatial.distance import cdist


def calculate_similarities(queries, features):
    """
      Function that generates video triplets from CC_WEB_VIDEO.

      Args:
        queries: indexes of the query videos
        features: global features of the videos in CC_WEB_VIDEO
      Returns:
        similarities: the similarities of each query with the videos in the dataset
    """
    similarities = []
    dist = np.nan_to_num(cdist(features[queries], features, metric='euclidean'))
    for i, v in enumerate(queries):
        sim = np.round(1 - dist[i] / dist.max(), decimals=6)
        similarities += [[(s, sim[s]) for s in sim.argsort()[::-1] if not np.isnan(sim[s])]]
    return similarities


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-es', '--evaluation_set', type=str, required=True,
                        help='Path to the .npy file that contains the global '
                             'video vectors of the CC_WEB_VIDEO dataset')
    parser.add_argument('-m', '--model_path', type=str, required=True,
                        help='Path to load the trained DML model')
    parser.add_argument('-f', '--fusion', type=str, default='Early',
                        help='Processed dataset. Options: Early and Late. Default: Early')
    parser.add_argument('-ef', '--evaluation_features', type=str,
                        help='Paths to the .npy files that contains the feature vectors '
                             'of the videos in the CC_WEB_VIDEO dataset. Each line of the '
                             'file have to contain the video id (name of the video file) '
                             'and the full path to the corresponding .npy file, separated '
                             'by a tab character (\\t)')
    parser.add_argument('-pl', '--positive_labels', type=str, default='ESLMV',
                        help='Labels in CC_WEB_VIDEO datasets that '
                             'considered posetive. Default=\'ESLMV\'')
    args = vars(parser.parse_args())

    print('Loading data...')
    cc_dataset = pk.load(open('datasets/cc_web_video.pickle', 'rb'))
    cc_features = load_features(args['evaluation_set'])

    print('Loading model...')
    model = DNN(cc_features.shape[1],
                args['model_path'],
                load_model=True,
                trainable=False)

    if args['fusion'].lower() == 'early':
        print('Fusion type: Early')
        print('Extract video embeddings...')
        cc_embeddings = model.embeddings(cc_features)
    else:
        print('Fusion type: Late')
        print('Extract video embeddings...')

        assert args['evaluation_features'] is not None, \
            'Argument \'--evaluation_features\' must be provided for Late fusion'
        feature_files = load_feature_files(args['evaluation_features'])

        cc_embeddings = np.zeros((len(cc_dataset['index']), model.embedding_dim))
        for i, video_id in enumerate(tqdm(cc_dataset['index'])):
            if video_id in feature_files:
                features = load_features(feature_files[video_id])
                embeddings = model.embeddings(normalize(features))
                embeddings = embeddings.mean(0, keepdims=True)
                cc_embeddings[i] = normalize(embeddings, zero_mean=False)

    print('\nEvaluation set file: ', args['evaluation_set'])
    print('Path to DML model: ', args['model_path'])
    print('Positive labels: ', args['positive_labels'])

    print('\nEvaluation Results')
    print('==================')
    similarities = calculate_similarities(cc_dataset['queries'], cc_embeddings)
    baseline_similarities = calculate_similarities(cc_dataset['queries'], cc_features)
    mAP_dml, pr_curve_dml = evaluate(cc_dataset['ground_truth'], similarities,
                                     positive_labels=args['positive_labels'], all_videos=False)
    mAP_base, pr_curve_base = evaluate(cc_dataset['ground_truth'], baseline_similarities,
                                       positive_labels=args['positive_labels'], all_videos=False)
    print('CC_WEB_VIDEO')
    print('baseline mAP: ', mAP_base)
    print('DML mAP: ', mAP_dml)
    plot_pr_curve(pr_curve_dml, pr_curve_base, 'CC_WEB_VIDEO')

    mAP_dml, pr_curve_dml = evaluate(cc_dataset['ground_truth'], similarities,
                                     positive_labels=args['positive_labels'], all_videos=True)
    mAP_base, pr_curve_base = evaluate(cc_dataset['ground_truth'], baseline_similarities,
                                       positive_labels=args['positive_labels'], all_videos=True)
    print('\nCC_WEB_VIDEO*')
    print('baseline mAP: ', mAP_base)
    print('DML mAP: ', mAP_dml)
    plot_pr_curve(pr_curve_dml, pr_curve_base, 'CC_WEB_VIDEO*')
