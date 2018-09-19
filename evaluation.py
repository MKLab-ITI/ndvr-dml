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

import argparse
import numpy as np
import pickle as pk

from model import DNN
from scipy.spatial.distance import cdist
from utils import plot_pr_curve, evaluate


def calculate_similarities(queries, features):
    """
      Function that generates video triplets from CC_WEB_VIDEO.

      Args:
        queries: indexes of the query videos
        features: global features of the videos in CC_WEB_VIDEO
      Returns:
        similarities: the similarities of each query with the videos in the dataset
    """
    similarities = dict()
    dist = np.nan_to_num(cdist(features[queries], features, metric='euclidean'))
    for i, v in enumerate(queries):
        sim = np.round(1 - dist[i] / dist.max(), decimals=6)
        similarities[i + 1] = [(s, sim[s]) for s in sim.argsort()[::-1] if not np.isnan(sim[s])]
    return similarities


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-es', '--evaluation_set', type=str, required=True,
                        help='Path to the .npy file that contains the global '
                             'video vectors of the CC_WEB_VIDEO dataset')
    parser.add_argument('-m', '--model_path', type=str, required=True,
                        help='Path to load the trained DML model')
    parser.add_argument('-pl', '--positive_labels', default='ESLMV',
                        help='Labels in CC_WEB_VIDEO datasets that '
                             'considered posetive. default=\'ESLMV\'')
    args = vars(parser.parse_args())

    print 'loading data...'
    cc_dataset = pk.load(open('datasets/cc_web_video.pickle', 'rb'))
    cc_features = np.load(args['evaluation_set'])

    model = DNN(cc_features.shape[1], None,
                args['model_path'],
                load_model=True,
                trainable=False)
    cc_embeddings = model.embeddings(cc_features)
    print 'Evaluation set file: ', args['evaluation_set']
    print 'Path to DML model: ', args['model_path']
    print 'Positive labels: ', args['positive_labels']

    print '\nEvaluation Results'
    print '=================='
    similarities = calculate_similarities(cc_dataset['queries'], cc_embeddings)
    mAP, pr_curve = evaluate(cc_dataset['ground_truth'], similarities,
                             positive_labels=args['positive_labels'], all_videos=False)
    print 'CC_WEB_VIDEO mAP: ', mAP
    plot_pr_curve(pr_curve, 'CC_WEB_VIDEO')

    mAP, pr_curve = evaluate(cc_dataset['ground_truth'], similarities,
                             positive_labels=args['positive_labels'], all_videos=True)
    print 'CC_WEB_VIDEO* mAP: ', mAP
    plot_pr_curve(pr_curve, 'CC_WEB_VIDEO*')
