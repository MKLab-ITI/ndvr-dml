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

import numpy as np
import pickle as pk
import matplotlib.pylab as plt

from sklearn.metrics import precision_recall_curve


def load_dataset(dataset):
    """
      Function that loads dataset object.

      Args:
        dataset: dataset name
      Returns:
        dataset object
    """
    return pk.load(open('datasets/{}.pickle'.format(dataset), 'rb'))


def load_feature_files(feature_files):
    """
      Function that loads the feature directories.

      Args:
        feature_files: file that contains the feature directories
      Returns:
        dictionary that contains the feature directories for each video id
      Raise:
        file is not in the right format
    """
    try:
        return {l.split('\t')[0]: l.split('\t')[1] for l in open(feature_files, 'rb').readlines()}
    except:
        raise Exception('''--feature_files provided is in wrong format. Each line of the 
        file have to contain the video id (name of the video file) 
        and the full path to the corresponding .npy file, separated
        by a tab character (\\t). Example:
                        
            23254771545e5d278548ba02d25d32add952b2a4	features/23254771545e5d278548ba02d25d32add952b2a4.npy
            468410600142c136d707b4cbc3ff0703c112575d	features/468410600142c136d707b4cbc3ff0703c112575d.npy
            67f1feff7f624cf0b9ac2ebaf49f547a922b4971	features/67f1feff7f624cf0b9ac2ebaf49f547a922b4971.npy
            7deff9e47e47c98bb341c4355dfff9a82bfba221	features/7deff9e47e47c98bb341c4355dfff9a82bfba221.npy
                                                      ...''')


def normalize(X):
    """
      Function that apply zero mean and l2-norm to every vector.

      Args:
        X: input feature vectors
      Returns:
        the normalized vectors
    """
    X -= X.mean(axis=1, keepdims=True)
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-15
    return X


def global_vector(video):
    """
      Function that calculate the global feature vector from the
      frame features vectors. First, all frame features vectors
      are normalized, then they are averaged on each dimension to
      produce the global vector, and finally the global vector is
      normalized again.

      Args:
        video: path to feature file of a video
      Returns:
        X: the normalized global feature vector
    """
    try:
        X = np.load(video)
        X = normalize(X)
        X = X.mean(axis=0, keepdims=True)
        X = normalize(X)
        return X
    except Exception as e:
        if video:
            print 'Can\'t load feature file {}\n{}'.format(video, e.message)
        return np.array([])


def plot_pr_curve(pr_curve, title):
    """
      Function that plots the PR-curve.

      Args:
        pr_curve: the values of precision for each recall value
        title: the title of the plot
    """
    plt.figure(figsize=(16, 9))
    plt.plot(np.arange(0.0, 1.05, 0.05),
             pr_curve, color='b', marker='o', linewidth=3, markersize=10)
    plt.grid(True, linestyle='dotted')
    plt.xlabel('Recall', color='k', fontsize=27)
    plt.ylabel('Precision', color='k', fontsize=27)
    plt.yticks(color='k', fontsize=20)
    plt.xticks(color='k', fontsize=20)
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(title, color='k', fontsize=27)
    plt.tight_layout()
    plt.show()


def evaluate(ground_truth, similarities, positive_labels='ESLMV', all_videos=False):
    """
      Function that plots the PR-curve.

      Args:
        ground_truth: the ground truth labels for each query
        similarities: the similarities of each query with the videos in the dataset
        positive_labels: labels that are considered positives
        all_videos: indicator of whether all videos are considered for the evaluation
        or only the videos in the query subset
      Returns:
        mAP: the mean Average Precision
        ps_curve: the values of the PR-curve
    """
    pr, mAP = [], 0.0
    for query_set, labels in ground_truth.iteritems():
        i = 0.0
        ri = 0
        s = 0.0
        y_target, y_score = [], []
        for video, sim in similarities[query_set]:
            if all_videos or video in labels:
                y_score += [sim]
                y_target += [0.0]
                ri += 1
                if video in labels and labels[video] in positive_labels:
                    i += 1.0
                    s += i / ri
                    y_target[-1] = 1.0

        mAP += s / np.sum([1.0 for label in labels.values() if label in positive_labels])

        precision, recall, thresholds = precision_recall_curve(y_target, y_score)
        p = []
        for i in xrange(20, 0, -1):
            idx = np.where((recall >= i*0.05))[0]
            p += [np.max(precision[idx])]
        pr += [p + [1.0]]

    return mAP / len(ground_truth), np.mean(pr, axis=0)[::-1]
