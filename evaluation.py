import argparse
import numpy as np
import pickle as pk
import matplotlib.pylab as plt

from model import DNN
from scipy.spatial.distance import cdist
from sklearn.metrics import precision_recall_curve

def plot_pr_curve(pr_curve, title):
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

def evaluate(ground_truth, similarities, positive_labels='ESLMV', all=False):
    AP = 0.0
    pr, mAP = [], 0.0
    for query_set, labels in ground_truth.iteritems():
        i = 0.0
        ri = 0
        s = 0.0
        y_target, y_score = [], []
        for video, sim in similarities[query_set]:
            if all or video in labels:
                y_score += [sim]
                y_target += [0.0]
                ri += 1
                if video in labels and labels[video] in positive_labels:
                    i += 1.0
                    s += i / ri
                    y_target[-1] = 1.0

        AP += s / np.sum([1.0 for label in labels.values() if label in positive_labels])

        precision, recall, thresholds = precision_recall_curve(y_target, y_score)
        p = []
        for i in xrange(20, 0, -1):
            idx = np.where((recall >= i*0.05))[0]
            p += [np.max(precision[idx])]
        pr += [p + [1.0]]

    return AP / len(ground_truth), np.mean(pr, axis=0)[::-1]

def calculate_similarities(queries, features):
    similarities = dict()
    for i, v in enumerate(queries):
        dist = cdist([features[v]], features, metric='euclidean')[0]
        sim = np.round(1 - dist / np.max(dist[~np.isnan(dist)]), decimals=6)
        similarities[i + 1] = [(s, sim[s]) for s in sim.argsort()[::-1] if not np.isnan(sim[s])]
    return similarities


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-es', '--evaluation_set', type=str, required=True)
    parser.add_argument('-m', '--model_path', type=str, required=True)
    parser.add_argument('-pl', '--positive_labels', default='ESLMV')
    parser.add_argument('-l', '--layers', default='2000,1000,500')
    args = vars(parser.parse_args())

    print 'loading data...'
    cc_dataset = pk.load(open('datasets/cc_web_video.pickle', 'rb'))
    cc_features = np.load(args['evaluation_set'])

    layers = [int(l) for l in args['layers'].split(',')]
    model = DNN(cc_features.shape[1], layers,
                args['model_path'],
                load_model=True,
                trainable=False)

    cc_embeddings = model.embeddings(cc_features)

    similarities = calculate_similarities(cc_dataset['queries'], cc_embeddings)
    mAP, pr_curve = evaluate(cc_dataset['ground_truth'], similarities,
                             positive_labels=args['positive_labels'], all=False)
    print 'CC_WEB_VIDEO mAP: ', mAP
    plot_pr_curve(pr_curve, 'CC_WEB_VIDEO')

    mAP, pr_curve = evaluate(cc_dataset['ground_truth'], similarities,
                             positive_labels=args['positive_labels'], all=True)
    print 'CC_WEB_VIDEO* mAP: ', mAP
    plot_pr_curve(pr_curve, 'CC_WEB_VIDEO*')
