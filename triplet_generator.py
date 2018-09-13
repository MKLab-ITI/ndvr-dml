import argparse
import numpy as np
import pickle as pk

from tqdm import tqdm, trange
from multiprocessing import Pool
from scipy.spatial.distance import cdist, euclidean

excluded_negatives = [6171, 97284]

def load_dataset(dataset):
    return pk.load(open('datasets/{}.pickle'.format(dataset), 'rb'))

def load_feature_files(feature_files):
    return {l[0]: l[1] for l in np.loadtxt(feature_files, delimiter='\t', dtype=str)}

def normalize(X):
    X -= X.mean(axis=1, keepdims=True)
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-15
    return X

def global_vector(video):
    try:
        X = np.load(video)['features']
        X = normalize(X)
        X = X.mean(axis=0, keepdims=True)
        return X
    except Exception as e:
        if video:
            print 'Can\'t load feature file {}\n{}'.format(video, e.message)
        return np.array([])


def extract_features(dataset, feature_files, cores):
    progress_bar = trange(len(dataset['index']))
    pool = Pool(cores)
    future = []
    for video_id in dataset['index']:
        future += [pool.apply_async(global_vector,
                        args=[feature_files.get(video_id)],
                        callback=(lambda *a: progress_bar.update()))]
    pool.close()
    pool.join()

    global_vectors = np.zeros((len(future), future[1].get().shape[1]))
    for i, f in enumerate(future):
        if f.get().size > 0:
            global_vectors[i] = f.get()
    pool.terminate()

    return global_vectors

def triplet_generator_vcdb(dataset, vcdb_features, threshold):
    core_dataset = vcdb_features[:528]
    distractors = vcdb_features[528:]

    triplets = []
    for video_pair in tqdm(dataset['video_pairs']):
        if video_pair['overlap'][0] > threshold and video_pair['overlap'][1] > threshold:
            video1 = core_dataset[video_pair['videos'][0]]
            video2 = core_dataset[video_pair['videos'][1]]
            pair_distance = euclidean(video1, video2)
            negative_distances = cdist(np.array([video1, video2]), distractors, metric='euclidean')
            hard_negatives = np.where(negative_distances[0] < pair_distance)[0] + 528
            triplets += [[video_pair['videos'][0], video_pair['videos'][1], negative]
                         for negative in hard_negatives if negative not in excluded_negatives]

            hard_negatives = np.where(negative_distances[1] < pair_distance)[0] + 528
            triplets += [[video_pair['videos'][1], video_pair['videos'][0], negative]
                         for negative in hard_negatives if negative not in excluded_negatives]
    return triplets

def triplet_generator_cc_web(dataset, cc_web_features):
    triplets = []
    for i, ground_truth in dataset['ground_truth'].iteritems():
        pos = [k for k, v in ground_truth.iteritems() if v in ['E', 'L', 'V', 'S', 'M']]
        neg = [k for k, v in ground_truth.iteritems() if v in ['X', '-1']]
        for q in tqdm(xrange(len(pos)), desc='Query {}'.format(i)):
            for p in xrange(q + 1, len(pos)):
                video1 = cc_web_features[pos[q]]
                video2 = cc_web_features[pos[p]]
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
    parser.add_argument('-f', '--feature_files', type=str, required=True)
    parser.add_argument('-o', '--output_dir', type=str, required=True)
    parser.add_argument('-d', '--dataset', type=str, required=True)
    parser.add_argument('-t', '--overlap_threshold', type=float)
    parser.add_argument('-c', '--cores', type=int, default=8)
    args = vars(parser.parse_args())

    dataset = load_dataset(args['dataset'])

    features = extract_features(dataset,
                load_feature_files(args['feature_files']), args['cores'])
    np.save(args['dataset']+'_global_features', features)

    if 'vcdb' in args['dataset'].lower():
        triplets = triplet_generator_vcdb(dataset, features, args['overlap_threshold'])
    elif 'cc_web_video' in args['dataset'].lower():
        triplets = triplet_generator_cc_web(dataset, features)
    np.save(args['dataset'] + '_triplets', triplets)
