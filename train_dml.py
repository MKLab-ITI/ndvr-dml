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
Tensorflow implementation of the Deep Metric Learning training process.
"""

import tqdm
import argparse
import numpy as np

from model import DNN


def train_dml_network(model, train_set, triplets, epochs, batch_sz):
    """
      Function that handles the training process.

      Args:
        model: the DML model
        train_set: the features of the training set
        triplets: the generated triplets
        epochs: the training epochs
        batch_sz: the batch size
    """

    # split triplets to create a validation set
    val_triplets = triplets[:10000]
    triplets = triplets[10000:]
    val_set = train_set[val_triplets.reshape(-1)]

    print '\nStart of DML Training'
    print '====================='
    n_batch = triplets.shape[0] / batch_sz + 1
    for i in xrange(epochs):
        np.random.shuffle(triplets)
        pbar = tqdm.trange(n_batch, desc='epoch {}'.format(i),
                                mininterval=1.0, unit='batch',
                                postfix={'loss': '', 'error': ''})
        for j in pbar:
            triplet_batch = triplets[j * batch_sz: (j + 1) * batch_sz]
            train_batch = train_set[triplet_batch.reshape(-1)]

            model.train(train_batch)

            if (j+1) % 25 == 0:
                loss, cost, error = model.test(val_set)
                pbar.set_postfix(loss=loss, error='{0:.2f}%'.format(error))
            if j % int(0.25 * n_batch + 1) == 0 and j > 0:
                model.save()

        model.save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-ts', '--train_set', type=str, required=True,
                        help='Path to the .npy file that contains the global '
                             'video vectors of the train set')
    parser.add_argument('-tr', '--triplets', type=str, required=True,
                        help='Path to the .npy file that contains the triplets '
                             'generated based on the train set')
    parser.add_argument('-m', '--model_path', type=str, required=True,
                        help='Directory where the generated files will be stored')
    parser.add_argument('-es', '--evaluation_set', type=str,
                        help='Path to the .npy file that contains the global '
                             'video vectors of the evaluation set')
    parser.add_argument('-et', '--evaluation_triplets', type=str,
                        help='Path to the .npy file that contains the triplets '
                             'generated based on the evaluation set')
    parser.add_argument('-ij', '--injection', type=int, default=10000,
                        help='Number of injected triplets generated from the '
                             'evaluation set. It is only applied when the '
                             'evaluation_set is provided. Default: 10000, Max:10000')
    parser.add_argument('-l', '--layers', default='2000,1000,500',
                        help='Number of neuron for each layer of the DML network, '
                             'separated by a comma \',\'. Default: 2000,1000,500')
    parser.add_argument('-e', '--epochs', type=int, default=10,
                        help='Number of epochs to train the DML network. Default: 10')
    parser.add_argument('-b', '--batch_sz', type=int, default=1000,
                        help='Number of triplets fed every training iteration. '
                             'Default: 1000')
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-6,
                        help='Learning rate of the DML network. Default: 10^-6')
    parser.add_argument('-wd', '--weight_decay', type=float, default=1e-5,
                        help='Regularization parameter of the DML network. Default: 10^-5')
    parser.add_argument('-g', '--gamma', type=float, default=1.0,
                        help='Margin parameter of the distance between the two pairs of '
                             'every triplet. Default: 1.0')
    args = vars(parser.parse_args())

    print 'Train set file: ', args['train_set']
    print 'Train triplet file: ', args['triplets']

    print 'loading data...'
    train_set = np.load(args['train_set'])
    triplets = np.load(args['triplets'])

    if args.get('evaluation_set'):
        args['injection'] = np.min([args['injection'], 10000])
        print 'Evaluation set file: ', args['evaluation_set']
        print 'Evaluation triplet file: ', args['evaluation_triplets']
        print 'Injected triplet: ', args['injection']
        print 'loading data...'
        evaluation_set = np.load(args['evaluation_set'])
        eval_triplets = np.load(args['evaluation_triplets']) + len(train_set)
        np.random.shuffle(eval_triplets)
        train_set = np.concatenate([train_set, evaluation_set], axis=0)
        triplets = np.concatenate([triplets, eval_triplets[:args['injection']]], axis=0)

    try:
        layers = [int(l) for l in args['layers'].split(',') if l]
    except Exception as e:
        raise Exception('--layers argument is in wrong format. Specify the number '
                        'of neurons in each layer separated by a comma \',\'')

    model = DNN(train_set.shape[1], layers,
                args['model_path'],
                learning_rate=args['learning_rate'],
                weight_decay=args['weight_decay'],
                gamma=args['gamma'])

    train_dml_network(model, train_set, triplets, args['epochs'], args['batch_sz'])
