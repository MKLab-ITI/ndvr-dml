import tqdm
import argparse
import numpy as np

from model import DNN

def train_network(model, train_set, triplets, epochs, batch_sz):

    print train_set.shape, triplets.shape
    val_triplets = triplets[:10000]
    triplets = triplets[10000:]
    val_set = train_set[val_triplets.reshape(-1)]

    print 'start training'
    n_batch = triplets.shape[0] / batch_sz + 1
    for i in xrange(epochs):
        np.random.shuffle(triplets)
        pbar = tqdm.trange(n_batch, desc='epoch {}'.format(i),
                                mininterval=1.0, unit='iter',
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
    parser.add_argument('-ts', '--train_set', type=str, required=True)
    parser.add_argument('-tr', '--triplets', type=str, required=True)
    parser.add_argument('-m', '--model_path', type=str, required=True)
    parser.add_argument('-es', '--evaluation_set', type=str)
    parser.add_argument('-et', '--evaluation_triplets', type=str)
    parser.add_argument('-ij', '--injection', type=int, default=1000)
    parser.add_argument('-l', '--layers', default='2000,1000,500')
    parser.add_argument('-e', '--epochs', type=int, default=10)
    parser.add_argument('-b', '--batch_sz', type=int, default=1000)
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-6)
    parser.add_argument('-wd', '--weight_decay', type=float, default=1e-5)
    parser.add_argument('-g', '--gamma', type=float, default=1.0)
    args = vars(parser.parse_args())

    print 'loading data...'
    train_set = np.load(args['train_set'])

    triplets = np.load(args['triplets'])

    if args.get('evaluation_set'):
        evaluation_set = np.load(args['evaluation_set'])
        eval_triplets = np.load(args['evaluation_triplets']) + len(train_set)
        np.random.shuffle(eval_triplets)
        train_set = np.concatenate([train_set, evaluation_set], axis=0)
        triplets = np.concatenate([triplets, args['triplets'][:args['injection']]], axis=0)

    layers = [int(l) for l in args['layers'].split(',')]
    model = DNN(train_set.shape[1], layers,
                args['model_path'],
                learning_rate=args['learning_rate'],
                weight_decay=args['weight_decay'],
                gamma=args['gamma'])

    train_network(model, train_set, triplets, args['epochs'], args['batch_sz'])
