import os
try:
   import cPickle as pickle
except:
   import pickle

import numpy as np

import mxnet as mx
from mxnet.test_utils import almost_equal, assert_almost_equal

from fold import Fold, rnn
from model import SimilarityTreeLSTM

def _indent(s_, numSpaces):
    s = s_.split('\n')
    s = [(numSpaces * ' ') + line for line in s]
    s = '\n'.join(s)
    return s

def str_tree(tree):
    if tree.children:
        return '{0}:\n{1}'.format(tree.idx, _indent('\n'.join(str_tree(c) for c in tree.children), 4))
    else:
        return str(tree.idx)

CURRENT_DIR = os.path.dirname(__file__)

def test_tree_lstm():

    l_sentences = mx.nd.load(os.path.join(CURRENT_DIR, 'l_sentences.nd'))
    r_sentences = mx.nd.load(os.path.join(CURRENT_DIR, 'r_sentences.nd'))
    with open(os.path.join(CURRENT_DIR, 'trees.pkl'), 'rb') as f:
        l_trees, r_trees = pickle.load(f)

    rnn_hidden_size, sim_hidden_size, num_classes = 150, 50, 5
    net = SimilarityTreeLSTM(sim_hidden_size, rnn_hidden_size, 2413, 300, num_classes)
    net.initialize(mx.init.Xavier(magnitude=2.24))
    sent = mx.nd.concat(l_sentences[0], r_sentences[0], dim=0)
    net(sent, len(l_sentences[0]), l_trees[0], r_trees[0])
    net.embed.weight.set_data(mx.nd.random.uniform(shape=(2413, 300)))

    def verify(batch_size):
        print('verifying batch size: ', batch_size)
        fold = Fold()
        num_samples = 100
        inputs = []
        fold_preds = []
        for i in range(num_samples):
            # get next batch
            l_sent = l_sentences[i]
            r_sent = r_sentences[i]
            sent = mx.nd.concat(l_sent, r_sent, dim=0)
            l_len = len(l_sent)
            l_tree = l_trees[i]
            r_tree = r_trees[i]

            inputs.append((sent, l_len, l_tree, r_tree))
            z_fold = net.fold_encode(fold, sent, l_len, l_tree, r_tree)
            fold_preds.append(z_fold)

            if (i+1) % batch_size == 0 or (i+1) == num_samples:
                fold_outs = fold([fold_preds])[0]
                outs = mx.nd.concat(*[net(sent, l_len, l_tree, r_tree)
                                      for sent, l_len, l_tree, r_tree in inputs], dim=0)
                if not almost_equal(fold_outs.asnumpy(), outs.asnumpy()):
                    print(fold_preds)
                    print('l_sents: ', l_sent, l_sentences[i-1])
                    print('r_sents: ', r_sent, r_sentences[i-1])
                    print('\n'.join((str(l_tree), str_tree(l_tree),
                                     str(r_tree), str_tree(r_tree),
                                     str(l_trees[i-1]), str_tree(l_trees[i-1]),
                                     str(r_trees[i-1]), str_tree(r_trees[i-1]),
                                     str(fold))))
                    assert_almost_equal(fold_outs.asnumpy(), outs.asnumpy())
                fold_preds = []
                inputs = []
                fold.reset()

    for batch_size in range(1, 6):
        verify(batch_size)


if __name__ == '__main__':
    import nose
    nose.runmodule()
