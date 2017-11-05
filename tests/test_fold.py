import collections
import numpy as np

import mxnet as mx
from mxnet import nd
from mxnet.gluon import nn, rnn, Block
from mxnet.test_utils import assert_almost_equal

import fold


class TestBlock(nn.Block):

    def __init__(self):
        super(TestBlock, self).__init__()
        self.embed = nn.Embedding(10, 3)
        self.out = nn.Dense(10)

    def concat(self, axis, *nodes):
        return nd.concat(*nodes, dim=axis)

    def embed(self, indices):
        return self.embed(indices)

    def embed2(self, indices):
        return self.embed(indices), self.embed(indices)

    def predict(self, embed):
        return self.out(embed)

    def dot(self, enc, embed):
        return nd.dot(enc, embed.T)

    def __repr__(self):
        return 'test_encoder'


def test_rnn():
    t = TestBlock()
    f = fold.Fold()
    v1 = f.add(t.embed2, 0, mx.cpu(), mx.nd.array([1]))[0]
    v2 = f.add(t.embed2, 0, mx.cpu(), mx.nd.array([2]))[1]
    r = v1
    for i in range(10):
        r = f.add(t.concat, 0, mx.cpu(), 1, v1, r)
        r = f.add(t.concat, 0, mx.cpu(), 1, r, v2)
    r = f.add(t.predict, 0, mx.cpu(), r)

    t.initialize()
    enc = f([[r]])[0]
    assert enc.shape == (1, 10)

def test_no_batch():
    t = TestBlock()
    f = fold.Fold()
    v = []
    for i in range(15):
        v.append(f.add(t.embed, 0, mx.cpu(), mx.nd.array([i % 10])))
    d = f.add(t.concat, 0, mx.cpu(), 0, *v).no_batch()
    res = []
    for i in range(100):
        res.append(f.add(t.dot, 0, mx.cpu(), v[i % 10], d))

    t.initialize()
    enc = f([res])[0]
    assert enc.shape == (100, 15)

def test_rnn_cell():
    cell = rnn.LSTMCell(7)
    cell.initialize()
    f = fold.Fold()
    regular_result = []
    fold_result = []
    for _ in range(3):
        length = np.random.randint(3, 20)
        input_data = mx.nd.random.uniform(shape=(1, length, 5))
        regular_result.extend(cell.unroll(length, input_data, merge_outputs=False)[0])
        state = cell.begin_state(1)
        outputs = []
        split_input = input_data.split(length, squeeze_axis=True)
        for i in range(length):
            out, state = f.add(cell, 0, mx.cpu(), split_input[i], state).split(2)
            state = state.split(2)
            outputs.append(out)
        fold_result.extend(outputs)
    result = f([fold_result], True)[0]
    assert_almost_equal(result.asnumpy(), mx.nd.concat(*regular_result, dim=0).asnumpy())

def test_combined():
    f = fold.Fold()
    t = TestBlock()
    cell = fold.rnn.FoldLSTMCell(20)
    cell.initialize()
    t.initialize()
    fold_output = []
    for _ in range(3):
        length = np.random.randint(3, 20)
        input_data = mx.nd.random.uniform(shape=(1, length, 5))
        cell_out = [f.add(t.predict, 0, mx.cpu(), r[0]) for r in cell.fold_unroll(f, length, input_data)]
        fold_output.extend(cell_out)
    f([fold_output])[0]


if __name__ == '__main__':
    import nose
    nose.runmodule()
