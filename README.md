# MXNet Gluon Dynamic-batching
This repository includes simplified implementation of Fold, a helper for [*dynamic batching*](https://arxiv.org/abs/1702.02181).

This animation from [Tensorflow Fold](https://github.com/tensorflow/fold) shows a [recursive neural network](https://en.wikipedia.org/wiki/Recursive_neural_network) run with dynamic batching. Operations of the signature at the same depth in the computation graph (indicated by color in the animiation) are batched together regardless of whether or not they appear in the same sample.

## Usage

This library performs dynamic batching by pooling the computation for multiple samples and merging
the shared operators through concatenation. For example:

```python
import fold
import mxnet as mx
from mxnet.gluon import nn

embed_layer = nn.Embedding(10, 1)
embed_layer.initialize()
fold_pool = fold.Fold()
lazy_values = []
lazy_results = []
for i in range(15):
    lazy_values.append(fold_pool.record(0, embed_layer, mx.nd.array([i % 10])))
shared_value = fold_pool.record(0, mx.nd.concat, *lazy_values).no_batch()
for i in range(100):
    lazy_results.append(fold_pool.record(0, mx.nd.dot, lazy_values[i % 10], shared_value))

# collect actual results
actual_result = fold_pool([lazy_results])[0]
```

Also, for use with Gluon RNN cells, there is the `fold_unroll` function:
```python
import random
cell = mx.gluon.rnn.LSTMCell(20)
fold_pool = fold.Fold()
batch_size = 5
for _ in range(batch_size):
    length = random.randint(1, 5)
    fold.fold_unroll(cell, fold_pool, length, mx.nd.random.uniform(shape=(length, 1, 10)),
                     layout='TNC')

# show the complete computation graph info
print(fold_pool)
```

## Performance

The following results are obtained from MXNet Gluon implementation of [treelstm.pytorch](https://github.com/dasguptar/treelstm.pytorch), which can be found in [example/tree_lstm](https://github.com/szha/mxnet-fold/tree/master/example/tree_lstm) folder. It implements Child-sum Tree-LSTM by [Tai et al.](https://arxiv.org/abs/1503.00075) on the semantic-relatedness task on SICK dataset. The performance is evaluated on:
- EC2 [c4.8xlarge](https://aws.amazon.com/ec2/instance-types/#c4) with
  - Intel Xeon E5-2666 v3 (Haswell) processors
  - Ubuntu 16.04

The following speed benchmark is performed with the following settings:
- Batch size: 256
- MXNet:
  - [mxnet-cu80mkl==0.12.0b20171030](https://pypi.python.org/pypi?:action=display&name=mxnet-cu80mkl&version=0.12.0b20171030) for GPU test.
  - [mxnet-mkl==0.12.0b20171030](https://pypi.python.org/pypi?:action=display&name=mxnet-mkl&version=0.12.0b20171030) for CPU test.


The following results are obtained on EC2 [c4.8xlarge](https://aws.amazon.com/ec2/instance-types/#c4) host.

| Implementation                         |     Training     |    Inference     |
|----------------------------------------|------------------|------------------|
| MXNet Gluon w/o Fold                   | 33.77 samples/s  | 50.46 samples/s  |
| MXNet Gluon w/o Fold Hybridized        | 66.79 samples/s  | 131.86 samples/s |
| MXNet Gluon w/ Fold                    | 201.11 samples/s | 315.54 samples/s |
