# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

from itertools import chain, tee
import sys

import mxnet as mx
from mxnet import nd
from mxnet.gluon import Block, nn, rnn
from mxnet.gluon.parameter import Parameter
from mxnet.gluon.rnn.rnn_cell import _format_sequence, _get_begin_state

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

class ChildSumLSTMCell(rnn.RecurrentCell):
    def __init__(self, hidden_size,
                 i2h_weight_initializer=None,
                 hs2h_weight_initializer=None,
                 hc2h_weight_initializer=None,
                 i2h_bias_initializer='zeros',
                 hs2h_bias_initializer='zeros',
                 hc2h_bias_initializer='zeros',
                 input_size=0, prefix=None, params=None):
        super(ChildSumLSTMCell, self).__init__(prefix=prefix, params=params)
        with self.name_scope():
            self._hidden_size = hidden_size
            self._input_size = input_size
            self.i2h = nn.Dense(4*hidden_size,
                                weight_initializer=i2h_weight_initializer,
                                bias_initializer=i2h_bias_initializer)
            self.hs2h = nn.Dense(3*hidden_size,
                                 weight_initializer=hs2h_weight_initializer,
                                 bias_initializer=hs2h_bias_initializer)
            self.hc2h = nn.Dense(hidden_size,
                                 weight_initializer=hc2h_weight_initializer,
                                 bias_initializer=hc2h_bias_initializer,
                                 flatten=False)

    def state_info(self, batch_size=0):
        return ({'shape': (batch_size, self._hidden_size), '__layout__': 'NC'},
                {'shape': (batch_size, self._hidden_size), '__layout__': 'NC'})

    def _alias(self):
        return 'childsum_lstm'

    def forward(self, F, inputs, *children_states):
        name = '{0}{1}_'.format(self.prefix, self._alias)
        # notation: N for batch size, C for hidden state dimensions, K for number of children.
        # FC for i, f, u, o gates (N, 4*C), from input to hidden
        i_act_i2h, f_act_i2h, u_act_i2h, o_act_i2h = self.i2h(inputs).split(num_outputs=4)

        iuo_i2h = F.concat(i_act_i2h, u_act_i2h, o_act_i2h, dim=1) # (N, C*3)

        # sum of children states
        children_states = list(pairwise(children_states))
        hs = F.add_n(*[state[0] for state in children_states], name='%shs'%name) # (N, C)
        # concatenation of children hidden states
        hc = F.concat(*(F.expand_dims(state[0], axis=1) for state in children_states), dim=1,
                      name='%shc') # (N, K, C)
        # concatenation of children cell states
        cc = F.concat(*(F.expand_dims(state[1], axis=1) for state in children_states), dim=1,
                      name='%scs') # (N, K, C)

        # calculate activation for forget gate. addition in f_act is done with broadcast
        f_act = f_act_i2h.expand_dims(1) + self.hc2h(hc) # (N, K, C)

        # FC for i, u, o gates, from summation of children states to hidden state
        iuo_i2h = iuo_i2h + self.hs2h(hs)

        i_act, u_act, o_act = F.SliceChannel(iuo_i2h, num_outputs=3,
                                             name='%sslice'%name) # (N, C)*3

        # calculate gate outputs
        in_gate = F.Activation(i_act, act_type='sigmoid', name='%si'%name)
        in_transform = F.Activation(u_act, act_type='tanh', name='%sc'%name)
        out_gate = F.Activation(o_act, act_type='sigmoid', name='%so'%name)
        forget_gates = F.Activation(f_act, act_type='sigmoid', name='%sf'%name) # (N, K, C)

        # calculate cell state and hidden state
        next_c = F._internal._plus(F.sum(forget_gates * cc, axis=1), in_gate * in_transform,
                                   name='%sstate'%name)
        next_h = F._internal._mul(out_gate, F.Activation(next_c, act_type='tanh'),
                                  name='%sout'%name)

        return next_h, [next_h, next_c]

    def encode(self, F, inputs, tree):
        root_input = inputs[tree.idx].expand_dims(0)
        if tree.children:
            root_children_states = chain.from_iterable(self.encode(F, inputs, c)[1]
                                                       for c in tree.children)
        else:
            with root_input.context:
                root_children_states = self.begin_state(1)

        return self(F, root_input, *root_children_states)

    def fold_encode(self, fold, F, inputs, tree):
        root_input = inputs[tree.idx]
        if tree.children:
            root_children_states = chain.from_iterable(
                    self.fold_encode(fold, F, inputs, c)[1].split(2)
                    for c in tree.children)
        else:
            with root_input.context:
                root_children_states = fold.add(self.begin_state,
                                                0, root_input.context, 1).no_batch().split(2)

        return fold.add(self, 0, root_input.context, F, root_input, *root_children_states)
