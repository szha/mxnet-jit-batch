import collections
import traceback
from itertools import chain

import mxnet as mx
from mxnet import nd
from mxnet.gluon import nn, Block
from mxnet.gluon.utils import _indent

__all__ = ['Fold']

class Fold(Block):
    r"""`Fold` supports dynamic-batching with NDArray.

    Parameters
    ----------
    ctx : mx.context.Context
        The context in which all operators are executed.

    Inputs:
        - **nodes**: list of list of Virtual nodes. All nodes in one child list must be batchable.
        - **reset**: boolean. Whether to clear the records in the fold after calling.

    Output:
        - **out**: list of NDArrays. Each NDArray represents the batched nodes in a child input list.
    """
    def __init__(self, ctx=mx.context.current_context()):
        self.steps = collections.defaultdict(
            lambda: collections.defaultdict(list))
        self.cached_nodes = collections.defaultdict(dict)
        assert isinstance(ctx, mx.context.Context), 'ctx must be a Context. Got: %s' % str(ctx)
        self.ctx = ctx

    def record(self, batch_axis, op, *args):
        """Record an operator call to the fold.

        Parameters
        ----------
        batch_axis : int
            Axis that represents a batch. Must be non-negative.
            All NDArray arguments must have length value 1 on that axis.
        op : callable
            The operator to be recorded.
        *args: object
            Arguments to the operator call.

        Returns
        -------
        node : Virtual
            An object that represents the output of the registered operation.
        Calling the fold on which the operator is registered with the Virtual object
        results in the corresponding result of the operator.
        """
        assert callable(op), 'op must be callable. Got: %s' % op
        flat_args, fmt = _flatten(args)
        op_sig = OpSig(op, fmt, batch_axis)
        flat_args_sig = tuple(hash(x) for x in flat_args)
        arg_types = tuple(_type_code(arg) for arg in flat_args)
        if flat_args_sig not in self.cached_nodes[op_sig]:
            step = max(chain.from_iterable(((0,), (arg.step+1 for arg, arg_type
                                                   in zip(flat_args, arg_types)
                                                   if arg_type == 1))))
            node = Virtual(op_sig, _arg_hash(flat_args), step, len(self.steps[step][op_sig]))
            self.steps[step][op_sig].append((flat_args, arg_types))
            self.cached_nodes[op_sig][flat_args_sig] = node
        return self.cached_nodes[op_sig][flat_args_sig]

    def forward(self, nodes, reset=True):
        """Apply current fold to given nodes."""
        values = {}
        if reset:
            self.cached_nodes = None
        ctx = self.ctx
        for step in range(len(self.steps)):
            values[step] = {}
            for op_signature in self.steps[step]:
                op, fmt, batch_axis = op_signature
                args, arg_types = zip(*self.steps[step][op_signature])
                try:
                    args, arg_size = _batch_args(zip(*args), zip(*arg_types), values, batch_axis)
                except Exception:
                    print("Error while executing node Step %d op: %s\n"
                          "(fmt: %s, axis: %d)\n"
                          "with args:\n%s" % (
                        step, op, fmt, batch_axis, self.steps[step][op_signature]))
                    print(traceback.format_exc())
                    raise
                if reset:
                    self.steps[step][op_signature] = None
                args, _ = _regroup(args, fmt)
                with ctx:
                    values[step][op_signature] = _split_batch(op(*args), batch_axis, arg_size)
        try:
            return [_batch_args([n], [(1,)], values, batch_axis)[0][0] for n in nodes]
        except Exception:
            print("Retrieving %s" % nodes)
            print(traceback.format_exc())
            for lst in nodes:
                if isinstance(lst[0], Virtual):
                    print(', '.join([str(x(values).size()) for x in lst]))
            raise
        finally:
            if reset:
                self.reset()

    def reset(self):
        """Clear this fold."""
        self.steps = collections.defaultdict(
            lambda: collections.defaultdict(list))
        self.cached_nodes = collections.defaultdict(dict)

    def __repr__(self):
        s = 'Fold-{steps} (\n{stepstr}\n)'
        result = ''
        for step in range(len(self.steps)):
            result += 'Step %d:\n' % step
            for op_signature in self.steps[step]:
                out = ',\n    '.join([str(arg.shape) if isinstance(arg, nd.NDArray) else str(arg)
                                 for arg in zip(*self.steps[step][op_signature])[0]])
                result += '    %s = %d x (%s)\n' % (
                    op_signature, len(self.steps[step][op_signature]), out)
        return _indent(s.format(steps=len(self.steps), stepstr=result), 4)

OpSig = collections.namedtuple('OpSig', ['op', 'fmt', 'batch_axis'])

class Virtual(object):
    def __init__(self, op_sig, arg_sig, step, index):
        self.op_sig = op_sig
        self.arg_sig = arg_sig
        self.step = step
        self.index = index
        self.split_idx = ()
        self.batch = True
        self._hash = None
        self.expand_axis = None

    def __getitem__(self, i):
        if isinstance(i, int):
            result = Virtual(self.op_sig, self.arg_sig, self.step, self.index)
            result.split_idx = self.split_idx + (i,)
            return result
        elif isinstance(i, slice):
            assert i.stop is not None and (i.start is None or i.stop > i.start), \
                    'Slice must have non-negative start and stop. Got: %s' % (i)
            nodes = []
            for idx in range(i.stop)[i]:
                nodes.append(Virtual(self.op_sig, self.arg_sig, self.step, self.index))
                nodes[-1].split_idx = self.split_idx + (idx,)
            return tuple(nodes)
        else:
            raise ValueError('Unsupported key for Virtual indexing: %s' % i)

    def expand_dims(self, axis):
        assert self.expand_dims is None, 'Dimension can be expanded only once.'
        self.expand_axis = axis
        return self

    def split(self, num):
        return self[:num]

    def no_batch(self):
        self.batch = False
        return self

    def __call__(self, values):
        out = values[self.step][self.op_sig][self.index]
        if self.split_idx:
            for i in self.split_idx:
                out = out[i]
        if self.expand_axis is not None:
            out = out.expand_dims(self.expand_axis)
        return out

    def __repr__(self):
        return "%s[%d][%d][%s]" % (self.op_sig, self.step, self.index, self.split_idx)

    def __hash__(self):
        if not self._hash:
            self._hash = hash((self.index, self.split_idx)) + self.arg_sig
        return self._hash

def _flatten(args):
    if isinstance(args, (list, tuple)):
        flat = []
        fmts = []
        for i in args:
            arg, fmt = _flatten(i)
            flat.extend(arg)
            fmts.append(fmt)
        return tuple(flat), tuple(fmts)
    else:
        return (args,), int(0)

def _regroup(args, fmt):
    if isinstance(fmt, int):
        if fmt == 0:
            return args[0], args[1:]
        return args[:fmt], args[fmt:]

    ret = []
    for i in fmt:
        res, args = _regroup(args, i)
        ret.append(res)
    return ret, args

def _split_batch(arg, batch_axis, arg_size):
    if isinstance(arg, nd.NDArray):
        return nd.split(arg, arg_size, axis=batch_axis) if arg_size > 1 else [arg]
    arg, fmt = _flatten(arg)
    result = zip(*(nd.split(x, arg_size, axis=batch_axis) if arg_size > 1 else [x] for x in arg))
    out = [_regroup(x, fmt)[0] for x in result]
    return out

def _batch_args(arg_lists, arg_types, values, batch_axis):
    res = []
    arg_size = 1
    for arg, arg_type in zip(arg_lists, arg_types):
        arg_size = len(arg)
        if arg_type[0] == 1 and arg[0].batch:
            out = nd.concat(*(x(values) for x in arg), dim=batch_axis)
        elif arg_type[0] == -1:
            out = nd.concat(*arg, dim=batch_axis)
        else:
            for i in range(2, len(arg)):
                assert arg[i] == arg[0], \
                    "Can not use more than one of no-batch argument, got: %s." % str(arg)
            out = arg[0]
            if arg_type[0] == 1:
                out = out(values)
        res.append(out)
    return tuple(res), arg_size

def _type_code(arg):
    return isinstance(arg, Virtual)-isinstance(arg, nd.NDArray)

def _arg_hash(flat_args):
    return sum(hash(a) for a in flat_args)
