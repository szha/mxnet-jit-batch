import mxnet as mx
from mxnet.gluon.rnn.rnn_cell import _format_sequence, _get_begin_state

def fold_unroll(cell, fold, length, inputs, begin_state=None, layout='NTC'):
    """Unrolls an RNN cell across time steps.

    Parameters
    ----------
    length : int
        Number of steps to unroll.
    inputs : Symbol, list of Symbol, or None
        If `inputs` is a single Symbol (usually the output
        of Embedding symbol), it should have shape
        (batch_size, length, ...) if `layout` is 'NTC',
        or (length, batch_size, ...) if `layout` is 'TNC'.

        If `inputs` is a list of symbols (usually output of
        previous unroll), they should all have shape
        (batch_size, ...).
    begin_state : nested list of Symbol, optional
        Input states created by `begin_state()`
        or output state of another cell.
        Created from `begin_state()` if `None`.
    layout : str, optional
        `layout` of input symbol. Only used if inputs
        is a single Symbol.
    merge_outputs : bool, optional
        If `False`, returns outputs as a list of Symbols.
        If `True`, concatenates output across time steps
        and returns a single symbol with shape
        (batch_size, length, ...) if layout is 'NTC',
        or (length, batch_size, ...) if layout is 'TNC'.
        If `None`, output whatever is faster.

    Returns
    -------
    outputs : list of Symbol or Symbol
        Symbol (if `merge_outputs` is True) or list of Symbols
        (if `merge_outputs` is False) corresponding to the output from
        the RNN from this unrolling.

    states : list of Symbol
        The new state of this RNN after this unrolling.
        The type of this symbol is same as the output of `begin_state()`.
    """
    cell.reset()

    inputs, _, F, batch_size = _format_sequence(length, inputs, layout, False)
    begin_state = _get_begin_state(cell, F, begin_state, inputs, batch_size)

    states = begin_state
    outputs = []
    for i in range(length):
        output, states = fold.record(0, cell, inputs[i], states).split(2)
        outputs.append(output)
        states = states.split(2)

    return outputs, states
