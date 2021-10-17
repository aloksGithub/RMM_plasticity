"""
Implements some finite state machine related tools inspired by the book

de la Higuera, C (2010). Grammatical Inference: Learning Automata and Grammars.
Cambridge University Press, Cambridge, UK. doi:10.1017/CBO9781139194655

"""

import numpy as np
import queue

__author__ = 'Benjamin Paaßen'
__copyright__ = 'Copyright 2020, Benjamin Paaßen'
__license__ = 'GPLv3'
__Version__ = '0.0.1'
__maintainer__ = 'Benjamin Paaßen'
__email__  = 'benjamin.paassen@sydney.edu.au'

def label_sequence(X, delta, rho, q_0 = 0):
    """ Labels the given input sequence using the given finite state
    machine.

    Parameters
    ----------
    X: class numpy.array
        An input sequence of symbol indices in the range [0, delta.shape[1]-1].
    delta: class numpy.array
        A len(states) x len(in_alphabet) table where delta[q, x] = q2 means
        that state q transfers to state q2 on input x.
    rho: class numpy.array
        A len(states) array where rho[q] = y means that state q outputs
        symbol y.
    q_0: int (default = 0)
        the starting state.

    Returns
    -------
    Y: class numpy.array
        An output sequence of output symbol indices as defined by rho.
        More precisely, Y[t] = rho[Q[t]].
    Q: class numpy.array
        A sequence of state indices, where Q[t] = delta[Q[t-1], X[t]].

    """
    Q = np.zeros(len(X), dtype=int)
    Q[0] = delta[q_0, X[0]]
    for t in range(1, len(X)):
        Q[t] = delta[int(Q[t-1]), X[t]]
    return rho[Q], Q

def label_sequences(Xs, delta, rho, q_0 = 0):
    """ Labels the given input sequences using the given finite state
    machine.

    Parameters
    ----------
    Xs: list OR numpy.array
        A list of input sequences, where each input sequence is a
        numpy.array of symbol indices in the range [0, delta.shape[1]-1].
    delta: class numpy.array
        A len(states) x len(in_alphabet) table where delta[q, x] = q2 means
        that state q transfers to state q2 on input x.
    rho: class numpy.array
        A len(states) array where rho[q] = y means that state q outputs
        symbol y.
    q_0: int (default = 0)
        the starting state.

    Returns
    -------
    Ys: list
        A list of output sequences, where each output sequence is an array
        of output symbol indices as defined by rho.
    Qs: list
        Q list of state equences, where each state sequence is an array of
        states assigned via delta.

    """
    Ys = []
    Qs = []
    for X in Xs:
        Y, Q = label_sequence(X, delta, rho, q_0)
        Ys.append(Y)
        Qs.append(Q)
    return Ys, Qs

def learn_fsm(Xs, Ys):
    """ Learns a finite state machine from a given set of input and output
    sequences.

    Note that this method assumes that the input data is consistent, i.e.
    there exist no two prefixes X[i][:t] == X[j][:t] such that
    Y[i][t] != Y[j][t].

    Parameters
    ----------
    Xs: list
        A list of input sequences, where each input sequence is a numpy.array
        of symbol indices, i.e. it is assumed that the union of all sequences
        is the set [0, ..., K] for some K.
    Ys: list
        A list of output sequences, where each output sequence is a numpy.array
        of symbol indices, i.e. it is assumed that the union of all sequences
        is the set [0, ..., L] for some L.

    Returns
    -------
    delta: class numpy.array
        The transition matrix of the resulting finite state machine which is
        consistent with the input if the input is consistent. We can not
        guarantee that this matrix is as short as possible because the learning
        scheme is greedy. But the number of states is at most as large as the
        set of all unique prefixes in Xs. Note that the start state is 0.
    rho: class numpy.array
        The output matrix of the resulting finite state machine.

    Raises
    ------
    ValueError
        If the input is not consistent.

    """
    if len(Xs) != len(Ys):
        raise ValueError('Expected as many input as output sequences but got %d input and %d output sequences.' % (len(Xs), len(Ys)))
    # first, find the size of the input alphabet
    n = 0
    for X in Xs:
        max_x = np.max(X)
        if max_x > n:
            n = max_x
    n += 1
    # initialize the output machine as a simple tree.
    delta = [np.full(n,-1)]
    rho   = [-1]
    for i in range(len(Xs)):
        if len(Xs[i]) != len(Ys[i]):
            raise ValueError('The %dth input and output sequence had different lengths (%d versus %d).' % (len(Xs[i]), len(Ys[i])))
        # start with the start state and iterate over the sequence
        q = 0
        for t in range(len(Xs[i])):
            x = Xs[i][t]
            # check if there is already a state defined for the current symbol
            if delta[q][x] < 0:
                # if not, create a new state
                delta[q][x] = len(delta)
                delta.append(np.full(n,-1))
                rho.append(Ys[i][t])
            q = delta[q][x]
    # transform into arrays
    delta = np.stack(delta, 0)
    rho = np.stack(rho, 0)
    # mark states into which we can merge ('red' states)
    red = [0]
    # mark states which we wish to merge ('blue' states)
    blue = queue.SimpleQueue()
    for x in range(n):
        blue.put((delta[0][x], 0, x))
    # then attempt to merge states according to breadth first search order
    while not blue.empty():
        # get the current state in the blue queue
        q, pred, x = blue.get()
        # if this state was already disconnected or is red, ignore it
        if delta[pred, x] != q or q in red:
            continue
        # iterate over all red states and try to merge
        merged = False
        for q2 in red:
            delta_merged = np.copy(delta)
            rho_merged = np.copy(rho)
            remaining = _merge(q2, q, pred, x, delta_merged, rho_merged)
            if remaining is not None:
                # push remaining states to blue
                for qnext, pred, x in remaining:
                    blue.put((qnext, pred, x))
                delta = delta_merged
                rho   = rho_merged
                merged = True
                break
        if not merged:
            # if no merge was possible, 'promote' q to a red state
            red.append(q)
            # and push all its children onto blue
            for x in range(n):
                if delta[q, x] >= 0:
                    blue.put((delta[q, x], q, x))

    # shorten the automaton to only the red states
    states = list(sorted(red))
    idx_to_shortened = {}
    for q in range(len(states)):
        idx_to_shortened[states[q]] = q
    delta = delta[states, :]
    for q in range(len(delta)):
        for x in range(n):
            if delta[q, x] >= 0:
                delta[q, x] = idx_to_shortened[delta[q, x]]
    rho   = rho[states]

    return delta, rho

def _merge(q, q2, pred, x, delta, rho):
    """ Attempts to merge the state q2 into q in the given finite state
    machine.

    """
    if delta[pred, x] != q2:
        raise ValueError('Internal error: delta[%g, %g] is not %g but %g' % (pred, x, q2, delta[pred, x]))
    if q == q2:
        # if both states are the same, the merge is trivial
        return []

    # merge the predecessor connection
    delta[pred, x] = q

    # check if the outputs are compatible
    if rho[q] < 0:
        # if rho[q] is undefined, plug in rho[q2]
        rho[q] = rho[q2]
    elif rho[q2] >= 0 and rho[q] != rho[q2]:
        return None

    # iterate over all children
    remaining = []
    for x in range(delta.shape[1]):
        # if delta[q, x] is undefined, plug in delta[q2, x]
        if delta[q, x] < 0:
            if delta[q2, x] >= 0:
                delta[q, x] = delta[q2, x]
                remaining.append((delta[q2, x], q, x))
            continue
        # if delta[q2, x] is undefined, continue
        if delta[q2, x] < 0:
            continue
        # otherwise try to merge both successors
        remaining_x = _merge(delta[q, x], delta[q2, x], q2, x, delta, rho)
        if remaining_x is None:
            return None
        remaining += remaining_x
    # return
    return remaining


def one_cyclic_paths(delta, q_0 = 0):
    """ Computes all paths that contain exactly one cycle in the finite state
    machine defined by the given transition matrix delta.

    Parameters
    ----------
    delta: class numpy.array
        A len(states) x len(in_alphabet) matrix mapping a current state and the
        current inpt symbol to a follow up state.
    q_0: int (default = 0)
        The index of the start state of the finite state machine.

    Returns
    -------
    paths: list
        The output list of all paths, where each path is in turn a list of
        tuples (q, x), where q is the current state index and x is the index
        of the previous symbol. In all paths, no state is repeated up to the
        last step, where one state is repeated.

    """
    # initialize output array
    paths = []
    # construct initial paths for all possible symbols
    for x in range(delta.shape[1]):
        q_next = delta[q_0, x]
        initial_path = [(q_next, x)]
        if q_next == q_0:
            # if we loop already, write the path to output
            paths.append(initial_path)
        else:
            # otherwise, start a recursive path computation
            _one_cyclic_paths(delta, initial_path, set([q_0, q_next]), paths)
    return paths


def _one_cyclic_paths(delta, path, state_set, paths):
    """ Computes all paths that contain exactly one cycle in the finite state
    machine defined by the given transition matrix delta.

    Parameters
    ----------
    delta: class numpy.array
        A len(states) x len(in_alphabet) matrix mapping a current state and the
        current inpt symbol to a follow up state.
    path: list
        The current path as a list of tuples (q, x) where q is the current
        state index and x is the previous input symbol index.
    state_set: set
        The set of all states already visited by the given path, i.e. the
        set of all first entries in path, plus the start state.
    paths: list
        The output list of all paths.

    """
    q = path[-1][0]
    # iterate over all possible next symbols
    for x in range(delta.shape[1]):
        q_next = delta[q, x]
        # continue the current path with that symbol
        next_path = path + [(q_next, x)]
        if q_next not in state_set:
            # extend the path further if it does not yet loop
            _one_cyclic_paths(delta, next_path, state_set | set([q_next]), paths)
        else:
            # otherwise write it to output
            paths.append(next_path)


def sample_fsm(num_states, num_in_symbols, num_out_symbols):
    """ Samples a random finite state machine with up to the given number of
    states and input as well as output alphabets of the given size.

    The starting state of the resulting FSM is always 0.

    Parameters
    ----------
    num_states: int
        The (maximum) number of states in the resulting finite state machine.
    num_in_symbols: int
        The size of the input alphabet for the resulting finite state machine.
    num_out_symbols: int
        The size of the output alphabet for the resulting finite state machine.

    Returns
    -------
    delta: class numpy.array
        A num_states x num_in_symbols transition matrix that is uniformly
        randomly sampled from the range [0, num_states-1]
    rho: class numpy.array
        A num_states-element array with entries uniformly randomly sampled
        from the range [0, num_out_symbols-1]

    """
    delta = np.random.randint(num_states, size = (num_states, num_in_symbols))
    rho   = np.random.randint(num_out_symbols, size = num_states)
    # check which states can actually be reached
    stk = [0]
    visited = set()
    while stk:
        q = stk.pop()
        visited.add(q)
        for x in range(num_in_symbols):
            if delta[q, x] not in visited:
                stk.append(delta[q, x])
    states = list(sorted(visited))
    # reduce the FSM to these states
    if len(states) < num_states:
        idx_to_shortened = {}
        for q in range(len(states)):
            idx_to_shortened[states[q]] = q
        delta = delta[states, :]
        for q in range(len(delta)):
            for x in range(num_in_symbols):
                delta[q, x] = idx_to_shortened[delta[q, x]]
        rho   = rho[states]
    return delta, rho
