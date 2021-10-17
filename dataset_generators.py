"""
This module accumulates the dataset generating functions for the benchmark
datasets, except for fsms and associative recall, which require special
handling.

"""

# Copyright (C) 2020
# Benjamin Paaßen
# The University of Sydney

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.


import random
import numpy as np
import csv

__author__ = 'Benjamin Paaßen'
__copyright__ = 'Copyright 2020, Benjamin Paaßen'
__license__ = 'GPLv3'
__Version__ = '0.0.1'
__maintainer__ = 'Benjamin Paaßen'
__email__  = 'benjamin.paassen@sydney.edu.au'

def generate_data(N, task_name):
    """ A utility function to call data generation functions with
    standard parameters.

    Parameters
    ----------
    N: int
        The number of sequences to be generated.
    task_name: str
        The name of the task (either 'latch', 'copy', 'repeat_copy', or
        'signal_copy').

    Returns
    -------
    Xs: list
        A N-element list of input sequences for the task.
    Qs: list
        A N-element list of state sequences for the task.
    Ys: list
        A N-element list of output sequences for the task.

    """
    if task_name == 'latch':
        return latch_task(N)
    elif task_name == 'copy':
        return copy_task(N)
    elif task_name == 'repeat_copy':
        return repeat_copy_task(N)
    elif task_name == 'signal_copy':
        return signal_copy_task(N)
    elif task_name == 'image_copy':
        return image_copy_task(N)
    else:
        raise ValueError('Unknown task: %s' % task_name)


def latch_task(N, max_len = 200, num_switches = 3, force_length = False):
    """ Generates N sequences for the latch task.

    The latch task is originally due to
    `Paaßen and Schulz (2020) <https://arxiv.org/abs/2003.04793>`.
    The input is a sequence of zeros, interrupted only by very few ones.
    The output should be zero until the first ones, then one until the next
    one, then back to zero, and so on.

    Note that it is ensured that equally many sequences for each
    length are generated.

    Parameters
    ----------
    N: int
        The number of sequences to be generated.
    max_len: int (default = 200)
        The maximum length of the input sequence.
    num_switches: int (default = 3)
        The number of ones in the input.
    force_length: bool (default = False)
        If set to true, all sequences have exactly max_len.

    Returns
    -------
    Xs: list
        The list of input sequences, each of size T x 1 with
        T in [num_switches * 3, max_len]. X is zero except at
        num_switches random locations where it is one.
    Qs: list
        The list of state sequences, each of length T. This is just
        a copy of Ys.
    Ys: list
        The list of output sequences of size T x 1. Each output
        sequence is zero until the first one in the input, then one
        until the next one, then back to zero, and so forth.

    """
    # start by generating the sequence lengths.
    if force_length:
        Ts = np.full(N, max_len, dtype=int)
    else:
        Ts = _permutation_sampling(N, num_switches * 3, max_len)
    # then generate the sequences
    Xs = []
    Qs = []
    Ys = []
    for j in range(N):
        T = Ts[j]
        # initialize the input and output sequence
        X = np.zeros((T, 1))
        Y = np.zeros((T, 1))
        # sample the switching points
        t_last = 0
        for j in range(num_switches):
            lo = int(T / num_switches) * j + int(0.25 * T / num_switches)
            hi = int(T / num_switches) * (j+1) - int(0.25 * T / num_switches)
            t  = random.randrange(lo, hi)
            X[t, :] = 1.
            Y[t_last:t, :] = j % 2
            t_last = t
        Y[t_last:, :] = num_switches % 2
        # make Q the copy of Y
        Q = Y.squeeze() + 1
        # append to output
        Xs.append(X)
        Qs.append(Q)
        Ys.append(Y)
    # return
    return Xs, Qs, Ys

def copy_task(N, n = 8, max_len = 20, force_length = False):
    """ Generates N sequences for the copy task.

    The copy task was originally devised by
    `Graves, Wayne, and Danihelka (2014) <https://arxiv.org/abs/1410.5401>`.
    Each input is a random bit sequences which should be written twice to
    the output. To mark beginning and end of the sequence we have a one on
    an extra input channel.

    Note that it is ensured that equally many sequences for each
    length are generated.

    Parameters
    ----------
    N: int
        The number of sequences to be generated.
    n: int (default = 8)
        The number of content dimensions.
    max_len: int (default = 20)
        The maximum length of the input sequence to be copied.
    force_length: bool (default = False)
        If set to true, all sequences have exactly max_len.

    Returns
    -------
    Xs: list
        The list of input sequences, each of size (2*T+2) x (n+1) with
        T in [1, max_len]. X[0, :n] will be zero, X[0, n] will be 1,
        X[1:T+1, :n] will be filled with random bits, and X[T+1:, :]
        will be zero except for X[T+1, n] = 1.
    Qs: list
        The list of state sequences, each of length 2*T+2. The sequences
        will be 1, ..., T twice.
    Ys: list
        The list of output sequences of size (2*T+2) x n. Y is twice
        X[:T+1, :n].

    """
    # start by generating the sequence lengths.
    if force_length:
        Ts = np.full(N, max_len, dtype=int)
    else:
        Ts = _permutation_sampling(N, 1, max_len)
    # then generate the sequences
    Xs = []
    Qs = []
    Ys = []
    for j in range(N):
        T = Ts[j]
        # initialize the input and output sequence
        X = np.zeros((2*T+2, n+1))
        Y = np.zeros((2*T+2, n))
        # fill the input sequence with random bits
        X[1:T+1, :n] = np.round(np.random.rand(T, n))
        # set a start of sequence marker
        X[0, n] = 1.
        # set an end-of-sequence marker
        X[T+1, n] = 1.
        # copy the input to the output
        Y[1:T+1, :] = X[1:T+1, :n]
        Y[T+2:, :]  = X[1:T+1, :n]
        # set desired states
        Q = np.tile(np.arange(1, T+2), 2)
        # append to output
        Xs.append(X)
        Qs.append(Q)
        Ys.append(Y)
    # return
    return Xs, Qs, Ys

def repeat_copy_task(N, n = 8, max_len = 10, max_repeats = 10, force_length = False):
    """ Generates N sequences for the repeat copy task.

    The repeat copy task was originally devised by
    `Graves, Wayne, and Danihelka (2014) <https://arxiv.org/abs/1410.5401>`.
    Each input is a random bit sequences which should be written multiple
    times to the output. The start of the sequence and each new repeat is
    marked with a one on an extra input channel.

    Note that it is ensured that equally many sequences for each
    length and number of repeats are generated.

    Parameters
    ----------
    N: int
        The number of sequences to be generated.
    n: int (default = 8)
        The number of content dimensions.
    max_len: int (default = 10)
        The maximum length of the input sequence to be copied.
    max_repeats: int (defualt = 10)
        The maximum number of times the input sequence is repeated.
    force_length: bool (default = False)
        If set to true, all sequences have exactly max_len and are
        repeated exactly max_repeats times.

    Returns
    -------
    Xs: list
        The list of input sequences, each of size ((R+1)*(T+1)) x (n+1) with
        R in [1, max_repeats] and T in [1, max_len].
        X[0, :], X[T+1, :], X[2*(T+1), :], ... will be a zero vector with
        one on the last channel. X[1:T+1, :n] is a sequence of random bits.
        All other entries of X are zero.
    Qs: list
        The list of state sequences, each of length (R+1)*(T+1). Each state
        sequence is 1, ..., T+1 repeated R+1 times.
    Ys: list
        The list of output sequences of size ((R+1)*(T+1)) x n. Y is
        R+1 times the sequence X[:T+1, :n].

    """
    # start by generating the sequence lengths and number of repeats.
    if force_length:
        Ts = np.full(N, max_len, dtype=int)
        Rs = np.full(N, max_repeats, dtype=int)
    else:
        Ts = _permutation_sampling(N, 1, max_len)
        Rs = _permutation_sampling(N, 1, max_repeats)
    # then generate the sequences
    Xs = []
    Qs = []
    Ys = []
    for j in range(N):
        T = Ts[j]
        R = Rs[j]
        # initialize the input and output sequence
        X = np.zeros(((R+1)*(T+1), n+1))
        Y = np.zeros(((R+1)*(T+1), n))
        # fill the input sequence with random bits
        X[1:T+1, :n] = np.round(np.random.rand(T, n))
        # and copy the sequence R+1 times to the output
        for r in range(R+1):
            lo = r*(T+1)
            hi = (r+1)*(T+1)
            # write a marker to the input
            X[lo, n] = 1.
            # copy the sequence to the output
            Y[lo+1:hi, :] = X[1:T+1, :n]
        # generate the state sequence
        Q = np.tile(np.arange(1, T+2), R+1)
        # append to output
        Xs.append(X)
        Qs.append(Q)
        Ys.append(Y)
    # return
    return Xs, Qs, Ys

def signal_copy_task(N, T_signal = 256, max_repeats = 10, force_length = False):
    """ Generates N sequences for the signal copy task.

    This task is newly devised for this paper. The input consists of two
    random wavelets on the first channel which are associated with markers
    on the second input channel. Then, the second channel contains repititions
    of these markers, whereupon the output should be a copy of the random wavelet
    associated with it.

    Parameters
    ----------
    N: int
        The number of sequences to be generated.
    T_signal: int (default = 256)
        The length of each signal block (must be at least 32 because
        that is the marker length).
    max_repeats: int (default = 10)
        The maximum number of times an input sequence is recalled from
        memory.
    force_length: bool (default = False)
        If set to true, all sequences have exactly max_repeats repeats.

    Returns
    -------
    Xs: list
        The list of input sequences, each of size T_signal * (R+3) x 2 with
        R in [1, max_repeats]. X[:T_signal, 0] and X[T_signal:2*T_signal, 0]
        are two random (but smooth) wavelets generated as affine combinations
        of sine waves of varying frequencies. X[T_signal-32:T_signal, 1]
        and X[2*T_signal-32:2*T_signal, 1] are marker signals of length 32.
        These two first input blocks serve as presentation of the input
        signals. All R remaining blocks are zero on the first channel and
        have one of the two marker signals on channel 1.
    Qs: list
        The list of state sequences, each of length T_signal * (R+3).
        The state is zero except at the end of each block of length T_signal
        where it is the index of the marker signal on the first input channel.
    Ys: list
        The list of output sequences of size T_signal * (R+3) x 1.
        In each block of length T_signal, the output is the input wavelet
        corresponding to the last marker.

    """
    # start by generating the number of repeats
    if force_length:
        Rs = np.full(N, max_repeats, dtype=int)
    else:
        Rs = _permutation_sampling(N, 1, max_repeats)
    # set up auxiliary concepts, in particular the markers and sine waves
    # of random frequencies
    T_marker = 32
    markers = [
        np.square(np.sin(np.arange(T_marker) / (T_marker - 1) * 2 * np.pi)),
        np.square(np.sin(np.arange(T_marker) / (T_marker - 1) * 4 * np.pi))
    ]
    for s in range(len(markers)):
        markers[s][-1] = -1.

    num_freqs = 8
    Sines = np.zeros((num_freqs, T_signal))
    for f in range(num_freqs):
        Sines[f, :] = np.sin(f*np.arange(T_signal)/T_signal*np.pi)

    # then generate the sequences
    Xs = []
    Qs = []
    Ys = []
    for j in range(N):
        num_repeats = Rs[j]
        num_signals = len(markers)
        # initialize the input and output sequence
        T = (num_signals + num_repeats + 1) * T_signal
        X = np.zeros((T, 2))
        Q = np.zeros(T)
        Y = np.zeros((T, 1))
        # do the initial signal presentations
        for s in range(num_signals):
            # generate a random signal via a random affine combination of sine waves
            coefs = np.random.randn(num_freqs)
            coefs /= np.sum(coefs)
            signal = np.dot(coefs, Sines)
            # put the signal on the first input channel
            lo = s*T_signal
            hi = (s+1)*T_signal
            X[lo:hi, 0] = signal
            # put the s-th marker on the second input channel
            # after the signal is completed
            X[hi-T_marker:hi, 1] = markers[s]
            # mark the state
            Q[hi] = s + 1
            # put the signal to the output as well
            Y[hi:hi+T_signal, 0] = signal
        # generate the output and state sequence
        for r in range(num_repeats):
            lo = (num_signals+r+1)*T_signal
            hi = (num_signals+r+2)*T_signal
            # sample the signal we wish to repeat
            s  = random.randrange(num_signals)
            # mark the state
            Q[lo] = s + 1
            # write the s-th marker on the second input channel
            X[lo-T_marker:lo, 1] = markers[s]
            # write the s-th signal on the output channel
            Y[lo:hi, 0] = X[s*T_signal:(s+1)*T_signal, 0]
        # append to output
        Xs.append(X)
        Qs.append(Q)
        Ys.append(Y)
    # return
    return Xs, Qs, Ys

_MNIST_SIZE_ = 28
_MNIST_ROWS_ = 70000

def image_copy_task(N, max_repeats = 10, force_length = False, sharpen = False, data_path = 'mnist_784.csv'):
    """ Generates N sequences for the image copy task.

    This task is newly devised for this paper. The input consists of
    a random MNIST image which shall be recalled whenever a special token
    occurs on the input.

    Note that this function requires the file 'mnist_784.csv' from
    https://www.openml.org/d/554 .

    Parameters
    ----------
    N: int
        The number of sequences to be generated.
    max_repeats: int (default = 10)
        The maximum number of times the input image shall be copied.
    force_length: bool (default = False)
        If set to true, all sequences have exactly max_repeats repeats.
    data_path: str (default = 'mnist_784.csv')
        Path to the mnist data set as csv file, where each image is a row
        with 785 entries, the last entry being the label.

    Returns
    -------
    Xs: list
        The list of input sequences, each of size 28 * (R+1) x 28 with
        R in [1, max_repeats]. X[:28, :] contains the MNIST image and
        X[28*r, :] for all r in {1, ..., R} contains a special marker token,
        and X is zero everywhere else.
    Qs: list
        The list of state sequences, each of length 28 * (R+1)
        The state is zero except at positions 28, 56, ..., R*28 where
        it is one.
    Ys: list
        The list of output sequences of size 28 * (R+1) x 28.
        In each block of length 28, the output is a copy of the MNIST image
        X[:28, :].

    """
    # start by generating the number of repeats
    if force_length:
        Rs = np.full(N, max_repeats, dtype=int)
    else:
        Rs = _permutation_sampling(N, 1, max_repeats)
    # select a random image for each sequence to be generated
    subset = np.random.choice(_MNIST_ROWS_, size = N, replace = False)
    # load these images from CSV
    Images = np.zeros((N, _MNIST_SIZE_, _MNIST_SIZE_))
    with open(data_path) as f:
        l = 0
        for j in np.argsort(subset):
            while l < subset[j]:
                f.__next__()
                l += 1
            line = np.fromstring(f.__next__(), dtype=int, sep=',')
            l += 1
            Images[j, :, :] = np.reshape(line[:-1], (_MNIST_SIZE_, _MNIST_SIZE_))
    # if so desired, we sharpen the images to be either 0 or 1 at each point
    if sharpen:
        Images[Images < 128] = 0
        Images[Images >= 128] = 1
    # start generating sequences
    Xs = []
    Qs = []
    Ys = []
    for j in range(N):
        # initialize the input and output sequence
        T = (Rs[j] + 1) * _MNIST_SIZE_
        X = np.zeros((T, _MNIST_SIZE_))
        Q = np.zeros(T)
        Y = np.zeros((T, _MNIST_SIZE_))
        # put the current image at the start of X
        X[:_MNIST_SIZE_, :] = Images[j, :, :]
        # then construct the rest of the sequence
        for r in range(Rs[j]):
            lo = (r+1)*_MNIST_SIZE_
            hi = (r+2)*_MNIST_SIZE_
            # put a special marker token at X[r*_MNIST_SIZE_, :] for each repititon
            if sharpen:
                X[lo, :] = 1
            else:
                X[lo, :] = 256
            # recall state 1 whenever that occurs
            Q[lo] = 1
            # and copy the image to the output
            Y[lo:hi, :] = Images[j, :, :]
        # append to training data
        Xs.append(X)
        Qs.append(Q)
        Ys.append(Y)

    return Xs, Qs, Ys

def _permutation_sampling(N, min_param, max_param):
    """ Samples an integer-valued parameter N times such that each possible
    parameter value occurs equally often.

    Parameters
    ----------
    N: int
        The number of samples.
    min_param: int
        The minimum parameter value.
    max_param: int
        The maximum parameter value (inclusive).

    Returns
    -------
    samples: class numpy.array
        The output integer array of size N containing all sampled values.

    """
    if max_param < min_param:
        raise ValueError('Maximum parameter value is smaller than minimum parameter value.')
    samples = np.zeros(N, dtype=int)
    param_range = max_param+1-min_param
    lo = 0
    hi = param_range
    while hi < N:
        samples[lo:hi] = np.random.permutation(param_range) + min_param
        lo += param_range
        hi += param_range
    if lo < N:
        samples[lo:] = np.random.choice(param_range, size = N - lo, replace = False) + min_param
    return samples
