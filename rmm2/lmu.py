"""
Implements Legendre delay networks as proposed by Voelker, Kajić, and Eliasmith
(2019).

https://papers.nips.cc/paper/9689-legendre-memory-units-continuous-time-representation-in-recurrent-neural-networks

"""

# Copyright (C) 2020
# Terrence C. Stewart
# Research Council of Canada and
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

__author__ = 'Terrence C. Stewart, Benjamin Paaßen'
__copyright__ = 'Copyright 2020, Terrence C. Stewart, Benjamin Paaßen'
__license__ = 'GPLv3'
__Version__ = '0.0.1'
__maintainer__ = 'Benjamin Paaßen'
__email__  = 'benjamin.paassen@sydney.edu.au'

import numpy as np
from scipy.linalg import expm
from scipy.sparse import bsr_matrix
from scipy.special import legendre

def initialize_reservoir(n, degree, T):
    """ Sets up a linear dynamical system which optimally stores the
    past theta time steps in memory for each input. Note that this assumes
    smoothness in the input signal. If the signal is not smooth, one still
    requires as many neurons as time steps.

    Parameters
    ----------
    n: int
        The number of input channels.
    degree: int
        The maximum degree of Legendre polynomials to represent the input
        signal. The memory dimension will be n * (degree + 1).
    T: int
        The length of the time window which we wish to store. Note that a
        signal beyond length T can _not_ be reconstructed because Legendre
        polynomials do not extrapolate. On the other hand, setting T too long
        means that less high-frequency polynomials are available. Therefore,
        it is crucial to set T 'just right' for your dataset.

    Returns
    -------
    U: class numpy.array
        The input-to-memory connection matrix.
    W: class scipy.sparse.csr_matrix
        The memory-to-memory/recurrent connection matrix.

    """

    q = degree+1

    # set up the continuous linear system according to the definitions in the
    # paper
    A = np.zeros((q, q))
    b = np.zeros(q)
    for i in range(q):
        coeff = (2*i+1)
        A[i, i+1:] = -coeff
        if i % 2 == 0:
            A[i, 1:i+1:2] = coeff
            A[i, :i+1:2] = -coeff
            b[i] = coeff
        else:
            A[i, 1:i+1:2] = -coeff
            A[i, :i+1:2] = coeff
            b[i] = -coeff

    # Handle the fact that we're discretizing the time step
    # https://en.wikipedia.org/wiki/Discretization#Discretization_of_linear_state_space_models
    Ad = expm(A / T)
    bd = np.dot(np.dot(np.linalg.inv(A), (Ad-np.eye(q))), b)

    # To generate the output, we need to copy these matrices n times.
    m = q * n
    U = np.zeros((m, n))
    for k in range(n):
        U[k*q:(k+1)*q, k] = bd
    # To set up W we use a sparse matrix representation
    rows = []
    cols = []
    data = []
    for k in range(n):
        for i in range(q):
            for j in range(q):
                rows.append(k*q + i)
                cols.append(k*q + j)
                data.append(Ad[i, j])
    W = bsr_matrix((data, (rows, cols)), blocksize = (q, q), shape = (m, m))

    return U, W

def reconstruct_signal(h, n, T, tau = None):
    """ Reconstructs a signal as a linear combination of Legendre polynomials
    over the time window tau.

    Note that this reconstruction tends to only work well over sufficiently
    long time windows and for sufficient smoothness in the input signal.
    If the signal is non-smooth, other reconstructions may be better.

    Parameters
    ----------
    h: class numpy.array
        An array of linear coefficients for the Legendre polynomials up to
        degree len(h)/n-1.
    n: int
        The dimensionality of the original signal.
    T: int
        The time window over which the Legendre polynomials are defined.
    tau: int (default = T)
        The time window over which the signal should be reconstructed.

    """
    if tau is None:
        tau = T

    # recover q
    if len(h) % n != 0:
        raise ValueError('len(h) = %d is not divisble by n = %d' % (len(h), n))

    q = int(len(h)/n)

    # compute the legendre polynomials in the desired range
    ts = np.arange(1, tau+1)
    L  = np.zeros((q, len(ts)))
    for p in range(q):
        # note that we use input -ts here because we look 'into the past'
        L[p, :] = legendre(p)(-2*ts/T+1)

    # then, the original signal should be recoverable by multiplying the
    # coefficients in h with the Legendre polyonomials
    Y = np.zeros((tau, n))
    for k in range(n):
        Y[:, k] = np.dot(h[k*q:(k+1)*q], L)
    # return the reconstructed signal
    return Y
