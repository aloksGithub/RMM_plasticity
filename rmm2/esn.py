"""
Provides echo state network classes according to the scikit-learn
interface.

"""

# Copyright (C) 2019
# Benjamin Paaßen, Alexander Schulz
# AG Machine Learning
# Bielefeld University

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

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, RegressorMixin

__author__ = 'Benjamin Paaßen'
__copyright__ = 'Copyright 2019-2020, Benjamin Paaßen, Alexander Schulz'
__license__ = 'GPLv3'
__Version__ = '1.0.0'
__maintainer__ = 'Benjamin Paaßen'
__email__  = 'bpaassen@techfak.uni-bielefeld.de'

def linreg(H, Y, regul):
    """ Applies linear regression to map the representation time series
    Phi to the output time series Y.

    Parameters
    ----------
    H: class numpy.array
        A T x m representation time series.
    Y: class numpy.array
        A T x K output time series.
    regul: float
        A L2 regularization strength.

    Returns
    -------
    V: class numpy.array
        A K x m matrix mapping representation to output dimensions,
        such that np.dot(H, V.T) is as similar as possible to Y.

    """
    C    = np.dot(H.T, H)
    Cinv = np.linalg.inv(C + regul * np.eye(H.shape[1]))
    Cout = np.dot(Y.T, H)
    return np.dot(Cout, Cinv)

def initialize_reservoir(m, n, radius, sparsity):
    """ Initializes a standard echo state network reservoir where U is a dense
    m x n matrix of Gaussian random numbers with standard deviation radius and
    W is a sparse m x m matrix with only a sparsity fraction of nonzero
    entries and spectral radius radius.

    Parameters
    ----------
    m: int
        the number of neurons.
    n: int
        the number of input channels.
    radius: float in range (0, 1.)
        The spectral radius of the recurrent matrix. Is enforced via eigenvalue
        decomposition.
    sparsity: float in range (0, 1]
        The fraction of nonzero elements in W.

    Returns
    -------
    U: class numpy.array
        An m x n matrix of Gaussian random numbers with zero mean and standard
        deviation radius.
    W: class scipy.sparse.csr_matrix
        A m x m sparse matrix of Gaussian random numbers, only sparsity*m^2 of
        which are nonzero, and with a spectral radius of radius.

    """
    # set up U
    U = np.random.randn(m, n) * radius
    # set up W
    W = np.zeros((m, m))
    nonzeros = max(1, int(sparsity * m))
    for i in range(m):
        # select the nonzero positions in row i at random
        pos = np.random.choice(m, size=nonzeros, replace = False)
        # initialize randomly
        W[i, pos] = np.random.randn(nonzeros)
    # compute the actual spectral radius
    rho = np.max(np.abs(np.linalg.eigvals(W)))
    # normalize the matrix by that and multiply with the desired spectral
    # radius
    W *= radius / rho
    # transform W into a sparse matrix
    W = csr_matrix(W)
    return U, W


class ESN(BaseEstimator, RegressorMixin):
    """ Implements an echo state network (Jaeger and Haas, 2004) with a
    given reservoir (or a standard ESN reservoir per default).

    In the following, let m be the number of neurons and n be the number of
    input channels.

    Parameters
    ----------
    U: class numpy.array OR int
        A m x n input-to-neurons connection matrix OR the number of neurons.
        In the latter case, this becomes
        initialize_reservoir(U, W, 0.9, 0.1)[0].
    W: class scipy.sparse.csr_matrix OR int
        A m x m sparse neuron-to-neuron/recurrent connection matrix OR the
        number of input channels. In the latter case, this becomes
        initialize_reservoir(U, W, 0.9, 0.1)[1].
    leak: float in range (0., 1.] (default = 1.)
        The 'leak rate' alpha, i.e. what fraction of the current memory
        content is overriden by the new values. This parameter can be used
        to 'smoothen' the reaction of the net, i.e. if the time scale of the
        output curve is much lower than the time scale of the input curve.
    regul: positive float (default = 1E-5)
        The L2 regularization strength for linear regression.
    input_normalization: bool (default = True)
        If True, the input is z-normalized before feeding it into the network.
        The mean and standard deviation are adjusted to the training data.
    washout: int (default = 0)
        The number of washout steps before training.
    nonlin: function (default = numpy.tanh)
        An elementwise nonlinearity.

    Attributes
    ----------
    V_: class numpy.array
        The reservoir-to-output matrix. Is set during training via linear
        regression.
    mu_: class numpy.array
        The mean of the training data for each feature. This is used for
        input_normalization. Is set during training.
    beta_: class numpy.array
        The precision of the training data for each feature.
        This is used for input_normalization. Is set during training.

    """
    def __init__(self, U, W, leak = 1., regul = 1E-5, input_normalization = True,
        washout = 0, nonlin = np.tanh):
        if isinstance(U, int):
            if not isinstance(W, int):
                raise ValueError('If U is a single integer, W must be as well.')
            m = U
            n = W
            self.U, self.W = initialize_reservoir(m, n, 0.9, 0.1)
        else:
            self.U = U
            self.W = W
        self.leak = leak
        self.regul = regul
        self.input_normalization = input_normalization
        self.washout = washout
        self.nonlin = nonlin

    def _apply_reservoir(self, X):
        """ Applies this networks reservoir to the given normalized input time series. """
        T = X.shape[0]
        # initialize output matrix
        H = np.zeros((T, self.U.shape[0]))
        # compute retain rate
        retain = 1. - self.leak
        if retain < 1E-3:
            retain = 0.
        # compute first state
        H[0, :] = self.nonlin(np.dot(self.U, X[0, :]))
        # compute remaining states
        for t in range(1, T):
            H[t, :] = self.leak * self.nonlin(np.dot(self.U, X[t, :]) + self.W * H[t-1, :]) + retain * H[t-1, :]

        return H

    def fit(self, X, Y):
        """ Fits this echo state network to the given data.

        T and K are arbitrary integers.

        Parameters
        ----------
        X: class numpy.array OR list
            A T x self.U.shape[1] input time series or a list of such matrices.
        Y: class numpy.array OR list
            A T x K output time series matrix or a list of such matrices.

        Returns
        -------
        self: class ESN
            this instance.

        """
        # prepare the input matrix
        if isinstance(X, list):
            Xs = X
        else:
            Xs = [X]
        # prepare input normalization
        n = Xs[0].shape[1]
        self.mu_   = np.zeros(n)
        self.beta_ = np.zeros(n)
        if self.input_normalization:
            # compute the mean first
            T = 0
            for X in Xs:
                T += len(X)
                self.mu_ += np.sum(X, axis=0)
            self.mu_ /= T
            self.mu_ = np.expand_dims(self.mu_, 0)
            # then compute precision
            for X in Xs:
                self.beta_ += np.sum(np.square(X - self.mu_), axis=0)
            self.beta_[self.beta_ < 1E-3] = T
            self.beta_ = np.expand_dims(np.sqrt(T / self.beta_), 0)
        else:
            self.mu_ = 0.
            self.beta_ = 1.

        # process all input matrices with a cycle reservoir with jumps
        # and concatenate them to one big output
        Hs = []
        for X in Xs:
            H = self._apply_reservoir((X - self.mu_) * self.beta_)
            # remove washout steps before recording the data
            Hs.append(H[self.washout:, :])
        H = np.concatenate(Hs, axis=0)

        # prepate the output
        if isinstance(Y, list):
            Ys = Y
        else:
            Ys = [Y]
        for j in range(len(Ys)):
            # remove the washout steps from each output matrix
            Ys[j] = Ys[j][self.washout:, :]
        Y = np.concatenate(Ys, axis=0)

        # after this preparation, we can perform linear regression to
        # generate the reservoir-to-output matrix
        self.V_ = linreg(H, Y, self.regul)

        return self

    def predict(self, X):
        """ Predicts on the given input time series using this ESN.

        Parameters
        ----------
        X: class numpy.array
            A T x self._U.shape[1] input time series.

        Returns
        -------
        Y: class numpy.array
            A T x len(self.V_) output time series.

        """
        # compute the reservoir representation after input normalization
        H = self._apply_reservoir((X - self.mu_) * self.beta_)
        # apply the output matrix.
        Y = np.dot(H, self.V_.T)
        # return
        return Y
