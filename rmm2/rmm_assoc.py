"""
Implements reservoir memory machines with associative memory.

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
from scipy.sparse import csr_matrix
from scipy.optimize import linprog
from scipy.spatial.distance import cdist
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.decomposition import PCA
from scipy.optimize import minimize
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from rmm2.pseudo_svm import PseudoSVM
import rmm2.esn as esn

__author__ = 'Benjamin Paaßen'
__copyright__ = 'Copyright 2020, Benjamin Paaßen'
__license__ = 'GPLv3'
__Version__ = '0.0.1'
__maintainer__ = 'Benjamin Paaßen'
__email__  = 'benjamin.paassen@sydney.edu.au'

class RMM(BaseEstimator, RegressorMixin):
    """ A reservoir memory machine which can losslessly recover past states.

    This architecture is an extension of an echo state network (Jaeger and
    Haas, 2004) to make it at least as powerful as a finite state machine.
    In particular, we equip a standard echo state network with an additional
    state set Q as well as a classifier q that maps from the state set of the
    echo state network into Q (plus a special default state). In each inference
    step, we then apply q to the current ESN state h_t to infer a discrete
    state q_t of our machine. If we did not see q_t before, we put h_t into
    memory. If we did q_t before, we override the current h_t with our stored
    value for q_t. In this fashion, we can losslessly recover past states,
    which in turn enables us to implement any finite state machine.

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
    nonlin: function (default = numpy.tanh)
        An elementwise nonlinearity.
    discrete_prediction: bool (default = False)
        A flag that marks wether the output should be a continuous prediction
        via linear regression or a discrete prediction via linear SVM.
    C: positive float (default = 100.)
        The regularization parameter for the state classifier.
    q_0: int (default = 0)
        The initial discrete state, i.e. whether the initial continuous state
        should be saved in memory.
    svm_kernel: string (default = 'linear')
        The kernel that the SVM should use to predict the write head, either
        'linear', 'rbf', or 'pseudo'.
        For a linear kernel, we use the sklearn LAPACK implementation which
        should guarantee linear scaling of training times with the data set
        size, whereas the rbf kernel may induce quadratic or even cubic
        runtimes. If 'pseudo' is selected, we use a custom SVM implementation
        based on linear regression which is guaranteed linear (and fast) in the
        number of data points but uses primitive linear regression as a
        surrogate of the decision function.
    init_state: class numpy.array (default = numpy.zeros(U.shape[0]))
        The initial continuous state.
    horizon: int (default = max(T))
        How far into the past association operators may look.
    max_negatives: int (default = 500)
        How many negative examples we use at most as reference for association
        learning. Smaller numbers make the result less reliable, but speed up
        computation.

    Attributes
    ----------
    V_: class numpy.array OR list
        The reservoir-to-output matrix OR a list of linear SVMs, one per
        output channel. Is set during training.
    c_: class sklearn.svm.LinearSVC
        The initial state classifier. Is set during training.
    memory_size_: int
        The size of the state memory. Is set during training.
    mu_: class numpy.array
        The mean of the training data for each feature. This is used for
        input_normalization.
    beta_: class numpy.array
        The precision of the training data for each feature.
        This is used for input_normalization.
    self.mh_: class numpy.array
        The mean of all continuous state vectors that should be associated.
    self.Wh_: class numpy.array
        The self.U.shape[0] x self.horizon linear transformation from the
        continuous state space to the latent space in which we perform
        associations.
    self.mm_: class numpy.array
        The mean of all vectors in memory that should be associated.
    self.Wm_: class numpy.array
        The self.U.shape[0] x self.horizon linear transformation from the
        memory to the latent space in which we perform associations.
    self.assoc_threshold_: float
        The association threshold. Whenever a continuous state vector is closer
        to a memory vector in latent space than this threshold, we override the
        continuous state with the memory vector.

    """
    def __init__(self, U, W, leak = 1., regul = 1E-5, input_normalization = True,
        nonlin = np.tanh, discrete_prediction = False, C = 100., q_0 = 0, svm_kernel = 'linear',
        init_state = None, horizon = None, max_negatives = 500):

        if isinstance(U, int):
            if not isinstance(W, int):
                raise ValueError('If U is a single integer, W must be as well.')
            m = U
            n = W
            self.U, self.W = esn.initialize_reservoir(m, n, 0.9, 0.1)
        else:
            self.U = U
            self.W = W
        self.leak = leak
        self.regul = regul
        self.input_normalization = input_normalization
        self.nonlin = nonlin
        self.discrete_prediction = discrete_prediction
        self.C = C
        self.q_0 = q_0
        self.svm_kernel = svm_kernel
        if init_state is None:
            self.init_state = np.zeros(self.U.shape[0])
        else:
            self.init_state = init_state
        self.horizon = horizon
        self.max_negatives = max_negatives

    def fit(self, X, Q, Y, verbose = False):
        """ Fits this reservoir memory machine to the given data.

        Parameters
        ----------
        X: class numpy.array OR list
            A T x n input time series or a list of such matrices.
        Q: class numpy.array OR list
            A T-element auxiliary time series of states or a list of such
            time series.
        Y: class numpy.array OR list
            A T x K output time series matrix or a list of such matrices.

        Returns
        -------
        self: class RMM
            this instance.

        """

        # prepare the input matrix
        if isinstance(X, list):
            Xs = X
            Qs = Q
            Ys = Y
            if len(Xs) != len(Ys):
                raise ValueError('Expected as many output as input time series, but got %d input time series and %d output time series' % (len(Xs), len(Ys)))
            if len(Xs) != len(Qs):
                raise ValueError('Expected as many state as input time series, but got %d input time series and %d state time series' % (len(Xs), len(Qs)))
        else:
            Xs = [X]
            Qs = [Q]
            Ys = [Y]

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

        # pre-search the number of unique states
        unique_states = np.unique(Qs[0])
        for i in range(1, len(Qs)):
            unique_states = np.union1d(unique_states, Qs[i])

        if 0 in unique_states:
            self.memory_size_ = len(unique_states)-1
        else:
            self.memory_size_ = len(unique_states)

        # compute retain rate
        retain = 1. - self.leak
        if retain < 1E-3:
            retain = 0.

        # set horizon
        if self.horizon is None:
            self.horizon = 0
            for i in range(len(Xs)):
                self.horizon = max(self.horizon, len(Xs[i]))

        # process all input matrices via teacher forcing and concatenate them
        # to one big state time series; in particular one for the initial state
        # classifier, one for the associative state classifier, and one for
        # the output.
        Hs_init  = []
        Qs_init  = []
        Hs_assoc = []
        Ms_assoc = []
        Qs_assoc = []
        Hs = []
        for i in range(len(Xs)):
            X = (Xs[i] - self.mu_) / self.beta_
            Q = Qs[i]
            T = len(X)
            if len(Q) != T:
                raise ValueError('%dth state series had length %d but the input time series had length %d.' % (i, len(Q), T))
            if len(Ys[i]) != T:
                raise ValueError('%dth output series had length %d but the input time series had length %d.' % (i, len(Ys[i]), T))

            # initialize state matrices
            H_raw = np.zeros((T, self.U.shape[0]))
            H     = np.zeros((T, self.U.shape[0]))
            # initialize targets for initial state classifier
            Q_init = []
            # initialize memory
            M = {}
            if self.q_0 > 0:
                M[self.q_0] = self.init_state
            # compute time steps
            for t in range(T):
                # compute the current raw state
                if t == 0:
                    H_raw[t, :] = self.leak * self.nonlin(np.dot(self.U, X[t, :]) + self.W * self.init_state) + retain * self.init_state
                else:
                    H_raw[t, :] = self.leak * self.nonlin(np.dot(self.U, X[t, :]) + self.W * H[t-1, :]) + retain * H[t-1, :]

                # check if it needs to be overridden by memory
                written_to_memory = False
                if Q[t] > 0:
                    if Q[t] in M:
                        # if we have already a state associated with q_t, recover
                        # that state
                        H[t, :] = M[Q[t]]
                        # use the raw H as training data for the association
                        # mechanism
                        Hs_assoc.append(H_raw[t, :])
                        Ms_assoc.append(M[Q[t]])
                        Qs_assoc.append(+1.)
                    else:
                        # otherwise, memorize the current state
                        H[t, :] = H_raw[t, :]
                        M[Q[t]] = H[t, :]
                        written_to_memory = True
                else:
                    H[t, :] = H_raw[t, :]
                # write negative training data for the association mechanism
                if not written_to_memory:
                    for key in M:
                        if key == Q[t]:
                            continue
                        Hs_assoc.append(H_raw[t, :])
                        Ms_assoc.append(M[key])
                        Qs_assoc.append(-1.)
                # write training data for initial state classifier
                if written_to_memory:
                    Q_init.append(+1)
                elif len(M) < self.memory_size_:
                    Q_init.append(-1)

            # append time series to the output
            Hs_init.append(H_raw[:len(Q_init),:])
            Qs_init.append(np.array(Q_init))
            Hs.append(H)

        # concatenate all time series
        H_init  = np.concatenate(Hs_init, 0)
        Q_init  = np.concatenate(Qs_init, 0)
        if len(Hs_assoc) > 0:
            H_assoc = np.stack(Hs_assoc, 0)
            M_assoc = np.stack(Ms_assoc, 0)
            Q_assoc = np.stack(Qs_assoc, 0)
        H     = np.concatenate(Hs, 0)

        # train the initial state SVM
        if self.svm_kernel == 'linear':
            self.c_ = LinearSVC(dual = False, C = self.C, max_iter = 10000)
        elif self.svm_kernel == 'rbf':
            self.c_ = SVC(C = self.C, max_iter = 10000)
        elif self.svm_kernel == 'pseudo':
            self.c_ = PseudoSVM(regul = self.regul)
        else:
            raise ValueError('Unsupported SVM kernel: %s' % self.svm_kernel)


        self.c_ = LinearSVC(dual = False, C = self.C, max_iter = 10000)
        if len(np.unique(Q_init)) == 1:
            # if we always write (or never), use a constant predictor
            self.c_.predict = lambda X : np.full(X.shape[0], Q_init[0])
        else:
            self.c_.fit(H_init, Q_init)

            acc = np.mean(Q_init == self.c_.predict(H_init))
            if verbose:
                print('write head accuracy: %g' % acc)

        # train the associative mechanism
        if np.any(Q_assoc > 0):
            # prepare linear operators which map from the current memory state
            # to the input tau steps ago for tau < self.horizon

            # to do so, we first compute the state sequences for all time series
            Hs_raw = []
            for i in range(len(Xs)):
                X = Xs[i]
                H_raw = []
                h = self.init_state
                for t in range(len(Xs[i])):
                    h = self.leak * self.nonlin(np.dot(self.U, X[t, :]) + self.W * h) + retain * h
                    H_raw.append(h)
                Hs_raw.append(np.stack(H_raw))

            # then iterate over all tau
            operators = []
            for tau in range(self.horizon):
                # accumulate the training data for the current tau
                H_tau = []
                X_tau = []
                for i in range(len(Xs)):
                    if tau >= len(Xs[i]):
                        continue
                    H_tau.append(Hs[i][tau:, :])
                    X_tau.append(Xs[i][:len(Xs[i])-tau, :])
                H_tau = np.concatenate(H_tau, 0)
                X_tau = np.concatenate(X_tau, 0)
                # infer the operator via linear regression
                operators.append(np.linalg.solve(np.dot(H_tau.T, H_tau) + self.regul * np.eye(H_tau.shape[1]), np.dot(H_tau.T, X_tau)))

            self.assoc_U_, self.assoc_V_, self.assoc_b_ = learn_association_mappings_alignment_(H_assoc, M_assoc, Q_assoc, operators, operators, self.regul, max_negatives = self.max_negatives, verbose = verbose)

            if verbose:
                print('association mapping shapes: %s and %s; threshold: %g' % (self.assoc_U_.shape, self.assoc_V_.shape, self.assoc_b_))

            d = np.sum(np.square(np.dot(H_assoc, self.assoc_U_) - np.dot(M_assoc, self.assoc_V_)), 1) - self.assoc_b_

            if verbose:
                print('read head accuracy: positive cases: %g negative cases: %g' % (np.mean(d[Q_assoc > 0.] < 0.), np.mean(d[Q_assoc < 0.] > 0.)))

        # prepate the output
        Y = np.concatenate(Ys, axis=0)

        # now, train the output layer
        if not self.discrete_prediction:
            # if the prediction is discrete, perform linear regression,
            # as in standard echo state networks
            C    = np.dot(H.T, H)
            Cout = np.dot(H.T, Y)
            self.V_ = np.linalg.solve(C + self.regul * np.eye(H.shape[1]), Cout)
            if verbose:
                print('output RMSE: %g' % np.sqrt(np.mean(np.square(Y - np.dot(H, self.V_)))))
        else:
            # otherwise, train a linear SVM for each output channel
            self.V_ = []
            for l in range(Y.shape[1]):
                self.V_.append(LinearSVC(dual = False, C = self.C, max_iter = 10000))
                unique_outputs = np.unique(Y[:, l])
                if len(unique_outputs) == 1:
                    # if there is only one possible output, don't train the
                    # svm and instead override the predict method with a
                    # constant prediction
                    self.V_[l].predict = lambda X : np.full(X.shape[0], unique_outputs[0])
                else:
                    self.V_[l].fit(H, Y[:, l])

        return self

    def predict(self, X, return_states = False):
        """ Predicts an output time series for the given input time series.

        In more detail, this function first applies the reservoir with the
        following mechanism:

        1. Initialize state h_0 as the zero vector, t as 1, and M as an empty
           dictionary.
        2. if t > T, return the current state sequence.
        3. Compute the next memory state h_t using the standard ESN formula.
        4. Classify h_t using self.c_. If the output is +1 and the memory is
           not full yet, write h_t into memory and go to 6.
        5. Classify h_t * M[k, :] using self.c_assoc_ for every k. If the
           output is +1 for any of these, override h_t with the M[k, :]
           vector with maximum value. Otherwise, leave h_t as is.
        6. Increment t and go to 2.

        Then, the states are transformed into outputs using self.V_, resulting
        in the output time series Y.

        Parameters
        ----------
        X: class numpy.array
            A T x self.U.shape[1] input time series.
        return_states: bool (default = False)
            If set to true, the predicted state sequence will be returned as
            second argument.

        Returns
        -------
        Y: class numpy.array
            A T x len(self.V_) output time series.
        Q: class numpy.array (optional)
            A T-element state time series. This is only returned if
            return_states is set to True.

        """
        # check input dimensionality
        if X.shape[1] != self.U.shape[1]:
            raise ValueError('Expected %d input dimensions but got %d' % (self.U.shape[1], X.shape[1]))
        T = X.shape[0]

        # normalize input
        if self.input_normalization:
            X = (X - self.mu_) * self.beta_

        # initialize state matrix, both continuous and discrete
        m = self.U.shape[0]
        H = np.zeros((T, m))
        Q = np.zeros(T, dtype=int)
        # initialize memory matrix
        M = np.zeros((self.memory_size_, m))
        # and a copy of the memory matrix after transformation into latent space
        if hasattr(self, 'assoc_V_'):
            M_latent = np.zeros((self.memory_size_, self.assoc_V_.shape[1]))
        k = 0
        if self.q_0 > 0:
            k += 1
        # compute retain rate
        retain = 1. - self.leak
        if retain < 1E-3:
            retain = 0.

        for t in range(T):
            # compute current state
            if t == 0:
                H[t, :] = self.leak * self.nonlin(np.dot(self.U, X[t, :]) + self.W * self.init_state) + retain * self.init_state
            else:
                H[t, :] = self.leak * self.nonlin(np.dot(self.U, X[t, :]) + self.W * H[t-1, :]) + retain * H[t-1, :]

            # check if the state should be written into memory
            if k < self.memory_size_ and self.c_.predict(np.expand_dims(H[t, :], 0)) > 0:
                M[k, :] = H[t, :]
                if hasattr(self, 'assoc_V_'):
                    M_latent[k, :] = np.dot(M[k, :], self.assoc_V_)
                k += 1
                Q[t] = k
            elif k > 0 and hasattr(self, 'assoc_U_'):
                # otherwise, check if the state should be associated with a
                # state in memory

                # compute the distance between memory and state in the latent space
                ds = np.sum(np.square(np.dot(H[t, :], self.assoc_U_)[None, :] - M_latent[:k, :]), 1) - self.assoc_b_
                # if any distance is smaller zero, associate the respective memory
                # content
                k_assoc = np.argmin(ds)
                if ds[k_assoc] < 0.:
                    H[t, :] = M[k_assoc, :]
                    Q[t] = k_assoc + 1

        # transform to output
        if not self.discrete_prediction:
            # if the prediction is continuous, we simply multiply with the
            # output matrix
            Y = np.dot(H, self.V_)
        else:
            # otherwise, we predict each channel via SVMs
            Ys = []
            for l in range(len(self.V_)):
                Ys.append(self.V_[l].predict(H))
            Y = np.stack(Ys, axis = 1)

        if return_states:
            return Y, Q
        else:
            return Y


def learn_association_mappings_alignment_(X, Y, Z, Us, Vs, regul = 1E-3, max_negatives = 500, verbose = False):
    """ Selects two linear mappings U and V as well as a threshold b, such
    that the distance between X * U and Y * V is small if and only if Z is
    positive.

    In more detail, this method assumes that two sets of mappings U_1, ..., U_K
    and V_1, ..., V_L are given and then selects optimal semi-positive weights
    w_kl for each possible pairing (U_k, V_l) as well as a threshold b such
    that the following loss is minimized

    .. math:: \\sum_i \\big[(d_i^2 - b) \\cdot z_i + 1\\big]_+ + \\sum_{k=1}^K \\sum_{l=1}^L w_{k, l}

    where

    .. math:: d_i^2 = \\sum_{k=1}^K \\sum_{l=1}^L w_{k, l} \\lVert U_k \\cdot \\vec x_i - V_l \\cdot \\vec y_i \\rVert^2

    The overall mappings U and V are then constructed as

    .. math:: U = (\\sqrt{w_{1, 1}} \\cdot U_1 , \\ldots , \\sqrt{w_{K, L}} \\cdot U_K)
    .. math:: V = (\\sqrt{w_{1, 1}} \\cdot V_1 , \\ldots , \\sqrt{w_{K, L}} \\cdot V_L)

    where we exclude entries where w_kl is zero.

    This problem can be formalized as a sparse linear program with slack
    variables for each row in X.

    Parameters
    ----------
    X: class numpy.array
        A N x m matrix of left-hand-side input data.
    Y: class numpy.array
        A N x n matrix of right-hand-side input data.
    Z: class numpy.array
        A N-element array with entries +1 if X[i, :] and Y[i, :] should be
        associated and -1 otherwise.
    Us: list
        A K-element list of m x r matrices which map from X into a latent
        space.
    Vs: list
        A L-element list of n x r matrices which map from Y into a latent
        space.
    regul: float
        The L1 regularization strength.
    max_negatives: int (default = 500)
        The maximum number of non-associated examples we use during learning.
        Since this uses a linear program that scales linearly with the number
        of negative examples, large numbers here can slow learning down
        substantially.

    Returns
    -------
    U: class numpy.array
        A m x (R*r) linear mapping from X into a latent space, where R is at
        most K * L.
    V: class numpy.array
        A n x (R*r) linear mapping from Y into a latent space, where R is at
        most K * L.
    b: float
        A threshold such that some vector x should be associated with some
        vector y if and only if ||U*x - V*y|| - b is smaller than zero.

    """

    # subsample the data if it is large
    neg_samples = np.where(Z < 0.)[0]
    if len(neg_samples) > max_negatives:
        pos_samples = np.where(Z > 0.)[0]
        if verbose:
            print('subsampling the data to %d negative examples plus %d positive examples' % (max_negatives, len(pos_samples)))
        neg_sub = np.random.choice(len(neg_samples), size = max_negatives, replace = False)
        sub_samples = np.concatenate((pos_samples, neg_samples[neg_sub]), 0)
        X = X[sub_samples, :]
        Y = Y[sub_samples, :]
        Z = Z[sub_samples]


    N = len(X)
    K = len(Us)
    L = len(Vs)
    # set up the linear program. We have N + len(Us) * len(Vs) + 1 variables,
    # the first N are slack variables for the margin violations, the next
    # len(Us) * len(Vs) variables are the actual weights, and the last
    # variable is for the threshold.
    num_vars = N + K * L + 1

    # we start with the objective function. This is the sum of all slack
    # variables plus the sum of all weights
    c = np.ones(num_vars)
    c[N:-1] *= regul
    c[-1] = 0.

    # next, we get started on the side constraints. In particular, our
    # side constraints are of the form
    # sum_{k, l} w_{k, l} * ||U_k * X[i, :] - V_l * Y[i, :]||^2 * Z[i]
    # - b * Z[i] - slack[i] <= -1

    # pre-compute the distances for all data points
    XUs = []
    for k in range(K):
        XUs.append(np.dot(X, Us[k]))
    YVs = []
    for l in range(L):
        YVs.append(np.dot(Y, Vs[l]))
    D = np.zeros((K, L, N))
    for k in range(K):
        for l in range(L):
            D[k, l, :] = np.sum(np.square(XUs[k] - YVs[l]), 1)

    # iterate over all data points
    rows = []
    cols = []
    vals = []
    for i in range(N):
        # establish the side constraint for point i

        # coefficient for slack variable
        rows.append(i)
        cols.append(i)
        vals.append(-1.)
        # coefficients for weights
        for k in range(K):
            for l in range(L):
                rows.append(i)
                cols.append(N + k*L+l)
                vals.append(D[k, l, i] * Z[i])
        # coefficient for threshold
        rows.append(i)
        cols.append(num_vars-1)
        vals.append(-Z[i])
    # set up side constraints
    A_ub = csr_matrix((vals, (rows, cols)), shape=(N, num_vars))
    b_ub = -np.ones(N)

    # set up bounds
    bounds = (0., None)

    res = linprog(c, A_ub, b_ub, bounds = bounds, method = 'interior-point', options = {'sparse' : True})

    # extract variables
    W = np.reshape(res.x[N:-1], (K, L))
    b = res.x[-1]

    # report associations if so desired
    if verbose:
        for k in range(K):
            for l in range(L):
                if W[k, l] > 1E-3 * np.max(W):
                    print('associated (%d, %d) with weight %g' % (k, l, W[k, l]))

    # construct output mappings
    U = []
    V = []
    for k in range(K):
        for l in range(L):
            if W[k, l] > 1E-3 * np.max(W):
                U.append(Us[k] * np.sqrt(W[k, l]))
                V.append(Vs[l] * np.sqrt(W[k, l]))
    U = np.concatenate(U, 1)
    V = np.concatenate(V, 1)
    return U, V, b


def learn_association_mappings_lmnn_(X, Y, Z, k = None, regul = 1E-5):
    """ Learns mappings U and V as well as a threshold gamma, such that
    the distance between X * U and Y * V is small if and only if
    Z is positive.

    In more detail, this method tries to enforce that for as many i as
    possible we obtain

    .. math:: (\\lVert U \cdot \\vec x_i - V \\cdot \\vec y_i \\rVert^2 - 1) \\cdot z_i < 0

    This is done by initializing U and V via PCA on X and Y and then
    opimizing the loss

    \\frac{1}{4} \\sum_i \\big[(\\lVert U \cdot \\vec x_i - V \cdot \\vec y_i \\rVert^2 - 1) \\cdot z_i + \\frac{1}{2} \\big]_+ + \\frac{C}{2} \\cdot (\\lVert U \\rVert_F^2 + \\lVert V \\rVert_F^2)

    where C is the parameter regul and where [x]_+ refers to the rectified
    linear unit max(0, x).

    Parameters
    ----------
    X: class numpy.array
        A N x m matrix of left-hand-side input data.
    Y: class numpy.array
        A N x n matrix of right-hand-side input data.
    Z: class numpy.array
        A N-element array with entries +1 if X[i, :] and Y[i, :] should be
        associated and -1 otherwise.
    k: int (default = min(m, n))
        The latent space dimension.
    regul: float (default = 1E-5)
        The Frobenius norm regularization.

    Returns
    -------
    U: class numpy.array
        A k x m linear mapping from X into the latent space.
    V: class numpy.array
        A k x n linear mapping from Y into the latent space.

    """
    # retrieve dimensionalities
    m = X.shape[1]
    n = Y.shape[1]
    if k is None:
        k = min(m, n)
    # compute class weights to balance
    weight = len(Z) / np.sum(Z > 0.)
    # set up initial parameters
    pca = PCA(n_components = k)
    pca.fit(X)
    Uinit = pca.components_
    pca.fit(Y)
    Vinit = pca.components_
    paramsinit = np.concatenate((Uinit.ravel(), Vinit.ravel()), 0)
    # set up objective function
    def objfun(params):
        # extract U, V, and gamma from a flattened vector
        U = np.reshape(params[:m*k], (k, m))
        V = np.reshape(params[m*k:], (k, n))
        # compute the epsilon values, i.e. the error for each point
        UX = np.dot(X, U.T)
        VY = np.dot(Y, V.T)
        dists = np.sum(np.square(UX - VY), 1)
        epsilons = .5 + (dists - 1.) * Z
        epsilons[epsilons < 0.] = 0.
        # emphasize errors via class weights
        epsilons[Z > 0.] *= weight
        # compute loss
        loss = np.sum(np.square(epsilons)) / 4 + np.sum(np.square(U)) / 2 + np.sum(np.square(V)) / 2
        # multiply epsilons with labels
        epsilons *= Z
        # compute gradients
        Xeps = X * epsilons[:, None]
        Yeps = Y * epsilons[:, None]
        CXY  = np.dot(Xeps.T, Y)
        CXX  = np.dot(Xeps.T, X)
        CYY  = np.dot(Yeps.T, Y)
        gradU = np.dot(U, CXX + regul * np.eye(m)) - np.dot(V, CXY.T)
        gradV = np.dot(V, CYY + regul * np.eye(n)) - np.dot(U, CXY)
        # flatten gradients and concatenate
        grad = np.concatenate((gradU.ravel(), gradV.ravel()), 0)
        return loss, grad

    print('loss before optimization: %g' % objfun(paramsinit)[0])

    # optimize via lbfgs
    res = minimize(objfun, paramsinit, jac = True, method = 'L-BFGS-B')

    print('loss after optimization: %g' % res.fun)

    # extract result
    U = np.reshape(res.x[:m*k], (k, m))
    V = np.reshape(res.x[m*k:], (k, n))
    # return
    return U.T, V.T, -1.

def learn_association_mappings_(X, Y, Z, k = None, regul = 1E-5):
    """ Learns a linear regression mapping from Y to X as well as a bias
    to distinguish associated and nonassociated cases.

    In particular, the bias is optimized for the F1 score, i.e. the harmonic
    mean between association recall and association precision.

    Parameters
    ----------
    X: class numpy.array
        A N x m matrix of left-hand-side input data.
    Y: class numpy.array
        A N x n matrix of right-hand-side input data.
    Z: class numpy.array
        A N-element array with entries +1 if X[i, :] and Y[i, :] should be
        associated and -1 otherwise.
    k: int (default = min(m, n))
        The latent space dimension.
    regul: float (default = 1E-5)
        The Frobenius norm regularization.

    Returns
    -------
    U: class numpy.array
        A k x m linear mapping from X into the latent space.
    V: class numpy.array
        A k x n linear mapping from Y into the latent space.
    b: float
        A bias value such that ||X[i, :]*U - Y[i, :]*V||^2 - b is smaller zero
        hopefully if and only if X[i, :] should be associated with Y[i, :].

    """
    print(X.shape)
    print(Y.shape)
    print(Z.shape)
    m = X.shape[1]
    n = Y.shape[1]
    if k is None:
        k = min(m, n)
    # set up auxiliary variables that we re-use multiple times during
    # optimization
    num_positives = np.sum(Z > 0.)
    num_negatives = np.sum(Z < 0.)
    Z_tilde = Z.copy()
    correction_factor = 0. * num_positives / num_negatives
    Z_tilde[Z < 0.] *= correction_factor

    XZ = X * np.expand_dims(Z_tilde, 1)
    YZ = Y * np.expand_dims(Z_tilde, 1)
    CXX = np.dot(XZ.T, X) + regul * np.eye(m)
    CYY = np.dot(YZ.T, Y) + regul * np.eye(n)
    CXY = np.dot(X.T, YZ)

    # set up initial V via PCA
    pca = PCA(n_components = k)
    pca.fit(Y)
    V = pca.components_.T
    # set up the initial locations of Y in the latent space
    YV = np.dot(Y, V)

    # start iterative optimization
    last_loss = np.inf
    it = 0
    while True:
        # update U via linear regression
        U = np.linalg.solve(CXX, np.dot(CXY, V))
        # update X position in the latent space
        XU = np.dot(X, U)
        # re-compute loss
        dists = np.sum(np.square(XU - YV), 1)
        loss  = np.sum(dists * Z_tilde) + regul * (np.sum(np.square(U)) + np.sum(np.square(V)))
        print('loss in iteration %d after updating U: %g' % (it + 1, loss))
        # stop optimization if loss does not decrease anymore
        if loss > last_loss - 1E-3:
            break
        last_loss = loss

        # update V via linear regression
        V = np.linalg.solve(CYY, np.dot(CXY.T, U))
        # update Y location in the latent space
        YV = np.dot(Y, V)
        # re-compute loss
        dists = np.sum(np.square(XU - YV), 1)
        loss  = np.sum(dists * Z_tilde) + regul * (np.sum(np.square(U)) + np.sum(np.square(V)))
        # stop optimization if loss does not decrease anymore
        print('loss in iteration %d after updating V: %g' % (it + 1, loss))
        if loss > last_loss - 1E-3:
            break
        last_loss = loss
        it += 1
        if it == 100:
            break

    # compute the distances
    # sort the distances
    idxs = np.argsort(dists)
    # compute recalls, precisions, and f1 scores
    offset = 0
    while Z[idxs[offset]] < 0.:
        offset += 1
    true_positives = np.cumsum(Z[idxs[offset:]] > 0.)
    recalls        = true_positives / num_positives
    precisions     = true_positives / np.arange(1, len(Z)+1-offset)
    f1s            = 2. * (recalls * precisions) / (recalls + precisions)
    import matplotlib.pyplot as plt
    plt.plot(XU[Z > 0., 0], XU[Z > 0., 1], 'rx')
    plt.plot(YV[Z > 0., 0], YV[Z > 0., 1], 'bx')
    plt.show()
    plt.plot(dists[idxs])
    plt.plot(recalls)
    plt.plot(precisions)
    plt.plot(f1s)
    plt.legend(['distances', 'recalls', 'precisions', 'F1'])
    plt.show()
    i = offset + np.argmax(f1s)
    # the actual threshold is then the mean between the distance value for i and
    # i+1
    b = - 0.5 * (dists[idxs[i]] + dists[idxs[i+1]])
    # return result
    return U, V, b


def learn_association_mapping_(X, Y, Z, regul = 1E-5):
    """ Learns linear regression mappings from X and Y to a latent space
    as well as a bias to distinguish associated and nonassociated cases.

    In particular, the bias is optimized for the F1 score, i.e. the harmonic
    mean between association recall and association precision.

    Parameters
    ----------
    X: class numpy.array
        A N x m matrix of left-hand-side input data.
    Y: class numpy.array
        A N x n matrix of right-hand-side input data.
    Z: class numpy.array
        A N-element array with entries +1 if X[i, :] and Y[i, :] should be
        associated and -1 otherwise.
    regul: float (default = 1E-5)
        The Frobenius norm regularization.

    Returns
    -------
    U: class numpy.array
        A n x m linear mapping from Y to X.
    b: float
        A bias value such that ||X[i, :] - Y[i, :] * W||^2 - b is smaller zero
        hopefully if and only if X[i, :] should be associated with Y[i, :].

    """
    # set up the linear regression
    num_positives = np.sum(Z > 0.)
    num_negatives = np.sum(Z < 0.)
    Z_tilde = Z
    Z_tilde[Z < 0.] *= regul * num_positives / num_negatives
    YZ = Y * Z_tilde[:, None]
    W = np.linalg.solve(np.dot(YZ.T, Y) + regul * np.eye(Y.shape[1]), np.dot(YZ.T, X))
    # compute the distances
    ds = np.sum(np.square(X - np.dot(Y, W)), 1)
    # sort the distances
    idxs = np.argsort(ds)
    # compute recalls, precisions, and f1 scores
    offset = 0
    while Z[idxs[offset]] < 0.:
        offset += 1
    true_positives = np.cumsum(Z[idxs[offset:]] > 0.)
    recalls        = true_positives / num_positives
    precisions     = true_positives / np.arange(1, len(Z)+1-offset)
    f1s            = 2. * (recalls * precisions) / (recalls + precisions)
    i = offset + np.argmax(f1s)
    # the actual threshold is then the mean between the distance value for i and
    # i+1
    b = - 0.5 * (ds[idxs[i]] + ds[idxs[i+1]])
    # return result
    return W, b

def learn_kernel_association_mapping_(X, Y, Z, regul = 1E-2):
    """ Learns a mapping from Y to X as well as a bias to distinguish
    associated and nonassociated cases via a dual SVM.

    In particular, the dual SVM tries to achieve the condition

    (X[i, :] * W * Y[i, :] + b) * Z[i, :] >= 1.

    for all i. This turns out to be equivalent with the dual SVM which
    receives the kernel (X * X.T) * (Y * Y.T) as input, where the middle
    multiplication is element-wise.

    Parameters
    ----------
    X: class numpy.array
        A N x m matrix of left-hand-side input data.
    Y: class numpy.array
        A N x n matrix of right-hand-side input data.
    Z: class numpy.array
        A N-element array with entries +1 if X[i, :] and Y[i, :] should be
        associated and -1 otherwise.
    regul: float (default = 1E-5)
        The Frobenius norm regularization.

    Returns
    -------
    W: class numpy.array
        A n x m linear mapping from Y to X.
    b: float
        A bias value such that (X[i, :] * W * Y[i, :] + b) * Z[i, :] >= 1
        whenever possible.

    """
    # set up the kernel matrix
    K = np.dot(X, X.T) * np.dot(Y, Y.T)
    # set up the dual SVM
    model = SVC(C = 1. / regul, kernel = 'precomputed', max_iter = -1, class_weight = 'balanced')
    # fit it
    model.fit(K, Z)
    # extract the resulting mapping
    W = np.dot(X[model.support_, :].T, model.dual_coef_.T * Y[model.support_, :])
    # extract the bias value
    b = model.intercept_

    print('kernel accuracy: %g' % model.score(K, Z))
    # return
    return W, b
