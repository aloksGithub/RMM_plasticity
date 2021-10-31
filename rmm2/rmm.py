"""
Implements reservoir memory machines.

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
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from rmm2.pseudo_svm import PseudoSVM
import rmm2.fsm as fsm
import rmm2.esn as esn
import math

__author__ = 'Benjamin Paaßen'
__copyright__ = 'Copyright 2020, Benjamin Paaßen'
__license__ = 'GPLv3'
__Version__ = '0.0.1'
__maintainer__ = 'Benjamin Paaßen'
__email__  = 'benjamin.paassen@sydney.edu.au'

def sigmoid(x, a, b):
    return 1/(1+math.exp(-(a*x+b)))

def get_delta_b(y, u, lr):
    delta_b = lr*(1-(2+1/u)*y+(y**2/u))
    return delta_b

def update_b(delta_b, b):
    return b + delta_b

def update_a(x, delta_b, a, lr):
    return a + lr/a + x*delta_b

def computeAndUpdate(x, a, b, u, lr):
    y = list(map(sigmoid, x, a, b))
    delta_b = list(map(get_delta_b, y, [u]*len(a), [lr]*len(a)))
    b = list(map(update_b, delta_b, b))
    a = list(map(update_a, x, delta_b, a, [lr]*len(a)))
    return y, a, b

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
        esn.initialize_reservoir(U, W, 0.9, 0.1)[0].
    W: class scipy.sparse.csr_matrix OR int
        A m x m sparse neuron-to-neuron/recurrent connection matrix OR the
        number of input channels. In the latter case, this becomes
        esn.initialize_reservoir(U, W, 0.9, 0.1)[1].
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
    discrete_prediction: bool (default = False)
        A flag that marks wether the output should be a continuous prediction
        via linear regression or a discrete prediction via linear SVM.
    C: positive float (default = 100.)
        The regularization parameter for the state classifier.
    q_0: int (default = 0)
        The initial state, i.e. whether self.init_state should be assigned
        to some special state.
    svm_kernel: string (default = 'linear')
        The kernel that the SVM should use, either 'linear', 'rbf', or 'pseudo'.
        For a linear kernel, we use the sklearn LAPACK implementation which
        should guarantee linear scaling of training times with the data set
        size, whereas the rbf kernel may induce quadratic or even cubic
        runtimes. If 'pseudo' is selected, we use a custom SVM implementation
        based on linear regression which is guaranteed linear (and fast) in the
        number of data points but uses primitive linear regression as a
        surrogate of the decision function.
    init_state: class numpy.array (default = numpy.zeros(U.shape[0]))
        The initial continuous state.

    Attributes
    ----------
    V_: class numpy.array OR list
        The reservoir-to-output matrix OR a list of linear SVMs, one per
        output channel. Is set during training.
    c_: class sklearn.svm.LinearSVC
        The state classifier. Is set during training.
    mu_: class numpy.array
        The mean of the training data for each feature. This is used for
        input_normalization.
    beta_: class numpy.array
        The precision of the training data for each feature.
        This is used for input_normalization.

    """
    def __init__(self, U, W, lr, u, leak = 1., regul = 1E-5, input_normalization = True,
        washout = 0, nonlin = np.tanh, discrete_prediction = False, C = 100., q_0 = 0,
        svm_kernel = 'linear', init_state = None):

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
        self.washout = washout
        self.nonlin = nonlin
        self.discrete_prediction = discrete_prediction
        self.C = C
        self.q_0 = q_0
        self.svm_kernel = svm_kernel
        if init_state is None:
            self.init_state = np.zeros(self.U.shape[0])
        else:
            self.init_state = init_state
        
        # intrinsic plasticity variables
        self.lr = lr
        self.u = u
        self.a = np.ones(len(W.todense()))
        # self.a = np.random.rand(len(W.todense()))
        self.b = np.zeros(len(W.todense()))
        # self.b = np.random.rand(len(W.todense()))

    def fit(self, X, Q, Y):
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

        # process all input matrices via teacher forcing and concatenate them
        # to one big state time series; in particular one for the classifier
        # containing raw states and one for the output containing recovered
        # states
        Hs_raw = []
        Hs = []
        Qs_cut = []
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
            # initialize memory
            M = {}
            if self.q_0 > 0:
                M[self.q_0] = self.init_state
            # compute retain rate
            retain = 1. - self.leak
            if retain < 1E-3:
                retain = 0.
            # compute time steps
            for t in range(T):
                # compute the current raw state
                if t == 0:
                    x = np.dot(self.U, X[t, :]) + self.W * self.init_state
                    sigmoidOutput, self.a, self.b = computeAndUpdate(x, self.a, self.b, self.u, self.lr)
                    H_raw[t, :] = self.leak * np.array(sigmoidOutput) + retain * self.init_state
                    # H_raw[t, :] = self.leak * self.nonlin(np.dot(self.U, X[t, :]) + self.W * self.init_state) + retain * self.init_state
                else:
                    x = np.dot(self.U, X[t, :]) + self.W * H[t-1, :]
                    sigmoidOutput, self.a, self.b = computeAndUpdate(x, self.a, self.b, self.u, self.lr)
                    H_raw[t, :] = self.leak * np.array(sigmoidOutput) + retain * H[t-1, :]
                    # H_raw[t, :] = self.leak * self.nonlin(np.dot(self.U, X[t, :]) + self.W * H[t-1, :]) + retain * H[t-1, :]
                # check if it needs to be overridden by memory
                if Q[t] > 0:
                    if Q[t] in M:
                        # if we have already a state associated with q_t, recover
                        # that state
                        H[t, :] = M[Q[t]]
                    else:
                        # otherwise, memorize the current state
                        H[t, :] = H_raw[t, :]
                        M[Q[t]] = H[t, :]
                else:
                    H[t, :] = H_raw[t, :]
            # remove washout steps before recording the data
            Hs_raw.append(H_raw[self.washout:, :])
            Hs.append(H[self.washout:, :])
            Qs_cut.append(Q[self.washout:])

        # concatenate all time series
        H_raw = np.concatenate(Hs_raw, axis=0)
        H     = np.concatenate(Hs, axis=0)
        Q     = np.concatenate(Qs_cut, axis=0)

        unique_states = np.unique(Q)
        if self.svm_kernel == 'linear':
            self.c_ = LinearSVC(dual = False, C = self.C, max_iter = 10000)
        elif self.svm_kernel == 'rbf':
            self.c_ = SVC(C = self.C, max_iter = 10000)
        elif self.svm_kernel == 'pseudo':
            self.c_ = PseudoSVM(regul = self.regul)
        else:
            raise ValueError('Unsupported SVM kernel: %s' % self.svm_kernel)
        if len(unique_states) == 1:
            # if there is only one state, don't train the svm
            # and override the predict method with a constant prediction
            self.c_.predict = lambda X : np.full(X.shape[0], unique_states[0])
        else:
            # train the state classifier based on H_raw
            self.c_.fit(H_raw, Q)

            Qpred = self.c_.predict(H_raw)
            recall    = np.mean(Qpred[Q > 0.5] == Q[Q > 0.5])
            precision = np.mean(Qpred[Qpred > 0.5] == Q[Qpred > 0.5])

            print('State prediction recall: %g; precision: %g' % (recall, precision))

        # prepate the output
        Ys_cut = []
        for Y in Ys:
            # remove the washout steps from each output matrix
            Ys_cut.append(Y[self.washout:, :])
        Y = np.concatenate(Ys_cut, axis=0)

        # now, train the output layer
        if not self.discrete_prediction:
            # if the prediction is discrete, perform linear regression,
            # as in standard echo state networks
            C    = np.dot(H.T, H)
            Cinv = np.linalg.inv(C + self.regul * np.eye(H.shape[1]))
            Cout = np.dot(Y.T, H)
            self.V_ = np.dot(Cout, Cinv)
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

        1. Initialize state h_0 as self.init_state, t as 1, and M as an empty
           dictionary.
        2. if t > T, return the current state sequence.
        3. Compute the next memory state h_t using the standard ESN formula.
        4. Classify h_t using self.c_ into a discrete state q_t.
        5. If q_t is 0 (the special default state), increment t and go to 2.
        6. If q_t is contained in M, set h_t to M[q_t]. Otherwise set M[q_t]
           to h_t.
        7. Increment t and go to 2.

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
        a = list(self.a)
        b = list(self.b)
        if X.shape[1] != self.U.shape[1]:
            raise ValueError('Expected %d input dimensions but got %d' % (self.U.shape[1], X.shape[1]))
        T = X.shape[0]

        # normalize input
        if self.input_normalization:
            X = (X - self.mu_) * self.beta_

        # initialize state matrix, both continuous and discrete
        H = np.zeros((T, self.U.shape[0]))
        Q = np.zeros(T)
        # initialize state map
        M = {}
        if self.q_0 > 0:
            M[self.q_0] = self.init_state
        # compute retain rate
        retain = 1. - self.leak
        if retain < 1E-3:
            retain = 0.
        # compute time steps
        for t in range(T):
            # compute the current raw state
            if t == 0:
                x = np.dot(self.U, X[t, :]) + self.W * self.init_state
                sigmoidOutput, a, b = computeAndUpdate(x, a, b, self.u, self.lr)
                H[t, :] = self.leak * np.array(sigmoidOutput) + retain * self.init_state
                # H[t, :] = self.leak * self.nonlin(np.dot(self.U, X[t, :]) + self.W * self.init_state) + retain * self.init_state
            else:
                x = np.dot(self.U, X[t, :]) + self.W * H[t-1, :]
                sigmoidOutput, a, b = computeAndUpdate(x, a, b, self.u, self.lr)
                H[t, :] = self.leak * np.array(sigmoidOutput) + retain * self.init_state
                # H[t, :] = self.leak * self.nonlin(np.dot(self.U, X[t, :]) + self.W * H[t-1, :]) + retain * H[t-1, :]
            # classify it
            Q[t] = self.c_.predict(np.expand_dims(H[t, :], 0))
            if Q[t] > 0:
                if Q[t] in M:
                    # if we have already a state associated with q_t, recover
                    # that state
                    H[t, :] = M[Q[t]]
                else:
                    # otherwise, memorize the current state
                    M[Q[t]] = H[t, :]

        # transform to output
        if not self.discrete_prediction:
            # if the prediction is continuous, we simply multiply with the
            # output matrix
            Y = np.dot(H, self.V_.T)
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


    def predict_debug(self, X, return_states = True):
        """ Predicts an output time series for the given input time series.

        In more detail, this function first applies the reservoir with the
        following mechanism:

        1. Initialize state h_0 as self.init_state, t as 1, and M as an empty
           dictionary.
        2. if t > T, return the current state sequence.
        3. Compute the next memory state h_t using the standard ESN formula. # - repeat this in debug mode - #
        4. Classify h_t using self.c_ into a discrete state q_t.
        5. If q_t is 0 (the special default state), increment t and go to 2.
        6. If q_t is contained in M, set h_t to M[q_t]. Otherwise set M[q_t]
           to h_t.
        7. Increment t and go to 2.

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
        H = np.zeros((T, self.U.shape[0]))
        Q = np.zeros(T)
        # initialize state map
        M = {}
        if self.q_0 > 0:
            M[self.q_0] = np.zeros(self.U.shape[0])
        # compute retain rate
        retain = 1. - self.leak
        if retain < 1E-3:
            retain = 0.
        # compute first state
        H[0, :] = self.nonlin(np.dot(self.U, X[0, :]))
        # check if this needs to go into memory
        Q[0] = self.c_.predict(np.expand_dims(H[0, :], 0))
        if (Q[0] > 0) & 0: ### --- dont use this in debug mode --- ###
            if Q[0] == self.q_0:
                # if self.q_0 is set, recall the zero vector
                H[0, :] = 0.
            else:
                # otherwise, write into memory
                M[Q[0]] = H[0, :]
        # compute remaining time steps
        for t in range(1, T):
            # compute next continuous memory state
            H[t, :] = self.leak * self.nonlin(np.dot(self.U, X[t, :]) + self.W * H[t-1, :]) + retain * H[t-1, :]
            # classify it
            Q[t] = self.c_.predict(np.expand_dims(H[t, :], 0))
            if (Q[t] > 0) & 0: ### --- dont use this in debug mode --- ###
                if Q[t] in M:
                    # if we have already a state associated with q_t, recover
                    # that state
                    H[t, :] = M[Q[t]]
                else:
                    # otherwise, memorize the current state
                    M[Q[t]] = H[t, :]

        # transform to output
        if not self.discrete_prediction:
            # if the prediction is continuous, we simply multiply with the
            # output matrix
            Y = np.dot(H, self.V_.T)
        else:
            # otherwise, we predict each channel via SVMs
            Ys = []
            for l in range(len(self.V_)):
                Ys.append(self.V_[l].predict(H))
            Y = np.stack(Ys, axis = 1)

        if return_states:
            return Y, Q, H
        else:
            return Y


def fsm_to_rmm(delta, rho, q_0 = 0, rmm = None):
    """ Transforms a finite state machine into a reservoir memory machine.

    Parameters
    ----------
    delta: class numpy.array
        A len(states) x len(in_alphabet) table where delta[q, x] = q2 means
        that state q transfers to state q2 on input x.
    rho: class numpy.array
        A len(states) array where rho[q] = y means that state q outputs
        symbol y.
    q_0: int (default = 0)
        the starting state.
    rmm: class RMM (default = RMM(input_normalization = False, discrete_prediction = True, q_0 = q_0 + 1))
        an initial reservoir memory machine that is not yet trained

    Returns
    -------
    rmm: class RMM
        The corresponding reservoir memory machine to the input finite state
        machine, where each input symbol is represented via one-hot coding
        and each output symbol via its index.

    """
    # ensure that delta and rho are consistent
    if delta.shape[0] != len(rho):
        raise ValueError('The delta table has %d rows but the rho table had %d elements' % (delta.shape[0], len(rho)))
    # Identify the paths with exactly one cycle
    paths = fsm.one_cyclic_paths(delta, q_0)
    # convert them to training data for a RMM
    Xs = []
    Qs = []
    Ys = []
    for path in paths:
        # initialize input matrix, state array, and output array
        X = np.zeros((len(path), delta.shape[1]))
        Q = np.zeros(len(path))
        Y = np.zeros((len(path), 1))
        # convert the path
        for t in range(len(path)):
            q, x = path[t]
            X[t, x] = 1.
            Q[t] = q + 1
            Y[t] = rho[q]
        # append to training data
        Xs.append(X)
        Qs.append(Q)
        Ys.append(Y)

#    print('--- training data ---')
#    for j in range(len(Xs)):
#        print('-- new time series --')
#        for t in range(len(Xs[j])):
#            print('t = %d, x = %s, q = %g, y = %g' % (t, str(Xs[j][t, :]), Qs[j][t], Ys[j][t]))

    # set up the reservoir memory machine
    if rmm is None:
        rmm = RMM(128, delta.shape[1], input_normalization = False,
            discrete_prediction = True, q_0 = q_0 + 1)
    # train the RMM
    rmm.fit(Xs, Qs, Ys)

    # return the rmm
    return rmm
