"""
Implements variants of the reservoir memory machine as learned systems,
either entirely learned or with a fixed reservoir but learned memory
access.

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

import torch
import numpy as np

__author__ = 'Benjamin Paaßen'
__copyright__ = 'Copyright 2020, Benjamin Paaßen'
__license__ = 'GPLv3'
__Version__ = '0.0.1'
__maintainer__ = 'Benjamin Paaßen'
__email__  = 'benjamin.paassen@sydney.edu.au'

class DeepMemoryMachine(torch.nn.Module):
    """ A deep variant of the reservoir memory machine which can losslessly
    recover past states.

    Parameters
    ----------
    U: class numpy.array OR int
        A fixed m x n input-to-neurons connection matrix OR the number of
        neurons. In the latter case, this becomes a GRU.
    W: class scipy.sparse.csr_matrix OR int
        A m x m sparse neuron-to-neuron/recurrent connection matrix OR the
        number of input channels.
    K: int 
        The number of rows in memory.
    L: int
        The number of output dimensions.
    nonlin: function (default = numpy.tanh)
        An elementwise nonlinearity. If the reservoir is a GRU, this has no
        effect.
    q_0: int (default = 0)
        The initial state, i.e. whether self.init_state should be assigned
        to some special state.
    init_state: class numpy.array (default = numpy.zeros(U.shape[0]))
        The initial continuous state.

    Attributes
    ----------
    V_: class torch.nn.Linear
       The reservoir-to-output matrix.
    C_: class torch.nn.Linear
        The reservoir-to-memory address matrix.

    """
    def __init__(self, U, W, K, L, nonlin = np.tanh, q_0 = 0, init_state = None):
        super(DeepMemoryMachine, self).__init__()
        # initialize recurrent part of the model
        if isinstance(U, int):
            if not isinstance(W, int):
                raise ValueError('If U is a single integer, W must be as well.')
            self.m = U
            self.n = W
            self.gru_ = torch.nn.GRU(self.n, self.m)
            self.numpy_reservoir = False
        else:
            self.U = U
            self.W = W
            self.n = U.shape[1]
            self.m = U.shape[0]
            self.nonlin = nonlin
            self.numpy_reservoir = True
        if init_state is None:
            self.init_state = torch.zeros(self.m)
        else:
            self.init_state = torch.tensor(init_state)
        # initialize memory mechanism
        self.K = K
        self.q_0 = q_0
        self.C_ = torch.nn.Linear(self.m, self.K+1)
        self.softmax = torch.nn.Softmax(dim=0)
        # initialize output mechanism
        self.L = L
        self.V_ = torch.nn.Linear(self.m, self.L)

    def _next_step(self, x, h):
        """ Computes the next recurrent state, either with a reservoir or
        with a pre-defined GRU.

        Parameters
        ----------
        h: class torch.Tensor
            The self.m dimensional current system state.
        x: class numpy.array
            The self.n dimensional current input vector.

        Returns
        -------
        h: class torch.Tensor
            The self.m dimensional next system state.

        """
        if self.numpy_reservoir:
            return torch.tensor(self.nonlin(np.dot(self.U, x) + self.W * h.detach().numpy()), dtype=torch.float)
        else:
            # expand h and x with extra dimensions to be compatible
            # with the torch GRU implementation
            h = h.unsqueeze(0).unsqueeze(1)
            x = torch.tensor(x, dtype=torch.float).unsqueeze(0).unsqueeze(1)
            # process with GRU
            _, h = self.gru_(x, h)
            # return the state without empty dimensions
            return h.squeeze(1).squeeze(0)

    def forward(self, X, return_states = False):
        """ Computes the output for the given input time series.

        In more detail, this function first applies the reservoir with the
        following mechanism:

        1. Initialize state h_0 as self.init_state, t as 1, M as a self.K x
           self.m zero matrix (except if self.q_0 > 0, in which case the
           self.q_0th row of M is filled with self.init_state), and beta as a
           self.K dimensional zero vector.
        2. if t > T, return the current state sequence.
        3. Compute the next memory state h_t using the recurrent network.
        4. Compute alpha_t as softmax(self.C_ * h_t)
        5. Generate a memory read vector as (alpha_t * beta).dot(M).
        6. Add alpha_t[l] * (1 - beta[l]) * h_t to M[l, :]. For all l.
        7. Add alpha_t[l] * (1 - beta[l]) to beta[l].
        8. Increment t and go to 2.

        Then, the states are transformed into outputs using self.V_, resulting
        in the output time series Y.

        Parameters
        ----------
        X: class numpy.array
            A T x self.n input time series.
        return_states: bool (default = False)
            If set to true, the T x self.K time series of alpha_t values will
            be returned as well.

        Returns
        -------
        Y: class torch.Tensor
            A T x self.L output time series.
        Q: class torch.Tensor (optional)
            A T x self.K time series of state access coefficients.

        """
        # check input dimensionality
        if X.shape[1] != self.n:
            raise ValueError('Expected %d input dimensions but got %d' % (self.n, X.shape[1]))
        T = X.shape[0]

        # initialize state list and predicted memory addresses
        H = []
        h = self.init_state
        Q = np.zeros(T)
        # initialize memory matrix
        M = {}
        if self.q_0 > 0:
            M[self.q_0] = self.init_state
        # iterate over all time steps
        for t in range(T):
            # compute the next recurrent state
            h = self._next_step(X[t, :], h)
            # compute the predicted memory address
            Q[t] = np.argmax(self.C_(h).detach().numpy())
            if Q[t] > 0:
                if Q[t] in M:
                    # if we have already a state associated with q_t, recover
                    # that state
                    h = M[Q[t]]
                else:
                    # otherwise, memorize the current state
                    M[Q[t]] = h
            # append state to the state sequence
            H.append(h)
        # concate H content
        H = torch.stack(H, 0)
        # apply output map
        Y = self.V_(H)
        # return
        if return_states:
            return Y, torch.stack(Q, 0)
        else:
            return Y

    def compute_teacher_forcing_loss(self, X, Q, Y, weight = 1.):
        """ Computes the crossentropy loss between predicted and desired
        memory addresses plus the mean square error between predicted and
        desired outputs under the assumption that the ground-truth memory
        addresses are used.

        Parameters
        ----------
        X: class numpy.array
            A T x self.n input time series.
        Q: class numpy.array
            A T element time series of ground-truth memory addresses.
        Y: class numpy.array
            A T x self.L time series of desired outputs.
        weight: float (default = 1.)
            A weight for the crossentropy loss. A higher weight means that
            state learning is more emphasized.

        Returns
        -------
        loss: class torch.Tensor
            The crossentropy loss between the predicted memory addresses
            and Q plus the mean square error between the predicted outputs
            and Y.

        """
        # check input dimensionality
        if X.shape[1] != self.n:
            raise ValueError('Expected %d input dimensions but got %d' % (self.n, X.shape[1]))
        T = X.shape[0]
        if np.any(Q > self.K):
            raise ValueError('Expected memory of size %d but got a memory address above it.' % self.K)
        if np.any(Q < 0):
            raise ValueError('Got a memory address below zero.')
        if len(Q) != T:
            raise ValueError('Got %d time steps in input but %d in address sequence' % (T, len(Q)))
        if Y.shape[1] != self.L:
            raise ValueError('Expected %d output dimensions but got %d' % (self.L, Y.shape[1]))
        if len(Y) != T:
            raise ValueError('Got %d time steps in input but %d in output' % (T, len(Y)))

        # initialize state list and predicted memory addresses
        H = []
        h = self.init_state
        Qpred = []
        # initialize memory matrix
        M = {}
        if self.q_0 > 0:
            M[self.q_0] = self.init_state
        # iterate over all time steps
        for t in range(T):
            # compute the next recurrent state
            h = self._next_step(X[t, :], h)
            # compute the memory address logits
            Qpred.append(self.C_(h))
            # update state based on memory
            if Q[t] > 0:
                if Q[t] in M:
                    # if we have already a state associated with q_t, recover
                    # that state
                    h = M[Q[t]]
                else:
                    # otherwise, memorize the current state
                    M[Q[t]] = h
            # append state to the state sequence
            H.append(h)
        # concate H and Qpred content
        H = torch.stack(H, 0)
        Qpred = torch.stack(Qpred, 0)
        # apply output map
        Ypred = self.V_(H)
        # compute loss and return
        xe  = torch.nn.functional.cross_entropy(Qpred, torch.tensor(Q, dtype=torch.long))
        mse = torch.nn.functional.mse_loss(Ypred, torch.tensor(Y, dtype=torch.float))
        #print('xe loss: %g, mse: %g' % (xe.item(), mse.item()))
        loss = weight * xe + mse
        return loss

class GRUInterface(torch.nn.Module):
    """ This is just a simpler-to-use interface to the torch GRU class to make
    it comparable with the interface for the DeepMemoryMachine above.

    Paramaters
    ----------
    m: int
        The number of neurons.
    n: int
        The number of input dimensions.
    L: int
        The number of output dimensions.

    Attributes
    ----------
    gru_: class torch.nn.GRU
        The actual GRU.
    V_: class torch.nn.Linear
        The L x m linear output layer.

    """
    def __init__(self, m, n, L):
        super(GRUInterface, self).__init__()
        self.m = m
        self.n = n
        self.L = L
        self.gru_ = torch.nn.GRU(self.n, self.m)
        self.V_ = torch.nn.Linear(self.m, self.L)

    def forward(self, X):
        # add another input dimension to make the input
        # compatible with the torch GRU implementation.
        X = torch.tensor(X, dtype=torch.float).unsqueeze(1)
        # process with GRU
        H, _ = self.gru_(X)
        # apply output layer
        return self.V_(H.squeeze(1))

