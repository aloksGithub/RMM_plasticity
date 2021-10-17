#!/usr/bin/python3
"""
Tests associative reservoir memory machines.

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

import unittest
import random
from sklearn.svm import SVC
from scipy.spatial.distance import cdist
from scipy.linalg import block_diag
import numpy as np
import rmm2.lmu as lmu
import rmm2.crj as crj
import rmm2.rmm_assoc as rmm_assoc

import matplotlib.pyplot as plt

__author__ = 'Benjamin Paaßen'
__copyright__ = 'Copyright 2020, Benjamin Paaßen'
__license__ = 'GPLv3'
__version__ = '0.0.1'
__maintainer__ = 'Benjamin Paaßen'
__email__  = 'benjamin.paassen@sydney.edu.au'

class TestRMM_Assoc(unittest.TestCase):

    def test_learn_association_mappings_alignment(self):
        # check a case of data where we need dimensions 1 and 2 from the first
        # dataset but dimensions 9 and 10 from another and our mappings are
        # unit vectors
        X = np.random.randn(400, 4)
        X[:200, 0] += 8
        X[200:, 1] += 8
        Y = np.random.randn(400, 10)
        Y[:100, 8] += 8
        Y[100:200, 9] += 8
        Y[200:300, 8] += 8
        Y[300:400, 9] += 8

        # we want to associate points 0-99 and 300-399 but not the rest
        Z = np.ones(400)
        Z[100:300] = -1

        # set up the mappings
        Us = []
        for k in range(X.shape[1]):
            Us.append(np.eye(X.shape[1])[k, :][:, None])

        Vs = []
        for l in range(Y.shape[1]):
            Vs.append(np.eye(Y.shape[1])[l, :][:, None])

        # learn the mapping
        U, V, b = rmm_assoc.learn_association_mappings_alignment_(X, Y, Z, Us, Vs)

        # check the association loss
        dists = np.sum(np.square(np.dot(X, U) - np.dot(Y, V)), 1)
        epsilons = (dists - b) * Z - b
        epsilons[epsilons < 0.] = 0.
        loss = np.sum(epsilons)
        self.assertTrue(loss < 1E-3)


    def test_manual_association(self):
        # check if we can manually set up a mapping between memory and state
        # such that the association works reliably.
        # In particular, we would like to associate the input in the last time
        # step to the input that happened two steps ago in each memory state.
        # If these are equal, the association mechanism should return +1,
        # otherwise -1.

        # set up a Legendre memory unit
        n = 8
        degree = 15
        U, W = lmu.initialize_reservoir(n, degree, 16)
        m = U.shape[0]

        # as test input we use random bit patterns.
        T = 10
        N = 100
        Xs = []
        Hs = []
        Qs = []
        for j in range(N):
            # draw a random sequence of T non-repeating integers in
            # the range 2^n
            x = np.random.choice(np.power(2, n), size = T, replace = False)
            X = np.zeros((T+1, n))
            for t in range(T):
                for j in range(n):
                    X[t, j] = x[t] % 2
                    x[t] = int(x[t] / 2)
            # set the Tth step to a random copy of the input
            i = random.randrange(T-1)
            X[T, :] = X[i, :]
            Xs.append(X)
            # compute the states
            H = np.zeros((T+1, m))
            for t in range(T+1):
                if t == 0:
                    H[t, :] = np.dot(U, X[t, :])
                else:
                    H[t, :] = np.dot(U, X[t, :]) + W * H[t-1, :]
            Hs.append(H)
            # record the item that we copied
            Qs.append(i)

        # train a linear regression that recovers from the final state
        # the input that just happened
        H_fin = np.stack([H[-1, :] for H in Hs[:N-10]], 0)
        X_fin = np.stack([X[-1, :] for X in Xs[:N-10]], 0)
        # but train it blockwise
        H_fin = np.reshape(H_fin, (H_fin.shape[0] * n, degree+1))
        X_fin = np.reshape(X_fin, (X_fin.shape[0] * n, 1))

        V_left = np.linalg.solve(np.dot(H_fin.T, H_fin) + np.eye(degree+1) * 1E-8, np.dot(H_fin.T, X_fin))
        state_rmse = np.sqrt(np.mean(np.square(X_fin - np.dot(H_fin, V_left))))
        self.assertTrue(state_rmse < 1E-3)

        # train another linear regression that recovers from any state
        # the input that happened before
        H = np.concatenate([H[1:-1, :] for H in Hs[:N-10]], 0)
        X_before = np.concatenate([X[:-2, :] for X in Xs[:N-10]], 0)
        # but train it blockwise
        H = np.reshape(H, (H.shape[0] * n, degree+1))
        X_before = np.reshape(X_before, (X_before.shape[0] * n, 1))
        V_right = np.linalg.solve(np.dot(H.T, H) + np.eye(degree+1) * 1E-8, np.dot(H.T, X_before))
        mem_rmse = np.sqrt(np.mean(np.square(X_before - np.dot(H, V_right))))
        self.assertTrue(mem_rmse < 1E-3)

        # then extend V_left and V_right to a block-diagonal matrix
        V_left = block_diag(*([V_left] * n))
        V_right = block_diag(*([V_right] * n))

        # now use the distance in the reconstruction space to decide which
        # memory state the final state should be associated to
        for j in range(N):
            # we are interested in the associated input for the final state
            x_left  = np.dot(Hs[j][-1, :], V_left)
            X_right = np.dot(Hs[j][1:-1, :], V_right)
            ds = np.sum(np.square(x_left[None, :] - X_right), 1)
            i_selec = np.argmin(ds)
            if i_selec != Qs[j]:
                print('distances for sequence %d: %s' % (j, str(ds)))
                print('i_selec = %d versus i_actual = %d' % (i_selec, Qs[j]))
                self.fail()


    def test_learn_association_mappings(self):
        # check if we can learn association mappings between memory and state
        # such that the association works reliably.
        # In particular, we would like to associate the input 0-2 steps ago
        # with the input 3-5 steps ago encoded in memory.
        # If these are equal, the association mechanism should return +1,
        # otherwise -1.

        # as test input we use random bit patterns.
        N = 100
        n = 6
        block_length = 3
        max_blocks = 6
        N = 100

        # set up a Legendre memory unit
        degree = 15
        U, W = lmu.initialize_reservoir(n+1, degree, 16)
        m = U.shape[0]

        Xs = []
        Hs = []
        Qs = []
        for j in range(N):
            T = max_blocks
            # initialize the input sequence
            X = np.zeros(((block_length)*(T+2)+1, n+1))
            # fill the input sequence with random bits
            X[:T*block_length, :n] = np.round(np.random.rand(T*block_length, n))
            # add the sequence end token
            X[T*block_length, n] = 1.
            # enumerate the blocks in the state sequence
            Q = np.zeros((block_length)*(T+2)+1)
            Q[(2*block_length-1):T*block_length:block_length] = np.arange(1, T)
            # then select a random item
            i = random.randrange(T-1)
            # and copy it once again to the input
            lo = T*block_length+1
            hi = (T+1)*block_length+1
            X[lo:hi,:] = X[(i*block_length):((i+1)*block_length), :]
            # set the associated state
            Q[hi] = i+1

            # construct the state sequence
            H = []
            h = np.zeros(U.shape[0])
            for t in range(len(X)):
                h = np.dot(U, X[t, :]) + W * h
                H.append(h)

            # append
            Xs.append(X)
            Hs.append(np.stack(H, 0))
            Qs.append(Q)

#            # FOR DEBUG PURPOSES: show an example sequence
#            plt.imshow(np.concatenate((X, Q[:, None] / np.max(Q)), 1))
#            plt.title('input')
#            plt.xlabel('bit')
#            plt.ylabel('time')
#            plt.show()
#            raise ValueError('stop')
#            # END DEBUG REGION

        # prepare the linear operators which map from the state at time t
        # to the input tau steps ago
        operators = []
        for tau in range(2*block_length):
            # accumulate the training data for the current tau
            H_tau = []
            X_tau = []
            for i in range(len(Xs)):
                H_tau.append(Hs[i][tau:, :])
                X_tau.append(Xs[i][:len(Xs[i])-tau, :])
            H_tau = np.concatenate(H_tau, 0)
            X_tau = np.concatenate(X_tau, 0)
            # infer the operator via linear regression
            V_tau = np.linalg.solve(np.dot(H_tau.T, H_tau) + 1E-8 * np.eye(H_tau.shape[1]), np.dot(H_tau.T, X_tau))
            # ensure that the operator works with very low error
            rmse_tau = np.sqrt(np.mean(np.square(np.dot(H_tau, V_tau) - X_tau)))
            self.assertTrue(rmse_tau < 0.1, 'unexpectedly high error in constructing lookback operator for step %d: %g' % (tau, rmse_tau))
            operators.append(V_tau)

        # prepare the training data for associations
        H_train = []
        M_train = []
        Q_train = []
        for j in range(N-10):
            M = np.zeros((max_blocks-1, U.shape[0]))
            k = 0
            for t in range(len(Xs[j])):
                if t > block_length and k < max_blocks-1 and t % block_length == block_length-1:
                    M[k, :] = Hs[j][t, :]
                    k += 1
                    continue
                for l in range(k):
                    H_train.append(Hs[j][t])
                    M_train.append(M[l, :])
                    if Qs[j][t] == l+1:
                        Q_train.append(+1)
                    else:
                        Q_train.append(-1)

        # concatenate the training input
        H_train = np.stack(H_train, 0)
        M_train = np.stack(M_train, 0)
        Q_train = np.array(Q_train)

        # construct the overall association mapping
        U, V, b = rmm_assoc.learn_association_mappings_alignment_(H_train, M_train, Q_train, operators, operators)

        b -= 0.9

#        # FOR DEBUG PURPOSES: define mapping manually
#        U = np.concatenate(operators[1:4], 1)
#        V = np.concatenate(operators[3:6], 1)
#        b = 1.
#        # END DEBUG REGION

        # check the train accuracy
        dists = np.sum(np.square(np.dot(H_train, U) - np.dot(M_train, V)), 1)
        self.assertTrue(np.mean(dists[Q_train > 0.] - b < 0.) > 0.99)
        self.assertTrue(np.mean(dists[Q_train < 0.] - b > 0.) > 0.99)

        # check how well this generalizes to test data
        for j in range(N):
            M = np.zeros((max_blocks, V.shape[1]))
            k = 0
            for t in range(len(Xs[j])):
                # write to memory every 3 steps
                if t > block_length and k < max_blocks-1 and t % block_length == block_length-1:
                    M[k, :] = np.dot(Hs[j][t, :], V)
                    k += 1
                    continue
                if k == 0:
                    continue
                # compute distances to memory
                h = np.dot(Hs[j][t, :], U)
                d = np.sum(np.square(h[None, :] - M[:k, :]), 1) - b
                # if we are in the association time step, ensure that an
                # association takes place
                if k == max_blocks-1 and Qs[j][t] > 0:
                    self.assertTrue(np.any(d < 0.), 'sequence %d: missing association with minimum distance %g.' % (j, np.min(d)))
                    i_selec = np.argmin(d)
                    i_true = int(Qs[j][t])-1
                    self.assertEqual(i_selec, i_true, 'sequence %d: Wanted memory entry %d but got %d by distance %g versus %g' % (j, i_true, i_selec, np.min(d), d[i_true]))
                else:

                    if np.any(d < 0.):
                        plt.imshow(np.stack((h, M[np.argmin(d), :])))
                        plt.show()

                    # otherwise, ensure that no association takes place
                    self.assertTrue(np.all(d > 0.), 'sequence %d: undesired association with memory entry %d at time step %d (distance: %g)' % (j, np.argmin(d), t, np.min(d)))

    def test_fit(self):
        # test whether our RMM implementation is able to select from a set of
        # available inputs the correct one

        # generate input data
        T = 10
        n = 8
        N = 100
        Xs = []
        Qs = []
        Ys = []
        for j in range(N):
            # draw a random sequence of T non-repeating integers in
            # the range 2^n
            x = np.random.choice(np.power(2, n)-1, size = T, replace = False)+1
            X = np.zeros((T+1, n))
            for t in range(T):
                for j in range(n):
                    X[t, j] = x[t] % 2
                    x[t] = int(x[t] / 2)
            # set the Tth step to a random copy of the input
            i = random.randrange(T-1)
            X[T, :] = X[i, :]
            # the state sequence should reflect the item index
            Q = np.arange(1, T+2)
            Q[T] = i+2
            # the output is just a copy of the input
            Y = X
            # append
            Xs.append(X)
            Qs.append(Q)
            Ys.append(Y)

        # set up model
        m = 128
        degree = int(m/n)-1
        U, W = lmu.initialize_reservoir(n, degree, 3)

        model = rmm_assoc.RMM(U, W, regul = 1E-7, input_normalization = False, C = 100., nonlin = lambda x : x, horizon = 2)

        # train the model on all sequences but the last 10 to ensure
        # some generalization
        model.fit(Xs[:N-10], Qs[:N-10], Ys[:N-10])

        # test the result
        errs = []
        for j in range(N):
            _, Qpred = model.predict(Xs[j], True)
            if np.any(np.abs(Qpred - Qs[j]) > 1E-3):
                print('wanted states: %s' % str(Qs[j].astype(int)))
                print('got states:    %s' % str(Qpred.astype(int)))
                errs.append(j)
        self.assertTrue(len(errs) < 5, 'Errors in predicting time series %s' % str(errs))

    def test_clock(self):
        # test whether we can make our initial state classifier return
        # a 1 every 3 steps, irrespective of the concrete input

        n = 6
        tau = 3
        T = 21
        N = 90
        Xs = []
        Qs = []
        Ys = []
        for j in range(N):
            # let the input sequence be completely random
            X = np.random.randint(2, size=(T, n+1))
            # and the state sequence should be a new state every tau steps
            Q = np.zeros(T)
            Q[tau-1:T:tau] = np.arange(1, int(T/tau)+1)
            X[:,n] = 0.
            X[-1,n] = 1.
            # the output is just a copy of the input
            Y = X
            # append
            Xs.append(X)
            Qs.append(Q)
            Ys.append(Y)

        # set up model
        m = 128

        degree = int(m/(n+1))-1
        U, W = lmu.initialize_reservoir(n+1, degree, T+1)

        e_n = np.zeros(n+1)
        e_n[n] = -1.
        init_state = np.dot(U, e_n)
        #init_state = None
        model = rmm_assoc.RMM(U, W, C = 1000., input_normalization = False, init_state = init_state, nonlin = lambda x : x)

        # train the model on all sequences but the last 10 to ensure
        # some generalization
        model.fit(Xs[:N-10], Qs[:N-10], Ys[:N-10])

        # test the result
        for j in range(N):
            _, Qpred = model.predict(Xs[j], True)
            np.testing.assert_allclose(Qpred, Qs[j])

    def test_associative_recall(self):
        # test whether our RMM implementation is able to solve the associative
        # recall task, i.e. to select the _next_ bit-block in memory after
        # seeing one
        n = 6
        block_length = 3
        max_blocks = 6
        N = 100
        Xs = []
        Qs = []
        Ys = []
        for j in range(N):
            # start by sampling the number of input blocks
            T = max_blocks#random.randrange(2, max_blocks+1)
            # initialize the input sequence
            X = np.zeros(((block_length)*(T+2)+1, n+1))
            # fill the input sequence with random bits
            X[:T*block_length, :n] = np.round(np.random.rand(T*block_length, n))
            # add the sequence end token
            X[T*block_length, n] = 1.
            # enumerate the blocks in the state sequence
            Q = np.zeros((block_length)*(T+2)+1)
            Q[(2*block_length-1):T*block_length:block_length] = np.arange(1, T)
            # then select a random item
            i = random.randrange(T-1)
            # and copy it once again to the input
            lo = T*block_length+1
            hi = (T+1)*block_length+1
            X[lo:hi,:] = X[(i*block_length):((i+1)*block_length), :]
            # set the associated state
            Q[hi-1] = i+1
            # copy the input sequence to the output
            Y = np.zeros(((block_length)*(T+2)+1, n))
            Y[block_length:, :] = X[:-block_length, :n]
            # finally, put the _next_  item in the input sequence to the output
            lo = hi
            i += 1
            Y[lo-1:, :] = X[(i*block_length)-1:((i+1)*block_length), :n]
            # append
            Xs.append(X)
            Qs.append(Q)
            Ys.append(Y)

#            # FOR DEBUG PURPOSES: Show an example sequence
#            plt.subplot(1, 2, 1)
#            plt.imshow(np.concatenate((X, Q[:, None] / np.max(Q)), 1))
#            plt.title('input')
#            plt.xlabel('bit')
#            plt.ylabel('time')
#            plt.subplot(1, 2, 2)
#            plt.imshow(Y)
#            plt.title('output')
#            plt.xlabel('bit')
#            plt.ylabel('time')
#            plt.show()
#            raise ValueError('stop')
#            # END DEBUG REGION


        # set up model
        m = 128

        degree = int(m/(n+1))-1
        U, W = lmu.initialize_reservoir(n+1, degree, block_length*max_blocks)

        e_n = np.zeros(n+1)
        e_n[n] = -1.
        init_state = np.dot(U, e_n)
        model = rmm_assoc.RMM(U, W, input_normalization = False, C = 1000., init_state = init_state, nonlin = lambda x : x, svm_kernel = 'rbf', horizon = 6)
        #model = rmm_assoc.RMM(m, n+1, input_normalization = False)

        # train the model on all sequences but the last 10 to ensure
        # some generalization
        model.fit(Xs[:N-10], Qs[:N-10], Ys[:N-10])

#        # FOR DEBUG PURPOSES: set association mapping manually
#        # prepare state sequences
#        Hs = []
#        for j in range(N):
#            h = np.zeros(U.shape[0])
#            H = []
#            for t in range(max_blocks*block_length):
#                h = np.dot(U, Xs[j][t, :]) + W * h
#                H.append(h)
#            Hs.append(np.stack(H))

#        # prepare the linear operators which map from the state at time t
#        # to the input tau steps ago
#        operators = []
#        for tau in range(2*block_length):
#            # accumulate the training data for the current tau
#            H_tau = []
#            X_tau = []
#            for i in range(len(Xs)):
#                H_tau.append(Hs[i][tau:, :])
#                X_tau.append(Xs[i][:max_blocks*block_length-tau, :])
#            H_tau = np.concatenate(H_tau, 0)
#            X_tau = np.concatenate(X_tau, 0)
#            # infer the operator via linear regression
#            V_tau = np.linalg.solve(np.dot(H_tau.T, H_tau) + 1E-8 * np.eye(H_tau.shape[1]), np.dot(H_tau.T, X_tau))
#            # ensure that the operator works with very low error
#            rmse_tau = np.sqrt(np.mean(np.square(np.dot(H_tau, V_tau) - X_tau)))
#            self.assertTrue(rmse_tau < 0.1, 'unexpectedly high error in constructing lookback operator for step %d: %g' % (tau, rmse_tau))
#            operators.append(V_tau)

#        model.assoc_U_ = np.concatenate(operators[0:3], 1)
#        model.assoc_V_ = np.concatenate(operators[3:6], 1)
#        model.assoc_b_ = 0.1
#        # END DEBUG REGION

        # test the result
        errs = []
        for j in range(N):
            Ypred, Qpred = model.predict(Xs[j], True)
            if np.any(np.abs(Qpred - Qs[j]) > 1E-3):
                print('wanted states (seq %d): %s' % (j, str(Qs[j].astype(int))))
                print('got states (seq %d):    %s' % (j, str(Qpred.astype(int))))
                errs.append(j)
        self.assertTrue(len(errs) < 11, 'Errors in predicting time series %s' % str(errs))



if __name__ == '__main__':
    unittest.main()
