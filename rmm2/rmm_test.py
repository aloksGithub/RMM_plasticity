#!/usr/bin/python3
"""
Tests reservoir memory machines.

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
from scipy.sparse import csr_matrix
import numpy as np
import rmm2.fsm as fsm
import rmm2.esn as esn
import rmm2.lmu as lmu
import rmm2.rmm as rmm

__author__ = 'Benjamin Paaßen'
__copyright__ = 'Copyright 2020, Benjamin Paaßen'
__license__ = 'GPLv3'
__version__ = '0.0.1'
__maintainer__ = 'Benjamin Paaßen'
__email__  = 'benjamin.paassen@sydney.edu.au'

class TestRMM(unittest.TestCase):

    def test_latch(self):
        # test whether our RMM implementation is able to implement a latching
        # switch, i.e. outputting a zero until we have a one at input and
        # then outputting one until the next input one

        # generate input data
        T = 100
        N = 10
        Xs = []
        Qs = []
        Ys = []
        for j in range(N):
            # initialize input and output sequence
            X = np.zeros((T, 1))
            Y = np.zeros((T, 1))
            # sample two switching points randomly
            t_1 = random.randrange(10, 40)
            t_2 = random.randrange(60, 90)
            # set input at precisely these locations to one
            X[0] = 1.
            X[t_1] = 1.
            X[t_2] = 1.
            # activate output between these points
            Y[t_1:t_2] = 1.
            # append
            Xs.append(X)
            Ys.append(Y)
            Qs.append(np.round(Y.squeeze() + 1))

        # set up model
        m = 64
        n = 1

        model = rmm.RMM(m, n, input_normalization = False)

        # train the model
        model.fit(Xs, Qs, Ys)

        # test the result
        for j in range(N):
            Yhat = model.predict(Xs[j])

            se = np.sum(np.square(Ys[j] - Yhat))
            self.assertTrue(se < 0.1, 'SE was too high, got %g; predicted vs actual:\n%s' % (se, str(np.concatenate([Yhat, Ys[j]], axis=1))))


    def test_repeat_copy(self):
        # test whether our RMM implementation can solve the repeat copy task,
        # i.e. read in a time series from the input stream and output it
        # multiple times again once a special marker appears in the input

        # generate input data
        T = 10
        max_repeats = 10
        n = 8
        N = 10
        Xs = []
        Qs = []
        Ys = []
        for j in range(N):
            # sample the number of repeats
            R = random.randrange(1, max_repeats+1)
            # initialize the input and output sequence
            X = np.zeros(((R+1)*(T+1), n+1))
            Q = np.zeros((R+1)*(T+1))
            Y = np.zeros(((R+1)*(T+1), n))
            # fill the input sequence with random bits
            X[1:T+1, :n] = np.round(np.random.rand(T, n))
            # and copy the sequence R times to the output
            for r in range(R):
                lo = (r+1)*(T+1)
                hi = (r+2)*(T+1)
                # before each repeat, indicate that another
                # output should follow on another channel
                X[lo, n] = 1.
                Q[lo] = 1.
                # then copy the input time series
                Y[lo+1:hi, :] = X[1:T+1, :n]
            Xs.append(X)
            Qs.append(Q)
            Ys.append(Y)

        # set up model

        # To make this more reliable, we pre-set the input and memory matrix
        # to reliably store the input information. In particular, we set U
        # and W such that the t+(T+2)*ith neuron stores the ith input
        # channel t steps in the past.

        m = n * (T+2) + 1

        # set up U matrix
        U = np.zeros((m, n+1))
        for i in range(n):
            U[(T+2) * i, i] = 1.
        U[m-1, n] = 1.
        # set up W matrix
        rows = []
        cols = []
        data = []
        for i in range(n):
            for t in range(T+2):
                rows.append(i*(T+2)+t+1)
                cols.append(i*(T+2)+t)
                data.append(1. / np.tanh(1.))
        rows.append(m-1)
        cols.append(m-1)
        data.append(1. / np.tanh(1.))
        W = csr_matrix((data, (rows, cols)), shape=(m, m))
        model = rmm.RMM(U, W, input_normalization = False)

        # train the model
        model.fit(Xs, Qs, Ys)

        # test the result
        for j in range(N):
            Ypred, Qpred = model.predict(Xs[j], True)

            np.testing.assert_allclose(Qpred, Qs[j])
            np.testing.assert_allclose(np.round(Ypred), Ys[j])

    def test_signal_copy(self):
        T_marker  = 32
        T_signal = 256

        # first, we define a fixed list of marker tokens, each a
        # continuous signal of length T_marker, which can then be associated
        # with an arbitrary input signal
        marker_tokens = [
            np.sin(np.arange(T_marker) / (T_marker - 1) * 2 * np.pi),
            np.exp(-np.square(np.arange(T_marker)-(T_marker-1)*0.5)/(T_marker/5)**2)
        ]
        num_signals = len(marker_tokens)
        num_repeats = 10
        N = 5

        # set up sine waves of various frequencies that are zero
        # at the start and the end of the signal
        num_freqs = 16
        Sines = np.zeros((num_freqs, T_signal))
        for f in range(num_freqs):
            Sines[f, :] = np.sin(f*np.arange(T_signal)/T_signal*np.pi)

        # generate data
        Xs = []
        Qs = []
        Ys = []
        for j in range(N):
            # initialize the input and output sequence
            T = (num_signals + num_repeats+1) * T_signal
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
                # put the s-th marker token on the second input channel
                # after the signal is completed
                X[hi-T_marker:hi, 1] = marker_tokens[s]
                # mark the state
                Q[hi] = s + 1
                # put the signal to the output as well
                Y[hi:hi+T_signal, 0] = signal
            # generate the output
            for r in range(num_repeats):
                lo = (num_signals+r+1)*T_signal
                hi = (num_signals+r+2)*T_signal
                # sample the signal we wish to repeat
                s  = random.randrange(num_signals)
                # mark the state
                Q[lo] = s + 1
                # write the s-th marker token on the second input channel
                X[lo-T_marker:lo, 1] = marker_tokens[s]
                # write the s-th signal on the output channel
                Y[lo:hi, 0] = X[s*T_signal:(s+1)*T_signal, 0]
            # append to training data
            Xs.append(X)
            Qs.append(Q)
            Ys.append(Y)

        # set up a RMM
        degree = 64 - 1
        U, W = lmu.initialize_reservoir(2, degree, T = 1.5 * T_signal)
        nonlin = lambda x : x
        model = rmm.RMM(U, W, regul = 1E-8, input_normalization = False, nonlin = nonlin, C = 100000., svm_kernel = 'rbf')

        # train
        model.fit(Xs, Qs, Ys)

        # test prediction
        for j in range(N):
            Yhat, Qhat = model.predict(Xs[j], True)
            np.testing.assert_allclose(Qhat, Qs[j])
            rmse = np.sqrt(np.mean(np.square(Yhat - Ys[j])))
            self.assertTrue(rmse < 0.1, 'rmse was unexpectedly high: %g' % rmse)

#            import matplotlib.pyplot as plt
#            plt.plot(Xs[j][:, 1])
#            plt.plot(Qs[j])
#            plt.plot(Qhat)
#            plt.legend(['input', 'state', 'predicted state'])
#            plt.show()
#            plt.plot(Ys[j])
#            plt.plot(Yhat)
#            plt.legend(['output', 'predicted output'])
#            plt.show()

    def test_fsm_to_rmm(self):
        # convert a trivial FSM with one state and one symbol for
        # input and output
        delta = np.array([[0]])
        rho   = np.array([0])
        model = rmm.fsm_to_rmm(delta, rho)
        X     = np.ones((10, 1))
        Y, Q  = model.predict(X, True)
        np.testing.assert_allclose(Y, np.zeros((len(X), 1)))
        np.testing.assert_allclose(Q, np.ones(len(X)))

        # convert the FSM which has only one symbol but only accepts
        # if the number of input symbols is even
        delta = np.array([[1], [0]])
        rho   = np.array([1, 0])
        model = rmm.fsm_to_rmm(delta, rho)
        # test with a long sequence that an echo state network would
        # not be able to memorize
        T = 1000
        X     = np.ones((T, 1))
        Y, Q  = model.predict(X, True)
        # compare with expected output
        expected_Y = np.zeros((T, 1))
        expected_Y[1::2, :] = 1.
        expected_Q = (2. - expected_Y).squeeze()
        np.testing.assert_allclose(Y, expected_Y)
        np.testing.assert_allclose(Q, expected_Q)

        # convert an FSM which outputs 1 whenever the number of bs is
        # divisible by 3
        delta = np.array([[0, 1], [1, 2], [2, 0]])
        rho   = np.array([1, 0, 0])

        model = rmm.RMM(128, 2, input_normalization = False, discrete_prediction = True, q_0 = 1, C = 10000.)
        model = rmm.fsm_to_rmm(delta, rho, rmm = model)
        # test with a long sequence that an echo state network would
        # not be able to memorize
        T = 100
        X = np.round(np.random.rand(T, 2))
        X[:, 1] = np.round(1. - X[:, 0])
        Y, Q = model.predict(X, True)
        # compare with expected output
        expected_Y = np.zeros((T, 1))
        expected_Q = np.zeros(T)
        q = 0
        for t in range(T):
            x = int(X[t, 1])
            q = delta[q, x]
            expected_Q[t] = q + 1
            expected_Y[t] = rho[q]

        np.testing.assert_allclose(Y, expected_Y)
        np.testing.assert_allclose(Q, expected_Q)


    def test_fsm_to_rmm_large(self):
        # in this test, we evaluate the FSM to RMM conversion on a larger
        # number of randomly sampled FSMs
        m = 128
        num_states = 3
        num_in_symbols = 2
        num_out_symbols = 2

        # for evaluation, we also set up a number of long input sequences
        # to ensure that the RMM doesn't just memorize the training data
        N_test = 20
        Xs_test = np.random.randint(num_in_symbols, size=(N_test, 2 * m, num_in_symbols))
        Xs_test[:, :, 1] = 1 - Xs_test[:, :, 0]

        # then start sampling FSMs
        for r in range(10):
            # sample a random FSM
            delta, rho = fsm.sample_fsm(num_states, num_in_symbols, num_out_symbols)
            # iniitalize a RMM model
            model = rmm.RMM(m, num_in_symbols, input_normalization = False, discrete_prediction = True, q_0 = 1, C = 10000.)
            # convert the FSM into RMM
            model = rmm.fsm_to_rmm(delta, rho, rmm = model)
            # evaluate the prediction
            Ys_test, Qs_test = fsm.label_sequences(Xs_test[:, :, 1], delta, rho)
            # compare with model prediction
            for i in range(N_test):
                Ypred, Qpred = model.predict(Xs_test[i, :, :], True)

                np.testing.assert_allclose(Ypred.squeeze(), Ys_test[i])
                np.testing.assert_allclose(Qpred, Qs_test[i]+1)

if __name__ == '__main__':
    unittest.main()
