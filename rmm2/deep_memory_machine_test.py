#!/usr/bin/python3
"""
Tests the deep memory machine.

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
import torch
import lmu
from dataset_generators import generate_data
import deep_memory_machine as dmm

__author__ = 'Benjamin Paaßen'
__copyright__ = 'Copyright 2020, Benjamin Paaßen'
__license__ = 'GPLv3'
__version__ = '0.0.1'
__maintainer__ = 'Benjamin Paaßen'
__email__  = 'benjamin.paassen@sydney.edu.au'

class TestDMM(unittest.TestCase):

    def test_gru_learning(self):
        # test gru learning on the latch task
        model = dmm.GRUInterface(n = 1, m = 64, L = 1)
        # set training hyperparams
        num_epochs = 3000
        minibatch_size = 8
        lr = 1E-3
        weight_decay = 1E-8
        loss_fun = torch.nn.MSELoss()
        optim = torch.optim.Adam(model.parameters(), lr = lr, weight_decay = weight_decay)
        # set up aux variables
        loss_avg = None
        avg_factor = 0.1
        # start training
        for epoch in range(num_epochs):
            optim.zero_grad()
            # generate minibatch of data
            Xs, Qs, Ys = generate_data(minibatch_size, 'latch')
            minibatch_loss = torch.zeros(1)
            for i in range(minibatch_size):
                # process current sequence with GRU
                Ypred = model(Xs[i])
                # compute loss
                loss  = loss_fun(Ypred, torch.tensor(Ys[i], dtype=torch.float))
                # add to minibatch
                minibatch_loss = minibatch_loss + loss
            # compute gradient
            minibatch_loss.backward()
            # perform optimization step
            optim.step()
            # record loss
            if loss_avg is None:
                loss_avg = minibatch_loss.item() / minibatch_size
            else:
                loss_avg = avg_factor * minibatch_loss.item() / minibatch_size + (1. - avg_factor) * loss_avg
            if (epoch + 1) % 10 == 0:
                print('moving average loss in epoch %d: %g' % (epoch+1, loss_avg))
            if loss_avg < 1E-3:
                print('ended training already after %d epochs because loss was very small.' % (epoch + 1))
                break

        # check prediction on a new minibatch
        Xs, Qs, Ys = generate_data(minibatch_size, 'latch')
        test_loss = 0.
        for i in range(minibatch_size):
            Ypred = model(Xs[i])
            test_loss += loss_fun(Ypred, torch.tensor(Ys[i], dtype=torch.float)).item()
        self.assertTrue(test_loss / minibatch_size < 1E-2)


    def test_dmm_learning(self):
        # test DMM learning on the latch task, where we use an LMU reservoir
        # U, W = lmu.initialize_reservoir(n = 1, degree = 15, T = 128)
        model = dmm.DeepMemoryMachine(U = 64, W = 1, K = 2, L = 1, nonlin = lambda x : x)
        # set training hyperparams
        num_epochs = 3000
        minibatch_size = 8
        lr = 1E-2
        weight_decay = 1E-8
        optim = torch.optim.Adam(model.parameters(), lr = lr, weight_decay = weight_decay)
        # set up aux variables
        loss_avg = None
        avg_factor = 0.1
        # start training
        for epoch in range(num_epochs):
            optim.zero_grad()
            # generate minibatch of data
            Xs, Qs, Ys = generate_data(minibatch_size, 'latch')
            minibatch_loss = torch.zeros(1)
            for i in range(minibatch_size):
                # compute teacher forcing loss
                loss = model.compute_teacher_forcing_loss(Xs[i], Qs[i], Ys[i])
                # add to minibatch
                minibatch_loss = minibatch_loss + loss
            # compute gradient
            minibatch_loss.backward()
            # perform optimization step
            optim.step()
            # record loss
            if loss_avg is None:
                loss_avg = minibatch_loss.item() / minibatch_size
            else:
                loss_avg = avg_factor * minibatch_loss.item() / minibatch_size + (1. - avg_factor) * loss_avg
            if (epoch + 1) % 10 == 0:
                print('moving average loss in epoch %d: %g' % (epoch+1, loss_avg))
            if loss_avg < 1E-3:
                print('ended training already after %d epochs because loss was very small.' % (epoch + 1))
                break

        # check prediction on a new minibatch
        Xs, Qs, Ys = generate_data(minibatch_size, 'latch')
        test_loss = 0.
        for i in range(minibatch_size):
            Ypred = model(Xs[i])
            test_loss += torch.nn.functional.mse_loss(Ypred, torch.tensor(Ys[i], dtype=torch.float)).item()
        self.assertTrue(test_loss / minibatch_size < 1E-2)

if __name__ == '__main__':
    unittest.main()
