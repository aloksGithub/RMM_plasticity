"""
Implements a pseudo support vector machine which learns the class prediction
matrix via linear regression and then selects the recognition threshold via
binary search. This is particularly useful for large data sets where a standard
SVM would take too long.

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


import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

import matplotlib.pyplot as plt

__author__ = 'Benjamin Paaßen'
__copyright__ = 'Copyright 2020, Benjamin Paaßen'
__license__ = 'GPLv3'
__Version__ = '0.0.1'
__maintainer__ = 'Benjamin Paaßen'
__email__  = 'benjamin.paassen@sydney.edu.au'

class PseudoSVM(BaseEstimator, ClassifierMixin):
    """ A pseudo support vector machine which classifies data via linear
    regression and thresholding.

    In particular, the classifier learns a num_features x (num_classes-1)
    matrix V_ which is multiplied with the input features. Then, it subtracts
    thresholds. Any data point that does not exceed 0 is assigned to the
    majority class. Otherwise, the class with maximum activation is assigned.

    Attributes
    ----------
    regul: float (default = 1E-8)
        The regular regression regularization parameter. Similar to 1. / C
        for SVM.
    balance_classes: bool (default = True)
        A flag whether class weights should be applied to balance our the
        influence of classes.
    classes_: class numpy.array
        The vector of possible class labels, where the last entry is the
        majority class.
    V_: class numpy.array
        The num_features x (num_classes-1) recognition matrix.
    thresholds_: class numpy.array
        A (num_classes-1) vector of class-wise recognition thresholds.

    """
    def __init__(self, regul = 1E-8, balance_classes = False, margin = 0.1):
        self.regul = regul
        self.balance_classes = balance_classes
        self.margin = margin

    def fit(self, X, y):
        """ Fits this PseudoSVM to the given data.

        Parameters
        ----------
        X: class numpy.array
            A num_data x num_features matrix representing the input data.
        y: class numpy.array
            A num_data vector containing the desired class labels.

        Returns
        -------
        self: class PseudoSVM
            this instance after training.

        """

        # determine the number of points in each class
        unique_classes = np.unique(y)
        class_nums = np.zeros(len(unique_classes))
        for l in range(len(unique_classes)):
            class_nums[l] = np.sum(y == unique_classes[l])
        # assign class weights
        weights = np.ones_like(y)
        if self.balance_classes:
            for l in range(len(unique_classes)):
                weights[y == unique_classes[l]] /= class_nums[l]
            weights = weights / len(unique_classes)
        else:
            weights = weights / len(y)
        # put the majority class at the end of the class list
        lmax = np.argmax(class_nums)
        self.classes_ = unique_classes
        self.classes_[lmax], self.classes_[-1] = self.classes_[-1], self.classes_[lmax]
        # design a new desired output matrix via one-hot coding for all
        # classes except the majority class
        L = len(unique_classes)-1
        Y = np.zeros((len(y), L))
        for l in range(L):
            Y[y == self.classes_[l], l] = 1.
        # perform weighted linear regression
        XwT = (X * np.expand_dims(weights, 1)).T
        self.V_ = np.linalg.solve(np.dot(XwT, X) + self.regul / len(y) * np.eye(X.shape[1]), np.dot(XwT, Y))
        # get class decision function values
        Ypred = np.dot(X, self.V_)
        # find the optimal recognition threshold for each class via binary
        # search
        self.thresholds_ = np.zeros(L)
        for l in range(L):
            ypred_l = Ypred[:, l]
            y_l = 2 * Y[:, l] - 1
            # start with the maximum and minimum predicted value
            hi = np.max(ypred_l)
            lo = np.min(ypred_l)
            hi_loss = compute_svm_loss_(ypred_l, y_l, hi, weights, self.margin)
            lo_loss = compute_svm_loss_(ypred_l, y_l, lo, weights, self.margin)
            loss_delta = np.abs(hi_loss - lo_loss)
            # continue search until loss doesn't change much anymore
            while loss_delta > 1E-3 / len(y):
                # compute the loss at the mid point between hi and lo
                mid = 0.5 * (hi + lo)
                mid_loss = compute_svm_loss_(ypred_l, y_l, mid, weights, self.margin)
                # check if we can reduce the loss by moving hi or lo to
                # mid
                if hi_loss > lo_loss and hi_loss > mid_loss:
                    hi = mid
                    hi_loss = mid_loss
                elif lo_loss > hi_loss and lo_loss > mid_loss:
                    lo = mid
                    lo_loss = mid_loss
                else:
                    # if neither is possible, the function does not have
                    # a unique low point and our search fails
                    raise ValueError('Internal error: Binary search failed.')
                loss_delta = np.abs(hi_loss - lo_loss)
            # take the threshold with lower loss
            if hi_loss > lo_loss:
                self.thresholds_[l] = lo
            else:
                self.thresholds_[l] = hi
        # return this instance
        return self

    def decision_function(self, X):
        """ Returns the decision function values of this pseudo-SVM.

        In particular, this returns np.dot(X, self.V_) - self.thresholds_


        Parameters
        ----------
        X: class numpy.array
            A num_data x num_features matrix representing the input data.

        Returns
        -------
        Y: class numpy.array
            A num_data x num_classes -1 matrix containing all decision function
            values.

        """
        return np.dot(X, self.V_) - np.expand_dims(self.thresholds_, 0) + 0.5 * self.margin

    def predict(self, X):
        """ Predicts class labels for the given input.

        Parameters
        ----------
        X: class numpy.array
            A num_data x num_features matrix representing the input data.

        Returns
        -------
        Y: class numpy.array
            A num_data vector containing predicted class labels.

        """
        # compute class predictions
        Ypred = self.decision_function(X)
        # per default, predict the majority class
        ypred = np.full(len(Ypred), self.classes_[-1])
        # retrieve all points that are not predicted as majority class
        nonzeros = np.where(np.max(Ypred, 1) >= 0.)[0]
        # amongst those, assign the correct classes
        ypred[nonzeros] = self.classes_[np.argmax(Ypred[nonzeros, :], 1)]
        # return
        return ypred

def compute_svm_loss_(ypred, y, threshold, weights, margin = 1E-3):
    """ Computes the SVM loss for the given decision function values, the given
    actual labels, and the given threshold. This loss is given as

    max(-(ypred[i] - threshold) * y[i] + margin, 0)^2 * weights[i]

    for each point i.

    Parameters
    ----------
    ypred: class numpy.array
        The decision function values for the current class, where higher values
        mean more confidence that this class should be predicted.
    y: class numpy.array
        The actual labels for the current class, where -1 means that the class
        should not be predicted and +1 means that it should be predicted.
    threshold: float
        The current threshold.
    weights: class numpy.array
        The weights for each point.
    margin: float (default = 1E-3)
        The margin value which punishes points that are classified correctly,
        but are very close to the decision boundary.

    Returns
    -------
    loss: float
        The SVM loss for the given inputs.

    """
    epsilons = -(ypred - threshold) * y + margin
    epsilons[epsilons < 0.] = 0.
    return np.dot(weights, np.square(epsilons))
