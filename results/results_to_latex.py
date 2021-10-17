""" Convers the results CSV files for all data sets to a LaTeX table
in the paper.

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

__author__ = 'Benjamin Paaßen'
__copyright__ = 'Copyright 2020, Benjamin Paaßen'
__license__ = 'GPLv3'
__version__ = '1.0.0'
__maintainer__ = 'Benjamin Paaßen'
__email__  = 'bpaassen@techfak.uni-bielefeld.de'

datasets = ['latch', 'copy', 'repeat_copy', 'signal_copy', 'image_copy', 'fsms', 'associative_recall']
dataset_labels = ['latch', 'copy', 'repeat copy', 'smooth recall', 'image copy', 'FSMs', 'assoc.\ recall']
models   = ['rand', 'CRJ', 'LDN', 'GRU', 'GRU-MM', 'rand-RMM', 'CRJ-RMM', 'LDN-RMM']
model_groups = [3, 2, 3]

# define a function to print a LaTeX table with models as columns and datasets
# as rows
def latex_table(data, highlight_lowest = True):
    # print table header
    print('\\begin{tabular}{c%s}' % ('c' * len(models)))
    print('dataset & %s \\\\' % ' & '.join(models))
    cmidrule_str = '\\cmidrule(lr){1-1}'
    col_idx = 2
    for col_offset in model_groups:
        cmidrule_str += '\\cmidrule(lr){%d-%d}' % (col_idx, col_idx + col_offset - 1)
        col_idx += col_offset
    print(cmidrule_str)
    # start iterating over rows, one per dataset
    for data_idx in range(len(datasets)):
        row = dataset_labels[data_idx]
        # compute the lowest value for this dataset
        data_for_row = data[:, data_idx, :].copy()
        data_for_row[np.all(np.isnan(data_for_row), 1), :] = np.inf
        lowest = np.min(np.nanmean(data_for_row, 1))
        # print all values
        for model_idx in range(len(models)):
            data_for_cell = data[model_idx, data_idx, :]
            # print only a dash if the data is missing
            if np.all(np.isnan(data_for_cell)):
                row += ' & -'
            else:
                # format mean and standard deviation, ignoring nan values
                data_string = '%.2f \\pm %.2f' % (np.nanmean(data_for_cell), np.nanstd(data_for_cell))
                # highlight the lowest result in each row
                if highlight_lowest and np.nanmean(data_for_cell) < lowest + 1E-3:
                    row += ' & $\\bm{%s}$' % data_string
                else:
                    row += ' & $%s$' % data_string
        row += '\\\\'
        print(row)
    print('\\end{tabular}')

num_repeats  = 20

errors   = np.full((len(models), len(datasets), num_repeats), np.nan)
runtimes = np.full((len(models), len(datasets), num_repeats), np.nan)

esn_model_idxs = [0, 1, 2, 5, 6, 7]
gru_model_idxs = [3, 4]

for data_idx in range(len(datasets)):
    dataset = datasets[data_idx]
    # load the errors for the task
    data = np.loadtxt('%s_errors.csv' % dataset, delimiter = '\t', skiprows = 1)
    errors[esn_model_idxs, data_idx, :] = data.T
    if data_idx < 4:
        data = np.loadtxt('%s_deep_errors.csv' % dataset, delimiter = '\t', skiprows = 1)
        errors[gru_model_idxs, data_idx, :len(data)] = data.T
    # load the runtimes for the task
    data = np.loadtxt('%s_runtimes.csv' % dataset, delimiter = '\t', skiprows = 1)
    runtimes[esn_model_idxs, data_idx, :] = data.T
    if data_idx < 4:
        data = np.loadtxt('%s_deep_runtimes.csv' % dataset, delimiter = '\t', skiprows = 1)
        runtimes[gru_model_idxs, data_idx, :len(data)] = data.T

# print error table
print('\nerrors:\n')
latex_table(errors)
# print runtime table
print('\n\nruntimes:\n')
latex_table(runtimes)
