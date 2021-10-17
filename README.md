# Reservoir Memory Machines as Neural Computers

Copyright (C) 2020  
Benjamin Paaßen, Alexander Schulz, Terrence C. Stewart  

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <http://www.gnu.org/licenses/>.

## Introduction

This is a reference implementation of _reservoir memory machines_ (RMMs)
as proposed in the paper 'Reservoir memory machines as neural computers',
submitted to the special issue 'New Frontiers in Extremely Efficient
Reservoir Computing' of the IEEE Transactions on Neural Networks and Learning
Systems (IEEE TNNLS).
We also provide here the experimental code for the experiments shown in the
paper.

If you use this code in academic work, please cite the paper

* Paaßen, Schulz, Stewart, and Hammer (2020). Reservoir Memory Machines as Neural Computers. IEEE Transactions on Neural Networks and Learning Systems. in print. doi:[10.1109/TNNLS.2021.3094139](https://doi.org/10.1109/TNNLS.2021.3094139). arXiv:[2009.06342][RMM]

## Quickstart Guide

To train your own reservoir memory machine, use the following code.

```
from rmm2.rmm import RMM
model = RMM(number_of_neurons, number_of_inputs)
model.fit(X, Q, Y)
Yhat = model.predict(X)
```

where `X` is the input time series, `Q` is a sequence of memory addresses which
defines when to store and read information, and `Y` is the desired output time
series, both with time steps as rows.

If you wish to use a Legendre delay network as reservoir, the code is slightly
longer:

```
import rmm2.lmu
from rmm2.rmm import RMM
U, W = lmu.initialize_reservoir(number_of_inputs, degree, T)
model = RMM(U, W)
model.fit(X, Q, Y)
Yhat = model.predict(X)
```

If you wish to use an associative reservoir memory machine, you can use the
same code but you need to import `from rmm2.rmm_assoc import RMM`.

## Contents

This repository contains the following files.

* `associative_recall.ipynb` : The experimental code for the
  _associative recall_ task from the paper.
* `dataset_generators.py` : An auxiliary file to generate data for the latch,
  copy, repeat copy, and signal copy (aka smooth associative recall) tasks.
* `deep_learning_experiments.ipynb` : A notebook for the deep learning
  experiments.
* `fsms.ipynb` : A notebook for the finite state machine experiments.
* `LICENSE.md` : A copy of the [GNU GPLv3][GPLv3].
* `README.md` : This file.
* `results` : A folder containing reference results on all tasks.
* `rmm2/crj.py` : An implementation of [cycle reservoirs with jumps (Rodan and Tiňo, 2012)][CRJ].
* `rmm2/deep_memory_machine.py` : A [pyTorch][torch] implementation of a
  deep learning variation of our reservoir memory machine.
* `rmm2/deep_memory_machine_test.py` : A unit test suite for
  `deep_memory_machine.py`.
* `rmm2/esn.py` : An implementation of [echo state networks (Jaeger and Haas, 2004)][ESN].
* `rmm2/fsm.py` : An implementation of [Moore machines (Moore, 1956)][FSM].
* `rmm2/lmu.py` : An implementation of [Legendre delay networks (Voelker, Kajić, and Eliasmith, 2019)][LMU].
* `rmm2/pseudo_svm.py` : A variation of a support vector machine via linear
  regression.
* `rmm2/rmm_assoc.py` : A reference implementation of the associative
  reservoir memory machine.
* `rmm2/rmm_assoc_test.py` : A unit test suite for `rmm_assoc.py`.
* `rmm2/rmm.py` : A reference implementation of the standard reservoir memory
  machine.
* `rmm2/rmm_test.py` : A unit test suite for `rmm_test.py`.
* `standard_rmm_experiments.ipynb` : A notebook for the latch, copy, repeat
  copy, and signal copy (aka smooth associative recall) experiments.

## Licensing

This library is licensed under the [GNU General Public License Version 3][GPLv3].

## Dependencies

This library depends on [NumPy][np] for matrix operations, on [scikit-learn][scikit]
for the base interfaces and on [SciPy][scipy] for optimization. The deep learning
experiments additionally depend on [pyTorch][torch].

## Literature

* Paaßen, Schulz, Stewart, and Hammer (2020). Reservoir Memory Machines as Neural Computers. IEEE Transactions on Neural Networks and Learning Systems. in print. doi:[10.1109/TNNLS.2021.3094139](https://doi.org/10.1109/TNNLS.2021.3094139). arXiv:[2009.06342][RMM]
* Jaeger and Haas (2004). Harnessing nonlinearity: Predicting chaotic systems and saving energy in wireless communication. Science, 304(5667), 78-80. doi:[10.1126/science.1091277][ESN]
* Moore (1956). Gedanken-experiments on sequential machines. Automata Studies (34), 129-153. [Link][FSM]
* Rodan and Tino (2012). Simple Deterministically Constructed Cycle Reservoirs with Regular Jumps. Neural Compuation, 24(7), 1822-1852. doi:[10.1162/NECO_a_00297][CRJ]
* Voelker, Kajić, and Eliasmith (2019). Legendre Memory Units: Continuous-Time Representation in Recurrent Neural Networks. In: Wallach, Larochelle, Beygelzimer, d'Alché-Buc, Fox, and Garnett. (eds.). Proceedings of the 32nd International Conference on Advances in Neural Information Processing Systems 32 (NeurIPS 2019), 15570--15579. [Link][LMU]

[scikit]: https://scikit-learn.org/stable/ "Scikit-learn homepage"
[np]: http://numpy.org/ "Numpy homepage"
[scipy]: https://scipy.org/ "SciPy homepage"
[torch]:https://pytorch.org/ "pyTorch homepage"
[GPLv3]: https://www.gnu.org/licenses/gpl-3.0.en.html "The GNU General Public License Version 3"
[CRJ]:https://doi.org/10.1162/NECO_a_00297 "Rodan and Tino (2012). Simple Deterministically Constructed Cycle Reservoirs with Regular Jumps. Neural Compuation, 24(7), 1822-1852. doi:10.1162/NECO_a_00297"
[ESN]:https://doi.org/10.1126/science.1091277 "Jaeger and Haas (2004). Harnessing nonlinearity: Predicting chaotic systems and saving energy in wireless communication. Science, 304(5667), 78-80. doi:10.1126/science.1091277"
[FSM]:http://www.jstor.org/stable/j.ctt1bgzb3s.8 "Moore (1956). Gedanken-experiments on sequential machines. Automata Studies (34), 129-153."
[LMU]:http://papers.nips.cc/paper/9689-legendre-memory-units-continuous-time-representation-in-recurrent-neural-networks "Voelker, Kajić, and Eliasmith (2019). Legendre Memory Units: Continuous-Time Representation in Recurrent Neural Networks. In: Wallach, Larochelle, Beygelzimer, d'Alché-Buc, Fox, and Garnett. (eds.). Proceedings of the 32nd International Conference on Advances in Neural Information Processing Systems 32 (NeurIPS 2019), 15570--15579."
[RMM]:https://arxiv.org/abs/2009.06342 "Paaßen, Schulz, Stewart, and Hammer (2020). Reservoir Memory Machines as Neural Computers. arXiv:2009.06342"
