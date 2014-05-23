Description
---

This is an algorithm that tunes a deep multiple kernel net by alternating optimization with the span bound. It is an attempt to extend deep learning to small sample sizes.

The algorithm is described in detail in Strobl EV, Visweswaran S. Deep Multiple Kernel Learning. ICMLA, 2013. http://arxiv.org/abs/1310.3101

Code
---

First, please install the MATLAB version of LIBSVM (http://www.csie.ntu.edu.tw/~cjlin/libsvm/). Then, download the entire package uploaded here (including the utility functions).

*Main Methods*

deepMKL_train.m - Trains the net. Each layer has an RBF, poly2, poly3, and linear kernel. How to adjust the learning rate: if the span bound is not on a decreasing trend, then your learning rate is probably too high. The default value works for most cases, but some adjustment may be needed.

deepMKL_test.m - Tests the net
