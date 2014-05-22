Description
---

This is an algorithm that optimizes a deep multiple kernel net by alternating optimization with the span bound. It is an attempt to extend deep learning to small sample sizes.

The algorithm is described in detail in Strobl EV, Visweswaran S. Deep Multiple Kernel Learning. ICMLA, 2013. http://arxiv.org/abs/1310.3101

I just uploaded this, so please let me know if you find any bugs or things that dont work.

Code
---

Requires the MATLAB version of LIBSVM (http://www.csie.ntu.edu.tw/~cjlin/libsvm/). Then, please download the entire package (including the utility functions)

*Main Methods*

deepMKL_train.m.m - Trains the net. Each layer has an RBF, poly2, poly3, and linear kernel. 

deepMKL_test.m.m - Tests the net
