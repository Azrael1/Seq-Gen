__author__ = 'azrael'
# linear.lua

import theano
import theano.tensor as T
import numpy as np
assert_op = T.opt.Assert()

class LinearLayer(object):
    def __init__(self, inputDimension, outputDimension):
        self.rng = np.random.RandomState(1)
        self.W = theano.shared(np.asarray(self.rng.uniform(-0.1, 0.1, (inputDimension, outputDimension)),
                                          theano.config.floatX), borrow=True, name='W')
        self.b = theano.shared(np.asarray(self.rng.uniform(-0.1, 0.1, outputDimension),
                                          theano.config.floatX), borrow=True, name='b')
        self.params = [self.W, self.b]

    def forward(self, input):
        """
        The input can be a vector/matrix. A matrix means that the whole batch is given as input.
        Vector means that only 1 sequence is given as input.
        :param input: TensorVariable of either 1 dim/2
        :return: TensorVariable of either 1 dim/2
        """
        input_ = assert_op(input, ((input.ndim == 1 or input.ndim == 2) and input.dtype == theano.config.floatX))
        return T.dot(input, self.W) + self.b
