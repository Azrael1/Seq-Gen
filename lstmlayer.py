__author__ = 'azrael'
import numpy as np
import theano
import theano.tensor as T
from preprocess import *

rng = np.random.RandomState(123)


def sigmoid(x):
    return T.nnet.sigmoid(x)


def tanh(x):
    return T.tanh(x)


class LSTMlayer():
    def __init__(self, rng, n_in, n_hid):
        self.n_in = n_in
        self.n_hid = n_hid
        self.W_xi = theano.shared(value=rng.uniform(-0.01, 0.01, size=(n_in, n_hid)), dtype=theano.config.floatX,
                                  borrow=True)
        self.W_xf = theano.shared(value=rng.uniform(-0.01, 0.01, size=(n_in, n_hid)), dtype=theano.config.floatX,
                                  borrow=True)
        self.W_xc = theano.shared(value=rng.uniform(-0.01, 0.01, size=(n_in, n_hid)), dtype=theano.config.floatX,
                                  borrow=True)
        self.W_xo = theano.shared(value=rng.uniform(-0.01, 0.01, size=(n_in, n_hid)), dtype=theano.config.floatX,
                                  borrow=True)
        self.W_hi = theano.shared(value=rng.uniform(-0.01, 0.01, size=(n_hid, n_hid)), dtype=theano.config.floatX,
                                  borrow=True)
        self.W_hf = theano.shared(value=rng.uniform(-0.01, 0.01, size=(n_hid, n_hid)), dtype=theano.config.floatX,
                                  borrow=True)
        self.W_hc = theano.shared(value=rng.uniform(-0.01, 0.01, size=(n_hid, n_hid)), dtype=theano.config.floatX,
                                  borrow=True)
        self.W_ho = theano.shared(value=rng.uniform(-0.01, 0.01, size=(n_hid, n_hid)), dtype=theano.config.floatX,
                                  borrow=True)
        self.W_ci = theano.shared(value=rng.uniform(-0.01, 0.01, size=(n_hid, n_hid)), dtype=theano.config.floatX,
                                  borrow=True)
        self.W_cf = theano.shared(value=rng.uniform(-0.01, 0.01, size=(n_hid, n_hid)), dtype=theano.config.floatX,
                                  borrow=True)
        self.W_co = theano.shared(value=rng.uniform(-0.01, 0.01, size=(n_hid, n_hid)), dtype=theano.config.floatX,
                                  borrow=True)
        self.W_hhi = theano.shared(value=rng.uniform(-0.01, 0.01, size=n_hid), dtype=theano.config.floatX, borrow=True)
        self.W_hhf = theano.shared(value=rng.uniform(-0.01, 0.01, size=n_hid), dtype=theano.config.floatX, borrow=True)
        self.W_hhc = theano.shared(value=rng.uniform(-0.01, 0.01, size=n_hid), dtype=theano.config.floatX, borrow=True)
        self.W_hho = theano.shared(value=rng.uniform(-0.01, 0.01, size=n_hid), dtype=theano.config.floatX, borrow=True)
        self.bi = theano.shared(value=rng.uniform(-0.01, 0.01, size=n_hid), dtype=theano.config.floatX, borrow=True)
        self.bf = theano.shared(value=rng.uniform(-0.01, 0.01, size=n_hid), dtype=theano.config.floatX, borrow=True)
        self.bc = theano.shared(value=rng.uniform(-0.01, 0.01, size=n_hid), dtype=theano.config.floatX, borrow=True)
        self.bo = theano.shared(value=rng.uniform(-0.01, 0.01, size=n_hid), dtype=theano.config.floatX, borrow=True)
        self.c0 = theano.shared(value=rng.uniform(-0.01, 0.01, size=n_hid), dtype=theano.config.floatX, borrow=True)
        self.h0 = theano.shared(value=rng.uniform(-0.01, 0.01, size=n_hid), dtype=theano.config.floatX, borrow=True)

    def MinibatchRecurrence(self, data, minibatch_length):
        # Split the data into minibatch and pass each Minibatch to SequenceRecurrence function
        # For now assumption that that len(data) is an integral multiple of minibatch_length.
        h_m, updates = theano.scan(fn=self.SequenceReccurence,
                                   sequences=[data[minibatch_length * k:(minibatch_length + 1) * k] for k in
                                              range(int(len(data) / minibatch_length))],
                                   )

    def SequenceRecurrence(self, x):
        # Add non_sequences once no compilation and other errors.
        # For now assumption that that len(x) is an integral multiple of n_in.
        [h_s, c_s], updates = theano.scan(fn=self.OneStep,
                                          sequences=[x[self.n_in * k: k*(self.n_in+1)] for k in range(int(len(x)/self.n_in)],
                                          outputs_info=[self.h0, self.c0])
        # Call SGD on output matrix h given the input matrix x
        return h_s

    def OneStep(self, x_t, h_tm1, c_tm1):
        i_t = sigmoid(T.dot(x_t, self.W_xi) + T.dot(h_tm1, self.W_hi) + T.dot(c_tm1, self.W_ci) + self.op(c_tm1,
                                                                                                          self.W_hhi) + self.bi)
        f_t = sigmoid(T.dot(x_t, self.W_xf) + T.dot(h_tm1, self.W_hf) + T.dot(c_tm1, self.W_cf) + self.op(c_tm1,
                                                                                                          self.W_hhf) + self.bf)
        c_t = f_t * c_tm1 + i_t * tanh(
            T.dot(x_t, self.W_xc) + T.dot(h_tm1, self.W_hc) + self.op(c_tm1, self.W_hhc) + self.bc)
        o_t = sigmoid(T.dot(x_t, self.W_xo) + T.dot(h_tm1, self.W_ho) + T.dot(c_t, self.W_ci) + self.op(c_tm1,
                                                                                                        self.W_hho) + self.bo)
        h_t = o_t * tanh(c_t)
        return h_t, c_t

    def op(self, c, Whh):
        mul = c * Whh
        mul = T.set_subtensor(mul[1:(self.n_hid - 1)], mul[0:(self.n_hid - 2)])
        mul = T.set_subtensor(mul[0:1], 0.)
        return mul

# Sanity check
data = load_dataset("/home/azrael/Documents/nn/andrejrec/data/warpeace_input.txt")
datax = remove_letters(data)
mappings = one_hot_vec(datax)
#79 dimensional mapping created, len(data)=3258213
