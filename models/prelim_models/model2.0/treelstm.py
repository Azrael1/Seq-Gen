__author__ = 'azrael'
# This code deals with developing a tree lstm.
# POTENTIAL IDEA:
# Instead of keeping the weight matrix for a single connection as a scalar, make it a matrix with dimensions=vector dims
# While this significantly increases the number of dimensions, it provides much better control over the manipulation of
# the word vectors. This theory does not agree with the current techniques and biological connections but, theoretically
# it should give much better performance.

# NOTES FOR COMPILATION AND RUN TIME:

# Please refer to " A brief overview of deep learning" by Ilya Sutskever for specific information on how to train the
# model with right weights and hyperparamaters.
# The assert op works correctly with " no optimization" . Please take care while compiling graph.
# small letters refer to subscript(the hidden unit), big letters refer to superscript(the input unit).
# WARNING: The TreeLSTMLayer has been defined with only a branching factor of 2 in mind. Take care. i.e Works good
# for Binarized Constituency Trees.
# Why has the right child been added to the left forget gate and vice-versa? Change it after running once.
import theano
import theano.tensor as T
import numpy as np

vec_dim = 50
assert_op = T.opt.assert_op()
rng = np.random.RandomState(1)

def sigmoid(x):
    return T.nnet.sigmoid(x)

def tanh(x):
    return T.tanh(x)

class TreeLSTMLayer(object):
    """
    Notations consistent with paper "LSTM over tree structures."
    """
    def __init__(self, n_in, n_hid):
        # For Binarized trees
        assert n_hid == n_in/2.0
        # Input should be of shape (vec_dim, n_in)
        self.input = assert_op(input, input.shape[1] == len(n_in))
        self.n_in = n_in
        self.n_hid = n_hid

        self.W_hiL = theano.shared(value=np.asarray(rng.uniform(n_hid), dtype='float32'), borrow=True, name='W_hiL')
        self.W_hiR = theano.shared(value=np.asarray(rng.uniform(n_hid), dtype='float32'), borrow=True, name='W_hiR')
        self.W_ciL = theano.shared(value=np.asarray(rng.uniform(n_hid), dtype='float32'), borrow=True, name='W_ciL')
        self.W_ciR = theano.shared(value=np.asarray(rng.uniform(n_hid), dtype='float32'), borrow=True, name='W_ciR')
        self.W_hflL = theano.shared(value=np.asarray(rng.uniform(n_hid), dtype='float32'), borrow=True, name='W_hflL')
        self.W_hfrL = theano.shared(value=np.asarray(rng.uniform(n_hid), dtype='float32'), borrow=True, name='W_hfrL')
        self.W_cflL = theano.shared(value=np.asarray(rng.uniform(n_hid), dtype='float32'), borrow=True, name='W_cflL')
        self.W_cfrL = theano.shared(value=np.asarray(rng.uniform(n_hid), dtype='float32'), borrow=True, name='W_cfrL')
        self.W_hflR = theano.shared(value=np.asarray(rng.uniform(n_hid), dtype='float32'), borrow=True, name='W_hflR')
        self.W_hfrR = theano.shared(value=np.asarray(rng.uniform(n_hid), dtype='float32'), borrow=True, name='W_hfrR')
        self.W_cflR = theano.shared(value=np.asarray(rng.uniform(n_hid), dtype='float32'), borrow=True, name='W_cflR')
        self.W_cfrR = theano.shared(value=np.asarray(rng.uniform(n_hid), dtype='float32'), borrow=True, name='W_cflR')
        self.W_hxL = theano.shared(value=np.asarray(rng.uniform(n_hid), dtype='float32'), borrow=True, name='W_hxL')
        self.W_hxR = theano.shared(value=np.asarray(rng.uniform(n_hid), dtype='float32'), borrow=True, name='W_hxR')
        self.W_hoL = theano.shared(value=np.asarray(rng.uniform(n_hid), dtype='float32'), borrow=True, name='W_hoL')
        self.W_hoR = theano.shared(value=np.asarray(rng.uniform(n_hid), dtype='float32'), borrow=True, name='W_hoR')
        self.W_co = theano.shared(value=np.asarray(rng.uniform(n_hid), dtype='float32'), borrow=True, name='W_co')
        self.bi = theano.shared(value=np.asarray(rng.uniform(n_hid), dtype='float32'), borrow=True, name='W_bi')
        self.bfl = theano.shared(value=np.asarray(rng.uniform(n_hid), dtype='float32'), borrow=True, name='W_bfl')
        self.bfr = theano.shared(value=np.asarray(rng.uniform(n_hid), dtype='float32'), borrow=True, name='W_bfr')
        self.bx = theano.shared(value=np.asarray(rng.uniform(n_hid), dtype='float32'), borrow=True, name='W_bx')
        self.bo = theano.shared(value=np.asarray(rng.uniform(n_hid), dtype='float32'), borrow=True, name='W_bo')
        self.params = [self.W_hiL, self.W_hiR, self.W_ciL, self.W_ciR, self.W_hflL, self.W_hfrL, self.W_cflL,
                       self.W_cfrL, self.W_hflR, self.W_hfrR, self.W_cflR, self.W_cfrR, self.W_hxL, self.W_hxR,
                       self.W_hoL, self.W_hoR, self.W_co, self.bi, self.bfl, self.bfr, self.bx, self.bo]

    def fwd(self, h_tm1, c_tm1):
        """
        Computes a single forward pass of the layer.
        """
        # Reshape the input to 2 parts: namely left and right and do the computation.
        h_tm1L, h_tm1R = T.reshape(h_tm1.T, (self.n_in/2, 2*vec_dim))[:, :vec_dim].T, \
                         T.reshape(h_tm1.T, (self.n_in/2, 2*vec_dim))[:, vec_dim:2*vec_dim].T
        c_tm1L, c_tm1R = T.reshape(c_tm1.T, (self.n_in/2, 2*vec_dim))[:, :vec_dim].T, \
                         T.reshape(c_tm1.T, (self.n_in/2, 2*vec_dim))[:, vec_dim:2*vec_dim].T

        i_t = sigmoid(h_tm1L * self.W_hiL + h_tm1R * self.W_hiR + c_tm1L * self.W_ciL + c_tm1R * self.W_ciR + self.bi)
        f_tL = sigmoid(h_tm1L * self.W_hflL + h_tm1R * self.W_hflR + c_tm1L * self.W_cflL + c_tm1R * self.W_cflR + self.bfl)
        f_tR = sigmoid(h_tm1L * self.W_hfrL + h_tm1R * self.W_hfrR + c_tm1L * self.W_cfrL + c_tm1R * self.W_cfrR + self.bfr)
        x_t = h_tm1L * self.W_hxL + h_tm1R * self.W_hxR + self.bx
        c_t = f_tL * c_tm1L + f_tR * c_tm1R + i_t * tanh(x_t)
        o_t = sigmoid(h_tm1L * self.W_hoL + h_tm1R * self.W_hiR + c_t * self.W_co + self.bo)
        h_t = o_t * tanh(c_t)
        h_t = assert_op(h_t, h_t.shape == (vec_dim, self.n_hid))
        c_t = assert_op(c_t, c_t.shape == (vec_dim, self.n_hid))

        return h_t, c_t, self.n_hid

class TreeLSTM(object):
    """
    Build a tree structured LSTM with each layer being an instance of class TreeLSTMlayer.
    """
    # Builds a tree structured LSTM based on the sentence length. Does not have paragraph support for now.
    # Does not have parse structured tree support. Works on 2**x (x is integer) length of sentences.
    def __init__(self, sen_vec):
        # sen_vec is a shared variable of shape (vec_dim, sentence_length)
        self.sen_vec = assert_op(sen_vec, sen_vec.shape[0] == 50)
        n_layers = T.log2(sen_vec.shape[1])
        self.n_in = sen_vec.shape[1]
        self.num_layers = assert_op(n_layers, n_layers.get_value() - int(n_layers.get_value()) == 0)
        self.params = []

    def tree_output(self):
        """
        Compute the output layerwise and feed it to the next layer.
        """
        n_t0 = self.n_in
        h_t0 = self.sen_vec
        c_t0 = T.zeros_like(h_t0)
        [h_t, c_t], _ = theano.scan(fn=self.layer_output, outputs_info=[n_t0, h_t0, c_t0],
                                    n_steps=self.num_layers)
        return h_t[-1]

    def layer_output(self, nin_tm1, h_tm1, c_tm1):
        n_hid = T.cast(nin_tm1/2.0, 'float32')
        layer = TreeLSTMLayer(nin_tm1, n_hid)
        self.params.append(layer.params)
        # The input is padded with zeros to keep a consistent shape over the scan recurrence.
        # So take out the part that is needed. Do your magic, again pad with zeros and return it.
        h_t, c_t, n_inx = layer.fwd(h_tm1[:, :nin_tm1], c_tm1[:, nin_tm1])
        h_tx = T.set_subtensor(T.zeros(vec_dim, self.n_in), h_t)
        c_tx = T.set_subtensor(T.zeros(vec_dim, self.n_in), c_t)
        return h_tx, c_tx, n_inx, theano.scan_module.until((h_t.shape[1] == 1) and (c_t.shape[1] == 1))

