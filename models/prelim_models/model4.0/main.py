__author__ = 'azrael'
"""
The code is quite complex. If you need help with understanding/implementing it, mail bhavishya.pohani@gmail.com.
There are 2 files needed for running this experiment.
1) The glove vectors file.
2) The text file on which training is done.
Github does not allow me to attach text files of such size. Mail me at the above email-id to get the data. 
"""
import theano.tensor as T
from preprocess import *
import matplotlib.pyplot as pyplot
from pylab import *

np.set_printoptions(threshold=500)
assert_op = T.opt.Assert()
n_in = 10
n_tree = 4
vec_file = 'path-to-vector-file'
vec_dims = 50
n_nodes = 100
rng = np.random.RandomState(123)
doc_path = 'path-to-text-file-on-which-training-occurs'
mappings_words, mappings_vec, input_text = load_vecs(vec_file, doc_path, vec_dims)
np_vecs = np.asarray(mappings_vec.get_value(), theano.config.floatX)

assert_check = False

def relu(x):
    alpha = 0
    return T.switch(x>0, x, alpha * x)

def activation(x):
    return T.nnet.sigmoid(x)

def sigmoid(x):
    return T.nnet.sigmoid(x)

def tanh(x):
    return T.tanh(x)

class TreeLSTMLayer(object):
    """
    This is a single layer of the Tree LSTM structure. This structure has been taken from the paper-
    " Long short term memory over tree structures" by X. Zhu, P. Sobhani, H. Guo. In particular, this is the S-LSTM
    from the paper. If you are looking at my project report, these layers are the circles in Figure 2.
    """
    def __init__(self, n_in, n_out, low, high, init, random_init):
        self.n = n_out

        weights_shape = (vec_dims, self.n/2)
        biases_shape = self.n/2
        # Required later for broadcasting purposes.
        if init and random_init=='gaussian':
            temp = 4*(6/float((n_nodes + n_in)**0.5))
            print 'The value of temp is', temp
            low = 0.0
            high = temp
            initmethod = rng.normal

        if init and random_init=='uniform':
            temp = 4*(6/float((n_nodes + n_in)**0.5))
            print 'The value of temp is', temp
            low = -temp
            high = temp
            initmethod = rng.uniform


        self.W_hiL = theano.shared(value=np.asarray(initmethod(low, high, weights_shape), dtype='float32'), borrow=True, name='W_hiL')
        self.W_hiR = theano.shared(value=np.asarray(initmethod(low, high, weights_shape), dtype='float32'), borrow=True, name='W_hiR')
        self.W_ciL = theano.shared(value=np.asarray(initmethod(low, high, weights_shape), dtype='float32'), borrow=True, name='W_ciL')
        self.W_ciR = theano.shared(value=np.asarray(initmethod(low, high, weights_shape), dtype='float32'), borrow=True, name='W_ciR')
        self.W_hflL = theano.shared(value=np.asarray(initmethod(low, high, weights_shape), dtype='float32'), borrow=True, name='W_hflL')
        self.W_hfrL = theano.shared(value=np.asarray(initmethod(low, high, weights_shape), dtype='float32'), borrow=True, name='W_hfrL')
        self.W_cflL = theano.shared(value=np.asarray(initmethod(low, high, weights_shape), dtype='float32'), borrow=True, name='W_cflL')
        self.W_cfrL = theano.shared(value=np.asarray(initmethod(low, high, weights_shape), dtype='float32'), borrow=True, name='W_cfrL')
        self.W_hflR = theano.shared(value=np.asarray(initmethod(low, high, weights_shape), dtype='float32'), borrow=True, name='W_hflR')
        self.W_hfrR = theano.shared(value=np.asarray(initmethod(low, high, weights_shape), dtype='float32'), borrow=True, name='W_hfrR')
        self.W_cflR = theano.shared(value=np.asarray(initmethod(low, high, weights_shape), dtype='float32'), borrow=True, name='W_cflR')
        self.W_cfrR = theano.shared(value=np.asarray(initmethod(low, high, weights_shape), dtype='float32'), borrow=True, name='W_cflR')
        self.W_hxL = theano.shared(value=np.asarray(initmethod(low, high, weights_shape), dtype='float32'), borrow=True, name='W_hxL')
        self.W_hxR = theano.shared(value=np.asarray(initmethod(low, high, weights_shape), dtype='float32'), borrow=True, name='W_hxR')
        self.W_hoL = theano.shared(value=np.asarray(initmethod(low, high, weights_shape), dtype='float32'), borrow=True, name='W_hoL')
        self.W_hoR = theano.shared(value=np.asarray(initmethod(low, high, weights_shape), dtype='float32'), borrow=True, name='W_hoR')
        self.W_co = theano.shared(value=np.asarray(initmethod(low, high, weights_shape), dtype='float32'), borrow=True, name='W_co')
        self.bi = theano.shared(value=np.asarray(np.zeros(biases_shape), dtype='float32'), borrow=True, name='bi')
        self.bfl = theano.shared(value=np.asarray(np.zeros(biases_shape), dtype='float32'), borrow=True, name='bfl')
        self.bfr = theano.shared(value=np.asarray(np.zeros(biases_shape), dtype='float32'), borrow=True, name='bfr')
        self.bx = theano.shared(value=np.asarray(np.zeros(biases_shape), dtype='float32'), borrow=True, name='bx')
        self.bo = theano.shared(value=np.asarray(np.zeros(biases_shape), dtype='float32'), borrow=True, name='bo')
        self.params = [self.W_hiL, self.W_hiR, self.W_ciL, self.W_ciR, self.W_hflL, self.W_hfrL, self.W_cflL,
                       self.W_cfrL, self.W_hflR, self.W_hfrR, self.W_cflR, self.W_cfrR, self.W_hxL, self.W_hxR,
                       self.W_hoL, self.W_hoR, self.W_co, self.bi, self.bfl, self.bfr, self.bx, self.bo]

    def fwd(self, h_tm1, c_tm1):
        if assert_check:
            h_tm1 = assert_op(h_tm1, T.eq(h_tm1.shape[0], vec_dims), T.eq(h_tm1.shape[1], self.n))
        h_tm1L, h_tm1R = T.reshape(h_tm1.T, (self.n/2, 2*vec_dims), name='h_tm1L')[:, :vec_dims].T, \
                         T.reshape(h_tm1.T, (self.n/2, 2*vec_dims), name='h_tm1R')[:, vec_dims:2*vec_dims].T

        c_tm1L, c_tm1R = T.reshape(c_tm1.T, (self.n/2, 2*vec_dims), name='c_tm1L')[:, :vec_dims].T, \
                         T.reshape(c_tm1.T, (self.n/2, 2*vec_dims), name='c_tm1R')[:, vec_dims:2*vec_dims].T

        if assert_check:
            h_tm1L = assert_op(h_tm1L, T.eq(h_tm1L.shape[0], vec_dims), T.eq(h_tm1L.shape[1], self.n/2))
            c_tm1L = assert_op(c_tm1L, T.eq(c_tm1L.shape[0], vec_dims), T.eq(c_tm1L.shape[1], self.n/2))

        i_t = sigmoid(h_tm1L * self.W_hiL + h_tm1R * self.W_hiR + c_tm1L * self.W_ciL + c_tm1R * self.W_ciR + self.bi)
        f_tl = sigmoid(h_tm1L * self.W_hflL + h_tm1R * self.W_hflR + c_tm1L * self.W_cflL + c_tm1R * self.W_cflR + self.bfl)
        f_tr = sigmoid(h_tm1L * self.W_hfrL + h_tm1R * self.W_hfrR + c_tm1L * self.W_cfrL + c_tm1R * self.W_cfrR + self.bfr)
        x_t = h_tm1L * self.W_hxL + h_tm1R * self.W_hxR + self.bx
        c_t = f_tl * c_tm1L + f_tr * c_tm1R + i_t * tanh(x_t)
        if assert_check:
            c_t = assert_op(c_t, T.eq(c_t.shape[0], vec_dims), T.eq(c_t.shape[1], self.n/2))
        o_t = sigmoid(h_tm1L * self.W_hoL + h_tm1R * self.W_hoR + c_t * self.W_co + self.bo)
        h_t = o_t * tanh(c_t)
        if assert_check:
            h_t = assert_op(h_t, T.eq(h_t.shape[0], vec_dims), T.eq(h_t.shape[1], self.n/2))
            c_t = assert_op(c_t, T.eq(c_t.shape[0], vec_dims), T.eq(c_t.shape[1], self.n/2))
        return h_t, c_t

class TreeLSTM(object):
    """
    This class combines layers of TreeLSTM. A binary tree is being currently used. So it works on 2**x (x is integer)
    length of sentences.
    """
    def __init__(self, n_tree, low, high, init=True, random_init='gaussian'):
        assert n_tree/int(n_tree) == 1
        self.n_in = n_tree
        self.params = []
        # Instead of adding layer by layer can use scan but it adds large overhead.
        self.layer1 = TreeLSTMLayer(1, self.n_in, low, high, init, random_init)
        self.layer2 = TreeLSTMLayer(self.n_in, self.n_in/2, low, high, init, random_init)
        self.params += self.layer1.params + self.layer2.params

    def tree_op(self, sen_vec):
        if assert_check:
            sen_vec = assert_op(sen_vec, T.eq(sen_vec.shape[0], vec_dims), T.eq(sen_vec.shape[1], self.n_in))
        h_t0 = sen_vec
        c_t0 = T.zeros_like(h_t0)
        layer1_h_t, layer1_c_t = self.layer1.fwd(h_t0, c_t0)
        layer2_h_t, layer2_c_t = self.layer2.fwd(layer1_h_t, layer1_c_t)

        h_t_ = layer2_h_t
        if assert_check:
            h_t_ = assert_op(h_t_, T.eq(h_t_.shape[0], vec_dims), T.eq(h_t_.shape[1], 1))
        return h_t_  # Extra dimension not removed. It is used later when the tree structure is combined with the LSTMStackedLayers.


class LSTMLayer(object):
    """
    This is a simple LSTM layer that unfolds in time. If you are looking at my project report, then this is the square
    box with LSTM written inside it in Figure 2.
    """
    def __init__(self, n_in, n_nodes, local, high, init=True, random_init='gaussian'):
        weights_shape = (n_in, n_nodes)
        wts_shape = (n_nodes, n_nodes)

        biases_shape = n_nodes
    #     Required later for broadcasting purposes.
        if init and random_init=='gaussian':
            temp = 4*(6/float((n_nodes + n_in)**0.5))
            print 'The value of temp is', temp
            local = 0.0
            high = temp
            initmethod = rng.normal

        if init and random_init=='uniform':
            temp = 4*(6/float((n_nodes + n_in)**0.5))
            print 'The value of temp is', temp
            local = -temp
            high = temp
            initmethod = rng.uniform

        self.W_xi = theano.shared(value=np.asarray(initmethod(local, high, weights_shape), dtype=theano.config.floatX), borrow=True, name='W_xi')
        self.W_xf = theano.shared(value=np.asarray(initmethod(local, high, weights_shape), dtype=theano.config.floatX), borrow=True, name='W_xf')
        self.W_xc = theano.shared(value=np.asarray(initmethod(local, high, weights_shape), dtype=theano.config.floatX), borrow=True, name='W_xc')
        self.W_xo = theano.shared(value=np.asarray(initmethod(local, high, weights_shape), dtype=theano.config.floatX), borrow=True, name='W_xo')
        self.W_hi = theano.shared(value=np.asarray(initmethod(local, high, wts_shape), dtype=theano.config.floatX), borrow=True, name='W_hi')
        self.W_hf = theano.shared(value=np.asarray(initmethod(local, high, wts_shape), dtype=theano.config.floatX), borrow=True, name='W_hf')
        self.W_hc = theano.shared(value=np.asarray(initmethod(local, high, wts_shape), dtype=theano.config.floatX), borrow=True, name='W_hc')
        self.W_ho = theano.shared(value=np.asarray(initmethod(local, high, wts_shape), dtype=theano.config.floatX), borrow=True, name='W_ho')
        self.W_ci = theano.shared(value=np.asarray(initmethod(local, high, wts_shape), dtype=theano.config.floatX), borrow=True, name='W_ci')
        self.W_cf = theano.shared(value=np.asarray(initmethod(local, high, wts_shape), dtype=theano.config.floatX), borrow=True, name='W_cf')
        self.W_co = theano.shared(value=np.asarray(initmethod(local, high, wts_shape), dtype=theano.config.floatX), borrow=True, name='W_co')
        self.bi = theano.shared(value=np.asarray(initmethod(local, high, biases_shape), dtype=theano.config.floatX),borrow=True, name='bi')
        self.bf = theano.shared(value=np.asarray(initmethod(local, high, biases_shape), dtype=theano.config.floatX), borrow=True, name='bf')
        self.bc = theano.shared(value=np.asarray(initmethod(local, high, biases_shape), dtype=theano.config.floatX), borrow=True, name='bc')
        self.bo = theano.shared(value=np.asarray(initmethod(local, high, biases_shape), dtype=theano.config.floatX), borrow=True, name='bo')
        self.h_t0 = theano.shared(value=np.zeros((vec_dims, n_nodes), theano.config.floatX), borrow=True, name='h_t0')
        self.c_t0 = theano.shared(value=np.zeros((vec_dims, n_nodes), theano.config.floatX), borrow=True, name='c_t0')


        self.params = [self.W_xi, self.W_xf, self.W_xc, self.W_xo, self.W_hi, self.W_hf, self.W_hc,
                       self.W_ho, self.W_ci, self.W_cf, self.W_co, self.bi, self.bf, self.bc, self.bo]


class LSTMStackedLayers(object):
    def __init__(self, n_nodes, low, high, init=True, random_init='gaussian'):
        if init and random_init=='gaussian':
            temp = 4*(6/float((n_nodes + n_in)**0.5))
            print 'The value of temp is', temp
            local = 0.0
            high = temp
            initmethod = rng.normal

        if init and random_init=='uniform':
            temp = 4*(6/float((n_nodes + n_in)**0.5))
            print 'The value of temp is', temp
            local = -temp
            high = temp
            initmethod = rng.uniform

        self.W = theano.shared(value=np.asarray(initmethod(local, high, n_nodes), theano.config.floatX), borrow=True, name='W')
        self.b = theano.shared(value=np.asarray(initmethod(local, high, 1), theano.config.floatX), borrow=True, name='b', broadcastable=(True,))

        self.l1 = LSTMLayer(1, n_nodes, low, high, init, random_init)
        self.l2 = LSTMLayer(n_nodes, n_nodes, low, high, init, random_init)

        self.params = self.l1.params + self.l2.params + [self.W, self.b]

    def op(self, x_t):
        # The reason that there are different functions for normal output and for validation output is that the structure
        # of scan changes in both. See that there are 1 sequences and 4 outputs_info in function op and that there are
        # 0 sequences and 5 outputs_info in function valid_op
        [h_t1, c_t1, h_t2, c_t2], upd = theano.scan(fn=self.oneStep, sequences=x_t, outputs_info=[self.l1.h_t0, self.l1.c_t0,
                                                                                                  self.l2.h_t0, self.l2.c_t0,
                                                                                                  ])
        op = T.dot(h_t2, self.W) + self.b
        assert op.ndim == 2
        if assert_check:
            op = assert_op(op, T.eq(op.shape[1], vec_dims))
        return op, upd

    def oneStep(self, x_t, l1h_tm1, l1c_tm1, l2h_tm1, l2c_tm1):
        assert l1h_tm1.ndim == 2
        assert l1c_tm1.ndim == 2
        assert l2h_tm1.ndim == 2
        assert l2c_tm1.ndim == 2

        x_t = x_t.dimshuffle(0,'x')
        assert x_t.ndim == 2

        if assert_check:
            x_t = assert_op(x_t, T.eq(x_t.shape[0], vec_dims), T.eq(x_t.shape[1], 1))

        # The reason that x_t.shape = (vec_dims, 1) is because this is later used in the dot product with the weights.
        i_t1 = sigmoid(T.dot(x_t, self.l1.W_xi) + T.dot(l1h_tm1, self.l1.W_hi) + T.dot(l1c_tm1, self.l1.W_ci) + self.l1.bi)
        f_t1 = sigmoid(T.dot(x_t, self.l1.W_xf) + T.dot(l1h_tm1, self.l1.W_hf) + T.dot(l1c_tm1, self.l1.W_cf) + self.l1.bf)
        temp = tanh(T.dot(x_t, self.l1.W_xc) + T.dot(l1h_tm1, self.l1.W_hc) + self.l1.bc)
        c_t1 = f_t1 * l1c_tm1 + i_t1 * temp
        o_t1 = sigmoid(T.dot(x_t, self.l1.W_xo) + T.dot(l1h_tm1, self.l1.W_ho) + T.dot(c_t1, self.l1.W_co) + self.l1.bo)
        h_t1 = o_t1 * tanh(c_t1)

        x_t2 = h_t1

        i_t2 = sigmoid(T.dot(x_t2, self.l2.W_xi) + T.dot(l2h_tm1, self.l2.W_hi) + T.dot(l2c_tm1, self.l2.W_ci) + self.l2.bi)
        f_t2 = sigmoid(T.dot(x_t2, self.l2.W_xf) + T.dot(l2h_tm1, self.l2.W_hf) + T.dot(l2c_tm1, self.l2.W_cf) + self.l2.bf)
        temp2 = tanh(T.dot(x_t2, self.l2.W_xc) + T.dot(l2h_tm1, self.l2.W_hc) + self.l2.bc)
        c_t2 = f_t2 * l2c_tm1 + i_t2 * temp2
        o_t2 = sigmoid(T.dot(x_t2, self.l2.W_xo) + T.dot(l2h_tm1, self.l2.W_ho) + T.dot(c_t2, self.l2.W_co) + self.l2.bo)
        h_t2 = o_t2 * tanh(c_t2)

        return h_t1, c_t1, h_t2, c_t2

    def valid_op(self, x_t0, n_steps):
        if assert_check:
            assert x_t0.ndim == 1
            x_t0 = assert_op(x_t0, T.eq(x_t0.shape[0], vec_dims))
        [x_t, h_t2, h_t1, c_t2, c_t1], upd = theano.scan(fn=self.valid_oneStep,
                                                         outputs_info=[x_t0, self.l2.h_t0, self.l1.h_t0, self.l2.c_t0, self.l1.c_t0],
                                                         n_steps=n_steps)
        op = x_t
        if assert_check:
            assert op.ndim == 2
            op = assert_op(op, T.eq(op.shape[0], n_steps), T.eq(op.shape[1], vec_dims))
        return op, upd


    def valid_oneStep(self, x_tm1, l2h_tm1, l1h_tm1, l2c_tm1, l1c_tm1):
        if assert_check:
            assert x_tm1.ndim == 1
            x_tm1 = T.eq(x_tm1, T.eq(x_tm1.shape[0], vec_dims))
        x_tm1 = x_tm1.dimshuffle(0,'x')
        i_t1 = sigmoid(T.dot(x_tm1, self.l1.W_xi) + T.dot(l1h_tm1, self.l1.W_hi) + T.dot(l1c_tm1, self.l1.W_ci) + self.l1.bi)
        f_t1 = sigmoid(T.dot(x_tm1, self.l1.W_xf) + T.dot(l1h_tm1, self.l1.W_hf) + T.dot(l1c_tm1, self.l1.W_cf) + self.l1.bf)
        c_t1 = f_t1 * l1c_tm1 + i_t1 * tanh(T.dot(x_tm1, self.l1.W_xc) + T.dot(l1h_tm1, self.l1.W_hc) + self.l1.bc)
        o_t1 = sigmoid(T.dot(x_tm1, self.l1.W_xo) + T.dot(l1h_tm1, self.l1.W_ho) + T.dot(c_t1, self.l1.W_co) + self.l1.bo)
        h_t1 = o_t1 * tanh(c_t1)

        x_t2 = h_t1

        i_t2 = sigmoid(T.dot(x_t2, self.l2.W_xi) + T.dot(l2h_tm1, self.l2.W_hi) + T.dot(l2c_tm1, self.l2.W_ci) + self.l2.bi)
        f_t2 = sigmoid(T.dot(x_t2, self.l2.W_xf) + T.dot(l2h_tm1, self.l2.W_hf) + T.dot(l2c_tm1, self.l2.W_cf) + self.l2.bf)
        c_t2 = f_t2 * l2c_tm1 + i_t2 * tanh(T.dot(x_t2, self.l2.W_xc) + T.dot(l2h_tm1, self.l2.W_hc) + self.l2.bc)
        o_t2 = sigmoid(T.dot(x_t2, self.l2.W_xo) + T.dot(l2h_tm1, self.l2.W_ho) + T.dot(c_t2, self.l2.W_co) + self.l2.bo)
        h_t2 = o_t2 * tanh(c_t2)
        x_t = T.dot(h_t2, self.W) + self.b
        if assert_check:
            assert x_t.ndim == 1
            x_t = assert_op(x_t, T.eq(x_t.shape[0], vec_dims))
        return x_t, h_t2, h_t1, c_t2, c_t1



class Model(object):
    """
    This class combines the Tree structured LSTMs(class TreeLSTM) and the LSTM(class LSTMStackedLayers) which expands in time.
    """
    def __init__(self, n_tree, n_nodes, low, high, init, random_init='gaussian'):
        assert (n_tree % 2 == 0), "Number of input nodes of LSTM tree has to be even"
        self.n_tree = n_tree
        self.n_nodes = n_nodes
        self.rnn_stacked = LSTMStackedLayers(n_nodes, low, high, init, random_init)
        self.tree = TreeLSTM(n_tree, low, high, init, random_init)

        self.params = self.tree.params + self.rnn_stacked.params


    def op(self, vecs):
        # vec_words should have shape- (Vector dimensions, Number of words)
        if assert_check:
            vecs = assert_op(vecs, T.eq(vecs.shape[0], vec_dims), T.eq(vecs.shape[1], n_in))
        vecs_tree = vecs[:, :n_tree]
        tree_op = self.tree.tree_op(vecs_tree)
        vecs_rem = vecs[:, n_tree:]
        x_t = T.concatenate([tree_op, vecs_rem], axis=1).T
        x_t = assert_op(x_t,  T.eq(x_t.shape[1], vec_dims), T.eq(x_t.shape[0], n_in-n_tree+1))

        op, upd = self.rnn_stacked.op(x_t)
        return op, upd

    def valid_op(self, x_t, n_steps):
        assert n_steps > 0
        assert x_t.ndim == 2
        if assert_check:
            x_t = assert_op(x_t, T.eq(x_t.shape[0], vec_dims), T.eq(x_t.shape[1], n_tree))
        tree_op = self.tree.tree_op(x_t)
        tree_op = tree_op.T[0]
        if assert_check:
            tree_op = assert_op(tree_op, T.eq(tree_op.shape[0], vec_dims))
        op, upd = self.rnn_stacked.valid_op(tree_op, n_steps)
        return op, upd


    def loss(self, y, output, loss='squared'):
        # assumes using squared loss, if not then uses cosine loss
        assert output.ndim == 2
        y_ = y.T
        if assert_check:
            # y.shape = output.shape = (n_timesteps, vec_dims)
            y_ = assert_op(y_, T.eq(y_.shape[1], vec_dims))
            y_ = assert_op(y_, T.eq(y_.shape[0], output.shape[0]))
            y_ = assert_op(y_, T.eq(y_.shape[1], output.shape[1]))
        if loss == 'squared':
            loss = T.mean(T.sum((output - y_)**2, axis=1), axis=0)
        elif loss == 'cosine':
            temp = T.sum(output * y_, axis=1)/(T.sqrt(T.sum(T.sqr(output), axis=1)) * T.sqrt(T.sum(T.sqr(y_), axis=1)) + 1e-8)
            loss = T.sum((1 - temp)/2)
        else:
            raise AssertionError('The loss type used is not compatible')

        assert loss.ndim == 0
        return loss


def form_trainingdata(n_in, text):
    words = text.split()
    n_sen = len(words)/int(n_in)
    print 'max_minibatch is', n_sen
    train_x = np.asarray(words)[:n_sen*n_in].reshape((n_sen, n_in))
    del words
    train_x_vec = np.asarray([sentence_vec(train_x[i], np_vecs, mappings_words) for i in range(len(train_x))], dtype=theano.config.floatX)
    shared_x = theano.shared(np.concatenate(train_x_vec, axis=1), name='shared_x', borrow=True)
    return shared_x, n_sen

train_x, max_minibatch = form_trainingdata(n_in, input_text)

def prediction(op):
    temp = np.argmin(np.sum((np_vecs-op)**2, axis=1))
    return temp

def gradient_updates_momentum(cost, params, learning_rate, momentum):
    assert 1 > momentum >= 0
    updates = []
    for param in params:
        param_update = theano.shared(param.get_value()*0., broadcastable=param.broadcastable)
        updates.append((param, param - learning_rate*param_update))
        updates.append((param_update, momentum * param_update + (1-momentum) * T.grad(cost, param)))
    return updates

def SGD(eta, n_epochs, valid_steps, momentum, low, high, init, random_init='gaussian'):
    t0 = time.time()
    index = T.iscalar('index')
    x, y, z, alpha = T.fmatrices('x', 'y', 'z', 'alpha')
    n_minibatch = max_minibatch - 2
    model = Model(n_tree, n_nodes, low, high, init, random_init)
    model_op, auto_upd = model.op(x)
    valid_op, valid_upd = model.valid_op(z, valid_steps)

    loss = model.loss(y, model_op)
    valid_loss = model.loss(alpha, valid_op)

    print "Updation to be compiled yet"

    params = model.params
    train_upd = gradient_updates_momentum(loss, params, eta, momentum) + auto_upd
    train_output = [model_op, loss]
    valid_output = [valid_op, valid_loss]

    print "Train function to be compiled"
    train_fn = theano.function([index], train_output, updates=train_upd,
                               givens={x: train_x[:, n_in * index: n_in * (index + 1)],
                                       y: train_x[:, (n_in * index + n_tree): (n_in * (index + 1) + 1)]}, name='train_fn')

    valid_fn = theano.function([index], valid_output, updates=valid_upd,
                               givens={z: train_x[:, n_tree * index: n_tree * (index + 1)],
                                       alpha: train_x[:, (n_in * index + n_tree): (n_in * index + n_tree + valid_steps)]}, name='valid_fn')

    print "Train function compiled"


    # Compilation over
    #################
    ## TRAIN MODEL ##
    #################
    print 'The compilation time is', time.time() - t0
    loss_list = []
    for i in range(n_epochs):
        epoch_loss = 0

        t1 = time.time()
        for idx in range(n_minibatch):
            print 'The current idx is ', idx,' and the epoch number is  ', i
            output, loss_ = train_fn(idx)[:-1], train_fn(idx)[-1]
            if idx%500 == 0:
                v_output, v_loss = valid_fn(idx/500)[:-1][0], valid_fn(idx/500)[-1]
                print 'v_pred is', ' '.join([mappings_words[prediction(abc)] for abc in v_output])
                print 'v_loss is', np.array(v_loss)
            print 'The loss is', loss_
            epoch_loss +=loss_
            loss_list.append(loss_)

            print '=='*20
        print 'The mean loss for the epoch was', epoch_loss/float(n_minibatch)
        print 'Time taken by this epoch is', time.time()-t1
        print '-'*50
    pyplot.plot(loss_list)
    pyplot.show()


if __name__ == "__main__":
    low, high = -1, 1
    SGD(10**-5, 20, 10, 0.8, low, high, True, 'uniform')

