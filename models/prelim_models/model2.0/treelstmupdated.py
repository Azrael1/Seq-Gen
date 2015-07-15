__author__ = 'azrael'

"""
POTENTIAL IDEAS:

1) Instead of keeping the weight matrix for a single connection as a scalar, make it a matrix with dimensions=vector dims
 While this significantly increases the number of dimensions, it provides much better control over the manipulation of
 the word vectors. This theory does not agree with the current techniques and biological connections but, theoretically
 it should give much better performance.
2) The model as of the moment takes n_in+1 words and feeds n_in words in the net and the last word as the output. So that
 single word is never fed into input. This should have a negative effect/ break the sequence. Test it and report results.
"""

"""
GC:

Monitor memory usage and decide if manual garbage collection is required. Avoid dictionary usage. Consumes 25 times
more memory than lists.
"""

"""
 NOTES FOR COMPILATION AND RUN TIME:

 Please refer to " A brief overview of deep learning" by Ilya Sutskever for specific information on how to train the
 model with right weights and hyperparamaters.
 This link provides a nice discussion on the problems of training RNNs - https://groups.google.com/forum/?utm_source=di
 gest&utm_medium=email/#!searchin/theano-users/scan$20function$20running$20only$20for$201$20step/theano-users/16AwnXu
 fKMY/3fjieA3mfIEJ
 The assert op works correctly with " no optimization" . Please take care while compiling graph.
 small letters refer to subscript(the hidden unit), big letters refer to superscript(the input unit).
 WARNING: The TreeLSTMLayer has been defined with only a branching factor of 2 in mind. Take care. i.e Works good
 for Binarized Constituency Trees/normal binary trees with fixed dimensionality.
 If gives something like <generic> type error, search for python arrays(not numpy's) and String values coded as
 theano.shared. Theano.shared values usually give this error.
 Why has the right child been added to the left forget gate and vice-versa? Change it after running once.
 The update rule for now(clipping the gradient in the final layer then passing it on to the layers) will not prevent
 exploding gradients. Clipping the gradients layer wise should do that,
 as done by Graham Taylor in his code.(https://github.com/gwtaylor/theano-rnn) Check once.
"""

"""
 CHANGES TO BE MADE IN CODE:

 1) Add minibatch support.
 2) Change error function to take into account angle and distance of the vectors.( Also use weights in this. Let the
 NN figure out what is best for it. Start from equal weights/see what Kai sheng tai has done.)
 3) Add paragraph support.
 4) Add regularization - relu, dropout, L1, L2, more stuff
 5) Active visualization for error rate, current preds.
 6) Increase dimensions of weights to vec_dims.(see POT IDEAS section)
 7) Instead of giving 2 words to a single neuron, give 1 word to single neuron. Can be done by adding 1 more layer at
 bottom. The dimensionality of the sentence will have to increase to work properly for this NN. Can be done by(example):
 suppose n_in = 4 words
 so make the number of layers = log2(4)+1 = 3 The bottom most layer will have inputs: [w1 0 w2 0 w3 0 w4 0]
"""
import theano.tensor as T
from preprocess1 import *

n_in = 4
vec_file = '/home/azrael/Documents/nn/code/vectors.6B.50d.txt'
vec_dims = 50
assert_op = T.opt.Assert()

rng = np.random.RandomState(1)
doc_path = '/home/azrael/Documents/nn/code/data_clean/prelim1/wikisample1.txt'
word_vecs, mappings_words, mappings_vec = load_vecs(vec_file, doc_path, vec_dims)

def sigmoid(x):
    return T.nnet.sigmoid(x)

class AssertShape(T.opt.Assert):
    def make_node(self, value, *shapes):
        assert len(shapes) == 1
        assert isinstance(shapes[0], (tuple, list))
        params = [value]
        for actual_size, expected_size in zip(value.shape, shapes[0]):
            params.append(T.eq(actual_size, expected_size))
        return super(AssertShape, self).make_node(*params)

assert_shape_op = AssertShape()

def tanh(x):
    return T.tanh(x)

def abs(x, axis=0):
    """
    Takes the matrix/vector x and finds the absolute along the axis.
    :param x: T.matrix
    :return: Absolute along the given axis. T.vector
    """
    x = assert_op(x, T.or_(T.eq(x.ndim, 2), T.eq(x.ndim, 1)))
    return T.sqrt(T.sum(T.sqr(x), axis))

def weight_gen(n_in):
    cmplte_weights = np.asarray(rng.uniform(-0.05, 0.05, ((2*n_in-1))), 'float32')
    seq_list = []
    r_sum = 0
    for i in range(int(np.log2(n_in))):
        y = cmplte_weights[r_sum : (r_sum + n_in/2**(i+1))]
        seq_list.append(theano.shared(y))
        r_sum += (n_in/2**(i+1))
    return seq_list

class TreeLSTMLayer(object):
    def __init__(self, n_in):
        self.n_in = n_in
        self.n_hid = n_in/2.0

        self.W_hiL = theano.shared(value=np.asarray(rng.uniform(self.n_hid), dtype='float32'), borrow=True, name='W_hiL')
        self.W_hiR = theano.shared(value=np.asarray(rng.uniform(self.n_hid), dtype='float32'), borrow=True, name='W_hiR')
        self.W_ciL = theano.shared(value=np.asarray(rng.uniform(self.n_hid), dtype='float32'), borrow=True, name='W_ciL')
        self.W_ciR = theano.shared(value=np.asarray(rng.uniform(self.n_hid), dtype='float32'), borrow=True, name='W_ciR')
        self.W_hflL = theano.shared(value=np.asarray(rng.uniform(self.n_hid), dtype='float32'), borrow=True, name='W_hflL')
        self.W_hfrL = theano.shared(value=np.asarray(rng.uniform(self.n_hid), dtype='float32'), borrow=True, name='W_hfrL')
        self.W_cflL = theano.shared(value=np.asarray(rng.uniform(self.n_hid), dtype='float32'), borrow=True, name='W_cflL')
        self.W_cfrL = theano.shared(value=np.asarray(rng.uniform(self.n_hid), dtype='float32'), borrow=True, name='W_cfrL')
        self.W_hflR = theano.shared(value=np.asarray(rng.uniform(self.n_hid), dtype='float32'), borrow=True, name='W_hflR')
        self.W_hfrR = theano.shared(value=np.asarray(rng.uniform(self.n_hid), dtype='float32'), borrow=True, name='W_hfrR')
        self.W_cflR = theano.shared(value=np.asarray(rng.uniform(self.n_hid), dtype='float32'), borrow=True, name='W_cflR')
        self.W_cfrR = theano.shared(value=np.asarray(rng.uniform(self.n_hid), dtype='float32'), borrow=True, name='W_cflR')
        self.W_hxL = theano.shared(value=np.asarray(rng.uniform(self.n_hid), dtype='float32'), borrow=True, name='W_hxL')
        self.W_hxR = theano.shared(value=np.asarray(rng.uniform(self.n_hid), dtype='float32'), borrow=True, name='W_hxR')
        self.W_hoL = theano.shared(value=np.asarray(rng.uniform(self.n_hid), dtype='float32'), borrow=True, name='W_hoL')
        self.W_hoR = theano.shared(value=np.asarray(rng.uniform(self.n_hid), dtype='float32'), borrow=True, name='W_hoR')
        self.W_co = theano.shared(value=np.asarray(rng.uniform(self.n_hid), dtype='float32'), borrow=True, name='W_co')
        self.bi = theano.shared(value=np.asarray(rng.uniform(self.n_hid), dtype='float32'), borrow=True, name='W_bi')
        self.bfl = theano.shared(value=np.asarray(rng.uniform(self.n_hid), dtype='float32'), borrow=True, name='W_bfl')
        self.bfr = theano.shared(value=np.asarray(rng.uniform(self.n_hid), dtype='float32'), borrow=True, name='W_bfr')
        self.bx = theano.shared(value=np.asarray(rng.uniform(self.n_hid), dtype='float32'), borrow=True, name='W_bx')
        self.bo = theano.shared(value=np.asarray(rng.uniform(self.n_hid), dtype='float32'), borrow=True, name='W_bo')
        self.params = [self.W_hiL, self.W_hiR, self.W_ciL, self.W_ciR, self.W_hflL, self.W_hfrL, self.W_cflL,
                       self.W_cfrL, self.W_hflR, self.W_hfrR, self.W_cflR, self.W_cfrR, self.W_hxL, self.W_hxR,
                       self.W_hoL, self.W_hoR, self.W_co, self.bi, self.bfl, self.bfr, self.bx, self.bo]

    def fwd(self, h_tm1, c_tm1):
        h_tm1L, h_tm1R = T.reshape(h_tm1.T, (self.n_in/2, 2*vec_dims), name='h_tm1L')[:, :vec_dims].T, \
                         T.reshape(h_tm1.T, (self.n_in/2, 2*vec_dims), name='h_tm1R')[:, vec_dims:2*vec_dims].T
        c_tm1L, c_tm1R = T.reshape(c_tm1.T, (self.n_in/2, 2*vec_dims), name='c_tm1L')[:, :vec_dims].T, \
                         T.reshape(c_tm1.T, (self.n_in/2, 2*vec_dims), name='c_tm1R')[:, vec_dims:2*vec_dims].T

        i_t = sigmoid(h_tm1L * self.W_hiL + h_tm1R * self.W_hiR + c_tm1L * self.W_ciL + c_tm1R * self.W_ciR + self.bi)
        f_tl = sigmoid(h_tm1L * self.W_hflL + h_tm1R * self.W_hflR + c_tm1L * self.W_cflL + c_tm1R * self.W_cflR + self.bfl)
        f_tr = sigmoid(h_tm1L * self.W_hfrL + h_tm1R * self.W_hfrR + c_tm1L * self.W_cfrL + c_tm1R * self.W_cfrR + self.bfr)
        x_t = h_tm1L * self.W_hxL + h_tm1R * self.W_hxR + self.bx
        c_t = f_tl * c_tm1L + f_tr * c_tm1R + i_t * tanh(x_t)
        o_t = sigmoid(h_tm1L * self.W_hoL + h_tm1R * self.W_hoR + c_t * self.W_co + self.bo)
        h_t = o_t * tanh(c_t)
        h_t_ = assert_shape_op(h_t, (vec_dims, self.n_hid))
        c_t_ = assert_shape_op(c_t, (vec_dims, self.n_hid))

        return h_t_, c_t_

class TreeLSTM(object):
    # Builds a tree structured LSTM based on the sentence length. Does not have paragraph support for now.
    # Does not have parse structured tree support. Works on 2**x (x is integer) length of sentences.
    def __init__(self, sen_vec, n_in):
        self.sen_vec = assert_shape_op(sen_vec, (vec_dims, n_in))
        n_layers = T.log2(sen_vec.shape[1])
        self.num_layers = T.cast(assert_op(n_layers, T.eq(n_layers, T.cast(n_layers, 'int32'))), 'int32')
        self.n_in = T.constant(n_in, name='n_in', dtype='int32')
        self.params = []
        self.output = self.tree_output()

    def tree_output(self):
        print 'Calling TreeLSTM.tree_output() ..'
        h_t0 = self.sen_vec
        c_t0 = T.zeros_like(h_t0)
        n_t0 = self.n_in
        rng_weights = weight_gen(self.n_in)

        [h_t, c_t], upd = theano.scan(fn=self.layer_output, sequences=[rng_weights],
                                      outputs_info=[h_t0, c_t0, n_t0],
                                      n_steps=self.num_layers, profile=True, name='TreeLSTM.tree_output.scan')
        self.pred = T.argmax(T.dot(h_t[-1].T, mappings_vec)/(abs(h_t[-1])*abs(mappings_vec, axis=0)))
        return h_t[-1][:, 0]

    def layer_output(self, rng_weight, h_tm1, c_tm1, nin_tm1, params_tm1):
        weight_array = rng_weight.get_value()

        W_hiL = theano.shared(value=weight_array[0*nin_tm1:1*nin_tm1], borrow=True, name='W_hiL')
        W_hiR = theano.shared(value=weight_array[1*nin_tm1:2*nin_tm1], borrow=True, name='W_hiR')
        W_ciL = theano.shared(value=weight_array[2*nin_tm1:3*nin_tm1], borrow=True, name='W_ciL')
        W_ciR = theano.shared(value=weight_array[3*nin_tm1:4*nin_tm1], borrow=True, name='W_ciR')
        W_hflL = theano.shared(value=weight_array[4*nin_tm1:5*nin_tm1], borrow=True, name='W_hflL')
        W_hfrL = theano.shared(value=weight_array[5*nin_tm1:6*nin_tm1], borrow=True, name='W_hfrL')
        W_cflL = theano.shared(value=weight_array[6*nin_tm1:7*nin_tm1], borrow=True, name='W_cflL')
        W_cfrL = theano.shared(value=weight_array[7*nin_tm1:8*nin_tm1], borrow=True, name='W_cfrL')
        W_hflR = theano.shared(value=weight_array[9*nin_tm1:9*nin_tm1], borrow=True, name='W_hflR')
        W_hfrR = theano.shared(value=weight_array[9*nin_tm1:10*nin_tm1], borrow=True, name='W_hfrR')
        W_cflR = theano.shared(value=weight_array[10*nin_tm1:11*nin_tm1], borrow=True, name='W_cflR')
        W_cfrR = theano.shared(value=weight_array[11*nin_tm1:12*nin_tm1], borrow=True, name='W_cflR')
        W_hxL = theano.shared(value=weight_array[12*nin_tm1:13*nin_tm1], borrow=True, name='W_hxL')
        W_hxR = theano.shared(value=weight_array[13*nin_tm1:14*nin_tm1], borrow=True, name='W_hxR')
        W_hoL = theano.shared(value=weight_array[14*nin_tm1:15*nin_tm1], borrow=True, name='W_hoL')
        W_hoR = theano.shared(value=weight_array[15*nin_tm1:16*nin_tm1], borrow=True, name='W_hoR')
        W_co = theano.shared(value=weight_array[16*nin_tm1:17*nin_tm1], borrow=True, name='W_co')
        bi = theano.shared(value=weight_array[17*nin_tm1:18*nin_tm1], borrow=True, name='W_bi')
        bfl = theano.shared(value=weight_array[18*nin_tm1:19*nin_tm1], borrow=True, name='W_bfl')
        bfr = theano.shared(value=weight_array[19*nin_tm1:20*nin_tm1], borrow=True, name='W_bfr')
        bx = theano.shared(value=weight_array[20*nin_tm1:21*nin_tm1], borrow=True, name='W_bx')
        bo = theano.shared(value=weight_array[21*nin_tm1:22*nin_tm1], borrow=True, name='W_bo')
        params = [W_hiL, W_hiR, W_ciL, W_ciR, W_hflL, W_hfrL, W_cflL, W_cfrL, W_hflR, W_hfrR, W_cflR, W_cfrR, W_hxL,
                  W_hxR, W_hoL, W_hoR, W_co, bi, bfl, bfr, bx, bo]

        h_tm1, c_tm1 = h_tm1[:, :nin_tm1], c_tm1[:, :nin_tm1]
        h_tm1L, h_tm1R = T.reshape(h_tm1.T, (nin_tm1/2, 2*vec_dims), name='h_tm1L')[:, :vec_dims].T, \
                         T.reshape(h_tm1.T, (nin_tm1/2, 2*vec_dims), name='h_tm1R')[:, vec_dims:2*vec_dims].T
        c_tm1L, c_tm1R = T.reshape(c_tm1.T, (nin_tm1/2, 2*vec_dims), name='c_tm1L')[:, :vec_dims].T, \
                         T.reshape(c_tm1.T, (nin_tm1/2, 2*vec_dims), name='c_tm1R')[:, vec_dims:2*vec_dims].T

        i_t = sigmoid(h_tm1L * W_hiL + h_tm1R * W_hiR + c_tm1L * W_ciL + c_tm1R * W_ciR + bi)
        f_tl = sigmoid(h_tm1L * W_hflL + h_tm1R * W_hflR + c_tm1L * W_cflL + c_tm1R * W_cflR + bfl)
        f_tr = sigmoid(h_tm1L * W_hfrL + h_tm1R * W_hfrR + c_tm1L * W_cfrL + c_tm1R * W_cfrR + bfr)
        x_t = h_tm1L * W_hxL + h_tm1R * W_hxR + bx
        c_t = f_tl * c_tm1L + f_tr * c_tm1R + i_t * tanh(x_t)
        o_t = sigmoid(h_tm1L * W_hoL + h_tm1R * W_hoR + c_t * W_co + bo)
        h_t = o_t * tanh(c_t)
        h_t_ = assert_shape_op(h_t, (vec_dims, nin_tm1/2))
        c_t_ = assert_shape_op(c_t, (vec_dims, nin_tm1/2))
        h_tx = T.set_subtensor(T.zeros((vec_dims, self.n_in), 'float32')[0:], h_t_)
        c_tx = T.set_subtensor(T.zeros((vec_dims, self.n_in), 'float32')[0:], c_t_)
        return h_tx, c_tx, nin_tm1/2, params

    def loss(self, y):
        print 'Calling TreeLSTM.loss() ..'
        # y is a single output. No support for minibatches as of yet.
        y_ = assert_shape_op(y, self.output.shape)
        loss = T.mean(self.output()**2 - y_**2)
        return loss


def form_dataset(doc, n_in):
    """
    Given a document and the number of input units, return the vector form  of the document segmented into units of
    length (n_in + 1)
    :param doc: String : Location of doc.
    :param n_in: Number of input units of the TreeLSTM
    :return: return the vector form of the document segmented into units of length(n_in + 1)
    """
    print 'Calling form_dataset()..'
    doc_obj = open(doc)
    data = tokenize(doc_obj.read().lower())
    data = data[:int(len(data)/(n_in+1)) * (n_in+1)]
    n_sen = len(data)/(n_in+1)
    data_x, data_y = np.asarray(data).reshape((n_sen, (n_in+1)))[:, :n_in], \
                     np.asarray(data).reshape((n_sen, (n_in+1)))[:, -1]
    data_x_vec = np.asarray([sentence_vec(data_x[i], word_vecs) for i in range(len(data_x))], dtype=theano.config.floatX)
    shared_x = theano.shared(np.concatenate(data_x_vec, axis=1), name='vec_data_x', borrow=True)
    shared_x_ = assert_op(shared_x, T.eq(shared_x.get_value().shape[0], vec_dims),
                          T.eq(shared_x.get_value().shape[1], n_sen*n_in))
    shared_y = theano.shared(np.asarray(sentence_vec(data_y, word_vecs),
                               dtype=theano.config.floatX), name='vec_data_y', borrow=True)
    shared_y_ = assert_op(shared_y, T.eq(shared_y.get_value().shape[0], vec_dims),
                          T.eq(shared_y.get_value().shape[1], n_sen))
    doc_obj.close()
    # Shape(vec_data_y) reshaped from Number of sentences * Vector Dimensions * 1 to Number of sentences * Vector Dims
    return shared_x_, shared_y_

train_x, train_y = form_dataset(doc_path, n_in)

def SGD(eta, minibatch_size, n_minibatch, n_epochs):
    # Testing and Validation data are the outputs of the last inputs.
    print 'Calling SGD() ..'
    index = T.iscalar('index')
    x, y = T.matrices('x', 'y')

    tree = TreeLSTM(x, n_in)
    updates = [(param, param - eta * gparam) for param, gparam in zip(tree.params, T.grad(tree.loss(y), tree.params))]
    train_fn = theano.function([index], tree.loss(y), updates=updates,
                               givens={x: train_x[:, minibatch_size * n_in * index: (minibatch_size + 1) * n_in * index],
                                       y: train_y[:, minibatch_size * index: (minibatch_size + 1) * index]}
                               )

    # Compilation over
    #################
    ## TRAIN MODEL ##
    #################

    for epoch in n_epochs:
        for idx in range(n_minibatch):
            train_fn(idx)

#SGD(0.01, 1, 10, 2)
#  THEANO_FLAGS='theano.config.floatX='float32', device='cpu', optimizer=None' python childtree.py
# import theano, numpy as np, theano.tensor as T
