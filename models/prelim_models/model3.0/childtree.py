__author__ = 'azrael'

"""
POTENTIAL IDEAS:

1) Instead of keeping the weight matrix for a single connection as a scalar, make it a matrix with dimensions=vector dims
 While this significantly increases the number of dimensions, it provides much better control over the manipulation of
 the word vectors. This theory does not agree with the current techniques and biological connections but, theoretically
 it should give much better performance. 
 Changed the weights to have more dimensions in the RNNLayer. but not the biases. Feel wrong doing that. Try experiments
 and report results. 
 
2) The model as of the moment takes n_in+1 words and feeds n_in words in the net and the last word as the output. So that
 single word is never fed into input. This should have a negative effect/ break the sequence. Test it and report results.

3) The next word is chosen on the basis of only the previous one(yes, they are all interrelated) but not directly related.
 I believe that this is what leads to such output in Karpathy's char- rnn(small group of words make sense but large
 groups don't)

 This can be changed by this method-

 Suppose there are 3 neurons. 1st is paragraph neuron, 2nd is sentence neuron, 3 word neuron. The output of the word
 neuron is the prediction of the next word.  Normally there is connection between 1st and 2nd, and 2nd and 3rd(recurrent
 neural network). But by enforcing a connection between 1st and 3rd, long term dependencies are enforced.
 
4) Instead of just updating the weights, updation of word vectors(given fixed weights) can be done. This should be done
 by first just running the code and learning some weights. Now keep this weights constant and update the word vectors.
 Should be done multiple times for good results.

5) Keep the unkown vectors as 0,0,0,.. and then update weights. Keep weights fixed and then update word vecs. This should
 be done only when sure of model performance.
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
 See http://deeplearning.net/software/theano/library/scan.html#optimizing-scan-s-performance for faster scan optimization.

"""

"""
 CHANGES TO BE MADE IN CODE:

 1) Add minibatch support.                                                                      IMP, NTR
 2) Add paragraph support.                                                                      IMP, TR
 3) Add regularization - relu, dropout, L1, L2, more stuff, adaptive delta, RMSProp             IMP, NTR
 4) Active visualization for error rate, current preds.                                         NIMP, TR
 5) Instead of giving 2 words to a single neuron, give 1 word to single neuron. Can be done by adding 1 more layer at
 bottom. The dimensionality of the sentence will have to increase to work properly for this NN. Can be done by(example):
 suppose n_in = 4 words                                                                         NIMP, TR
 so make the number of layers = log2(4)+1 = 3 The bottom most layer will have inputs: [w1 0 w2 0 w3 0 w4 0]
 6) Test and validation outputs have to be found by feeding the output into the input. Suppose there are total 6 input
 neurons and they generate a output of 1 word. So take out the first word in the previous input sequence and shift the
 input sequence.                                                                                DONE, IMP, NTR
 7) Save model parameters and reuse them when using an addition in the code. Should increase speed a lot. IMP, NTR
 8) Take note of proper initialization of weights and biases. Compare using of sigmoid v/s tanh v/s relu v/s another
 activation functions.                                                                          IMP, TR
 9) Try different types of relationships, i.e. many-to-one/ using the whole last output and averaging/adding weights
  to get the final output.                                                                      NIMP, TR
 10) Add Bidirectional LSTM support.                                                            IMP, TR
 11) Instead of using Simple Neurons, use LSTMs.                                                IMP, NTR
 12) Increase the dimensionality of the hidden to hidden connections. At this moment, it is n * n where n is the number
  of hidden nodes. That means that a vector in the output state is going to be multiplied with a vector of dimensionality
  1. That does not allow room for internal changes in representation.
 13) It is possible to keep multiple nodes instead of a single LSTM in the tree structure. But I have already increased
 the dimensionality of the LSTM to vec_dims. Give it a try at the end.                          NIMP, NTR
 ----------------DONE---------------
 1) Change error function to take into account angle and distance of the vectors.( Also use weights in this. Let the
 NN figure out what is best for it. Start from equal weights/see what Kai sheng tai has done.)
 DONE.
 Upon analysis, it was found that that allocating weights to angle and distance in the loss function leads to depression
 of + errors and increment of - errors. So just 0.5 attached to both(constant).
 2) Increase dimensions of weights to vec_dims.(see POT IDEAS section)
 DONE.
 3) The angle between the two vectors can not be found accurately only using dot product. Use cross product as well.
  Find the accurate angle. If sin(theta) > 0: The angle is between 0 and 180 degrees. If sin(theta) < 0: The angle is
  between 0 and -180 degrees. Using -180 to 180 better(compared to 0 to 360 degrees) since need to bring to 0 and I
  think the negative sign(-180) will reinforce better gradients.
 DONE(Partially)
  It is impossible to 50/300 dimensional cross product with only 2 vectors. See this page-
  https://mathoverflow.net/questions/94312/n-dimensional-cross-product-reference-request
  Using only cos angle. This means that program would not be able to identify between vectors having 2 degrees and 358 degrees.
  So maybe will try to decrease 358(2 degrees) but instead end up increasing it. Test that can be run to check this-
  --- Use a single sentence and keep giving it to the network. keep updating the network. Keep note of angle and distance difference.

"""

"""
MAKING THE CODE GIVE BETTER OUTPUTS

1) Proper loss function(something that can control each node). See what Richard socher gave in the glove paper.
2) More dimensionality(2 ways to implement this)--
        1.. Mix fully connected layers with tree layers
        2.. Increasing dimensionality of weights as expressed above
        3.. From the output of the tree, feed that into a layered lstm layer.
3) Use an encoder decoder approach and see results. Suppose you have an encoder, then use multiple decoders. One decoder
 predicts future, one decoder predicts past, and one decoder predicts present.
"""

import theano.tensor as T
from preprocess1 import *

"""
I have seen somewhere that the hiddent to hidden matrix is identity type? is this true.?
In RNNStackedLayers, the bias dimensions are kept as vec_dims, when it should be actulaly 1 (only one output unit).
When I keep it as one, then Shape error shows up.
"""

"""
PROBLEMS IN OUTPUT:

1) Seems to be due to a combination of weights and learning rate. Properly initialize weights and learning rate.
2) The for loop does not seem to be properly working. Looks like it is giving the same input/ compilation is happening
"""

n_in = 8
n_in_tree = 4
vec_file = '/home/azrael/Documents/nn/code/vectors.6B.50d.txt'
vec_dims = 50
assert_op = T.opt.Assert()
n_nodes = 500
rng = np.random.RandomState(1)
doc_path = '/home/azrael/Documents/nn/code/wikisample.txt'
word_vecs, mappings_words, mappings_vec = load_vecs(vec_file, doc_path, vec_dims)
np_array_vecs = mappings_vec.get_value()

print 'The length of the dictionary of words is', len(mappings_words)

def sigmoid(x):
    return T.nnet.sigmoid(x)

def tanh(x):
    return T.tanh(x)

class AssertShape(T.opt.Assert):
    def make_node(self, value, *shapes):
        assert len(shapes) == 1
        assert isinstance(shapes[0], (tuple, list))
        params = [value]
        for actual_size, expected_size in zip(value.shape, shapes[0]):
            params.append(T.eq(actual_size, expected_size))
        return super(AssertShape, self).make_node(*params)

assert_shape_op = AssertShape()

def absolute(x, axis=0):
    """
    Takes the matrix/vector x and finds the absolute along the axis.
    :param x: T.matrix
    :return: Absolute along the given axis. T.vector
    """
    x = assert_op(x, T.or_(T.eq(x.ndim, 2), T.eq(x.ndim, 1)))
    return T.sqrt(T.sum(T.sqr(x), axis))

def weight_gen(n_in):
    cmplte_weights = np.asarray(rng.uniform(-0.05, 0.05, (2*n_in-1)), 'float32')
    seq_list = []
    r_sum = 0
    for i in range(int(np.log2(n_in))):
        y = cmplte_weights[r_sum : (r_sum + n_in/2**(i+1))]
        seq_list.append(theano.shared(y))
        r_sum += (n_in/2**(i+1))
    return seq_list

class TreeLSTMLayer(object):
    def __init__(self, n_in_tree):
        print 'Initializing layer.....'
        self.n_in = n_in_tree

        weights_shape = (vec_dims, self.n_in/2)
        biases_shape = self.n_in/2
        # Required later for broadcasting purposes.
        low = -0.5
        high = 0.5

        self.W_hiL = theano.shared(value=np.asarray(rng.uniform(low, high, weights_shape), dtype='float32'), borrow=True, name='W_hiL')
        self.W_hiR = theano.shared(value=np.asarray(rng.uniform(low, high, weights_shape), dtype='float32'), borrow=True, name='W_hiR')
        self.W_ciL = theano.shared(value=np.asarray(rng.uniform(low, high, weights_shape), dtype='float32'), borrow=True, name='W_ciL')
        self.W_ciR = theano.shared(value=np.asarray(rng.uniform(low, high, weights_shape), dtype='float32'), borrow=True, name='W_ciR')
        self.W_hflL = theano.shared(value=np.asarray(rng.uniform(low, high, weights_shape), dtype='float32'), borrow=True, name='W_hflL')
        self.W_hfrL = theano.shared(value=np.asarray(rng.uniform(low, high, weights_shape), dtype='float32'), borrow=True, name='W_hfrL')
        self.W_cflL = theano.shared(value=np.asarray(rng.uniform(low, high, weights_shape), dtype='float32'), borrow=True, name='W_cflL')
        self.W_cfrL = theano.shared(value=np.asarray(rng.uniform(low, high, weights_shape), dtype='float32'), borrow=True, name='W_cfrL')
        self.W_hflR = theano.shared(value=np.asarray(rng.uniform(low, high, weights_shape), dtype='float32'), borrow=True, name='W_hflR')
        self.W_hfrR = theano.shared(value=np.asarray(rng.uniform(low, high, weights_shape), dtype='float32'), borrow=True, name='W_hfrR')
        self.W_cflR = theano.shared(value=np.asarray(rng.uniform(low, high, weights_shape), dtype='float32'), borrow=True, name='W_cflR')
        self.W_cfrR = theano.shared(value=np.asarray(rng.uniform(low, high, weights_shape), dtype='float32'), borrow=True, name='W_cflR')
        self.W_hxL = theano.shared(value=np.asarray(rng.uniform(low, high, weights_shape), dtype='float32'), borrow=True, name='W_hxL')
        self.W_hxR = theano.shared(value=np.asarray(rng.uniform(low, high, weights_shape), dtype='float32'), borrow=True, name='W_hxR')
        self.W_hoL = theano.shared(value=np.asarray(rng.uniform(low, high, weights_shape), dtype='float32'), borrow=True, name='W_hoL')
        self.W_hoR = theano.shared(value=np.asarray(rng.uniform(low, high, weights_shape), dtype='float32'), borrow=True, name='W_hoR')
        self.W_co = theano.shared(value=np.asarray(rng.uniform(low, high, weights_shape), dtype='float32'), borrow=True, name='W_co')
        self.bi = theano.shared(value=np.asarray(rng.uniform(low, high, biases_shape), dtype='float32'), borrow=True, name='bi')
        self.bfl = theano.shared(value=np.asarray(rng.uniform(low, high, biases_shape), dtype='float32'), borrow=True, name='bfl')
        self.bfr = theano.shared(value=np.asarray(rng.uniform(low, high, biases_shape), dtype='float32'), borrow=True, name='bfr')
        self.bx = theano.shared(value=np.asarray(rng.uniform(low, high, biases_shape), dtype='float32'), borrow=True, name='bx')
        self.bo = theano.shared(value=np.asarray(rng.uniform(low, high, biases_shape), dtype='float32'), borrow=True, name='bo')
        self.params = [self.W_hiL, self.W_hiR, self.W_ciL, self.W_ciR, self.W_hflL, self.W_hfrL, self.W_cflL,
                       self.W_cfrL, self.W_hflR, self.W_hfrR, self.W_cflR, self.W_cfrR, self.W_hxL, self.W_hxR,
                       self.W_hoL, self.W_hoR, self.W_co, self.bi, self.bfl, self.bfr, self.bx, self.bo]
        print 'Initialization of layer successful....'

    def fwd(self, h_tm1, c_tm1):
        print 'Doing forward pass...'
        h_tm1_ = assert_op(h_tm1, T.eq(h_tm1.shape[0], vec_dims), T.eq(h_tm1.shape[1], self.n_in))
        print 'n-in', self.n_in
        h_tm1L, h_tm1R = T.reshape(h_tm1_.T, (self.n_in/2, 2*vec_dims), name='h_tm1L')[:, :vec_dims].T, \
                         T.reshape(h_tm1_.T, (self.n_in/2, 2*vec_dims), name='h_tm1R')[:, vec_dims:2*vec_dims].T

        c_tm1L, c_tm1R = T.reshape(c_tm1.T, (self.n_in/2, 2*vec_dims), name='c_tm1L')[:, :vec_dims].T, \
                         T.reshape(c_tm1.T, (self.n_in/2, 2*vec_dims), name='c_tm1R')[:, vec_dims:2*vec_dims].T

        h_tm1L_ = assert_op(h_tm1L, T.eq(h_tm1L.shape[0], vec_dims), T.eq(h_tm1L.shape[1], self.n_in/2))
        c_tm1L_ = assert_op(h_tm1L, T.eq(c_tm1L.shape[0], vec_dims), T.eq(c_tm1L.shape[1], self.n_in/2))

        i_t = sigmoid(h_tm1L_ * self.W_hiL + h_tm1R * self.W_hiR + c_tm1L_ * self.W_ciL + c_tm1R * self.W_ciR + self.bi)
        f_tl = sigmoid(h_tm1L_ * self.W_hflL + h_tm1R * self.W_hflR + c_tm1L * self.W_cflL + c_tm1R * self.W_cflR + self.bfl)
        f_tr = sigmoid(h_tm1L_ * self.W_hfrL + h_tm1R * self.W_hfrR + c_tm1L * self.W_cfrL + c_tm1R * self.W_cfrR + self.bfr)
        x_t = h_tm1L_ * self.W_hxL + h_tm1R * self.W_hxR + self.bx
        c_t = f_tl * c_tm1L_ + f_tr * c_tm1R + i_t * tanh(x_t)
        c_t_ = assert_op(c_t, T.eq(c_t.shape[0], vec_dims), T.eq(c_t.shape[1], self.n_in/2))
        o_t = sigmoid(h_tm1L_ * self.W_hoL + h_tm1R * self.W_hoR + c_t_ * self.W_co + self.bo)
        h_t = o_t * tanh(c_t)
        h_t_ = assert_shape_op(h_t, (vec_dims, self.n_in/2))
        c_t_ = assert_shape_op(c_t, (vec_dims, self.n_in/2))
        print 'Forward pass successful...'
        return h_t_, c_t_

class TreeLSTM(object):
    # Builds a tree structured LSTM based on the sentence length. Does not have paragraph support for now.
    # Does not have parse structured tree support. Works on 2**x (x is integer) length of sentences.
    def __init__(self, n_in_tree):
        print 'Initialing TreeLSTM...'
        assert n_in_tree/int(n_in_tree) == 1
        self.n_in = n_in_tree
        self.params = []
        print 'Successfully initialized'

    def tree_output(self, sen_vec):
        print 'Calculating tree output...'
        sen_vec_ = assert_shape_op(sen_vec, (vec_dims, self.n_in))
        h_t0 = sen_vec_
        c_t0 = T.zeros_like(h_t0)
        layer1 = TreeLSTMLayer(self.n_in)
        layer2 = TreeLSTMLayer(self.n_in/2)
        self.params += layer1.params
        self.params += layer2.params
        layer1_h_t, layer1_c_t = layer1.fwd(h_t0, c_t0)
        layer2_h_t, layer2_c_t = layer2.fwd(layer1_h_t, layer1_c_t)
        h_t_ = layer2_h_t
        h_t__ = assert_op(h_t_, T.eq(h_t_.shape[0], vec_dims), T.eq(h_t_.shape[1], 1))
        print 'Completed tree output...'
        return h_t__  # Extra dimension not removed. It is used later in SentenceLSTMLayers.

    def pred(self, output):
        W1, W2 = 0.5, 0.5
        pred = T.argmin(W1 * T.arccos(T.dot(output, mappings_vec)/(absolute(output)*absolute(mappings_vec, axis=0))) +
                        W2 * (absolute(mappings_vec, axis=0) - absolute(output)))
        return pred

    def loss(self, y, output):
        print 'Calling TreeLSTM.loss()...'
        # y is a single output. No support for minibatches as of yet.
        W_dist, W_angle = 0.5, 0.5
        y_ = assert_op(y, T.eq(y.shape[0], output.shape[0]), T.eq(y.shape[1], 1))
        loss = W_dist * T.mean((output - y_[:, 0])**2) + W_angle * (T.dot(output, y[:, 0])/
                                                                         (absolute(output)*absolute(y[:0])))
        return loss

class RNNLayer(object):
    """
    This is a single layer which simply unfolds itself in time.
    """
    def __init__(self, input_size, n_nodes):
        self.n_nodes = n_nodes
        low, high = -0.5, 0.5
        temp_imp_size = n_nodes
        if input_size == 1:
            temp_imp_size = vec_dims
        self.W_x = theano.shared(value=np.asarray(rng.uniform(low, high, (temp_imp_size, n_nodes)), 'float32'), borrow=True, name='W_x')
        self.W_h = theano.shared(value=np.asarray(rng.uniform(low, high, (n_nodes, n_nodes)), 'float32'), borrow=True, name='W_h')
        self.b = theano.shared(value=np.asarray(rng.uniform(low, high,  n_nodes), 'float32'), borrow=True, name='b')
        self.h_t0 = theano.shared(value=np.asarray(np.zeros((vec_dims, n_nodes)), 'float32'), borrow=True, name='h_t0')
        self.params = [self.W_x, self.W_h, self.b]

class RNNStackedLayers(object):
    """
    Combination of stacked layers of type RNNLayer.
    """
    def __init__(self, n_nodes):
        # For now fixed number of layers. later, using scan can produce variable
        # number of layers.
        low, high = -0.5, 0.5
        self.W = theano.shared(value=np.asarray(rng.uniform(low, high, (vec_dims, n_nodes)), 'float32'), borrow=True, name='W')
        self.b = theano.shared(value=np.asarray(rng.uniform(low, high, vec_dims), 'float32'), borrow=True, name='b')
        self.layer1 = RNNLayer(1, n_nodes)
        self.layer2 = RNNLayer(n_nodes, n_nodes)

        self.params = self.layer1.params + self.layer2.params + [self.W, self.b]

    def output(self, x_t):
        [layer1_out, layer2_out], updates = theano.scan(fn=self.oneStep, sequences=x_t, outputs_info=[self.layer1.h_t0, self.layer2.h_t0],
                                                        non_sequences=[self.layer1.W_x, self.layer1.W_h, self.layer1.b,
                                                                       self.layer2.W_x, self.layer2.W_h, self.layer2.b])
        model_out = T.sum(input=layer2_out[-1] * self.W, axis=1) + self.b
        return assert_op(model_out, T.eq(model_out.ndim, 1), T.eq(model_out.shape[0], vec_dims))

    def oneStep(self, x_t, h1_tm1, h2_tm1, W1_x, W1_h, b1, W2_x, W2_h, b2):
        h1_tm1_ = assert_op(h1_tm1, T.eq(h1_tm1.shape[0], vec_dims), T.eq(h1_tm1.shape[1], n_nodes))
        h2_tm1_ = assert_op(h2_tm1, T.eq(h2_tm1.shape[0], vec_dims), T.eq(h2_tm1.shape[1], n_nodes))

        h1_t = sigmoid(T.dot(x_t, W1_x) + T.dot(h1_tm1_, W1_h) + b1)
        h2_t = sigmoid(T.dot(h1_t, W2_x) + T.dot(h2_tm1_, W2_h) + b2)
        return h1_t, h2_t

    def valid_output(self, x_t0, n_steps):
        [x_t, layer1_out, layer2_out], updates = theano.scan(fn=self.valid_oneStep, outputs_info=[x_t0, self.layer1.h_t0, self.layer2.h_t0],
                                                             non_sequences=[self.layer1.W_x, self.layer1.W_h, self.layer1.b,
                                                                       self.layer2.W_x, self.layer2.W_h, self.layer2.b],
                                                             n_steps=n_steps)

        return assert_op(x_t, T.eq(x_t.shape[0], n_steps), T.eq(x_t.shape[1], vec_dims))

    def valid_oneStep(self, x_tm1, h1_tm1, h2_tm1, W1_x, W1_h, b1, W2_x, W2_h, b2):
        h1_tm1_ = assert_op(h1_tm1, T.eq(h1_tm1.shape[0], vec_dims), T.eq(h1_tm1.shape[1], n_nodes))
        h2_tm1_ = assert_op(h2_tm1, T.eq(h2_tm1.shape[0], vec_dims), T.eq(h2_tm1.shape[1], n_nodes))

        h1_t = sigmoid(T.dot(x_tm1, W1_x) + T.dot(h1_tm1_, W1_h) + b1)
        h2_t = sigmoid(T.dot(h1_t, W2_x) + T.dot(h2_tm1_, W2_h) + b2)
        x_t = T.sum(h2_t * self.W, axis=1) + self.b
        return x_t, h1_t, h2_t

    def pred(self, output):
        W1 = 0.5
        W2 = 0.5
        pred = T.argmin(W1 * T.arccos(T.dot(output, mappings_vec)/(absolute(output)*absolute(mappings_vec, axis=0))) +
                        W2 * (absolute(mappings_vec, axis=0) - absolute(output)))
        return pred

    def loss(self, y, output):
        print 'Calling TreeLSTM.loss()...'
        # y is a single output. No support for minibatches as of yet.
        W_dist, W_angle = 0.5, 0.5
        y_ = assert_op(y, T.eq(y.shape[0], output.shape[0]), T.eq(y.shape[1], 1))
        loss = W_dist * T.mean((output - y_[:, 0])**2) + W_angle * (T.dot(output, y[:, 0])/(absolute(output)*absolute(y[:0])))
        return loss

class SentenceLSTMLayers(object):
    def __init__(self, n_in_tree, n_nodes):
        print 'Calling SentenceLSTMLayers.__init__()...'
        assert (n_in_tree % 2 == 0), "Number of input nodes of LSTM tree has to be even"
        self.n_in_tree = n_in_tree
        self.n_nodes = n_nodes
        self.lstm_tree = TreeLSTM(n_in_tree)
        self.rnn_stacked = RNNStackedLayers(n_nodes)
        self.params = self.lstm_tree.params + self.rnn_stacked.params
        print 'Successful initialization of SentenceLSTMLayers'

    def output(self, vec_words, n_steps=0, valid=False):
        # vec_words should have shape- (Vector dimensions, Number of words)
        print 'Calling SentenceLSTMLayers.output()...'

        if not valid:
            vec_words_ = assert_op(vec_words.T, T.eq(vec_words.shape[0], vec_dims), T.gt(vec_words.shape[1], self.n_in_tree))
            vec_words_tree = vec_words[:, :self.n_in_tree]
            vec_words_rem = vec_words_[self.n_in_tree:]
            lstm_tree_output = self.lstm_tree.tree_output(vec_words_tree)
            x_t = T.concatenate([lstm_tree_output.T, vec_words_rem])
            x_t_ = assert_op(x_t, T.eq(x_t.shape[0], n_in-n_in_tree+1), T.eq(x_t.shape[1], vec_dims))
            output = self.rnn_stacked.output(x_t_)
        else:
            assert n_steps > 0
            assert vec_words.ndim == 1
            vec_words_ = assert_op(vec_words, T.eq(vec_words.shape[0], vec_dims))
            x_t_ = vec_words_
            output = self.rnn_stacked.valid_output(x_t_, n_steps)
        print 'SentenceLSTMLayers.output() successful'
        return output

    def pred(self, output):
        W1 = 0.5
        W2 = 0.5
        pred = T.argmin(W1 * T.arccos(T.dot(output, mappings_vec.T)/(absolute(output)*absolute(mappings_vec, axis=1))) +
                        W2 * (absolute(mappings_vec, axis=1) - absolute(output)))
        return pred

    def loss(self, y, output):
        print 'Calling SentenceLSTMLayers.loss()...'
        # y is a single output. No support for minibatches as of yet.
        W_dist, W_angle = 0.5, 0.5
        assert output.ndim == 1
        y_ = assert_op(y, T.eq(y.shape[0], output.shape[0]), T.eq(y.shape[1], 1))
        loss = T.mean((output - y_[:, 0])**2) + T.arccos(T.dot(output, y[:, 0])/(absolute(output)*absolute(y[:, 0])))
        loss_ = assert_op(loss, T.eq(loss.ndim, 0))
        print 'The dimensions of loss is', loss.ndim
        print 'SentenceLSTMLayers.loss() call successful'
        return loss_

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
    print 'Length of data is:', len(data)
    n_sen = len(data)/(n_in+1)
    print 'N_sen is:', n_sen
    train_x, train_y = np.asarray(data).reshape((n_sen, (n_in+1)))[:, :n_in], \
                     np.asarray(data).reshape((n_sen, (n_in+1)))[:, -1]
    train_x_vec = np.asarray([sentence_vec(train_x[i], word_vecs) for i in range(len(train_x))], dtype=theano.config.floatX)
    print 'Shape of train_x_vec is:', train_x_vec.shape
    shared_x = theano.shared(np.concatenate(train_x_vec, axis=1), name='shared_x', borrow=True)
    val = shared_x.get_value()
    print 'Printing shape of shared_x :', val.shape
    shared_x_ = assert_op(shared_x, T.eq(shared_x.get_value().shape[0], vec_dims),
                          T.eq(shared_x.get_value().shape[1], n_sen*n_in))
    shared_y = theano.shared(np.asarray(sentence_vec(train_y, word_vecs),
                               dtype=theano.config.floatX), name='shared_y', borrow=True)
    shared_y_ = assert_op(shared_y, T.eq(shared_y.get_value().shape[0], vec_dims),
                          T.eq(shared_y.get_value().shape[1], n_sen))
    doc_obj.close()
    # Shape(shared_y) reshaped from Number of sentences, Vector Dimensions, 1) to (Number of sentences, Vector Dims)
    print 'form_dataset completed successfuly ...'
    return shared_x_, shared_y_, train_y

train_x, train_y, word_labels = form_dataset(doc_path, n_in)

def SGD(eta, n_minibatch, n_epochs, valid_steps, valid_interval, valid_headwords):

    # Testing and Validation data are the outputs of the last inputs.
    print 'Calling SGD() ..'
    t0 = time.time()
    index = T.iscalar('index')
    x, y = T.matrices('x', 'y')
    print 'The length of valid_headwords is', len(valid_headwords)
    print 'n_epochs * n_minibatch * n_in is', n_epochs * n_minibatch * n_in / valid_interval
    # minibatch size needs to be added later in assert statement.
    assert len(valid_headwords) >= (n_epochs * n_minibatch)/ valid_interval, 'The length of valid_headwords need to be' \
                                                                                       'greater than n_epochs * n_minibatch/ valid_interval'
    # Need to put  (* minibatch_size) when support for minibatches is introduced.
    model = SentenceLSTMLayers(n_in_tree, n_nodes)
    model_output = model.output(x)
    valid_output = model.output(x, n_steps=valid_steps, valid=True)

    loss = model.loss(y, model_output)
    pred = model.pred(model_output)
    validation_pred = model.pred(valid_output)

    params = model.params
    updates = [(param, param - eta * gparam) for param, gparam in zip(params, T.grad(loss, params))]
    # No need to return the updates by scan function within RNNStackedLayers. Only needed when shared variables are
    # updated within step function. This is not the case here. See this source

    train_fn = theano.function([index], [loss, pred], updates=updates,
                               givens={x: train_x[:, n_in * index: n_in * (index + 1)],
                                       y: train_y[:, index: index + 1]})

    valid_fn = theano.function([x], validation_pred)

    # Compilation over
    #################
    ## TRAIN MODEL ##
    #################
    words_seen = 0
    match_idx = 0
    valid_idx = 0
    training_loss = []
    for i in range(n_epochs):
        print 'The current epoch number is', i
        t1 = time.time()
        for idx in range(n_minibatch):
            train_loss, train_pred = train_fn(idx)
            training_loss.append(train_loss)
            if (i*n_minibatch+idx) % valid_interval == 0:
                valid_op = []
                valid_headword = valid_headwords[valid_idx]
                valid_input = word_vecs[valid_headword]
                print '---------------------------------------------------------------'
                assert valid_input.shape[0] == vec_dims, 'ASSERTION 1 FALSE'
                print 'The validation headword for validation index =', valid_idx, 'is', \
                       valid_headword
                valid_pred = valid_fn(valid_input)
                valid_op.append(valid_pred)

                print 'The validation prediction is', ' '.join([mappings_words[idx] for idx in valid_op])
        print 'Time taken by this epoch is', time.time()-t1

    print 'Time taken by __main__ and training is', time.time()-t0
    print 'Total words seen is', words_seen
    print 'Total words matched is', match_idx
    print 'Ratio is', match_idx/words_seen

if __name__=="__main__":
    valid_headwords = mappings_words[:]
    SGD(0.1, 20000, 700, 10, 3333, valid_headwords)

# THEANO_FLAGS='theano.config.floatX='float32', device='cpu', optimizer='None'' python childtreesimple.py
# import theano, numpy as np, theano.tensor as T
# x_t_ = assert_op(x_t, T.or_(T.and_(T.eq(x_t.ndim, 1), T.eq(x_t.shape[0], vec_dims)),
#                                T.and_(T.and_(T.eq(x_t.ndim, 2), T.eq(x_t.shape[0], vec_dims)), T.eq(x_t.shape[1], self.n_nodes))))
