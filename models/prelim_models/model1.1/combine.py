__author__ = 'azrael'
import theano
import theano.tensor as T
import time
from preprocess import *

vector_dim = 79

def sigmoid(x):
    return T.nnet.sigmoid(x)

def tanh(x):
    return T.tanh(x)


doc = load_dataset("/home/azrael/Documents/nn/andrejrec/data/warpeace_input.txt")
doc_clean = remove_letters(doc)
mappings = one_hot_vec(doc_clean)
train_len = 200
valid_len = 200
test_len = 200
seq_len = n_in = n_hid = n_out = 200


def string_vec(abc):
    return np.array([mappings[z][0] for z in list(abc)], dtype=theano.config.floatX).T


def string_idx(op):
    # given a string find the indexes of the letters in the string.
    return np.array([mappings[z][1] for z in list(op)], dtype='int32').T  # int required for indexes.


def idx_string(idx_list):
    return ''.join([list(mappings)[idx] for idx in idx_list])


class LSTMlayer(object):
    def __init__(self, rng, n_in, n_hid):
        self.n_in = n_in
        self.n_hid = n_hid
        self.W_xi = theano.shared(
            value=np.asarray(rng.uniform(-0.01, 0.01, size=n_in), dtype=theano.config.floatX),
            borrow=True, name='W_xi')
        self.W_xf = theano.shared(
            value=np.asarray(rng.uniform(-0.01, 0.01, size=n_in), dtype=theano.config.floatX),
            borrow=True, name='W_xf')
        self.W_xc = theano.shared(
            value=np.asarray(rng.uniform(-0.01, 0.01, size=n_in), dtype=theano.config.floatX),
            borrow=True, name='W_xc')
        self.W_xo = theano.shared(
            value=np.asarray(rng.uniform(-0.01, 0.01, size=n_in), dtype=theano.config.floatX),
            borrow=True, name='W_xo')
        self.W_hi = theano.shared(
            value=np.asarray(rng.uniform(-0.01, 0.01, size=(n_hid, n_hid)), dtype=theano.config.floatX),
            borrow=True, name='W_hi')
        self.W_hf = theano.shared(
            value=np.asarray(rng.uniform(-0.01, 0.01, size=(n_hid, n_hid)), dtype=theano.config.floatX),
            borrow=True, name='W_hf')
        self.W_hc = theano.shared(
            value=np.asarray(rng.uniform(-0.01, 0.01, size=(n_hid, n_hid)), dtype=theano.config.floatX),
            borrow=True, name='W_hc')
        self.W_ho = theano.shared(
            value=np.asarray(rng.uniform(-0.01, 0.01, size=(n_hid, n_hid)), dtype=theano.config.floatX),
            borrow=True, name='W_ho')
        self.W_ci = theano.shared(
            value=np.asarray(rng.uniform(-0.01, 0.01, size=(n_hid, n_hid)), dtype=theano.config.floatX),
            borrow=True, name='W_ci')
        self.W_cf = theano.shared(
            value=np.asarray(rng.uniform(-0.01, 0.01, size=(n_hid, n_hid)), dtype=theano.config.floatX),
            borrow=True, name='W_cf')
        self.W_co = theano.shared(
            value=np.asarray(rng.uniform(-0.01, 0.01, size=(n_hid, n_hid)), dtype=theano.config.floatX),
            borrow=True, name='W_co')
        self.W_hhi = theano.shared(value=np.asarray(rng.uniform(-0.01, 0.01, size=n_hid), dtype=theano.config.floatX),
                                   borrow=True, name='W_hhi')
        self.W_hhf = theano.shared(value=np.asarray(rng.uniform(-0.01, 0.01, size=n_hid), dtype=theano.config.floatX),
                                   borrow=True, name='W_hhf')
        self.W_hhc = theano.shared(value=np.asarray(rng.uniform(-0.01, 0.01, size=n_hid), dtype=theano.config.floatX),
                                   borrow=True, name='W_hhc')
        self.W_hho = theano.shared(value=np.asarray(rng.uniform(-0.01, 0.01, size=n_hid), dtype=theano.config.floatX),
                                   borrow=True, name='W_hho')
        self.bi = theano.shared(value=np.asarray(rng.uniform(-0.01, 0.01, size=n_hid), dtype=theano.config.floatX),
                                borrow=True, name='bi')
        self.bf = theano.shared(value=np.asarray(rng.uniform(-0.01, 0.01, size=n_hid), dtype=theano.config.floatX),
                                borrow=True, name='bf')
        self.bc = theano.shared(value=np.asarray(rng.uniform(-0.01, 0.01, size=n_hid), dtype=theano.config.floatX),
                                borrow=True, name='bc')
        self.bo = theano.shared(value=np.asarray(rng.uniform(-0.01, 0.01, size=n_hid), dtype=theano.config.floatX),
                                borrow=True, name='bo')
        self.c0 = theano.shared(
            value=np.asarray(rng.uniform(-0.01, 0.01, size=(vector_dim, n_hid)), dtype=theano.config.floatX),
            borrow=True, name='c0')
        self.h0 = theano.shared(
            value=np.asarray(rng.uniform(-0.01, 0.01, size=(vector_dim, n_hid)), dtype=theano.config.floatX),
            borrow=True, name='h0')

        self.params = [self.bo, self.bc, self.bf, self.bi, self.W_hho, self.W_hhc, self.W_hhf, self.W_hhi, self.W_co,
                       self.W_cf, self.W_ci, self.W_ho, self.W_hc, self.W_hf, self.W_hi,
                       self.W_xo, self.W_xc, self.W_xf, self.W_xi]

    def SequenceRecurrence(self, x):
        # Add non_sequences once no compilation and other errors.
        # For now assumption that that len(x) is an integral multiple of n_in.
        [h_s, c_s], updates = theano.scan(fn=self.OneStep,
                                          sequences=x,
                                          outputs_info=[self.h0, self.c0])
        # Call SGD on output matrix h given the input matrix x

        return h_s[0]

    def OneStep(self, x_t, h_tm1, c_tm1):
        i_t = sigmoid(x_t * self.W_xi + T.dot(h_tm1, self.W_hi) + T.dot(c_tm1, self.W_ci) + self.op(c_tm1,
                                                                                                    self.W_hhi) + self.bi)
        # print 'i_t', i_t.eval({x_t:return_x_val(0,seq_len)[0], c_tm1:self.c0.get_value(), h_tm1:self.h0.get_value()}).shape
        f_t = sigmoid(x_t * self.W_xf + T.dot(h_tm1, self.W_hf) + T.dot(c_tm1, self.W_cf) + self.op(c_tm1,
                                                                                                    self.W_hhf) + self.bf)
        # print 'f_t', f_t.eval({x_t:return_x_val(0,seq_len)[0], c_tm1:self.c0.get_value(), h_tm1:self.h0.get_value()}).shape
        c_t = f_t * c_tm1 + i_t * tanh(
            x_t * self.W_xc + T.dot(h_tm1, self.W_hc) + self.op(c_tm1, self.W_hhc) + self.bc)
        # print 'c_t', c_t.eval({x_t:return_x_val(0,seq_len)[0], c_tm1:self.c0.get_value(), h_tm1:self.h0.get_value(), f_t:np.asarray(a=np.random.uniform(-0.1, 0.1, size=(79, 200)), dtype='float32')}).shape
        o_t = sigmoid(x_t * self.W_xo + T.dot(h_tm1, self.W_ho) + T.dot(c_t, self.W_co) + self.op(c_tm1,
                                                                                                  self.W_hho) + self.bo)
        # print 'o_t', o_t.eval({x_t:return_x_val(0,seq_len)[0], c_tm1:self.c0.get_value(), h_tm1:self.h0.get_value(), c_t:np.asarray(a=np.random.uniform(-0.1, 0.1, size=(79, 200)), dtype='float32')}).shape
        h_t = o_t * tanh(c_t)
        # print 'h_t', h_t.eval({c_t:np.asarray(a=np.random.uniform(-0.1, 0.1, size=(79, 200)), dtype='float32'), o_t:np.asarray(a=np.random.uniform(-0.1, 0.1, size=(79, 200)), dtype='float32')}).shape
        # print ip.eval({ip:string_vec(doc_clean[0:seq_len])}).shape

        return h_t, c_t

    def op(self, c, W_hh):
        mul = c * W_hh
        mul = T.set_subtensor(mul[:, 1:self.n_hid], mul[:, 0:self.n_hid - 1])
        mul = T.set_subtensor(mul[:, 0:1], 0.)
        return mul


class LogReg(object):
    def __init__(self, rng, ip, n_in, n_out):
        self.W = theano.shared(
            value=np.asarray(rng.uniform(low=-0.01, high=0.01, size=n_in), dtype=theano.config.floatX), name='W',
            borrow=True)
        self.b = theano.shared(
            value=np.asarray(rng.uniform(low=-0.01, high=0.01, size=n_in), dtype=theano.config.floatX), name='b',
            borrow=True)
        self.output = T.nnet.softmax((ip * self.W + self.b).T)
        self.pred = T.argmax(self.output, axis=1)  # NOTE: INT64 made into INT32
        # self.pred is a list of the indexes of one-hot vectors(in the output).
        self.params = [self.W, self.b]

    def loss(self, y):  # target_op is indexes of letters done using string_idx
        return -T.sum(T.log(self.output.T)[y, T.arange(y.shape[0])])


# def errors(self, y):
#        return T.mean(T.neq(self.pred, y))


class LSTMNetwork:
    def __init__(self, rng, ip, n_in, n_hid, n_out):
        # create functionality for n_layers later.
        self.lstm_layer = LSTMlayer(rng, n_in, n_hid)
#        print ip.eval({ip: [string_vec(doc_clean[0:seq_len])]}).shape
#        print self.lstm_layer.SequenceRecurrence(ip).eval({ip: [string_vec(doc_clean[0:seq_len])]}).shape
        self.log_reg_layer = LogReg(rng, self.lstm_layer.SequenceRecurrence(ip), n_in, n_out)
        #       self.errors = self.log_reg_layer.errors
        self.loss = self.log_reg_layer.loss
        self.params = self.lstm_layer.params + self.log_reg_layer.params
        self.pred = self.log_reg_layer.pred


def return_x_val(indx1, indx2):
    return np.asarray(a=[string_vec(doc_clean[indx1:indx2])], dtype='float32')


def return_y_val(indx1, indx2):
    return np.asarray(a=string_idx(doc_clean[indx1:indx2]), dtype='int32')


def prep_data(train_len, valid_len, test_len, seq_len):
    # given the location of dataset form the x and y input and output sequences. make sure total_len = minibatch/seq_len*n
    itx1, itx2 = 0, train_len
    ity1, ity2 = seq_len, train_len + seq_len
    ivx1, ivx2 = train_len + seq_len, train_len + valid_len + seq_len
    ivy1, ivy2 = train_len + 2 * seq_len, train_len + valid_len + 2 * seq_len
    ittx1, ittx2 = train_len + valid_len + 2 * seq_len, train_len + valid_len + 2 * seq_len + test_len
    itty1, itty2 = train_len + valid_len + 3 * seq_len + test_len, train_len + valid_len + 3 * seq_len + 2 * test_len
    train_x = theano.shared(value=return_x_val(itx1, itx2), borrow=True, name='train_x')
    train_y = theano.shared(value=return_y_val(ity1, ity2), borrow=True, name='train_y')
    valid_x = theano.shared(value=return_x_val(ivx1, ivx2), borrow=True, name='valid_x')
    valid_y = theano.shared(value=return_y_val(ivy1, ivy2), borrow=True, name='valid_y')
    test_x = theano.shared(value=return_x_val(ittx1, ittx2), borrow=True, name='test_x')
    test_y = theano.shared(value=return_y_val(itty1, itty2), borrow=True, name='test_y')
    return train_x, train_y, valid_x, valid_y, test_x, test_y


train_x, train_y, valid_x, valid_y, test_x, test_y = prep_data(train_len, valid_len, test_len, seq_len)


def SGD(learning_rate, n_epochs):
    # x = theano.tensor.TensorType(dtype='float32', broadcastable=(True, False, False))()
    c_time_start = time.time()
    x = T.tensor3('x', dtype='float32')
    y = T.ivector('y')
    index = T.iscalar('index')
    rng = np.random.RandomState(1)
    net = LSTMNetwork(rng, x, n_in, n_hid, n_in)
    #    errors = net.errors(y)

    test_net = theano.function(inputs=[index], outputs=[net.loss(y), net.pred],
                               givens={x: test_x[:, :, index * seq_len: (index + 1) * seq_len],
                                       y: test_y[index * seq_len: (index + 1) * seq_len]})

    valid_net = theano.function(inputs=[index], outputs=[net.loss(y), net.pred],
                                givens={x: valid_x[:, :, index * seq_len: (index + 1) * seq_len],
                                        y: valid_y[index * seq_len: (index + 1) * seq_len]})

    grad_params = T.grad(cost=net.loss(y), wrt=net.params)

    updates = [(param, param - learning_rate * gparam) for param, gparam in zip(net.params, grad_params)]

    train_net = theano.function(inputs=[index], outputs=net.loss(y),
                                givens={x: train_x[:, :, index * seq_len: (index + 1) * seq_len],
                                        y: train_y[index * seq_len: (index + 1) * seq_len]}, updates=updates,
                                mode='DebugMode'
                                )
    print 'Compiling time is ', time.time()-c_time_start
    ##############
    ##TRAIN MODEL#
    ##############
    print '... Training the model!!'
    start_time = time.time()
    for epoch in range(n_epochs):
        for seqidx in range(int(train_len / seq_len)):
            train_loss = train_net(seqidx)
        valid_loss, valid_pred = valid_net(epoch)
        print 'epoch no is', epoch, 'valid loss is ', valid_loss, 'output is ', idx_string(valid_pred)
        print 'Number of seconds model has been running %%', time.time() - start_time
    #              'The output of this validation input is %f', (epoch, valid_loss, valid_errors, idx_string(net.pred))
    ##############
    ##TEST MODEL##
    ##############
    for seqtestidx in range(int(test_len / seq_len)):
        test_loss, test_pred = test_net(seqtestidx)
        print 'test loss is',test_loss ,'and prediction is', idx_string(test_pred)

# 'The output of this validation input is %f', (test_loss, test_errors )


SGD(0.001, 1)
