__author__ = 'azrael'
import theano
import theano.tensor as T

assert_op = T.opt.Assert()

def CMulTensors(tensor1, tensor2):
    """
    Supports multiplication of only 2 matrix at the moment. Will suffice.
    :param tensor1: Theano.tensor.tensorVariable
    :param tensor2: same
    :return: same
    """
    # Implement check that the tensors hold the same number of elements.
    tensor1_ = assert_op(tensor1, tensor1.shape[0]*tensor1.shape[1]==tensor2.shape[0]*tensor2.shape[1])
    # Reshape into shape of tensor1
    tensor2_ = T.reshape(tensor2, (tensor1_.shape[0], tensor1_.shape[1]), ndim =2)
    return tensor1_*tensor2_


def CAddTensors(tensor1, tensor2):
    """
    Supports addition of only 2 matrix at the moment. Will suffice. To add more, just add them 2 at a time.
    :param tensor1: Theano.tensor.tensorVariable
    :param tensor2: same
    :return: same
    """
    # Implement check that the tensors hold the same number of elements.
    tensor1_ = assert_op(tensor1, tensor1.shape[0]*tensor1.shape[1]==tensor2.shape[0]*tensor2.shape[1])
    # Reshape into shape of tensor1
    tensor2_ = T.reshape(tensor2, (tensor1_.shape[0], tensor1_.shape[1]), ndim =2)
    return tensor1_+tensor2_

