__author__ = 'azrael'
# Conditioned for 50 dimensions word vectors.


def load_word_vecs(file):
    """
    Load glove word vectors from this file
    :param file: String : file location
    :return: Dictionary of words and their corresponding vectors
    """
    dict_word_vec = {}
    temp_file = open(file).read().split()
    for i in range(len(file)/51):
        dict_word_vec[temp_file[51*i]] = temp_file[51*i + 1: 51*1 + 51]

    return dict_word_vec
