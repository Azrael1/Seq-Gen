__author__ = 'azrael'

import theano
import numpy as np
import nltk
import time

"""
SOME NOTES ABOUT USING NUMPY:

Used numpy as shown in preprocess.py. Didn't work out. Memory usage too much. 10x factor slower.
Another note is that when process ends, does not always free memory. Why is this? (numpy)
Results:
azrael@azrael-G551JK:~/Documents/nn/code/prelim2.0$ python preprocess1.py (without numpy)
Using gpu device 0: GeForce GTX 850M
Number of Vec tokens are  20400000
Length of input dict is  3109
Length of word vecs is  400000
Length of mappings vec is  400000
Length of mappings words is  400000
Completed insertion of new words in time 23.1122250557
Final length of mappings words is  400735
Time taken by whole process is in minutes  0.447935120265

azrael@azrael-G551JK:~/Documents/nn/code/prelim2.0$ python preprocess.py (with numpy)
Using gpu device 0: GeForce GTX 850M
Number of Vec tokens are  20400000
Length of input dict is  3109
Length of word vecs is  400000
Length of mappings vec is  400000
Length of mappings words is  400000
Completed insertion of new words in time 229.8882792
Final length of mappings words is  400735
Size of word_vecs in MB is 25
Size of mappings vec in MB is  641.176 (not sure about using nbytes check it properly)
Size of mappings words in MB is  27
Time taken by whole process is in minutes  3.92276798487

These results are using 50d word vecs and a 61KB text file. Maybe using a bigger file will have some changes?
"""

def tokenize(string):
    pattern = r'''(?x)    # set flag to allow verbose regexps
         ([A-Z]\.)+        # abbreviations, e.g. U.S.A.
       | \w+(-\w+)*        # words with optional internal hyphens
       | \$?\d+(\.\d+)?%?  # currency and percentages, e.g. $12.40, 82%
       | \.\.\.            # ellipsis
       | [][.,;"'?():-_`]  # these are separate tokens; includes ], [
     '''
    str_tokens = nltk.regexp_tokenize(string, pattern)
    return str_tokens

def load_vecs(vec_file, input_file, vec_dims):
    word_vecs = {}
    mappings_words = []
    mappings_vecs = []
    K = (vec_dims + 1)
    vec_object = open(vec_file)
    input_object = open(input_file)
    vec_tokens = vec_object.read().lower().split()
    input_tokens = tokenize(input_object.read().lower())
    input_dict = list(set(input_tokens))
    for i in range(len(vec_tokens)/K):
        word_vecs[vec_tokens[K*i]] = vec_tokens[(K*i+1): K*(i+1)]
        mappings_words.append(vec_tokens[K*i])
        mappings_vecs.append(vec_tokens[K*i+1: K*(i+1)])
    for word in input_dict:
        if word not in mappings_words:
            vec_word = np.random.uniform(-0.05, 0.05, vec_dims)
            mappings_vecs.append(vec_word)
            mappings_words.append(word)
            word_vecs[word] = vec_word

    req_words = []
    req_vecs = []
    req_wordvecs = {}
    for word in input_dict:
        idx = mappings_words.index(word)
        req_words.append(word)
        temp_vec = mappings_vecs[idx]
        req_vecs.append(temp_vec)
        req_wordvecs[word] = temp_vec
    vec_object.close()
    input_object.close()
    return req_wordvecs, req_words, \
           theano.shared(np.asarray(req_vecs, theano.config.floatX), borrow=True, name='mappings_vecs')


def sentence_vec(sentence, dict_word_vec):
    """
    Convert sentence to list of vectors.
    :param sentence: List of strings
    :return:List of vectors.
    """
    list_vec = []
    for word in sentence:
        list_vec.append(dict_word_vec[word])
    # Done in order to make shape (vec_dims, n_words)
    return np.asarray(list_vec, theano.config.floatX).T

if __name__ == "__main__":
    t0 = time.time()
    vec_file = '/home/azrael/Documents/nn/code/vectors.6B.50d.txt'
    data_file = '/home/azrael/Documents/nn/code/wikisample.txt'
    vec_dims = 50
    load_vecs(vec_file, data_file, vec_dims)
    print 'Time taken by whole process is in minutes ', (time.time()-t0)/60.0
