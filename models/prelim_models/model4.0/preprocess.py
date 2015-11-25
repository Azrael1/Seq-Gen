__author__ = 'azrael'

import theano
import numpy as np
import nltk
import time
import re

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


def tokeniize(string):
    # This regular expression handles words like "breech-loading" and converts them into "breech loading"
    # Numbers are not coverted at the moment. Makes no sense. They are not present in glove vectors.
    temp = re.sub(r'(?<=[a-zA-Z])[-](?=[a-zA-Z])', r' ', string)
    temp_ = re.sub(r'(?<=[a-zA-Z])[.](?=[a-zA-Z])', r'. ', temp)
    temp__ = re.sub(r'(?<=[a-zA-Z])[/](?=[a-zA-Z])', r' / ', temp_)

    tokens = nltk.tokenize.word_tokenize(temp__)
    return tokens


def load_vecs(vec_file, input_file, vec_dims, ignore_unks=True):
    t0 = time.time()
    low, high = -0.1, 0.1
    mappings_words, mappings_vecs, unks = [], [], []
    K = (vec_dims + 1)
    vec_obj = open(vec_file)
    ip_obj = open(input_file)
    vec_toks = vec_obj.read().lower().split()
    # The following line first decodes the bytes into unicode and then encodes the unicode into ascii. The values
    # outside ascii are ignored. 0.1% data loss occurs.
    for i in range(len(vec_toks) / K):
        mappings_words.append(vec_toks[K * i])
        mappings_vecs.append(np.asarray(vec_toks[(K * i + 1): K * (i + 1)], theano.config.floatX))

    ip_text = ip_obj.read().decode('utf-8').encode('ascii', 'ignore').lower()

    ip_toks = tokeniize(ip_text)
    ip_dict = list(set(ip_toks))

    if ignore_unks:
        print 'The unknown tokens are going to be ignored.'
        # Ignoring the unknown words. Replacing them by blank.
        for word in ip_dict:
            if word not in mappings_words:
                unks.append(word)

        ip_toks[:] = [x for x in ip_toks if not x in unks]
        ip_dict = set(list(ip_toks))
        ip_text = " ".join(ip_toks)

    else:
        j = 0
        for word in ip_dict:
            if word not in mappings_words:
                j += 1
                vec_word = np.random.uniform(low, high, vec_dims)
                mappings_vecs.append(vec_word)
                mappings_words.append(word)

        print 'The number of words that are present in the input dictionary but not in glove vectors are ', j
        print 'The ratio is ', j / float(len(ip_dict))
    req_words, req_vecs = [], []

    # Normalize the input matrix by making mean 0 and dividing by the Standard Deviation.
    mappings_vecs = (mappings_vecs - np.mean(mappings_vecs))/np.std(mappings_vecs)


    # Make only the words which are used in the document shared variables. Decreases memory consumption.
    for word in ip_dict:
        idx = mappings_words.index(word)
        req_words.append(word)
        temp_vec = mappings_vecs[idx]
        req_vecs.append(temp_vec)
    req_vecs = np.array(req_vecs)
    vec_obj.close()
    ip_obj.close()
    print 'The time required in preprocessing the document is', time.time()- t0
    return req_words, \
           theano.shared(np.asarray(req_vecs, theano.config.floatX), borrow=True, name='mappings_vecs'), ip_text


def sentence_vec(sentence, np_vecs, mappings_words):
    """
    Convert sentence to list of vectors.
    :param sentence: List of strings
    :return:List of vectors.
    """
    list_vec = []
    for word in sentence:
        list_vec.append(np_vecs[mappings_words.index(word)])
    # Done in order to make shape (vec_dims, n_words)
    return np.asarray(list_vec, theano.config.floatX).T


if __name__ == "__main__":
    t0 = time.time()
    vec_file = '/home/azrael/Documents/nn/seq_gen/data/vectors.6B.50d.txt'
    data_file = '/home/azrael/Documents/nn/seq_gen/data/smallwikisample.txt'
    vec_dims = 50
    load_vecs(vec_file, data_file, vec_dims)
    print 'Time taken by whole process is in minutes ', (time.time() - t0) / 60.0
