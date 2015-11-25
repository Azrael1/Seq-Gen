__author__ = 'azrael'

import theano
import numpy as np
import nltk
import time
import re

def tokeniize(string):
    # This regular expression handles words like "breech-loading" and converts them into "breech loading"
    # Numbers are not converted at the moment. Makes no sense. They are not present in glove vectors.
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
