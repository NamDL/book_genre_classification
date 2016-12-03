__author__ = 'supriyaanand'

import gensim
import logging
import nltk
import os
import sys
import numpy as np
from nltk.tokenize import WordPunctTokenizer
import string

def iterate_through_docs(topdir, stoplist):
    '''
    :param topdir: top level directory containing individual files each housing a document
    :param stoplist: stopwords needed to be removed from the data
    :return: generator object to iterate through the dataset
    '''
    word_punct_tokenizer = WordPunctTokenizer()
    for fn in os.listdir(topdir):
        if '.txt' not in fn:
            continue
        fh = open(os.path.join(topdir, fn), 'r')
        text = fh.readlines()
        fh.close()
        for line in text:
            line = line.strip()
            line = word_punct_tokenizer.tokenize(line)
            yield [x.lower() for x in line if x.lower() not in stoplist]

class corpus_cl(object):
    '''
    Defines an iterator to run through the dataset
    '''
    def __init__(self, topdir, stoplist):
        self.topdir = topdir
        self.stoplist = stoplist
        self.dictionary = gensim.corpora.Dictionary(iterate_through_docs(topdir, stoplist))
        self.dictionary.filter_extremes(no_below=5)

    def __iter__(self):
        for tokens in iterate_through_docs(self.topdir, self.stoplist):
            yield self.dictionary.doc2bow(tokens)


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)

# Fix seed for Numpy random generator
FIXED_SEED = 42  # the answer to life, universe and everything

# Before training, set the fixed seed for numpy
np.random.seed(FIXED_SEED)

stoplist = nltk.corpus.stopwords.words("english")
stoplist.extend(list(string.punctuation)) # add punctuation to stop word list
stoplist = set(stoplist) # convert to set for efficient querying

corpus = corpus_cl(sys.argv[1], stoplist)

corpus.dictionary.save("corpus.dict")
gensim.corpora.MmCorpus.serialize("corpus.mm", corpus)

dictionary = gensim.corpora.Dictionary.load("./corpus.dict")

NUM_TOPICS = 5 # determine 5 topical word sets from each genre

eta = None

corpus = gensim.corpora.MmCorpus("./corpus.mm")

# Project to LDA space
lda_model = gensim.models.LdaModel(corpus, id2word=dictionary, num_topics=NUM_TOPICS, passes=5, eta=eta)

print("Topics word distribution")
lda_model.print_topics(NUM_TOPICS, num_words=50)


