__author__ = 'supriyaanand'

import string
import operator
import os
import sys
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
import sklearn.datasets
from sklearn import metrics


class doc_sim_calc:
    '''
    Classify books into various genres using cosine similarity between document vectors and vectors from a generic
    document most representative of a genre
    '''
    def __init__(self):
        '''
        Set up preprocessors
        :return: self
        '''
        self.stemmer = nltk.stem.porter.PorterStemmer()
        self.strip_punct_dict = dict((ord(char), None) for char in string.punctuation)
        self.vectorizer = TfidfVectorizer(tokenizer=self.preprocess, stop_words='english')
        self.data = None

    def stem_tokens(self, tokens):
        '''
        :param tokens: list of tokens composing the document
        :return: list of stemmed tokens using the Porter stemmer
        '''
        return [self.stemmer.stem(tok) for tok in tokens]

    def preprocess(self, text):
        '''
        :param text: document to be preprocessed
        :return: list of stemmed tokens from the document with punctuation stripped
        '''
        return self.stem_tokens(
            nltk.word_tokenize(text.lower().translate(self.strip_punct_dict)))

    def cosine_sim(self, doc_a, doc_b):
        '''
        Calculate cosine similarity between the two given documents
        :param doc_a: document representing a genre
        :param doc_b: test document
        :return: cosine similarity score
        '''
        vector_mat = self.vectorizer.fit_transform([doc_a, doc_b])
        return ((vector_mat * vector_mat.T).A)[0, 1]

    def read_test_data(self, data_dir):
        '''
        Load test data
        :param data_dir: direcory holding the test data hosted in folders bearing class names
        :return: self
        '''
        self.data = sklearn.datasets.load_files(data_dir, encoding='latin1')

    def sim_based_predictions(self, genre_based_doc_dir):
        '''
        Computes evaluation metrics post similarity comparisons with each document representative of a genre
        :param genre_based_doc_dir: Directory hosting documents representative of each genre
        :return: self
        '''
        class_docs = {} # dict to record class representative documents

        # initialize classes
        classes = self.data.target_names

        for cl in classes:
            with open(os.path.join(genre_based_doc_dir, cl + "_doc.txt")) as fh:
                class_docs[cl] = " ".join(fh.readlines())

        pred = [] # list to contain predictions for each test data point

        for dp in self.data.data:
            score_class = {}
            for cl in classes:
                score_class[cl] = self.cosine_sim(class_docs[cl], dp)
            sorted_scores = sorted(score_class.items(), key=operator.itemgetter(1))
            pred.append(sorted_scores[-1][0])

        y_test = [self.data.target_names[x] for x in self.data.target] # record true labels for the dataset

        print(metrics.classification_report(y_test, pred, target_names=self.data.target_names))
        print(metrics.confusion_matrix(y_test, pred))


if __name__ == '__main__':
    # arg1 should be the directory containing the documents to be clustered
    # arg2 should be the directory containing the representative documents for each genre titled 'genre_doc.txt'
    # where genre should have the same labels as the test directory contents
    doc_sim_clf = doc_sim_calc()
    doc_sim_clf.read_test_data(sys.argv[1])
    doc_sim_clf.sim_based_predictions(sys.argv[2])
