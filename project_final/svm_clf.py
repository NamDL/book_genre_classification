import sys
import string
from sklearn.feature_extraction.text import TfidfVectorizer
import sklearn.datasets
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
import nltk
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


class svm_classifier:
    def __init__(self, train_data_dir, test_data_dir):
        '''
        Sets up data vectors
        :param train_data_dir: direcory holding the training data hosted in folders bearing class names
        :param test_data_dir: direcory holding the test data hosted in folders bearing class names
        :return: self
        '''
        # setup pre-processors
        self.stemmer = nltk.stem.porter.PorterStemmer()
        self.strip_punct_dict = dict((ord(char), None) for char in string.punctuation)
        self.vectorizer = TfidfVectorizer(tokenizer=self.preprocessor, stop_words='english')
        # self.vectorizer = TfidfVectorizer() # baseline cacl with no preprocessing
        self.train_data = sklearn.datasets.load_files(train_data_dir, encoding='latin1')
        self.test_data = sklearn.datasets.load_files(test_data_dir, encoding='latin1')
        self.X_train = self.train_data.data
        self.X_test = self.test_data.data
        self.y_train = self.train_data.target
        self.y_test = self.test_data.target

    def stem_tokens(self, tokens):
        '''
        :param tokens: list of tokens composing the document
        :return: list of stemmed tokens using the Porter stemmer
        '''
        return [self.stemmer.stem(tok) for tok in tokens]

    def preprocessor(self, text):
        '''
        :param text: document to be preprocessed
        :return: list of stemmed tokens from the document with punctuation stripped
        '''
        return self.stem_tokens(
            nltk.word_tokenize(text.lower().translate(self.strip_punct_dict)))

    def pre_process_data(self):
        '''
        :return: self, sets up pre-processed data vectors
        '''
        self.vectors = self.vectorizer.fit_transform(self.X_train)
        self.preprocessor = preprocessing.Normalizer().fit(self.vectors)
        self.X_normalized = self.preprocessor.transform(self.vectors)
        self.vectors_test = self.vectorizer.transform(self.X_test)
        self.X_test_scaled = self.preprocessor.transform(self.vectors_test)

    def baseline_classifier_metrics(self):
        '''
        Calculates baseline metrics of precision, recall and F1 score on the test data set using Multinomial Naive Bayes
        :return: self
        '''
        clf = MultinomialNB(alpha=.01)
        clf.fit(self.vectors, self.y_train)
        pred = clf.predict(self.vectors_test)
        print(metrics.classification_report(self.y_test, pred,
                                            target_names=self.train_data.target_names))

    def svm_classifier_metrics(self):
        '''
        SVM classifier training and evaluation
        :return: self
        '''
        self.clf = svm.SVC(decision_function_shape='ovr', kernel='rbf', C=3,
                           gamma=0.2)
        self.clf.fit(self.vectors, self.y_train)
        pred = self.clf.predict(self.vectors_test)
        print(metrics.classification_report(self.y_test, pred,
                                            target_names=self.train_data.target_names))
        print(metrics.confusion_matrix(self.y_test, pred))

    def grid_search_params(self):
        '''
        Determine parameters by cross-validation
        :return: self
        '''
        self.clf = svm.SVC(decision_function_shape='ovr')
        tuned_parameters = {'kernel': ['rbf'], 'gamma': [1e-3, 1e-2],
                             'C': [1, 3, 7, 9]}

        print(self.clf.get_params().keys())
        gs_clf = GridSearchCV(self.clf, tuned_parameters, n_jobs=-1)
        gs_clf = gs_clf.fit(self.vectors, self.train_data.target)

        print("best score", gs_clf.best_score_)
        for param_name in sorted(tuned_parameters.keys()):
            print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))

def k_means_clusters(data):
    '''
    K means clustering on the given dataset with a visualization using matplotlib
    :param data:
    :return: self
    '''
    # reduce high dimensional data to 2 dimensions using PCA
    reduced_data = PCA(n_components=2).fit_transform(data.toarray())
    kmeans = KMeans(init='k-means++', n_clusters=5, n_init=10) # Set cluster size to 5, num of iterations to 10
    kmeans.fit(reduced_data)

    # Step size of the mesh. Decrease to increase the quality of the VQ
    h = .02  # point in the mesh [x_min, x_max]x[y_min, y_max].

    # Plot the decision boundary. For that, we will assign a color to each
    x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
    y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Obtain labels for each point in mesh
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

    # Color plot
    Z = Z.reshape(xx.shape)
    plt.figure(1)
    plt.clf()
    plt.imshow(Z, interpolation='nearest',
               extent=(xx.min(), xx.max(), yy.min(), yy.max()),
               cmap=plt.cm.Paired,
               aspect='auto', origin='lower')

    plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
    # Plot the centroids
    centroids = kmeans.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=169, linewidths=3,
                color='w', zorder=10)
    plt.title('K-means clustering on the dataset (PCA-reduced data)\n'
              'Centroids are marked with white cross')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.show()

if __name__ == '__main__':
    # arg1 should be the directory containing the training dataset
    # arg2 should be the directory containing the test dataset
    clf = svm_classifier(sys.argv[1], sys.argv[2]) # training and test data directories respectively
    clf.pre_process_data()
    clf.baseline_classifier_metrics()
    clf.svm_classifier_metrics()
    # clf.grid_search_params()
    # k_means_clusters(clf.vectors_test)
