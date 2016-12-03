import os
import string
import pickle
from sklearn.datasets import load_files
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report as clsr
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords as sw
from nltk.corpus import wordnet as wn
from nltk import wordpunct_tokenize
from nltk import WordNetLemmatizer
from nltk import sent_tokenize
from nltk import pos_tag
from sklearn.base import BaseEstimator, TransformerMixin


def identity(arg):
    return arg

class NLTKPreprocessor(BaseEstimator, TransformerMixin):

    
    def __init__(self, stopwords=None, punct=None,
                 lower=True, strip=True):
        self.lower      = lower
        self.strip      = strip
        self.stopwords  = stopwords or set(sw.words('english'))
        self.punct      = punct or set(string.punctuation)
        self.lemmatizer = WordNetLemmatizer()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [
            list(self.tokenize(str(doc))) for doc in X
        ]

    def tokenize(self, document):
        # Break the document into sentences
        for sent in sent_tokenize(document):
            # Break the sentence into part of speech tagged tokens
            for token, tag in pos_tag(wordpunct_tokenize(sent)):
                # Apply preprocessing to the token
                token = token.lower() if self.lower else token
                token = token.strip() if self.strip else token
                token = token.strip('_') if self.strip else token
                token = token.strip('*') if self.strip else token

                # If stopword, ignore token and continue
                if token in self.stopwords:
                    continue

                # If punctuation, ignore token and continue
                if all(char in self.punct for char in token):
                    continue

                # Lemmatize the token and yield
                lemma = self.lemmatize(token, tag)
                yield lemma

    def lemmatize(self, token, tag):
        tag = {
            'N': wn.NOUN,
            'V': wn.VERB,
            'R': wn.ADV,
            'J': wn.ADJ
        }.get(tag[0], wn.NOUN)

        return self.lemmatizer.lemmatize(token, tag)


TRAIN_DATA = "C:\\Users\\Shobha Rani\\Desktop\\shobha\\NLP_544\\Project\\Data\\Train"
TEST_DATA = "C:\\Users\\Shobha Rani\\Desktop\\shobha\\NLP_544\\Project\\Data\\Test"

train = load_files(TRAIN_DATA)
X_train = train.data
y_train = train.target

test = load_files(TEST_DATA)
X_test = test.data
y_test = test.target

model = Pipeline([
           ('preprocessor', NLTKPreprocessor()),
            ('vectorizer', TfidfVectorizer(decode_error='ignore', tokenizer=identity, preprocessor=None, lowercase=False, ngram_range=(1, 2))),
            ('classifier', RandomForestClassifier(n_estimators=100)),
        ])

model.fit(X_train, y_train)

print("Classification Report:\n")

y_pred = model.predict(X_test)
print(clsr(y_test, y_pred, target_names=test.target_names))

print("COMPLETE")