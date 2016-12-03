import os
import time
import string
import pickle

from sklearn.datasets import load_files
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report as clsr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import train_test_split as tts

LOAD_FILES_ROOT = "C:\\Users\\Shobha Rani\\Desktop\\shobha\\NLP_544\\Project\\TrainData"

res = load_files(LOAD_FILES_ROOT)
X = res.data
y = res.target

labels = LabelEncoder()
y = labels.fit_transform(y)

X_train, X_test, y_train, y_test = tts(X, y, test_size=0.30)

model = Pipeline([
            ('vectorizer', TfidfVectorizer(decode_error='ignore', stop_words='english')),
            ('classifier', RandomForestClassifier(n_estimators=100)),
        ])

model.fit(X_train, y_train)

print("Classification Report:\n")

y_pred = model.predict(X_test)
print(clsr(y_test, y_pred, target_names=res.target_names))

print("COMPLETE")