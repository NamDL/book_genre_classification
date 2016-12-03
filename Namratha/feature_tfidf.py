
import os, sys, random, numpy, re, copy
from sklearn import datasets
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.datasets import fetch_20newsgroups
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer

wordSet={}
testDataList=[]
testClassList=[]
trainClassList=[]
trainDataList=[]
fileTrainList={}
fileTestList={}
traincount=0
testcount=0
countElse=0
n_samples = 2000
n_features = 100
n_topics = 10
n_top_words = 20

def getData(fileList,testORtrain):
    global trainClassList
    global testClassList
    global testDataList
    global trainDataList
    global countElse

    wordnet_lemmatizer = WordNetLemmatizer()
    porter_stemmer = PorterStemmer()
    dataList=[]
    classList=[]
    content=""
    lemmas=[]
    
    fileItems=list(fileList.items())
    random.shuffle(fileItems)
    
    for key,value in fileItems:
        genre=key
        filePath=value
        if "Romance" in genre:
            classList.append("Romance")
        elif "Humor" in genre:
            classList.append("Humor")
        elif "Religion" in genre:
            classList.append("Religion")
        elif "Mystery" in genre:
            classList.append("Mystery")
        elif "SciFi" in genre:
            classList.append("Science Fi")
        else:
            classList.append("***********"+str(countElse))
            
        with open(filePath, 'r',encoding="latin1") as content_file:
            content = content_file.read()
        '''words=content.split(" ")        
        for word in words:
            lemmas.append(porter_stemmer.stem(word))'''
            
        dataList.append(content)
    
    if testORtrain=="test":
        countElse+=1
        testClassList=copy.deepcopy(classList)
        testDataList=copy.deepcopy(dataList)
    if testORtrain=="train":
        trainClassList=copy.deepcopy(classList)
        trainDataList=copy.deepcopy(dataList)

    '''for f, b in zip(classList, dataList):
        print(f, re.sub('[^\x00-\x7F]','', b))'''

def analyzeData():
    #tfidf_vectorizer = TfidfVectorizer(max_features=n_features,analyzer='word',stop_words='english',max_df=0.98, ngram_range=(1,2), min_df=0.01)
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,2))
    tfidfTrain = tfidf_vectorizer.fit_transform(trainDataList)
    classArray=numpy.asarray(trainClassList)
    model=LogisticRegression()
    model.fit(tfidfTrain, classArray) 
    tfidfTest= tfidf_vectorizer.transform(testDataList)    
    predicted = model.predict(tfidfTest)
    print("\n                       Stop Word Removal\n")
    print("")
    print(metrics.classification_report(testClassList, predicted))


trainRoot_dir=sys.argv[1]
testRoot_dir=sys.argv[2]

for roota, subdirsa, filesa in os.walk(trainRoot_dir):    
    for subdira in subdirsa:
        for filenamea in os.listdir(os.path.join(roota, subdira)):
            fileTrainPath=os.path.join(roota,subdira,filenamea)
            fileTrainList[subdira+""+str(traincount)]=fileTrainPath
            traincount+=1
            
for roots, subdirss, filess in os.walk(testRoot_dir):    
    for subdirs in subdirss:
        for filenames in os.listdir(os.path.join(roots, subdirs)):
            fileTestPath=os.path.join(roots,subdirs,filenames)
            fileTestList[subdirs+""+str(testcount)]=fileTestPath
            testcount+=1
print("\n")            
print("Traing Samples "+str(traincount))
print("Testing Samples "+str(testcount))
getData(fileTrainList,"train")
getData(fileTestList,"test")
analyzeData()

