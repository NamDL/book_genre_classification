
import os, sys, random, numpy, re, copy
from sklearn import datasets
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.datasets import fetch_20newsgroups
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import wordnet as wn
import nltk, gensim
from nltk.corpus import stopwords

testDataList=[]
testClassList=[]
trainClassList=[]
trainDataList=[]
fileTrainList={}
fileTestList={}
traincount=0
testcount=0
countElse=0
testCount=0
model_path="D:\\GoogleNews-vectors-negative300.bin"
model=gensim.models.word2vec.Word2Vec.load_word2vec_format(model_path, binary=True)
model.init_sims(replace=True)
engDict=set(w.lower() for w in nltk.corpus.words.words())


def getWordnetValue(content,genre):
    wordList=[]
    stop = set(stopwords.words('english'))
    words=[i for i in content.lower().split() if i not in stop]
    mysScore=0
    romScore=0
    sciScore=0
    relScore=0
    humScore=0    
    for word in words:
        werd=re.sub('[^A-Za-z0-9]+', '', word)
        if werd in model.vocab and werd:            
            mysScore+=model.similarity('thriller',werd)
            romScore+=model.similarity('love',werd)
            sciScore+=model.similarity('fantasy',werd)
            relScore+=model.similarity('religion',werd)
            humScore+=model.similarity('funny',werd)
    wordList.append(mysScore)
    wordList.append(romScore)
    wordList.append(sciScore)
    wordList.append(relScore)
    wordList.append(humScore)
    return wordList   
        

def getData(fileList,testORtrain):
    global trainClassList
    global testClassList
    global testDataList
    global trainDataList
    global countElse
    global testCount

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
        elif "Humour" in genre:
            classList.append("Humor")
        elif "Religion" in genre:
            classList.append("Religion")
        elif "Mystery" in genre:
            classList.append("Mystery")
        elif "Sci-fi" in genre:
            classList.append("Science Fi")
        else:
            classList.append("***********"+str(countElse))
            
        with open(filePath, 'r',encoding="latin1") as content_file:
            content = content_file.read()   
        dataList.append(getWordnetValue(content,genre))
        
    if testORtrain=="test":
        countElse+=1
        testClassList=copy.deepcopy(classList)
        testDataList=copy.deepcopy(dataList)
    if testORtrain=="train":
        trainClassList=copy.deepcopy(classList)
        trainDataList=copy.deepcopy(dataList)

def analyzeData():
    classArray=numpy.asarray(trainClassList)
    trainDataArray=numpy.asarray(trainDataList)
    model=LogisticRegression()
    model.fit(trainDataArray, classArray)
    testDataArray=numpy.asarray(testDataList)
    print(testDataList)
    predicted = model.predict(testDataArray)
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
#posTagging()
analyzeData()


