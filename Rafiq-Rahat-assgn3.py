from __future__ import division, unicode_literals
import math
from textblob import TextBlob as tb
import sys  
import re
from collections import defaultdict
import nltk
from sklearn.feature_extraction import DictVectorizer
from numpy import array


from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score




from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.svm import SVC, LinearSVC
from sklearn import cross_validation
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier


from nltk.stem.lancaster import LancasterStemmer
from textblob import TextBlob as tb
from nltk import word_tokenize

reload(sys)  
sys.setdefaultencoding('utf8')

numOfTopWords = 3
st = LancasterStemmer()


idfWordHash = {}
bigramHash = {}

def WriteWordsIDFNcontainingToFile(trueList,decepTiveList):
    wordHash = {}
    for true in trueList:
        true = true.strip()
        words = true.split(" ")
        for word in words:
            wordHash[word] = 1 
        
    for deceptive in decepTiveList:
        deceptive = deceptive.split()
        words = true.split(" ")
        for word in words:
            wordHash[word] = 1
    
    f = open("words_idf_values.txt","w")
    g = open("words_ncontaining_values.txt","w")
    
    for word in wordHash.keys():
        print word
        f.write(str(word)+","+str(idf(word, trueList))+","+str(idf(word, decepTiveList))+"\n")
        g.write(str(word)+","+str(n_containing(word, trueList))+","+str(n_containing(word, decepTiveList))+"\n")
    
    f.close()
    g.close()


def tf(word, blob):
    return blob.words.count(word) / len(blob.words)

def n_containing(word, bloblist):
    return sum(1 for blob in bloblist if word in blob)

def idf(word, bloblist):
    return math.log(len(bloblist) / (1 + n_containing(word, bloblist)))

def tfidf(word, blob, bloblist):
    return tf(word, blob) * idf(word, bloblist)


def getPOSTags(featureDictionary,line):
    text = word_tokenize(str(line))
    taggedwords = nltk.pos_tag(text)
    for wordTuple in taggedwords:
        if featureDictionary[wordTuple[1]] is None:
            featureDictionary[wordTuple[1]] = 1
        else:
            featureDictionary[wordTuple[1]] = featureDictionary[wordTuple[1]] + 1
            


def generateIDFWordHash():
    f = open("words_idf_values.txt","r")
    for singleLine in f:
        values = singleLine.split(",")
        word = values[0]
        trueIDF = values[1]
        deceptiveIDF = values[2]
        if abs(float(trueIDF)-float(deceptiveIDF)) > 0.0:
            idfWordHash[word] = 1 

def getTopIDFWords(featureDictionary,line):        
    words = line.split(" ")
    for word in words:
        if word in idfWordHash.keys():
            if featureDictionary[word] is None:
                featureDictionary[word] = 1
            else:
                featureDictionary[word] = featureDictionary[word] + 1
                

def generatenContainingWordHash():
    f = open("words_ncontaining_values.txt","r")
    for singleLine in f:
        values = singleLine.split(",")
        word = values[0]
        trueIDF = values[1]
        deceptiveIDF = values[2]
        if abs(float(trueIDF)-float(deceptiveIDF)) > 4:
            idfWordHash[word] = 1 

def getTopnContainingWords(featureDictionary,line):        
    words = line.split(" ")
    for word in words:
        if word in idfWordHash.keys():
            featureDictionary[word] = 1    
    
    

def biGrams(featureDictionary,line):
    tokens = nltk.word_tokenize(str(line))
    bgs = nltk.bigrams(tokens)
    fdist = nltk.FreqDist(bgs)
    for k,v in fdist.items():
        bigram = str(k)
        bigramHash[bigram] = 1
        if featureDictionary[bigram] is None:
            featureDictionary[bigram] = int(v)
        else:
            featureDictionary[bigram] = featureDictionary[bigram] + int(v)



def bareUniGrams(featureDictionary,line):
    words = line.split(" ")
    for word in words:
        if featureDictionary[word] is None:
            featureDictionary[word] = 1
        else:
            featureDictionary[word] = featureDictionary[word] + 1



def uniGramsWithStems(featureDictionary,line):
    words = line.split(" ")
    for word in words:
        word = st.stem(word)
        if featureDictionary[word] is None:
            featureDictionary[word] = 1
        else:
            featureDictionary[word] = featureDictionary[word] + 1




def readInputFiles(trueFileName,deceptiveFileName):
    trueList =[]
    deceptiveList = []
    f = open(trueFileName,"r")
    for line in f:
        line = line.strip()
        line = line[8:]
        line = line.lower()
        line = re.sub(r'\W+', ' ', line)
        trueList.append(tb(line))
    f.close()
    
    f = open(deceptiveFileName,"r")
    for line in f:
        line = line.strip()
        line = line[8:]
        line = line.lower()
        line = re.sub(r'\W+', ' ', line)
        deceptiveList.append(tb(line))
    f.close()
    
    return (trueList,deceptiveList)
        




def startAllClassifiersWithCrossValidation(totalData,totalLabel):
    print "classifier work is starting"
    print "length of total data  set :"+str(len(totalData))
    print "length of total label set :"+str(len(totalLabel))
    
    v = DictVectorizer(sparse=True)
    X = v.fit_transform(totalData)
    X = X.todense()
    Y = array(totalLabel)
    print "shape of X: "+str(X.shape)
    print "shape of Y: "+str(Y.shape)
    
    classifierList = []
    classifierList.append((Perceptron(n_iter=150), "Perceptron"))
    classifierList.append((RandomForestClassifier(n_estimators=100), "Random forest"))
    classifierList.append((MultinomialNB(),"NaiveBayes"))
    
    X_training = X[0:385]
    Y_training = Y[0:385]
    X_test = X[385:]
    Y_test = Y[385:]
    
    f = open("allClassifier_accuracy_ncontaining.txt","w")
    
    for clf in classifierList:
        scores = cross_validation.cross_val_score(clf[0], X, Y, cv=10)
        print clf[1]
        print scores.mean()
        f.write(str(clf[1])+" "+str(scores.mean())+"\n")
        print "_________________________________"
    f.close()



def startAllClassifiers(totalData,totalLabel,testData):
    
    
    print "classifier work is starting"
    print "length of total data  set :"+str(len(totalData))
    print "length of total label set :"+str(len(totalLabel))
    
    v = DictVectorizer(sparse=True)
    X = v.fit_transform(totalData)
    X = X.todense()
    Y = array(totalLabel)
    print "shape of X: "+str(X.shape)
    print "shape of Y: "+str(Y.shape)
    
    classifierList = []
    #classifierList.append((Perceptron(n_iter=150), "Perceptron"))
    #random forest performed the best
    classifierList.append((RandomForestClassifier(n_estimators=100), "Random forest"))
    #classifierList.append((MultinomialNB(),"NaiveBayes"))
    f = open("Rafiq-Rahat-assgn3-out.txt","w")
    for test in testData:
        temp = []
        temp.append(test[1])
        X_test = v.fit_transform(temp)
        X_test = X_test.todense()
        for clf in classifierList:
            clf[0].fit(X, Y)
            res = None
            if  clf[0].predict(X_test)[0] == 1:
                res = "T"
            else:
                res = "F"
            f.write(str(test[0])+"\t"+str(res)+"\n")
            print test[0]
            print res
        
    f.close()
    
    
    
    


def featureExtraction(trueList,deceptiveList):
    
    index = 0
    
    trainingData = []
    trainingLabel = []
    
    while index < len(trueList):
        featureDictionary = defaultdict(int)
        #bareUniGrams(featureDictionary,trueList[index])
        #uniGramsWithStems(featureDictionary,trueList[index])
        #biGrams(featureDictionary,trueList[index])
        #getPOSTags(featureDictionary, trueList[index])
        #getTopIDFWords(featureDictionary, trueList[index])
        getTopnContainingWords(featureDictionary, trueList[index])
        trainingData.append(featureDictionary)
        trainingLabel.append(1)
        
        
        
        featureDictionary = defaultdict(int)
        #bareUniGrams(featureDictionary,deceptiveList[index])
        #uniGramsWithStems(featureDictionary,deceptiveList[index])
        #biGrams(featureDictionary,deceptiveList[index])
        #getPOSTags(featureDictionary, deceptiveList[index])
        #getTopIDFWords(featureDictionary, deceptiveList[index])
        getTopnContainingWords(featureDictionary, deceptiveList[index])
        trainingData.append(featureDictionary)
        trainingLabel.append(0)
        
        
        
        index = index + 1
    return (trainingData,trainingLabel)


def readTestFile(filename):
    testData = []
    f = open(filename,"r")
    for line in f:
        featureDictionary = defaultdict(int)
        line = line.strip()
        reviewid = line[:8]
        line = line[8:]
        line = line.lower()
        line = re.sub(r'\W+', ' ', line)
        for key in idfWordHash.keys():
            featureDictionary[key] = 0
        words = line.split(" ")
        for word in words:
            if word in idfWordHash.keys():
                featureDictionary[word] = 1
        tup = (reviewid,featureDictionary)
        testData.append(tup)
        
    f.close()
    return testData



  

trueFileName = "hotelT-train.txt"
deceptiveFileName = "hotelF-train.txt"  

trueList,decepTiveList = readInputFiles(trueFileName, deceptiveFileName)

generatenContainingWordHash()
totalData, totalLabel = featureExtraction(trueList,decepTiveList)

testData = readTestFile("HW2-testset.txt")





startAllClassifiers(totalData,totalLabel,testData)

#startAllClassifiersWithCrossValidation(totalData,totalLabel)
#WriteWordsIDFNcontainingToFile(trueList,decepTiveList)



print "done"