import nltk
import csv
import pandas as pd
import math
import numpy as np
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer

dataFile = "data.csv"
testFile = "test.csv"

def readFile(fileName) :
    file = open(fileName, 'r+')
    fList = file.readlines()
    file.close()
    return fList

def removeStopWordsAndPunctuations(context) :
    stop_words = set(stopwords.words('english')) 
    
    tokenizer = RegexpTokenizer(r'\w+')
    context = tokenizer.tokenize(context)
    
    filtered_sentence = [] 
    
    for w in context: 
        if w not in stop_words: 
            filtered_sentence.append(w)
    return filtered_sentence

def lowerAllCharacters(context) :
    context = context.lower()
    return context

def lemmatizeWords(wordList) : 
    lemmatizer = WordNetLemmatizer()
    for i in range(len(wordList)):
        wordList[i] = lemmatizer.lemmatize(wordList[i])
    return wordList

def preprocessData(context) : 
    context = lowerAllCharacters(context)   
    wordList = removeStopWordsAndPunctuations(context)
    wordList = lemmatizeWords(wordList)
    return wordList
    
    
class Classifier :
    def __init__(self, dataFile, testFile) :
        [self.travelTrainData, self.travelEvaluateData, self.businessTrainData, self.businessEvaluateData, self.styleTrainData, self.styleEvaluateData] = self.getDatas(dataFile)
        # print(len(self.travelTrainData))
        # print(len(self.travelEvaluateData))
        
        # print(len(self.businessTrainData))
        # print(len(self.businessEvaluateData))
        
        # print(len(self.styleTrainData))
        # print(len(self.styleEvaluateData))
        self.pt = (len(self.travelTrainData) / (len(self.travelTrainData) + len(self.businessTrainData) + len(self.styleTrainData)))
        self.pb = (len(self.businessTrainData) / (len(self.travelTrainData) + len(self.businessTrainData) + len(self.styleTrainData)))
        self.ps = (len(self.styleTrainData) / (len(self.travelTrainData) + len(self.businessTrainData) + len(self.styleTrainData)))
        print("data.csv reading done.")
        self.ptWords = self.calculatePWords(self.travelTrainData)
        print("words' probability of travels done.")
        self.pbWords = self.calculatePWords(self.businessTrainData)
        print("words' probability of buisiness done.")
        self.psWords = self.calculatePWords(self.styleTrainData)
        print("words' probability of style & beauty done.")
        
    
    def getDatas(self, dataFile) : 
        col_list = ["index", "category", "headline", "short_description"]
        df = pd.read_csv(dataFile, usecols=col_list)
        travelData = []
        businessData = []
        styleData = []
        for index, dfRow in df.iterrows() : 
            if dfRow[1] == "TRAVEL" : 
                travelData.append(dfRow)
            elif dfRow[1] == "BUSINESS" : 
                businessData.append(dfRow)
            elif dfRow[1] == "STYLE & BEAUTY" : 
                styleData.append(dfRow)
        
        travelSize = len(travelData)
        businessSize = len(businessData)
        styleSize = len(styleData)
        
        
        maxSize = max(travelSize, businessSize, styleSize)
        
        if (travelSize < maxSize) : 
            l1 = list(range(travelSize))
            travelIndices = np.random.choice(l1, maxSize)
            travelData2 = []
            for i in range(maxSize) : 
                travelData2.append(travelData[travelIndices[i]])
            travelData = travelData2
        
        if (businessSize < maxSize) : 
            l1 = list(range(businessSize))
            businessIndices = np.random.choice(l1, maxSize)
            businessData2 = []
            for i in range(maxSize) : 
                businessData2.append(businessData[businessIndices[i]])
            businessData = businessData2
                
        if (styleSize < maxSize) : 
            l1 = list(range(styleSize))
            styleIndices = np.random.choice(l1, maxSize)
            styleData2 = []
            for i in range(maxSize) : 
                styleData2.append(styleData[styleIndices[i]])
            styleData = styleData2
        
        travelSize = len(travelData)
        businessSize = len(businessData)
        styleSize = len(styleData)
            
        # print(len(travelData))
        # print(len(businessData))
        # print(travelData[8800])
        # print(businessData[8800])
        # print(styleData[8800])
        
        # print(len(styleData))
            
                
        # print(businessData2)
        # print("**************************")
        
        travelTrainData = []
        businessTrainData = []
        styleTrainData = []
        travelEvaluateData = []
        buisinessEvaluateData = []
        styleEvaluateData = []
        
        for i in range(travelSize) :
            if (i <= (int)(travelSize * 0.8)) :
                travelTrainData.append(travelData[i])
            else:
                travelEvaluateData.append(travelData[i])
        for i in range(businessSize) :
            if (i <= (int)(businessSize * 0.8)) :
                businessTrainData.append(businessData[i])
            else:
                buisinessEvaluateData.append(businessData[i])
        for i in range(styleSize) :
            if (i <= (int)(styleSize * 0.8)) :
                styleTrainData.append(styleData[i])
            else:
                styleEvaluateData.append(styleData[i])
                
        return [travelTrainData, travelEvaluateData, businessTrainData, buisinessEvaluateData, styleTrainData, styleEvaluateData]
    
    def getProcessedWords(self, dataSet) :
        words = []
        #row[2] -> headlines
        #row[3] -> short_description
        i = 0
        for row in dataSet :  
            i += 1 
            headWordList = []
            descWordList = []
            if (not (isinstance(row[2], float) and math.isnan(row[2]))) : 
                headWordList = preprocessData(row[2])
            if (not (isinstance(row[3], float) and math.isnan(row[3]))) : 
                descWordList = preprocessData(row[3])
            
            currentWords = headWordList + descWordList
            words = words + currentWords
                # print(words)
                # print("___________________________________________________________")
                # print(headWordList)
                # print(descWordList)
                
                
        return words

    def calculatePWords(self, dataSet) : 
        words = self.getProcessedWords(dataSet)
        wordsSize = len(words)
        wordDic = {}
        for word in words :
            if (word not in wordDic) :
                wordDic[word] = 1/wordsSize
            else: 
                wordDic[word] += 1/wordsSize
        for word in wordDic:
            wordDic[word] = math.log(wordDic[word], 10)
        return wordDic 
    
    def evaluateTravelP(self, wordList) :
        pLog = 0
        for word in wordList : 
            if (word in self.ptWords) : 
                pLog += self.ptWords[word]
            else:
                pLog += -6
        return pLog
    
    def evaluateBusinessP(self, wordList) :
        pLog = 0
        for word in wordList : 
            if (word in self.pbWords) : 
                pLog += self.pbWords[word]
            else:
                pLog += -6
        return pLog
    
    def evaluateStyleP(self, wordList) :
        pLog = 0
        for word in wordList : 
            if (word in self.psWords) : 
                pLog += self.psWords[word]
            else:
                pLog += -6
        return pLog
     
    def classify2EvaluateDatas(self, data, dataType) :
        i = 0
        travelCount = 0
        businessCount = 0
        for row in data:
            headWordList = []
            descWordList = []
            
            if (type(row[2]) == list) :
                headWordList = row[2]
            elif (not (isinstance(row[2], float) and math.isnan(row[2]))) : 
                headWordList = preprocessData(row[2])
            
            if (type(row[3]) == list) :
                descWordList = row[3]
            elif (not (isinstance(row[3], float) and math.isnan(row[3]))) : 
                descWordList = preprocessData(row[3])
            
            currentWords = headWordList + descWordList
            data[i][2] = currentWords
            i += 1
            travleTotalP = self.evaluateTravelP(currentWords)
            businessTotalP = self.evaluateBusinessP(currentWords)
            if (travleTotalP >= businessTotalP) :
                travelCount += 1
            else:
                businessCount += 1
        return [travelCount, businessCount]
    
    def classify3EvaluateDatas(self, data, dataType) :
        travelCount = 0
        businessCount = 0
        styleCount = 0
        for row in data:
            
            currentWords = row[2]
            travleTotalP = self.evaluateTravelP(currentWords)
            businessTotalP = self.evaluateBusinessP(currentWords)
            styleTotalP = self.evaluateStyleP(currentWords)
            if (max(travleTotalP, businessTotalP, styleTotalP) == travleTotalP) :
                travelCount += 1
            elif (max(travleTotalP, businessTotalP, styleTotalP) == businessTotalP):
                businessCount += 1
            else: 
                styleCount += 1
        return [travelCount, businessCount, styleCount]      
            
    
    def classify2(self) :
        travelPredics = self.classify2EvaluateDatas(self.travelEvaluateData, "travel") 
        businessPredicts = self.classify2EvaluateDatas(self.businessEvaluateData, "business")
        print("travel recall : {:.3f}".format(travelPredics[0] / (travelPredics[0] + travelPredics[1]) * 100))
        print("travel precision : {:.3f}".format(travelPredics[0] / (travelPredics[0] + businessPredicts[0]) * 100))
        
        print("business recall : {:.3f}".format(businessPredicts[1] / (businessPredicts[0] + businessPredicts[1]) * 100))
        print("business precision : {:.3f}".format(businessPredicts[1] / (travelPredics[1] + businessPredicts[1]) * 100))
        
        print("accuracy: {:.3f}".format((travelPredics[0] + businessPredicts[1]) / (travelPredics[0] + travelPredics[1] + businessPredicts[0] + businessPredicts[1]) * 100))
        
        
    def classify3(self) :
        travelPredics = self.classify3EvaluateDatas(self.travelEvaluateData, "travel") 
        businessPredicts = self.classify3EvaluateDatas(self.businessEvaluateData, "business")
        stylePredict = self.classify3EvaluateDatas(self.styleEvaluateData, "style")
        
        print("travel recall : {:.3f}".format(travelPredics[0] / (travelPredics[0] + travelPredics[1] + travelPredics[2]) * 100))
        print("travel precision : {:.3f}".format(travelPredics[0] / (travelPredics[0] + businessPredicts[0] + stylePredict[0]) * 100))
        
        print("business recall : {:.3f}".format(businessPredicts[1] / (businessPredicts[0] + businessPredicts[1] + businessPredicts[2]) * 100))
        print("business precision : {:.3f}".format(businessPredicts[1] / (travelPredics[1] + businessPredicts[1] + stylePredict[1]) * 100))
        
        print("style recall : {:.3f}".format(stylePredict[2] / (stylePredict[0] + stylePredict[1] + stylePredict[2]) * 100))
        print("style precision : {:.3f}".format(stylePredict[2] / (travelPredics[2] + businessPredicts[2] + stylePredict[2]) * 100))
        
        print("accuracy: {:.3f}".format((travelPredics[0] + businessPredicts[1] + stylePredict[2]) / (travelPredics[0] + travelPredics[1] + travelPredics[2] + businessPredicts[0] + businessPredicts[1] + businessPredicts[2] + stylePredict[0] + stylePredict[1] + stylePredict[2]) * 100))
        
            
        
        
cl = Classifier(dataFile, testFile)  
cl.classify2()  
print("______________")
cl.classify3()   