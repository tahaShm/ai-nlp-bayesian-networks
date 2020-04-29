import nltk
import csv
import pandas as pd
import math
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
        # print(self.pt, self.pb, self.ps)
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
        
        # print(len(travelData))
        # print(len(businessData))
        # print(len(styleData))
        
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
            if (not (isinstance(row[3], float) and math.isnan(row[3]))) : 
                headWordList = preprocessData(row[2])
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
        counter = 0
        wordDic = {}
        for word in words :
            if (word not in wordDic) :
                wordDic[word] = 1/wordsSize
            else: 
                wordDic[word] += 1/wordsSize
        return wordDic  
            
        
        
cl = Classifier(dataFile, testFile)       