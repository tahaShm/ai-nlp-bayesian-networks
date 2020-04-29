import nltk
import csv
import pandas as pd
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
            
        
        
cl = Classifier(dataFile, testFile)       