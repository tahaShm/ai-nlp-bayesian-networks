import nltk
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer

dataFile = "data.csv"
test = "test.csv"

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
    