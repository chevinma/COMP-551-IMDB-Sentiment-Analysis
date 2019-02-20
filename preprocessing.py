import csv
import numpy as np
import math
import matplotlib as plt
import re
import os
import string
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import *
from nltk.tokenize import sent_tokenize as st, word_tokenize as wt
import nltk

nopunct = string.punctuation.maketrans('', '', string.punctuation)
stop_words = set(nltk.corpus.stopwords.words('english'))
#ps = SnowballStemmer("english")
ps = PorterStemmer()

#list of data for the negative and positive training sets (separate)
negTrain = list()
posTrain = list()

#open negative training sets and add them as strings onto negative list
for filename in os.listdir('./train/neg'):
    file = open('./train/neg/' + filename, encoding="utf8")
    negTrain.append(file.read())
    file.close()
#open positive training sets and add them as strings onto positive list
for filename in os.listdir('./train/pos'):
    file = open('./train/pos/' + filename, encoding="utf8")
    posTrain.append(file.read())
    file.close()

#returns total counts of each word across all docs
def wordCounts(data):
    #initiate list for counting word frequencies in the list of documents
    count = {}
    for rawtext in data:
        #remove line breaks, indenting, punctuation, contractions
        text = processText(rawtext)
        #adds all stems that aren't stopwords
        tokens = wt(text)
        stemtokens = [ps.stem(word) for word in tokens if not word in stop_words]
        docwords = {}
        #adds all bigrams in the form [not ___] (excluding stop words)
        stemtokens += [tokens[i] + ps.stem(tokens[i + 1]) for i in range(len(tokens)-1) if tokens[i] == 'not' and not tokens[i + 1] in stop_words]
        for stem in stemtokens:
            if not stem in count:
                count[stem] = 0
            if not stem in docwords:
                docwords[stem] = 0
        for word in docwords:
            for stem in stemtokens:
                if word == stem:
                    docwords[word] += 1
            count[word] += docwords[word]
    return count

#returns number of times each word appears in docs
def word_frequencies(data):
    #initiate list for counting word frequencies in the list of documents
    count = {}
    for rawtext in data:
        #remove line breaks, indenting, punctuation, contractions
        text = processText(rawtext)
        #adds all stems that aren't stopwords
        tokens = wt(text)
        stemtokens = [ps.stem(word) for word in tokens if not word in stop_words]
        #adds all bigrams in the form [not ___] (excluding stop words)
        stemtokens += [tokens[i] + ps.stem(tokens[i + 1]) for i in range(len(tokens)-1) if tokens[i] == 'not' and not tokens[i + 1] in stop_words]
        for stem in stemtokens:
            if not stem in count:
                count[stem] = 0
        for word in count:
            if word in stemtokens:
                count[word] += 1
    return count


#remove line breaks, indenting, punctuation, contractions
def processText(text):
    text = re.sub("<.*>", ' ', text)
    text = re.sub("n't", ' not', text)
    text = re.sub("'ve", ' have', text)
    text = text.translate(nopunct).lower()
    return text


#iterates through all documents in data (as list) and stores word counts with tf-idf
def tfidfCounts(data):
    #initiate list for counting word frequencies in the list of documents
    count = {}
    for rawtext in data:
        #remove line breaks, indenting, punctuation, contractions
        text = processText(rawtext)
        #adds all stems that aren't stopwords
        tokens = wt(text)
        stemtokens = [ps.stem(word) for word in tokens if not word in stop_words]
        docwords = {}
        #adds all bigrams in the form [not ___] (excluding stop words)
        stemtokens += [tokens[i] + ps.stem(tokens[i + 1]) for i in range(len(tokens)-1) if tokens[i] == 'not' and not tokens[i + 1] in stop_words]
        for stem in stemtokens:
            if not stem in count:
                count[stem] = 0
            if not stem in docwords:
                docwords[stem] = 0
        for word in docwords:
            for stem in stemtokens:
                if word == stem:
                    docwords[word] += 1
            count[word] += math.log(len(stemtokens)/(docwords[word] + 1))*docwords[word]
    return count

[neg_words, pos_words] = [word_frequencies(negTrain[5000:len(negTrain)]), word_frequencies(posTrain[5000:len(posTrain)])]
#[negWords, posWords] = [tfidfCounts(negTrain[5000:len(negTrain)]), tfidfCounts(posTrain[5000:len(posTrain)])]

with open('negwordcounts.csv', 'w', encoding="utf8") as csv_file:
    fieldnames = ['Word', 'Frequency']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    for key in neg_words.keys():
        writer.writerow({'Word': key,'Frequency': neg_words[key]})

with open('poswordcounts.csv', 'w', encoding="utf8") as csv_file:
    fieldnames = ['Word', 'Frequency']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    for key in pos_words.keys():
        writer.writerow({'Word': key,'Frequency': pos_words[key]})
print(len(neg_words))
print(len(pos_words))
