import csv
import numpy as np
import math
import matplotlib as plt
import os
import string
import nltk
import re
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import *
from nltk.tokenize import sent_tokenize as st, word_tokenize as wt

#intiate stemmer, define list of stopwords, definte translation for punctuation removal
ps = SnowballStemmer("english")
#ps = PorterStemmer()
stop_words = set(nltk.corpus.stopwords.words('english'))
punct_remove = string.punctuation.maketrans('', '', string.punctuation)


#intiate lists
neg_list = list()
pos_list = list()
#negTest = list()
#posTest = list()

#open negative training sets and add them as strings onto negative list
for filename in os.listdir('./train/neg'):
    file = open('./train/neg/' + filename, encoding="utf8")
    neg_list.append(file.read())
    file.close()
#open positive training sets and add them as strings onto positive list
for filename in os.listdir('./train/pos'):
    file = open('./train/pos/' + filename, encoding="utf8")
    pos_list.append(file.read())
    file.close()


#generates a list of words from the frequencies positive and negative frequency lists
def word_ranking(hash1, hash2):
    polarity = {}
    for hash in hash1:
        polarity[hash] = hash1[hash]
    for hash in hash2:
        if hash in hash1:
            polarity[hash] = hash1[hash] + hash2[hash]
        else:
            polarity[hash] = hash2[hash]
    return polarity


#extracts the n top words from the list of words by ranking
def top_words(hash, n):
    topWords = {}
    n = min(len(hash), n)
    for i in range(n):
        #topWords.append(hash.pop(max(hash, key = hash.get)))
        top = max(hash, key = hash.get)
        topWords[top] = hash[top]
        hash.pop(top)
    return topWords


#extracts the n min words from the list of words by ranking
def min_words(hash, n):
    topWords = {}
    n = min(len(hash), n)
    for i in range(n):
        #topWords.append(hash.pop(max(hash, key = hash.get)))
        top = min(hash, key = hash.get)
        topWords[top] = hash[top]
        hash.pop(top)
    return topWords


#removes punctuation, < > strings, contractions
def text_process(text):
    #remove line breaks, indenting, punctuation, contractions
    text = re.sub("<.*>", ' ', text)
    text = re.sub("n't", ' not', text)
    text = re.sub("'ve", ' have', text)
    text = text.translate(punct_remove).lower()
    return text


def hash2matdict(hashtable):
    translator = {}
    inverse = {}
    i = 0
    for hash in hashtable:
        translator[int(i)] = hash
        inverse[hash] = i
        i += 1
    print(translator)
    return [translator, inverse]


def compute_adj(docs, words):
    n = len(words)
    data = np.zeros([n, n])
    for text in docs:
        text = text_process(text)
        tokens = wt(text)
        stemtokens = [ps.stem(word) for word in tokens if not word in stop_words]
#        stemtokens += [tokens[i] + ps.stem(tokens[i + 1]) for i in range(len(tokens) - 1) if tokens[i] == 'not' and not tokens[i + 1] in stop_words]
        for i in range(0, n):
            if words[i] in stemtokens:
                data[i, i] += 1
                for j in range(i):
                    if words[j] in stemtokens:
                        data[i, j] += 1
    for i in range(n):
        for j in range(i):
            data[j, i] = data[i, j]
    data /= len(docs)
    return data



def csvtohash(csvname):
    with open(csvname, 'r', encoding="utf8") as hashfile:
        text = hashfile.read()
    text = text.replace(',', ' ')
    tokens = wt(text)
    tokens = tokens[2:len(tokens)]
    hashsize = int(len(tokens)/2)
    hashmap = {}
    for i in range(hashsize):
        hashmap[tokens[2*i]] = float(tokens[2*i + 1])
    return hashmap


def word_scoring(adjmat, translator):
    scores = {}
    for feat1 in translator:
        entropy = 0
        key = translator[feat1]
        for feat2 in translator:
            if not feat1 == feat2 and adjmat[feat1, feat2] > 0:
                entropy += adjmat[feat1, feat2]*math.log(adjmat[feat1,feat2]/((adjmat[feat1, feat1]*adjmat[feat2, feat2])), 2)
        scores[key] = entropy
    return scores


neg_freq_list = csvtohash('negwordcounts.csv')
pos_freq_list = csvtohash('poswordcounts.csv')

train = neg_list[6250:len(neg_list)] + pos_list[6250:len(pos_list)]

ranking = word_ranking(neg_freq_list, pos_freq_list)
top = top_words(ranking, 1000)
[translator, inverse] = hash2matdict(top)

adjmat = compute_adj(train, translator)
scores = word_scoring(adjmat, translator)
bestfeat = top_words(scores, 500)
newtrans = {}
for key in bestfeat:
    index = int(inverse[key])
    newtrans[index] = key
scores = word_scoring(adjmat, newtrans)
bestfeat = top_words(scores, 250)


with open('bestfeatures.csv', 'w', encoding="utf8") as csv_file:
    fieldnames = ['Word', 'Frequency']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    for key in bestfeat.keys():
        writer.writerow({'Word': key,'Frequency': bestfeat[key]})

