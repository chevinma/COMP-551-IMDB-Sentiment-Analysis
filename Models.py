import os
import string
import nltk
import re
from nltk.stem import *
from nltk.tokenize import sent_tokenize as st, word_tokenize as wt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
import pandas as pd
import time


# Use SnowballStemmer to stem input comments.
ps = SnowballStemmer("english")
# Use nltk's predefined stopword list as our stop_words set.
stop_words = set(nltk.corpus.stopwords.words('english'))

# Remove all occurrences of punctuation with this function.
punct_remove = string.punctuation.maketrans('', '', string.punctuation)

# Initiate list
raw_x_train = list()
y_train = list()
x_test = list()
y_test = list()
pos_list = list()

x_real_test = list()

# Open Negative Training Data and append entries to list.
for filename in os.listdir('./train/neg'):
    file = open('./train/neg/' + filename, encoding="utf8")
    raw_x_train.append(file.read())
    y_train.append(0)
    file.close()

# Open Positive Training Data and append entries to list.
for filename in os.listdir('./train/pos'):
    file = open('./train/pos/' + filename, encoding="utf8")
    raw_x_train.append(file.read())
    y_train.append(1)
    file.close()

for i in range(25000):
    file = open("./test/%d.txt"%(i), encoding="utf8")
    x_real_test.append(file.read())
    file.close()


'''
Process text before pipeline.
'''
def stem_words(data, linebreak=False, notcontract=True,
               havecontract=True, punctuation=True):
    def feature_tokens(tokens):
        stemtokens = list()
        for i in range(len(tokens)):
            if tokens[i] == 'not':
                i += 1
                continue
            if tokens[i] not in stop_words:
                stemmed = ps.stem(tokens[i])
                if len(stemmed) > 2:
                    stemtokens.append(stemmed)
        return stemtokens

    def processText(text, linebreak=False, notcontract=True,
                    havecontract=True, punctuation=True):
        if linebreak:   text = re.sub("<.*>", ' ', text)
        if notcontract: text = re.sub("n't", ' not', text)
        if havecontract: text = re.sub("'ve", ' have', text)
        if punctuation: text = text.translate(punct_remove).lower()
        return text

    # initiate list for counting word frequencies in the list of documents
    new_train = list()
    for rawtext in data:
        # remove line breaks, indenting, punctuation, contractions
        text = processText(rawtext, linebreak, notcontract, havecontract,
                           punctuation)

        # adds all stems that aren't stopwords
        tokens = wt(text)
        stemtokens = feature_tokens(tokens)
        new_train.append(' '.join(stemtokens))

    return new_train

prestart = time.time()
new_train = stem_words(raw_x_train)

X_train, X_test, y_train, y_test = train_test_split(new_train, y_train, train_size=0.8, test_size=0.2)


pclf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('norm', Normalizer()),
    ('clf', LogisticRegression()),
])
preend = time.time()

fitstart = time.time()
pclf.fit(X_train, y_train)
fitend = time.time()

predstart = time.time()
y_pred = pclf.predict(X_test)
predend = time.time()

end = time.time()

print("Pre-Process Time: {}\nTraining Time: {}\nPrediction Time: {}".format(preend-prestart,
                                                                            fitend-fitstart, predend-predstart))
print(metrics.classification_report(y_test, y_pred))



#data_to_submit = pd.DataFrame({'Id': [i for i in range(len(y_pred))], 'Category': y_pred})

#data_to_submit.to_csv("./out.csv", index=False)

