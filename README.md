# COMP 551 - Mini-Project 2
This repository contains source code for Mini-Project 2 of McGill University's COMP 551 Course (Applied Machine Learning). The goal of this project is to classify IMDB comments as positive or negative (Sentiment Analysis). 

## Provided Files
`preprocessing.py` - This file preprocesses our data. More information about how data is pre-processed may be found in 
the report for this project. This generates the required `negwordcounts.csv` and `poswordcounts.csv` for running the 
Bernouilli Naive Bayes model.
`NaiveBayes.py` - This file trains the Bernouilli Naive Bayes model on the training set.
`Models.py` - This file contains the source code for the Logistic Regression model eventually used for submission to the Kaggle Competition.
`Experiments.ipynb` - This file is a Jupyter notebook containing experiments run for this project.
`test/` , `train/` - These directories containing test and training data, respectively.

## Installing Requirements
We use multiple libraries for this project - we predominantly use sk-learn to train our models and nltk for preprocessing
(stemming, lemmatizing, ...). 

All requirements are kept in `requirements.txt` and may be installed with `pip install --user -r requirements.txt`.