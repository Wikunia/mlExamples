#https://www.kaggle.com/c/word2vec-nlp-tutorial/details/part-1-for-beginners-bag-of-words

# Import the pandas package, then use the "read_csv" function to read the labeled training data
import pandas as pd
from bs4 import BeautifulSoup
import re
import nltk
from nltk.corpus import stopwords # Import the stop word list
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import json
import codecs
from sklearn.externals import joblib
from helper_01_dataCleaning import review_to_words
import argparse


###########################################################################
# Load RandomForestClassifier
###########################################################################

#Restore random_forest_classifier
forest = joblib.load('output/train_data_random_forest_classifier.joblib')

with open('additionalSamples/testData.txt', 'r') as myfile:
    test=myfile.read()

#command line options
#parser = argparse.ArgumentParser()
#parser.add_argument("-i" , "--input", type=str, help="Review statement")
#args = parser.parse_args()
#test  = args.input

#vocabulary
with codecs.open('output/train_data_vocab.json', 'r', encoding='utf-8') as f:
    vocab = json.load(f)


vectorizer = CountVectorizer(analyzer = "word",   \
                             vocabulary = vocab,   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 5000)

############################################################
#Make predictions
############################################################

# Create an empty list and append the clean reviews one by one
clean_test_reviews = []

print "Cleaning and parsing the test set movie reviews...\n"
clean_review = review_to_words( test )
clean_test_reviews.append( clean_review )

# Get a bag of words for the test set, and convert to a numpy array
test_data_features = vectorizer.transform(clean_test_reviews)
test_data_features = test_data_features.toarray()

# Use the random forest to make sentiment label predictions
result = forest.predict(test_data_features)

# Copy the results to a pandas dataframe with an "id" column and
# a "sentiment" column

print result

if result[0] == 0 :
    print "[%i] %s : %s" % (result[0] , "nagative" , test )
else :
    print "[%i] %s : %s" % (result[0] , "possitive" , test)
