#https://www.kaggle.com/c/word2vec-nlp-tutorial/details/part-1-for-beginners-bag-of-words

# Import the pandas package, then use the "read_csv" function to read the training data
import pandas as pd
import json
import codecs
from helper_01_dataCleaning import *
import nltk.data

########################################################
# Clean all the data
# See method "review_to_words" in helper_01_dataCleaning
########################################################

train = pd.read_csv("../../data/labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
test = pd.read_csv( "../../data/testData.tsv", header=0, delimiter="\t", quoting=3 )
unlabeled_train = pd.read_csv( "../../data/unlabeledTrainData.tsv", header=0,  delimiter="\t", quoting=3 )

# Verify the number of reviews that were read (100,000 in total)
print "Read %d labeled train reviews, %d labeled test reviews, " \
 "and %d unlabeled reviews\n" % (train["review"].size,
 test["review"].size, unlabeled_train["review"].size )

# Load the punkt tokenizer
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

# Initialize an empty list to hold the clean list of sentences
sentences = []

# Loop over each review
print "Parsing sentences from training set"
for review in train["review"]:
    sentences += review_to_sentences(review, tokenizer)

print "Parsing sentences from unlabeled set"
for review in unlabeled_train["review"]:
    sentences += review_to_sentences(review, tokenizer)

#Serialize cleaned data
with codecs.open('output/sentences_list.json',  'w', encoding='utf-8') as f:
	json.dump(sentences, f,  indent=3)

print "Check how many sentences we have in total - should be around 850,000+"
print len(sentences)
