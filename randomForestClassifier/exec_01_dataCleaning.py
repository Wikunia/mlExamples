#https://www.kaggle.com/c/word2vec-nlp-tutorial/details/part-1-for-beginners-bag-of-words

# Import the pandas package, then use the "read_csv" function to read the labeled training data
import pandas as pd
from bs4 import BeautifulSoup
import re
import nltk
from nltk.corpus import stopwords # Import the stop word list
import json
import codecs
from helper_01_dataCleaning import review_to_words

########################################################
# Clean all the data
# See method "review_to_words" in helper_01_dataCleaning
########################################################

train = pd.read_csv("../data/labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)

# Get the number of reviews based on the dataframe column size
num_reviews = train["review"].size

# Initialize an empty list to hold the clean reviews
clean_train_reviews = []

print "Cleaning and parsing the training set movie reviews...\n"

# Loop over each review; create an index i that goes from 0 to the length
# of the movie review list
for i in xrange( 0, num_reviews ):
    # Call our function for each one, and add the result to the list of
    # clean reviews
    clean_train_reviews.append( review_to_words( train["review"][i] ) )
    # If the index is evenly divisible by 1000, print a message
    if( (i+1)%1000 == 0 ):
        print "Review %d of %d\n" % ( i+1, num_reviews )

#Serialize cleaned data
with codecs.open('output/clean_train_reviews.json',  'w', encoding='utf-8') as f:
	json.dump(clean_train_reviews, f,  indent=3)
