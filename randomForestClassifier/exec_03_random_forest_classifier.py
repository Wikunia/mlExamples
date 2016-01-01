#https://www.kaggle.com/c/word2vec-nlp-tutorial/details/part-1-for-beginners-bag-of-words

# Import the pandas package, then use the "read_csv" function to read the labeled training data
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import json
import codecs
from sklearn.externals import joblib

###########################################################################
# Load Bag_of_Words_model and train data
###########################################################################

train_data_features = np.load('output/train_data_features.npy')
#labeld training dataset
train = pd.read_csv("../data/labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)

############################################################
#Random Forest classifier
############################################################


print "...Random Forest classifier start"

#Initialize a Random Forest classifier with 100 trees.
forest = RandomForestClassifier(n_estimators = 100)

"""
    Initialize a Random Forest classifier with 100 trees.
    Random_state fixed to make it reproducible, but the prediction would be
    incorrect afterwards. Without radom_state=42 the results are much better.
    #forest = RandomForestClassifier(n_estimators = 100, random_state=42)
"""

# Fit the forest to the training set, using the bag of words as
# features and the sentiment labels as the response variable
#
# This may take a few minutes to run
forest = forest.fit( train_data_features, train["sentiment"] )

print "...Random Forest classifier finished"


#persist to disk
joblib.dump(forest, 'output/train_data_random_forest_classifier.joblib')
