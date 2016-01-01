#https://www.kaggle.com/c/word2vec-nlp-tutorial/details/part-1-for-beginners-bag-of-words

# Import the pandas package, then use the "read_csv" function to read the labeled training data
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import json
import codecs
from sklearn.externals import joblib

###########################################################################
# Load cleaned data
###########################################################################

with codecs.open('output/clean_train_reviews.json', 'r', encoding='utf-8') as f:
	clean_train_reviews = json.load(f)

###########################################################################
# Creating Features from a Bag of Words (Using scikit-learn)
###########################################################################

print "Creating the bag of words...\n"
# Initialize the "CountVectorizer" object, which is scikit-learn's
# bag of words tool.
vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 5000)

# fit_transform() does two functions: First, it fits the model
# and learns the vocabulary; second, it transforms our training data
# into feature vectors. The input to fit_transform should be a list of
# strings.
train_data_features = vectorizer.fit_transform(clean_train_reviews)

# Take a look at the words in the vocabulary
vocab = vectorizer.get_feature_names()

# Numpy arrays are easy to work with, so convert the result to an
# array
train_data_features = train_data_features.toarray()

print (train_data_features)
print (train_data_features.shape)

# Sum up the counts of each vocabulary word
#dist = np.sum(train_data_features, axis=0)

# For each, print the vocabulary word and the number of times it
# appears in the training set
#for tag, count in zip(vocab, dist):
#    print count, tag

#Serialize data features
np.save('output/train_data_features.npy',train_data_features)

with codecs.open('output/train_data_vocab.json',  'w', encoding='utf-8') as f:
	json.dump(vocab, f,  indent=3)

joblib.dump(vectorizer, 'output/train_data_vectorizer.joblib')
