# https://www.tensorflow.org/versions/master/tutorials/word2vec/index.html#vector-representations-of-words
# https://tensorflow.googlesource.com/tensorflow/+/master/tensorflow/examples/tutorials/word2vec/word2vec_basic.py

import os
import cPickle as pickle
import argparse
from exec_01_w2v_cleanData import *
from exec_02_w2v_model_train import *
from exec_03_w2v_visualize_embedings import *

def clean_model_train_persist_visualize(searchWord1, searchWord2):

    ########################################################
    # read the previous cleaned data into a list<string>
    ########################################################

    with open('../../data/text8', 'r') as myfile:
        words=myfile.read().split()

    #print('Data size', len(words))
    #print (type(words))
    #print (words[:5])

    #####################################################################
    # Step 1: Build the dictionary and replace rare words with UNK token.
    #  == Bag of Words + Stopwords
    #####################################################################

    vocabulary_size = 50000
    data, count, dictionary, reverse_dictionary = build_dataset(words, vocabulary_size,persist=True)

    ##############################################################################
    # Step 2: Choose the words for which similarity should be evaluated.
    ##############################################################################

    # prediction set to sample nearest neighbors.
    #valid_examples = np.array([dictionary['four'], dictionary['american']])
    valid_examples = np.array([dictionary[searchWord1], dictionary[searchWord2]])
    #valid_examples = np.array([21, 64]) # alternative the ids of the words can be directly used

    ##############################################################################
    # Step 3: Generate a training batch for the skip-gram model.
    ##############################################################################

    batch, labels = generate_batch(data, batch_size=8, num_skips=2, skip_window=1) #ndarray
    #print_generate_batch(batch, labels, reverse_dictionary)

    ##############################################################################
    # Step 4: Build and train a skip-gram model. Persist result to disk.
    ##############################################################################

    final_embedings_similarity, final_embeddings_normalized  = \
        build_tain_model(valid_examples,data,dictionary, reverse_dictionary,vocabulary_size,persist=True)

    print ("final_embeddings_similarity : " , final_embedings_similarity.shape , " " ,
        type(final_embedings_similarity)) # numpy.ndarray'>    (50000, 128)
    print ("final_embeddings_normalized (trained vectors)" ,
        type(final_embeddings_normalized) , "  " , final_embeddings_normalized.shape)  #  numpy.ndarray'>    (50000, 128)


    #print ("valid_embeddings (dictonary words)" , type(valid_embeddings) , "  " , valid_embeddings.shape) # Tensor
    #print ("Similar (" ,reverse_dictionary[dictionary["four"]], ") : ", final_embeddings[0])

    ##############################################################################
    # Step 7: Visualize the embeddings.
    ##############################################################################

    visualize_similarities_text(final_embedings_similarity,reverse_dictionary,valid_examples)

    visualize_plot_with_labels(final_embeddings_normalized, reverse_dictionary)



def load_visualize(searchWord1, searchWord2):

    dictionary = pickle.load( open( "output/dictionary.pickle", "rb"))
    reverse_dictionary = pickle.load( open( "output/reverse_dictionary.pickle", "rb"))
    final_embedings_similarity = pickle.load( open( "output/final_embedings_similarity.pickle", "rb"))
    final_embeddings_normalized = pickle.load( open( "output/final_embeddings_normalized.pickle", "rb"))

    # prediction set to sample nearest neighbors.
    valid_examples = np.array([dictionary[searchWord1], dictionary[searchWord2]])

    visualize_similarities_text(final_embedings_similarity,reverse_dictionary,valid_examples)
    visualize_plot_with_labels(final_embeddings_normalized, reverse_dictionary)


##################################################################################################
# command line options
##################################################################################################

parser = argparse.ArgumentParser()
parser.add_argument("-search1" , "--searchWord1", type=str, help="word1 to search for related vectors")
parser.add_argument("-search2" , "--searchWord2", type=str, help="word2 to search for related vectors")
parser.add_argument("-onlyVis" , "--onlyVisualize", type=str, help="only visualize related vectors")
args = parser.parse_args()
searchWord1  = args.searchWord1
searchWord2  = args.searchWord2
onlyVisualize = args.onlyVisualize

##################################################################################################
# Execute
##################################################################################################

#searchWord1 = 'four' # four, pictures
#searchWord2 = 'american' # american

if (onlyVisualize == "false"):
    # Execute 01: This function must only be called once, to persist the similarity embeddings numpy
    clean_model_train_persist_visualize(searchWord1,searchWord2)
if (onlyVisualize == "true"):
    # This function can load the similarity embedings from disk and visualize them
    load_visualize(searchWord1,searchWord2)
