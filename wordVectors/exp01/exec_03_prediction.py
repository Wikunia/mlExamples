#https://www.kaggle.com/c/word2vec-nlp-tutorial/details/part-2-word-vectors

from gensim.models import word2vec


######################################################################################
# Restore word2vector model (trained model with some semantic understanding of words,)
######################################################################################

model = word2vec.Word2Vec()
model = model.load("output/IMDB_300features_40minwords_10context.cnpy")

################################################################################################
#Make predictions: doesnt_match() deduce which word in a set is most dissimilar from the others
################################################################################################

"""
 The "doesnt_match" function will try to deduce which word in a set is most dissimilar from the others:
"""
print("\n###################### doesnt_match  ######################\n")
result = model.doesnt_match("man woman child kitchen".split())
print ("doesnt_match(man, woman, child, kitchen) : %s \n" % (result))

result = model.doesnt_match("Good, excellent, worst, blue".split())
print ("doesnt_match(Good, excellent, worst, blue) : %s \n" % (result))

result = model.doesnt_match("chocolate earth meat fruit".split())
print ("doesnt_match(chocolate, earth, meat, fruit) : %s \n" % (result))

result = model.doesnt_match("france england germany berlin".split())
print ("doesnt_match(france, england, germany, berlin) : %s \n" % (result))

result = model.doesnt_match("paris berlin london austria".split())
print ("doesnt_match(paris, berlin, london, austria) : %s \n" % (result))

##########################################################################
#Make predictions: most_similar()  insight into the model's word clusters
##########################################################################

"""
 The "most_similar" function will print the word cluster of similarity
"""
print("\n###################### most_similar   ######################\n")

print ("most_similar(man) :\n %s \n" % (model.most_similar("man")))
print ("most_similar(queen) :\n %s \n" % (model.most_similar("queen")))
print ("most_similar(awful) :\n %s \n" % (model.most_similar("awful")))
print ("most_similar(positive=[woman, king], negative=[man], topn=1) :\n %s \n"
        % (model.most_similar(positive=['woman', 'king'], negative=['man'], topn=1)))


##########################################################################
#Make predictions: similarity()  distance between words
##########################################################################

"""
 The "similarity" function will print the distance between word similarity
"""

print("\n###################### similarity   ######################\n")

print ("model.similarity('woman', 'man') :\n %s \n" % (model.similarity('woman', 'man')))

##########################################################################
#Raw output
##########################################################################

"""
 raw NumPy vector of a word
"""
#print("\n###################### Raw output   ######################\n")

#print ("model['man'] : \n %s \n \n" % (model['man'])) # Individual word vectors can be accessed by vocabulary
#print ("model['man'] : \n %s \n \n" % (model.syn0[1])) # Or accessed by index
