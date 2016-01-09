# https://www.tensorflow.org/versions/master/tutorials/word2vec/index.html#vector-representations-of-words
# https://tensorflow.googlesource.com/tensorflow/+/master/tensorflow/examples/tutorials/word2vec/word2vec_basic.py

import collections
import numpy as np
import os
import random
import cPickle as pickle

#####################################################################
# Step 2: Build the dictionary and replace rare words with UNK token.
#  == Bag of Words + Stopwords
#####################################################################

def build_dataset(words, vocabulary_size, persist=False):
  count = [['UNK', -1]]
  """
  Leaving out the argument  most_common() would produce a list of all the items, in order of frequency.
  most_common(vocabulary_size - 1) limits the list ( variable: count ) to a fixed size of 5k
  """
  count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
  #print (count[:3])

  """
  create a dictionary with the 5k words.
  Word 1 will set value to 0    {'UNK': 0}
  Word 2 will set value to 1    {'the': 1, 'UNK': 0}
  Word 3 will set value to 2    {'of': 2, 'the': 1, 'UNK': 0}
  Word 4 will set value to 3    {'and': 3, 'of': 2, 'the': 1, 'UNK': 0}
  """
  dictionary = dict()
  i=0
  for word, _ in count:
    #if i < 5:
    # print (dictionary)
    i=i+1
    dictionary[word] = len(dictionary)


  """
  replace words in the data which are not in the dictionary with UNK.
  replace the numeric values, the variable "data" is of type list<interger> .
  """
  data = list()
  unk_count = 0
  for word in words:
    if word in dictionary:
      index = dictionary[word]
    else:
      index = 0  # dictionary['UNK']
      unk_count = unk_count + 1
    data.append(index)
  #print(data[:5])
  #print (dictionary['anarchism'])  # First word in the input text. Maps to index/position 5239 in the dictionary.
  #print(data[0]) # 5239 is saved as first element in the data-list
  """
  set variable to the frequency of NUM occurrence
  """
  count[0][1] = unk_count
  #print ("UNK count: " , count[0][1])
  #print (data.count(0))

  """
  zip()  takes two lists as arguments and combines them into a new list [(l1[0],l2[0]), (l1[1],l2[1]) ...]
  dict() takes one list in zip-format and converts it into a dictionary
  Reverse because, previously key=String, value=Int and afterwards key=Int, value=String.
  Reverse because, key- and value-order have been reversed.
  """
  reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
  del words  # Hint to reduce memory.

  #persist to disk
  if persist == True:
     pickle.dump(reverse_dictionary,open('output/reverse_dictionary.pickle','wb'))
     pickle.dump(dictionary,open('output/dictionary.pickle','wb'))


  return data, count, dictionary, reverse_dictionary




##############################################################################
# Step 3: Function to generate a training batch for the skip-gram model.
##############################################################################

"""
 create training examples (one-hot vectors) from plain text, used afterwards as input for skip gram.
 specific two one-hot vectors will be created with identical context but different shapes.
 variable batch (ndarray) is one column of size batch_size.
 variable labels (ndarray) is one row of size batch_size.

 skip_window: window of words to the left and to the right of a target word.
 batch_size: size of the ndarray vector
"""
data_index = 0

def generate_batch(data, batch_size, num_skips, skip_window):
  global data_index
  assert batch_size % num_skips == 0
  assert num_skips <= 2 * skip_window
  batch = np.ndarray(shape=(batch_size), dtype=np.int32)
  labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
  span = 2 * skip_window + 1 # [ skip_window target skip_window ]
  buffer = collections.deque(maxlen=span)
  for _ in range(span):
    buffer.append(data[data_index])
    data_index = (data_index + 1) % len(data)
  for i in range(batch_size // num_skips):
    target = skip_window  # target label at the center of the buffer
    targets_to_avoid = [ skip_window ]
    for j in range(num_skips):
      while target in targets_to_avoid:
        target = random.randint(0, span - 1)
      targets_to_avoid.append(target)
      batch[i * num_skips + j] = buffer[skip_window]
      labels[i * num_skips + j, 0] = buffer[target]
    buffer.append(data[data_index])
    data_index = (data_index + 1) % len(data)
  return batch, labels


"""
    transform the cleaned data on a format that can be used by skip-gram
"""
def print_generate_batch(batch, labels, reverse_dictionary):
    print("type(batch): %s , type(lables) : %s"  % (type(batch),type(labels)) )
    print ("batch.shape : %s   , labels.shape : %s " % (batch.shape,labels.shape))
    print (batch)  # horizontal  _ (column)
    print (labels) # vertical  | (row)

    for i in range(8):
      """
       labels[i, 0] : because the labels vector was defined as single row (shape=(batch_size, 1)), the
       second index to access the values must be 0.
      """
      print(batch[i], '->', labels[i, 0])   # numernic values
      print(reverse_dictionary[batch[i]], '->', reverse_dictionary[labels[i, 0]])  # text values
