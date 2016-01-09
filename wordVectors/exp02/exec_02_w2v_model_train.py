# https://www.tensorflow.org/versions/master/tutorials/word2vec/index.html#vector-representations-of-words
# https://tensorflow.googlesource.com/tensorflow/+/master/tensorflow/examples/tutorials/word2vec/word2vec_basic.py
# https://groups.google.com/a/tensorflow.org/forum/?utm_source=digest&utm_medium=email/#!topic/discuss/nbJ-n_SZcwU

from __future__ import absolute_import

import tensorflow.python.platform

import math
import cPickle as pickle
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import random
from exec_01_w2v_cleanData import *


batch_size = 128
embedding_size = 128  # Dimension of the embedding vector.
skip_window = 1       # How many words to consider left and right.
num_skips = 2         # How many times to reuse an input to generate a label.

def build_tain_model(valid_examples,data,dictionary, reverse_dictionary,vocabulary_size,persist=False):

    num_sampled = 64    # Number of negative examples to sample.

    graph = tf.Graph()

    """
      A TensorFlow graph is a description of computations:
        - builds the mathematical graph with nodes(ops) containing computation(math)-functions
        - input for the ops are tensors (typed multi-dimensional array)
        - output of the ops are tensors
    """
 ###########################################################################################
 # Step 1: Define "source ops" as graph nodes
 # source ops do not need any input and pass their output to other ops that do computation.
 # Variables maintain state across executions of the graph.
 # placeholder: "feed"-operations, must be called with "run(feed,feed_dict)"
 ###########################################################################################

    with graph.as_default():
      # Input data.
      train_inputs = tf.placeholder(tf.int32, shape=[batch_size])      # Training vector for variable batch ( ndarray<Integer>)
      train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])   # Training vector for variable labels ( ndarray<String>)
      valid_dataset = tf.constant(valid_examples, dtype=tf.int32)      # predict sample nearest neighbors array for int-values in valid_examples

      # tensor of shape (50000,128) filled with random uniform values between -1 and 1
      embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
      # output weights ( NCE == noise-contrastive training objective.)
      nce_weights = tf.Variable(
          tf.truncated_normal([vocabulary_size, embedding_size],
              stddev=1.0 / math.sqrt(embedding_size)))
      #bias
      nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

 ###########################################################################################
 # Step 2: Define "computation ops"  as graph nodes
 # computation ops take source ops as input parameters in their constructors
 ###########################################################################################

      # Look up integer representation of embedding word vector for inputs.
      # shape=TensorShape([Dimension(128), Dimension(128)]), dtype=float32)
      embed = tf.nn.embedding_lookup(embeddings, train_inputs)

      # "loss node": try to predict the target word using the noise-contrastive training objective
      # Compute the average NCE loss for the batch.
      # tf.nce_loss automatically draws a new sample of the negative labels each
      # time we evaluate the loss.
      loss = tf.reduce_mean(tf.nn.nce_loss(nce_weights, nce_biases,
            embed, train_labels, num_sampled, vocabulary_size))

      # "gradient node": required to compute and update the parameters, etc.
      # Construct the SGD optimizer using a learning rate of 1.0.  (SGD ==  stochastic gradient descent)
      optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

      # Compute the cosine similarity between minibatch examples and all embeddings.
      norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))

      normalized_embeddings = embeddings / norm

      valid_embeddings = tf.nn.embedding_lookup(
          normalized_embeddings, valid_dataset)

      similarity = tf.matmul(
          valid_embeddings, normalized_embeddings, transpose_b=True)

##############################################################################
# Step 3: Execute the computation-nodes
#         output: numpy ndarray
##############################################################################

    num_steps = 100001
    #num_steps = 1


    # Enter a Session with a "with" block. The Session closes automatically at the end of the with block.
    # Launch the default graph.
    #session = tf.Session()
    with tf.Session(graph=graph) as session:
          # We must initialize all variables before we use them.
          tf.initialize_all_variables().run()
          print("Initialized")

          average_loss = 0
          for step in xrange(num_steps):
            batch_inputs, batch_labels = generate_batch(data,batch_size, num_skips, skip_window)
            """
            feed_dict to push data into the placeholders (top of this page  - train_inputs = tf.placeholder(...) -
            and calling session.run with this new data in a loop.
            """
            feed_dict = {train_inputs : batch_inputs, train_labels : batch_labels}


            """
            Execute the computation nodes (1) optimizer (2) loss

            feed_dict: substituting the values in feed_dict for the corresponding placeholders.

            We perform one update step by evaluating the optimizer op (including it
            in the list of returned values for session.run()

            '_' : don't care variable. In this case, the output of optimizer-computation is not relevant and stored in variable _
            """
            _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
            average_loss += loss_val

            if step % 10000 == 0:
              if step > 0:
                average_loss = average_loss / 10000
              # The average loss is an estimate of the loss over the last 2000 batches.
              print("Average loss at step ", step, ": ", average_loss)
              average_loss = 0

          final_embedings_similarity = similarity.eval()  # similarity vector
          final_embeddings_normalized = normalized_embeddings.eval()

           #persist to disk
          if persist == True:
               pickle.dump(final_embedings_similarity,open('output/final_embedings_similarity.pickle','wb'))
               pickle.dump(final_embeddings_normalized,open('output/final_embeddings_normalized.pickle','wb'))

    return final_embedings_similarity, final_embeddings_normalized
