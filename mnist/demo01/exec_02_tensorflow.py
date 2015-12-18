import tensorflow as tf
import input_data
import numpy as np
import math
from sys import argv
from array import array
import cv2

"""
a placeholder for our image data: None stands for an unspecified number of images 784 = 28*28 pixel
"""
x = tf.placeholder("float", [None, 784])

# we need our weights for our neural net
W = tf.Variable(tf.zeros([784,10]))
# and the biases
b = tf.Variable(tf.zeros([10]))

"""
softmax provides a probability based output we need to multiply the image values x and the weights
and add the biases (the normal procedure, explained in previous articles)
"""
y = tf.nn.softmax(tf.matmul(x,W) + b)

"""
y_ will be filled with the real values which we want to train (digits 0-9)
for an undefined number of images
"""
y_ = tf.placeholder("float", [None,10])

# initialize all variables
init = tf.initialize_all_variables()

# create a session
sess = tf.Session()
sess.run(init)


###############################################################################
#Train on mnist dataset
###############################################################################

# create a MNIST_data folder with the MNIST dataset if necessary
mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True)

"""
we use the cross_entropy function which we want to minimize to improve our model
"""
cross_entropy = -tf.reduce_sum(y_*tf.log(y))

"""
use a learning rate of 0.01 to minimize the cross_entropy error
"""
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# use 1000 batches with a size of 100 each to train our net
for i in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  # run the train_step function with the given image values (x) and the real output (y_)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})


###############################################################################
# manipulate custom image to fit requirements
###############################################################################

# create an array where we can store our 1 pictures
images = np.zeros((1,784))

# load normalized image
input_file = "../numbers/handwritten/pro-img/basic/manualInput/singleFileInputImage.png"
gray = cv2.imread(input_file, cv2.CV_LOAD_IMAGE_GRAYSCALE)
flatten = gray.flatten()

images[0] = flatten


###############################################################################
# evaluate custom re-fitted image to predict handwritten value
###############################################################################



"""
the prediction will be an array with one value, which show the predicted number
"""
prediction = tf.argmax(y,1)
#print "%s" % (prediction)
"""
we want to run the prediction and the accuracy function using our generated arrays (images and correct_vals)
"""
print sess.run(prediction, feed_dict={x: images})
#print sess.run(accuracy, feed_dict={x: images, y_: correct_vals})
