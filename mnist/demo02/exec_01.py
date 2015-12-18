import input_data
import cv2
import numpy as np
import math
from scipy import ndimage
from sys import argv
from array import array
import argparse
from tensorflow.python.platform import gfile
from matplotlib import pyplot
import matplotlib as mpl
import tensorflow as tf

#Load original "4" from MNIST data set and provide it as input to tensorflow


inputFileName = "../MNIST_data/extract/files/9998.png"

i = 0
'''
Must be loaded in GRAYSCALE, else error: "ValueError: could not broadcast input array from shape (2352) into shape (784)"
"gray.size" will be "2352" if not loaded in "GRAYSCALE" because of RGB channles.
'''
gray = cv2.imread(inputFileName,cv2.CV_LOAD_IMAGE_GRAYSCALE)
height, width = gray.shape
print "height: %s   width: %s   size: %s" % (height, width, gray.size)

# save the processed images
cv2.imwrite("../numbers/handwritten/pro-img/basic/manualInput/singleFileInputImage.png", gray)


"""
all images in the training set have an range from 0-1
and not from 0-255 so we divide our flatten images
(a one dimensional vector with our 784 pixels)
to use the same 0-1 based range
"""
flatten = gray.flatten() # / (1.0 - 255.0 )

print "flatten.size: %s" % flatten.size

#############################################
# ML starts
#############################################


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
