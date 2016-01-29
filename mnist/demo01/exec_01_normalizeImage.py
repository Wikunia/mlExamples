import input_data
import cv2
import numpy as np
import math
from scipy import ndimage
from sys import argv
from array import array
import argparse


##############################
# helper to plot the image
##############################

def show(image):
    """
    Render a given numpy.uint8 2D array of pixel data.
    """
    fig = pyplot.figure()
    ax = fig.add_subplot(1,1,1)
    imgplot = ax.imshow(image, cmap=mpl.cm.Greys)
    imgplot.set_interpolation('nearest')
    ax.xaxis.set_ticks_position('top')
    ax.yaxis.set_ticks_position('left')
    pyplot.show()

###############################################################################
# manipulate custom image to fit MNIST requirements
###############################################################################

#command line options


parser = argparse.ArgumentParser()
parser.add_argument("-i" , "--input", type=str, help="input filename")
args = parser.parse_args()
inputFileName  = args.input

# template for 28x28 image box
def getBestShift(img):
    cy,cx = ndimage.measurements.center_of_mass(img)

    rows,cols = img.shape
    shiftx = np.round(cols/2.0-cx).astype(int)
    shifty = np.round(rows/2.0-cy).astype(int)
    return shiftx,shifty


def shift(img,sx,sy):
    rows,cols = img.shape
    M = np.float32([[1,0,sx],[0,1,sy]])
    shifted = cv2.warpAffine(img,M,(cols,rows))
    return shifted

i = 0
# read the image in an array
gray = cv2.imread(inputFileName, cv2.CV_LOAD_IMAGE_GRAYSCALE)

# rescale it, and convert black-linecolor to white-linecolor
gray = cv2.resize(255-gray, (28, 28))

#show(gray)


"""
 the trained images are white digits on a black background.
 our current images are white digets on a gray background.
 therefore we need change the background from gray to black.
"""
(thresh, gray) = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)


"""
put the pictures in a 28x28 box aligned in the center
"""
while np.sum(gray[0]) == 0:
    gray = gray[1:]

while np.sum(gray[:,0]) == 0:
    gray = np.delete(gray,0,1)

while np.sum(gray[-1]) == 0:
    gray = gray[:-1]

while np.sum(gray[:,-1]) == 0:
    gray = np.delete(gray,-1,1)

rows,cols = gray.shape

if rows > cols:
    factor = 20.0/rows
    rows = 20
    cols = int(round(cols*factor))
    # first cols than rows
    gray = cv2.resize(gray, (cols,rows))
else:
    factor = 20.0/cols
    cols = 20
    rows = int(round(rows*factor))
    # first cols than rows
    gray = cv2.resize(gray, (cols, rows))

colsPadding = (int(math.ceil((28-cols)/2.0)),int(math.floor((28-cols)/2.0)))
rowsPadding = (int(math.ceil((28-rows)/2.0)),int(math.floor((28-rows)/2.0)))
#gray = np.lib.pad(gray,(rowsPadding,colsPadding),'constant')
(thresh, gray) = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
gray = np.lib.pad(gray,(rowsPadding,colsPadding),'constant',constant_values=0)

shiftx,shifty = getBestShift(gray)
shifted = shift(gray,shiftx,shifty)
gray = shifted


# save the processed images
cv2.imwrite("../numbers/handwritten/pro-img/basic/manualInput/singleFileInputImage.png", gray)
