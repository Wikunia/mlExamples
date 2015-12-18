import input_data
import cv2
import numpy as np
import math
from scipy import ndimage
from sys import argv
from array import array
import struct

# Extract mnist images
def extract_images():
     images_file_path = "../MNIST_data/extract/t10k-images-idx3-ubyte"
     images_save_folder = "../MNIST_data/extract/files"
     with open(images_file_path, "rb") as images_file:
        # 32 bit integer magic number
        images_file.read(4)
        # 32 bit integer number of images
        images_file.read(4)
        # 32 bit number of rows
        images_file.read(4)
        # 32 bit number of columns
        images_file.read(4)
        # every image contain 28 x 28 = 784 byte, so read 784 bytes each time
        count = 1
        image = np.zeros((28, 28, 1), np.uint8)
        image_bytes = images_file.read(784)
        while image_bytes:
            image_unsigned_char = struct.unpack("=784B", image_bytes)
            for i in range(784):
                image.itemset(i, image_unsigned_char[i])
            image_save_path = "%s/%d.png" % (images_save_folder, count)
            cv2.imwrite(image_save_path, image)
            print(image_save_path)
            image_bytes = images_file.read(784)
            count += 1


#Extract the images
extract_images()
