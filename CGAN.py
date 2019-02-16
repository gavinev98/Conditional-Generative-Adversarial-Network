import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation, Input
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Conv2DTranspose, concatenate, merge, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
import keras.layers.advanced_activations
from keras.models import Model
import pandas as pd
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from keras.preprocessing import image
import pandas as pd
import numpy as np
from scipy.misc import imresize
import glob
import os
import cv2
from PIL import Image
from keras.applications.vgg16 import VGG16
from keras.optimizers import RMSprop
from keras.applications.vgg16 import preprocess_input
import glob

photo_path = "/Users/Gavin Everett/Desktop/GavinsCGANFYP/FullyFormattedPhotos"
sketches_path = "/Users/Gavin Everett/Desktop/GavinsCGANFYP/FullyFormattedSketch"


listPhotos = os.listdir(photo_path)
listSketches = os.listdir(sketches_path)

# Creating arrays to hold reformatted names for images
images_arr = []
sketch = []

# Loop of the directory of images and store each in array.
for photos in glob.glob(photo_path + '\\*'):

        im = Image.open(photos)
        tempImage1 = image.img_to_array(im)
        images_arr.append(tempImage1)

        images_arr = np.array(images_arr)

        images_arr = images_arr.astype('float32')

        # acquire the mean
        meanOfImage = np.mean(images_arr)
        stdOfImage = np.std(images_arr)

        images_arr = (images_arr.astype(np.float32) - 127.5) / 127.5


# Loop over directory of sketches and store each in array.
for sketches in glob.glob(sketches_path + '\\*'):

        im = Image.open(sketches)
        tempImage2 = image.img_to_array(im)

        sketch.append(tempImage2)

        sketch = np.array(sketch)

        sketch = sketch.astype('float32')

        # acquire the mean
        meanOfSketch = np.mean(sketch)
        stdOfSketch = np.std(sketch)

        sketch = (sketch.astype(np.float32) - 127.5) / 127.5


        print(sketch)


def generator():

        # Defining the shape of the images to be input into the generator network.
        image_input=(128, 128, 3)
        img_rows = 128, 128
        img_cols = 128, 128

        # Using input method to feed in the rows and columns.
        # Each of the layers in the model will contain a conv layer, bias, and activation layer.
        inputs = Input(img_rows, img_cols, 3)
        convolution1 = Conv2D(32, (7, 7), strides=(1, 1), padding="same")(inputs)
        convolution1 = BatchNormalization()(convolution1)
        convolution1 = Activation("relu")(convolution1)

        convolution2 = Conv2D(64, (3, 3), strides=(1, 1), padding="same")(convolution1)
        convolution2 = BatchNormalization()(convolution2)
        convolution2 = Activation("relu")(convolution2)
        
        convolution3 = Conv2D(128, (3,3), strides=(2, 2), padding="same")(convolution2)
        convolution3 = BatchNormalization()(convolution3)
        convolution3 = Activation("relu")(convolution3)

        test1 = Conv2D(128, (3, 3), strides=(2, 2), padding="same")(convolution3)
        test1 = BatchNormalization()(test1)
        test1 = Activation("relu")(test1)
        test2 = Conv2D(128, (3, 3), strides=(2, 2), padding="same")(test1)
        test2 = BatchNormalization()(test1)
        test2 = Activation("relu")(test1)
        # Merging layers together using keras function
        merging1 = keras.layers.add([test1, test2])

        test1 = Conv2D(128, (3, 3), strides=(2, 2), padding="same")(merging1)
        test1 = BatchNormalization()(test1)
        test1 = Activation("relu")(test1)
        test2 = Conv2D(128, (3, 3), strides=(2, 2), padding="same")(test1)
        test2 = BatchNormalization()(test1)
        test2 = Activation("relu")(test1)
        # Merging layers together using keras function
        merging2 = keras.layers.add([test1, test2])

        test1 = Conv2D(128, (3, 3), strides=(2, 2), padding="same")(merging2)
        test1 = BatchNormalization()(test1)
        test1 = Activation("relu")(test1)
        test2 = Conv2D(128, (3, 3), strides=(2, 2), padding="same")(test1)
        test2 = BatchNormalization()(test1)
        test2 = Activation("relu")(test1)
        # Merging layers together using keras function
        merging3 = keras.layers.add([test1, test2])

        test1 = Conv2D(128, (3, 3), strides=(2, 2), padding="same")(merging3)
        test1 = BatchNormalization()(test1)
        test1 = Activation("relu")(test1)
        test2 = Conv2D(128, (3, 3), strides=(2, 2), padding="same")(test1)
        test2 = BatchNormalization()(test1)
        test2 = Activation("relu")(test1)
        # Merging layers together using keras function
        merging4 = keras.layers.add([test1, test2])

        test1 = Conv2D(128, (3, 3), strides=(2, 2), padding="same")(merging4)
        test1 = BatchNormalization()(test1)
        test1 = Activation("relu")(test1)
        test2 = Conv2D(128, (3, 3), strides=(2, 2), padding="same")(test1)
        test2 = BatchNormalization()(test1)
        test2 = Activation("relu")(test1)
        # Merging layers together using keras function
        merging5 = keras.layers.add([test1, test2])

        test1 = Conv2D(128, (3, 3), strides=(2, 2), padding="same")(merging5)
        test1 = BatchNormalization()(test1)
        test1 = Activation("relu")(test1)
        test2 = Conv2D(128, (3, 3), strides=(2, 2), padding="same")(test1)
        test2 = BatchNormalization()(test1)
        test2 = Activation("relu")(test1)
        # Merging layers together using keras function
        merging6 = keras.layers.add([test1, test2])

        transpose1 = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding="same")(merging6)
        transpose1 = BatchNormalization()(transpose1)
        transpose1 = Activation("relu")(transpose1)

        transpose2 = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding="same")(transpose1)
        transpose2 = BatchNormalization()(transpose1)
        transpose2 = Activation("relu")(transpose2)


























