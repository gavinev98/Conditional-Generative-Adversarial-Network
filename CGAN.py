import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation, Input
from keras.layers import Conv2D, Conv2DTranspose
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model
from keras.preprocessing import image
import numpy as np
import os
from PIL import Image
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

        # Inputs will be 128 rows , 128 cols, channels 3, as recommended.
        inputs = Input((128, 128, 3))
        # Layer 1
        conv1 = Conv2D(32, (7, 7), strides=(1, 1), padding="same")(inputs)
        conv1 = BatchNormalization()(conv1)
        conv1 = Activation("relu")(conv1)

        # Layer 2
        conv2 = Conv2D(64, (3, 3), strides=(2, 2), padding="same")(conv1)
        conv2 = BatchNormalization()(conv2)
        conv2 = Activation("relu")(conv2)

        # Layer 3
        conv3 = Conv2D(128, (3, 3), strides=(2, 2), padding="same")(conv2)
        conv3 = BatchNormalization()(conv3)
        conv3 = Activation("relu")(conv3)

        # Layer 4
        conv4 = Conv2D(128, (3, 3), padding="same")(conv3)
        conv4 = BatchNormalization()(conv4)
        conv4 = Activation("relu")(conv4)

        # Layer 5
        conv5 = Conv2D(128, (3, 3), padding="same")(conv4)
        conv5 = BatchNormalization()(conv5)
        conv5 = Activation("relu")(conv5)

        # Layer 6
        conv6 = Conv2D(128, (3, 3), padding="same")(conv5)
        conv6 = BatchNormalization()(conv6)
        conv6 = Activation("relu")(conv6)

        # Transpose Layers
        transpose1 = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding="same")(conv6)
        transpose1 = BatchNormalization()(transpose1)
        transpose1 = Activation("relu")(transpose1)

        # Transpose Layers
        transpose2 = Conv2DTranspose(32, (3, 3), strides=(2, 2), padding="same")(transpose1)
        transpose2 = BatchNormalization()(transpose2)
        transpose2 = Activation("relu")(transpose2)

        # Layer 7
        conv7 = Conv2D(3, (3, 3), strides=(1, 1), padding="same")(transpose2)
        conv7 = BatchNormalization()(conv7)
        conv7 = Activation("relu")(conv7)

        model = Model(inputs=[inputs], outputs=[conv7])
        print(model.summary())
        return model



def discriminator():

    # Discriminator to also take same input structure.
    image_input = (128, 128)
    input = Input(image_input, 3)

    convolution1 = Conv2D(64, (4, 4), strides=(2, 2))(input)
    convolution1 = Activation(LeakyReLU(alpha=.2))(convolution1)

    convolution2 = Conv2D(64, (4, 4), strides=(2, 2))(input)
    convolution2 = Activation(LeakyReLU(alpha=.2))(convolution2)

    convolution3 = Conv2D(64, (4, 4), strides=(2, 2))(input)
    convolution3 = BatchNormalization()(convolution1)
    convolution3 = Activation(LeakyReLU(alpha=.2))(convolution1)

    convolution4 = Conv2D(64, (4, 4), strides=(2, 2))(input)
    convolution4 = BatchNormalization()(convolution1)
    convolution4 = Activation(LeakyReLU(alpha=.2))(convolution1)

    finalOutput = Flatten()(convolution4)
    finalOutput = Dense(1, activation='sigmoid')(finalOutput)


def createfullmode(generator, discriminator):

        # Setting the inputs
        inputs = Input((128, 128, 3))

        # Acquire the generator
        generator_model = generator(inputs)
        # Acquire the inputs for the discriminator
        discriminator_model = discriminator(inputs)

        # Setting the trainable to false for discriminator.
        discriminator_model.trainable = False

        # Acquiring the output produced by the discriminator (probability)
        output_gan = discriminator_model(generator_model)

        # create the model
        # input is the noise and output is the probability of real or fake.
        ganfinal_Model = Model(inputs=inputs, outputs=output_gan)

        return ganfinal_Model













if __name__ == "__main__":
    generator()



