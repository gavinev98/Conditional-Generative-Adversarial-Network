import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation, Input
from keras.layers import Conv2D, Conv2DTranspose
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model
from keras.preprocessing import image
from keras import backend as K
import numpy as np
import os
from PIL import Image
import glob

photo_path = "/Users/Gavin Everett/Desktop/GavinsCGANFYP/photo/"
sketches_path = "/Users/Gavin Everett/Desktop/GavinsCGANFYP/sketch/"


listPhotos = [photo_path + i for i in os.listdir(photo_path)]
listSketches = [sketches_path + i for i in os.listdir(sketches_path)]

# Creating arrays to hold reformatted names for images
images_arr = []
sketch = []

# Defining number of epochs
num_of_epochs = 25
# Defining batch size.
batch_size = 25

# Loop of the directory of images and store each in array.
for i in range(len(listPhotos)):

        #Get Image
        store_Images = listPhotos[i];

        image_1 = image.load_img(store_Images, target_size=(128,128,3))

        image_1 = image.img_to_array(image_1)

        images_arr.append(image_1)

images_arr = np.array(images_arr)

images_arr = images_arr.astype('float32')

images_arr = (images_arr.astype(np.float32) - 128) / 128


# Loop over directory of sketches and store each in array.
for i in range(len(listSketches)):

        #Get Image
        store_Sketch = listSketches[i];


        image_2 = image.load_img(store_Sketch, target_size=(128,128,3))

        sketch.append(image_2)

sketch = np.array(sketch)

sketch = sketch.astype('float32')

# Rescaling images between -1 to 1.
sketch = (sketch.astype(np.float32) - 128) / 128


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

    # Layer 1
    convolution1 = Conv2D(64, (4, 4), strides=(2, 2))(input)
    convolution1 = Activation(LeakyReLU(alpha=.2))(convolution1)
    # Layer 2
    convolution2 = Conv2D(64, (4, 4), strides=(2, 2))(convolution1)
    convolution2 = Activation(LeakyReLU(alpha=.2))(convolution2)
    # Layer 3
    convolution3 = Conv2D(64, (4, 4), strides=(2, 2))(convolution2)
    convolution3 = BatchNormalization()(convolution3)
    convolution3 = Activation(LeakyReLU(alpha=.2))(convolution3)
    # Layer 4
    convolution4 = Conv2D(64, (4, 4), strides=(2, 2))(convolution3)
    convolution4 = BatchNormalization()(convolution4)
    convolution4 = Activation(LeakyReLU(alpha=.2))(convolution4)
    # Output Layer / flatten and sigmoid activation used.

    finalOutput = Flatten()(convolution4)
    finalOutput = Dense(1, activation='sigmoid')(finalOutput)

    #create model
    model = Model(inputs=[input], outputs=[finalOutput])

    return model;

def createfullmodel(generator_input, discriminator_input):
    # https: // www.datacamp.com / community / tutorials / generative - adversarial - networks
        # Setting the inputs
        inputs = Input((128, 128, 3))

        # Acquire the generator
        generator_model = generator_input(inputs)

        # Setting the trainable to false for discriminator.
        discriminator_input.trainable = False

        # Acquire the inputs for the discriminator
        discriminator_model = discriminator_input(generator_model)

        # create the model
        # input is the noise and output is the probability of real or fake.
        ganfinal_Model = Model(inputs=inputs, outputs=[generator_model, discriminator_model])

        return ganfinal_Model


# https://keras.io/losses/
def mean_squared_error(y_true, y_pred):
    return 10 * K.mean(K.square(y_pred - y_true), axis=-1)


def training_loop():

    # defining the looping structure
        # Looping over with number of epochs currently set to 128.
    for epoch in range(num_of_epochs):

        # Splitting training data into batches of 128 as specified.
        batch_count = sketch.shape[0] // batch_size

        #  Defining a batch from the sketches and a batch from the real data.
        for batch in range(batch_count):

            # Acquiring the generator network.
            generator_input = generator()

            # Get a batch of size 128 from the sketches from generator.
            sketch_x = sketch[batch * batch_size:(batch + 1)*[batch_size]]

            # Generate fake images using prediction
            generate_images = generator_input.predict(sketch_x, verbose=1);

            







                



        # Acquire the training data


def loss_functions():















if __name__ == "__main__":
    generator()



