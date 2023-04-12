# Classes and functions from the Keras library which is used for building and training deep learning models in Python.
from keras.models import load_model
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers import Dropout
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense


def construct_model(input_shape, num_classes, name):
    
    # Creating a sequential model. A sequential model is a linear stack of layers, where the output of one layer is the input of the next.
    model = Sequential(name=name)

    # Add a convolutional layer with 32 filters, a kernel size of 3x3, and a ReLU activation function. 
    # The ReLU activation function is a simple equation that takes the input of a neuron and returns the input if it is positive, and returns 0 if it is negative.
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape)) # input is a 28x28 image with 1 color channel.

    # Add a max pooling layer with a pool size of 2x

    # This layer applies a max operation over a 2x2 window of the input, reducing the spatial dimensions of the input by half.
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Add a convolutional layer with 64 filters, a kernel size of 3x3, and a ReLU activation function
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))

    # Add a max pooling layer with a pool size of 2x2
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Flatten the output from the previous layers
    model.add(Flatten())

    # Add a dropout layer to prevent overfitting
    model.add(Dropout(0.5))

    # Add a fully connected layer with 128 units and a ReLU activation function. This layer has 128 neurons and it is fully connected to the previous layer
    model.add(Dense(128, activation='relu'))

    # Add a final output layer with num_classes number of units and a softmax activation function The softmax function is used to convert the output of the final layer into probability distribution over 10 possible classes.
    model.add(Dense(num_classes, activation='softmax'))

    # # Complete model 
    # model.summary()
    
    return model