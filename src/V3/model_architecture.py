# Classes and functions from the Keras library which is used for building and training deep learning models in Python.
# from keras.models import load_model
# from keras.models import model_from_json
# from keras.models import Sequential
# from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3


def construct_model(input_shape, num_classes, name):
    
    # Creating a sequential model. A sequential model is a linear stack of layers, where the output of one layer is the input of the next.
#     model = Sequential(name=name)

#     # Add a convolutional layer with 32 filters, a kernel size of 3x3, and a ReLU activation function. 
#     # The ReLU activation function is a simple equation that takes the input of a neuron and returns the input if it is positive, and returns 0 if it is negative.
#     model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape)) # input is a 28x28 image with 1 color channel.

#     # Add a max pooling layer with a pool size of 2x

#     # This layer applies a max operation over a 2x2 window of the input, reducing the spatial dimensions of the input by half.
#     model.add(MaxPooling2D(pool_size=(2, 2)))

#     # Add a convolutional layer with 64 filters, a kernel size of 3x3, and a ReLU activation function
#     model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))

#     # Add a max pooling layer with a pool size of 2x2
#     model.add(MaxPooling2D(pool_size=(2, 2)))

#     # Flatten the output from the previous layers
#     model.add(Flatten())

#     # Add a dropout layer to prevent overfitting
#     model.add(Dropout(0.5))

#     # Add a fully connected layer with 128 units and a ReLU activation function. This layer has 128 neurons and it is fully connected to the previous layer
#     model.add(Dense(128, activation='relu'))

#     # Add a final output layer with num_classes number of units and a softmax activation function The softmax function is used to convert the output of the final layer into probability distribution over 10 possible classes.
#     model.add(Dense(num_classes, activation='softmax'))

#     # Complete model 
#     model.summary()
    
    
    model = InceptionV3(include_top=False, input_shape=input_shape, weights=None)
    
    # Freeze the weights of the pre-trained layers
    for layer in model.layers:
        layer.trainable = False
    
     # Add a convolutional layer with 32 filters, a kernel size of 3x3, and a ReLU activation function. 
    # The ReLU activation function is a simple equation that takes the input of a neuron and returns the input if it is positive, and returns 0 if it is negative.
    x = model.output
    x = Conv2D(32, kernel_size=(3, 3), activation='relu')(x)
    
     # Add a max pooling layer with a pool size of 2x2
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Add a convolutional layer with 64 filters, a kernel size of 3x3, and a ReLU activation function
    x = Conv2D(64, kernel_size=(3, 3), activation='relu')(x)

    # Add a max pooling layer with a pool size of 2x2
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Flatten the output from the previous layers
    x = Flatten()(x)

    # Add a dropout layer to prevent overfitting
    x = Dropout(0.5)(x)

    # Add a fully connected layer with 128 units and a ReLU activation function. This layer has 128 neurons and it is fully connected to the previous layer
    x = Dense(128, activation='relu')(x)
    
     # Add a final output layer with num_classes number of units and a softmax activation function The softmax function is used to convert the output of the final layer into probability distribution over 10 possible classes.
    output = Dense(num_classes, activation='softmax')(x)
    
    
    # Define the final model
    model = Model(inputs=model.input, outputs=output)
    
     # Complete model 
    # model.summary()
    
#     #Load the pre-trained ResNet50 model
#     pretrained_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)

#     # Freeze the weights of the pre-trained layers
#     for layer in pretrained_model.layers:
#         layer.trainable = False

#     # Add new layers to the model for your specific task
#     x = pretrained_model.output
#     x = Flatten()(x)
#     x = Dense(128, activation='relu')(x)
#     output = Dense(num_classes, activation='softmax')(x)

#     # Define the final model
#     model = tf.keras.models.Model(inputs=pretrained_model.input, outputs=output)


    
    return model