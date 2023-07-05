# Classes and functions from the Keras library which is used for building and training deep learning models in Python.
# from keras.models import load_model
# from keras.models import model_from_json
from keras.models import Sequential
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input

from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense, UpSampling2D, LSTM, TimeDistributed
import tensorflow as tf
from tensorflow.keras.models import Model
# from tensorflow.keras.applications.inception_v3 import InceptionV3
# from keras.applications.resnet50 import ResNet50



def construct_model(input_shape, num_classes, name):
    
    #Creating a sequential model. A sequential model is a linear stack of layers, where the output of one layer is the input of the next.
#     model = Sequential(name=name)

#     # Add a convolutional layer with 32 filters, a kernel size of 3x3, and a ReLU activation function. 
#     # The ReLU activation function is a simple equation that takes the input of a neuron and returns the input if it is positive, and returns 0 if it is negative.
#     model.add(Conv2D(32, kernel_size=(7, 7), activation='relu', input_shape=input_shape)) # input is a 28x28 image with 1 color channel.

#     # Add a max pooling layer with a pool size of 2x
#     # This layer applies a max operation over a 2x2 window of the input, reducing the spatial dimensions of the input by half.
#     model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
    
#     # Add a convolutional layer with 64 filters, a kernel size of 3x3, and a ReLU activation function
#     model.add(Conv2D(64, kernel_size=(7, 7), activation='relu'))

#     # Add a max pooling layer with a pool size of 2x2
#     model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

#     # Flatten the output from the previous layers
#     model.add(Flatten())

#     # Add a dropout layer to prevent overfitting
#     model.add(Dropout(0.5))

#     # Add a fully connected layer with 128 units and a ReLU activation function. This layer has 128 neurons and it is fully connected to the previous layer
#     model.add(Dense(128, activation='relu'))
    
#     # Add a final output layer with num_classes number of units and a softmax activation function The softmax function is used to convert the output of the final layer into probability distribution over 10 possible classes.
#     model.add(Dense(num_classes, activation='softmax'))

# #     # Complete model 
#     # model.summary()


####### Standard architecture
    print("Compiling model...\n")

    model = Sequential(name=name)
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    

####### Standard architecture with LSTM

    # model = Sequential(name=name)
    # model.add(TimeDistributed(Conv2D(32, kernel_size=(7, 7), activation='relu'), input_shape=input_shape))
    # model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2), strides=2)))
    # model.add(TimeDistributed(Conv2D(64, kernel_size=(7, 7), activation='relu')))
    # model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2), strides=2)))
    # model.add(TimeDistributed(Flatten()))
    # model.add(LSTM(128, activation='relu', return_sequences=True))
    # model.add(Dropout(0.5))
    # model.add(TimeDistributed(Dense(num_classes, activation='softmax')))

    
#######
# Sequential model but slightly more complicated


    # model = Sequential(name=name)
    # model.add(Conv2D(32, kernel_size=(7, 7), activation='relu', input_shape=input_shape))  
    # model.add(Conv2D(32, kernel_size=(7, 7), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Conv2D(64, kernel_size=(7, 7), activation='relu'))
    # model.add(Conv2D(64, kernel_size=(7, 7), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Conv2D(128, kernel_size=(7, 7), activation='relu'))
    # model.add(Conv2D(128, kernel_size=(7, 7), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Flatten())
    # model.add(Dropout(0.5))
    # model.add(Dense(256, activation='relu'))
    # model.add(Dense(128, activation='relu'))
    # model.add(Dense(num_classes, activation='softmax'))



#######

# VGG16

#     model = Sequential(name=name)
#     model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same', input_shape=input_shape))
#     model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
#     model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

#     model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
#     model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
#     model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

#     model.add(Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'))
#     model.add(Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'))
#     model.add(Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'))
#     model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

#     model.add(Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same'))
#     model.add(Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same'))
#     model.add(Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same'))
#     model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

#     model.add(Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same'))
#     model.add(Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same'))
#     model.add(Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same'))
#     model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

#     model.add(Flatten())
#     model.add(Dense(4096, activation='relu'))
#     model.add(Dropout(0.5))
#     model.add(Dense(4096, activation='relu'))
#     model.add(Dropout(0.5))
#     model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.legacy.Adam(), metrics=['accuracy'])
    print("Done :) \n")

    
    
    # ========


    
    
#     # Load the ResNet50 model pre-trained on ImageNet dataset
#     resnet = ResNet50(include_top=False, weights='imagenet', input_shape = (356, 398, 1))

#     # Freeze the weights of the ResNet50 layers to prevent them from being updated during training
#     for layer in resnet.layers:
#         layer.trainable = False

#     # Define the input shape of your model
#     input_shape = input_shape

#     # Define the number of classes in your dataset
#     num_classes = 3

#     # Create a new input layer with the same shape as your grayscale images
#     input_layer = Input(shape=input_shape)

#     # Add a 2D upsampling layer to add two additional color channels to your grayscale images
#     upsample_layer = UpSampling2D(size=(1, 1, 3))(input_layer)

#     # Preprocess the input using the preprocess_input function from the ResNet50 module
#     preprocessed_input = preprocess_input(upsample_layer)

#     # Load the ResNet50 model pre-trained on ImageNet dataset, with input shape (None, None, 3) to match the output of the UpSampling2D layer
#     resnet_model = ResNet50(weights='imagenet', include_top=False, input_shape=(None, None, 3))

#     # Freeze the weights of the ResNet50 layers to prevent them from being updated during training
#     for layer in resnet_model.layers:
#         layer.trainable = False

#     # Connect the preprocessed input to the ResNet50 model
#     resnet_output = resnet_model(preprocessed_input)

#     # Flatten the output of the ResNet50 model
#     flatten_layer = Flatten()(resnet_output)

#     # Add a dense layer with 128 units and ReLU activation function
#     dense_layer = Dense(128, activation='relu')(flatten_layer)

#     # Add a dropout layer to prevent overfitting
#     dropout_layer = Dropout(0.5)(dense_layer)

#     # Add the final output layer with num_classes units and softmax activation function
#     output_layer = Dense(num_classes, activation='softmax')(dropout_layer)

#     # Create a new model that includes the ResNet50 model as a feature extractor
#     model = Model(inputs=input_layer, outputs=output_layer)
    
#     print("Compiling model...\n")
#     # Compile the model with appropriate optimizer, loss function, and metrics
#     model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


    
    
#     model = InceptionV3(include_top=False, input_shape=input_shape, weights=None)
    
#     # Freeze the weights of the pre-trained layers
#     for layer in model.layers:
#         layer.trainable = False
    
#      # Add a convolutional layer with 32 filters, a kernel size of 3x3, and a ReLU activation function. 
#     # The ReLU activation function is a simple equation that takes the input of a neuron and returns the input if it is positive, and returns 0 if it is negative.
#     x = model.output
#     x = Conv2D(32, kernel_size=(3, 3), activation='relu')(x)
    
#      # Add a max pooling layer with a pool size of 2x2
#     x = MaxPooling2D(pool_size=(2, 2))(x)

#     # Add a convolutional layer with 64 filters, a kernel size of 3x3, and a ReLU activation function
#     x = Conv2D(64, kernel_size=(3, 3), activation='relu')(x)

#     # Add a max pooling layer with a pool size of 2x2
#     x = MaxPooling2D(pool_size=(2, 2))(x)

#     # Flatten the output from the previous layers
#     x = Flatten()(x)

#     # Add a dropout layer to prevent overfitting
#     x = Dropout(0.5)(x)

#     # Add a fully connected layer with 128 units and a ReLU activation function. This layer has 128 neurons and it is fully connected to the previous layer
#     x = Dense(128, activation='relu')(x)
    
#      # Add a final output layer with num_classes number of units and a softmax activation function The softmax function is used to convert the output of the final layer into probability distribution over 10 possible classes.
#     output = Dense(num_classes, activation='softmax')(x)
    
    
#     # Define the final model
#     model = Model(inputs=model.input, outputs=output)
    
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