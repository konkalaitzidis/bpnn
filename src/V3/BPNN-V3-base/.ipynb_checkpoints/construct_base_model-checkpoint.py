from keras.models import Sequential
from keras.layers import Dense, Flatten


def base_model(num_classes, name, input_shape):
    
    model = Sequential(name=name)
    model.add(Flatten(input_shape=input_shape))
    model.add(Dense(units=128, activation='relu'))
    model.add(Dense(units=num_classes, activation='softmax'))
    
    return model