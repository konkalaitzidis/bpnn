from keras.utils import to_categorical

def model_preprocessing(images, 
                        labels, 
                        df_new_annotations_unique, 
                        num_classes):
    # ensuring that the pixel values are float numbers. This is a common preprocessing step for image data
    images = images.astype('float32')

    # Finding number of classes and converting labels to categorical values
    # Converting labels to categorical.
    labels = to_categorical(labels, num_classes)
    
    return images, labels