from tensorflow.keras.utils import to_categorical

def model_preprocessing(
    images, 
    labels, 
    df_new_annotations_unique, 
    # Convert pixel values to float32
):
    images = images.astype('float32')

    # Finding number of classes and converting labels to categorical values
    # Converting labels to categorical.
    labels = to_categorical(labels, num_classes)
    
    return images, labels