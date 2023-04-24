from keras.utils import to_categorical



# def model_preprocessing(train_images, val_images, train_labels, val_labels, df_new_annotations_unique):
#     # ensuring that the pixel values are float numbers. This is a common preprocessing step for image data
#     train_images = train_images.astype('float32')
#     val_images = val_images.astype('float32')

#     # Finding number of classes and converting labels to categorical values
#     # How many distinct behaviors do we have?
#     no_of_behaviors = df_new_annotations_unique

#     # Define the number of classes
#     num_classes = len(no_of_behaviors)

#     # Converting labels to categorical.
#     train_labels = to_categorical(train_labels, num_classes)
#     val_labels = to_categorical(val_labels, num_classes)
    
#     return train_images, val_images, train_labels, val_labels, num_classes

def model_preprocessing(images, labels, df_new_annotations_unique):
    # ensuring that the pixel values are float numbers. This is a common preprocessing step for image data
    images = images.astype('float32')

    # Finding number of classes and converting labels to categorical values
    # How many distinct behaviors do we have?
    no_of_behaviors = df_new_annotations_unique

    # Define the number of classes
    num_classes = len(no_of_behaviors)

    # Converting labels to categorical.
    labels = to_categorical(labels, num_classes)
    
    return images, labels, num_classes