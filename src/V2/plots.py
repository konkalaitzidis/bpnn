import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
# for generating a confusion matrix
from sklearn.metrics import confusion_matrix
from keras.models import load_model



def plot_confusion_matrix(x, no_of_behaviors, train_labels, val_labels, train_images, val_images):
    
    dir_path = "/home/dmc/Desktop/kostas/direct-Behavior-prediction-from-miniscope-calcium-imaging-using-convolutional-neural-networks/src/V2/cm"
    
    # format the train_labels and val_labels to be appropriate for the predict method
    # Reshape train_labels and val_labels if they have only one dimension
    if len(train_labels.shape) == 1:
        train_labels = np.reshape(train_labels, (-1, 1))
    if len(val_labels.shape) == 1:
        val_labels = np.reshape(val_labels, (-1, 1))

    train_labels = np.argmax(train_labels, axis=1)
    val_labels = np.argmax(val_labels, axis=1)

    model = load_model('BPNN_V2_model.h5')
    
    # Predict the class labels of the training images using the trained model
    train_predicted_labels = np.argmax(model.predict(train_images), axis=1)
    # Predict the class labels of the validation images using the trained model
    val_predicted_labels = np.argmax(model.predict(val_images), axis=1)
    

    # Calculate the confusion matrix for training data
    cm_train = confusion_matrix(train_labels, train_predicted_labels)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_train, annot=True, cmap='Blues', fmt='g', xticklabels=no_of_behaviors, yticklabels=no_of_behaviors)
    plt.title('Confusion Matrix - Training')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.savefig(dir_path+"/"+'cm_train'+str(x)+'.png', bbox_inches='tight', dpi=300)
    plt.show()
    
    # Repeat for validation data
    cm_val = confusion_matrix(val_labels, val_predicted_labels)

    # Plot the confusion matrix as a heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_val, annot=True, cmap='Blues', fmt='g', xticklabels=no_of_behaviors, yticklabels=no_of_behaviors)
    plt.title('Confusion Matrix - Validation')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.savefig(dir_path+"/"+'cm_val'+str(x)+'.png', bbox_inches='tight', dpi=300)
    plt.show()
    

    

    
def plot_accuracy(x, history):
    dir_path = "/home/dmc/Desktop/kostas/direct-Behavior-prediction-from-miniscope-calcium-imaging-using-convolutional-neural-networks/src/V2/accuracy"

    plt.plot(history['accuracy'])
    plt.plot(history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig(dir_path+"/"+'model_accuracy_'+str(x)+'.png', bbox_inches='tight', dpi=300)
    return plt.show()


def plot_loss(x, history):
    dir_path = "/home/dmc/Desktop/kostas/direct-Behavior-prediction-from-miniscope-calcium-imaging-using-convolutional-neural-networks/src/V2/loss"

    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.savefig(dir_path+"/"+'loss_'+str(x)+'.png', bbox_inches='tight', dpi=300)
    return plt.show()