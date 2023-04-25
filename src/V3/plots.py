import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from keras.models import load_model
# for generating a confusion matrix
from sklearn.metrics import confusion_matrix, f1_score
from matplotlib.gridspec import GridSpec
from save_model_info import save_training_info
from skimage.transform import resize



def plot_confusion_matrix(experiment_ID, no_of_behaviors, train_labels, val_labels, train_images, val_images, base_model_cm_dir, model_path, model_version):
    
    dir_path = "/home/dmc/Desktop/kostas/direct-Behavior-prediction-from-miniscope-calcium-imaging-using-convolutional-neural-networks/src/V2/output/cm"
    
    # format the train_labels and val_labels to be appropriate for the predict method
    # Reshape train_labels and val_labels if they have only one dimension
    if len(train_labels.shape) == 1:
        train_labels = np.reshape(train_labels, (-1, 1))
    if len(val_labels.shape) == 1:
        val_labels = np.reshape(val_labels, (-1, 1))

    train_labels = np.argmax(train_labels, axis=1)
    val_labels = np.argmax(val_labels, axis=1)

    model = load_model(model_path+"/"+model_version+".h5")
    
    # Predict the class labels of the training images using the trained model
    train_predicted_labels = np.argmax(model.predict(train_images), axis=1)
    # Predict the class labels of the validation images using the trained model
    val_predicted_labels = np.argmax(model.predict(val_images), axis=1)
    

    # Calculate the confusion matrix for training data
    cm_train = confusion_matrix(train_labels, train_predicted_labels)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_train, annot=True, cmap='Blues', fmt='g', xticklabels=no_of_behaviors, yticklabels=no_of_behaviors)
    plt.title('Confusion Matrix - Training, Location Labels')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.savefig(base_model_cm_dir+"/"+'cm_train_'+str(experiment_ID)+'.png', bbox_inches='tight', dpi=300)
    plt.show()
    
    # Repeat for validation data
    cm_val = confusion_matrix(val_labels, val_predicted_labels)

    # Plot the confusion matrix as a heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_val, annot=True, cmap='Blues', fmt='g', xticklabels=no_of_behaviors, yticklabels=no_of_behaviors)
    plt.title('Confusion Matrix - Validation, Location Labels')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.savefig(base_model_cm_dir+"/"+'cm_val_'+str(experiment_ID)+'.png', bbox_inches='tight', dpi=300)
    plt.show()
    f1_score_val = f1_score(val_labels, val_predicted_labels, average='micro')
    print("F1 score is: {:.3f}" .format(f1_score_val))
    return f1_score_val
    # from save_model_info import save_training_info
    # return f1_score

    
# def plot_accuracy(x, history):
#     dir_path = "/home/dmc/Desktop/kostas/direct-Behavior-prediction-from-miniscope-calcium-imaging-using-convolutional-neural-networks/src/V2/accuracy"

#     plt.plot(history['accuracy'])
#     plt.plot(history['val_accuracy'])
#     plt.title('Model accuracy, Turning Labels')
#     plt.ylabel('Accuracy')
#     plt.xlabel('Epoch')
#     plt.legend(['Train', 'Validation'], loc='upper left')
#     plt.savefig(dir_path+"/"+'model_accuracy_'+str(x)+'.png', bbox_inches='tight', dpi=300)
#     return plt.show()

def plot_accuracy_k_fold(experiment_ID, model_acc_dir, train_acc_all, val_acc_all, num_folds):
    fig, ax = plt.figure(figsize=(8, 6)) 
    sns.set_style('whitegrid')
    sns.set_palette('husl')
    for i in range(num_folds):
        sns.lineplot(x=range(num_epochs), y=train_acc_all[i], label=f'Train Acc Fold {i+1}',  ax=ax)
        sns.lineplot(x=range(num_epochs), y=val_acc_all[i], label=f'Val Acc Fold {i+1}',  ax=ax)
    # plt.title('Training and Validation Accuracy')
    # plt.xlabel('Epoch')
    # plt.ylabel('Accuracy')
    ax.set(title='Training and Validation Accuracy', xlabel='Epoch', ylabel='Accuracy')
    plt.legend(loc='lower right')
    plt.savefig(model_acc_dir+"/"+"training-validation-Accuracy_"+str(experiment_ID)+".svg", bbox_inches='tight', dpi=300)
    return plt.show()
    


def plot_loss_k_fold(experiment_ID, model_acc_dir, train_acc_all, val_acc_all, num_folds):

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.set_style('whitegrid')
    sns.set_palette('husl')
    for i in range(num_folds):
        sns.lineplot(x=range(num_epochs), y=train_acc_all[i], label=f'Train Acc Fold {i+1}', ax=ax)
        sns.lineplot(x=range(num_epochs), y=val_acc_all[i], label=f'Val Acc Fold {i+1}', ax=ax)
    ax.set(title='Training and Validation Loss', xlabel='Epoch', ylabel='Accuracy')
    plt.legend(loc='lower right')
    plt.savefig(model_acc_dir+"/"+"training-validation-loss_"+str(experiment_ID)+".svg", bbox_inches='tight', dpi=300)
    return plt.show()





def plot_accuracy(experiment_ID, history, base_model_acc_dir, i):
    dir_path = "/home/dmc/Desktop/kostas/direct-Behavior-prediction-from-miniscope-calcium-imaging-using-convolutional-neural-networks/src/V2/output/accuracy"
    
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(10,6))
    sns.lineplot(x=range(1, len(history['accuracy'])+1), y=history['accuracy'], label=f'Train Fold {i+1}')
    sns.lineplot(x=range(1, len(history['val_accuracy'])+1), y=history['val_accuracy'], label=f'Val Fold {i+1}')
    ax.set_title('Model Accuracy '+"Fold"+str(i))
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    plt.savefig(base_model_acc_dir+"/"+"model_accuracy_"+str(experiment_ID)+".png", bbox_inches='tight', dpi=300)
    plt.legend(loc='lower right')
    return plt.show()


def plot_loss(experiment_ID, history, base_model_loss_dir, i):
    dir_path = "/home/dmc/Desktop/kostas/direct-Behavior-prediction-from-miniscope-calcium-imaging-using-convolutional-neural-networks/src/V2/output/loss"
    fig, ax = plt.subplots(figsize=(10,6))
    sns.set_style('whitegrid')
    sns.set_palette('husl')

    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('Model loss, Turning Labels')
    plt.ylabel('loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.savefig(base_model_loss_dir+"/"+'loss_'+str(experiment_ID)+'.png', bbox_inches='tight', dpi=300)
    return plt.show()



def plot_first_frames(images, labels, vmin, vmax, data_file):

    fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(15, 5))
    axes = axes.flatten()
    fig.suptitle("The First 5 Images from "+str(data_file), fontsize=16)

    # Generate a list of 5 random integers between 0 and the length of the images variable
    indices = [0, 1, 2, 3, 4] #np.random.randint(0, len(images), 5) 5

    # Loop over the indices and plot each frame with its corresponding label
    for i, index in enumerate(indices):

        axes[i].imshow(images[index], vmin = vmin, vmax = vmax)
        label_name = labels[indices[i]]

        if label_name == 0:
            label_name = "Main Corr"
        elif label_name == 1:
            label_name = "Left Corr"
        else:
            label_name = "Right Corr"


        # Convert binary labels to the desired format
        if isinstance(labels[indices[i]], np.ndarray):
            label_name = np.argmax(labels[indices[i]])
        axes[i].set_title("Label: " + str(label_name), fontsize=12)
        axes[i].tick_params(axis='both', which='both', length=0)
        axes[i].set_xticklabels([])
        axes[i].set_yticklabels([])



    plt.subplots_adjust(wspace=0.05, hspace=0)
    plt.show()


    
def plot_random_frames(images, labels, vmin, vmax, data_file):

    fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(15, 5))
    axes = axes.flatten()
    fig.suptitle("5 random images from "+str(data_file), fontsize=16)

    # Generate a list of 5 random integers between 0 and the length of the images variable
    indices = np.random.randint(0, len(images), 5)

    # Loop over the indices and plot each frame with its corresponding label
    for i, index in enumerate(indices):
        axes[i].imshow(images[index], vmin = vmin, vmax = vmax)
        label_name = labels[indices[i]]
        
        if label_name == 0:
            label_name = "Main Corr"
        elif label_name == 1:
            label_name = "Left Corr"
        else:
            label_name = "Right Corr"

        # Convert binary labels to the desired format
        if isinstance(labels[indices[i]], np.ndarray):
            label_name = np.argmax(labels[indices[i]])
        axes[i].set_title("Label: " + str(label_name), fontsize=12)
        axes[i].tick_params(axis='both', which='both', length=0)
        axes[i].set_xticklabels([])
        axes[i].set_yticklabels([])

    plt.subplots_adjust(wspace=0.05, hspace=0)
    plt.show()
    # plt.savefig('five_random_frames.png')

    
def plot_image_pixel_values(img):
    downsampled_img = resize(img, (int(img.shape[0]/16), int(img.shape[1]/16)), anti_aliasing=True)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax = sns.heatmap(downsampled_img, cmap="gray", annot=True,  fmt=".2f", linewidth=.5, annot_kws={"fontsize": 6})
    ax.set_title("Image Heatmap")
    ax.set(xlabel="width", ylabel="height")
                


                

# def plot_frames(images, labels, indices, title):
#     # Create a FacetGrid with a 1 row and 5 columns
#     g = sns.FacetGrid(data=pd.DataFrame({'image': images[indices], 'label': labels[indices]}),
#                       col='label', col_wrap=5, height=3, aspect=1)
#     # Map the images to the grid
#     g.map(sns.imshow, 'image', cmap='gray')
#     # Set the title of each subplot
#     g.set_titles('{col_name}')
#     # Set the main title of the plot
#     g.fig.suptitle(title, fontsize=14)
#     # Tighten the layout
#     plt.tight_layout()
#     # Show the plot
#     plt.show()

# def plot_first_frames(images, labels):
#     images_flat = images.reshape(images.shape[0], -1)
#     # Generate a list of 5 random integers between 0 and the length of the images variable
#     indices = [0, 1, 2, 3, 4] #np.random.randint(0, len(images), 5) 5
#     # Plot the frames with corresponding labels
#     plot_frames(images, labels, indices, "First Five Frames")

# def plot_random_frames(images, labels):
#     images_flat = images.reshape(images.shape[0], -1)
#     # Generate a list of 5 random integers between 0 and the length of the images variable
#     indices = np.random.randint(0, len(images), 5)
#     # Plot the frames with corresponding labels
#     plot_frames(images, labels, indices, "Five Random Frames")
