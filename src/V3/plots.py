import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from keras.models import load_model
# for generating a confusion matrix
from sklearn.metrics import confusion_matrix, f1_score
from matplotlib.gridspec import GridSpec
# from save_model_info import save_training_info
from skimage.transform import resize
from sklearn.metrics import confusion_matrix
from matplotlib.backends.backend_pdf import PdfPages


#=============================================#


# def plot_cm_k_fold(conf_matrices, no_of_behaviors, num_classes, experiment_ID, model_cm_dir):

def plot_cm_k_fold(model_cm_dir, fold, cm, conf_matrices, train_labels_names):

    print("Plotting confusion matrix\n")
    
    with PdfPages(model_cm_dir+'/Confusion_Matrices_Per_Fold.pdf') as pdf:
        # plot the confusion matrix for each fold
        for fold, cm in enumerate(conf_matrices):
            print(f"Plotting the confusion matrix for fold {fold}")
            f = plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
            plt.title(f'Confusion Matrix - Fold {fold}')
            plt.xlabel('Predicted Labels')
            plt.xticks(np.arange(len(train_labels_names)) + 0.5, train_labels_names, rotation=90, fontsize=6)
            plt.ylabel('True Labels')
            plt.yticks(np.arange(len(train_labels_names)) + 0.5, train_labels_names, rotation=0, fontsize=6)
            # Save the plot in the pdf
            pdf.savefig(f, bbox_inches='tight')
            plt.show()
    

    # no_of_behaviors = no_of_behaviors
    # n_classes = num_classes   
    # mean_conf_matrix = np.mean(conf_matrices, axis=0)
    # plt.figure(figsize=(8, 6))
    # sns.heatmap(mean_conf_matrix, annot=True, cmap='Blues', fmt='g', xticklabels=no_of_behaviors, yticklabels=no_of_behaviors)
    # plt.title('Confusion Matrix - K-fold, Location Labels')
    # plt.xlabel('Predicted Labels')
    # plt.ylabel('True Labels')
    # plt.savefig(model_cm_dir+"/"+'cm_val_'+str(experiment_ID)+'.svg', bbox_inches='tight', dpi=300)
    # plt.show()

#=============================================#

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
    plt.savefig(base_model_cm_dir+"/"+'cm_train_'+str(experiment_ID)+'.svg', bbox_inches='tight', dpi=300)
    plt.show()
    
    # Repeat for validation data
    cm_val = confusion_matrix(val_labels, val_predicted_labels)

    # Plot the confusion matrix as a heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_val, annot=True, cmap='Blues', fmt='g', xticklabels=no_of_behaviors, yticklabels=no_of_behaviors)
    plt.title('Confusion Matrix - Validation, Location Labels')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.savefig(base_model_cm_dir+"/"+'cm_val_'+str(experiment_ID)+'.svg', bbox_inches='tight', dpi=300)
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

#=============================================#

# def plot_accuracy_k_fold(experiment_ID, model_acc_dir, train_acc_all, val_acc_all, num_folds, num_epochs):
#     fig, axs = plt.subplots(nrows=num_folds, figsize=(8, 6*num_folds)) 
#     sns.set_style('whitegrid')
#     sns.set_palette('husl')
        
#     for i in range(num_folds):
#         axs[i].plot(range(len(train_acc_all[i])), train_acc_all[i], label=f'Train Acc')
#         axs[i].plot(range(len(val_acc_all[i])), val_acc_all[i], label=f'Val Acc')
#         axs[i].set_title(f'Fold {i+1}', fontsize=14)
#         axs[i].set_xlabel('Epoch', fontsize=12)
#         axs[i].set_ylabel('Accuracy', fontsize=12)
#         axs[i].legend(loc='lower right')
    
#     plt.suptitle('Training and Validation Accuracy Per Fold', fontsize=16)
#     plt.savefig(model_acc_dir+"/"+"training-validation-accuracy_"+str(experiment_ID)+".svg", bbox_inches='tight', dpi=300)
#     plt.show()


def plot_accuracy_k_fold(experiment_ID, model_acc_dir, train_acc_all, val_acc_all, num_folds, num_epochs, min_length):
    fig = plt.figure(figsize=(4, 3)) 
    sns.set_style('whitegrid')
    sns.set_palette('husl')
    
#     # Find the minimum length of the train and validation accuracy lists
#     min_length = min([len(train_acc_all[i]) for i in range(num_folds)] + [len(val_acc_all[i]) for i in range(num_folds)])
    
    # Create arrays to hold the averaged results
    train_acc_avg = np.zeros(min_length)
    val_acc_avg = np.zeros(min_length)
    
    
    # Average the results across all folds
    for i in range(num_folds):
        train_acc_fold = np.array(train_acc_all[i][:min_length])
        val_acc_fold = np.array(val_acc_all[i][:min_length])
        train_acc_avg += train_acc_fold
        val_acc_avg += val_acc_fold
        
        # # Plot individual fold results
        # sns.lineplot(x=range(min_length), y=train_acc_fold, label=f'Train Acc Fold {i+1}', alpha=0.7)
        # sns.lineplot(x=range(min_length), y=val_acc_fold, label=f'Val Acc Fold {i+1}', alpha=0.7)
        
    train_acc_avg /= num_folds
    val_acc_avg /= num_folds
    
    # Plot the averaged results
    sns.lineplot(x=range(min_length), y=train_acc_avg, label='Train Acc', linewidth=2.5, color='#880454')
    sns.lineplot(x=range(min_length), y=val_acc_avg, label='Val Acc', linewidth=2.5, color='#2596be')
    
    plt.title('Training and Validation Accuracy (5-Fold CV)', fontsize=8)
    plt.xlabel('Epoch', fontsize=6)
    plt.ylabel('Accuracy', fontsize=6)
    plt.legend(loc='lower right')
    plt.savefig(model_acc_dir+"/"+"training-validation-accuracy_"+str(experiment_ID)+".svg", bbox_inches='tight', dpi=300)
    plt.show()
    
    return train_acc_avg, val_acc_avg
    
    
#=============================================#

def plot_loss_k_fold(experiment_ID, model_loss_dir, train_loss_all, val_loss_all, num_folds, num_epochs, min_length):

    fig, ax = plt.subplots(figsize=(4, 3))
    sns.set_style('whitegrid')
    sns.set_palette('husl')
    
    
    # Create arrays to hold the averaged results
    train_loss_avg = np.zeros(min_length)
    val_loss_avg = np.zeros(min_length)
    
    # Average the results across all folds
    for i in range(num_folds):
        train_loss_fold = np.array(train_loss_all[i][:min_length])
        val_loss_fold = np.array(val_loss_all[i][:min_length])
        train_loss_avg += train_loss_fold
        val_loss_avg += val_loss_fold
        
        # # Plot individual fold results
        # sns.lineplot(x=range(min_length), y=train_loss_fold, label=f'Train Loss Fold {i+1}', alpha=0.7)
        # sns.lineplot(x=range(min_length), y=val_loss_fold, label=f'Val Loss Fold {i+1}', alpha=0.7)
    
    train_loss_avg /= num_folds
    val_loss_avg /= num_folds
    
    # Plot the averaged results
    sns.lineplot(x=range(min_length), y=train_loss_avg, label='Train Loss', linewidth=2.5, color='#880454')
    sns.lineplot(x=range(min_length), y=val_loss_avg, label='Val Loss', linewidth=2.5, color='#2596be')
    
    
    plt.title('Training and Validation Loss (5-Fold CV)', fontsize=8)
    plt.xlabel('Epoch', fontsize=6)
    plt.ylabel('Loss', fontsize=6)
    plt.legend(loc='lower right')
    plt.savefig(model_loss_dir+"/"+"training-validation-loss_"+str(experiment_ID)+".svg", bbox_inches='tight', dpi=300)
    plt.show()

    return train_loss_avg, val_loss_avg
    
    # for i in range(num_folds):
    #     sns.lineplot(x=range(num_epochs), y=train_loss_all[i], label=f'Train Loss Fold {i+1}')
    #     sns.lineplot(x=range(num_epochs), y=val_loss_all[i], label=f'Val Loss Fold {i+1}')
    # plt.title('Training and Validation Loss Per Fold', fontsize=14)
    # plt.xlabel('Epoch', fontsize=12)
    # plt.ylabel('Loss', fontsize=12)
    # # ax.set(title='Training and Validation Accuracy', xlabel='Epoch', ylabel='Accuracy')
    # plt.legend(loc='lower right')
    # plt.savefig(model_loss_dir+"/"+"training-validation-loss_"+str(experiment_ID)+".svg", bbox_inches='tight', dpi=300)
    # plt.show()
    
    
    
# def plot_accuracy_k_fold(experiment_ID, model_acc_dir, train_acc_all, val_acc_all, num_folds, num_epochs):
#     fig = plt.figure(figsize=(12, 6))
#     sns.set_style('whitegrid')
#     sns.set_palette('husl')
    
#     # Find the minimum length of the train and validation accuracy lists
#     min_length = min([len(train_acc_all[i]) for i in range(num_folds)] + [len(val_acc_all[i]) for i in range(num_folds)])
    
#     # Create arrays to hold the averaged results
#     train_acc_avg = np.zeros(min_length)
#     val_acc_avg = np.zeros(min_length)
    
#     # Average the results across all folds
#     for i in range(num_folds):
#         train_acc_fold = np.array(train_acc_all[i][:min_length])
#         val_acc_fold = np.array(val_acc_all[i][:min_length])
#         train_acc_avg += train_acc_fold
#         val_acc_avg += val_acc_fold
        
#         # Plot individual fold results
#         sns.lineplot(x=range(min_length), y=train_acc_fold, label=f'Train Acc Fold {i+1}', alpha=0.7)
#         sns.lineplot(x=range(min_length), y=val_acc_fold, label=f'Val Acc Fold {i+1}', alpha=0.7)
    
#     train_acc_avg /= num_folds
#     val_acc_avg /= num_folds
    
#     # Plot the averaged results
#     sns.lineplot(x=range(min_length), y=train_acc_avg, label='Train Acc', linewidth=2.5)
#     sns.lineplot(x=range(min_length), y=val_acc_avg, label='Val Acc', linewidth=2.5)
    
#     plt.title('Training and Validation Accuracy', fontsize=14)
#     plt.xlabel('Epoch', fontsize=12)
#     plt.ylabel('Accuracy', fontsize=12)
#     plt.legend(loc='lower right')
#     plt.savefig(model_acc_dir+"/"+"training-validation-accuracy_"+str(experiment_ID)+".svg", bbox_inches='tight', dpi=300)
#     plt.show()

    
    
    
#=============================================#
    
def plot_average_accuracy_k_fold(experiment_ID, model_average_acc_dir, train_acc_all, val_acc_all, num_folds, num_epochs):
    fig = plt.figure(figsize=(8, 6)) 
    sns.set_style('whitegrid')
    sns.set_palette('husl')
    # for i in range(num_folds):
    #     sns.lineplot(x=range(num_epochs), y=train_acc_all[i], label=f'Train Acc Fold {i+1}')
    #     sns.lineplot(x=range(num_epochs), y=val_acc_all[i], label=f'Val Acc Fold {i+1}')
    mean_train_acc = np.mean(train_acc_all, axis=0)
    mean_val_acc = np.mean(val_acc_all, axis=0)
    sns.lineplot(x=range(num_epochs), y=mean_train_acc, label='Average Train Acc')
    sns.lineplot(x=range(num_epochs), y=mean_val_acc, label='Average Val Acc')
    plt.title('Average Training and Validation Accuracy', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend(loc='lower right')
    plt.savefig(model_average_acc_dir+"/"+"average-training-validation-accuracy_"+str(experiment_ID)+".svg", bbox_inches='tight', dpi=300)
    plt.show()
    return mean_train_acc, mean_val_acc
    

    

#=============================================#

def plot_average_loss_k_fold(experiment_ID, model_average_loss_dir, train_loss_all, val_loss_all, num_folds, num_epochs):

    fig = plt.figure(figsize=(8, 6)) 
    sns.set_style('whitegrid')
    sns.set_palette('husl')
    # for i in range(num_folds):
    #     sns.lineplot(x=range(num_epochs), y=train_acc_all[i], label=f'Train Acc Fold {i+1}')
    #     sns.lineplot(x=range(num_epochs), y=val_acc_all[i], label=f'Val Acc Fold {i+1}')
    mean_train_loss = np.mean(train_loss_all, axis=0)
    mean_val_loss = np.mean(val_loss_all, axis=0)
    sns.lineplot(x=range(num_epochs), y=mean_train_loss, label='Average Train Loss')
    sns.lineplot(x=range(num_epochs), y=mean_val_loss, label='Average Val Loss')
    plt.title('Average Training and Validation Loss', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(loc='lower right')
    plt.savefig(model_average_loss_dir+"/"+"average-training-validation-loss_"+str(experiment_ID)+".svg", bbox_inches='tight', dpi=300)
    return mean_train_loss, mean_val_loss
    plt.show()

#=============================================#



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

        axes[i].imshow(images[index], vmin = vmin, vmax = vmax) #,  interpolation = 'none'
        label_name = labels[indices[i]]

        # if label_name == 0:
        #     label_name = "Main Corr"
        # elif label_name == 1:
        #     label_name = "Left Corr"
        # else:
        #     label_name = "Right Corr"

        if label_name == 0:
            label_name = "Grooming"
        elif label_name == 1:
            label_name = "Frozen"
        elif label_name == 2:
            label_name = "Not Moving"
        elif label_name == 3:
            label_name = "Moving"
        elif label_name == 4:
            label_name = "Right Turn"            
        else:
            label_name = "Left Turn"
            
        # Convert binary labels to the desired format
        if isinstance(labels[indices[i]], np.ndarray):
            label_name = np.argmax(labels[indices[i]])
        axes[i].set_title("Label: " + str(label_name), fontsize=12)
        # axes[i].tick_params(axis='both', which='both', length=0)
        # axes[i].set_xticklabels([])
        # axes[i].set_yticklabels([])
        # plt.subplots_adjust(wspace=0.05, hspace=0)
    plt.tight_layout()
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
        
        # if label_name == 0:
        #     label_name = "Main Corr"
        # elif label_name == 1:
        #     label_name = "Left Corr"
        # else:
        #     label_name = "Right Corr"
        
        if label_name == 0:
            label_name = "Grooming"
        elif label_name == 1:
            label_name = "Frozen"
        elif label_name == 2:
            label_name = "Not Moving"
        elif label_name == 3:
            label_name = "Moving"
        elif label_name == 4:
            label_name = "Right Turn"            
        else:
            label_name = "Left Turn"

        # Convert binary labels to the desired format
        if isinstance(labels[indices[i]], np.ndarray):
            label_name = np.argmax(labels[indices[i]])
        axes[i].set_title("Label: " + str(label_name), fontsize=12)
        axes[i].tick_params(axis='both', which='both', length=0)
        axes[i].set_xticklabels([])
        axes[i].set_yticklabels([])

    # plt.subplots_adjust(wspace=0.05, hspace=0)
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
