import csv  # CSV module is used for working with CSV (Comma Separated Values) files in Python.
# from f1_score import f1_score
import os
import pickle
import time  # for time-related functions
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


import tensorflow.compat.v1 as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.model_selection import KFold
# Train and evaluate the model for each fold
from keras.models import load_model

# import NWBDataGenerator and use it in your code
from nwb_data_generator import NWBDataGeneratorTime, NWBDataGenerator
from model_architecture import construct_model
from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix, f1_score



from plots import plot_cm_k_fold
from class_balance import check_class_imbalance_k_fold

import gc

from matplotlib.backends.backend_pdf import PdfPages


def run_k_fold(params,
               train_loss_all,
               val_loss_all,
               train_acc_all,
               val_acc_all,
               average_score_list,
               experiment_ID):
    
    
    images = params['images']
    labels = params['labels']
    num_folds = params['number_of_folds']
    shuffle = params['shuffle_data']
    input_shape = params['input_shape']
    num_classes = params['number_of_classes']
    name = params['model_name']
    epochs = params['epochs']
    no_of_behaviors = params['behaviours'],
    df_new_annotations = params['df_new_annotations']
    df_new_annotations_unique = params['unique_annotations']
    df_new_annotations_check = params['check_annotations']
    df_new_annotations_names = params['label_names']
    output_dir = params['output_directory']
 

    
    # Define the EarlyStopping callback to stop training when the validation loss stops decreasing
    early_stopping = EarlyStopping(monitor='val_loss', patience=6, verbose=1, mode='min')

    # Define the KFold cross-validator
    kf = KFold(n_splits=num_folds, shuffle=shuffle)
    
    conf_matrices = []
    
    import matplotlib.backends.backend_pdf as pdf_backend

    
    with PdfPages('class_distributions.pdf') as pdf:

        for fold, (train_idx, test_idx) in enumerate(kf.split(images), 1):


            # loop control
            print(f'\n\n\nFold {fold}/{num_folds}\n')

            # if fold == 4:
            #     break

            # train_idx, test_idx = indices[fold]

            save_dir = output_dir
            
            
            train_labels_fold = df_new_annotations[train_idx]
            test_labels_fold = df_new_annotations[test_idx]
            
            train_labels_names = df_new_annotations_names
            # train_labels_names = df_new_annotations_check['state_name'].unique()
            # test_labels_names = df_new_annotations_check['state_name']
                        
            train_class_counts = train_labels_fold.value_counts()
            test_class_counts = test_labels_fold.value_counts()

            no_of_labels = len(train_labels_names)
            
            
            fig = check_class_imbalance_k_fold(train_class_counts, test_class_counts, fold, num_folds, experiment_ID, save_dir, df_new_annotations_unique, df_new_annotations_check, train_labels_names, no_of_labels)
            

            # Save the plot in the pdf
            pdf.savefig(fig, bbox_inches='tight')
            plt.show()



            print("Splitting data with NWBDataGenerator\n")

            # Define the data generators for the training and validation sets
            train_generator = NWBDataGeneratorTime(images, labels, train_idx)
            val_generator = NWBDataGeneratorTime(images, labels, test_idx)

            print("Length of training and validation sets:", len(train_idx), len(test_idx))



            # construct model
            print("Creating Model\n")
            model = construct_model(input_shape, num_classes, name)

            # reset the weights.    

            # Training model
            print("Training model. Go grab a coffee or take a walk.")

            # history = model.fit(train_images, train_labels, epochs=epochs, batch_size=batch_size, validation_data=(val_images, val_labels), callbacks=[early_stopping]) # callbacks=[early_stopping])

            history = model.fit(train_generator, 
                                epochs=epochs,
                                validation_data=val_generator,
                                callbacks=[early_stopping])




            # all_histories.append(history)  # save the history object to the list


             # Append the training and validation loss and accuracy values to the corresponding lists
            train_loss_all.append(np.array(history.history['loss']))
            val_loss_all.append(np.array(history.history['val_loss']))
            train_acc_all.append(np.array(history.history['accuracy']))
            val_acc_all.append(np.array(history.history['val_accuracy']))


            # Evaluate the model on the validation set


            print("\nEvaluating model...")
            accuracy_score = model.evaluate(val_generator, verbose=0)
            print(f'Validation loss: {accuracy_score[0]:.4f}')
            print(f'Validation accuracy: {accuracy_score[1]:.4f}\n')
            average_score_list.append(accuracy_score[1])



            # accuracy_score_list.append(accuracy_score)



            # get the true labels and predicted labels for the validation set
            print("Generating predictions on validation data")
            true_labels = val_generator.get_labels()
            predictions = model.predict(val_generator)

            y_true = np.array([np.argmax(x) for x in true_labels])
            y_pred = np.array([np.argmax(x) for x in predictions])

            # print(np.mean(y_true == y_pred))

            # calculate the confusion matrix for this fold
            cm = confusion_matrix(y_true, y_pred)
            conf_matrices.append(cm)


            model_cm_dir = "/home/dmc/Desktop/kostas/direct-Behavior-prediction-from-miniscope-calcium-imaging-using-convolutional-neural-networks/src/V3/output/cm"

            # plot_cm_k_fold(val_generator, no_of_behaviors, experiment_ID, model_cm_dir, model)

            # del train_images, train_labels, val_images, val_labels, model, history, accuracy_score
            del train_generator, val_generator, model, history, accuracy_score
            gc.collect()
        
    
    print("\nDone!\n")
    
    # no_of_behaviors = ['Main Corr', 'Left Corr', 'Right Corr']
    
    
    
    
    
    model_cm_dir = "/home/dmc/Desktop/kostas/direct-Behavior-prediction-from-miniscope-calcium-imaging-using-convolutional-neural-networks/src/V3/output/cm"
    
    plot_cm_k_fold(conf_matrices, no_of_behaviors, num_classes, experiment_ID, model_cm_dir)
        
    return train_loss_all, val_loss_all, train_acc_all, val_acc_all, average_score_list, conf_matrices
 
    

    
    
    
    
    
    
    
#==== run k-fold without time element ====#
    
def run_k_fold_basic(params,
               train_loss_all,
               val_loss_all,
               train_acc_all,
               val_acc_all,
               average_score_list,
               experiment_ID):
    
    # Initializing parameters
    images = params['images']
    labels = params['labels']
    num_folds = params['number_of_folds']
    shuffle = params['shuffle_data']
    input_shape = params['input_shape']
    num_classes = params['number_of_classes']
    name = params['model_name']
    epochs = params['epochs']
    no_of_behaviors = params['behaviours']
    df_new_annotations = params['df_new_annotations']
    df_new_annotations_unique = params['unique_annotations']
    df_new_annotations_check = params['check_annotations']
    df_new_annotations_names = params['label_names']
    output_dir = params['output_directory']

    
    # Define the EarlyStopping callback to stop training when the validation loss stops decreasing
    early_stopping = EarlyStopping(monitor='val_loss', patience=6, verbose=1, mode='min')

    # Define the KFold cross-validator
    kf = KFold(n_splits=num_folds, shuffle=shuffle)
    
    
    conf_matrices = []
    f1_score_val_list = []
    
    model_balance_dir = "/home/dmc/Desktop/kostas/direct-Behavior-prediction-from-miniscope-calcium-imaging-using-convolutional-neural-networks/src/V3/"+str(output_dir)+"/balance"
    
    
    with PdfPages(model_balance_dir+'/class_distributions.pdf') as pdf:
        for fold, (train_idx, test_idx) in enumerate(kf.split(images), 1):

            # loop control
            print(f'\n\n\nFold {fold}/{num_folds}\n')
            
            # save balance output
            save_dir = "/home/dmc/Desktop/kostas/direct-Behavior-prediction-from-miniscope-calcium-imaging-using-convolutional-neural-networks/src/V3/"+str(output_dir)+"/balance"           
            
            train_labels_fold = df_new_annotations[train_idx]
            test_labels_fold = df_new_annotations[test_idx]
            
            train_labels_names = df_new_annotations_names
            # train_labels_names = df_new_annotations_check['state_name'].unique()
            # test_labels_names = df_new_annotations_check['state_name']
                        
            train_class_counts = train_labels_fold.value_counts()
            test_class_counts = test_labels_fold.value_counts()

            no_of_labels = no_of_behaviors #no_of_behaviors
            
            
            fig = check_class_imbalance_k_fold(train_class_counts, test_class_counts, fold, num_folds, experiment_ID, save_dir, df_new_annotations_unique, df_new_annotations_check, train_labels_names, no_of_labels)
            

#             # Set the file name and full path of the PDF file
#             file_name = str(experiment_ID)+'_class_distribution_.pdf'
#             file_path = os.path.join(save_dir, file_name)

#             # Save the figure as a PDF in the specified directory
#             fig.savefig(file_path, bbox_inches='tight')
#             plt.show()


            # Save the plot in the pdf
            pdf.savefig(fig, bbox_inches='tight')
            plt.show()
            
            print("Splitting data with NWBDataGenerator\n")

            # Define the data generators for the training and validation sets
            train_generator = NWBDataGenerator(images, labels, train_idx)
            val_generator = NWBDataGenerator(images, labels, test_idx)

            print("Length of training and validation sets:", len(train_idx), len(test_idx))



            # construct model
            print("Creating Model\n")
            model = construct_model(input_shape, num_classes, name)

            

            # Training model
            print("Training model. Go grab a coffee or take a walk.")

            # history = model.fit(train_images, train_labels, epochs=epochs, batch_size=batch_size, validation_data=(val_images, val_labels), callbacks=[early_stopping]) # callbacks=[early_stopping])

            history = model.fit(train_generator, 
                                epochs=epochs,
                                validation_data=val_generator,
                                callbacks=[early_stopping])

            # all_histories.append(history)  # save the history object to the list


             # Append the training and validation loss and accuracy values to the corresponding lists
            train_loss_all.append(np.array(history.history['loss']))
            val_loss_all.append(np.array(history.history['val_loss']))
            train_acc_all.append(np.array(history.history['accuracy']))
            val_acc_all.append(np.array(history.history['val_accuracy']))


            # Evaluate the model on the validation set
            print("\nEvaluating model...")
            accuracy_score = model.evaluate(val_generator, verbose=0)
            print(f'Validation loss: {accuracy_score[0]:.4f}')
            print(f'Validation accuracy: {accuracy_score[1]:.4f}\n')
            average_score_list.append(accuracy_score[1])

            # accuracy_score_list.append(accuracy_score)

            # get the true labels and predicted labels for the validation set
            print("Generating predictions on validation data")
            true_labels = val_generator.get_labels()
            predictions = model.predict(val_generator)

            y_true = np.array([np.argmax(x) for x in true_labels])
            y_pred = np.array([np.argmax(x) for x in predictions])

            # print(np.mean(y_true == y_pred))

            # calculate the confusion matrix for this fold
            cm = confusion_matrix(y_true, y_pred)
            conf_matrices.append(cm)
            
            # find the f1 score
            f1_score_val = f1_score(y_true, y_pred, average='micro')
            print("F1 score is: {:.3f}" .format(f1_score_val))
            f1_score_val_list.append(f1_score_val)
            
            
            # del train_images, train_labels, val_images, val_labels, model, history, accuracy_score
            del train_generator, val_generator, model, history, accuracy_score
            gc.collect()
    
    print("\nDone!\n")
    
    
    # plot_cm_k_fold(conf_matrices, no_of_behaviors, num_classes, experiment_ID, model_cm_dir)
        
    # Plot Confusion Matrices for all Folds
    
    # specify the file path where you want to save the PDF
    model_cm_dir = "/home/dmc/Desktop/kostas/direct-Behavior-prediction-from-miniscope-calcium-imaging-using-convolutional-neural-networks/src/V3/"+str(output_dir)+"/cm"
    

    # create a PdfPages object to save the figures
    
    plot_cm_k_fold(model_cm_dir, fold, cm, conf_matrices, train_labels_names)
    
            
    # plot the mean confusion matrix
    # mean_cm = np.mean(conf_matrices, axis=0)
    # plt.figure(figsize=(8, 6))
    # sns.heatmap(mean_cm, annot=True, cmap='Blues', fmt='g')
    # plt.title('Mean Confusion Matrix - K-Fold Cross Validation')
    # plt.xlabel('Predicted Labels')
    # plt.xticks(np.arange(len(train_labels_names)), train_labels_names, rotation=90, fontsize=6)
    # plt.ylabel('True Labels')
    # plt.yticks(np.arange(len(train_labels_names)), train_labels_names, rotation=0, fontsize=6)
    # plt.show()
    
    
    # dir_name = "/home/dmc/Desktop/kostas/direct-Behavior-prediction-from-miniscope-calcium-imaging-using-convolutional-neural-networks/src/V3/output/pickles"
  
    return train_loss_all, val_loss_all, train_acc_all, val_acc_all, average_score_list, conf_matrices, f1_score_val_list


    
    
    

    # plot_confusion_matrix(experiment_ID, no_of_behaviors, train_generator, val_generator, train_labels, val_labels, train_images, val_images, model_cm_dir, model_path, model_version)

    
    
    
    
    
    
#     dir_name = "/home/dmc/Desktop/kostas/direct-Behavior-prediction-from-miniscope-calcium-imaging-using-convolutional-neural-networks/src/V2/output/pickles"
#     model_save_name = str(model_version)+'.h5'
#     print("Now saving model data to pickles. Please wait...")
#     model.save(f"{save_dir}/{model_save_name}")
    
#     dir_name_pickles = "src/V2/output/pickels"
#     # Save the history object to a pickle file
#     with open(os.path.join(save_dir, 'history.pkl'), 'wb') as f:
#         pickle.dump(history.history, f)
    
#     with open(os.path.join(save_dir, 'train_images.pkl'), 'wb') as f:
#         pickle.dump(train_images, f)

#     with open(os.path.join(save_dir, 'val_images.pkl'), 'wb') as f:
#         pickle.dump(val_images, f)

#     with open(os.path.join(save_dir, 'train_labels.pkl'), 'wb') as f:
#         pickle.dump(train_labels, f)
        
#     with open(os.path.join(save_dir, 'val_labels.pkl'), 'wb') as f:
#         pickle.dump(val_labels, f)
        
#     print("Done!")

    
#     return history