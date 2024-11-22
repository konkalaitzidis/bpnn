import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import datetime


def check_class_imbalance_k_fold(train_class_counts, 
                                 test_class_counts, 
                                 fold, 
                                 num_folds, 
                                 experiment_ID, 
                                 save_dir, 
                                 df_new_annotations_unique, 
                                 df_new_annotations_check, 
                                 train_labels_names, 
                                 no_of_labels):
    

    fig = plt.figure(figsize=(8, 6))
    plt.bar(train_class_counts.index, train_class_counts.values, color='b', alpha=0.5, label='Train')
    plt.bar(test_class_counts.index, test_class_counts.values, color='r', alpha=0.5, label='Test')
    plt.xlabel('Class Labels')
    plt.ylabel('Number of Instances')
    plt.title(f'Distribution of Class Labels in Fold {fold}/{num_folds}')
    plt.xticks(np.arange(len(train_labels_names)), train_labels_names, rotation=90, fontsize=8)
    # plt.xticks(np.arange(len(train_class_counts.index)), train_class_counts.index)
    plt.legend()
    
    return fig
    

    
def check_class_imbalance_old_merged(df_new_annotations, 
                                     experiment_ID, 
                                     save_dir, 
                                     df_new_annotations_unique, 
                                     df_new_annotations_check, 
                                     no_of_labels, 
                                     names_of_labels, 
                                     data_file, 
                                     label_names):
    """
    Check class imbalance in the old merged dataset and plot the distribution.

    Parameters:
    df_new_annotations (pd.DataFrame): DataFrame containing new annotations.
    experiment_ID (str): ID of the experiment.
    save_dir (str): Directory to save the plots.
    df_new_annotations_unique (pd.DataFrame): DataFrame containing unique new annotations.
    df_new_annotations_check (pd.DataFrame): DataFrame to check new annotations.
    no_of_labels (int): Number of labels.
    names_of_labels (list): List of label names.
    data_file (str): Name of the data file.
    label_names (list): List of label names.

    Returns:
    tuple: class_counts (pd.Series), total_counts (int)
    """

    total_counts = 0
    class_counts = pd.value_counts(df_new_annotations_check['state_id'])
    for i in range(len(df_new_annotations_unique)):
        total_counts = total_counts + class_counts.get(i, 0)
        total_counts = total_counts + class_counts.get(i, 0)

    # calculate the percentage of each class in the dataset
    class_percents = pd.value_counts(df_new_annotations, normalize=True) * 100

    # create a bar chart of class percentages using Seaborn
    sns.set_style("whitegrid")
    plt.figure(figsize=(4, 5))
    f = sns.barplot(x=class_percents.index, y=class_percents.values, palette='Paired')
    plt.xlabel('Behaviour Labels', fontsize = 12)
    plt.ylabel('Percentage of Instances', fontsize = 12)
    plt.title('Distribution of Class Labels (n='+str(no_of_labels)+'), '+str(data_file))

    f.set_xticklabels(names_of_labels)

    patches = f.patches
    for i in range(len(patches)):
        x = patches[i].get_x() + patches[i].get_width()/2
        y = patches[i].get_height()+.5
        f.annotate('{:.1f}%'.format(class_percents[i]), (x, y), ha='center', fontsize = 10)
    
    # Get the current timestamp
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    # Save the plot with the timestamp appended to the file name
    plt.savefig(f"{save_dir}/{experiment_ID}_class_distribution_{timestamp}.png", dpi=600, bbox_inches='tight')
    
    # plt.savefig(f"{save_dir}/{experiment_ID}_class_distribution.png", dpi=100, bbox_inches='tight')
    plt.show()
    class_counts = pd.value_counts(df_new_annotations_check['state_id'])
    total_counts = total_counts + class_counts.get(i, 0)
    
    
    

def check_class_imbalance_old(df_new_annotations, 
                              experiment_ID, 
                              save_dir, 
                              df_new_annotations_unique, 
                              df_new_annotations_check, 
                              no_of_labels, 
                              data_file, 
                              label_names):
    """
    Check class imbalance in the old dataset and plot the distribution.

    Parameters:
    df_new_annotations (pd.DataFrame): DataFrame containing new annotations.
    experiment_ID (str): ID of the experiment.
    save_dir (str): Directory to save the plots.
    df_new_annotations_unique (pd.DataFrame): DataFrame containing unique new annotations.
    df_new_annotations_check (pd.DataFrame): DataFrame to check new annotations.
    no_of_labels (int): Number of labels.
    data_file (str): Name of the data file.
    label_names (list): List of label names.

    Returns:
    tuple: class_counts (pd.Series), total_counts (int)
    """
    
    total_counts = 0
    for i in range(len(df_new_annotations_unique)):
        class_counts = pd.value_counts(df_new_annotations_check['state_id'])
        total_counts = total_counts + class_counts[i]

    # calculate the percentage of each class in the dataset
    class_percents = pd.value_counts(df_new_annotations, normalize=True) * 100

     # create a bar chart of class percentages using Seaborn
    sns.set_style("whitegrid")
    plt.figure(figsize=(8, 6))
    f = sns.barplot(x=class_percents.index, y=class_percents.values, color="#6495ED")
    plt.xlabel('Behaviour Labels', fontsize = 12)
    plt.ylabel('Percentage of Instances', fontsize = 12)
    plt.title('Distribution of Class Labels (n='+str(no_of_labels)+'), '+str(data_file))
    
    f.set_xticklabels(df_new_annotations_check['state_name'].unique(), rotation = 90)

    patches = f.patches
    for i in range(len(patches)):
        x = patches[i].get_x() + patches[i].get_width()/2
        y = patches[i].get_height()+.5
        f.annotate('{:.1f}%'.format(class_percents[i]), (x, y), ha='center', fontsize = 6)
    
    plt.axhline(y=10, color='gray', linestyle='--')
        # Get the current timestamp
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    # Save the plot with the timestamp appended to the file name
    plt.savefig(f"{save_dir}/{experiment_ID}_class_distribution_{timestamp}.svg", dpi=600, bbox_inches='tight')
    
    plt.show()
    
    return class_counts, total_counts




def check_class_imbalance_new(df_new_annotations, 
                              experiment_ID, 
                              save_dir, 
                              df_new_annotations_unique, 
                              df_new_annotations_check,
                              no_of_labels, 
                              data_file, 
                              label_names):
    """
    Check class imbalance in the new dataset and plot the distribution.

    Parameters:
    df_new_annotations (pd.DataFrame): DataFrame containing new annotations.
    experiment_ID (str): ID of the experiment.
    save_dir (str): Directory to save the plots.
    df_new_annotations_unique (pd.DataFrame): DataFrame containing unique new annotations.
    df_new_annotations_check (pd.DataFrame): DataFrame to check new annotations.
    no_of_labels (int): Number of labels.
    data_file (str): Name of the data file.
    label_names (list): List of label names.

    Returns:
    tuple: class_counts (pd.Series), total_counts (int)
    """
    
    total_counts = 0
    for i in range(len(df_new_annotations_unique)):
        class_counts = pd.value_counts(df_new_annotations_check['state_id'])
        total_counts = total_counts + class_counts[i]

    # calculate the percentage of each class in the dataset
    class_percents = pd.value_counts(df_new_annotations, normalize=True) * 100

    
     # create a bar chart of class percentages using Seaborn
    sns.set_style("whitegrid")
    plt.figure(figsize=(8, 6))
    f = sns.barplot(x=class_percents.index, y=class_percents.values, palette='viridis')
    plt.xlabel('Class Labels', fontsize=12)
    plt.ylabel('Percentage of Instances', fontsize=12)
    plt.title('Distribution of Class Labels (n='+str(no_of_labels)+'), '+str(data_file))
    # rename the x-axis labels
    # for i in range(len(df_new_annotations_unique)):
    f.set_xticklabels(label_names)
    

    patches = f.patches
    for i in range(len(patches)):
        x = patches[i].get_x() + patches[i].get_width()/2
        y = patches[i].get_height()+.5
        f.annotate('{:.1f}%'.format(class_percents[i]), (x, y), ha='center')
    
    
    # Get the current timestamp
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    # Save the plot with the timestamp appended to the file name
    plt.savefig(f"{save_dir}/{experiment_ID}_class_distribution_{timestamp}.png", dpi=100, bbox_inches='tight')
    
    # plt.savefig(f"{save_dir}/{experiment_ID}_class_distribution.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    return class_counts, total_counts
