import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import datetime





def check_class_imbalance_k_fold(train_class_counts, test_class_counts, fold, num_folds, experiment_ID, save_dir, df_new_annotations_unique, df_new_annotations_check, train_labels_names, no_of_labels):
    

    fig = plt.figure(figsize=(8, 6))
    plt.bar(train_class_counts.index, train_class_counts.values, color='b', alpha=0.5, label='Train')
    plt.bar(test_class_counts.index, test_class_counts.values, color='r', alpha=0.5, label='Test')
    plt.xlabel('Class Label (n='+str(no_of_labels)+')')
    plt.ylabel('Number of Instances')
    plt.title(f'Distribution of Class Labels in Fold {fold}/{num_folds}')
    plt.xticks(np.arange(len(train_labels_names)), train_labels_names, rotation=90, fontsize=8)
    # plt.xticks(np.arange(len(train_class_counts.index)), train_class_counts.index)
    plt.legend()
    
    return fig
    


    
    
def check_class_imbalance_old_merged(df_new_annotations, experiment_ID, save_dir, df_new_annotations_unique, df_new_annotations_check, no_of_labels, names_of_labels, data_file):

    total_counts = 0
    for i in range(len(df_new_annotations_unique)):
        class_counts = pd.value_counts(df_new_annotations_check['state_id'])
        total_counts = total_counts + class_counts[i]

    # calculate the percentage of each class in the dataset
    class_percents = pd.value_counts(df_new_annotations, normalize=True) * 100

    
    
     # create a bar chart of class percentages using Seaborn
    sns.set_style("whitegrid")
    plt.figure(figsize=(8, 6))
    f = sns.barplot(x=class_percents.index, y=class_percents.values, palette='pastel')
    plt.xlabel('Behaviour Labels')
    plt.ylabel('Percentage of Instances')
    plt.title('Distribution of Class Labels (n='+str(no_of_labels)+'), '+str(data_file))
    # rename the x-axis labels
    # for i in range(len(df_new_annotations_unique)):
    
    f.set_xticklabels(names_of_labels)

    patches = f.patches
    for i in range(len(patches)):
        x = patches[i].get_x() + patches[i].get_width()/2
        y = patches[i].get_height()+.5
        f.annotate('{:.1f}%'.format(class_percents[i]), (x, y), ha='center', fontsize = 10)
    
    # Get the current timestamp
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    # Save the plot with the timestamp appended to the file name
    plt.savefig(f"{save_dir}/{experiment_ID}_class_distribution_{timestamp}.png", dpi=100, bbox_inches='tight')
    
    # plt.savefig(f"{save_dir}/{experiment_ID}_class_distribution.png", dpi=100, bbox_inches='tight')
    plt.show()
    
    return class_counts, total_counts
    
    
    

def check_class_imbalance_old(df_new_annotations, experiment_ID, save_dir, df_new_annotations_unique, df_new_annotations_check, no_of_labels, data_file):
    total_counts = 0
    for i in range(len(df_new_annotations_unique)):
        class_counts = pd.value_counts(df_new_annotations_check['state_id'])
        total_counts = total_counts + class_counts[i]

    # calculate the percentage of each class in the dataset
    class_percents = pd.value_counts(df_new_annotations, normalize=True) * 100

    # sns.color_palette("rocket")

     # create a bar chart of class percentages using Seaborn
    sns.set_style("whitegrid")
    plt.figure(figsize=(8, 6))
    f = sns.barplot(x=class_percents.index, y=class_percents.values, palette='pastel')
    plt.xlabel('Behaviour Labels')
    plt.ylabel('Percentage of Instances')
    plt.title('Distribution of Class Labels (n='+str(no_of_labels)+'), '+str(data_file))
    # rename the x-axis labels
    # for i in range(len(df_new_annotations_unique)):
    
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
    plt.savefig(f"{save_dir}/{experiment_ID}_class_distribution_{timestamp}.png", dpi=100, bbox_inches='tight')
    
    # plt.savefig(f"{save_dir}/{experiment_ID}_class_distribution.png", dpi=100, bbox_inches='tight')
    plt.show()
    
    return class_counts, total_counts



def check_class_imbalance_new(df_new_annotations, experiment_ID, save_dir, df_new_annotations_unique, df_new_annotations_check, no_of_labels, data_file):
    
    total_counts = 0
    for i in range(len(df_new_annotations_unique)):
        class_counts = pd.value_counts(df_new_annotations_check['state_id'])
        total_counts = total_counts + class_counts[i]

    # calculate the percentage of each class in the dataset
    class_percents = pd.value_counts(df_new_annotations, normalize=True) * 100

    
    
     # create a bar chart of class percentages using Seaborn
    sns.set_style("whitegrid")
    plt.figure(figsize=(8, 6))
    f = sns.barplot(x=class_percents.index, y=class_percents.values, palette='pastel')
    plt.xlabel('Class Labels')
    plt.ylabel('Percentage of Instances')
    plt.title('Distribution of Class Labels (n='+str(no_of_labels)+'), '+str(data_file))
    # rename the x-axis labels
    # for i in range(len(df_new_annotations_unique)):
    f.set_xticklabels(df_new_annotations_check['state_name'].unique())
    

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
#     class_counts = pd.value_counts(df_new_annotations)
#     total_counts = class_counts[0] + class_counts[1] + class_counts[2] + class_counts[3] + class_counts[4] + class_counts[5]

#     # calculate the percentage of each class in the dataset
#     class_percents = pd.value_counts(df_new_annotations, normalize=True) * 100

    
    
#      # create a bar chart of class percentages using Seaborn
#     plt.figure(figsize=(8, 6))
#     f = sns.barplot(x=class_percents.index, y=class_percents.values, palette='Set2')
#     plt.xlabel('Class Label')
#     plt.ylabel('Percentage of Instances')
#     plt.title('Distribution of Class Labels')
#     # rename the x-axis labels
#     f.set_xticklabels(['Grooming', 'Immobile', 'Still', 'Moving', 'Right Turn', 'Left Turn'])
#     plt.savefig(f"{save_dir}/{experiment_ID}_class_distribution.png", dpi=300, bbox_inches='tight')

#     patches = f.patches
#     for i in range(len(patches)):
#         x = patches[i].get_x() + patches[i].get_width()/2
#         y = patches[i].get_height()+.5
#         f.annotate('{:.1f}%'.format(class_percents[i]), (x, y), ha='center')

#     plt.show()
    
#     return class_counts, total_counts


# def check_distribution_among_datasets(labels, experiment_ID, save_dir, dataset_type):
    
#     # calculate the percentage of each class in the dataset
#     class_percents = pd.value_counts(labels, normalize=True) * 100
    
#     # create a bar chart of class percentages using Seaborn
#     plt.figure(figsize=(8, 6))
#     sns.barplot(x=class_percents.index, y=class_percents.values, palette='Set2')
#     plt.xlabel('Class Label')
#     plt.ylabel('Percentage of Instances')
#     plt.title('Distribution of Class Labels in '+ str(dataset_type))
    
    

#     plt.savefig(f"{save_dir}/{experiment_ID}_class_distribution_"+str(dataset_type)+".png", dpi=300, bbox_inches='tight')

#     # display the plot
#     plt.show()
    
#     return 