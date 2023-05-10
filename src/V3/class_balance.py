import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def check_class_imbalance_old(df_new_annotations, experiment_ID, save_dir):
    class_counts = pd.value_counts(df_new_annotations)
    total_counts = class_counts[0] + class_counts[1] + class_counts[2]

    # calculate the percentage of each class in the dataset
    class_percents = pd.value_counts(df_new_annotations, normalize=True) * 100

    
    
     # create a bar chart of class percentages using Seaborn
    plt.figure(figsize=(8, 6))
    f = sns.barplot(x=class_percents.index, y=class_percents.values, palette='Set2')
    plt.xlabel('Class Label')
    plt.ylabel('Percentage of Instances')
    plt.title('Distribution of Class Labels')
    # rename the x-axis labels
    f.set_xticklabels(['Main Corr', 'Left Corr', 'Right Corr'])
    plt.savefig(f"{save_dir}/{experiment_ID}_class_distribution.png", dpi=300, bbox_inches='tight')

    patches = f.patches
    for i in range(len(patches)):
        x = patches[i].get_x() + patches[i].get_width()/2
        y = patches[i].get_height()+.5
        f.annotate('{:.1f}%'.format(class_percents[i]), (x, y), ha='center')

    plt.show()
    
    return class_counts, total_counts



def check_class_imbalance_new(df_new_annotations, experiment_ID, save_dir):
    class_counts = pd.value_counts(df_new_annotations)
    total_counts = class_counts[0] + class_counts[1] + class_counts[2] + class_counts[3] + class_counts[4] + class_counts[5]

    # calculate the percentage of each class in the dataset
    class_percents = pd.value_counts(df_new_annotations, normalize=True) * 100

    
    
     # create a bar chart of class percentages using Seaborn
    plt.figure(figsize=(8, 6))
    f = sns.barplot(x=class_percents.index, y=class_percents.values, palette='Set2')
    plt.xlabel('Class Label')
    plt.ylabel('Percentage of Instances')
    plt.title('Distribution of Class Labels')
    # rename the x-axis labels
    f.set_xticklabels(['Grooming', 'Immobile', 'Still', 'Moving', 'Right Turn', 'Left Turn'])
    plt.savefig(f"{save_dir}/{experiment_ID}_class_distribution.png", dpi=300, bbox_inches='tight')

    patches = f.patches
    for i in range(len(patches)):
        x = patches[i].get_x() + patches[i].get_width()/2
        y = patches[i].get_height()+.5
        f.annotate('{:.1f}%'.format(class_percents[i]), (x, y), ha='center')

    plt.show()
    
    return class_counts, total_counts


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