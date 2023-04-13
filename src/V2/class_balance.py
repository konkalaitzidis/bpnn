import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt


def check_class_imbalance(df_new_annotations, experiment_ID, save_dir):
    class_counts = pd.value_counts(df_new_annotations)
    total_counts = class_counts[0] + class_counts[1] + class_counts[2]

    # calculate the percentage of each class in the dataset
    class_percents = pd.value_counts(df_new_annotations, normalize=True) * 100

    # create a bar chart of class percentages using Seaborn
    plt.figure(figsize=(8, 6))
    sns.barplot(x=class_percents.index, y=class_percents.values, palette='Set2')
    plt.xlabel('Class Label')
    plt.ylabel('Percentage of Instances')
    plt.title('Distribution of Class Labels')
    
    # dir_name = "/home/dmc/Desktop/kostas/direct-Behavior-prediction-from-miniscope-calcium-imaging-using-convolutional-neural-networks/src/V2/output/balance"
    plt.savefig(f"{save_dir}/{experiment_ID}_class_distribution.png", dpi=300, bbox_inches='tight')

    # display the plot
    plt.show()

    # print the percentage of each class
    print("Behavior Main Corridor is {:.1f}%" .format((class_counts[0]/total_counts)*100))
    print("Behavior Left Corridor is {:.1f}%" .format((class_counts[1]/total_counts)*100))
    print("Behavior Right Corridor is {:.1f}%" .format((class_counts[2]/total_counts)*100))
    
    return class_counts, total_counts



def check_distribution_among_datasets(labels, experiment_ID, save_dir, dataset_type):
    
    # calculate the percentage of each class in the dataset
    class_percents = pd.value_counts(labels, normalize=True) * 100
    
    # create a bar chart of class percentages using Seaborn
    plt.figure(figsize=(8, 6))
    sns.barplot(x=class_percents.index, y=class_percents.values, palette='Set2')
    plt.xlabel('Class Label')
    plt.ylabel('Percentage of Instances')
    plt.title('Distribution of Class Labels in '+ str(dataset_type))
    
    # dir_name = "/home/dmc/Desktop/kostas/direct-Behavior-prediction-from-miniscope-calcium-imaging-using-convolutional-neural-networks/src/V2/output/balance"
    plt.savefig(f"{save_dir}/{experiment_ID}_class_distribution_"+str(dataset_type)+".png", dpi=300, bbox_inches='tight')

    # display the plot
    plt.show()
    
    return 