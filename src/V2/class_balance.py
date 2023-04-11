import matplotlib.pyplot as plt
import pandas as pd


def check_class_imbalance(df_new_annotations):
    class_counts = pd.value_counts(df_new_annotations)
    total_counts = class_counts[0] + class_counts[1] + class_counts[2]

    
    # calculate the percentage of each class in the dataset
    class_percents = pd.value_counts(df_new_annotations, normalize=True) * 100

    # create a bar chart of class percentages
    plt.bar(class_percents.index, class_percents.values)

    # add axis labels and a title
    plt.xlabel('Class Label')
    plt.ylabel('Percentage of Instances')
    plt.title('Distribution of Class Labels')

    # display the plot
    plt.show()

    print("Behavior Forward is {:.1f}%" .format((class_counts[0]/total_counts)*100))
    print("Behavior Right is {:.1f}%" .format((class_counts[1]/total_counts)*100))
    print("Behavior Left is {:.1f}%" .format((class_counts[2]/total_counts)*100))
    
    return class_counts, total_counts