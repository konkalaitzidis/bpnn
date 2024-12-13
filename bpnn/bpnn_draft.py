import os
import pandas as pd
import numpy as np
from src.load_calcium_video import load_video_data, load_one_video
from src.align_behavior_to_calcium import align_files_old_labels, align_files_new_labels
from src.class_balance import check_class_imbalance_old, check_class_imbalance_new, check_class_imbalance_old_merged
from src.preprocessing_model import model_preprocessing
from src.plots import plot_first_frames, plot_random_frames
from src.run_BPNN import run

import tensorflow.compat.v1 as tf

def check_gpu_availability(GPU_usage):
    if GPU_usage:
        print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
        tf.ones(1) + tf.ones(1)
    else:
        print("\nNo GPU available")

def create_output_directory(experiment_ID):
    output_dir = f"{experiment_ID}_output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        os.makedirs(os.path.join(output_dir, "balance"))
        os.makedirs(os.path.join(output_dir, "accuracy"))
        os.makedirs(os.path.join(output_dir, "loss"))
        os.makedirs(os.path.join(output_dir, "cm"))
        os.makedirs(os.path.join(output_dir, "architecture"))
        os.makedirs(os.path.join(output_dir, "pickles"))
    return output_dir

def load_calcium_video(multiple_videos, video_name_list, video_data_list):
    if multiple_videos:
        video_paths = ["path/animal1-learnday8.nwb", "path/animal1-learnday9.nwb", "path/animal1-learnday10.nwb"]
        fov_info = pd.read_csv('path/aligned_videos_animal1.csv')
        images = load_video_data(video_paths, fov_info, video_name_list, video_data_list)
    else:
        video_path = ["path/animal1-learnday10.nwb"]
        images = load_one_video(video_path, video_name_list)
    return images

def align_behavior_labels(multiple_videos, labels_type_new, merge_labels, num_of_videos):
    bonsai_paths = ["path/bonsai_path.csv"]
    if multiple_videos:
        bonsai_paths *= 3
    if labels_type_new:
        h5_path = "path/behavior_segmentation_arrowmaze.h5"
        return align_files_new_labels(bonsai_paths, num_of_videos, h5_path, multiple_videos)
    else:
        behavior_paths = ["path/animal3learnday11.h5"] if not multiple_videos else ["path/behavior_segmentation_arrowmaze.h5"] * 3
        return align_files_old_labels(bonsai_paths, behavior_paths, num_of_videos, merge_labels)

def check_class_balance(labels_type_new, merge_labels, df_new_annotations, experiment_ID, save_dir, df_new_annotations_unique, df_new_annotations_check, no_of_labels, data_file, label_names):
    if labels_type_new:
        return check_class_imbalance_new(df_new_annotations, experiment_ID, save_dir, df_new_annotations_unique, df_new_annotations_check, no_of_labels, data_file, label_names)
    elif merge_labels:
        names_of_labels = 'Main Corr', 'Left Corr', 'Right Corr'
        return check_class_imbalance_old_merged(df_new_annotations, experiment_ID, save_dir, df_new_annotations_unique, df_new_annotations_check, no_of_labels, names_of_labels, data_file, label_names)
    else:
        return check_class_imbalance_old(df_new_annotations, experiment_ID, save_dir, df_new_annotations_unique, df_new_annotations_check, no_of_labels, data_file, label_names)

def main():
    # Step 1: Resource Allocation (GPU/CPU)
    check_gpu_availability(GPU_usage)

    # Step 2: Experiment Configuration and Logging
    experiment_ID = 'test_1.0'
    data_file = 'Animal_Doe'
    experiment_name = f"{data_file}_{experiment_ID}"
    train_test_split_strategy = "k-fold"
    name = 'BPNN_revamp'
    labels_type_new = True
    merge_labels = False
    multiple_videos = False

    # Step 3: Create output directory
    output_dir = create_output_directory(experiment_ID)

    # Step 4: Data Loading
    video_name_list = []
    video_data_list = []
    images = load_calcium_video(multiple_videos, video_name_list, video_data_list)

    # Step 5: Importing Behaviour Labels
    num_of_videos = 1
    df_new_annotations, df_new_annotations_check = align_behavior_labels(multiple_videos, labels_type_new, merge_labels, num_of_videos)

    # Step 6: Datafile structure validation and preparation
    reordered_annotations = df_new_annotations_check.drop_duplicates().sort_values('state_id').reset_index(drop=True)
    reordered_annotations.at[0, 'state_name'] = 'Grooming'
    reordered_annotations.at[1, 'state_name'] = 'Frozen'
    reordered_annotations.at[2, 'state_name'] = 'Not moving'
    reordered_annotations.at[3, 'state_name'] = 'Moving'
    reordered_annotations.at[4, 'state_name'] = 'Right turn'
    reordered_annotations.at[5, 'state_name'] = 'Left turn'

    # Step 7: Alignment of Behavioural Classification with CI movies
    df_new_annotations = df_new_annotations.reset_index(drop=True)
    df_new_annotations_unique = reordered_annotations['state_id'].unique()
    no_of_labels = len(reordered_annotations)

    # Step 8: Representation of behavioural instances in the dataset
    save_dir = f"/home/dmc/Desktop/kostas/direct-Behavior-prediction-from-miniscope-calcium-imaging-using-convolutional-neural-networks/src/V3/{output_dir}/balance"
    class_counts, total_counts = check_class_balance(labels_type_new, merge_labels, df_new_annotations, experiment_ID, save_dir, df_new_annotations_unique, df_new_annotations_check, no_of_labels, data_file, reordered_annotations['state_name'])

    # Step 9: Preparation of Model Training Data
    labels = df_new_annotations
    if len(labels) == len(images):
        print("Labels and Images have the same length:", len(images))
    plot_first_frames(images, labels, data_file)
    plot_random_frames(images, labels, data_file)
    num_classes = len(reordered_annotations['state_name'])
    images, labels = model_preprocessing(images, labels, df_new_annotations_unique, num_classes)

    # Step 10: BPNN model training
    is_basic_BPNN = False
    channel_dimension = 1 if is_basic_BPNN else 7
    input_shape = (images.shape[1], images.shape[2], channel_dimension)
    num_folds = 2
    epochs = 1
    shuffle = False
    df_new_annotations_names = reordered_annotations['state_name'].unique()
    no_of_behaviors = len(reordered_annotations['state_name'])

    if is_basic_BPNN:
        print(f"Running {data_file} on the BPNN, with {num_folds}-Fold CV, {epochs} Epochs, and Early Stopping")
    else:
        print(f"Running {data_file} on the BPNNt, with {num_folds}-Fold CV, {epochs} Epochs, and Early Stopping")

    train_loss_all, val_loss_all, train_acc_all, val_acc_all, average_score_list, conf_matrices, f1_score_val_list, train_labels_names = run(
        is_basic_BPNN,
        reordered_annotations['state_name'],
        labels_type_new,
        shuffle,
        images,
        labels,
        num_folds,
        input_shape,
        num_classes,
        name,
        epochs,
        no_of_behaviors,
        df_new_annotations,
        df_new_annotations_unique,
        df_new_annotations_check,
        output_dir,
        experiment_ID
    )

    # Step 11: Saving Results
    f1_score_mean = np.mean(f1_score_val_list)
    print(f"F1 Score Mean: {f1_score_mean}")

if __name__ == "__main__":
    main()