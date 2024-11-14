# Behaviour Prediction Neural Network (BPNN) toolkit

This repository contains the code for the BPNN toolkit developed for the MSc Thesis work at Karolinska Institutet and Stockholm Universitet titled "Direct Behaviour Prediction from Miniscope Calcium Imaging Using Convolutional Neural Networks". The interested reader can access the Thesis print via this link: https://drive.google.com/file/d/1avRTjyJE3bpw_BOo2nvuhFnxl442z4bC/view?usp=sharing 

## Abstract
### Background: 
Neurodegenerative diseases, including Parkinson's, continue to affect millions worldwide, driving neuroscience research to develop effective and personalised treatments. To study and understand the neural circuits in the human brain associated with the emergence and progression of these diseases, researchers are using neuroimaging techniques like calcium Imaging in disease-relevant animal models to link an organism's neural activity with its behaviour.
### Aim: 
Calcium imaging techniques present various limitations for researchers, such as complex-to-use pipelines often limited to extracting cell body information without directly inferring any behavioural correlates. This thesis investigates the potential application of advanced deep learning techniques, such as convolutional neural networks, in improving calcium imaging analysis by reducing pre-processing requirements and directly arriving at behavioural correlations from animal neural activity.
### Methods: 
In this study, previously collected calcium imaging datasets from behavioural assays of freely moving animals are repurposed and used to train a CNN-based tool called the BPNN (Behavioural Prediction Neural Network). Additionally, the performance of the BPNN is compared and evaluated with current state-of-the-art methods applied in neuroscience research.
### Results: 
Several experiments were performed to evaluate the BPNN's capacity to predict behaviour compared to current methods. However, issues related to overfitting arose, which may have been caused by technical discrepancies or other biological artefacts produced during the calcium imaging recording sessions. Despite this, the BPNN produced similar or better results in predicting animal behaviour, with an F1-score of 0.56 compared to the F1-score of 0.41 of an existing calcium imaging analysis tool concerning the same biological task.
### Conclusion: 
The best-performing configuration of the BPNN model demonstrated a limited yet plausible ability to establish links between neural activity and specific animal behaviours, indicating the potential applicability of CNNs in behaviour prediction assignments. However, further research is required to address current technical and biological limitations to reaffirm the postulations of this study.
<!-- 
## Installation
To get started, follow these steps:

1. Clone the Repository. Open your terminal and navigate to a directory of your choice. Then clone the repository with:
    ```sh
    git clone https://github.com/konkalaitzidis/bpnn.git
    ```
    ```sh
    cd bpnn.git
    ```

2. Set up a Virtual Environment. Create and activate a Virtual Environment for managing dependencies (recommended). Replace `name_of_venv` with your desired environment name.

    Linux/macOS:
    ```sh
    python -m venv name_of_venv
    ```
    ```sh
    source name_of_venv/bin/activate 
    ```
    Windows: 
    ```sh
    python -m venv name_of_venv
    ```
    ```sh
    name_of_venv\Scripts\activate
    ```

3. Install Dependencies. Install the required packages by running the following command:
    ```sh
    pip install -r requirements.txt
    ```

# continue updating the readme -->

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


