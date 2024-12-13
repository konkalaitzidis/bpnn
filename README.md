# Behaviour Prediction Neural Network (BPNN) Toolkit

Welcome to the repository for the BPNN toolkit, developed during my MSc Thesis work at the Karolinska Institute and Stockholm University. This project explores the potential use of convolutional neural networks (CNNs) for intrepreting animal behaviour directly from calcium imaging movies.

---

## About the Project

- **Title**: Direct Behaviour Prediction from Miniscope Calcium Imaging Using Convolutional Neural Networks
- **Purpose**:  
  - Investigate the application of CNNs in linking neural activity with behaviour.
  - Reduce pre-processing requirements in calcium imaging analysis.
  - Compare the performance of BPNN with state-of-the-art calcium imaging analysis methods.
- **Thesis Document**: Access the thesis [here](https://www.kostaskal.com/bpnn).

---

## Abstract

### Background  
Neurodegenerative diseases, including Parkinson's, affect millions worldwide, driving neuroscience research to develop effective and personalised treatments. Techniques like calcium imaging are used to link neural activity with behaviour in animal models, but they often face challenges such as complex workflows often limited to extracting cell body information without directly inferring any behavioural correlates. 

### Aim  
To address these challenges, this Thesis employs CNNs to simplify calcium imaging analysis and directly infer behavioural correlates.

### Methods  
Previously collected calcium imaging datasets from behavioural assays were repurposed to train the BPNN, a CNN-based tool. Its performance was compared with existing methods in neuroscience research.

### Results  
Despite challenges like overfitting, which may have been caused by technical discrepancies or other biological artefacts produced during the calcium imaging recording sessions, the BPNN demonstrated competitive performance in predicting animal behaviour, achieving an F1-score of 0.56 compared to 0.41 from an existing tool.

### Conclusion  
The BPNN model shows promise in linking neural activity with behaviour. However, further research is required to address current technical and biological limitations to reaffirm the postulations of this study.

---

## Tech Stack

- **Programming Languages**: Python
- **Frameworks**: TensorFlow, Keras
- **Data Handling**: NumPy, Pandas
- **Visualization**: Matplotlib, Seaborn
- **Environment Management**: Virtualenv

---

## Features

- CNN-based calcium imaging analysis pipeline.
- Competitive performance with state-of-the-art methods that emply SVMs.
- Modular and extensible codebase to further research.

---

<!-- ## Installation

To get started with the BPNN toolkit:

1. **Clone the Repository**  
   ```sh
   git clone https://github.com/konkalaitzidis/bpnn.git
   cd bpnn
   ```

2. **Set Up a Virtual Environment**  
   - **Linux/macOS**:  
     ```sh
     python -m venv venv
     source venv/bin/activate
     ```
   - **Windows**:  
     ```sh
     python -m venv venv
     venv\Scripts\activate
     ```

3. **Install Dependencies**  
   ```sh
   pip install -r requirements.txt
   ```

4. **Run the Toolkit**  
   Execute the main script to start the analysis:
   ```sh
   python main.py
   ```

--- -->

## Future Enhancements

- Currently, this work is being under a complete revamp and awaits official release. 
- Address overfitting and technical discrepancies in calcium imaging data.
- Integrate additional datasets for improved model generalizability.
- Train and test 2p CI data.
- Develop an interactive interface for easier usability.

---

## Author

**Konstantinos Kalaitzidis**  
- [GitHub](https://github.com/konkalaitzidis)  
- [LinkedIn](https://linkedin.com/in/konstantinoskalaitzidis)  
- [Personal Website](https://www.kostaskal.com)

**Special thanks to my supervisor at the Meletis Lab, Emil Wärnberg (Post-doctoral Fellow, Ölveczky Lab, Harvard University), whose continuous support and oversight have made this work possible as well for the joyful, thought-provoking, and intellectually illuminating collaboration.**

---

## License

This project is licensed under the [MIT License](LICENSE).

<!-- If you encounter any issues or have feedback, please feel free to reach out! -->
