# AI701_Project_G22
## Ablation Study and Comparative Analysis of Machine Learning and Deep Learning Models for Restaurant Review Sentiment Classification
### Project Overview

This project explores sentiment analysis on restaurant reviews using four models:
- Logistic Regression
- Support Vector Machine (SVM)
- Long Short-Term Memory (LSTM)
- BERT (Bidirectional Encoder Representations from Transformers)
  
The objective is to classify sentiments as either positive or negative (binary classification) and analyze the performance of each model through an ablation study. The models are evaluated based on metrics like accuracy, precision, recall, and F1-score.

### Repository Structure

The repository contains the following files and directories:
#### Code 
- Logistic_regression_CODE.ipynb: Implementation of Logistic Regression with ablation configurations.
- SVM_code.ipynb: Implementation of SVM with ablation configurations.
- LSTM_code.ipynb: Implementation of LSTM model with ablation configurations.
- bert_code.ipynb: Implementation of BERT.
- preprocessingbert.ipynb: Preprocessing steps for BERT.

#### Models
- best_logistic_model.joblib: Saved Logistic Regression model.
- best_svm_model.joblib: Saved SVM model.
- LSTM_Best.h5: Saved LSTM model.
- BERT Saved Model: Due to file size limitations, the saved BERT model is on Google Drive. Access it here: https://drive.google.com/drive/folders/1H6ZziJkkRokt_TBc3PVzMVYyAROvhpXC?usp=drive_link

#### Datasets
- Restaurant reviews.csv: Dataset used for the whole project. Obtained from: 
- df_cleaned_bert.csv: Preprocessed dataset for BERT.

### Instructions to Run
1. Download the Repository
   - Clone the repository or download the files directly from GitHub.
   - Download the pre-trained BERT model from Google Drive.

2. Run Models
   - Open the Jupyter notebooks in local environment or upload them to Google Colab.
   - Use the pre-trained models to skip training by loading the provided weights: The pre-trained models are ready for evaluation, simply load them to avoid retraining(saving time and computational resources).

### Results
- The ablation study provided insights into the best parameter combinations for the models to improve their performances. 
- BERT outperformed all other models used in this project.

### Authors
This project was implemented by: Khawla Almarzooqi, Ayesha Alhammadi, Aljalila Aladawi. 


