# Call Center Efficiency Enhancement Project

## Overview

This project aims to improve call center efficiency and customer satisfaction by analyzing and reducing **Average Handle Time (AHT)** and **Average Speed to Answer (AST)**. The focus is on identifying the key factors that contribute to extended call durations, such as agent performance, different types of call reasons, and customer sentiment. The project also seeks to reduce unnecessary agent interventions through enhanced **Interactive Voice Response (IVR)** options, promoting self-service for recurring, easily solvable issues.

## Key Tasks

### 1. AHT and AST Analysis
- **Objective**: Analyze the factors contributing to long AHT and AST, and quantify the percentage difference between the most frequent and least frequent call reasons.
- **Approach**: By analyzing agent performance, customer sentiment, and call reasons, the project identifies which call reasons have the longest durations and explores factors that lead to these extended times.
- **Outcome**: This helps in understanding the efficiency of the call center and determining areas where improvements can be made.

### 2. IVR System Improvement
- **Objective**: Identify common, repetitive issues that can be resolved without agent involvement and propose enhancements to the IVR system.
- **Approach**: Call transcripts and metadata are analyzed to find self-solvable issues. Recommendations are provided for optimizing IVR self-service options to reduce agent workload.
- **Outcome**: Reduces agent intervention for recurring issues, leading to improved efficiency and faster resolutions.

### 3. Call Reason Classification
- **Objective**: Automate the categorization of incoming calls based on transcripts and metadata to streamline processes and reduce manual tagging efforts.
- **Approach**: Machine learning models, such as `RandomForestClassifier`, are trained to predict the **primary call reason** from features derived from call transcripts and metadata.
- **Outcome**: Automating call classification helps in directing customers more efficiently to the appropriate resources, improving both operational efficiency and customer service.

## Frameworks and Tools

The following frameworks and libraries are used in this project:

- **Pandas**: Data manipulation and preprocessing.
- **NumPy**: Numerical computations.
- **Scikit-learn**: Machine learning, text vectorization (TF-IDF), model building, and evaluation.
- **Matplotlib** & **Seaborn**: Data visualization (AHT analysis, confusion matrices).
- **NLTK**: Natural language processing for text cleaning and stopword removal.
- **Joblib**: Saving and loading machine learning models and vectorizers.
- **WordCloud**: Visualizing common words in transcripts.
  

## Data Preprocessing

### 1. **Text Cleaning**
- Call transcripts are cleaned by removing punctuation, stopwords, and irrelevant words to prepare them for vectorization.

### 2. **TF-IDF Vectorization**
- Transcripts are vectorized using `TfidfVectorizer` to transform text data into numerical features that can be fed into machine learning models.

### 3. **Handling Missing Values**
- Rows with missing `primary_call_reason` values (target variable) are removed. Additionally, columns like `call_duration`, `AHT`, and `AST` are calculated based on call timestamps.

## Machine Learning Models

- **RandomForestClassifier**: Used for predicting the primary call reason. The model is trained on features extracted from call transcripts and metadata.
- **Evaluation**: Classification performance is evaluated using metrics like precision, recall, F1-score, and confusion matrix. Batch processing is used to train the model in batches if the dataset is large.

## Visualizations

- **AHT Analysis**: Visualizes the average AHT for different call reasons and agents to understand performance bottlenecks.
- **Word Cloud**: Displays the most common terms in call transcripts, particularly for self-solvable issues.

## Results

By implementing this project, the call center can:
- **Reduce AHT and AST** by understanding key drivers of extended call durations.
- **Optimize IVR systems**
## Installation

To run this project, ensure the following libraries are installed:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn nltk imbalanced-learn joblib xgboost
