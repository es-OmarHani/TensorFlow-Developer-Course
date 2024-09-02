# TensorFlow Projects

This repository contains a series of Jupyter notebooks demonstrating various aspects of TensorFlow and its applications. Each notebook covers different aspects of machine learning, from fundamental operations to advanced techniques. Below is a brief overview of what each notebook entails.

## 00_tensorflow_fundamentals.ipynb

### Description
This notebook introduces fundamental TensorFlow operations. It covers:

- **Introduction to Tensors:** Creating tensors, getting information from tensors, and manipulating tensors.
- **Tensors and NumPy:** Interoperability between TensorFlow and NumPy.
- **Using GPUs with TensorFlow:** Accelerating computations using GPUs.

## 01_linear_regression_with_NN.ipynb

### Description
This notebook walks through linear regression using neural networks in TensorFlow. Topics include:

- **Architecture of a Regression Model:** Understanding input and output shapes.
- **Creating and Compiling a Model:** Steps for setting up, compiling, and evaluating a regression model.
- **Custom Data:** Creating custom data for model training and evaluation.
- **Visualization:** Visualizing training curves and comparing predictions to ground truth.
- **Model Persistence:** Saving and loading models.

## 02_Classification_NN_with_tensorflow.ipynb

### Description
This notebook covers various classification problems with TensorFlow:

- **Binary and Multi-class Classification:** Models for predicting classes based on input data.
- **Architecture and Modeling Steps:** Creating, compiling, and improving classification models.
- **Evaluation:** Visualizing training curves and comparing predictions to ground truth.

## 03_Computer_vision_and_CNN_with_tensorflow.ipynb

### Description
This notebook explores convolutional neural networks (CNNs) for computer vision tasks:

- **Dataset and CNN Architecture:** Getting a dataset and building a CNN model.
- **Modeling Steps:** Preparing data, creating, fitting, and evaluating CNN models.
- **Improving Models:** Techniques for improving CNN performance.

## 04-Transfer_learning_Feature_Extraction_part_1.ipynb

### Description
This notebook introduces transfer learning with a focus on feature extraction:

- **Transfer Learning Basics:** Leveraging pre-trained models for new tasks.
- **Building a Feature Extraction Model:** Using TensorFlow Hub and comparing results.
- **TensorBoard:** Tracking and visualizing model training results.

## 05-Transfer_learning_fine_tuning_part2.ipynb

### Description
This notebook extends the previous transfer learning techniques to fine-tuning:

- **Fine-tuning:** Unfreezing and tuning pre-trained model layers.
- **Keras Functional API:** Building models using the Keras Functional API.
- **Model Experiments:** Comparing feature extraction and fine-tuning models with various data samples.
- **Data Augmentation:** Techniques for increasing dataset diversity.

## 06-Transfer_Learning_Scaling_up_part_3.ipynb

### Description
This notebook scales up from small to large datasets using transfer learning:

- **Scaling Up:** From a small dataset to the full Food101 dataset.
- **Model Performance:** Evaluating and improving model performance with more data.
- **Mixed Precision Training:** Enhancing training efficiency.

## 07-Milestone_Project_Food_Vision_Big_Food101.ipynb

### Description
This project builds a large-scale Food Vision model using the full Food101 dataset:

- **Food Vision Big:** Using all images to beat DeepFood paper results.
- **Model Improvements:** Techniques like prefetching and mixed precision training.
- **Comparative Analysis:** Comparing Food Vision mini and Food Vision Big.

## 08-NLP_With_Tensorflow.ipynb

### Description
This notebook deals with natural language processing (NLP) and natural language understanding (NLU):

- **NLP Basics:** Tokenization, embeddings, and building text models.
- **Modeling:** Dense, LSTM, GRU, Conv1D, and transfer learning models.
- **Evaluation:** Comparing model performance and combining models into ensembles.

## 09-Milestone_Project_2_skimlit.ipynb

### Description
This project focuses on the SkimLit model for classifying sentences in medical abstracts:

- **Dataset:** PubMed 200k RCT dataset.
- **Modeling:** Building and evaluating models for sequential sentence classification.
- **Output:** Classifying sentences into abstract roles to aid literature review.

## 10_Time_Series_Data.ipynb

### Description
This notebook introduces time series forecasting with a focus on predicting Bitcoin prices:

- **Time Series Problems:** Classification vs. forecasting.
- **Data Preparation:** Loading and formatting time series data.
- **Modeling:** Creating and evaluating various deep learning models for time series forecasting.
- **Prediction:** Making forecasts and discussing uncertainty.

---

Feel free to explore each notebook to understand TensorFlow's capabilities and how they can be applied to different types of data and problems.
