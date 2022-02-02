# Movie Sentiment Classification using BiLSTMs, BERT, RoBERTa

## Table of Contents: 
* Overview of Project

* Data Description 
* Libraries used

* Structure of the Approach

* Conclusion



## Overview of Project:
The main task corresponds to a multi-class text classification on Movie Reviews from the Rotten Tomatoes dataset. We will be using deep learning techniques for sentiment classification 
We will be comparing the performace of following models
* BiLSTM

* BERT
* RoBERTa (Robustly Optimized BERT Pre-training Approach)

## Data Description:   
For this project , I will be using the data from Kaggle <a href='https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews/data'>Sentiment Analysis on Movie Reviews</a>.<br>
The main task corresponds to a multi-class text classification on Movie Reviews Competition and the dataset contains 156.060 instances. The sentiment labels are:

0 → Negative      </br>
1 → Somewhat negative  </br>
2 → Neutral </br>
3 → Somewhat positive </br>
4 → Positive </br>


## Libraries used:
* Numpy
* Pandas
* Matplotlib

* tensorflow
* keras
* nltk
* huggingface transformers library 

## Structure of the Approach

* ### 1. Importing Necessary Libraries
Getting all the required python libraries required for the implementation of the project

+
* ### 2. Importing dataset and Doing a primary analysis
We check how many data points are there, number of missing values, data type of features etc.

* ### 3. EDA and Feature Engineering
We find out the number of instances for every emotion , and also check the length of the phrases present. Based on that we make a conclusion that there are nearly half of the instances are of neutral class and also that most phrases have length <40. We then eventually remove all the phrases with length more than 40.


* ### 4. Dividing dataset into train and validation split using <tt>**StratifiedSplit()**</tt> as we find that there is an imabalance in the instances for every class


* ### 5. Text processing with techniques like lemmatization , stop word removal , padding , tokenization

* ### 6. Building the model
* ### 7. Evaluating the performance
Plotted graphs to see the pattern of different parameters that were stored in a list during forward propagation

## Analysis and Conclusion

Based on the training conducted above , we find out that LSTM has an accuracy of 0.67 , BERT has , RoBERT has . Comparing them XYZ has the best performance in our case
