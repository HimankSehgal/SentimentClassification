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
The instances of every class till this step comes out to be 
2    79525
3    32810
1    27195
4     9146
0     7032
Hence there is an imbalance. To deal with this we use the <tt>StratifiedSplit()</tt> method to ensure proportionate distribution in train and val set

* ### 5. Text processing with techniques like lemmatization , stop word removal , padding , tokenization
The text that we have cannot be directly input into the model , we need to perform pre processing. Lemmatization help to reduce words like going , go, gone to its lemma form of go. Stop words like I,the don't add much value hence they can be removed. Then we convert them to token numbers as we cannot directly input text into the model

* ### 6. Building the model
For BiLSTM , we take a stacked BiLSTM of 2 layers with 128 neurons which then followed by 2 linear layers with 50 and 5 units. 
For BERT models,  we take the pretrained models with <tt>trainable = True </tt> and add a final layer with 5 neurons 
* ### 7. Evaluating the performance
We display the confusion matrix and classification matrix for predictions on the validation set for each of the three models



## Analysis and Conclusion

Based on the training conducted above , we find out that LSTM has an accuracy of 0.67 , BERT has , RoBERT has . Comparing them XYZ has the best performance in our case
