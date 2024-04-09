# Script: student_ML_skeleton

# Description: A script to read the extracted data from extract.py's output file and to 
#              train and test models on.

# Author: Eric Ciccotelli, 2021
# Organization: Manhattan College
# Contact: eciccotelli01@manhattan.edu

# Maintainer: Daniel Simpson, 2023
# Organization: Tennessee Technological University
# Contact: dnsimpson42@tntech.edu

# imports
import os
import json
import pandas as pd
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.preprocessing import StandardScaler

# metrics imports
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score

# data parsing
# open data extracted from Cuckoo reports
with open('extractedInfo.json') as f:
    data = json.load(f)

completeCallList = data['completeCallList']
allFilesList = data['allFilesList']
finishedRows = data['finishedRows']
malwareList = data['malwareList']

# create dataframe 
df = pd.DataFrame(columns=completeCallList)

# put statistics from API calls in working dataframe
count = 0
for malwareSampleData in finishedRows:
    df.loc[count] = malwareSampleData
    count+=1

# add truth label to dataset, take off as needed
df['Malware'] = malwareList

# print complete dataframe head & tail
print("\n\nComplete Dataframe\n")
print(df)

# drop truth label from training set, define training and testing sets
X = df.drop('Malware', axis=1)  
y = df['Malware'] 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 2020, stratify=y)

# scale data
ss = StandardScaler()
X_train_scaled = ss.fit_transform(X_train)
X_test_scaled = ss.transform(X_test)
y_train = np.array(y_train)

# API calls (features) obtained for graphics
features = X_test.columns

# ===== Random Forest Classifier example =====

# define RFC
rfc = RandomForestClassifier()
rfc.fit(X_train_scaled, y_train)
rfc_pred = rfc.predict(X_test_scaled)

# RFC metrics
rfc_acc = accuracy_score(y_test, rfc_pred)
rfc_precision = precision_score(y_test, rfc_pred)
rfc_recall = recall_score(y_test, rfc_pred)
rfc_f1 = f1_score(y_test, rfc_pred)

# print RFC scores
print('\n\nRandom Forest Scores\nAccuracy: %.3f' % rfc_acc)
print('Precision: %.3f' % rfc_precision)
print('Recall: %.3f' % rfc_recall)
print('F1 score: %.3f' % rfc_f1)

# ====================================================================
# Given the random forest classifier example above, try importing and 
# testing other models from scikit-learn below. 
# https://scikit-learn.org/stable/supervised_learning.html











