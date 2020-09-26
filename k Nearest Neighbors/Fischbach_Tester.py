"""
Naive kNN Custom Implementation Tester
Ryan Fischbach

To Run, type in Terminal/CMD:
python Fischach_FromScratchTester.py {Absolute Path To Directory Containing CSVs} {Value of K}
ie:
python Fischbach_FromScratchTester.py /users/rtf/Desktop 10

Note: The Fischbach_kNNFromScratch.py needs to be in the same directory as this file.
"""

#Import Required Libraries (numpy, pandas, os, sys, sklearn, Fischbach_kNNFromScratch)
import numpy as np
import pandas as pd
import os
import sys
import Fischbach_kNNFromScratch
import sklearn.metrics
import sklearn.preprocessing 
from Fischbach_kNNFromScratch import kNN
import matplotlib.pyplot as plt


#Query path to data
try:
  dir = sys.argv[1]
  k = sys.argv[2]

  if int(k) < 1:
    print("Please enter a k value of at least 1.")
    sys.exit(1)

  if(not os.path.isdir(dir)):
    print("Please enter a valid absolute path to the directory housing cycling data in the command line followed by the k.")
    sys.exit(1)
    
except:
  print("Please enter a valid absolute path to the directory housing cycling data in the command line followed by the k.")
  sys.exit(1)

try:
  os.chdir(dir)
  
  traindata = pd.read_csv("train_activities.csv")
  testdata = pd.read_csv("test_activities.csv")
except:
  print("CSV could not be read. Please enter a valid absolute path to the directory of the cycling data housing train.csv and test.csv.")
  sys.exit(1)
  
# Preprocess The Dataset

#Remove target variable
y_train = traindata['avg_power']
del traindata['avg_power']
y_test = testdata['avg_power']
del testdata['avg_power']

#Discretize the target
mean_train = np.mean(np.array(y_train))
y_train = np.where(y_train >= mean_train, 1, 0)
y_test = np.where(y_test >= mean_train, 1, 0)

#Remove duplicates
traindata.drop_duplicates()
testdata.drop_duplicates()

#Determine and mitigate missing values
#print(traindata.isnull().sum())
#print(testdata.isnull().sum())

#Irrelevant attributes identified and removed
del traindata['activity_id']
del traindata['category_id']
del traindata['athlete_id']
del traindata['filename']
del traindata['timestamp']

del testdata['activity_id']
del testdata['category_id']
del testdata['athlete_id']
del testdata['filename']
del testdata['timestamp']

#Redundant attributes removed
del traindata['moving_time'] #Highly correlated (0.97) with moving time
del traindata['elevation_loss'] #Highly correlated (0.99) with elevation gain

del testdata['moving_time']
del testdata['elevation_loss']

#Encode categorical attributes (no categorical attributes to encode.)
#print(traindata)

#Rescale numeric attributes
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.metrics import accuracy_score
scaler = MinMaxScaler() 
traindata_scaled = scaler.fit_transform(traindata)
testdata_scaled = scaler.fit_transform(testdata)

model = kNN()

y_pred = model.predict(traindata_scaled, y_train.reshape((1937, 1)), testdata_scaled, int(k))

#Code to test different values of k
"""
kr = range(1,21)
scores = []
for kloop in kr:
  y_predict = model.predict(traindata_scaled, y_train.reshape((1937, 1)), testdata_scaled, kloop)
  scores.append(metrics.accuracy_score(y_predict, y_test))

plt.plot(kr, scores)
plt.xlabel("k Nearest Neighbors Value")
plt.ylabel("The Accuracy of the Classification")
plt.title("kNN From Scratch: Classification Accuracy on the Value of k")
plt.savefig('scratch.pdf')
"""

#Determine accuracy score
print("Accuracy Score for kNN with k =", k, "from scratch =", accuracy_score(y_test.reshape((25,1)), y_pred))
