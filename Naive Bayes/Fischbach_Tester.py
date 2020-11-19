"""
Naive NB Custom Implementation Tester
Ryan Fischbach

To Run, type in Terminal/CMD:
python Fischach_Tester.py {Absolute Path To Directory Containing CSVs}
ie:
python Fischbach_Tester.py /users/rtf/Desktop

Note: The Fischbach_NBFromScratch.py needs to be in the same directory as this file.
"""

#Import Required Libraries (numpy, pandas, os, sys, sklearn, Fischbach_NBFromScratch)
import numpy as np
import pandas as pd
import os
import sys
import Fischbach_NBFromScratch
import sklearn.metrics
import sklearn.preprocessing 
from Fischbach_NBFromScratch import NaiveBayes
import matplotlib.pyplot as plt


#Query path to data
try:
  dir = sys.argv[1]

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

#Encode numeric attributes to categorical
from sklearn.preprocessing import KBinsDiscretizer
import sklearn.metrics as metrics
from sklearn.metrics import accuracy_score
est = KBinsDiscretizer(n_bins=4, encode='ordinal', strategy='uniform')
X_test_cat = est.fit_transform(testdata)
X_train_cat = est.fit_transform(traindata)

#Create Naive Bayes class from tester
model = NaiveBayes()
#Train model with training samples and labels
model.train(X_train_cat, y_train.reshape((y_train.shape[0], 1)))
y_pred= model.predict(X_test_cat)

#Determine accuracy score
print("Accuracy Score for NB from scratch =", accuracy_score(y_test.reshape((25,1)), y_pred))

"""
r = range(2,6)
scores = []
for loop in r:
  est = KBinsDiscretizer(n_bins=loop, encode='ordinal', strategy='uniform')
  X_test_cat = est.fit_transform(testdata)
  X_train_cat = est.fit_transform(traindata)

  #Create Naive Bayes class from tester
  model = NaiveBayes()
  #Train model with training samples and labels
  model.train(X_train_cat, y_train.reshape((y_train.shape[0], 1)))
  y_pred= model.predict(X_test_cat)
  scores.append(metrics.accuracy_score(y_pred, y_test))

plt.plot(r, scores)
plt.xlabel("Number of Bins Used To Discretize Attributes")
plt.ylabel("The Accuracy of the Classification")
plt.title("NB From Scratch: Classification Accuracy on the Number of Bins")
plt.savefig('scratch.pdf')
"""


