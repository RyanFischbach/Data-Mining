"""
Off the Shelf NB Implementation Tester
Ryan Fischbach

To Run, type in Terminal/CMD:
python Fischach_OffTheShelf.py {Absolute Path To Directory Containing CSVs}
ie:
python Fischbach_OffTheShelf.py /users/rtf/Desktop
"""

# Import relevant libraries. Libraries needed: numpy, pandas, seaborn, sys, os, sklearn, matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import sys
import os
import sklearn as sklearn
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

"""
#Loop through kbin values
from sklearn.naive_bayes import CategoricalNB
from sklearn import metrics
from sklearn.preprocessing import KBinsDiscretizer 
r = range(2,6)
scores = []
for loop in r:
  est = KBinsDiscretizer(n_bins=loop, encode='ordinal', strategy='uniform')
  X_test_cat = est.fit_transform(testdata)
  X_train_cat = est.fit_transform(traindata)

  cnb = CategoricalNB()
  cnb.fit(X_train_cat, y_train)
  y_pred = cnb.predict(X_test_cat)
  scores.append(metrics.accuracy_score(y_pred, y_test))

plt.plot(r, scores)
plt.xlabel("Number of Bins Used To Discretize Attributes")
plt.ylabel("The Accuracy of the Classification")
plt.title("NB From Scratch: Classification Accuracy on the Number of Bins")
plt.savefig('scikitlearn.pdf')
"""

from sklearn.preprocessing import KBinsDiscretizer  
est = KBinsDiscretizer(n_bins=4, encode='ordinal', strategy='uniform')
X_test_cat = est.fit_transform(testdata)
X_train_cat = est.fit_transform(traindata)
#Try NB from SKLearn
from sklearn.naive_bayes import CategoricalNB
from sklearn import metrics

cnb = CategoricalNB()
cnb.fit(X_train_cat, y_train)
y_predict = cnb.predict(X_test_cat)

print("Accuracy Score for SKLearn NB =", metrics.accuracy_score(y_test, y_predict))
