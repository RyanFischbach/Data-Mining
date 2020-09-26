"""
Off the Shelf kNN Implementation Tester
Ryan Fischbach

To Run, type in Terminal/CMD:
python Fischach_OffTheShelf.py {Absolute Path To Directory Containing CSVs} {Value of K}
ie:
python Fischbach_OffTheShelf.py /users/rtf/Desktop 5
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

#traindata.head()

"""# Preprocess The Dataset"""

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

#Correlation matrix to identify redundant attributes
#cm = traindata.corr()
#f, ax = mpl.subplots(figsize=(12, 9))
#sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 8})
#mpl.clf()

#Redundant attributes removed
del traindata['moving_time'] #Highly correlated (0.97) with moving time
del traindata['elevation_loss'] #Highly correlated (0.99) with elevation gain

del testdata['moving_time']
del testdata['elevation_loss']

#Encode categorical attributes (no categorical attributes to encode.)
#print(traindata)

#Rescale numeric attributes
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler() 
traindata_scaled = scaler.fit_transform(traindata)
testdata_scaled = scaler.fit_transform(testdata)

#print(traindata_scaled)

#Try kNN from sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

model = KNeighborsClassifier(n_neighbors=int(k))
model.fit(traindata_scaled, y_train)
y_predict = model.predict(testdata_scaled)


kr = range(1,21)
scores = []
for kloop in kr:
  model = KNeighborsClassifier(n_neighbors=kloop, weights='distance')
  model.fit(traindata_scaled, y_train)
  y_predict = model.predict(testdata_scaled)
  scores.append(metrics.accuracy_score(y_predict, y_test))

plt.plot(kr, scores)
plt.xlabel("k Nearest Neighbors Value")
plt.ylabel("The Accuracy of the Classification")
plt.title("Scikit Learn: Classification Accuracy on the Value of k")
plt.savefig('scikitlearn.pdf')



print("Accuracy Score for kNN with k =", k, "with sklearn =", metrics.accuracy_score(y_test, y_predict))

#Try logistic regression
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(random_state=0).fit(traindata_scaled, y_train)
y_predict = lr.predict(testdata_scaled)
print("Accuracy Score for Logistic Regression Classifier with sklearn =", metrics.accuracy_score(y_test, y_predict))

#Try stochastic gradient descent
from sklearn.linear_model import SGDClassifier
clf = SGDClassifier(loss="hinge", penalty="l2", max_iter=2000)
clf.fit(traindata_scaled, y_train)
y_predict = clf.predict(testdata_scaled)
print("Accuracy Score for Stochastic Gradient Descent Classifier with sklearn =", metrics.accuracy_score(y_test, y_predict))

#Try decisiontree
from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf.fit(traindata_scaled, y_train)
y_predict = clf.predict(testdata_scaled)
print("Accuracy Score for Decision Tree Classifier with sklearn =", metrics.accuracy_score(y_test, y_predict))




