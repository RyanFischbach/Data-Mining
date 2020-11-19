"""
From Scratch KMeans Implementation Tester
Ryan Fischbach
Dr. Khuri
CSC373
10/30/2020

Citations:
Chapter 7
Data Mining
Tan et al.
"""

# Import relevant libraries. Libraries needed: numpy, pandas, seaborn, sys, os, sklearn 
import numpy as np
import pandas as pd
import seaborn as sns
import sys
import os
import sklearn as sklearn
import matplotlib.pyplot as mpl
import Fischbach_KMeansFromScratch
from Fischbach_KMeansFromScratch import KMeans

#Query path to data
try:
  dir = sys.argv[1]
  k = int(sys.argv[2])

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

#traindata = pd.read_csv('/content/train_activities.csv')
#testdata = pd.read_csv('/content/test_activities.csv')

traindata.head()

#Remove duplicates
traindata.drop_duplicates()
testdata.drop_duplicates()

#Determine and mitigate missing values
traindata.isnull().sum()
testdata.isnull().sum()

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

#Rescale data
from sklearn.preprocessing import StandardScaler

traindata = StandardScaler().fit_transform(traindata)
testdata = StandardScaler().fit_transform(testdata)

#Reduce dimensionality
from sklearn.decomposition import PCA

reduced_data = PCA(n_components=2).fit_transform(testdata)

#Run Custom KMeans
kmeans = KMeans()
pred, wss = kmeans.cluster(testdata, k)

#determine cluster info
for displayClusters in range(k):
  # determine samples in the cluster and total number of samples in each cluster
  samplesInCluster = (np.where(pred == displayClusters))[0]
  numSamplesInCluster = samplesInCluster.shape[0]
  print("Within Cluster", displayClusters, "Sum of Squares: ", wss[displayClusters])
  print(numSamplesInCluster, "samples in Cluster", displayClusters)


