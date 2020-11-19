"""
KMeans Best K Finder
Ryan Fischbach
Dr. Khuri
CSC373
10/30/2020

To find the best value of k, two pdfs will be created to the directory where the train and test sets are located.
Scratch.pdf is the elbow curve for the implementation from scratch, OffTheShelf.pdf is the elbow curve for the
off the shelf SKLearn implementation. To find the best value of k, locate the "bend" in the curve for each.

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
import matplotlib.pyplot as plt
import Fischbach_KMeansFromScratch
from Fischbach_KMeansFromScratch import KMeans

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

#Code to test different values of k with custom implementation
kmeansS = KMeans()
kr = range(1, 10)
scoresFromScratch = []
for kloop in kr:
  
  pred, wss = kmeansS.cluster(testdata, kloop)
  scoresFromScratch.append(np.nansum(wss))

print("Best Value of k for From Scratch Implementation is: ", np.argmin(scoresFromScratch)+1)


#Run SKLearn's KMeans
from sklearn.cluster import KMeans
from scipy.spatial import distance
kr = range(1, 5)
scoresSKLearn = []

for kloop in kr:
  kmeans = KMeans(n_clusters=kloop+1).fit(testdata)
  pred = kmeans.predict(testdata)
  centers = kmeans.cluster_centers_
  sumOfSquares = 0
  scoresSKLearn.append(kmeans.inertia_)

  # determine cluster info
  for displayClusters in range(kloop):
    # determine samples in the cluster and total number of samples in each cluster
    samplesInCluster = (np.where(pred == displayClusters))[0]
    numSamplesInCluster = samplesInCluster.shape[0]

    # compute sum of squares for cluster
    for sampleIndex in range(numSamplesInCluster):
      sumOfSquares += np.square(distance.euclidean(centers[displayClusters], samplesInCluster[sampleIndex]))
print("Best Value of k for SKLearn is: ", np.argmin(scoresSKLearn)+1)
"""

#PCA visualizaiton
clusters, wss = kmeansS.cluster(testdata, 2)
pred = KMeans(n_clusters=2).fit_predict(testdata)

plt.subplot(1, 2, 1)
plt.scatter(reduced_data[:,0],reduced_data[:,1],c=clusters)
plt.title('Scratch Implementation, K=2')
plt.xlabel('PCA Dimension 1')
plt.ylabel('PCA Dimension 2')

plt.subplot(1, 2, 2)
plt.scatter(reduced_data[:,0],reduced_data[:,1],c=pred)
plt.title('SKLearn Implementation, K=2')
plt.xlabel('PCA Dimension 1')
plt.savefig('ComparisonViz.pdf')
"""
