"""
SocialNetworkProfiles_Fischbach.py

Ryan Fischbach
Dr. Khuri
CSC373
11/13/2020

Assignment 5: SocialNetworkProfiles.csv Analysis Via Clustering

This script preprocesses the purchases.csv file, prints useful information
and outputs useful plots, clusters the data, then outputs useful plots.

To run, type "Python socialnetworkprofiles_fischbach.py {absolute path to socialnetworkprofiles.csv}" into the CLI.

Note: the visualizations and outputs will be saved in the same directory as the data.
"""

# Import relevant libraries. Libraries needed: numpy, pandas, seaborn, sys, os, sklearn, matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import sys
import os
import matplotlib.pyplot as mpl
import sys
import os

try:
    dir = sys.argv[1]

    if (not os.path.isdir(dir)):
        print(
            "Please enter a valid absolute path to the directory housing socialnetworkprofiles.csv.")
        sys.exit(1)

except:
    print("Please enter a valid absolute path to the directory housing socialnetworkprofiles.csv after the .py file. For help see comments.")
    sys.exit(1)

try:
    os.chdir(dir)

    # Read data
    unprocesseddata = pd.read_csv('social_network_profiles.csv')
except:
    print("CSV could not be read. Please enter a valid absolute path to the directory of socialnetworkprofiles.csv and make sure the file is there..")
    sys.exit(1)

data = unprocesseddata.copy()

#data.head()

#data.info()

#data.describe()

#data.columns

"""The dataset has 40 columns, with the last 36 being binary/dummy variables indicating if a sample (person) has a relation to those items/words. The first 4 columns contain identifying information. There are 30,000 samples before removing duplicates, suggesting 30,000 people are contained within these profiles on the social network."""

#Remove duplicates
data.drop_duplicates(inplace=True)

"""After dropping duplicates, there are 29,350 samples (people) in this dataset."""

#Determine missing values
#data.isnull().any()

#Mitigate missing values because these two columns with NAs (gender, age) contain valuable information on identifying patterns about samples and for clustering.
data.dropna(axis=0, how='any', inplace=True)

#len(data)

#Binarize gender
data["gender"] = (data["gender"] == 'M').astype(int)

#Get variance of attributes
#data.var(axis=0)

#Subset data to remove near zero variance attributes
clusterdata = data.copy()[['gradyear', 'age', 'friends', 'gender']]

"""
#Correlation Matrix
cm = clusterdata.corr()
f, ax = mpl.subplots(figsize=(12, 9))
sns.heatmap(cm, vmax=.8, square=True);


#Get highest correlations to see if there are redundant attributes
c = clusterdata.corr().abs()
s = c.unstack()
so = s.sort_values(kind="quicksort")

so.tail(n=(clusterdata.columns.size + 1))
"""

#Skewness
#print("Skewness of Gradyear: %f" % clusterdata['gradyear'].skew())
#print("Kurtosis of Gradyear: %f" % clusterdata['gradyear'].kurt())

#print("Skewness of Age: %f" % clusterdata['age'].skew())
#print("Kurtosis of Age: %f" % clusterdata['age'].kurt())

#scatterplot using non-indicator variables
#sns.set()
#cols = ['gradyear','age', 'friends']
#sns.pairplot(clusterdata[cols], height = 2.5)
#mpl.show()

#Investigate differences between men and women
"""
subdataMale = clusterdata.where(clusterdata['gender'] == 1)
subdataFemale = clusterdata.where(clusterdata['gender'] == 0)

fig, axes = mpl.subplots(nrows=1, ncols=2)
fig.set_size_inches(10,10)
axes[0].matshow(subdataMale.corr())
axes[0].set_title("Male Correlation Matric")
axes[1].matshow(subdataFemale.corr())
axes[1].set_title("Female Correlation Matric")
#mpl.show()

#Investigate differences between many friends and less friends
subdataMoreFriends = clusterdata.where(clusterdata['friends'] >= clusterdata['friends'].mean())
subdataLessFriends = clusterdata.where(clusterdata['friends'] < clusterdata['friends'].mean())

fig, axes = mpl.subplots(nrows=1, ncols=2)
fig.set_size_inches(12,12)
axes[0].matshow(subdataMoreFriends.corr())
axes[0].set_title(">= Mean Correlation Matric")
axes[1].matshow(subdataLessFriends.corr())
axes[1].set_title("< Mean Correlation Matric")
#mpl.show()

#Investigate differences between old and young
subdataOld = clusterdata.where(clusterdata['age'] >= clusterdata['age'].mean())
subdataYoung = clusterdata.where(clusterdata['age'] < clusterdata['age'].mean())

fig, axes = mpl.subplots(nrows=1, ncols=2)
fig.set_size_inches(12,12)
axes[0].matshow(subdataOld.corr())
axes[0].set_title(">= Mean Correlation Matrix: Age")
axes[1].matshow(subdataYoung.corr())
axes[1].set_title("< Mean Correlation Matrix: Age")
mpl.show()
"""

#Rescale data
from sklearn.preprocessing import StandardScaler

scaleddata = StandardScaler().fit_transform(clusterdata)

#scaleddata.shape

#Run SKLearn's KMeans
from sklearn.cluster import KMeans

kr = range(2, 11)
scores = []

for kloop in kr:
  kmeans = KMeans(n_clusters=kloop+1).fit(scaleddata)
  pred = kmeans.predict(scaleddata)
  scores.append(kmeans.inertia_)

mpl.plot(kr,scores)
mpl.title('Elbow Plot For KMeans')
mpl.xlabel('Value of K')
mpl.ylabel('Inertia')
mpl.savefig('ElbowPlot.pdf', dpi=400)
print("Elbow Plot For KMeans Clustering with k = 5 saved as 'ElbowPlot.pdf' in current directory.")

#Plot average silhoutte score of clustering
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_samples, silhouette_score

range_n_clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10]
scores = []

#produce silhouette avg for clustering and append to list
for n_clusters in range_n_clusters:
    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(scaleddata)
    scores.append(silhouette_score(scaleddata, cluster_labels))

#plot silhouette scores
mpl.plot(range_n_clusters,scores)
mpl.title('Silhouette Score on the Value of K')
mpl.xlabel('Value of K')
mpl.ylabel('Silhoutte Score')
mpl.savefig('SilhouetteScores.pdf', dpi=400)
print("Silhouette Scores for KMeans Clustering with k = 5 saved as 'SilhouetteScores.pdf' in current directory.")

#Reduce dimensionality
from sklearn.decomposition import PCA
from matplotlib.pyplot import figure
pca = PCA(n_components=2).fit(scaleddata)
reduced_data = pca.transform(scaleddata)

#Print most influential attributes to each component
#print(pca.components_)

"""
#PCA visualizaiton
for i in range(2, 10):
  pred = KMeans(n_clusters=i).fit_predict(scaleddata)
  mpl.subplot(2, 4, (i-1))
  mpl.scatter(reduced_data[:,0],reduced_data[:,1],c=pred, cmap='viridis')
figure(num=None, figsize=(25, 25))
#mpl.show()

#Gender on age visualization
pred = KMeans(n_clusters=5).fit_predict(scaleddata)
data = pd.DataFrame(data=data)

mpl.scatter(data['age'],data['gradyear'],c=pred, cmap='viridis')
mpl.title('Gradyear on Age, K=5')
mpl.xlabel('Age')
mpl.ylabel('Gradyear')
mpl.yticks(ticks=[2006, 2007, 2008, 2009])
#mpl.show()

#age and friends visualization
pred = KMeans(n_clusters=5).fit_predict(scaleddata)

mpl.scatter(data['friends'],data['age'],c=pred, cmap='viridis')
mpl.title('Age on Friends, K=5')
mpl.xlabel('Friends')
mpl.ylabel('Age')
#mpl.show()

#friends and gender visualization
pred = KMeans(n_clusters=5).fit_predict(scaleddata)

mpl.scatter(data['friends'],data['gender'],c=pred, cmap='viridis')
mpl.title('Gender on Friends, K=5')
mpl.xlabel('Friends')
mpl.ylabel('Gender')
#mpl.show()

#age and gender visualization
pred = KMeans(n_clusters=5).fit_predict(scaleddata)

mpl.scatter(data['age'],data['gender'],c=pred, cmap='viridis')
mpl.title('Gender on Age, K=5')
mpl.xlabel('Age')
mpl.ylabel('Gender')
#mpl.show()
"""

#Run Hierarchical Clustering
from sklearn.cluster import AgglomerativeClustering

prediction = AgglomerativeClustering().fit_predict(scaleddata)

#Visualize dendrogram to determine the optimal number of clusters
import scipy.cluster.hierarchy as shc

#mpl.figure(figsize=(10, 7))
#mpl.title("Profiles Dendrogram")
#dend = shc.dendrogram(shc.linkage(scaleddata, method='ward'))
print("Dendrogram not created because of the time taken to compute. To enable it, uncomment the line 263.")

#Calculate similarity between agglomerative and kmeans clustering
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans

#Perform clustering from both methods
predictionHClustering = AgglomerativeClustering(n_clusters=5).fit_predict(scaleddata)

kmeans = KMeans(n_clusters=5).fit(scaleddata)
predictionKMeans = kmeans.predict(scaleddata)

#Reduce dimensionality
from sklearn.decomposition import PCA
from matplotlib.pyplot import figure
pca = PCA(n_components=2).fit(scaleddata)
reduced_data = pca.transform(scaleddata)

#Plot side by side of resulting clusters from both methods in 2 dimensional space via pca
fig, axes = mpl.subplots(nrows=1, ncols=2)
fig.set_size_inches(8,8)
fig.suptitle('Comparison of KMeans and Agglomerative Clustering, K=5')
axes[0].scatter(reduced_data[:,0],reduced_data[:,1],c=predictionKMeans, cmap='viridis')
axes[0].set_title("KMeans Clustering")
axes[1].scatter(reduced_data[:,0],reduced_data[:,1],c=predictionHClustering, cmap='viridis')
axes[1].set_title("Agglomerative Clustering")
mpl.savefig('Comparison.pdf', dpi=400)
print("Comparison of KMeans and Agglomerative Clustering, k = 5 saved as 'Comparison.pdf' in current directory.")

#Make final clustering groups
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=5).fit(scaleddata)
predictionKMeans = kmeans.predict(scaleddata)

#Make interpretations by analyzing each cluster

#Cluster 0 = young females (avg gender = 0.03, avg age = 17.70, avg friends = 112)
#Cluster 1 = younge females (avg gender = 0, avg age = 16.29, avg friends = 22.97, graduated 2008, more into sports)
#Cluster 2 = young males (avg gender = 1, avg age = 17.5)
#Cluster 3 = young females (avg gender = 0, avg age = 18.2, avg friends = 21.6, graduated 2006)
#Cluster 4 = old females (gender avg = 0.24, age avg = 101)

avgList = pd.DataFrame(columns=data.columns)
avgList.insert(0, "Cluster", -1)
print(data.columns.dtype)
for i in range(0,5):
  sample = data.iloc[predictionKMeans==i].mean()
  avgList.loc[i] = sample.T
  avgList['Cluster'].iloc[i] = i

#way to compare the average values for each cluster to determine their defining characteristics
avgList.to_csv('clusterDescriptions.csv')
print("Cluster Descriptions saved to current director as: 'clusterDescriptions.csv'")
