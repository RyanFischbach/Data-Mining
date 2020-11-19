"""
TweetCluster_Fischbach.py

Ryan Fischbach
Dr. Khuri
CSC373
11/13/2020

Assignment 5: Tweet Clustering

This script preprocesses the 'elections.csv', 'covid.csv', and 'usopen.csv' files, prints useful information
and outputs useful plots, clusters the data, then outputs useful plots.

To run, type "Python tweetcluster_fischbach.py {absolute path to the 3 csvs}" into the CLI.

Note: the visualizations and outputs will be saved in the same directory as the data.
"""

#! pip3 install nltk

# Import relevant libraries. Libraries needed: numpy, pandas, seaborn, sys, os, sklearn, matplotlib, nltk, re
import numpy as np
import pandas as pd
import seaborn as sns
import sys
import os
import matplotlib.pyplot as mpl
import sys
import os
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import Fischbach_KMeansFromScratch
from Fischbach_KMeansFromScratch import KMeansFromScratch

try:
    dir = sys.argv[1]

    if (not os.path.isdir(dir)):
        print(
            "Please enter a valid absolute path to the directory housing the 3 csv files.")
        sys.exit(1)

except:
    print("Please enter a valid absolute path to the directory housing the csv files after the .py file. For help see comments.")
    sys.exit(1)

try:
    os.chdir(dir)

    # Read data
    electionsdata = pd.read_csv('elections.csv')
    coviddata = pd.read_csv('covid.csv')
    usopendata = pd.read_csv('usopen.csv')
except:
    print("CSVs could not be read. Please enter a valid absolute path to the directory of the csv files and make sure the file is there..")
    sys.exit(1)

#drop duplicates
coviddata.drop_duplicates(inplace=True)
electionsdata.drop_duplicates(inplace=True)
usopendata.drop_duplicates(inplace=True)

#filter to just tweets
coviddata = coviddata['tweet']
electionsdata = electionsdata['tweet']
usopendata = usopendata['tweet']

#combine samples
combinedData = pd.concat((coviddata, electionsdata, usopendata), axis=0)
data = pd.DataFrame(data=combinedData)
data['tweet'].astype('string')

labels = np.zeros(((len(data)), 1))
labels[0:(len(coviddata)-1)] = 0
labels[(len(coviddata)):(len(electionsdata)+(len(coviddata)-1)) ] = 1
labels[(len(electionsdata)+(len(coviddata))):(len(electionsdata)+len(coviddata)+ len(usopendata) -1)] = 2

#download nltk package

nltk.download('stopwords')
nltk.download('punkt')

#citation: https://pythonhealthcare.org/2018/12/14/101-pre-processing-data-tokenization-stemming-and-removal-of-stop-words/

#remove punctuation and numbers
def text_lowercase(text):
    return text.lower()

data.loc[:,"tweet"] = data.tweet.apply(lambda x : " ".join(re.findall('[\w]+',x)))

#to lowercase
data.loc[:,"tweet"] = data.tweet.apply(lambda x : str.lower(x))

#tokenize words
data['tweet'] = data.apply(lambda row: nltk.word_tokenize(row['tweet']), axis=1)

#remove stopwords
stops = set(stopwords.words("english"))

def remove_stops(row):
    my_list = row['tweet']
    meaningful_words = [w for w in my_list if not w in stops]
    return (meaningful_words)

data['tweet'] = data.apply(remove_stops, axis=1)

#stem words
from nltk.stem import PorterStemmer
stemming = PorterStemmer()

def stem_list(row):
    my_list = row['tweet']
    stemmed_list = [stemming.stem(word) for word in my_list]
    return (stemmed_list)

data['tweet'] = data.apply(stem_list, axis=1)

#rejoin tokenized words
def rejoin_words(row):
    my_list = row['tweet']
    joined_words = ( " ".join(my_list))
    return joined_words

data['tweet'] = data.apply(rejoin_words, axis=1)

#encode words with tfidf
from sklearn.feature_extraction.text import TfidfVectorizer

#create a vectorizer and use ngram size 1
vectorizer = TfidfVectorizer(ngram_range=(1,1))

matrixData = vectorizer.fit_transform(data.tweet)
matrixData = np.array(matrixData.toarray())

#sanity check... shape is (number of samples, dictionary size)
matrixData.shape

#shuffle data
from sklearn.utils import shuffle
x, y = shuffle(matrixData, labels, random_state=0)

benchmark = pd.DataFrame(data=x, columns=vectorizer.vocabulary_)
benchmark.insert(0, 'label', y)
benchmark.to_csv('benchmarkDataset.csv')
print("Benchmark Dataset saved as 'benchmarkDataset.csv' to current directory.")

#SKLearn Clustering
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3).fit(x)
predictionKMeans = kmeans.predict(x)

#From scratch clustering
kmeansFromScratch = KMeansFromScratch()
pred = kmeansFromScratch.cluster(x, 3)

#Reduce dimensionality
from sklearn.decomposition import PCA
pca = PCA(n_components=2).fit(x)
reduced_data = pca.transform(x)

#Plot side by side of resulting clusters from both methods in 2 dimensional space via pca
fig, axes = mpl.subplots(nrows=1, ncols=2)
fig.set_size_inches(8,8)
fig.suptitle('Comparison of SKLearn KMeans (k = 3) and True Labels')
axes[0].scatter(reduced_data[:,0],reduced_data[:,1],c=predictionKMeans, cmap='viridis')
axes[0].set_title("KMeans Clustering")
axes[1].scatter(reduced_data[:,0],reduced_data[:,1],c=labels, cmap='viridis')
axes[1].set_title("True Labels")
mpl.savefig('ComparisonSKLearn.pdf', dpi=400)
print("Comparison of KMeans and True Labels, k = 3 saved as 'ComparisonSKLearn.pdf' in current directory.")

#Plot side by side of resulting clusters from both methods in 2 dimensional space via pca
fig, axes = mpl.subplots(nrows=1, ncols=2)
fig.set_size_inches(8,8)
fig.suptitle('Comparison of From Scratch KMeans (k = 3) and True Labels')
axes[0].scatter(reduced_data[:,0],reduced_data[:,1],c=pred, cmap='viridis')
axes[0].set_title("KMeans Clustering")
axes[1].scatter(reduced_data[:,0],reduced_data[:,1],c=labels, cmap='viridis')
axes[1].set_title("True Labels")
mpl.savefig('ComparisonFromScratch.pdf', dpi=400)
print("Comparison of KMeans and True Labels, k = 3 saved as 'ComparisonFromScratch.pdf' in current directory.")

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x, labels, test_size=0.4, random_state=1)

from sklearn.linear_model import SGDClassifier

sgd = SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42, max_iter=5, tol=None)
sgd.fit(X_train, y_train)
y_pred = sgd.predict(X_test)

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, confusion_matrix

print('SGD classifier accuracy %s' % accuracy_score(y_pred, y_test))
dict = classification_report(y_test, y_pred)
print(dict)
