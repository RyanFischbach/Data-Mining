"""
Classify_Fischbach.py

Ryan Fischbach
Dr. Khuri
CSC373
12/4/2020

Final Project: Scraping Tweets For Classification Of Stock Price

This script takes in the preprocessed_stocks.csv and preprocessed_tweets.csv files and classifies them.

To run, type "Python classify_fischbach.py {absolute directory path containing CSVs}" into the CLI.
"""

# Import relevant libraries
import numpy as np
import pandas as pd
import sys
import os
import datetime
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

try:
    dir = sys.argv[1]

    if (not os.path.isdir(dir)):
        print(
            "Please enter a valid absolute path to the directory housing the csvs.")
        sys.exit(1)

except:
    print("Please enter a valid absolute path to the directory of the directory housing csvs after the .py file.")
    sys.exit(1)

try:
    os.chdir(dir)

    # Read data
    #combinedTweets = pd.read_csv('fiveday_preprocessed_tweets.csv')
    #preprocessedStocks = pd.read_csv('fiveday_preprocessed_stocks.csv')

    #Read in preprocessed csv data
    combinedTweets = pd.read_csv('preprocessed_tweets.csv')
    preprocessedStocks = pd.read_csv('preprocessed_stocks.csv')
except:
    print("CSV could not be read. Please enter a valid absolute path to the directory of purchases.csv and make sure the file is there..")
    sys.exit(1)



#Set preprocessed data's index to it's date for easy filtering
combinedTweets.set_index(['Date'], inplace=True)
preprocessedStocks.set_index(['Date'], inplace=True)

#Try using sentiment to predict
noTweets = combinedTweets.drop(columns=['Tweets'], axis=1)

#Subset data to yield train/test split (roughly 70/30)
x_train = combinedTweets[combinedTweets.index < '2020-03-01']
y_train = preprocessedStocks[preprocessedStocks.index < '2020-03-01']
x_test = combinedTweets[combinedTweets.index >= '2020-03-01']
y_test = preprocessedStocks[preprocessedStocks.index >= '2020-03-01']

x_train_notweets = noTweets[noTweets.index < '2020-03-01']
x_test_notweets = noTweets[noTweets.index >= '2020-03-01']

#create df to store scores
models_df = pd.DataFrame(index=range(5))
entries = []

"""
#Create pipeline to determine best parameters for KNN. This is commented out because this is a brute force approach.
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', KNeighborsClassifier()),
])

#create parameter list to explore
parameters = {
    'clf__n_neighbors': (3, 5, 7, 9, 11, 13, 15),
}

#perform exploration 
grid_search_tune = GridSearchCV(pipeline, parameters, cv=2, n_jobs=2, verbose=3)
grid_search_tune.fit(x_train_notweets, y_train.Labels)

#output but parameters found
print("Best parameters set:")
print(grid_search_tune.best_score_)
print(grid_search_tune.best_params_)
"""
#normalize attributes with scaler
scaler = StandardScaler()
model = KNeighborsClassifier(n_neighbors=5)
model_name = model.__class__.__name__
#perform classification five times to determine average accuracy
for fold in range(5):
  model.fit(scaler.fit_transform(x_train_notweets), y_train.Labels)
  y_pred = model.predict(scaler.transform(x_test_notweets))
  entries.append((model_name, accuracy_score(y_test.Labels, y_pred)))

#Create pipeline to determine best tfidf parameters for GaussianNB. This is commented out because this is a brute force approach.
"""
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', GaussianNB()),
])

#create parameter list to explore
parameters = {
    'tfidf__max_df': (0.5, 0.6, 0.7, 0.8, 0.9),
    'tfidf__min_df': (0.05, 0.1, 0.2, 0.3, 0.4, 0.5),
    'tfidf__ngram_range': [(1, 1), (1, 2), (2, 2), (1, 3), (1, 4), (2, 4), (4, 8), (4, 10), (6, 8), (6, 10), (8, 10), (1, 8), (2, 8)]
}

#perform exploration 
grid_search_tune = GridSearchCV(pipeline, parameters, cv=2, n_jobs=2, verbose=3)
grid_search_tune.fit(x_train.CombTweets, y_train.Labels)

#output but parameters found
print("Best parameters set:")
print(grid_search_tune.best_score_)
print(grid_search_tune.best_params_)
"""
#The best parameters for Gaussian NB were determine to be: {'tfidf__max_df': 0.7, 'tfidf__min_df': 0.2, 'tfidf__ngram_range': (1, 2)}

#encode words with tfidf
from sklearn.metrics import classification_report

vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=0.2, max_df=0.7, sublinear_tf=True)

tfidf_train = vectorizer.fit_transform(x_train.Tweets).toarray()
tfidf_test = vectorizer.transform(x_test.Tweets).toarray()

#add in count and sentiment
tfidf_train = np.concatenate((tfidf_train, x_train.Count.to_numpy().reshape((len(x_train), 1))), axis=1)
tfidf_test = np.concatenate((tfidf_test, x_test.Count.to_numpy().reshape((len(x_test), 1))), axis=1)
tfidf_train = np.concatenate((tfidf_train, x_train.OpenPctChange.to_numpy().reshape((len(x_train), 1))), axis=1)
tfidf_test = np.concatenate((tfidf_test, x_test.OpenPctChange.to_numpy().reshape((len(x_test), 1))), axis=1)
tfidf_train = np.concatenate((tfidf_train, x_train.Volume.to_numpy().reshape((len(x_train), 1))), axis=1)
tfidf_test = np.concatenate((tfidf_test, x_test.Volume.to_numpy().reshape((len(x_test), 1))), axis=1)
model = GaussianNB()

model_name = model.__class__.__name__
#perform classification five times to determine average accuracy
for fold in range(5):
  model.fit(tfidf_train, y_train.Labels)
  y_pred = model.predict(tfidf_test)
  entries.append((model_name, accuracy_score(y_test.Labels, y_pred)))

#Create pipeline to determine best tfidf parameters for RandomForest. This is commented out because this is a brute force approach.
"""
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=1000)),
    ('clf', RandomForestClassifier(random_state=0)),
])

#create parameter list to explore
parameters = {
    'tfidf__max_df': (0.5, 0.6, 0.7, 0.8, 0.9),
    'tfidf__min_df': (0.05, 0.1, 0.2, 0.3, 0.4, 0.5),
    'tfidf__ngram_range': [(1, 1), (1, 2), (2, 2), (1, 3), (1, 4), (2, 4), (4, 8), (4, 10), (6, 8), (6, 10), (8, 10), (1, 8), (2, 8)],
    'clf__n_estimators': (10, 25, 50, 75, 100, 200, 500, 1000),
    'clf__max_depth': (3, 5, 10, 15, 20, 25)
}

#perform exploration 
grid_search_tune = GridSearchCV(pipeline, parameters, cv=2, n_jobs=2, verbose=3)
grid_search_tune.fit(x_train.CombTweets, y_train.Labels)

#output but parameters found
print("Best parameters set:")
print(grid_search_tune.best_score_)
print(grid_search_tune.best_params_)
"""
#The best parameters for RandomForest were determine to be: {'tfidf__max_df': 0.5, 'tfidf__min_df': 0.1, 'tfidf__ngram_range': (1, 1)}

#encode words with tfidf
vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=0.1, max_df=0.8)

tfidf_train = vectorizer.fit_transform(x_train.Tweets).toarray()
tfidf_test = vectorizer.transform(x_test.Tweets).toarray()

#add in count and sentiment
tfidf_train = np.concatenate((tfidf_train, x_train.Count.to_numpy().reshape((len(x_train), 1))), axis=1)
tfidf_test = np.concatenate((tfidf_test, x_test.Count.to_numpy().reshape((len(x_test), 1))), axis=1)
tfidf_train = np.concatenate((tfidf_train, x_train.OpenPctChange.to_numpy().reshape((len(x_train), 1))), axis=1)
tfidf_test = np.concatenate((tfidf_test, x_test.OpenPctChange.to_numpy().reshape((len(x_test), 1))), axis=1)
tfidf_train = np.concatenate((tfidf_train, x_train.Volume.to_numpy().reshape((len(x_train), 1))), axis=1)
tfidf_test = np.concatenate((tfidf_test, x_test.Volume.to_numpy().reshape((len(x_test), 1))), axis=1)

model = RandomForestClassifier(n_estimators=100, max_depth=3)

model_name = model.__class__.__name__
for fold in range(5):
  model.fit(tfidf_train, y_train.Labels)
  y_pred = model.predict(tfidf_test)
  entries.append((model_name, accuracy_score(y_test.Labels, y_pred)))

#Create pipeline to determine best tfidf parameters for LinearSVC. This is commented out because this is a brute force approach.
"""
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LinearSVC()),
])

#create parameter list to explore
parameters = {
    'tfidf__max_df': (0.5, 0.6, 0.7, 0.8, 0.9),
    'tfidf__min_df': (0.05, 0.1, 0.2, 0.3, 0.4, 0.5),
    'tfidf__ngram_range': [(1, 1), (1, 2), (2, 2), (1, 3), (1, 4), (2, 4), (4, 8), (4, 10), (6, 8), (6, 10), (8, 10), (1, 8), (2, 8)]
}

#perform exploration 
grid_search_tune = GridSearchCV(pipeline, parameters, cv=2, n_jobs=2, verbose=3)
grid_search_tune.fit(x_train.Tweets, y_train.Labels)

#output but parameters found
print("Best parameters set:")
print(grid_search_tune.best_score_)
print(grid_search_tune.best_params_)
"""
#The best parameters for LinearSVC were determine to be: {'tfidf__max_df': 0.8, 'tfidf__min_df': 0.2, 'tfidf__ngram_range': (1, 2)}

#encode words with tfidf
vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=0.2, max_df=0.8)

tfidf_train = vectorizer.fit_transform(x_train.Tweets).toarray()
tfidf_test = vectorizer.transform(x_test.Tweets).toarray()

#add in count and sentiment
tfidf_train = np.concatenate((tfidf_train, x_train.Count.to_numpy().reshape((len(x_train), 1))), axis=1)
tfidf_test = np.concatenate((tfidf_test, x_test.Count.to_numpy().reshape((len(x_test), 1))), axis=1)
tfidf_train = np.concatenate((tfidf_train, x_train.OpenPctChange.to_numpy().reshape((len(x_train), 1))), axis=1)
tfidf_test = np.concatenate((tfidf_test, x_test.OpenPctChange.to_numpy().reshape((len(x_test), 1))), axis=1)
tfidf_train = np.concatenate((tfidf_train, x_train.Volume.to_numpy().reshape((len(x_train), 1))), axis=1)
tfidf_test = np.concatenate((tfidf_test, x_test.Volume.to_numpy().reshape((len(x_test), 1))), axis=1)

model = LinearSVC()

model_name = model.__class__.__name__
for fold in range(5):
  model.fit(tfidf_train, y_train.Labels)
  y_pred = model.predict(tfidf_test)
  entries.append((model_name, accuracy_score(y_test.Labels, y_pred)))

#Create pipeline to determine best tfidf parameters for LogisticRegression. This is commented out because this is a brute force approach.
"""
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression(random_state=0)),
])

#create parameter list to explore
parameters = {
    'tfidf__max_df': (0.5, 0.6, 0.7, 0.8, 0.9),
    'tfidf__min_df': (0.05, 0.1, 0.2, 0.3, 0.4, 0.5),
    'tfidf__ngram_range': [(1, 1), (1, 2), (2, 2), (1, 3), (1, 4), (2, 4), (4, 8), (4, 10), (6, 8), (6, 10), (8, 10), (1, 8), (2, 8)]
}

#perform exploration 
grid_search_tune = GridSearchCV(pipeline, parameters, cv=2, n_jobs=2, verbose=3)
grid_search_tune.fit(x_train.Tweets, y_train.Labels)

#output but parameters found
print("Best parameters set:")
print(grid_search_tune.best_score_)
print(grid_search_tune.best_params_)
"""
#The best parameters for LinearSVC were determine to be: {'tfidf__max_df': 0.8, 'tfidf__min_df': 0.2, 'tfidf__ngram_range': (1, 2)}

#encode words with tfidf
vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=0.2, max_df=0.8)

tfidf_train = vectorizer.fit_transform(x_train.Tweets).toarray()
tfidf_test = vectorizer.transform(x_test.Tweets).toarray()

#add in count and sentiment
tfidf_train = np.concatenate((tfidf_train, x_train.Count.to_numpy().reshape((len(x_train), 1))), axis=1)
tfidf_test = np.concatenate((tfidf_test, x_test.Count.to_numpy().reshape((len(x_test), 1))), axis=1)
tfidf_train = np.concatenate((tfidf_train, x_train.OpenPctChange.to_numpy().reshape((len(x_train), 1))), axis=1)
tfidf_test = np.concatenate((tfidf_test, x_test.OpenPctChange.to_numpy().reshape((len(x_test), 1))), axis=1)
tfidf_train = np.concatenate((tfidf_train, x_train.Volume.to_numpy().reshape((len(x_train), 1))), axis=1)
tfidf_test = np.concatenate((tfidf_test, x_test.Volume.to_numpy().reshape((len(x_test), 1))), axis=1)

model = LogisticRegression()
model_name = model.__class__.__name__
for fold in range(5):
  model.fit(tfidf_train, y_train.Labels)
  y_pred = model.predict(tfidf_test)
  entries.append((model_name, accuracy_score(y_test.Labels, y_pred)))

models_df = pd.DataFrame(entries, columns=['model_name', 'accuracy'])

#Determine mean and std dev of accuracy from each model
#Citation: https://www.kaggle.com/selener/multi-class-text-classification-tfidf
mean_accuracy = models_df.groupby('model_name').accuracy.mean()
std_accuracy = models_df.groupby('model_name').accuracy.std()
acc = pd.concat([mean_accuracy, std_accuracy], axis= 1, 
          ignore_index=True)
acc.columns = ['Mean Accuracy', 'Std Deviation']
print('Average Accuracy of Models Over 5 Iterations')
print(acc)

#plot models
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8,5))
sns.boxplot(x='model_name', y='accuracy',
            data=models_df, 
            color='lightblue').set_title('Performance of Models on TFIDF Vectorized Data')
plt.title("ACCURACY\n", size=14);

#Try temporal cross validation to yield best model
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit()

#encode words with tfidf using determined hyperparameters
vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=0.1, max_df=0.5, sublinear_tf=True)

models = [GaussianNB(),
          LogisticRegression(),
          LinearSVC(),
          RandomForestClassifier(n_estimators=100, max_depth=3),
          SGDClassifier()
]

cv_df = pd.DataFrame(index=range(5 * len(models)))
entries = []

for model in models:
  model_name = model.__class__.__name__

  for train_index, test_index in tscv.split(combinedTweets.Tweets):
    X_train, X_test = combinedTweets.Tweets[train_index], combinedTweets.Tweets[test_index]
    y_train, y_test = preprocessedStocks.Labels[train_index], preprocessedStocks.Labels[test_index]
    tfidf_train = vectorizer.fit_transform(X_train).toarray()
    tfidf_test = vectorizer.transform(X_test).toarray()
    model.fit(tfidf_train, y_train)
    accuracy = accuracy_score(model.predict(tfidf_test), y_test)
    entries.append((model_name, accuracy))
  
cv_df = pd.DataFrame(entries, columns=['model_name', 'accuracy'])

mean_accuracy = cv_df.groupby('model_name').accuracy.mean()
std_accuracy = cv_df.groupby('model_name').accuracy.std()

acc = pd.concat([mean_accuracy, std_accuracy], axis= 1, 
          ignore_index=True)
acc.columns = ['Mean Accuracy', 'Standard deviation']
print('Temporal Cross Validation Accuracy')
print(acc)

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8,5))
sns.boxplot(x='model_name', y='accuracy', 
            data=cv_df, 
            color='lightblue')
plt.title("ACCURACY\n", size=14);
