"""
Preprocess_Fischbach.py

Ryan Fischbach
Dr. Khuri
CSC373
12/4/2020

Final Project: Scraping Tweets For Classification Of Stock Price

This script takes in the amazonstock.csv and amazontweets.csv files and preprocesses them for classification.

To run, type "Python preprocess_fischbach.py {absolute directory path containing CSVs}" into the CLI.
"""

#!pip install tweet-preprocessor

# Import relevant libraries
import numpy as np
import pandas as pd
import sys
import os
import nltk
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
import re
import datetime
import sklearn

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
    stocks = pd.read_csv('amazonstock.csv')
    tweets = pd.read_csv('amazontweets.csv')
except:
    print("CSV could not be read. Please enter a valid absolute path to the directory of purchases.csv and make sure the file is there..")
    sys.exit(1)

#Preprocess Stock Information to Generate Labels

#Drop na and duplicate values
stocks.dropna(inplace=True)
stocks.drop_duplicates(inplace=True)

#Create date thresholds based on tweets
minDate = '2017-12-01'
maxDate = '2020-12-01'

#Check data types on attributes
stocks.dtypes

#Convert date column to datetime to allow for comparison
stocks['Date'] = pd.to_datetime(stocks['Date'])

#Check data types on attributes to ensure successful change
stocks.dtypes

#Filter data above minimum date and under max date
stocks = stocks[(stocks['Date'] >= minDate) & (stocks['Date'] < maxDate)]
stocks = stocks.reset_index(drop=True)
stocks.sort_values(by=['Date'], inplace=True)

#define percent close stock change
def percentStockChangeClose(df, i):
  return((stocks.iloc[i]['Close'] - stocks.iloc[i-1]['Close']) / stocks.iloc[i-1]['Close'])

#create labels array
labels = np.zeros((len(stocks),1))

#determine labels
for i in range(len(stocks)):
  label = None
  if(i != 0):
    #2% is an arbitrary cutoff but after research this appears to be a good target to prevent normal market volatility from interfering
    if(percentStockChangeClose(stocks, i) > 0.02):
      label = 1
    #elif(percentStockChange(stocks, i) < -0.02):
      #label = -1
    else:
      label = 0

    labels[i-1] = label

#create dataframe from labels
labelsDf = pd.DataFrame(data=labels, columns=['Labels'])
preprocessedStocks = pd.concat([stocks['Date'], labelsDf], axis=1)

#Sort values by date
preprocessedStocks.sort_values(by=['Date'], inplace=True)

#Preprocess tweets
tweets = pd.read_csv('amazontweets.csv')

len(tweets)

#Drop duplicates and na values
tweets.drop_duplicates(inplace=True)
tweets.dropna(inplace=True)

#Convert date format
tweets['date'] = pd.to_datetime(tweets['date'], dayfirst=True)

#Convert date to the same format as stock market (daily time window)
tweets['date'] = tweets['date'].apply(lambda t: t.strftime('%Y-%m-%d'))

#remove id attribute
tweets = tweets[['date', 'tweet']]

tweets.sort_values(by=['date'], inplace=True)

#Combine tweets from dates into 1 corpus per date and determine count of tweets per day
tweetDict = {}
countDict = {}

for i, row in tweets.iterrows():
  date = row['date']
  if row['date'] in tweetDict:
    tweetDict[date] = (tweetDict[date] + row['tweet'])
    countDict[date] = countDict[date] + 1
  else:
    tweetDict[date] = row['tweet']
    countDict[date] = 1

#turn dictionaries back into pandas dataframe
combinedTweets = pd.DataFrame(data=[tweetDict, countDict])

#swap axes to have dates as sample index
combinedTweets = combinedTweets.swapaxes("index", "columns")
combinedTweets.reset_index(inplace=True)
combinedTweets.rename(columns={"index": "Date", 0: "Tweets", 1: "Count"}, inplace=True)
combinedTweets['Date'] = pd.to_datetime(combinedTweets['Date'])

#Begin text preprocessing
combinedTweets['Tweets'] = combinedTweets['Tweets'].astype('string')

#download nltk package
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('movie_reviews')

#citation: https://pythonhealthcare.org/2018/12/14/101-pre-processing-data-tokenization-stemming-and-removal-of-stop-words/

#Citation: https://www.kaggle.com/sreejiths0/efficient-tweet-preprocessing
import preprocessor as p
#apply twitter specific preprocessing
def preprocess_tweet(row):
    text = row
    text = p.clean(text)
    return text

combinedTweets['Tweets'] = combinedTweets.Tweets.apply(preprocess_tweet)

#Tokenize
def identify_tokens(row):
    tweet = row['Tweets']
    tokens = nltk.word_tokenize(tweet)
    # taken only words (not punctuation)
    token_words = [w for w in tokens if w.isalpha()]
    return token_words

combinedTweets['Tweets'] = combinedTweets.apply(identify_tokens, axis=1)

#Stem
from nltk.stem import PorterStemmer
stemming = PorterStemmer()

def stem_list(row):
    my_list = row['Tweets']
    stemmed_list = [stemming.stem(word) for word in my_list]
    return (stemmed_list)

combinedTweets['Tweets'] = combinedTweets.apply(stem_list, axis=1)

#remove stopwords
from nltk.corpus import stopwords
stops = stopwords.words(['english', 'spanish'])
stops.extend(['amazon', 'breaking', 'news']) 

def remove_stops(row):
    my_list = row['Tweets']
    meaningful_words = [w for w in my_list if not w in stops]
    return (meaningful_words)

combinedTweets['Tweets'] = combinedTweets.apply(remove_stops, axis=1)

#rejoin words
def rejoin_words(row):
    my_list = row['Tweets']
    joined_words = ( " ".join(my_list))
    return joined_words

combinedTweets['Tweets'] = combinedTweets.apply(rejoin_words, axis=1)

#filter out empty tweets
combinedTweets = combinedTweets[combinedTweets['Tweets'].str.len() > 0]

#Remove tweets that exist when market is closed
combinedTweets.set_index(['Date'], inplace=True)
preprocessedStocks.set_index(['Date'], inplace=True)

ixs = combinedTweets.index.intersection(preprocessedStocks.index)
combinedTweets = combinedTweets.loc[ixs]
preprocessedStocks = preprocessedStocks.loc[ixs]

#see labels
#np.unique(preprocessedStocks.Labels.to_numpy(), return_counts=True)

df_majority = preprocessedStocks[preprocessedStocks.Labels==0]
df_minority = preprocessedStocks[preprocessedStocks.Labels!=0]
from sklearn.utils import resample

#downsample majority class
df_majority_downsampled = resample(df_majority, 
                                 replace=False,    # sample without replacement
                                 n_samples=len(df_minority),     # to match minority class size
                                 random_state=123) # reproducible results
#recombine                      
df_downsampled = pd.concat([df_majority_downsampled, df_minority])

#sort
df_downsampled.sort_index(inplace=True)
combinedTweets.sort_index(inplace=True)

#remove items where there is no intersection
ixs = combinedTweets.index.intersection(df_downsampled.index)
combinedTweets = combinedTweets.loc[ixs]
df_downsampled = df_downsampled.loc[ixs]

#add percent change in open and yesterday's volume to input data
for i, rows in combinedTweets.iterrows():
  currentiloc = combinedTweets.index.get_loc(i)
  yesterdayOpen = stocks.iloc[(currentiloc-1)].Open
  todayOpen = stocks.iloc[currentiloc].Open
  combinedTweets.loc[i, 'OpenPctChange'] = (todayOpen - yesterdayOpen)/yesterdayOpen
  combinedTweets.loc[i, 'Volume'] = stocks.iloc[currentiloc-1].Volume

#create 3 day rolling average dataframe of tweets
threeDayRollingTweets = combinedTweets.copy()
threeDayRollingTweets.reset_index(inplace=True)

#perform rolling
for i in range(len(threeDayRollingTweets)):
  if(i > 3):
    threeDayRollingTweets.loc[i, 'CombTweets'] = threeDayRollingTweets.loc[i, 'Tweets'] + threeDayRollingTweets.loc[i-2, 'Tweets'] + threeDayRollingTweets.loc[i-1, 'Tweets'] + threeDayRollingTweets.loc[i-3, 'Tweets'] + threeDayRollingTweets.loc[i-4, 'Tweets']
    threeDayRollingTweets.loc[i, 'CombCount'] = threeDayRollingTweets.loc[i, 'Count'] + threeDayRollingTweets.loc[i-2, 'Count'] + threeDayRollingTweets.loc[i-1, 'Count'] + threeDayRollingTweets.loc[i-3, 'Count'] + threeDayRollingTweets.loc[i-4, 'Count']

#drop top two rows with no combined tweets
threeDayRollingTweets = threeDayRollingTweets.iloc[4:]
threeDayRollingTweets.set_index('Date', inplace=True)

#remove items where there is no intersection
ixs = threeDayRollingTweets.index.intersection(df_downsampled.index)
threeDayRollingTweets = threeDayRollingTweets.loc[ixs]
threeDayRollingStocks = df_downsampled.loc[ixs]

#Perform sentiment analysis to add another attribute for classification using NaiveBayesAnalyzer from textblob
#Commented out because sentiment analysis takes a very long time
"""
import textblob
from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer
from textblob.decorators import requires_nltk_corpus

def sentimentAnalysis(row):
  blob = TextBlob(row.Tweets, analyzer=NaiveBayesAnalyzer())
  row['Sentiment'] = blob.sentiment.p_neg
  return row

combinedTweets = combinedTweets.apply(sentimentAnalysis, axis=1)
threeDayRollingTweets['Tweets'] = threeDayRollingTweets['CombTweets']
threeDayRollingTweets = threeDayRollingTweets.apply(sentimentAnalysis, axis=1)
"""

#export preprocessed tweets
combinedTweets.to_csv('preprocessed_tweets.csv')
print("'preprocessed_tweets.csv' saved to current directory." )

#output preprocessed stocks
df_downsampled.to_csv('preprocessed_stocks.csv')
print("'preprocessed_stocks.csv' saved to current directory." )

#output three day rolling preprocessed stocks
threeDayRollingStocks.to_csv('fiveday_preprocessed_stocks.csv')
print("'fiveday_preprocessed_stocks.csv' saved to current directory." )

#output three day rolling preprocessed tweets
threeDayRollingTweets.to_csv('fiveday_preprocessed_tweets.csv')
print("'fiveday_preprocessed_tweets.csv' saved to current directory." )
