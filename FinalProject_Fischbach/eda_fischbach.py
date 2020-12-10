"""
Preprocess_Fischbach.py

Ryan Fischbach
Dr. Khuri
CSC373
12/4/2020

Final Project: Scraping Tweets For Classification Of Stock Price

This script takes in the preprocessed_stocks.csv and preprocessed_tweets.csv files and generates visualizations and other EDA to better understand the data.

To run, type "Python eda_fischbach.py {absolute directory path containing CSVs}" into the CLI.
"""

# Import relevant libraries
import numpy as np
import pandas as pd
import sys
import os
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
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

    #Read in preprocessed csv data
    combinedTweets = pd.read_csv('preprocessed_tweets.csv')
    preprocessedStocks = pd.read_csv('preprocessed_stocks.csv')
except:
    print("CSV could not be read. Please enter a valid absolute path to the directory of purchases.csv and make sure the file is there..")
    sys.exit(1)

#Convert date format
combinedTweets['Date'] = pd.to_datetime(combinedTweets['Date'], dayfirst=True)
preprocessedStocks['Date'] = pd.to_datetime(preprocessedStocks['Date'], dayfirst=True)

# Import the wordcloud library
from wordcloud import WordCloud

# Join the different processed titles together.
long_string = ','.join(list(combinedTweets['Tweets'].values))
# Create a WordCloud object
wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=3, contour_color='steelblue')
# Generate a word cloud
wordcloud.generate(long_string)
# Visualize the word cloud
wordcloud.to_image()

import matplotlib.dates as mdates
import matplotlib.cbook as cbook

years = mdates.YearLocator()   # every year
months = mdates.MonthLocator()  # every month
years_fmt = mdates.DateFormatter('%Y')

fig, ax = plt.subplots()
ax.plot('Date', 'Count', data=combinedTweets)

# format the ticks
ax.xaxis.set_major_locator(years)
ax.xaxis.set_major_formatter(years_fmt)
ax.xaxis.set_minor_locator(months)

ax.set_xlabel('Date')
ax.set_ylabel('Number of Tweets')
ax.set_title('Number of Tweets on Date')

# format the coords message box
ax.format_xdata = mdates.DateFormatter('%Y-%m-%d')
ax.format_ydata = lambda x: '$%1.2f' % x  # format the price.
ax.grid(True)

# rotates and right aligns the x labels, and moves the bottom of the
# axes up to make room for them
fig.autofmt_xdate()

plt.show()

import matplotlib.dates as mdates
import matplotlib.cbook as cbook

fig, ax = plt.subplots()
ax.hist('Labels', data=preprocessedStocks)

# format the ticks
ax.set_xlabel('Label')
ax.set_ylabel('Number of Occurences')
ax.set_title('Number of Occurences of Each Label')
ax.set_xticklabels(['> 2% Change', '<= 2% Change'])

# format the coords message box
ax.set_xticks((0,1))
ax.grid(True)

# rotates and right aligns the x labels, and moves the bottom of the
# axes up to make room for them

plt.show()

import matplotlib.dates as mdates
import matplotlib.cbook as cbook

fig, ax = plt.subplots()
ax.hist('Sentiment', data=combinedTweets)

# format the ticks
ax.set_ylabel('Occurences')
ax.set_title('Number of Occurences of Each Sentiment')
ax.set_xticklabels(['Negative', 'Positive'])

# format the coords message box
ax.set_xticks((0,1))
ax.grid(True)

# rotates and right aligns the x labels, and moves the bottom of the
# axes up to make room for them

plt.show()

# Commented out IPython magic to ensure Python compatibility.
# Determine most popular words
from sklearn.feature_extraction.text import CountVectorizer
sns.set_style('whitegrid')
# %matplotlib inline
# Helper function
def plot_10_most_common_words(count_data, count_vectorizer):
    import matplotlib.pyplot as plt
    words = count_vectorizer.get_feature_names()
    total_counts = np.zeros(len(words))
    for t in count_data:
        total_counts+=t.toarray()[0]
    
    count_dict = (zip(words, total_counts))
    count_dict = sorted(count_dict, key=lambda x:x[1], reverse=True)[0:10]
    words = [w[0] for w in count_dict]
    counts = [w[1] for w in count_dict]
    x_pos = np.arange(len(words)) 
    
    plt.figure(2, figsize=(15, 15/1.6180))
    plt.subplot(title='10 most common words')
    sns.set_context("notebook", font_scale=1.25, rc={"lines.linewidth": 2.5})
    sns.barplot(x_pos, counts, palette='husl')
    plt.xticks(x_pos, words, rotation=90) 
    plt.xlabel('words')
    plt.ylabel('counts')
    plt.show()
# Initialise the count vectorizer with the English stop words
count_vectorizer = CountVectorizer(stop_words='english')
# Fit and transform the processed titles
count_data = count_vectorizer.fit_transform(combinedTweets.Tweets)
# Visualise the 10 most common words
plot_10_most_common_words(count_data, count_vectorizer)

import warnings
warnings.simplefilter("ignore", DeprecationWarning)
# Load the LDA model from sk-learn
from sklearn.decomposition import LatentDirichletAllocation as LDA
 
# Helper function
def print_topics(model, count_vectorizer, n_top_words):
    words = count_vectorizer.get_feature_names()
    for topic_idx, topic in enumerate(model.components_):
        print("\nTopic #%d:" % topic_idx)
        print(" ".join([words[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
        
# Tweak the two parameters below
number_topics = 5
number_words = 10
# Create and fit the LDA model
lda = LDA(n_components=number_topics, n_jobs=-1)
lda.fit(count_data)
# Print the topics found by the LDA model
print("Topics found via LDA:")
print_topics(lda, count_vectorizer, number_words)
