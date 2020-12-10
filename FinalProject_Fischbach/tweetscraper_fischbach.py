"""
TweetScraper_Fischbach.py

Ryan Fischbach
Dr. Khuri
CSC373
12/4/2020

Final Project: Scraping Tweets For Classification Of Stock Price

This script uses the snscrape library to pull 1,000 tweets from the last week.

To run, type "Python tweetscraper_fischbach.py {absolute directory path for CSVs to be output}" into the CLI.
"""

#SNScrape is needed to run this script
#! pip3 install snscrape

import snscrape.modules.twitter as sntwitter
import csv
import sys
import os
maxTweets = 50000

#Query path to data
try:
  dir = sys.argv[1]

  if(not os.path.isdir(dir)):
    print("Please enter a valid absolute path for CSV output.")
    sys.exit(1)
    
except:
  print("Please enter a valid absolute path for CSV output after the .py in the CLI.")
  sys.exit(1)

try:
  os.chdir(dir)
except:
  print("Could not change directory. Please enter a valid absolute path for the output.")
  sys.exit(1)

#Use csv writer to get amazon tweets
csvFile = open('amazontweets.csv', 'a', newline='', encoding='utf8')
csvWriter = csv.writer(csvFile)
csvWriter.writerow(['id','date','tweet']) 

for i,tweet in enumerate(sntwitter.TwitterSearchScraper('amazon + since:2017-12-01 until:2020-12-01 lang:en -filter:links -filter:replies').get_items()):
        if i > maxTweets :
            break
        csvWriter.writerow([tweet.id, tweet.date, tweet.content])
csvFile.close()
print("50,000 tweets including 'amazon breaking news' saved as 'amazontweets.csv' in output directory.")
