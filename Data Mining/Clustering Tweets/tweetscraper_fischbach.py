"""
TweetScraper_Fischbach.py

Ryan Fischbach
Dr. Khuri
CSC373
11/13/2020

Assignment 5: Scraping Tweets For Clustering

This script uses the snscrape library to pull 1,000 tweets about the 'election', 'covid', and 'usopen'.

To run, type "Python tweetscraper_fischbach.py {absolute directory path for CSVs to be output}" into the CLI.
"""

#SNScrape is needed to run this script
#! pip3 install snscrape

import snscrape.modules.twitter as sntwitter
import csv
import sys
import os
maxTweets = 1000

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

#Use csv writer to get election tweets
csvFile = open('elections.csv', 'a', newline='', encoding='utf8')
csvWriter = csv.writer(csvFile)
csvWriter.writerow(['id','date','tweet',]) 

for i,tweet in enumerate(sntwitter.TwitterSearchScraper('election + place:01fbe706f872cb32 + since:2020-10-31 until:2020-11-12 -filter:links -filter:replies').get_items()):
        if i > maxTweets :
            break  
        csvWriter.writerow([tweet.id, tweet.date, tweet.content])
csvFile.close()
print("1,000 election tweets saved as 'elections.csv' in output directory.")

#Use csv writer to get covid tweets
csvFile = open('covid.csv', 'a', newline='', encoding='utf8')
csvWriter = csv.writer(csvFile)
csvWriter.writerow(['id','date','tweet',]) 

for i,tweet in enumerate(sntwitter.TwitterSearchScraper('covid  + since:2020-3-01 until:2020-11-12 -filter:links -filter:replies').get_items()):
        if i > maxTweets :
            break  
        csvWriter.writerow([tweet.id, tweet.date, tweet.content])
csvFile.close()
print("1,000 covid tweets saved as 'elections.csv' in output directory.")

#Use csv writer to get usopen tweets
csvFile = open('usopen.csv', 'a', newline='', encoding='utf8')

csvWriter = csv.writer(csvFile)
csvWriter.writerow(['id','date','tweet',]) 

for i,tweet in enumerate(sntwitter.TwitterSearchScraper('usopen + since:2020-3-01 until:2020-11-12 -filter:links -filter:replies').get_items()):
        if i > maxTweets :
            break  
        csvWriter.writerow([tweet.id, tweet.date, tweet.content])
csvFile.close()
print("1,000 usopen tweets saved as 'elections.csv' in output directory.")
