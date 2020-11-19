"""
Purchases_Fischbach.py

Ryan Fischbach
Dr. Khuri
CSC373
11/13/2020

Assignment 5: Purchases.csv Analysis Via Association Analysis

This script preprocesses the purchases.csv file, prints useful information
and outputs useful plots, then generates itemsets and mines rules.

To run, type "Python purchases_fischbach.py {asbsolute path to directory with purchases.csv}" into the CLI.

Note: plots will be saved in the same directory as the data

Necessary packages:
pandas
mlxtend
seaborn
matplotlib.pyplot
"""

#pip install mlxtend

# ! pip install mlxtend
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import seaborn as sns
import sys
import os
import matplotlib.pyplot as plt
import numpy as np

#Create list of column names
namesList = []
itemName = 'Item '
for i in range(1,33):
  namesList.append((itemName + str(i)))

try:
    dir = sys.argv[1]

    if (not os.path.isdir(dir)):
        print(
            "Please enter a valid absolute path to the directory housing purchases.csv.")
        sys.exit(1)

except:
    print("Please enter a valid absolute path to the directory of the directory housing purchases.csv after the .py file.")
    sys.exit(1)

try:
    os.chdir(dir)

    # Read data
    data = pd.read_csv('purchases.csv', error_bad_lines=False, header=None, sep=',', names=namesList)
except:
    print("CSV could not be read. Please enter a valid absolute path to the directory of purchases.csv and make sure the file is there..")
    sys.exit(1)

#Get total number of transactions
totalNumberOfTransactions = len(data)
print("Total Number of Transactions =", totalNumberOfTransactions)

#remove na values
data = data.apply(lambda x: list(x.dropna().values), axis=1)

#print transaction list data
data

#Apply transaction encoder to create sparse matrix
te = TransactionEncoder()
te_ary = te.fit(data.to_list()).transform(data.to_list())
df = pd.DataFrame(te_ary, columns=te.columns_)
df = df.astype('int')

#Get info about transaction encoded dataframe
#df.info()
#df.describe()

#Density of matrix (proportion of nonzero matrix cells)
density = df[df==1].count().sum() / df.count().sum()
print("Density of transaction matrix =", density)

#Total number of items purchased
purchasedItemCount = df[df==1].count().sum()
print("Total number of items purchased =", purchasedItemCount)

#Average transaction width
averageTransactionWidth = df[df==1].count(axis=1).mean()
print("Average number of items per transaction =", averageTransactionWidth)

#Max transaction width
maxTransactionWidth = df[df==1].count(axis=1).max()
print("Maximum number of items in a transaction =", maxTransactionWidth)

#Min transaction width
minTransactionWidth = df[df==1].count(axis=1).min()
print("Minimum number of items in a transaction =", minTransactionWidth)

#20 most popular items
df.sum().to_frame('Frequency').sort_values('Frequency',ascending=False)[:20].plot(kind='bar',
                                                                                  figsize=(12,8), 
                                                                                  title="Most Frequent Items")
plt.savefig('MostFrequentItems.pdf', dpi=400)
print("Visualization of 20 most frequently purchased items saved as 'MostFrequentItems.pdf' in current directory.")

#Generate minsup and minconf
minsup = 0.01
minconf = 0.25

#minsup = 0.01 means the item shows up roughly in ~100 transactions
#0.01 is the lowest confidence that produces association rules with antecedent itemsets greater than length 1
#0.25 confidence is a good threshold for identifying meaninful association rather than chance

#Get frequent itemsets
frequent_itemsets = apriori(df, min_support=minsup, use_colnames=True)

frequent_itemsets

#Create single itemsets list
frequent_itemsets["itemset_len"] = frequent_itemsets["itemsets"].apply(lambda x: len(x))
oneItemFrequentItemsets = frequent_itemsets[(frequent_itemsets['itemset_len'] > 1)]

#Create association rules from frequent itemsets
ar = association_rules(frequent_itemsets, min_threshold=minconf)
ar

#From https://www.srose.biz/research-analysis/market-basket-analysis-in-python/

## Apply the Apriori algorithm with a support value of 0.0095
frequent_itemsets = apriori(df, min_support = 0.0095, 
                            use_colnames = True)

# Generate association rules without performing additional pruning
rules = association_rules(frequent_itemsets, metric='support', 
                          min_threshold = 0.0)

# Generate scatterplot using support and confidence
plt.figure(figsize=(10,6))
sns.scatterplot(x = "support", y = "confidence", data = rules)
plt.margins(0.01,0.01)
plt.title = "Confidence of Rule on Support"
plt.savefig('ConfidenceOnSupport.pdf', dpi=400)
print("Visualization of Confidence on Support saved as 'ConfidenceOnSupport.pdf' in current directory.")

#Create another list with the length of the itemset
ar["antecedent_len"] = ar["antecedents"].apply(lambda x: len(x))

#Show only itemsets greater than length 1
results = ar[(ar['antecedent_len'] > 1)]

#sort by confidence
filteredResults = results.sort_values(by="confidence", axis=0, ascending=False)
filteredResults.to_csv('rules.csv')
print("Result: Top Rules Sorted By Confidence saved as 'rules.csv' in current directory.")

filteredResults
