"""
Naive NB Implementation
Ryan Fischbach
Dr. Khuri
CSC373
10/9/2020

Citations:
Bayesian Classification 
Nir Friedman, Ron Kohavi
August 29, 1999
Referenced To Implement Code
"""

# Import required libraries
import numpy as np
import pandas as pd

class NaiveBayes(object):

    # determine label from nearest neighbors array
    def train(self, x_train, y_train):
        x_train = np.array(x_train)
        y_train = np.array(y_train)

        priors = {}
        likelihood = {}

        if x_train.shape == ():
            return -1

        if y_train.shape == ():
            return -1

        #compute prior class probabilities
        classes, count = np.unique(y_train, return_counts=True)
        sum = y_train.shape[0]
        for i in range(len(classes)):
            priors[classes[i]] = count[i] / sum

        #combine x and y
        combined_train = np.concatenate((x_train, y_train), axis=1)

        #loop through each column
        for columnIndex in range(0, x_train.T.shape[0]):
            #get that column through index
            column = x_train.T[columnIndex]

            #compute number of unique values of that attribute
            v = len(np.unique(column))
            columnUniques = np.unique(column)

            #loop through targets
            for i in classes:
                #determine how many total samples are in that target in each column
                samplesInClass = np.where(combined_train[:, 1] == i)
                samplesInClassNum = samplesInClass[0].shape[0]

                #loop through unique column values
                for j in range(len(columnUniques)):
                    attributes = np.where(combined_train[:, 0] == columnUniques[j])

                    #find number of occurences of this unique column value in total samples in target
                    crossover = np.intersect1d(attributes, samplesInClass)

                    #turn indexes into strings to hash
                    colString = str(columnIndex)
                    attributeString = columnUniques[j].astype(str)
                    targetString = i.astype(str)
                    combined_string = colString + '|' + attributeString + '|' + targetString
                    #put into dictionary
                    likelihood[combined_string] = ((crossover.shape[0] + 1) / (samplesInClassNum + v))
 
        d = dict(enumerate(classes))
        #Save model as CSVs to prevent loss
        df = pd.DataFrame.from_dict(priors, orient="index")
        df.to_csv("priors.csv")
        df = pd.DataFrame.from_dict(likelihood, orient="index")
        df.to_csv("likelihood.csv")
        df = pd.DataFrame.from_dict(d, orient="index")
        df.to_csv("classes.csv")

    # predict label of X_test via computing nearest neighbors
    def predict(self, x_test):
        
        # Check for an empty arrays
        x_test = np.array(x_test)

        if x_test.shape == ():
            return -1

        #Get model from CSVs
        df = pd.read_csv("priors.csv", index_col=0)
        priors = df.to_dict("split")
        priors = dict(zip(priors["index"], priors["data"]))

        df = pd.read_csv("likelihood.csv", index_col=0)
        likelihood = df.to_dict("split")
        likelihood = dict(zip(likelihood["index"], likelihood["data"]))

        df = pd.read_csv("classes.csv", index_col=0)
        classes = df.to_dict("split")
        classes = dict(zip(classes["index"], classes["data"]))

        #generate prediction array
        y_pred = np.zeros((x_test.shape[0], 1))

        #loop through each sample
        for rowIndex in range(x_test.shape[0]):

            #assign initial probability matrix
            prob = np.ones(((len(priors)), 1))

            #loop through each target class
            for i in range(len(priors)):
                priorsTuple = priors[i]
                priorsVal = priorsTuple[0]
                prob[i] = priorsVal

                #loop through each column
                for columnIndex in range(x_test.shape[1]):
                    stringAtt = str(x_test[rowIndex, columnIndex])
                    stringCol = str(columnIndex)
                    stringClass = str((classes[i])[0])
                    combstring = stringCol + '|' + stringAtt + '|' + stringClass
                    prob[i] = (prob[i] * likelihood[combstring])

            #assign predictions        
            y_pred[rowIndex] = np.argmax(prob)

        #return predictions array
        return y_pred
