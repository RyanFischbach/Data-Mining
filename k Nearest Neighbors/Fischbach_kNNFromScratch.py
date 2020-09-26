"""
Naive kNN Implementation
Ryan Fischbach
Dr. Khuri
CSC373
9/25/2020

Citations:
STAT 479: Machine Learning Lecture Notes From Sebastian Raschka Referenced To Code
"""

# Import required libraries
import numpy as np
import pandas as pd
import math


class kNN(object):

    # computer euclidean distance
    @staticmethod
    def distance(xi, xq):
        return np.sqrt(np.sum(np.square(xi - xq)))

    # determine label from nearest neighbors array
    @staticmethod
    def count(neighbors, x_train, y_train):
        labels = list()

        #loop through neighbors and find corresponding label
        for x in neighbors:
            labels.append(y_train[x])

        #determine if 1 or 0 is higher and assign corresponding label
        if labels.count(1) > labels.count(0):
            return 1
        else:
            return 0

    # predict label of X_test via computing nearest neighbors
    def predict(self, x_train, y_train, x_test, k):
        # Check for an empty arrays
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        x_test = np.array(x_test)

        if x_train.shape == ():
            return -1

        if y_train.shape == ():
            return -1

        if x_test.shape == ():
            return -1

        #generate prediction array
        y_pred = np.zeros((x_test.shape[0], 1))

        #loop through test set
        for i in range(x_test.shape[0]):

            #generate a new neighbors dictionary {id:distance}
            neighbors = {}

            #loop through train data for each test point
            for j in range(x_train.shape[0]):
                #compute distance
                current_distance = self.distance(x_test[i], x_train[j])
                #arbitrarily assign the first k items in the train set as neighbors
                if(len(neighbors) < k):
                    neighbors[j] = current_distance
                else:
                    #determine the current max distance in the neighbors array
                    maxv = max(neighbors, key=neighbors.get)
                    #if that max distance is greater than current, replace it
                    if current_distance < neighbors[maxv]:
                        neighbors.pop(maxv)
                        neighbors[j] = current_distance
            #after all train data is looped through, determine a label                
            y_pred[i] = self.count(neighbors, x_train, y_train)

        #return predictions array
        return y_pred
