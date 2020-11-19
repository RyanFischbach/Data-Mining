"""
KMeans Implementation From Scratch
Ryan Fischbach
Dr. Khuri
CSC373
10/30/2020

Citations:
Chapter 7
Data Mining
Tan et al.
"""

# Import required libraries
import numpy as np
import random
from scipy.spatial import distance

class KMeans(object):

    # computer euclidean distance
    @staticmethod
    def distance(xi, xq):
        return np.sqrt(np.sum(np.square(xi - xq)))

    # get initial centroids
    @staticmethod
    def getInitialCentroids(x, k):
        # Initial centroids
        centroids = {}

        # Create range and shuffle to ensure randomness/no duplicates
        indexOfRandomSamples = list(range(0, x.shape[0]))
        random.shuffle(indexOfRandomSamples)
        # Loop k times to assign k centroids
        for counter in range(k):
            centroids[counter] = x[indexOfRandomSamples[counter]]

        return centroids

    # perform iterations over samples and compute clusters
    def iterate(self, x, clusters, centroids):
        # loop through samples
        for sampleCounter in range(x.shape[0]):
            lowestDistance = np.inf
            lowestCentroidIndex = -1

            # loop through clusters
            for clusterCounter in range(len(centroids)):
                # compute distance
                current_distance = self.distance(centroids[clusterCounter], x[sampleCounter])
                if current_distance < lowestDistance:
                    lowestDistance = current_distance
                    lowestCentroidIndex = clusterCounter

            clusters[sampleCounter] = lowestCentroidIndex

        return clusters

    # output cluster information
    def clusterInformation(self, clusters, centroids):

        wss = list()
        
        # loop through centroids
        for displayClusters in range(len(centroids)):
            # determine samples in the cluster and total number of samples in each cluster
            samplesInCluster = (np.where(clusters == displayClusters))[0]
            numSamplesInCluster = samplesInCluster.shape[0]
            sumOfSquares = 0

            # compute sum of squares for cluster
            for sampleIndex in range(numSamplesInCluster):
                sumOfSquares += np.square(distance.euclidean(centroids[displayClusters], samplesInCluster[sampleIndex]))

            # output information
            #print("Within Cluster", displayClusters, "Sum of Squares: ", sumOfSquares)
            #print(numSamplesInCluster, "samples in Cluster", displayClusters)
            wss.append(sumOfSquares)
            
        return wss    

    # cluster x based on similarities of attributes
    def cluster(self, x, k):
        # Check for an empty arrays
        x = np.array(x)

        if x.shape == ():
            return -1

        if k < 1:
            return -1

        if k > x.shape[0]:
            return -1

        # randomly assign initial centroids
        centroids = self.getInitialCentroids(x, k)
        clusters = np.empty((x.shape[0], 1))
        lastIterationsClusters = None

        # loop 15 times, stopping criteria #2
        for iterations in range(15):
            clusters = self.iterate(x, clusters, centroids)

            # if clusters don't change, break loop. Stopping criteria #1
            if np.array_equal(clusters, lastIterationsClusters):
                break

            lastIterationsClusters = clusters.copy()

            # recalculate centroid means
            for recalcCentroidCounter in range(len(centroids)):
                centroids[recalcCentroidCounter] = np.nanmean(x[np.where(clusters == recalcCentroidCounter)], axis=0)

        # get cluster information
        wss = self.clusterInformation(clusters, centroids)

        # return cluster assignments
        return clusters.T, wss
