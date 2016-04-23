#with the initialization of the centroids from the function kmeansplusplus
#plug in the original data(dataSet), initializtions(initial) and the dimension of the data(d)

import numpy as np
import pandas as pd
from pandas import DataFrame
#distance square
def dist_sq(a, b):
    return np.sum((a-b)**2)

def kmeans(dataSet, initial, d):
    centroids=initial
    k=centroids.shape[0]
    #Initialize book keeping vars.
    iterations = 0
    oldCentroids = np.zeros(initial.shape)
    
    # Run the main k-means algorithm
    while not shouldStop(oldCentroids, centroids, iterations):
        # Save old centroids for convergence test. Book keeping.
        oldCentroids = centroids
        iterations += 1
        
        # Assign labels to each datapoint based on centroids
        l= getLabels(dataSet, centroids)
        
        # Assign centroids based on datapoint labels
        centroids = getCentroids(dataSet, l, k, d)
        
    # We can get the labels too by calling getLabels(dataSet, centroids)
    return centroids, np.array(l)

# Function: Should Stop
# -------------
# Returns True or False if k-means is done. K-means terminates either
# because it has run a maximum number of iterations OR the centroids
# stop changing.
def shouldStop(oldCentroids, centroids, iterations):
    if iterations > 50: return True
    return oldCentroids.all == centroids.all

# Function: Get Labels
# -------------
# Returns a label for each piece of data in the dataset. 
def getLabels(dataSet, centroids):
    # For each element in the dataset, chose the closest centroid. 
    # Make that centroid the element's label.
    l=[]
    for i in range(dataSet.shape[0]):
        #arg min as the label
        l.append(np.argmin(list(dist_sq(dataSet[i],c) for c in centroids)))
    return l

# Function: Get Centroids
# -------------
# Returns k random centroids, each of dimension n.
def getCentroids(dataSet, labels, k, d):
    # Each centroid is the arithmetic mean of the points that
    # have that centroid's label. Important: If a centroid is empty (no points have
    # that centroid's label) you should randomly re-initialize it.
    data_new = DataFrame(dataSet.copy())
    data_new['Labels'] = labels
    data_new = np.array(data_new.groupby(['Labels']).mean().iloc[:,:2])
    # if a centroid is empty, reinitialize it 
    if len(np.unique(labels))<k:
        diff=k-len(np.unique(labels))
        data_new=np.vstack([data_new,np.random.random([diff,d])])    
    return data_new
