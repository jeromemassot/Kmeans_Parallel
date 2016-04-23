import numpy as np
import pandas as pd
from pandas import DataFrame

#distance square
def dist_sq(a, b):
    return np.sum((a-b)**2)
#minimum distance square for every point to the centroid
def point_sq(data,centroid):
    dist=[]
    for i in range(data.shape[0]):
        dist.append(min(dist_sq(data[i],c) for c in centroid))
    return dist
        
#calculate probability
def dist_prob_plus(Dist):
    return Dist/np.sum(Dist)

#calculate probability
def dist_prob_parallel(Dist,l):
    return l*Dist/np.sum(Dist)


def kmeansplusplus(data, k, d):
    #make a copy of the data
    data_copy=data.copy()
    #step 1: sample a point uniformly at random from x
    index=int(np.random.choice(data_copy.shape[0],1))
    centroid=data_copy[index]
    #once the centroid is determined, delete it from the copy 
    data_copy=np.delete(data_copy,index,axis=0)
    #step 2: while c<k, sample x from X with probability d^2/phi_x(C)
    for number in range(k-1):
        #calculate the square difference for every point in the copy to its nearest center
        distance=point_sq(data_copy,centroid)
        #calculate the probability
        prob=dist_prob_plus(distance).tolist()
        #randomly sample another centroid
        index=int(np.random.choice(data_copy.shape[0],1,prob))
        #add the new centroid
        centroid=np.vstack([centroid,data_copy[index]])
        #delete the new centroid from the copy
        data_copy=np.delete(data_copy,index,axis=0)
    return centroid

import math
def kmeansparallel(data, k, l, d):
    #step 1: sample a point uniformly at random from X
    index=int(np.random.choice(data.shape[0],1))
    centroid=np.array(data[index])
    data_copy=data.copy()
    data_copy=np.delete(data_copy,index,axis=0)
    
    #step 2: calculate the cost and number of iterations(log(cost))
    cost=np.sum(point_sq(data_copy,centroid))
    iteration=math.ceil(np.log(cost))
    
    #step 3
    for number in range(iteration):
        #calculate phi_X(C)
        distance=point_sq(data_copy,centroid)
        #calculate the probability
        prob=dist_prob_parallel(distance,l).tolist()
        for n in range(data_copy.shape[0]):
            #if the probability is greater than the random uniform
            if prob[n]>np.random.uniform():
                #add the point to C
                centroid=np.vstack([centroid,np.array(data_copy[n])])
                #delete that point from the copy
                data_copy=np.delete(data_copy,n,axis=0)
    
    #step 4: assign the weights
    weight_size=centroid.shape[0]
    weight=np.zeros(weight_size)
    for i in range(data_copy.shape[0]):
        index_w=np.argmin(list(dist_sq(data_copy[i],c) for c in centroid))
        weight[index_w]=weight[index_w]+1
    
    #step 5: recluster the weighted points in C into k clusters
    #reinitialize k centroids
    new_centroids=np.zeros([k,d])
    for cluster in range(k):
        #according to the weights from step 4, calculate the probability that a point is sampled from C
        prob_w=list(weight/sum(weight))
        #sample a new centroid
        new_index=np.random.choice(centroid.shape[0],1,prob_w)
        #store the new centroid
        new_centroids[cluster]=centroid[new_index]
        #delete the new centroid from the centroid
        centroid=np.delete(centroid,new_index,axis=0)
        #delete the correponding weight
        weight=np.delete(weight,new_index,axis=0)
    return new_centroids

    
#with the initialization of the centroids from the function kmeansplusplus
#plug in the original data(dataSet), initializtions(initial) and the dimension of the data(d)
def kmeans(dataSet, initial, d):
    centroids=initial
    k=centroids.shape[0]
    # Initialize book keeping vars.
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
    