import numpy as np
import pandas as pd
from pandas import DataFrame
import math

##calculate the distance square
def dist_sq(a, b, axis = 0):
    '''a,b are numpy arrays with the same shape'''
    return np.sum((a-b)**2,axis)

#minimum distance square for every point to the centroid
def point_sq(data,centroid):
    dist=[min(dist_sq(centroid, d, axis = 1)) for d in data]
    return dist

#calculate probability
def dist_prob_plus(Dist):
    return Dist/np.sum(Dist)

##calculate the cost
def cost(data,centroid):
    '''The function caculates the sum of the distance square of each point in the data set to its closest centroid
    parameters: data and centroids
    output:cost(a number)
    '''
    return np.sum([min(dist_sq(centroid, d, axis = 1)) for d in data])


##calculate the parallel probability
def dist_prob_parallel(data,centroid,l):
    '''The function caculates the sum of the distance square to the closest centroid for each point in the data set
    parameters: data and centroids
    output:probability with length=len(data)
    '''
    phi= cost(data,centroid)
    return np.array([(min(dist_sq(centroid, d, axis = 1)))*l/phi for d in data])


    
#weight probability
def weight_prob(data,centroid):
    closest_center = [np.argmin(dist_sq(centroid, d, axis = 1)) for d in data]
    ## number of points which is closest to each s in c
    number_= np.array([closest_center.count(i) for i in range(len(centroid))]).astype(float)
    ## return normalized weight
    return number_/np.sum(number_)


#step 5: recluster the weighted points in C into k clusters
#reinitialize k centroids
def reassign_centroids(data,centroid,k,l,w):
    new_centroids = data[np.random.choice(range(len(centroid)),size=1,p=w),]
    potential_centroids = centroid
    for i in range(k-1):
        prob = dist_prob_parallel(potential_centroids,new_centroids,l) * w
        new_centroid = data[np.random.choice(range(len(centroid)),size=1,p=prob/np.sum(prob)),]
        new_centroids = np.vstack((new_centroids,new_centroid))
    return new_centroids



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



def kmeansparallel(data, k, l, r):
    #step 1: sample a point uniformly at random from X
    centroid=np.array(data[np.random.choice(range(len(data)),1),])
    
    #step 2: calculate number of iteration
    iteration= np.ceil(np.log(cost(data,centroid))).astype(int)  
    
    #step 3: Get initial Centroids C
    for round in range(r):
        for i in range(iteration):
            centroid_added = data[dist_prob_parallel(data,centroid,l)>np.random.uniform(size = len(data)),]
            centroid = np.vstack((centroid,centroid_added))  
    
    #step 4: calculate the weight probability
    w=weight_prob(data,centroid)
    
    #step 5: recluster the weighted points in C into k clusters
    #reinitialize k centroids
    final_centroids=reassign_centroids(data,centroid,k,l,w)
    
    return final_centroids

    
#with the initialization of the centroids from the function kmeansplusplus
#plug in the original data(dataSet), initializtions(initial) and the dimension of the data(d)
def kmeans(dataSet, initial, k, d):
    centroids=initial
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
    data_new = DataFrame(dataSet.copy())
    data_new['Labels'] = labels
    data_new = np.array(data_new.groupby(['Labels']).mean().iloc[:,:d])
  
    return data_new
    