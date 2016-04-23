import numpy as np
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
def dist_prob(Dist,l):
    return l*Dist/np.sum(Dist)
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
        prob=dist_prob(distance,l).tolist()
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

    