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
def dist_prob(Dist):
    return Dist/np.sum(Dist)

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
        prob=dist_prob(distance).tolist()
        #randomly sample another centroid
        index=int(np.random.choice(data_copy.shape[0],1,prob))
        #add the new centroid
        centroid=np.vstack([centroid,data_copy[index]])
        #delete the new centroid from the copy
        data_copy=np.delete(data_copy,index,axis=0)
    return centroid
