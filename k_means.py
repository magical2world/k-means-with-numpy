#-*-coding: utf-8 -*-
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

class kmeans():
    def __init__(self,x,num_classification):
        self.x=np.array(x)
        self.num_classification=num_classification
    def random_inital(self):#randomly select the initial value
        return random.sample(self.x,self.num_classification)
    def distance(self,x,y):
        return np.mean(np.square(x-y))#compute L2 distance
    def compute_mean(self,x):
        return np.mean(x,0)
    def classification(self,num,i):
        return np.where(num==i)#return samples belong to i
    def fit(self,max_train_step=10000):
        average = np.array(self.random_inital())
        new_average=np.zeros(average.shape)
        num=np.zeros(self.x.shape[0])
        while(1):
            max_train_step=max_train_step-1
            for i in xrange(self.x.shape[0]):
                distance=[]
                for j in xrange(len(average)):
                    distance.append(self.distance(self.x[i],average[j]))
                    num[i]=np.array(np.where(distance==np.min(distance)))[0][0]
            for j in xrange(len(average)):
                classification=np.array(self.classification(num,j))
                new_average[j]=np.mean(self.x[classification])#compute new average
            if np.sum(average-new_average)==0:
                break#if the old_average=new_average,stop iteration
            average = new_average#update average
            if max_train_step==0:
                break
        return num,average

def main():
    x,y=make_blobs(n_samples=1500,random_state=170)

    cluster=kmeans(x,3)
    num,average=cluster.fit()
    print num
    plt.scatter(x[:,0],x[:,1],c=num)
    plt.show()

if __name__=="__main__":
    main()









