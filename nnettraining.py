# -*- coding: utf-8 -*-
"""
Created on Tue May  3 02:59:06 2022

@author: omerw
"""

import numpy as np 
import matplotlib.pyplot as plt
import scipy.io as sio
from mpl_toolkits import mplot3d
from sklearn import svm
from sklearn import metrics
from numpy import genfromtxt
from random import sample
from sklearn import model_selection
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
import torch
import pandas as pd
from collections import Counter
from torch.utils.data import DataLoader, TensorDataset,Dataset
import torch.nn.functional as F
import math
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier 
from neuralnet import *
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn import model_selection
from sklearn.linear_model import SGDClassifier
from sklearn.kernel_approximation import Nystroem

filenames=[#'diabetes_012_health_indicators_BRFSS2015.csv', 
           'diabetes_binary_5050split_health_indicators_BRFSS2015.csv'
           #, 'diabetes_binary_health_indicators_BRFSS2015.csv'
           ]
classes=[3,2,2]

for name in filenames:
    
    data = genfromtxt(name, delimiter=',', skip_header = 1)
    y=data[:,0]
    X=data[:,1:]
   
    scaler=preprocessing.StandardScaler().fit(X)
    X=scaler.transform(X) 
    
    
    current_class=classes[filenames.index(name)]
    name1=name[0:-4]
    accuracies_lr=[]
    accuracies_epochs=[]
    accuracies_bsize=[]
    accuracies_num_neurons=[]
    accuracies_num_layers=[]
    
    num_neurons=[32,64,100, 128]
    num_layers=[1,2,3]
    lrates=[0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125, 0.0039,0.002]
    epochs=[1,5,10,15,20,25]
    bsizes=[50,100,200,400,800,1600,3200,6400]
    
    # best_lr=0.015625
    # best_epochs=10
    # best_bsize=4000
    # best_neurons=100
    # best_layers=2
    
    for i in lrates:
        accuracies_lr.append(validationneuralNet(X,y,current_class, 12, i, 50, 64,2)) 
    print(accuracies_lr)
    best_lr=lrates[accuracies_lr.index(max(accuracies_lr))]
    print('best lr: ', best_lr)
    
    for i in epochs:
        accuracies_epochs.append(validationneuralNet(X,y,current_class, i, best_lr, 50,64,2)) 
    best_epochs=epochs[accuracies_epochs.index(max(accuracies_epochs))]
    print('best epochs: ', best_epochs)
   
    for i in bsizes:
        accuracies_bsize.append(validationneuralNet(X,y,current_class, 12, best_lr, i,64,2)) #use batch size of 100, and epochs =20
   
    best_bsize=bsizes[accuracies_bsize.index(max(accuracies_bsize))]
    print('best batchsize: ', best_bsize)
   
    for i in num_neurons:
        accuracies_num_neurons.append(validationneuralNet(X,y, current_class, 12, best_lr, 500, i,2))
  
    best_neurons=num_neurons[accuracies_num_neurons.index(max(accuracies_num_neurons))]
    print('best neurons: ', best_neurons)
  
    for i in num_layers:
        accuracies_num_layers.append(validationneuralNet(X,y, current_class, 12, best_lr, 500, 64,i))
  
    best_layers=num_layers[accuracies_num_layers.index(max(accuracies_num_layers))]
    print('best layers: ', best_layers)
  
    
    print(accuracies_lr)
    print(accuracies_epochs)
    print(accuracies_bsize)
    print(accuracies_num_neurons)
    print(accuracies_num_layers)
  
    
    plt.xlabel("learning rates")
    plt.ylabel("Classification accuracy.")        
    plt.scatter(lrates, accuracies_lr)
    plt.show()

    plt.xlabel("Epochs")
    plt.ylabel("Classification accuracy.")        
    plt.scatter(epochs, accuracies_epochs)
    plt.show()
    

    plt.xlabel("Batch sizes")
    plt.ylabel("Classification accuracy.")        
    plt.scatter(bsizes, accuracies_bsize)
    plt.show()
    
    plt.xlabel("Number of neurons per hidden layer")
    plt.ylabel("Classification accuracy.")        
    plt.scatter(num_neurons, accuracies_num_neurons)
    plt.show()
    
    plt.xlabel("Number of hidden layers")
    plt.ylabel("Classification accuracy.")        
    plt.scatter(num_layers, accuracies_num_layers)
    plt.show()
    

    # accuracies_lr=[48.12420367190812, 91.72038201738651, 76.52936111872005, 76.81370179537608, 76.13464570190423, 77.05841550143336, 73.39318413784672]
    # accuracies_epochs=[31.87862498231716, 88.13737331132623, 72.52935191463895, 95.36568194337369, 89.39966501146608, 94.92020421855052]
    # accuracies_bsize=[77.89668859172157, 83.83683685346082, 97.5711846632797, 99.67039365254556]
    # accuracies_num_neurons=[80.56333518282005, 92.51136473912534, 93.55664881808593, 80.22661648174801]
    # accuracies_num_layers=[76.504014079843, 77.94824765300935, 70.0565826887642]
    
  
    #accuracies_lr=[26.001555789837216, 15.167491805866723, 31.138884741494376, 68.82707586543873, 69.08845432064577, 70.40712974132731, 69.68585480802189, 70.2114910951516, 70.21826924058128]
    #accuracies_epochs=[70.12838852747309,68.84865449340238,70.0594395554989,69.51832399819412,69.2906151587614, 69.51847389132341]
    #accuracies_bsize=[69.88513869049537, 69.97981621050774, 69.68175477006506, 69.84237226786357, 69.56716570127193, 70.26153864623575, 71.24829809537547, 66.77400719479017]
    #accuracies_num_neurons=[69.91525894581657, 69.89743872433037, 69.96199726958933, 69.57693949497207]
    #accuracies_num_layers=[70.19501266861727, 70.59399365678742, 70.55129108246597]
    
    
    # accuracies_lr=[26.001555789837216, 15.167491805866723, 31.138884741494376, 68.82707586543873, 69.08845432064577, 70.40712974132731, 69.68585480802189, 70.2114910951516, 70.21826924058128]
    # accuracies_epochs=[70.12838852747309,68.84865449340238,70.0594395554989,69.51832399819412,69.2906151587614, 69.51847389132341]
    # accuracies_bsize=[69.88513869049537, 69.97981621050774, 69.68175477006506, 69.84237226786357, 69.56716570127193, 70.26153864623575, 71.24829809537547, 66.77400719479017]
    # accuracies_num_neurons=[69.91525894581657, 69.89743872433037, 69.96199726958933, 69.57693949497207]
    # accuracies_num_layers=[70.19501266861727, 70.59399365678742, 70.55129108246597]
    # accuracies_lr=[26.001555789837216, 15.167491805866723, 31.138884741494376, 68.82707586543873, 69.08845432064577, 70.40712974132731, 69.68585480802189, 70.2114910951516, 70.21826924058128]
    # accuracies_epochs=[70.12838852747309,68.84865449340238,70.0594395554989,69.51832399819412,69.2906151587614, 69.51847389132341]
    # accuracies_bsize=[69.88513869049537, 69.97981621050774, 69.68175477006506, 69.84237226786357, 69.56716570127193, 70.26153864623575, 71.24829809537547, 66.77400719479017]
    # accuracies_num_neurons=[69.91525894581657, 69.89743872433037, 69.96199726958933, 69.57693949497207]
    # accuracies_num_layers=[70.19501266861727, 70.59399365678742, 70.55129108246597]
