# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 17:06:27 2022

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
import math
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
import torch
from sklearn import preprocessing
import pandas as pd
from collections import Counter
from torch.utils.data import DataLoader, TensorDataset,Dataset
import torch.nn.functional as F
import math
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn import model_selection
from torch.utils.data.dataset import Subset

class DataSet (Dataset):
    def __init__(self, X, transform=None):   
        self.x_data=torch.from_numpy(X).float()
        self.n_samples = self.x_data.shape[0]
        self.transform=transform
    def __getitem__(self, index):
        sample= self.x_data[index] 
        if self.transform:
            sample=self.transform(sample)
        return sample
    def __len__(self):
        return self.n_samples


class Feedforward(torch.nn.Module):
    def __init__(self, listneurons):
        super(Feedforward, self).__init__()
        self.input_size=listneurons[0]
        self.hidden_layers=len(listneurons)-2
        layers=[]
        for i in range(0, len(listneurons)-1):
           layers.append(torch.nn.Linear(listneurons[i], listneurons[i+1]))
           if i==len(listneurons)-2:
               break 
           layers.append(torch.nn.ReLU())
           
        self.module_list=torch.nn.ModuleList(layers)
        #self.relu=torch.nn.ReLU()
        #self.softmax=torch.nn.Softmax(dim=-1)
    
    def forward(self,x):
        output=x
        for layer in self.module_list:
            output=layer(output)
       # output=self.sotmax(intermediate)
        return output

def evaluateModel(data_loader, best_model):
    with torch.no_grad():
        n_correct=0
        n_samples=0
        for y in data_loader:
            #x=y[:, 0:21]
            labels=y[:,-1]
            outputs=best_model(y[:, 0:21])
            _, predictions=torch.max(outputs,1)
            n_samples+=labels.shape[0]
            n_correct+=(predictions==labels).sum().item()
        acc=100.0*n_correct/n_samples
        return acc

def trainandtestNeuralNet(X_train, X_test,y_train, y_test, classes, no_of_epochs, lr, batchSize, n_per_layer, num_layers):
  
    input_size=21
    num_epochs=no_of_epochs
    learning_rate=lr
    batchsize=batchSize 
    X_train1, X_validation1, y_train1, y_validation1=model_selection.train_test_split(X_train,y_train, test_size=0.1)
    X_train1=np.concatenate((X_train1,y_train1[:,None]),axis=1)
    X_test1=np.concatenate((X_test,y_test[:,None]),axis=1)
    X_validation1=np.concatenate((X_test,y_test[:,None]), axis=1)
    best_overall_accuracy=-float('inf')
    
    trainingset=DataSet(X_train1, transform=None)
    testset=DataSet(X_test1, transform=None)
    validationset=DataSet(X_validation1, transform=None)
    
    train_loader= DataLoader(trainingset, batch_size=batchsize, shuffle=True)
    test_loader = DataLoader(testset, batch_size=batchsize, shuffle=False)
    validation_loader=DataLoader(validationset, batch_size=batchSize, shuffle=False)
    
    num_neurons=[21]
    for l in range(0, num_layers):
        num_neurons.append(n_per_layer)
    num_neurons.append(classes)
    
    best_test_accuracy=-float('inf')
    
    for epoch in range(num_epochs):
        model=Feedforward(num_neurons)
        criterion=torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
        for y in train_loader:
            outputs=model(y[:, 0:21])
            loss=criterion(outputs,y[:,-1].long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        acc=evaluateModel(validation_loader, model)
        if acc>best_test_accuracy:
            best_test_accuracy=acc
            #print(f'Best accuracy in so far in epoch {epoch} is {acc}')
            torch.save(model.state_dict(), 'best_model.pth')

    #Testing and evaluation
    best_model=Feedforward(num_neurons)
    best_model.load_state_dict(torch.load('best_model.pth'))
    best_model.eval()
    
    return evaluateModel(test_loader, best_model)



def validationneuralNet (X,y, classes, no_of_epochs, lr, batchSize, n_per_layer, num_layers):
    
    input_size=21
    num_epochs=no_of_epochs
    learning_rate=lr
    batchsize=batchSize 
    kf=KFold(n_splits=10, random_state=None, shuffle=False)
    X=np.concatenate((X,y[:,None]),axis=1)
    sum_avg_accuracies=0
    
    for _fold, (train_index, test_index) in enumerate(kf.split(X)):
     
        trainingset=Subset(torch.from_numpy(X).float(),train_index)
        validationset=Subset(torch.from_numpy(X).float(),test_index)
        train_loader= DataLoader(trainingset, batch_size=batchsize, shuffle=True)
        validation_loader = DataLoader(validationset, batch_size=batchsize, shuffle=False)
    
        num_neurons=[21]
        for l in range(0, num_layers):
            num_neurons.append(n_per_layer)
        num_neurons.append(classes)
        best_validaccuracy=-float('inf')
        sum_v_over_epochs=0
        for epoch in range(num_epochs):
            model=Feedforward(num_neurons)
            criterion=torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
            for y in train_loader:
                outputs=model(y[:, 0:21])
                loss=criterion(outputs,y[:,-1].long())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
            acc=evaluateModel(validation_loader, model)
            sum_v_over_epochs+=acc
        
            if acc>best_validaccuracy:
                best_validaccuracy=acc
                
        sum_v_over_epochs=sum_v_over_epochs/num_epochs
        sum_avg_accuracies+=sum_v_over_epochs
        
    print("done")
    return sum_avg_accuracies/10

