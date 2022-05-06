# -*- coding: utf-8 -*-
"""
Created on Sat Apr 30 15:28:57 2022

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

def logisticregression(X,y):
    kfold=model_selection.KFold(n_splits=10)
    lmodel=linear_model.LogisticRegression(multi_class='ovr', solver='lbfgs')
    #lmodel.fit(X_train, y_train)
    results_kfold=model_selection.cross_val_score(lmodel, X,y,cv=kfold)
    return results_kfold.mean()*100.0

def SVM(X,y, kernel_choice):
    kfold=model_selection.KFold(n_splits=10)

    svc_model=None
    results_kfold=None
    if kernel_choice=='l':    
        svc_model=SGDClassifier(max_iter=5000)
        results_kfold=model_selection.cross_val_score(svc_model, X,y,cv=kfold)

    if kernel_choice=='r':
        fmap=Nystroem (kernel='rbf', n_components=300)
        svc_model=SGDClassifier(max_iter=5000)
        results_kfold=model_selection.cross_val_score(svc_model, fmap.fit_transform(X),y,cv=kfold)
    if kernel_choice=='s':
        fmap=Nystroem (kernel='sigmoid', n_components=300)
        svc_model=SGDClassifier(max_iter=5000)
        results_kfold=model_selection.cross_val_score(svc_model, fmap.fit_transform(X),y,cv=kfold)
    if kernel_choice=='p':
        fmap=Nystroem (kernel='poly', n_components=300, degree=100)
        svc_model=SGDClassifier(max_iter=5000)
        results_kfold=model_selection.cross_val_score(svc_model, X,y,cv=kfold)
    return results_kfold.mean()*100.0

def naivebayes(X,y):
    kfold=model_selection.KFold(n_splits=10)
    nb=GaussianNB()
    #nb.fit(X_train, y_train)
    results_kfold=model_selection.cross_val_score(nb, X,y,cv=kfold)
    return results_kfold.mean()*100.0
    
def decisiontree(X,y):
    kfold=model_selection.KFold(n_splits=10)
    dtree=DecisionTreeClassifier()
    #dtree.fit(X_train, y_train)
    results_kfold=model_selection.cross_val_score(dtree,X,y,cv=kfold)
    #predictions=dtree.predict(X_test)
    return results_kfold.mean()*100.0
  
def traintestMethods(X_train, X_test, y_train, y_test):
    scores=[]
    
    lmodel=linear_model.LogisticRegression(multi_class='ovr', solver='lbfgs')
    lmodel.fit(X_train, y_train)
    score=lmodel.score(X_test,y_test)
    #print('Test set score of logistic regression is: ', score)
    scores.append(score*100)
    
    dtree=DecisionTreeClassifier()
    dtree.fit(X_train, y_train)
    score=dtree.score(X_test,y_test)
    #print('Test set score of decision trees is: ', score)
    scores.append(score*100)
    
    nb=GaussianNB()
    nb.fit(X_train, y_train)
    score=nb.score(X_test,y_test)
    #print('Test set score of naive Bayes (gaussian) is: ', score)
    scores.append(float(score)*100)

    
    svc_model=SGDClassifier(max_iter=5000)
    svc_model.fit(X_train, y_train)
    svc_pred=svc_model.predict(X_test)
    score=format(metrics.accuracy_score(y_test, svc_pred))
    #print("Accuracy score for linear SVM=", score)
    scores.append(float(score)*100)

    fmap=Nystroem (kernel='rbf', n_components=300)
    svc_model=SGDClassifier(max_iter=5000)
    svc_model.fit(fmap.fit_transform(X_train), y_train)
    svc_pred=svc_model.predict(fmap.fit_transform(X_test))
    score=format(metrics.accuracy_score(y_test, svc_pred))
    #print("Accuracy score for RBF kernel SVM=", score)
    scores.append(float(score)*100)

    fmap=Nystroem (kernel='sigmoid', n_components=300)
    svc_model=SGDClassifier(max_iter=5000)
    svc_model.fit(fmap.fit_transform(X_train), y_train)
    svc_pred=svc_model.predict(fmap.fit_transform(X_test))
    score=format(metrics.accuracy_score(y_test, svc_pred))
    #print("Accuracy score for Sigmoid kernel SVM=", score)
    scores.append(float(score)*100)

    fmap=Nystroem (kernel='poly', n_components=300, degree=50)
    svc_model=SGDClassifier(max_iter=5000)
    svc_model.fit(fmap.fit_transform(X_train), y_train)
    svc_pred=svc_model.predict(fmap.fit_transform(X_test))
    score=format(metrics.accuracy_score(y_test, svc_pred))
    #print("Accuracy score for polynomial kernel SVM=", score)
    scores.append(float(score)*100)

    return scores

def crossValidateMethods (X,y):
    all_methods_scores=[0 for i in range(0,8)]
    for i in range (0,5):
        scores=[]
        scores.append(validationneuralNet(X,y,2, 5, 0.015625, 500,100,2))
        x=['DNN', 'LReg', 'DTree', 'NBayes', 'SVM(L)', 'SVM(RBF)', 'SVM(Si)', 'SVM(P)']
        scores.append(logisticregression(X,y))
        scores.append(decisiontree(X,y))
        scores.append(naivebayes(X,y))
        scores.append(SVM(X,y,'l'))
        scores.append(SVM(X,y,'r'))
        scores.append(SVM(X,y,'s'))
        scores.append(SVM(X,y,'p'))
        all_methods_scores=[all_methods_scores[j]+scores[j] for j in range(0,8)]
        
    b=[a/5  for a in all_methods_scores]
    return b

def generateTestData(X,y):
    X_train, X_test, y_train, y_test=model_selection.train_test_split(X,y, test_size=0.1)
    return X_train, X_test, y_train, y_test

filenames=[#'diabetes_012_health_indicators_BRFSS2015.csv', 
           'diabetes_binary_5050split_health_indicators_BRFSS2015.csv'
           #, 'diabetes_binary_health_indicators_BRFSS2015.csv'
           ]
classes=[3,2,2]

#all_methods_scores=[2264.172560113154, 2244.653465346535,1971.4851485148515,2160.8062234794907,2232.729844413013,1506.2659123055164,1524.9929278642148,1474.7100424328146]

for name in filenames:
    
    data = genfromtxt(name, delimiter=',', skip_header = 1)
    y=data[:,0]
    X=data[:,1:]
    scaler=preprocessing.StandardScaler().fit(X)
    X=scaler.transform(X)     
    current_class=classes[filenames.index(name)]
    name1=name[0:-4]
    
    # kfold=model_selection.KFold(n_splits=10)
    # lmodel=linear_model.LogisticRegression(multi_class='ovr',solver='liblinear',penalty='l1', C=0.0001)
    # #lmodel.fit(X_train, y_train)
    # results_kfold=model_selection.cross_val_score(lmodel, X,y,cv=kfold)
    # print(results_kfold.mean()*100.0)
    
    # X_train,X_test,y_train, y_test=generateTestData(X,y)
    # lmodel=linear_model.LogisticRegression(multi_class='ovr', solver='liblinear',penalty='l1', C=0.0001)
    # lmodel.fit(X_train, y_train)
    # print(lmodel.score(X_test,y_test))
    #print('Test set score of logistic regression is: ', score)


    #CROSS VALIDATE METHODS STORE MEAN CLASSIFICATION ACCURACIES OVER 15 RUNS IN A LIST
    print(crossValidateMethods(X, y))
    # #b=[69.84991887002695, 69.37140908278732, 61.9970908700918, 70.37850562939607, 69.05176363199443, 67.22277773964976, 68.66991327954815, 68.87352043896264]
      
    # #TEST THE PERFORMANCE OF VARIOUS METHODS ON THE DATA SET.
    runs=30
    lr=0.015625 
    epochs=15
    batchsize=500
    neurons_per_layer=100
    num_layers=2
    all_methods_scores=[0 for i in range(0,8)]
    
    for k in range(0,runs):
        X_train,X_test,y_train, y_test=generateTestData(X,y)
        print (len(y_test))
        nnet_accuracy=trainandtestNeuralNet(X_train, X_test,y_train, y_test, current_class, epochs, lr, batchsize, neurons_per_layer, num_layers)
        scores=traintestMethods(X_train, X_test, y_train, y_test)
        scores=[nnet_accuracy]+scores
        all_methods_scores=[all_methods_scores[i]+scores[i] for i in range(0,8)]
    b=[a/30  for a in all_methods_scores]
    x=['DNN', 'LReg', 'DTree', 'NBayes', 'SVM(L)', 'SVM(RBF)', 'SVM(Si)', 'SVM(P)']
    plt.xlabel('Classification method')
    plt.ylabel('Mean Classification Accuracy')
    plt.scatter(x,b, marker='x')
    plt.show()    
    