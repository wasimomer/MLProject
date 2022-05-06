# MLProject
Machine Learning Project for CS 6140 @ Northeastern. The data set diabetes _ binary _ 5050split _ health _ indicators _ BRFSS2015.csv in this repository is obtained from https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset?select=diabetes_binary_health_indicators_BRFSS2015.csv.

# File unified.py
The file unified.py consists of a list of implementations and cross validation methods for the following classifiers: SVM, Naive Bayes, Logistic Regression, and Decision Trees. 

The function crossValidateMethods takes as input a data set X with corresponding labels y. It returns a list of scores corresponding to cross validation accuracies of methods including logistic regression, decision trees, naive bayes, and SVM (with linear, rbf, sigmoid and polynomial kernels). It uses the functions logisticregression, decisiontree, SVM and naivebayes which implement k fold cross validation with k=10 when given as input a data set X with corresponding set of labels y. All these functions return the mean classification accuracy for each function obtained over all k folds. 

The function traintestMethods takes as input a set of training examples, test examples, training labels and test labels. It returns a vector of scores corresponding to accuracy of the classification methods namely Logistic Regression, Decision Trees and SVM (linear, polynomial, sigmoid and rbf). 

When the file is executed, the output list returned by crossValidateMethods is printed. Moreover, the test accuracies for all methods over 30 runs are plotted and returned as output. Depending on the system, the total execution time is likely to be around 30 minutes. 

# File neuralnet.py

This file implements the helper functions and some of the main subroutines we use to train, cross-validate and test neural networks. We implement the DataSet method which uses the Dataset class available in Pytorch. 

The architecture of the neural network is implemented usign the class Feedforward. 

The function evaluateModel takes as input a data loader object and a model, and returns as output the accuracy of the model on the data set that is captured by the data_loader object.

The function trainandtestNeuralNet takes as input a set of training examples, test examples, training labels, test labels, number of classes (2 for binary classfication), number of epochs, learning rate, batch size, neurons per hidden layer and the number of hidden layers. It trains the neural network parametrized by the input arguments and returns as ouput the classfication accuracy of the trained neural network on the test data set.

The function validationneuralNet takes as input a set of training data, with labels, number of epochs, learning rate, batch size, neurons per hidden layer and number of hidden layers. It performs k fold cross valudation with k=10 and returns the mean classification accuracy over the 10 folds. The number of passes that the function makes through each training set of a single fold is given by the input argument number of epochs.

# File nnettraining.py
This is a python script which we used to perform hyper-parameter tuning on our neural network. It takes a significant amount of time to execute depending on the underlying system and generates the plots of classification accuracies against different learning rates, number of epochs, batch sizes, number of neurons per hidden layer and number of hidden layers. 
