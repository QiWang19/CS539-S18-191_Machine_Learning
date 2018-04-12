import math
import numpy as np
from collections import Counter
# Note: please don't add any new package, you should solve this problem using only the packages above.
#-------------------------------------------------------------------------
'''
    Problem 1: Naive Bayes Classifier (with discrete attributes)
    In this problem, you will implement the naive Bayes classification method. 
    In the data1.csv file, we have a collection of email spam detection data. 
    The class label indicate whether or not an email is spam (1: spam; 0: not spam).
    Each email has many features where each feature represents whether or not a certain word has appeared in the email.
    You could test the correctness of your code by typing `nosetests -v test1.py` in the terminal.
    Note: please don't use any existing package for classification problems, implement your own version.
'''

#-----------------------------------------------
def prob_smooth(X,c=2,k=1):
    '''
        Estimate the probability distribution of a random variable with Laplace smoothing.
        Input:
            X: the observed values of training samples, an integer numpy vector of length n. 
                Here n is the number of training instances. Each X[i] = 0,1, ..., or c-1. 
            c: the number of possible values of the variable, an integer scalar.
            k: the parameter of Laplace smoothing, an integer, denoting the number of imagined instances observed for each possible value. 
        Output:
            P: the estimated probability distribution, a numpy vector of length c.
                Each P[i] is the estimated probability of the i-th value.
    '''
    #########################################
    ## INSERT YOUR CODE HERE
    #P = []
    parray = []
    P = np.zeros(c)

    hashmap = dict()
    for i in range(0, len(X) ):
        if hashmap.has_key(X[i]) == False:
            hashmap[X[i]] = 0
        hashmap[X[i]] = hashmap[X[i]] + 1

    i = 0
    for (key, val) in hashmap.items():
       P[i] = val
       parray.append(val)
       i = i + 1
    parray = np.array(P)

    P = (parray + k) / (float(len(X)) + c * k)

    #########################################
    return P
    


#--------------------------
def class_prior(Y):
    '''
        Estimate the prior probability of Class labels: P(Class=y).
        Here we assume this is binary classification problem.
        Input:
            Y : the labels of training instances, an integer numpy vector of length n. 
                Here n is the number of training instances. Each Y[i] = 0 or 1. 
        Output:
            PY: the prior probability of each class, a numpy vector of length c.
    '''
    #########################################
    ## INSERT YOUR CODE HERE
    arr = [0] * 2
    for ele in Y:
        if ele != 0:
            arr[1] += 1
        else:
            arr[0] += 1
    nparr = np.array(arr)
    PY = nparr/float(len(Y))
    #########################################
    return PY


#--------------------------
def conditional_prob(X,Y,k=1):
    '''
        Estimate the conditional probability of P(X=x|Class=y) for each value of attribute X given each class.
        Input:
            X : the values of one attribute for training instances, an integer numpy vector of length n. 
                n is the number of training instances. 
                Here we assume X is a binary variable, with 0 or 1 values. 
            Y : the labels of training instances, an integer numpy vector of length n. 
                Each Y[i] = 0,1
            k: the parameter of Laplace smoothing, an integer, denoting the number of imagined instances observed for each possible value of X given each value of Y.
        Output:
            PX_Y: the probability of P(X|Class), a numpy array (matrix) of shape 2 by 2.
                  PX_Y[i,j] represents the probability of X=j given the class label is i.
    '''
    #########################################
    ## INSERT YOUR CODE HERE

    arr = [[0 for col in range(2)] for row in range(2)]
    for idx in range(0, len(Y)):
        arr[Y[idx]][X[idx]] += 1
    nparr = np.array(arr)
    PX_Y = (nparr + k) / (float(len(Y)) / 2 + 2 * k)
    #########################################
    return PX_Y


#--------------------------
def train(X,Y,k=1):
    '''
        Training the model parameters on a training dataset.
        Input:
            X : the values of attributes for training instances, an integer numpy matrix of shape p by n. 
                p is the number of attributes. 
                n is the number of training instances. 
                Here we assume X is binary-valued, with 0 or 1 elements. 
            Y : the labels of training instances, an integer numpy vector of length n. 
                Each Y[i] = 0,1
            k: the parameter of Laplace smoothing, an integer, denoting the number of imagined instances observed for each possible value. 
        Output:
            PX_Y: the estimated probability of P(X|Class), a numpy array of shape p by 2 by 2.
                  PX_Y[i,j,k] represents the probability of the i-th attribute to have value k given the class label is j.
            PY: the estimated prior probability distribution of the class labels, a numpy vector of length 2.
                Each PY[i] is the estimated probability of the i-th class.
    '''
    #########################################
    ## INSERT YOUR CODE HERE
    p, n = X.shape
    PX_Y = np.zeros((p, 2, 2))
    for i in range(0, p):
        PX_Y[i, :, :] = conditional_prob(X[i, :], Y, k)
    PY = class_prior(Y)


    #########################################
    return PX_Y, PY


#--------------------------
def inference(X,PY, PX_Y):
    '''
        Given a trained model, predict the label of one test instance in the test dataset.
        Input:
            X : the values of attributes for one test instance, an integer numpy vector of length p. 
                p is the number of attributes. 
                Here we assume X is binary-valued, with 0 or 1 elements. 
            PX_Y: the estimated probability of P(X|Class), a numpy array of shape p by 2 by 2.
                  PX_Y[i,j,k] represents the probability of the i-th attribute to have value k given the class label is j.
            PY: the estimated prior probability distribution of the class labels, a numpy vector of length 2.
                Each PY[i] is the estimated probability of the i-th class.
        Output:
            Y: the predicted class label, an integer scalar or value 0 or 1.
            P: the probability P(class | X), a float array of length 2.
                P[i] is the probability of the instance X in the i-th class.
    '''
    #########################################
    ## INSERT YOUR CODE HERE

    i = 0
    temp = PY
    while i < X.shape[0] :
        temp = temp * PX_Y[i, :, X[i]]
        i+=1

    P = temp / sum(temp)
    Y = np.argmax(temp)
    #########################################
    return Y, P


#--------------------------
def predict(X,PY, PX_Y):
    '''
        Given a trained model, predict the labels of test instances in the test dataset.
        Input:
            X : the values of attributes for test instances, an integer numpy matrix of shape p by n. 
                p is the number of attributes. 
                n is the number of test instances. 
                Here we assume X is binary-valued, with 0 or 1 elements. 
            PX_Y: the estimated probability of P(X|Class), a numpy array of shape p by 2 by 2.
                  PX_Y[i,j,k] represents the probability of the i-th attribute to have value k given the class label is j.
            PY: the estimated prior probability distribution of the class labels, a numpy vector of length 2.
                Each PY[i] is the estimated probability of the i-th class.
        Output:
            Y: the predicted class labels, a numpy vector of length n.
               Each Y[i] is the predicted label of the i-th instance.
    '''
    #########################################
    ## INSERT YOUR CODE HERE

    m, n = X.shape
    Y = np.zeros(n)
    for i in range(0, n):

        Y[i] = inference(X[:, i], PY, PX_Y)[0]


    #########################################
    return Y

