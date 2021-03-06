import math
import numpy as np
from problem2 import DT,Node
#-------------------------------------------------------------------------
'''
    Problem 5: Boosting (on continous attributes). 
               We will implement AdaBoost algorithm in this problem.
    You could test the correctness of your code by typing `nosetests -v test5.py` in the terminal.
'''

#-----------------------------------------------
class DS(DT):
    '''
        Decision Stump (with contineous attributes) for Boosting.
        Decision Stump is also called 1-level decision tree.
        Different from other decision trees, a decision stump can have at most one level of child nodes.
        In order to be used by boosting, here we assume that the data instances are weighted.
    '''
    #--------------------------
    @staticmethod
    def entropy(Y, D):
        '''
            Compute the entropy of the weighted instances.
            Input:
                Y: a list of labels of the instances, a numpy array of int/float/string values.
                D: the weights of instances, a numpy float vector of length n
            Output:
                e: the entropy of the weighted samples, a float scalar
            Hint: you could use np.unique(). 
        '''
        #########################################
        ## INSERT YOUR CODE HERE

        list_y = set([Y[i] for i in range(len(Y))])
        e = 0
        for y_value in list_y:
            sub_y = D[Y == y_value]
            prob = np.sum(sub_y)
            if prob != 0:
                e += -prob * np.log2(prob)




        #########################################
        return e 
            
    #--------------------------
    @staticmethod
    def conditional_entropy(Y,X,D):
        '''
            Compute the conditional entropy of y given x on weighted instances
            Input:
                Y: a list of values, a numpy array of int/float/string values.
                X: a list of values, a numpy array of int/float/string values.
                D: the weights of instances, a numpy float vector of length n
            Output:
                ce: the weighted conditional entropy of y given x, a float scalar
        '''
        #########################################
        ## INSERT YOUR CODE HERE

        x_list = set([X[i] for i in range(X.shape[0])])
        ce = 0.0

        for x_value in x_list:
            sub_y = Y[X == x_value]
            sub_d = D[X == x_value]
            dSum = np.sum(sub_d)
            if dSum != 0:
                sub_d_in = sub_d / dSum
            else:
                sub_d_in = sub_d
            temp_ce = DS.entropy(sub_y, sub_d_in)
            ce += dSum * temp_ce



    
        #########################################
        return ce 

    #--------------------------
    @staticmethod
    def information_gain(Y,X,D):
        '''
            Compute the information gain of y after spliting over attribute x
            Input:
                X: a list of values, a numpy array of int/float/string values.
                Y: a list of values, a numpy array of int/float/string values.
                D: the weights of instances, a numpy float vector of length n
            Output:
                g: the weighted information gain of y after spliting over x, a float scalar
        '''
        #########################################
        ## INSERT YOUR CODE HERE




        g = DS.entropy(Y, D) - DS.conditional_entropy(Y, X, D)
        #########################################
        return g

    #--------------------------
    @staticmethod
    def best_threshold(X,Y,D):
        '''
            Find the best threshold among all possible cutting points in the continous attribute of X. The data instances are weighted. 
            Input:
                X: a list of values, a numpy array of int/float values.
                Y: a list of values, a numpy array of int/float/string values.
                D: the weights of instances, a numpy float vector of length n
            Output:
            Output:
                th: the best threhold, a float scalar. 
                g: the weighted information gain by using the best threhold, a float scalar. 
        '''
        #########################################
        ## INSERT YOUR CODE HERE

        th = -float('inf')
        g = -1.
        ent = DS.entropy(Y, D)
        cp = DT.cutting_points(X, Y)
        # print(cp)
        # conditional entropy
        for i_cp in cp:
            if cp.size == 0: break
            X_1 = np.empty(shape=[1, 0])
            X_2 = np.empty(shape=[1, 0])
            Y_1 = np.empty(shape=[1, 0])
            Y_2 = np.empty(shape=[1, 0])
            D_1 = np.empty(shape=[1, 0])
            D_2 = np.empty(shape=[1, 0])
            for j in range(X.shape[0]):
                if X[j] <= i_cp:
                    X_1 = np.append(X_1, 1)
                    Y_1 = np.append(Y_1, Y[j])
                    D_1 = np.append(D_1, D[j])
                else:
                    X_2 = np.append(Y_2, 1)
                    Y_2 = np.append(Y_2, Y[j])
                    D_2 = np.append(D_2, D[j])

            d_sub_1_sum = np.sum(D_1)
            d_sub_2_sum = np.sum(D_2)
            if d_sub_1_sum != 0:
                d_sub_1_in = D_1 / d_sub_1_sum
            else:
                d_sub_1_in = D_1

            if d_sub_2_sum != 0:
                d_sub_2_in = D_2 / d_sub_2_sum
            else:
                d_sub_2_in = D_2
            temp_ent_1 = DS.entropy(Y_1, d_sub_1_in)
            temp_ent_2 = DS.entropy(Y_2, d_sub_2_in)

            cent = d_sub_1_sum * temp_ent_1 + d_sub_2_sum * temp_ent_2
            g_new = ent - cent

            if g_new > g:
                g = g_new
                th = i_cp

        #########################################
        return th,g 
     
    #--------------------------
    def best_attribute(self,X,Y,D):
        '''
            Find the best attribute to split the node. The attributes have continous values (int/float). The data instances are weighted.
            Here we use information gain to evaluate the attributes. 
            If there is a tie in the best attributes, select the one with the smallest index.
            Input:
                X: the feature matrix, a numpy matrix of shape p by n. 
                   Each element can be int/float/string.
                   Here n is the number data instances in the node, p is the number of attributes.
                Y: the class labels, a numpy array of length n. Each element can be int/float/string.
                D: the weights of instances, a numpy float vector of length n
            Output:
                i: the index of the attribute to split, an integer scalar
                th: the threshold of the attribute to split, a float scalar
        '''
        #########################################
        ## INSERT YOUR CODE HERE


        th = -float('inf')
        g = -1.
        for index in range(X.shape[0]):
            thNext, g_new = self.best_threshold(X[index, :], Y, D)
            if g_new >= g:
                g = g_new
                th = thNext
                i = index




                #########################################
        return i, th
             
    #--------------------------
    @staticmethod
    def most_common(Y,D):
        '''
            Get the most-common label from the list Y. The instances are weighted.
            Input:
                Y: the class labels, a numpy array of length n.
                   Each element can be int/float/string.
                   Here n is the number data instances in the node.
                D: the weights of instances, a numpy float vector of length n
            Output:
                y: the most common label, a scalar, can be int/float/string.
        '''
        #########################################
        ## INSERT YOUR CODE HERE

        uniqueY = np.unique(Y)
        y_dict = dict()

        for y in uniqueY:
            indices = np.argwhere(Y == y).flatten()
            y_dict[y] = sum(D[indices])

        y = max(y_dict, key=y_dict.get)


        #########################################
        return y
 

    #--------------------------
    def build_tree(self, X,Y,D):
        '''
            build decision stump by overwritting the build_tree function in DT class.
            Instead of building tree nodes recursively in DT, here we only build at most one level of children nodes.
            Input:
                X: the feature matrix, a numpy matrix of shape p by n. 
                   Each element can be int/float/string.
                   Here n is the number data instances in the node, p is the number of attributes.
                Y: the class labels, a numpy array of length n. Each element can be int/float/string.
                D: the weights of instances, a numpy float vector of length n
            Return:
                t: the root node of the decision stump. 
        '''
        #########################################
        ## INSERT YOUR CODE HERE
    
        # if Condition 1 or 2 holds, stop splitting 
        t = Node(X, Y)
        t.isleaf = False
        t.p = DS.most_common(Y, D)
        t.X = X
        t.Y = Y
        # if Condition 1 or 2 holds, stop splitting
        if DT.stop1(Y) or DT.stop2(X):
            t.isleaf = True
            return t

        # find the best attribute to split
        t.i, t.th = self.best_attribute(X, Y, D)

        # configure each child node
        t.C1, t.C2 = DT.split(t.X, t.Y, t.i, t.th)
        t.C1.isleaf = True
        t.C2.isleaf = True

        lessIndex = np.argwhere(X[t.i] < t.th).flatten()
        greaterIndex = np.argwhere(X[t.i] >= t.th).flatten()
        t.C1.p = DS.most_common(t.C1.Y, D[lessIndex])
        t.C2.p = DS.most_common(t.C2.Y, D[greaterIndex])


        #########################################
        return t
    
 

#-----------------------------------------------
class AB(DS):
    '''
        AdaBoost algorithm (with contineous attributes).
    '''

    #--------------------------
    @staticmethod
    def weighted_error_rate(Y,Y_,D):
        '''
            Compute the weighted error rate of a decision on a dataset. 
            Input:
                Y: the class labels, a numpy array of length n. Each element can be int/float/string.
                Y_: the predicted class labels, a numpy array of length n. Each element can be int/float/string.
                D: the weights of instances, a numpy float vector of length n
            Output:
                e: the weighted error rate of the decision stump
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        e = np.sum(D[Y != Y_])

        #########################################
        return e

    #--------------------------
    @staticmethod
    def compute_alpha(e):
        '''
            Compute the weight a decision stump based upon weighted error rate.
            Input:
                e: the weighted error rate of a decision stump
            Output:
                a: (alpha) the weight of the decision stump, a float scalar.
        '''
        #########################################
        ## INSERT YOUR CODE HERE

        if e == 0.:
            a = 500
        elif e == 1.:
            a = -500
        else:
            a = 0.5 * np.log((1 - e) / e)



        #########################################
        return a

    #--------------------------
    @staticmethod
    def update_D(D,a,Y,Y_):
        '''
            update the weight the data instances 
            Input:
                D: the current weights of instances, a numpy float vector of length n
                a: (alpha) the weight of the decision stump, a float scalar.
                Y: the class labels, a numpy array of length n. Each element can be int/float/string.
                Y_: the predicted class labels by the decision stump, a numpy array of length n. Each element can be int/float/string.
            Output:
                D: the new weights of instances, a numpy float vector of length n
        '''
        #########################################
        ## INSERT YOUR CODE HERE

        for i in range(len(Y)):
            if Y[i] == Y_[i]:
                D[i] = D[i] * np.exp(-a)
            else:
                D[i] = D[i] * np.exp(a)

        if sum(D) != 1 and sum(D) != 0:
            D *= 1 / sum(D)




        #########################################
        return D

    #--------------------------
    @staticmethod
    def step(X,Y,D):
        '''
            Compute one step of Boosting.  
            Input:
                X: the feature matrix, a numpy matrix of shape p by n. 
                   Each element can be int/float/string.
                   Here n is the number data instances in the node, p is the number of attributes.
                Y: the class labels, a numpy array of length n. Each element can be int/float/string.
                D: the current weights of instances, a numpy float vector of length n
            Output:
                t:  the root node of a decision stump trained in this step
                a: (alpha) the weight of the decision stump, a float scalar.
                D: the new weights of instances, a numpy float vector of length n
        '''
        #########################################
        ## INSERT YOUR CODE HERE

        ds = DS()
        t = ds.build_tree(X, Y, D)
        Y_ = DS.predict(t, X)
        e = AB.weighted_error_rate(Y, Y_, D)
        a = AB.compute_alpha(e)
        D = AB.update_D(D, a, Y, Y_)


        #########################################
        return t,a,D

    
    #--------------------------
    @staticmethod
    def inference(x,T,A):
        '''
            Given a bagging ensemble of decision trees and one data instance, infer the label of the instance. 
            Input:
                x: the attribute vector of a data instance, a numpy vectr of shape p.
                   Each attribute value can be int/float
                T:  the root nodes of decision stumps, a list of length n_tree. 
                A: the weights of the decision stumps, a numpy float vector of length n_tree.
            Output:
                y: the class label, a scalar of int/float/string.
        '''
        #########################################


        y_list = list()
        for t in T:
            while (not t.isleaf):
                if x[t.i] <= t.th:
                    t = t.C1
                else:
                    t = t.C2
            y_list.append(t.p)

        list_y = list(set([y_list[i] for i in range(len(y_list))]))
        vote_obj = list_y[0]
        vote_num = 0
        for y_value in list_y:
            vote_temp = 0
            for idx in range(len(A)):
                if y_list[idx] == y_value:
                    vote_temp += A[idx]

            if vote_temp >= vote_num:
                vote_num = vote_temp
                vote_obj = y_value
        y = vote_obj

    
        #########################################
        return y
 

    #--------------------------
    @staticmethod
    def predict(X,T,A):
        '''
            Given an AdaBoost and a dataset, predict the labels on the dataset. 
            Input:
                X: the feature matrix, a numpy matrix of shape p by n. 
                   Each element can be int/float/string.
                   Here n is the number data instances in the node, p is the number of attributes.
                T:  the root nodes of decision stumps, a list of length n_tree. 
                A: the weights of the decision stumps, a numpy float vector of length n_tree.
            Output:
                Y: the class labels, a numpy array of length n. Each element can be int/float/string.
        '''
        #########################################
        ## INSERT YOUR CODE HERE


        Y = list()
        for i in range(X.shape[1]):
            Y.append(AB.inference(X[:, i], T, A))
        Y = np.asarray(Y)

 
        #########################################
        return Y 
 

    #--------------------------
    @staticmethod
    def train(X,Y,n_tree=10):
        '''
            train adaboost.
            Input:
                X: the feature matrix, a numpy matrix of shape p by n. 
                   Each element can be int/float/string.
                   Here n is the number data instances in the node, p is the number of attributes.
                Y: the class labels, a numpy array of length n. Each element can be int/float/string.
                n_tree: the number of trees in the ensemble, an integer scalar
            Output:
                T:  the root nodes of decision stumps, a list of length n_tree. 
                A: the weights of the decision stumps, a numpy float vector of length n_tree.
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        A = np.zeros(n_tree)
        D = np.ones(len(Y)) / len(Y)
        # iteratively build decision stumps

        T = list()
        for i in range(n_tree):
            t, a, D = AB.step(X, Y, D)
            T.append(t)
            A[i] = a

        # initialize weight as 1/n

        # iteratively build decision stumps

        #########################################
        return T, A
   



 
