import math
import numpy as np
from collections import Counter
# Note: please don't add any new package, you should solve this problem using only the packages above.
#-------------------------------------------------------------------------
'''
    Problem 3: Hidden Markov Model (HMM)
    In this problem, you will implement a hidden markov model. 
    You could test the correctness of your code by typing `nosetests -v test3.py` in the terminal.
    Note: please don't use any existing package for HMM, implement your own version.
    Hint: For details of HMM, please read: http://cs.rochester.edu/u/james/CSC248/Lec11.pdf
'''

#-----------------------------------------------
def forward_prob(Ev,I,T,Em):
    '''
        Given a HMM and a sequence of evidence, compute the forward probability of the last hidden state (X_{n-1}) using Forward Algorithm: P(X_{n-1}, e0, e1, ..., e_{n-1}), return the state of Xt with the maximium probability. Here e0 represents the observed evidence at step 0. n is the number of steps in total.
        Input:
            Ev: the observed evidence sequence, an integer numpy vector of length n. 
                Here n is the number of time steps. Each X[i] = 0,1, ..., or p-1. 
                p is the number of possible values of the evidence variable, an integer scalar.
            I : the initial probability distribution of the hidden variable, a float vector of length c. 
                c is the number of possible values of the hidden variable, an integer scalar.
            T : the transition probablity of the hidden variable, a float numpy matrix of shape c by c
                T[i,j] represents the probability of P(Xt = j | Xt-1 = i).
            Em: the emission probability of the evidence variable, a float numpy matrix of shape c by p.
                Em[i,j] represents the probability of P(Et = j | Xt = i)
        Output:
            X: the most likely state of the hidden variable in the last time step, an integer scalar (value = 0, 1, ..., or, c-1)
            a: the forward probabilities at the last time step, a float vector of length c.
                a[i] = P(X_{n-1} = i, e0, e1, ..., e_{n-1})
    '''
    #########################################
    ## INSERT YOUR CODE HERE
    arr = I * Em[:,Ev[0]]
    for i in range(1, len(Ev)):
        temp = arr.reshape(-1,1) * T

        arr = np.sum(temp, axis = 0) * Em[:,Ev[i]]
    X = np.argmax(arr)
    a = arr


    #########################################
    return X, a


#-----------------------------------------------
def backward_prob(Ev,T,Em):
    '''
        Given a HMM and a sequence of evidence, compute the backward probability of the first hidden state (X_0) using Backward Algorithm: P(e1, ..., e_{n-1}| X_0 = i), return the state of X-0 with the maximium probability. Here e0 represents the observed evidence at step 0. n is the number of steps in total.
        Input:
            Ev: the observed evidence sequence, an integer numpy vector of length n. 
                Here n is the number of time steps. Each X[i] = 0,1, ..., or p-1. 
                p is the number of possible values of the evidence variable, an integer scalar.
            T : the transition probablity of the hidden variable, a float numpy matrix of shape c by c
                T[i,j] represents the probability of P(Xt = j | Xt-1 = i).
            Em: the emission probability of the evidence variable, a float numpy matrix of shape c by p.
                Em[i,j] represents the probability of P(Et = j | Xt = i)
        Output:
            X: the most likely state of the hidden variable in the first time step, an integer scalar (value = 0, 1, ..., or, c-1)
            b: the backward probabilities at the first time step, a float vector of length c.
                b[i] = P(e1, e2, ..., e_{n-1} | X_0 = i)
    '''
    #########################################
    ## INSERT YOUR CODE HERE
     
    m, n = T.shape
    arri = np.ones((1, m))
    idx = len(Ev) - 1
    while idx > 0:
        arrm = (T * Em[:,Ev[idx]]) * arri
        arri = np.sum(arrm, axis = 1)
        idx -= 1
    b = arri
    X = np.argmax(arri)

    #########################################
    return X, b


#-----------------------------------------------
def forward_backward_prob(Ev,I,T,Em,i):
    '''
        Given a HMM and a sequence of evidence, compute the forward-backward probability of the i-th hidden state (X_i) using Forward-Backward Algorithm: P(X_i, e0, e1, ..., e_{n-1}), return the state of Xi with the maximium probability. Here e0 represents the observed evidence at step 0. n is the number of steps in total.
        Input:
            Ev: the observed evidence sequence, an integer numpy vector of length n. 
                Here n is the number of time steps. Each X[i] = 0,1, ..., or p-1. 
                p is the number of possible values of the evidence variable, an integer scalar.
            I : the initial probability distribution of the hidden variable, a float vector of length c. 
                c is the number of possible values of the hidden variable, an integer scalar.
            T : the transition probablity of the hidden variable, a float numpy matrix of shape c by c
                T[i,j] represents the probability of P(Xt = j | Xt-1 = i).
            Em: the emission probability of the evidence variable, a float numpy matrix of shape c by p.
                Em[i,j] represents the probability of P(Et = j | Xt = i)
            i : the target time step to etimate probability of hidden state (at step i), an integer scalar of value 0, 1, ..., or n-1
        Output:
            X: the most likely state of the hidden variable at the i-th time step, an integer scalar (value = 0, 1, ..., or, c-1)
            p: the forward-backward probabilities at the i-th time step, a float vector of length c.
                p[k] = P(X_i = k, e0, e1, ..., e_{n-1})
            a: the forward probabilities at the i-th time step, a float vector of length c.
                a[k] = P(X_i = k, e0, e1, ..., e_{i})
            b: the backward probabilities at the i-th time step, a float vector of length c.
                b[k] = P(e_{i+1}, ..., e_{n-1} | X_i = k)
    '''
    #########################################
    ## INSERT YOUR CODE HERE

    aa = forward_prob(Ev[0:i + 1], I, T, Em)[1]
    XX1, bb = backward_prob(Ev[i:], T, Em)
    X = np.argmax(aa * bb)
    a = aa
    b = bb
    p = aa * bb
    #########################################
    return X, p, a, b


#-----------------------------------------------
def most_probable_pass(Ev,I,T,Em):
    '''
        Given a HMM and a sequence of evidence, compute the most probable path of the hidden states using Viterbi Algorithm.
        Input:
            Ev: the observed evidence sequence, an integer numpy vector of length length.
                Here length is the number of time steps. Each X[i] = 0,1, ..., or p-1.
                p is the number of possible values of the evidence variable, an integer scalar.
            I : the initial probability distribution of the hidden variable, a float vector of length col.
                col is the number of possible values of the hidden variable, an integer scalar.
            T : the transition probablity of the hidden variable, a float numpy matrix of shape col by col
                T[i,j] represents the probability of P(Xt = j | Xt-1 = i).
            Em: the emission probability of the evidence variable, a float numpy matrix of shape col by p.
                Em[i,j] represents the probability of P(Et = j | Xt = i)
        Output:
            X: the hidden state trajectory with maximum joint probability, an integer vector of length length.
                X[i] represents the hidden state value of the i-th step in the most probable path.
    '''
    #########################################
    ## INSERT YOUR CODE HERE



    #########################################
    return X
    



#-----------------------------------------------
def compute_gamma(Ev,I,T,Em):
    '''
        Given a HMM and a sequence of evidence, estimate the parameter of HMM with the highest likelihood of observing the sequence using EM (Baum-Welch) Algorithm. Compute the gamma values for the E-step of EM algorithm.
        Input:
            Ev: the observed evidence sequence, an integer numpy vector of length length. 
                Here length is the number of time steps. Each X[i] = 0,1, ..., or p-1. 
            I : the current estimation of initial probability, a float vector of length col.
            T : the current estimation of transition probablity, a float numpy matrix of shape col by col
                T[i,j] represents the probability of P(Xt = j | Xt-1 = i).
            Em: the current estimation of emission probability, a float numpy matrix of shape col by p.
                Em[i,j] represents the probability of P(Et = j | Xt = i)
        Output:
            gamma: the gamma probability, a numpy matrix of shape length by col
                    gamma[t,i] denotes P(X_t =  i | Ev, HMM_parameters)
            alpha: the alpha probabilities at each time step, a numpy matrix of shape length by col
                    alpha[t] denotes the forward probability at time step t.
            beta: the beta probabilities at each time step, a numpy matrix of shape length by col
                    beta[t] denotes the backward probability at time step t.
    '''
    #########################################
    ## INSERT YOUR CODE HERE

    length = len(Ev)

    col, row = T.shape

    gamma = np.zeros((length, col))
    alpha = np.zeros((length, col))
    beta = np.zeros((length, col))

    for i in range(0, length):
        temp, p, alpha[i, :], beta[i, :] = forward_backward_prob(Ev, I, T, Em, i)
        gamma[i, :] = p / np.sum(p)








    #########################################
    return gamma, alpha, beta 




#-----------------------------------------------
def compute_xi(Ev,T,Em,alpha,beta):
    '''
        Given a HMM and a sequence of evidence, estimate the parameter of HMM with the highest likelihood of observing the sequence using EM (Baum-Welch) Algorithm. Compute the xi values for the E-step of EM algorithm.
        Input:
            Ev: the observed evidence sequence, an integer numpy vector of length n. 
                Here n is the number of time steps. Each X[i] = 0,1, ..., or p-1. 
            I : the current estimation of initial probability, a float vector of length c. 
            T : the current estimation of transition probablity, a float numpy matrix of shape c by c
                T[i,j] represents the probability of P(Xt = j | Xt-1 = i).
            Em: the current estimation of emission probability, a float numpy matrix of shape c by p.
                Em[i,j] represents the probability of P(Et = j | Xt = i)
            alpha: the alpha probabilities at each time step, a numpy matrix of shape n by c
                    alpha[t] denotes the forward probability at time step t.
            beta: the beta probabilities at each time step, a numpy matrix of shape n by c
                    beta[t] denotes the backward probability at time step t.
        Output:
            xi: the xi probability, a numpy matrix of shape n-1 by c by c
                    xi[t,i,j] denotes P(X_t =  i, X_{t+1} = j | Ev, HMM_parameters)
    '''
    #########################################
    ## INSERT YOUR CODE HERE

    xi = np.zeros((len(Ev) - 1, T.shape[0], T.shape[0]))
    row, col = T.shape
    for i in range(0, len(Ev) - 1):
        xi[i, :, :] =( (alpha[i,:].reshape(-1, 1) * T) * Em[:,Ev[i + 1]]) * beta[i + 1, :]
        temp = np.sum(xi[i, :, :])

        xi[i, :, :] = xi[i, :, :] / temp


    #########################################
    return xi 



#-----------------------------------------------
def E_step(Ev,I,T,Em):
    '''
        Given a HMM and a sequence of evidence, estimate the parameter of HMM with the highest likelihood of observing the sequence using EM (Baum-Welch) Algorithm. This function is the E-step of EM algorithm.
        Input:
            Ev: the observed evidence sequence, an integer numpy vector of length n. 
                Here n is the number of time steps. Each X[i] = 0,1, ..., or p-1. 
            I : the current estimation of initial probability, a float vector of length c. 
            T : the current estimation of transition probablity, a float numpy matrix of shape c by c
                T[i,j] represents the probability of P(Xt = j | Xt-1 = i).
            Em: the current estimation of emission probability, a float numpy matrix of shape c by p.
                Em[i,j] represents the probability of P(Et = j | Xt = i)
        Output:
            gamma: the gamma probability, a numpy matrix of shape n by c
                    gamma[t,i] denotes P(X_t =  i | Ev, HMM_parameters)
            xi: the xi probability, a numpy matrix of shape n-1 by c by c
                    xi[t,i,j] denotes P(X_t =  i, X_{t+1} = j | Ev, HMM_parameters)
    '''
    #########################################
    ## INSERT YOUR CODE HERE

    gamma, alpha, beta = compute_gamma(Ev, I, T, Em)
    xi = compute_xi(Ev, T, Em, alpha, beta)

    #########################################
    return gamma, xi



#-----------------------------------------------
def update_I(gamma):
    '''
        In the M-step of EM algorithm, update initial probabilities with gamma values.
        Input:
            gamma: the gamma probability, a numpy matrix of shape n by c
                    gamma[t,i] denotes P(X_t =  i | Ev, HMM_parameters)
        Output:
            I : the new estimation of initial probability, a float vector of length c. 
    '''
    #########################################
    ## INSERT YOUR CODE HERE

    I = gamma[0]

    #########################################
    return I




#-----------------------------------------------
def update_T(gamma,xi):
    '''
        In the M-step of EM algorithm, update transition probabilities with gamma values.
        Input:
            gamma: the gamma probability, a numpy matrix of shape n by c
                    gamma[t,i] denotes P(X_t =  i | Ev, HMM_parameters)
            xi: the xi probability, a numpy matrix of shape n-1 by c by c
                    xi[t,i,j] denotes P(X_t =  i, X_{t+1} = j | Ev, HMM_parameters)
        Output:
            T : the updated estimation of transition probablity, a float numpy matrix of shape c by c
                T[i,j] represents the probability of P(Xt = j | Xt-1 = i).
    '''
    #########################################
    ## INSERT YOUR CODE HERE


    T = np.divide(np.sum(xi, axis=0), (np.sum(gamma, axis=0) - gamma[-1, :]).reshape(-1, 1))



    #########################################
    return T 

#-----------------------------------------------
def update_Em(Ev,gamma,p):
    '''
        In the M-step of EM algorithm, update emission probabilities with gamma values.
        Input:
            Ev: the observed evidence sequence, an integer numpy vector of length n. 
                Here n is the number of time steps. Each X[i] = 0,1, ..., or p-1. 
            gamma: the gamma probability, a numpy matrix of shape n by c
                    gamma[t,i] denotes P(X_t =  i | Ev, HMM_parameters)
            p: the number of possible values of the evidence variable, an integer scalar.
        Output:
            Em: the new estimation of emission probability, a float numpy matrix of shape c by p.
                Em[i,j] represents the probability of P(Et = j | Xt = i)
    '''
    #########################################
    ## INSERT YOUR CODE HERE
    m, n = gamma.shape
    Em = np.zeros((n, p))

    for i in range(0, m):
        Em[:, Ev[i]] += gamma[i, :]
    Em = np.divide(Em, np.sum(gamma, axis=0).reshape(-1, 1))


    #########################################
    return Em 




#-----------------------------------------------
def M_step(Ev, gamma,xi,p):
    '''
        Given a HMM and a sequence of evidence, estimate the parameter of HMM with the highest likelihood of observing the sequence using EM (Baum-Welch) Algorithm. This function is the M-step of EM algorithm.
        Input:
            Ev: the observed evidence sequence, an integer numpy vector of length n. 
                Here n is the number of time steps. Each X[i] = 0,1, ..., or p-1. 
            gamma: the gamma probability, a numpy matrix of shape n by c
                    gamma[t,i] denotes P(X_t =  i | Ev, HMM_parameters)
            xi: the xi probability, a numpy matrix of shape n-1 by c by c
                    xi[t,i,j] denotes P(X_t =  i, X_{t+1} = j | Ev, HMM_parameters)
            p: the number of possible values of the evidence variable, an integer scalar.
        Output:
            I : the new estimation of initial probability, a float vector of length c. 
            T : the new estimation of transition probablity, a float numpy matrix of shape c by c
                T[i,j] represents the probability of P(Xt = j | Xt-1 = i).
            Em: the new estimation of emission probability, a float numpy matrix of shape c by p.
                Em[i,j] represents the probability of P(Et = j | Xt = i)
    '''
    #########################################
    ## INSERT YOUR CODE HERE
    Em = update_Em(Ev, gamma, p)


    I = update_I(gamma)
    T = update_T(gamma, xi)

    #########################################
    return I, T, Em 





#-----------------------------------------------
def EM(Ev,c,p, num_iter=10):
    '''
        Given a HMM and a sequence of evidence, estimate the parameter of HMM with the highest likelihood of observing the sequence using EM (Baum-Welch) Algorithm.
        Input:
            Ev: the observed evidence sequence, an integer numpy vector of length n. 
                Here n is the number of time steps. Each X[i] = 0,1, ..., or p-1. 
            c: the number of possible values of the hidden variable, an integer scalar.
            p: the number of possible values of the evidence variable, an integer scalar.
            num_iter: the number of iterations in EM, an integer scalar.
        Output:
            I : the initial probability distribution of the hidden variable, a float vector of length c. 
            T : the transition probablity of the hidden variable, a float numpy matrix of shape c by c
                T[i,j] represents the probability of P(Xt = j | Xt-1 = i).
            Em: the emission probability of the evidence variable, a float numpy matrix of shape c by p.
                Em[i,j] represents the probability of P(Et = j | Xt = i)
    '''

    # initialize parameters 
    # (This is for testing purpose. In real-world cases, we should randomly initialize the parameters.)
    I = np.arange(float(c))+1.
    I = I /sum(I)
    T = np.arange(float(c*c)).reshape((c,c))+1.
    for i in xrange(c):
        T[i] = T[i]/sum(T[i])
    Em = np.ones((c,p))/p

    #########################################
    ## INSERT YOUR CODE HERE






    #########################################
    return I,T,Em




