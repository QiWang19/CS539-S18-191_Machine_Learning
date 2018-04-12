import numpy as np
#-------------------------------------------------------------------------
'''
    Problem 2: PageRank Algorithm (Markov Chain and Stationary Probability Distribution) 
    In this problem, we implement the pagerank algorithm, which create a markov chain and use random walk to compute the stationary probability distribution.
    You could test the correctness of your code by typing `nosetests -v test2.py` in the terminal.
'''

#--------------------------
def compute_S(A):
    '''
        compute the transition matrix S from addjacency matrix A, which solves sink node problem by filling the all-zero columns in A.
        S[j][i] represents the probability of moving from node i to node j.
        If node i is a sink node, S[j][i] = 1/n.
        Input: 
                A: adjacency matrix, a (n by n) numpy matrix of binary values. If there is a link from node i to node j, A[j][i] =1. Otherwise A[j][i]=0 if there is no link.
        Output: 
                S: transition matrix, a (n by n) numpy matrix of float values.  S[j][i] represents the probability of moving from node i to node j.
    The values in each column of matrix S should sum to 1.
    '''
    #########################################
    ## INSERT YOUR CODE HERE
    # arr = [[0 for i in range(len(A))] for r in range(len(A)) ]
    # cnt = [0 for i in range(len(A)) ]
    # for i in range(len(A)):
    #     for j in range(len(A)):
    #         if A[j,i] == 1:
    #             cnt[i] = cnt[i] + 1
    #
    #
    # for i in range(len(A)):
    #     for j in range(len(A)):
    #         if cnt[i] == 0:
    #             arr[j][i] = 1. / len(A)
    #         elif cnt[i] != 0:
    #             arr[j][i] = A[j,i] / cnt[i]
    #
    # S = np.mat(arr)
    m, n = A.shape
    S = np.divide(A , np.sum(A, 0))
    S[np.isnan(S)] = float(1) / n
    #########################################
    return S

#--------------------------
def compute_G(S, alpha = 0.95):
    '''
        compute the pagerank transition Matrix G from matrix S, which solves the sing region problem.
        G[j][i] represents the probability of moving from node i to node j.
        If node i is a sink node, S[j][i] = 1/n.
        Input: 
                S: transition matrix, a (n by n) numpy matrix of float values.  S[j][i] represents the probability of moving from node i to node j.
                alpha: a float scalar value, which is the probability of choosing option 1 (randomly follow a link on the page)
        Output: 
                G: the transition matrix, a (n by n) numpy matrix of float values.  G[j][i] represents the probability of moving from node i to node j.
    The values in each column of matrix G should sum to 1.
    '''
    #########################################
    ## INSERT YOUR CODE HERE
    length = len(S)
    arr = np.zeros((S.shape))
    for i in range(len(S)):
        for j in range(length):
            arr[i,j] = (1. / len(S))

    mm = np.zeros((S.shape))
    for i in range(length):
        for j in range(length):
            mm[i,j] = ((1.0 - alpha) * arr[i,j] )

    G = np.add((alpha * S), mm)


    #########################################
    return G



#--------------------------
def random_walk_one_step(G, x_i):
    '''
        compute the result of one step random walk.
        Input: 
                G: transition matrix, a (n by n) numpy matrix of float values.  G[j][i] represents the probability of moving from node i to node j.
                x_i: pagerank scores before the i-th step of random walk. a numpy vector of shape (n by 1).
        Output: 
                x_i_plus_1: pagerank scores after the i-th step of random walk. a numpy vector of shape (n by 1).
    '''
    #########################################
    ## INSERT YOUR CODE HERE
    x_i_plus_1 = np.matmul(G, x_i)

    #########################################
    return x_i_plus_1


#--------------------------
def random_walk(G, x_0, max_steps=10000):
    '''
        compute the result of multi-step random walk until reaching stationary distribution. 
        The random walk should stop if the score vector x no longer change (converge) after one step of random walk, or the number of iteration reached max_steps.
        Input: 
                G: transition matrix, a (n by n) numpy matrix of float values.  G[j][i] represents the probability of moving from node i to node j.
                x_0: the initial pagerank scores. a numpy vector of shape (n by 1).
                max_steps: the maximium number of random walk steps. an integer value.  
        Output: 
                x: the final pagerank scores after multiple steps of random walk. a numpy vector of shape (n by 1).
                n_steps: the number of steps actually used (for example, if the vector x no longer changes after 3 steps of random walk, return the value 3. 
        Hint: you could use np.allclose(x, previous_x,atol=1e-4) function to determine when to stop the random walk iterations.
    '''
    #########################################
    ## INSERT YOUR CODE HERE


    x = x_0
    nextx = x_0
    n_steps = 0
    while n_steps < max_steps:
        x = np.dot(G, x)
        n_steps += 1
        if np.allclose(x, nextx):
            return x, n_steps
        nextx = x

    #########################################
    return x, n_steps

#--------------------------
def pagerank(A, alpha = 0.95):
    ''' 
        The final PageRank algorithm, which solves both the sink node problem and sink region problem.
        Given an adjacency matrix A, compute the pagerank score (stationary probability distribution) in the network. 
        Input: 
                A: adjacency matrix, a numpy matrix of binary values. If there is a link from node i to node j, A[j][i] =1. Otherwise A[j][i]=0 if there is no link.
                alpha: a float scalar value, which is the probability of choosing option 1 (randomly follow a link on the page)
        Output: 
                x: the stationary probability distribution, a numpy vector of float values, such as np.array([[.3], [.5], [.2]])
    '''
    
    # Initialize the score vector with all one values
    num_nodes, _ = A.shape # get the number of nodes (n)

    x_0 =  np.ones((num_nodes,1))/num_nodes # create a unique distribution as the initial probablity distribution.
    # x_0 = np.ones((num_nodes, 1))
    #########################################
    ## INSERT YOUR CODE HERE

    S = compute_S(A)

    G = compute_G(S, alpha)

    x, n_steps = random_walk(G, x_0)




    #########################################
    return x

