from problem2 import *
import sys

'''
    Unit test 2: 
    This file includes unit tests for problem2.py. 
    You could test the correctness of your code by typing `nosetests -v test2.py` in the terminal.
'''

#-------------------------------------------------------------------------
def test_python_version():
    ''' ---------- Problem 2 (30 points in total) ------------'''
    assert sys.version_info[0]==2 # require python 2 (instead of python 3)


#-------------------------------------------------------------------------
def test_compute_S():
    '''(5 points) compute_S'''

    # adjacency matrix of shape (3 by 3)
    A = np.mat( [ [0., 1., 0.],
                  [1., 0., 0.],
                  [1., 1., 0.]])

    # call the function
    S = compute_S(A) 

    # test whether or not S is a numpy matrix 
    assert type(S) == np.matrixlib.defmatrix.matrix

    # test the shape of the matrix 
    assert S.shape == (3,3)
    
    # check the correctness of the result 
    S_real=np.mat([[ 0. ,  0.5,  0.333333 ],
                   [ 0.5,  0. ,  0.333333 ],
                   [ 0.5,  0.5,  0.333333 ]] )
    assert np.allclose(S, S_real,atol=1e-2)
   
    #---------------------
    # test with another matrix (no sink node)

    # test on another adjacency matrix of shape (2 by 2)
    A = np.mat( [ [0., 1.],
                  [1., 0.]])
    
    # call the function again
    S = compute_S(A) 

    # test the shape of the matrix 
    assert S.shape == (2,2)

    # check the correctness of the result 
    assert np.allclose(S, A,atol=1e-2)

    #---------------------
    # test with a sink node 

    # test on another adjacency matrix of shape (2 by 2)
    A = np.mat( [ [0., 0.],
                  [1., 0.]])

    # call the function again
    S = compute_S(A) 

    # test the shape of the matrix 
    assert S.shape == (2,2)

    # check the correctness of the result 
    S_real = np.mat( [ [0., 0.5],
                       [1., 0.5]])
    assert np.allclose(S, S_real,atol=1e-2)





#-------------------------------------------------------------------------
def test_compute_G():
    '''(5 points) compute_G'''

    # adjacency matrix of shape (3 by 3)
    A = np.mat([[0., 1., 1.],
                [1., 0., 0.],
                [1., 1., 0.]])

    S = compute_S(A)
    # call the function
    G = compute_G(S, 1.0) 

    # test whether or not G is a numpy matrix 
    assert type(G) == np.matrixlib.defmatrix.matrix

    # test the shape of the matrix
    assert G.shape == (3,3)
    
    # check the correctness of the result 
    G_real=np.mat([[ 0. ,  0.5,  1. ],
                   [ 0.5,  0. ,  0. ],
                   [ 0.5,  0.5,  0. ]] )
    assert np.allclose(G, G_real,atol=1e-2)
   
    #---------------------
    # test with another matrix (no sink node)

    # test on another adjacency matrix of shape (2 by 2)
    A = np.mat( [ [0., 1.],
                  [1., 0.]])
    
    # call the function again
    S = compute_S(A)
    G = compute_G(S, 0.5) 

    # test the shape of the matrix
    assert G.shape == (2,2)

    # check the correctness of the result 
    G_real = [[ 0.25, 0.75],
              [ 0.75, 0.25]]
    assert np.allclose(G, G_real,atol=1e-2)

    #---------------------
    # test with a sink node 

    # test on another adjacency matrix of shape (2 by 2)
    A = np.mat( [ [0., 0.],
                  [1., 0.]])

    # call the function again
    S = compute_S(A)
    G = compute_G(S,0.5) 

    # test the shape of the matrix
    assert G.shape == (2,2)

    # check the correctness of the result 
    G_real = np.mat( [ [0.25, 0.5],
                       [0.75, 0.5]])
    assert np.allclose(G, G_real,atol=1e-2)

    # call the function again
    G = compute_G(A,0.) 

    # check the correctness of the result 
    G_real = np.mat( [ [0.5, 0.5],
                       [0.5, 0.5]])
    assert np.allclose(G, G_real,atol=1e-2)





#-------------------------------------------------------------------------
def test_random_walk_one_step():
    ''' (5 points) random_walk_one_step'''

    # transition matrix of shape (3 by 3) 
    P = np.mat([[ 0. ,  0.5,  1. ],
                [ 0.5,  0. ,  0. ],
                [ 0.5,  0.5,  0. ]] )

    # an all-one vector of shape (3 by 1)
    x_i =  np.asmatrix(np.ones((3,1)))
    
    # call the function 
    x_i_plus_1 = random_walk_one_step(P, x_i) 

    # test whether or not x_i_plus_1 is a numpy matrix 
    assert type(x_i_plus_1) == np.matrixlib.defmatrix.matrix

    # check the shape of the vector
    assert x_i_plus_1.shape == (3,1)

    # check the correctness of the result 
    x_real=np.mat([[1.5],
                   [0.5],
                   [1.0]] )
    assert np.allclose(x_real, x_i_plus_1,atol=1e-2)

    #---------------------
    # test with another matrix

    # another transition matrix of shape (2 by 2) 
    P = np.mat([[ 0.1,  0.4],
                [ 0.9,  0.6]])

    # an all-one vector of shape (2 by 1)
    x_i =  np.asmatrix(np.ones((2,1)))

    # call the function 
    x_i_plus_1 = random_walk_one_step(P, x_i) 

    # check the shape of the vector
    assert x_i_plus_1.shape == (2,1)

    # check the correctness of the result 
    x_real=np.mat([[0.5],
                   [1.5]] )
    assert np.allclose(x_real, x_i_plus_1,atol=1e-2)
 


#-------------------------------------------------------------------------
def test_random_walk():
    ''' (5 points) random_walk'''

    # a transition matrix of shape (3 by 3) 
    P = np.mat([[ 0. ,  0.5,  1. ],
                [ 0.5,  0. ,  0. ],
                [ 0.5,  0.5,  0. ]] )
 
    # an all-one vector of shape (3 by 1)
    x_0 =  np.asmatrix(np.ones((3,1)))

    # call the function 
    x, n_steps = random_walk(P, x_0) 

    # check the shape of the vector
    assert x.shape == (3,1)

    # check number of random walks 
    # assert n_steps == 18
    assert n_steps < 100

    # check the correctness of the result 
    x_real = np.mat( [[ 1.33333333],
                      [ 0.66666667],
                      [ 1.        ]] )
    assert np.allclose(x_real, x,atol=1e-3)
    
    #---------------------
    # test max_steps
    x, n_steps = random_walk(P, x_0,max_steps=2)

    # check number of random walks 
    assert n_steps == 2

    # check the correctness of the result 
    x_real = np.mat( [[ 1.25],
                      [ 0.75],
                      [ 1.  ]] )
    assert np.allclose(x_real, x)

    #---------------------
    # test with another matrix

    # another transition matrix of shape (2 by 2) 
    P = np.mat([[ 0.5,  0.5],
                [ 0.5,  0.5]])

    # an all-one vector of shape (2 by 1)
    x_0 =  np.asmatrix(np.ones((2,1)))

    # call the function 
    x, n_steps = random_walk(P, x_0) 

    # test whether or not x is a numpy matrix 
    assert type(x) == np.matrixlib.defmatrix.matrix

    # check the shape of the vector
    assert x.shape == (2,1)

    # check number of random walks 
    assert n_steps == 1

    # check the correctness of the result 
    x_real=np.array([[1.],
                     [1.]] )
    assert np.allclose(x_real, x,atol=1e-2)



#-------------------------------------------------------------------------
def test_pagerank():
    '''(5 points) pagerank'''

    # adjacency matrix of shape (3 by 3)
    A = np.mat( [ [0., 1., 1.],
                  [1., 0., 0.],
                  [1., 1., 0.]])
    
    # call the function
    x= pagerank(A, 1.0) 

    # test whether or not x is a numpy matrix
    assert type(x) == np.matrixlib.defmatrix.matrix

    # test the shape of the vector
    assert x.shape == (3,1)

    # check the correctness of the result 
    x_real = np.mat( [[ 0.44441732], 
                      [ 0.22224935], 
                      [ 0.33333333]])



    print x
    assert np.allclose(x_real, x, atol = 1e-2)

    # call the function
    x= pagerank(A, 0.0) 

    # check the correctness of the result 
    x_real = np.mat( [[ .33],
                      [ .33],
                      [ .33]] )
    assert np.allclose(x_real, x,atol=1e-2)

#-------------------------------------------------------------------------
def test_dataset2():
    '''(5 points) dataset2'''

    A = np.asmatrix(np.loadtxt('data2.csv', delimiter=','))
    # call the function
    x = pagerank(A,alpha=1) 
    # find the top ranked webpages
    x = x.T
    ids = np.argsort(x.getA1())
    sorted_ids = ids[::-1].tolist() 

    # test the result 
    assert sorted_ids[0]== 461 # the id of the top ranked webpage 
    assert sorted_ids[1]== 210 # the id of second ranked page 
    assert sorted_ids[2]== 811 



