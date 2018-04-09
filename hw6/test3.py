from problem3 import *
import numpy as np
import sys

'''
    Unit test 3:
    This file includes unit tests for problem3.py.
    You could test the correctness of your code by typing `nosetests -v test3.py` in the terminal.
'''

#-------------------------------------------------------------------------
def test_python_version():
    ''' ----------- Problem 3 (50 points in total)---------------------'''
    assert sys.version_info[0]==2 # require python 2 (instead of python 3)

#-------------------------------------------------------------------------
def test_forward_prob():
    ''' (5 points) forward_probability'''
    I = np.array([0.1, 0.9]) # initial probs
    T = np.array([[0.75, 0.25], # transition probs. / 2 states
                  [0.32, 0.68]])
    Em = np.array([[0.8, 0.1, 0.1], #  emission (observation) probs. / 3 obs modes
                   [0.1, 0.2, 0.7]])

    Ev = np.array([0]) # observed evidence
    Xt,a = forward_prob(Ev,I,T,Em)
    assert Xt==1
    assert np.allclose(a,[.08,.09],atol=1e-2)

    I = np.array([0.3, 0.7]) # initial probs
    Ev = np.array([0]) # observed evidence
    Xt,a = forward_prob(Ev,I,T,Em)
    assert Xt==0
    assert np.allclose(a,[.24,.07],atol=1e-2)


    I = np.array([0.5, 0.5]) # initial probs
    Ev = np.array([0]) # observed evidence
    Xt,a = forward_prob(Ev,I,T,Em)
    assert Xt==0
    assert np.allclose(a,[.4,.05],atol=1e-2)

    I = np.array([0.5, 0.5]) # initial probs
    Ev = np.array([0,2]) # observed evidence
    Xt,a = forward_prob(Ev,I,T,Em)
    assert Xt==1
    assert np.allclose(a,[.0316,.0937],atol=1e-3)

    Ev = np.array([0,1,2,1,0]) # observed evidence
    Xt,a = forward_prob(Ev,I,T,Em)
    assert Xt==0

    Ev = np.array([0,1,2,1]) # observed evidence
    Xt,a = forward_prob(Ev,I,T,Em)
    assert Xt==1
    Ev = np.array([0,1,2]) # observed evidence
    Xt,a = forward_prob(Ev,I,T,Em)
    assert Xt==1
    Ev = np.array([0,1,2,0]) # observed evidence
    Xt,a = forward_prob(Ev,I,T,Em)
    assert Xt==0
    Ev = np.array([0,1,1,1]) # observed evidence
    Xt,a = forward_prob(Ev,I,T,Em)
    assert Xt==1
    Ev = np.array([0,1,1,0]) # observed evidence
    Xt,a = forward_prob(Ev,I,T,Em)
    assert Xt==0
    assert np.allclose(a,[.00327,.000436],atol=1e-4)


#-------------------------------------------------------------------------
def test_backward_prob():
    ''' (5 points) backward_probability'''
    T = np.array([[0.75, 0.25], # transition probs. / 2 states
                  [0.32, 0.68]])
    Em = np.array([[0.8, 0.1, 0.1], #  emission (observation) probs. / 3 obs modes
                   [0.1, 0.2, 0.7]])

    Ev = np.array([0]) # observed evidence
    X,b = backward_prob(Ev,T,Em)
    assert X==0 # when there is a tie, select the smallest index
    assert np.allclose(b,[1,1],atol=1e-2)

    Ev = np.array([0,2]) # observed evidence
    X,b = backward_prob(Ev,T,Em)
    assert X==1
    assert np.allclose(b,[.25,.508],atol=1e-3)

    Ev = np.array([0,1,1,0]) # observed evidence
    X,b = backward_prob(Ev,T,Em)
    assert X==1
    assert np.allclose(b,[.00793,.01073],atol=1e-4)


#-------------------------------------------------------------------------
def test_forward_backward_prob():
    ''' (3 points) forward_backward_probability'''
    I = np.array([0.1, 0.9]) # initial probs
    T = np.array([[0.75, 0.25], # transition probs. / 2 states
                  [0.32, 0.68]])
    Em = np.array([[0.8, 0.1, 0.1], #  emission (observation) probs. / 3 obs modes
                   [0.1, 0.2, 0.7]])

    Ev = np.array([0]) # observed evidence
    X,p,a,b = forward_backward_prob(Ev,I,T,Em,0)
    assert X==1
    assert np.allclose(p,[.08,.09],atol=1e-2)
    assert np.allclose(a,[.08,.09],atol=1e-2)
    assert np.allclose(b,[1.,1.],atol=1e-2)

    I = np.array([0.5, 0.5]) # initial probs
    Ev = np.array([0]) # observed evidence
    X,p,a,b = forward_backward_prob(Ev,I,T,Em,0)
    assert X==0
    assert np.allclose(p,[.4,.05],atol=1e-2)
    assert np.allclose(a,[.4,.05],atol=1e-2)
    assert np.allclose(b,[1.,1.],atol=1e-2)


    I = np.array([0.5, 0.5]) # initial probs
    Ev = np.array([0,2]) # observed evidence
    X,p,a,b = forward_backward_prob(Ev,I,T,Em,1)
    assert X==1
    assert np.allclose(p,[.0316,.0937],atol=1e-3)
    assert np.allclose(a,[.0316,.0937],atol=1e-3)
    assert np.allclose(b,[1.,1.],atol=1e-2)

    X,p,a,b = forward_backward_prob(Ev,I,T,Em,0)
    assert X==0
    assert np.allclose(p,[.1,.0254],atol=1e-3)
    assert np.allclose(a,[.4,.05],atol=1e-3)
    assert np.allclose(b,[.25,.508],atol=1e-3)

    Ev = np.array([0,1,1,0]) # observed evidence
    X,p,a,b = forward_backward_prob(Ev,I,T,Em,1)
    assert X==0
    assert np.allclose(p,[.00199,.00172],atol=1e-4)
    assert np.allclose(a,[.0316,.0268],atol=1e-3)
    assert np.allclose(b,[.063075,.064064],atol=1e-3)


#-------------------------------------------------------------------------
def test_most_probable_pass():
    ''' (5 points) most_probable_pass'''
    I = np.array([0.5, 0.5]) # initial probs
    T = np.array([[0.75, 0.25], # transition probs. / 2 states
                  [0.32, 0.68]])
    Em = np.array([[0.8, 0.1, 0.1], #  emission (observation) probs. / 3 obs modes
                   [0.1, 0.2, 0.7]])
    Ev = np.array([0,1,2,1,0]) # observed evidence
    X = most_probable_pass(Ev,I,T,Em)
    assert np.allclose(X,[0,1,1,1,0],atol=1e-1)

    I = np.array([0., 1.]) # initial probs
    X = most_probable_pass(Ev,I,T,Em)
    assert np.allclose(X,[1,1,1,1,0],atol=1e-1)

    Ev = np.array([2,2,2,2,2]) # observed evidence
    X = most_probable_pass(Ev,I,T,Em)
    assert np.allclose(X,[1,1,1,1,1],atol=1e-1)

    Ev = np.array([0,0,0,0,0]) # observed evidence
    X = most_probable_pass(Ev,I,T,Em)
    assert np.allclose(X,[1,0,0,0,0],atol=1e-1)

#-------------------------------------------------------------------------
def test_gamma():
    ''' (5 points) compute_gamma'''
    I = np.array([0.1, 0.9]) # initial probs
    T = np.array([[0.75, 0.25], # transition probs. / 2 states
                  [0.32, 0.68]])
    Em = np.array([[0.8, 0.1, 0.1], #  emission (observation) probs. / 3 obs modes
                   [0.1, 0.2, 0.7]]) 
    Ev = np.array([0,1]) # observed evidence
    g,a,b = compute_gamma(Ev,I,T,Em)
    assert np.allclose(g,[[.398,.602],[.354,.646]],atol=1e-2)
    assert np.allclose(a,[[.08,.09],[.00888,.01624]],atol=1e-3)
    assert np.allclose(b,[[.125,.168],[1.,1.]],atol=1e-3)


#-------------------------------------------------------------------------
def test_xi():
    ''' (5 points) compute_xi'''
    I = np.array([0.1, 0.9]) # initial probs
    T = np.array([[0.75, 0.25], # transition probs. / 2 states
                  [0.32, 0.68]])
    Em = np.array([[0.8, 0.1, 0.1], #  emission (observation) probs. / 3 obs modes
                   [0.1, 0.2, 0.7]]) 
    Ev = np.array([0,1]) # observed evidence
    g,a,b = compute_gamma(Ev,I,T,Em)
    x = compute_xi(Ev,T,Em,a,b)
    assert np.allclose(x,[[[.2388,.159],[.11,.49]]],atol=1e-2)

    Ev = np.array([0,1,0]) # observed evidence
    g,a,b = compute_gamma(Ev,I,T,Em)
    x = compute_xi(Ev,T,Em,a,b)
    assert np.allclose(x,[[[.3468,.1199],[.1665,.3668]],[[.4928,.0205],[.3845,.1021]]],atol=1e-2)



#-------------------------------------------------------------------------
def test_E_step():
    ''' (2 points) E_step'''
    I = np.array([0.1, 0.9]) # initial probs
    T = np.array([[0.75, 0.25], # transition probs. / 2 states
                  [0.32, 0.68]])
    Em = np.array([[0.8, 0.1, 0.1], #  emission (observation) probs. / 3 obs modes
                   [0.1, 0.2, 0.7]]) 
    Ev = np.array([0,1]) # observed evidence
    g,x = E_step(Ev,I,T,Em)
    assert np.allclose(g,[[.398,.602],[.354,.646]],atol=1e-2)
    assert np.allclose(x,[[[.2388,.159],[.11,.49]]],atol=1e-2)



#-------------------------------------------------------------------------
def test_update_I():
    ''' (5 points) update_I'''
    g=np.array([[.1,.9],[.2,.8]])
    I = update_I(g)
    assert np.allclose(I,[.1,.9],atol=1e-1)
    
#-------------------------------------------------------------------------
def test_update_T():
    ''' (5 points) update_T'''

    I = np.array([0.1, 0.9]) # initial probs
    T = np.array([[0.75, 0.25], # transition probs. / 2 states
                  [0.32, 0.68]])
    Em = np.array([[0.8, 0.1, 0.1], #  emission (observation) probs. / 3 obs modes
                   [0.1, 0.2, 0.7]]) 
    Ev = np.array([0,1]) # observed evidence
    g,a,b = compute_gamma(Ev,I,T,Em)
    x = compute_xi(Ev,T,Em,a,b)
    T = update_T(g,x)
    assert np.allclose(T,[[.6,.4],[.19,.81]],atol=1e-2)

#-------------------------------------------------------------------------
def test_update_Em():
    ''' (5 points) update_Em'''
    g= np.array([[.1,.9],[.2,.8]])
    Ev = np.array([0,1]) # observed evidence
    Em = update_Em(Ev,g,2)
    assert np.allclose(Em,[[.3333,.66667],[.5294,.4706]],atol=1e-3)
  

    Em = update_Em(Ev,g,3)
    assert np.allclose(Em,[[.3333,.66667,0],[.5294,.4706,0]],atol=1e-3) 

    I = np.array([0.1, 0.9]) # initial probs
    T = np.array([[0.75, 0.25], # transition probs. / 2 states
                  [0.32, 0.68]])
    Em = np.array([[0.8, 0.1, 0.1], #  emission (observation) probs. / 3 obs modes
                   [0.1, 0.2, 0.7]]) 
    Ev = np.array([0,1,0]) # observed evidence
    g,a,b = compute_gamma(Ev,I,T,Em)
    Em = update_Em(Ev,g,2)
 
    assert np.allclose(Em,[[.7236,.2763],[.5741,.4259]],atol=1e-3)


#-------------------------------------------------------------------------
def test_M_step():
    ''' (2 points) M_step'''
    I = np.array([0.1, 0.9]) # initial probs
    T = np.array([[0.75, 0.25], # transition probs. / 2 states
                  [0.32, 0.68]])
    Em = np.array([[0.8, 0.1, 0.1], #  emission (observation) probs. / 3 obs modes
                   [0.1, 0.2, 0.7]]) 
    Ev = np.array([0,1,0]) # observed evidence
    g,a,b = compute_gamma(Ev,I,T,Em)
    x = compute_xi(Ev,T,Em,a,b)
    I,T,Em = M_step(Ev,g,x,2)
    assert np.allclose(I,[.467,.533],atol=1e-2)
    assert np.allclose(T,[[.857,.143],[.54,.46]],atol=1e-2)
    assert np.allclose(Em,[[.7236,.2763],[.5741,.4259]],atol=1e-3)

#-------------------------------------------------------------------------
def test_EM():
    ''' (3 points) EM'''

    Ev = np.array([0,1]) # observed evidence
    I,T,Em = EM(Ev,c=2,p=3,num_iter=1)
    assert np.allclose(I,[1./3,2./3],atol=1e-2)
    assert np.allclose(Em,[[.457,.543,0],[.525,.475,0]],atol=1e-2)
    assert np.allclose(T,[[1./3,2./3],[.429,.571]],atol=1e-2)

    I,T,Em = EM(Ev,c=2,p=3,num_iter=2)
    assert np.allclose(I,[0.3,0.7],atol=1e-2)
    assert np.allclose(Em,[[.408,0.592,0],[.55,.45,0]],atol=1e-2)
    assert np.allclose(T,[[.364,.636],[.462,.538]],atol=1e-2)

    Ev = np.array([0,1,0,1]) # observed evidence
    I,T,Em = EM(Ev,c=2,p=3,num_iter=20)
    assert np.allclose(I,[0,1],atol=1e-2)
    assert np.allclose(Em,[[0,1,0],[1,0,0]],atol=1e-2)
    assert np.allclose(T,[[0,1],[1,0]],atol=1e-2)

