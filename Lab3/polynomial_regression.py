import numpy as np
import numpy.linalg as lg
from itertools import chain, combinations
from itertools import combinations_with_replacement as combinations_w_r

# polynomial regression mapping for 1 dimension
def phi(x, n):
    """
    description: polynomial regression mapping for 1 dimension
    args:
        x: the entry variable
        n: the degree of the output polynome
    return: the vector of the form [1, x, x^2, ..., x^n]
    """
    return np.array([x**i for i in range(n+1)]).T

# polynomial regression hypothesis 
def hs(x,w):
    """
    description: polynomial regression hypothesis
    args:
        x: vector of features
        w: vector of weights
    return: matrcial product of x and w.T
    """
    return w.T @ x

# error function between real y and predicted y
def e(x,y,w):
    """
    description: error function between real y and predicted y
    args:
        x: vector of features
        y: label (scalar, 1 or -1)
        w: vector of weights
    return: error between real y and predicted y
    """
    return y - hs(x,w)

# empirical error function
def loss(X,Y,w):
    '''
    description: empirical error function
    args:
        X: list of vectors x_i
        Y: list of labels (scalars, 1 or -1) y_i
        w: vector of weights
    return: average empirical error
    '''
    n = len(X) # size of data sample
    error = [(e(X[i], Y[i], w))**2 for i in range(len(X))] # Mean-Squared Error (MSE)
    return np.sum(error)/n

# Linear regression for polynomial tasks (1 dimension)
def LinearRegressionforPoly(X,Y,n):
    '''
    description: Linear regression for polynomial tasks (1 dimension)
    args:
        X: vector of scalars x_i
        Y: vector of scalars y_i (labels, 1 or 0)
        n: degree of polynomial mapping
    return: vector of weights w and empirical error
    '''
    # mapping the dataset to an upper degree
    X_ = np.asarray([phi(x,n) for x in X])
    # computes the Hessian matrix of the loss function applied to the hypothesis
    A = np.dot(X_.T,X_) 
    # computes the second term of the linear system A.w = b
    b = np.dot(X_.T, Y) 
    # computes the pseudoinverse of A using a Singular-Value Decomposition algorithm
    Aplus = np.linalg.pinv(A)
    # solve the linear system A+.w = b
    w = np.dot(Aplus,b)
    return w, loss(X_,Y,w)

# polynomial mapping for multidimensional input
def PolynomialExpansion( X, max_degree ,min_degree=0):
        
        """
        
        Parameters :
        ------------
        
        X : the input matrix that contains all examples , of shape(#exapmles , #features)
        
        
        """
        
        n_features= X.shape[1]
        n_examples= X.shape[0]
        
        # combinations_w_r  = combination with repitition (=in french : "arrangement avec répétiton")
        
        comb = combinations_w_r
        start = 1
        
        iter = chain.from_iterable(
            comb(range(n_features), i) for i in range(start, max_degree + 1)
        )
        
        #convert the 'iter' into a list
        
        list_combinations =list(iter)
        
        print("the combinations are  = ", list_combinations)
        print("len(list_iter)=", len( list_combinations) )
        print()
        
        XP=np.zeros( [n_examples,len(list_combinations)] )
    
        
     
        
        for i, comb in enumerate(list_combinations):
               """print("hello we are in for loop of Polynomial expension ")
               print("comb = ",comb)
               print("X[:,comb].prod(1) = ", X[:,comb].prod(1))"""
               XP[:, i] = X[:, comb].prod(1)
        
        #now we'll add the bias term
        XP=np.insert(XP,0,1,axis=1)
        print("we have finished the polynomial expension")
        return XP

# polynomial regression 
def PolynomialRegression(X,Y,q):
    """
    description: polynomial regression for multidimensional input using algebraic solution
    args:
        X: list of vectors x_i
        Y: list of label y_i
        n: polynom's degree
    return: vector of weights w and empirical error
    """
    # mapping the dataset to an upper degree
    X_ = PolynomialExpansion(X,q)
    # computes the Hessian matrix of the loss function applied to the hypothesis
    A = np.dot(X_.T,X_) 
    # computes the second term of the linear system A.w = b
    b = np.dot(X_.T, Y) 
    # computes the pseudoinverse of A using a Singular-Value Decomposition algorithm
    Aplus = np.linalg.pinv(A)
    # solve the linear system A+.w = b
    w = np.dot(Aplus,b)
    return w, loss(X_,Y,w)