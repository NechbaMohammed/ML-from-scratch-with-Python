import numpy as np
from itertools import chain, combinations
from itertools import combinations_with_replacement as combinations_w_r

# sigmoid activation function
def hw(x,w):
    '''
    description: sigmoid activation function
    args:
        x: a vector
        w: weight vector
    return: sigmoid of x
    '''
    return 1/(1 + np.exp(-w.T @ x))

# empirical loss of the logistic regression
def Ls(X,Y,w):
    """
    description: empirical loss of the logistic regression
    args:
        X: a list of vectors x_i
        Y: the list of labels y_i associated
        w: weight vector
    return: empirical loss of the logistic regression
    """
    return np.mean( [-Y[i] * np.log(hw(X[i], w)) - (1 - Y[i]) * np.log(1 - hw(X[i], w)) for i in range(len(Y))] )

# gradient of cost function
def DLs(X,Y,w):
    """
    description: gradient of cost function
    args:
        x: a vector x_i
        y: the label y_i associated
        w: weight vector
    return: gradient of cost function
    """
    grad = [] # gradient vector
    d, m = len(w), len(Y) # dimension of data, size of data sample
    for j in range(d): 
        grad.append( np.mean([ (hw(X[i], w) - Y[i])*X[i][j] for i in range(m) ]) )
    return np.array(grad)

# logistic regression algorithm
def LogisticRegression(X, Y, lr = 0.1, Tmax = 1000, eps = 0.2):
    """
    description: logistic regression algorithm
    args:
        X: a list of vectors x_i
        Y: the list of labels y_i associated
        lr: learning rate
        Tmax: maximum number of iterations
        eps: threshold for stopping criterion (precision factor)
    return:
        w: vector of weights after training
        t: number of iterations
        ls: empirical loss
    """
    t = 0 # iteration counter
    w = np.zeros(X.shape[1]) # initialize weights vector
    ls = Ls(X,Y,w) # empirical loss
    while(ls > eps and t < Tmax): # stopping criterion
        if not(t%100) : print("iter:",t,"\t| empirical loss: ", "{0:.6f}".format(ls)) # print loss every 100 iterations
        w -= lr * DLs(X,Y,w) # update weights with gradient descent
        ls = Ls(X,Y,w) # update empirical loss
        t += 1 # increment iteration counter
    return w, t, ls

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

# define a function to plot the decision boundary (NOT FINAL)
def plotDecisionBoundary(w0, X_, axes):
    u = np.linspace(-1, 1.5, 50)
    v = np.linspace(-1, 1.5, 50)
    U,V = np.meshgrid(u,v)
    # convert U, V to vectors for calculating additional features
    # using vectorized implementation
    U = np.ravel(U)
    V = np.ravel(V)
    Z = np.zeros((len(u) * len(v)))
    
    Z = X_.dot(w0)
    
    # reshape U, V, Z back to matrix
    U = U.reshape((len(u), len(v)))
    V = V.reshape((len(u), len(v)))
    Z = Z.reshape((len(u), len(v)))
    
    cs = axes.contour(U,V,Z,levels=[0],cmap= "Greys_r")
    axes.legend(labels=['1', '0', 'Decision Boundary'])
    return cs