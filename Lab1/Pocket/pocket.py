import numpy as np

# perceptron sign 
def hs(x,w):
    """
    args:
        x: vector of features
        w: vector of weights
    return: 1 if h(x,w) > 0, -1 otherwise
    """
    if np.sign(w.T @ x) > 0 : return 1
    return -1

# empirical loss function
def loss(X,Y,w):
    '''
    args:
        X: list of vectors x_i
        Y: list of labels (scalars, 1 or 0) y_i
        w: vector of weights
    returns: average empirical loss
    '''
    n = len(X) # size of data sample
    misclassified = [1 if hs(X[i], w) != Y[i] else 0 for i in range(len(X))]
    return sum(misclassified)/n

# Single Layer Perceptron with Pocket
def Pocket(X,Y,w,Tmax=8):
    '''
    args:
        X: list of vectors x_i
        Y: list of labels (scalars, 1 or 0) y_i
        w: vector of weights
        Tmax: maximum number of iterations
    returns: 
        w: vector of weights after training
        t: number of iterations
        list_w: list of weights

    '''
    n, t = len(X), 0
    w0 = w.copy()
    list_w = []
    while t < Tmax:
        for i in range(n):
            if hs(X[i], w0) * Y[i] < 0 : 
                w0 += X[i]*Y[i]
               
        t += 1
        if loss(X,Y,w0) < loss(X,Y,w) :
             w = w0.copy()
             list_w.append(w)
    return w, t, list_w
    