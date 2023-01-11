import numpy as np

# pereceptron hypothesis
def h(x,w):
    """
    args:
        x: vector of features
        w: vector of weights
    returns: scalar value of the hypothesis
    """
    return np.dot(x,w)


# hypothesis sign
def hs(x,w):
    """
    args:
        x: vector of features
        w: vector of weights
    return: 1 if h(x,w) > 0, -1 otherwise
    """
    if np.sign(h(x,w)) > 0 : return 1
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
    print(w,X[0])
    misclassified = [1 if hs(X[i], w) != Y[i] else 0 for i in range(len(Y))]
    return sum(misclassified)/n

# Single Layer Perceptron
def PLA(X,Y,w):
    '''
    args:
        X: list of vectors x_i
        Y: list of labels (scalars, 1 or 0) y_i
        w: vector of weights
    returns: 
        w: vector of weights after training
        t: number of iterations
        list_w : list of weights

    '''
    list_w = []
    n, t = len(X), 0
    while loss(X,Y,w) != 0:
        print(loss(X,Y,w))

        for i in range(n):
            if hs(X[i], w) * Y[i] < 0 : 
                w += X[i]*Y[i]
                list_w.append(w)
        t += 1
    return w, list_w, t

    