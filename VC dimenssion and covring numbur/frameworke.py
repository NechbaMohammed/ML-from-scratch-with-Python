import numpy as np
import random
import math
import itertools

def create_combi(n, arr, i, combi):
    """
    Create all possible combinations of 0 and 1 for the last column of the array
    args:
        n: the number of rows of the array
        arr: the array to create combinations
        i: the current row to create combinations
        combi: the list to store all combinations
    """
    if i == n:
        combi.append(np.copy(arr))
        return
    arr[i][-1] = 0
    create_combi(n, arr, i + 1, combi)
    arr[i][-1] = 1
    create_combi(n, arr, i + 1, combi)
    

def shatter(classifier, data):
    """
    Check if the classifier can shatter the data
    args:
        classifier: the classifier to check
        data: the data to check
    return: 
        True if the classifier can shatter the data
    """
    all_combi = []
    X = np.copy(data)
    create_combi(len(X), X, 0, all_combi)
    all_combi = np.asarray(all_combi)
    MAX = all_combi.shape[0]
    for i in range(MAX):
        if all(all_combi[i][:, -1] == 0) or all(all_combi[i][:, -1] == 1) :
            continue
        classifier.fit(all_combi[i][:, :-1], all_combi[i][:, -1])
        y_ = classifier.predict(all_combi[i][:, :-1])
        if not all(y_ == all_combi[i][:, -1]):
            return False
    return True

def VC_dimension(classifier, data):
    """
    Calculate the VC dimension of the classifier
    args:
        classifier: the classifier to calculate
        data: the data to calculate
    return:
        the VC dimension of the classifier
    """
    vc = 1
    for k in range(2, data.shape[0] + 1):
        A = list(itertools.combinations(data, k))
        i = 1
        for subset in A:
            if shatter(classifier, np.asarray(subset)):
                vc += 1
                break
            if i == len(A):
                return vc
            i += 1
    return vc

def sample_complexity_bound(d, epsilon, delta):
    """ 
    sample complexity bound
    args:
        d: vc dimension
        epsilon: accuracy
        delta: confidence
    return:
        sample complexity
    """
    return math.ceil(d* np.log(1/(epsilon**delta))/epsilon)
def  realisation(classifier,m, epsilon, s,S):
        #print(s)
        classifier.fit(s[:, :-1], s[:, -1])
        #print("Pass")
        hs = classifier.predict(s[:, :-1])
        ls = np.sum(hs == s[:, -1])/len(hs)
        ld = Ld(classifier,S,m)
        if abs(ls-ld) < epsilon:
            return 0
        else:
            return 1

def Ld(classifier,S_sauf_s_train, m):
    #print(S_sauf_s_train)
    ls_cumulu = 0
    cpt=0
    for s in S_sauf_s_train :
            s= np.asarray(s)
            if all(s[:, -1] == 0) or all(s[:, -1] == 1) :
                continue
            cpt += 1
  
            y_pred = classifier.predict(s[:, :-1])
            ls_cumulu += np.sum(y_pred == s[:, -1])

    return ls_cumulu/(m*cpt)

def Remove(S,s):
   
    for i in range(len(S)):
       if np.equal(np.asarray(s),np.asarray(S[i])).all():
            S.pop(i)
            break
    return S

def frameworke_UC(classifier, data, epsilon, delta):
    d = VC_dimension(classifier, data)
    m  =  sample_complexity_bound(d,epsilon,delta)
    print("m",m)
    if m<= data.shape[0] and m>1:
        X_failing = 0
        cpt=0
        S= list(itertools.combinations(data, m))
        for i in range(len(S)):
            s = np.asarray(S[i])
            if all(s[:, -1] == 0) or all(s[:, -1] == 1) :
                continue

            cpt += 1
            X_failing += realisation(classifier,m, epsilon, s, Remove(S.copy(),s))
        Proba_failing = X_failing/cpt
        if Proba_failing<=delta:
            print("L'apprentissage des classificateur est garantie selon la methode de UC avec une confiance de ",1-delta," et une precision de ",epsilon)

        else:
            print("n'est pas d'apprentissage",Proba_failing)
    else:
        print("la taille de data ne permet pas l'apprentissage")


from sklearn.linear_model import Perceptron
from sklearn.datasets import make_blobs
import numpy as np

data = np.array([[0,0,1],[0.01, 0.02,1],[ 0.03, 0.04,0],[ 0.05, 0.06,0],[ 0.07, 0.08,0],[ 0.09, 0.10,1],[0.11, 0.12,1],[0.13, 0.14,1],[0.15, 0.16,0],[0.17, 0.18,1],[0.19, 0.20,0],[0.21, 0.22,1]
,[0.012, 0.002,1],[0.03, 0.04,0],[0.055, 0.86,0],[ 0.77, 0.08,1]])

# Création d'un perceptron
perceptron =  Perceptron(tol=1e-3, random_state=0)
frameworke_UC(perceptron, data, 0.1, 0.1)



    



