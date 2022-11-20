# Used Libraries
# calculations
import random
import math
# plotting vizualisation
import matplotlib.pyplot as plt
# numpy pour les opérations d'algébre linéaire sur les vecteurs et les matrices.
import numpy as np
# Pour que le générateur des nombres aléatoires donne toujours les mêmes nombres, pour éviter d'avoir des résultats différents à chaque exécution.
np.random.seed(0)
# On définit les données d'apprentissage avec deux features (2D).
X=np.array([[-0.72232557, -0.86682292],
        [-0.46936249, -1.45052712],
        [-0.62229699, -0.80261809],
        [ 1.85068362,  1.65668028],
        [-0.30871073, -1.28512984],
        [ 1.52320383,  0.89838313],
        [ 1.45705372, -1.22454469],
        [-0.20163733, -0.33670491],
        [ 0.57098597, -1.36774281],
        [ 0.43999614,  1.49893207],
        [-0.6826726 ,  1.73311864],
        [ 1.62513666, -0.98503269],
        [ 1.18499772,  0.67625894],
        [ 0.29314011, -1.12354835],
        [ 0.61041414,  0.24255405],
        [-0.88145096,  0.64314982],
        [ 0.49850695, -0.82070808],
        [ 0.15872877,  1.35120513],
        [-0.81580377,  0.39439876],
        [-0.58819338, -0.46409956],
        [-0.94122622, -0.96598244],
        [-0.07314702,  0.64674146],
        [-0.90313908, -1.59208177],
        [-0.42846429, -0.66034948],
        [-3.11539885, -0.65583869],
        [-0.2869104 , -0.94033802],
        [ 0.28147691, -1.48564574],
        [ 0.96048994, -1.1239507 ],
        [ 0.71330315,  0.84302527],
        [ 1.6965842 , -1.33839981],
        [-0.95979091,  0.77886686],
        [-0.28515981,  0.50357571],
        [ 1.76892688, -0.90391041],
        [ 0.37698455,  1.37701509],
        [-0.77922552, -1.23625614],
        [ 0.52919039,  0.57437908],
        [ 1.53631261,  0.36185235],
        [ 0.30078867,  1.76541865],
        [-0.82986122, -1.52932409],
        [ 0.19128238, -0.78018795],
        [-1.51537058,  1.15483795],
        [-0.45290225,  0.73725906],
        [-0.18890174,  0.10266195],
        [ 0.14055728,  1.44716524],
        [ 2.37003419,  1.36292231],
        [-1.4641154 , -0.83274709],
        [-0.73341669, -0.34802686],
        [-1.77068289,  1.47834414],
        [ 0.86229771,  1.72248271],
        [ 2.25629503,  1.75153904]])
# On définit les labels associés aux données d'apprentissage, c'est un problème de classification binaire, ces données sont choisis de telle sorte qu'on peut les séparer en deux classes linéarement par un hyperplan càd on n'a pas du bruit. 
y= np.array([0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1,
        0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1,
        1, 0, 0, 1, 1, 1])
# On fait une première visualisation des données.
# On stocke dans la table red les points qu'ont pour label -1.
red = X[y == 0]
# On stocke dans la table blue les points qu'ont pour label 1.
blue = X[y == 1]
# On définit la taille de la figure.
fig = plt.figure(figsize=(8, 6))
# On visualise les points négatifs en coleur rouge.
plt.plot(red[:, 0], red[:, 1], 'r.')
# On visualise les points positifs en coleur bleu.
plt.plot(blue[:, 0], blue[:, 1], 'b.')
# On ajoute un titre à la figure.
plt.title("Visualisation des données utilisées")
# On affiche la figure finale.
plt.show(block=False)
plt.pause(4)
plt.close()

def _get_cls_map(y):
    # chagement des labels y=0 en y=-1 de notre base de donnée
    return np.where(y <= 0, -1, 1)

# Ajout de x0=1 à chaque example de notre base de donnée pour transformer (hs(x)=w.x + b) en (hs(x)=w.x)
def add_1_x(X):
    x = []
    for i in range(len(X)):
        x.append(np.insert(X[i], 0, 1))
    x = np.array(x)
    return x


def indecatrice(w, x, y):
    if np.sign(np.dot(w, x)) != y:
        return 1
    return 0

def Ls(w, x, y):
    n = len(y)
    s = 0
    for i in range(n):
        s += indecatrice(w, x[i], y[i])
    return s/n

def PLA(X, y):
    y = _get_cls_map(y)
    X = add_1_x(X)
    # PLA
    w = np.array([np.zeros(len(X[0]))])
    w[0][0] = 1
    iteration = 0
    w_list = []
    while Ls(w, X, y) != 0:
        for i in range(len(y)):
            estimator = np.dot(w, X[i])
            if np.sign(estimator)*y[i] <= 0:
                w += y[i]*X[i]
                w_list.append([np.array(w),Ls(w,X,y)])
                # On incrémente le compteur des corrections. 
                iteration += 1
                # À chaque modification de wt on afficher l'erreur d'approximation.
                print("Erreur d'approximation",Ls(w,X,y))
                

    # Assigner les valeurs optimales
    return np.array(w), w_list


w,w_list=PLA(X,y)

# Plotting the Animated plot (GIF)
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
# Turn off matplotlib plot in Notebook
plt.ioff()
# Pass the ffmpeg path
#plt.rcParams['animation.ffmpeg_path'] = '/Users/hp/anaconda3/Lib/site-packages/ffmpeg'

x1 = []
y1 = []
for point in X:
    x1.append(point[0])
    y1.append(point[1])
colors = y
fig, ax = plt.subplots(figsize=(8, 6))
fig.set_tight_layout(True)

ax.scatter(x1, y1, c=y, alpha=0.9)
alpha = 1
beta = 1
def decision_boundary(x): return alpha*x + beta


x2 = np.array([-3, 2.5])
y2 = decision_boundary(x2)

line, = ax.plot(x2, y2, 'r-', linewidth=1)


def update(w):
    plt.title("hyperplan définie par: w="+str(w[0])+" \n Erreur d'approximation ="+str(w[1]))
    w=w[0]
    alpha = -w[0][1]/w[0][2]
    beta = -w[0][0]/w[0][2]
    def decision_boundary(x): return alpha*x + beta
    y2 = decision_boundary(x2)
    line.set_ydata(y2)


anim = FuncAnimation(fig, update, repeat=True, frames=w_list, interval=3500)
plt.show()
