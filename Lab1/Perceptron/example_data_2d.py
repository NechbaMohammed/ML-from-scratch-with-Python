# Used Libraries
from perceptron import*
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
X=np.array([[1,-0.72232557, -0.86682292],
        [1,-0.46936249, -1.45052712],
        [1,-0.62229699, -0.80261809],
        [1, 1.85068362,  1.65668028],
        [1,-0.30871073, -1.28512984],
        [1,1.52320383,  0.89838313],
        [1, 1.45705372, -1.22454469],
        [1,-0.20163733, -0.33670491],
        [1, 0.57098597, -1.36774281],
        [1, 0.43999614,  1.49893207],
        [1,-0.6826726 ,  1.73311864],
        [1, 1.62513666, -0.98503269],
        [1, 1.18499772,  0.67625894],
        [1, 0.29314011, -1.12354835],
        [ 1,0.61041414,  0.24255405],
        [1,-0.88145096,  0.64314982],
        [1, 0.49850695, -0.82070808],
        [1, 0.15872877,  1.35120513],
        [1,-0.81580377,  0.39439876],
        [1,-0.58819338, -0.46409956],
        [1,-0.94122622, -0.96598244],
        [1,-0.07314702,  0.64674146],
        [1,-0.90313908, -1.59208177],
        [1,-0.42846429, -0.66034948],
        [1,-3.11539885, -0.65583869],
        [1,-0.2869104 , -0.94033802],
        [1, 0.28147691, -1.48564574],
        [1, 0.96048994, -1.1239507 ],
        [1, 0.71330315,  0.84302527],
        [1, 1.6965842 , -1.33839981],
        [1,-0.95979091,  0.77886686],
        [1,-0.28515981,  0.50357571],
        [ 1,1.76892688, -0.90391041],
        [1, 0.37698455,  1.37701509],
        [1,-0.77922552, -1.23625614],
        [1, 0.52919039,  0.57437908],
        [ 1,1.53631261,  0.36185235],
        [1, 0.30078867,  1.76541865],
        [1,-0.82986122, -1.52932409],
        [ 1,0.19128238, -0.78018795],
        [1,-1.51537058,  1.15483795],
        [1,-0.45290225,  0.73725906],
        [1,-0.18890174,  0.10266195],
        [1, 0.14055728,  1.44716524],
        [1, 2.37003419,  1.36292231],
        [1,-1.4641154 , -0.83274709],
        [1,-0.73341669, -0.34802686],
        [1,-1.77068289,  1.47834414],
        [1, 0.86229771,  1.72248271],
        [ 1,2.25629503,  1.75153904]])
# On définit les labels associés aux données d'apprentissage, c'est un problème de classification binaire, ces données sont choisis de telle sorte qu'on peut les séparer en deux classes linéarement par un hyperplan càd on n'a pas du bruit. 
y= np.array([-1, -1, -1, 1, -1, 1, -1, -1, -1, 1, 1,-1, 1, -1, 1, 1, -1, 1, 1, -1, -1, 1,
        -1,-1, -1,-1, -1, -1, 1, -1, 1, 1, -1, 1, -1, 1, 1, 1, -1, -1, 1, 1, 1, 1,
        1, -1,-1, 1, 1, 1])
# On fait une première visualisation des données.

# On définit la taille de la figure.
fig = plt.figure(figsize=(8, 6))
for i in range(len(X)):
    if y[i] == 1 :
        plt.plot(X[i][1],X[i][2], "or")
    else :
        plt.plot(X[i][1],X[i][2], "og")
plt.show()


w = np.zeros(3)

w,w_list,t=PLA(X,y,w)

# Plotting the Animated plot (GIF)
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
# Turn off matplotlib plot in Notebook
plt.ioff()

x1 = []
y1 = []
for point in X:
    x1.append(point[1])
    y1.append(point[2])
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
    plt.title("hyperplan définie par: w="+str(w)+" \n Erreur d'approximation ="+str(loss(X,y,w)))
    alpha = -w[1]/w[2]
    beta = -w[0]/w[2]
    def decision_boundary(x): return alpha*x + beta
    y2 = decision_boundary(x2)
    line.set_ydata(y2)


anim = FuncAnimation(fig, update, repeat=False, frames=w_list, interval=3500)
plt.show()
