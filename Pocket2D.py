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
X=np.array([[-0.77653462, -1.76676856],
       [ 1.31480261, -1.32745378],
       [-0.81123925, -0.67986712],
       [ 1.91694867, -1.6599487 ],
       [-0.19468199, -1.33727661],
       [ 2.83010233,  0.43049509],
       [ 1.2741358 , -0.58728677],
       [ 0.1587766 ,  0.84431014],
       [-1.56068564, -1.29936497],
       [-0.21974939, -1.16094751],
       [-0.5935691 , -1.54254941],
       [ 0.33564679, -0.60148532],
       [-0.57415246,  1.26522078],
       [ 1.72305766,  1.51790662],
       [-1.58411805, -0.66931982],
       [ 1.57054243, -0.98425197],
       [ 0.54174332, -1.07782211],
       [ 0.54309691,  0.66143595],
       [ 1.79022524,  1.80597759],
       [-1.00013348,  0.32830595],
       [-1.49895481, -0.85283815],
       [ 1.10832789, -1.6523427 ],
       [-1.97107947, -1.29503691],
       [-0.45439286,  1.6755548 ],
       [-0.97483312,  0.93419291],
       [ 0.73102938, -1.39979353],
       [-1.32093769, -1.95289308],
       [ 0.64000688,  1.15479203],
       [ 1.15665497,  1.12203097],
       [ 0.10349708,  1.13618392],
       [-1.8772425 ,  2.80280889],
       [-0.69067583, -0.24090055],
       [-1.71172162, -1.36080809],
       [-0.36551575,  1.68597564],
       [ 0.13945338,  0.07833992],
       [-1.67986965,  0.51770032],
       [-0.80204854, -0.86027948],
       [-0.68489991,  0.27878456],
       [ 1.38440936, -1.32421466],
       [-1.93387441,  1.91363366],
       [-0.89107725,  1.50747386],
       [-0.34979529, -0.55520347],
       [ 0.12888045,  2.29778138],
       [-0.44518251,  1.55879059],
       [-0.82942575, -0.46483013],
       [ 1.17288821, -0.27822978],
       [ 0.5118538 , -1.96754755],
       [ 1.0094058 , -0.82131697],
       [-0.38519034,  2.11460754],
       [ 0.62814203, -0.01439766]])
# On définit les labels associés aux données d'apprentissage, c'est un problème de classification binaire, ces données sont choisis de telle sorte qu'on peut les séparer en deux classes linéarement par un hyperplan càd on n'a pas du bruit. 
y= np.array([0, 0, 0, 0, 0,  1, 0,  1, 0,  1, 0, 0,  1,  1, 0, 0, 0,
        1,  1,  1, 0, 0, 0,  1,  1, 0, 0,  1,  1,  1,  1, 0, 0,  1,
        1,  1, 0,  1, 0,  1,  1,  1,  1,  1, 0, 0, 0, 0,  1,  1])
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

def _get_cls_map( y):
    return np.where(y <= 0, -1, 1)

#Ajout de x0=1 à chaque example de notre base de donnée pour transformer (hs(x)=w.x + b) en (hs(x)=w.x)
def add_1_x( X):
    x=[]
    for i in range(len(X)):
                    x.append(np.insert(X[i],0,1))
    x = np.array(x)
    return x
def indecatrice( w, x, y):
        if np.sign(np.dot(w,x))!=y :
            return 1
        return 0
def Ls( w, x, y):
        n = len(y)
        s=0
        for i in range(n):
            s+= indecatrice( w,x[i],y[i])
        return s/n
def Pocket( X, y,max_iter):
    y = _get_cls_map(y)
    X = add_1_x(X)
    w1 = np.array([np.zeros(len(X[0]))])
    w1[0][0]=0.1
    w_list=[]

    wt = w1.copy()
    for j in range(max_iter):
        
        for i in range(len(y)):
            #i=random.random()
            estimator = np.dot(wt,X[i])
            if np.sign(estimator) < 0 and y[i]>0:
                wt+= X[i]
            elif np.sign(estimator) > 0 and y[i]<0:
                wt-= X[i]
            if Ls(wt, X, y)<Ls(w1, X, y) :
                w1 = wt.copy()
                w_list.append([w1,Ls(w1, X, y)])
                 # À chaque modification de wt on afficher l'erreur d'approximation.
                print("Erreur d'approximation",Ls(w1,X,y))
        
    #Assigner les valeurs optimales
    return np.array(w1),w_list
    
max_iter = 8
w,w_list=Pocket(X,y,max_iter)

# Plotting the Animated plot (GIF)
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

# Turn off matplotlib plot in Notebook
plt.ioff()
# Pass the ffmpeg path
#plt.rcParams['animation.ffmpeg_path'] = '/Users/hp/anaconda3/Lib/site-packages/ffmpeg'

x1=[]
y1=[]
for point in X:
    x1.append(point[0])
    y1.append(point[1])
colors = y

fig, ax = plt.subplots(figsize=(8,6))
fig.set_tight_layout(True)

ax.scatter(x1, y1, c = y, alpha=0.8)
alpha = 1 
beta =  1
decision_boundary = lambda x : alpha*x + beta
x2 = np.array([-3,2.5])
y2 = decision_boundary(x2)

line, = ax.plot(x2, y2, 'r-', linewidth=1)
def update(w):
    plt.title("hyperplan définie par: w="+str(w[0])+" \n Erreur d'approximation ="+str(w[1]))
    w=w[0]
    alpha = -w[0][1]/w[0][2]
    beta = -w[0][0]/w[0][2]
    decision_boundary = lambda x : alpha*x + beta
    y2 = decision_boundary(x2)
    line.set_ydata(y2)
    


anim = FuncAnimation(fig, update, repeat=True, frames=w_list, interval=2500)
plt.show()
