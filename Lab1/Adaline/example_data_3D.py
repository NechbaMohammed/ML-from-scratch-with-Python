#Used Libraries:
from adaline import*
#calculations
import numpy as np
import pandas as pd
import random
import math

# matplotlib pour la visualisation des données et des séparateurs.
import matplotlib.pyplot as plt
# numpy pour les opérations d'algébre linéaire sur les vecteurs et les matrices.
import numpy as np
# Pour que le générateur des nombres aléatoires donne toujours les mêmes nombres, pour éviter d'avoir des résultats différents à chaque exécution.
np.random.seed(0)
# On définit les données d'apprentissage avec trois  features (3D).
X=np.array([[ 1.01549349e+00,  1.57846028e+00, -1.09717313e+00],
       [ 1.72538101e+00, -1.08136162e+00, -5.24580948e-02],
       [-8.60690813e-01,  1.55785787e+00,  3.92180913e-01],
       [ 6.31645697e-01, -1.09784521e+00, -6.74809029e-01],
       [-1.26759251e+00, -1.21433640e+00,  5.54849998e-01],
       [ 5.24683252e-01,  1.51347542e+00,  9.01593361e-02],
       [-9.94644379e-01,  5.61356397e-02, -5.77075963e-01],
       [ 1.65755706e+00, -1.40681829e+00,  6.75035632e-01],
       [-1.03861083e+00, -9.75925376e-02, -1.55092569e+00],
       [ 1.43283138e+00,  2.01372026e+00, -9.92427015e-01],
       [ 3.21321436e-01, -5.82904214e-01,  7.58357785e-01],
       [-1.02661164e+00, -8.34899993e-01, -5.11417994e-01],
       [ 2.22922904e+00,  4.07940644e-01, -7.80816174e-01],
       [ 9.01888845e-01, -9.66604408e-01,  7.82143135e-01],
       [-1.16117432e+00, -9.94205632e-01, -8.65821114e-01],
       [-7.25743358e-01, -1.13409594e+00, -1.26661669e-01],
       [ 1.91922893e+00,  5.97523247e-01,  3.04983754e-01],
       [-1.00121777e+00,  1.54214105e+00,  1.22967241e+00],
       [-8.76798839e-01,  3.16602440e+00, -3.41897229e-01],
       [ 1.39638119e+00,  2.10023252e+00,  4.15589959e-01],
       [-9.81807702e-01,  9.57271102e-01,  2.59967911e-01],
       [ 2.41173828e+00,  1.10592111e+00, -1.42566326e+00],
       [ 1.70001884e-01, -6.10984610e-01, -2.34689434e-02],
       [-1.32307096e+00, -1.27576031e+00, -4.17494228e-01],
       [-1.27613739e+00, -7.61712802e-01,  1.01419168e-01],
       [ 1.23503936e+00, -1.18361597e+00,  6.04976876e-01],
       [-1.01933636e+00,  2.27096375e-01, -9.38109136e-01],
       [-5.72170035e-01, -1.30416673e+00,  9.13130560e-02],
       [-8.36763148e-01,  3.06544669e+00, -1.59212509e-01],
       [ 9.34023251e-01, -7.29995901e-01,  1.13428049e+00],
       [ 8.99826839e-01,  7.00306343e-01,  5.30123699e-01],
       [-1.04492782e+00,  1.18647017e+00,  1.29095043e+00],
       [-1.00721880e+00,  1.11066349e+00,  6.04680194e-01],
       [-6.07200574e-01, -6.43033084e-01,  1.23218423e-01],
       [ 2.40986465e+00, -1.63100065e+00,  1.26361373e+00],
       [-7.88869564e-01, -7.86347081e-01,  5.75850368e-01],
       [-9.55687295e-01,  1.14973699e+00, -2.94094482e-03],
       [-9.75704375e-01,  2.54374533e+00,  3.96732608e-01],
       [-1.50610005e+00, -1.11540208e+00, -1.04769817e-01],
       [-1.44734048e+00, -1.10480029e+00, -6.22778104e-01],
       [-1.02659339e+00, -1.56509264e-01, -1.76377928e+00],
       [ 2.26557893e+00, -1.34613250e+00, -8.65064531e-01],
       [-6.68121132e-01, -7.44460531e-01, -1.96216018e+00],
       [-1.06832185e+00, -8.82806694e-01,  5.32913420e-01],
       [ 9.33782947e-01, -1.12899838e-01,  1.72466156e-01],
       [ 5.61447730e-01, -9.42352841e-01, -3.27485240e-01],
       [ 9.62461898e-01, -1.10986586e+00, -1.31430544e+00],
       [ 1.84588947e+00,  9.69975224e-01,  8.77690626e-01],
       [ 1.78237540e+00,  1.98220957e+00, -1.81973707e-01],
       [ 9.21437579e-01, -8.65511041e-01,  2.20837683e+00]])
# On définit les labels associés aux données d'apprentissage, c'est un problème de classification binaire, ces des données sont choisis de telle sorte qu'on ne peut pas les séparer en deux classe linéarement par un hyperplan càd on a du bruit. 
y=np.array([ 1, 0,  1, 0, 0,  1,  1, 0,  1,  1, 0, 0,  1, 0, 0, 0,  1,
        1,  1,  1,  1,  1, 0, 0, 0, 0,  1, 0,  1, 0,  1,  1,  1, 0,0,
       0, 0,  1,  1,  0,  1, 0, 0, 0,  1, 0, 0,  1,  1,  1]) 
# On fait une première visualisation des données.
# On stocke dans la table red les points qu'ont pour label -1.
red = X[y == 0]
# On stocke dans la table blue les points qu'ont pour label 1.
blue = X[y == 1]
# On définit la taille de la figure.
fig = plt.figure(figsize=(8, 6))
ax=plt.axes(projection="3d")
# On visualise les points négatifs en coleur rouge.
ax.plot3D(red[:, 0], red[:, 1],red[:,2], 'r.')
# On visualise les points positifs en coleur bleu.
ax.plot3D(blue[:, 0], blue[:, 1],blue[:,2], 'b.')
# On ajoute un titre à la figure.
plt.title("Visualisation des données utilisées")
# On affiche la figure finale.
plt.show(block=False)
plt.pause(4)
plt.close()


 #chagement des labels y=0 en y=-1 dans notre base de donnée    
def _get_cls_map( y):
        return np.where(y <= 0, -1, 1)

 #Ajout de x0=1 à chaque example de notre base de donnée pour transformer (hs(x)=w.x + b) en (hs(x)=w.x)
def add_1_x( X):
        x=[]
        for i in range(len(X)):
                        x.append(np.insert(X[i],0,1))
        x = np.array(x)
        return x

X = add_1_x(X)
y = _get_cls_map(y)

w = np.zeros(X.shape[1])   
w,t,w_list=Adaline(X,y,w)

def visualise_hyperplan(w):
    # On extrait le biais
    b = w[0]
    # On veut dessiner la surface dans l'espace définie par les trois axes: x (feature1), y (feauture2) et z (feauture3).
    # La surface est une fonction linéaire. Pour dessiner cette fonction on aure besoin d'un ensembe des couples (i,j)  qu'sont défines sur les deux axes x*y.
    # linspace Renvoie des nombres régulièrement espacés sur un intervalle spécifié.
    nbrs = np.linspace(-2.5,2.5,20)
    # Création des couples.
    i,j = np.meshgrid(nbrs,nbrs)
    # Les images de ces couple sont définies sur le dernier axe z.
    # Pour calculer ces images: l'eqution de la surface est wx+b=0 => w1x1+w2x2+w3x3+b=0 =>  w3x3= -w1x1-w2x2-b => x3 = -(w1/w3)x1-(w2/w3)x2- b/w3 .
    z_values = lambda x,y: (-b-w[1]*x -w[2]*y) /w[3]
    # On défine la taille de la figure.
    fig = plt.figure(figsize=(16, 12))
    # Il faut créer des axes 3D.
    ax=plt.axes(projection="3d")
    # On visualise les points négatifs en coleur rouge.
    ax.plot3D(red[:, 0], red[:, 1],red[:,2], 'r.')
    # On visualise les points positifs en coleur bleu.
    ax.plot3D(blue[:, 0], blue[:, 1],blue[:,2], 'b.')
    # On visualise la surface.
    ax.plot_surface(i,j, z_values(i,j),color='green' )
    # On définit l'angle de la figure.
    ax.view_init(30, 60)
    # On ajoute un titre à la figure.
    plt.title("hyperplan définie par: w="+str(w)+" \n Erreur d'approximation ="+str(loss(X,y,w)))
    # On affiche la figure.
    plt.show()
   
visualise_hyperplan(w)
