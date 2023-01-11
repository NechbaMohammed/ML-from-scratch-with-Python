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
# On définit les données d'apprentissage avec trois  features (3D).
X=np.array([[1,4.8, 3.4, 1.9],
       [1,4.6, 3.1, 1.5],
       [1,5.1, 3.5, 1.4],
       [1,4.4, 3. , 1.3],
       [1,4.4, 2.9, 1.4],
       [1,6. , 2.9, 4.5],
       [1,4.6, 3.4, 1.4],
       [1,5.6, 2.9, 3.6],
       [1,5.5, 3.5, 1.3],
       [1,5.5, 2.5, 4. ],
       [1,6.3, 3.3, 4.7],
       [1,5.7, 2.8, 4.1],
       [1,6.5, 2.8, 4.6],
       [1,5. , 3.5, 1.6],
       [1,7. , 3.2, 4.7],
       [1,5.8, 2.7, 4.1],
       [1,5.1, 3.8, 1.6],
       [1,6.2, 2.2, 4.5],
       [1,5.9, 3. , 4.2],
       [1,6.2, 2.9, 4.3],
       [1,5.7, 2.6, 3.5],
       [1,4.5, 2.3, 1.3],
       [1,6.6, 2.9, 4.6],
       [1,5.3, 3.7, 1.5],
       [1,5.1, 2.5, 3. ],
       [1,4.9, 2.4, 3.3],
       [1,6.6, 3. , 4.4],
       [1,5.2, 4.1, 1.5],
       [1,5.6, 2.7, 4.2],
       [1,5.2, 2.7, 3.9],
       [1,6.1, 2.9, 4.7],
       [1,5.4, 3. , 4.5],
       [1,4.9, 3.6, 1.4],
       [1,4.7, 3.2, 1.6],
       [1,4.9, 3. , 1.4],
       [1,6.9, 3.1, 4.9],
       [1,5.1, 3.7, 1.5],
       [1,4.7, 3.2, 1.3],
       [1,5.1, 3.3, 1.7],
       [1,6.3, 2.3, 4.4],
       [1,6.1, 3. , 4.6],
       [1,6.4, 2.9, 4.3],
       [1,6.7, 3.1, 4.7],
       [1,5.8, 2.7, 3.9],
       [1,5.4, 3.4, 1.7],
       [1,5. , 2. , 3.5],
       [1,6.1, 2.8, 4. ],
       [1,5.8, 4. , 1.2],
       [1,5.8, 2.6, 4. ],
       [1,6.4, 3.2, 4.5]])
# On définit les labels associés aux données d'apprentissage, c'est un problème de classification binaire, ces des données sont choisis de telle sorte qu'on peut les séparer en deux classe linéarement par un hyperplan càd on n'a pas du bruit. 
y=np.array([-1, -1, -1, -1, -1, 1, -1, 1, -1, 1, 1, 1, 1, -1, 1, 1, -1, 1, 1, 1, 1, -1,
       1, -1, 1, 1, 1, -1, 1, 1, 1, 1, -1, -1, -1, 1, -1, -1, -1, 1, 1, 1, 1, 1,
       -1, 1, 1, -1, 1, 1])
# On fait une première visualisation des données.
# On stocke dans la table red les points qu'ont pour label -1.
red = X[y == -1]
# On stocke dans la table blue les points qu'ont pour label 1.
blue = X[y == 1]
# On définit la taille de la figure.
fig = plt.figure(figsize=(8, 6))
ax=plt.axes(projection="3d")

# On visualise les points négatifs en coleur rouge.
ax.plot3D(red[:, 1], red[:, 2],red[:,3],'r.')
# On visualise les points positifs en coleur bleu.
ax.plot3D(blue[:, 1], blue[:, 2],blue[:,3],'b.')
# On ajoute un titre à la figure.
plt.title("Visualisation des données utilisées")
# On affiche la figure finale.
plt.show(block=False)
plt.pause(4)
plt.close()

w = np.zeros(4)
w,w_list,t=PLA(X,y,w)


def visualise_hyperplan(w):
    # On extrait le biais
    b = w[0]
    # On veut dessiner la surface dans l'espace définie par les trois axes: x (feature1), y (feauture2) et z (feauture3).
    # La surface est une fonction linéaire. Pour dessiner cette fonction on aure besoin d'un ensembe des couples (i,j)  qu'sont défines sur les deux axes x*y.
    # linspace Renvoie des nombres régulièrement espacés sur un intervalle spécifié.
    nbrs = np.linspace(1,7,20)
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
    ax.plot3D(red[:, 1], red[:, 2],red[:,3], 'r.')
    # On visualise les points positifs en coleur bleu.
    ax.plot3D(blue[:, 1], blue[:, 2],blue[:,3], 'b.')
    # On visualise la surface.
    ax.plot_surface(i,j, z_values(i,j),color='green' )
    # On définit l'angle de la figure.
    ax.view_init(30, 60)
    # On ajoute un titre à la figure.
    plt.title("hyperplan définie par: w="+str(w)+" \n Erreur d'approximation ="+str(loss(X,y,w)))
    # On affiche la figure.
    if w[1]!=0:
	     plt.show(block=False)
	     plt.pause(4)
	     plt.close()
    else :
            plt.show()
for w in w_list:
       visualise_hyperplan(w)

