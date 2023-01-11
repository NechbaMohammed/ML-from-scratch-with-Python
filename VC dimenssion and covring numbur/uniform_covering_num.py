import itertools
import numpy as np
import matplotlib.pyplot as plt

#Notre class d'hypothèses H qui regroupe les séparateur linéaire du type h(x)=ax+b avec des conditions sur a et b
class H:
    #constructeur de la class
    def __init__(self,a,b):
        self.a=np.array(a)
        self.b=b
    #la fonction de prediction  
    def predict(self,A):
        A=np.array(A)
        return self.a@(A.T) + self.b



#la fonction HA qui génère l'ensemble HA=((h(a1),...,h(am));h in H) la restriction de H sur l'echantillon A
def HA(A,H):
    #On creee l'ensemble H_A
    H_A=list()
    A=np.array(A)
    #On genere toutes les hypothèses de la class d'hypothèse H
    all_h = All_h(H,len(A[0]))
    #On itère  sur tout les h de H, pour faire des predictions sur A et ajouter le vecteur prediction (h(a1),...,h(am)) dans HA
    for h in all_h:
        #prediction de h sur A
        h_a=list(h.predict(A))
        #si cette predictions a dèja été faite on l'ajoute pas dans H_A  car H_A est un ensemble donc il ne doit pas contenir de duplication
        if H_A.count(h_a) == 0:
            H_A.append(h_a)
    return H_A 



#la fonction All_h qui ayant la class d'hypothèse H et la taille de A retourne tout les hypothèses h de H dans une liste
def All_h(H,m):
    #On initialize par une liste vide
    all_h=[]
    #on gènere tout les a possible comme définit par notre H (a est un vecteur car "ax" dans la formule "ax+b" est un produit vectoriel)
    #ici la condition sur a est qu'elle est la longueur de A càd m et que a = [z,...,z] avec z dans l'intervalle [0,11]
    a=[[i for j in range(m)] for i in range(11)]
    #la condition sur b est que c'est un entier paire entre 0 et 20
    b=[2*i for i in range(11)]
    for a_ in a:
        for b_ in b:
            #pour tout ces a et b fixé on gènère une hypothèse h et on l'ajoute a all_h
            all_h.append(H(a_,b_))
    #On retourne toutes les hypothèses h de H dans une liste all_h
    return all_h



def Ne(W, epsilon, d = 2):
    """
    Cette fonction calcule le nombre minimale de boules de rayons epsilone et de centre c dans C qui est inclus dans W capable de couvrir tout le domaine W 
    args:
        W: le domaine qu'on veut couvrir dans notre cas il sera fini et discret)
        epsilon: le rayon des boules de couvertures
        d: la distance utilisé dans notre espace metrique
    returns:
        ne: le nombre de boule minimale pour couvrire le domaine W
    """
    #On créee une copy de notre domaine W qu'on veut couvrire
    points  = W.copy()
    #généralement W est fini soit m sa taille
    m = len(points)
    #on colorie au début les points non couvert par le noire ( càd on ajoute une composante zero à chaque point du domaine W )
    for i in range(len(points)):
        points[i].append(0)
    #on intialise le covering number au début par 0
    ne = 0
    #On initialise le nombre de point couvert (colorié en blanc ) par 0 car au début tout les points sont non couverts (coloriés en noire)
    n_cov = 0
    #Tant qu'il existe au moins un point non couvert on prend le points avec le plus de voisins à epsilone près, on coloris ces points en blanc et on incremente le covering number par 1
    while n_cov < len(points):
        #on creee une liste vide qui va contenir tout les points non couverts avec leurs nombres des voisins et les coordonnées des voisins aussi.
        current_cov = list()
        #on itère sur chaque points du domaine 
        for p in points:
            #on enregistre les coordonnées des voisins de ce point p si celui ci n'a pas été dèja couvert dans la liste voisinage_p.
            #les voisin sont les points situé à une distance inférieur à epsilone du point p
            voisinage_p = [other for other in points if (np.linalg.norm(np.asarray(other[:-1]) - np.asarray(p[:-1]), ord = d) <= epsilon) and (other[-1] != 1)]
            #on enregistre la liste contenant les coordonnées du point p actuel, le nombres de voisins qu'il possède avec leurs coordonnées.
            current_cov.append([p, len(voisinage_p), voisinage_p])
        #On trie cette liste par ordre decroissant du nombre de voisin    
        current_cov.sort(key=lambda x: x[1], reverse=True)
        #On colorie tout ces points en 1 pour ne pas les traiter encore une fois 
        for e in points:
            if e in current_cov[0][2]:
                e[-1] = 1
        #on ajoute le nombre de points traités pendant cette itération au nombre de points traité au total 
        n_cov += current_cov[0][1]
        #finalement on incrémente le covering number par 1
        ne += 1
    return ne



def uniform_covering_numbers(H, epsilon, m, S, d = 2):
    """
    fonction qui calcule le uniform covering number de H à partir d'un echantillon de taille m et d'une distance d 
    args:
        H: la class d'hopothèse
        epsilon: le rayons des boules avec lesquelles nous ferons le pavage de H
        m: la cardinalité de l'espace à paver H (dans notre cas H sear toujours fini car S est fini)
        S: l'echantillon S échantilooné avec une districution D inconnu à partir de X
        d: la distance utilisé dans notre espace metrique (par défaut on utilise la distance Euclidiene)
        returns:
            max_cov_nbr: le uniform covering number de la class d'hypothèse H
    """
    #On génère tout les échantillons A de taille m à partir de notre échantillon de données S
    All_A = np.asarray(list(itertools.combinations(S, m)))
    
    #On initialise max covering number par 1 car  le covering number se situe entre 1 et +inf
    max_cov_nbr = 1
    #On itère sur chaque combinaison de A de taille m pour calculer H_A = {(h(a1), h(a2), ... , h(am)); h in H} et donc le covering number de H_A
    for A in All_A :        
        #On gènère le H_A = {(h(a1), h(a2), ... , h(am)); h in H} en appelant la fonction HA()
        H_A = HA(A,H)        
        #On calcule le covering number en appelant la fonction Ne avec un epsilon donnée et la distance euclidienne
        cov_nbr_A = Ne(H_A.copy(), epsilon, d)
        #si ce covering number calculé pour ce A est supérieur au maximum des covering numbers alors on le prend comme maximum
        if cov_nbr_A > max_cov_nbr:
            max_cov_nbr = cov_nbr_A        
    #puis on retourne le covering number uniforme qui est par définition le maximum des covering number trouvé On aura jamais le cas +inf car notre H_A est génèralement fini donc au maximum cn=max(taille(H_A))
    return max_cov_nbr



# generate 10 random points in the 2D space
S = np.random.rand(4, 2)

petit_rayon = 0.5
grand_rayon = 10
m = 2 # number of points in the subset
ne_petit_rayon = uniform_covering_numbers(H, petit_rayon , m, S)
ne_grand_rayon = uniform_covering_numbers(H, grand_rayon , m, S)


print("uniform covering number avec un rayon petit= ",ne_petit_rayon)
print("uniform covering number un rayon grand= ",ne_grand_rayon)


print("En utilisant un petit rayon nous devons trouver un uniform covering number plus grand! ")
print("c'est exactement le cas! ")

