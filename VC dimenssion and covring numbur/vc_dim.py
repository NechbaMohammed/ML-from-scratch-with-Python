import itertools
import time 
import numpy as np
from sklearn.linear_model import Perceptron



def vc_dimension(h, S,timeSearch=3600):

# """  Cette fonction calcule la dimension VC d'un classificateur en utilisant la méthode du shattering.
#   Args:
#   - h: l'hypothèse h à laquelle nous voulons calculer la dimension vc
#   - S: un ensemble de points de données sous forme de matrice numpy de taille (n, d) où n est le nombre de points et d est la dimension de l'espace de caractéristiques
#   - timeSearch: le temps de calcul souhaité (car le calcul devient rapidement exponentiel le calcul d'une vc de 6 est quasi-impossible)
#   Returns:
#   - vc: la dimension VC du classificateur 
#   - timelaps: le temps écoulé pour calculer la vc dimension


    #les dimensions du jeu de données
    n, d = S.shape
    #tout h à une vc d'au moins 1 donc on initialize par 1
    vc = 1
    #cette variable servira pour sortir de la boucle après avoir tester tout les échantillon A de taille m sont trouvé un qui est shatteré par h
    cantshatter=False
    #cette variable enregistre le temps de commencement du calcul
    start=time.perf_counter()
    #tant qu'on peut shatter on augmente la dimension vc
    while not cantshatter:
        #le nombre d'example à echantilloner pour construire A (la longeur de A à tester si elle va etre shatterer par h)
        m=vc+1
        
        #Si le temps de calucl dépasse le temps voulue par l'utilisateur on renvoie la vc, ansi vc réelle >= vc renvoyé
        if(time.perf_counter()-start>timeSearch):
            return (vc,time.perf_counter()-start)
        
        #cette variable permet de continuer la recherche d'un A de taille vc+1 si le present A ne peut pas etre shattered par h
        contsearch=True  
        
        #On itère sur toute les combinaison possible de A de longueur vc+1 parmi toute notre jeu de donnée S
        for A in itertools.combinations(S, m):
            
            print(round(time.perf_counter()-start,6), " s\n")
            #Si le temps de calucl dépasse le temps voulue par l'utilisateur on renvoie la vc, ansi vc réelle >= vc renvoyé
            if(time.perf_counter()-start>timeSearch):
                return (vc,time.perf_counter()-start)
            
            #pour chaque A on suppose au début qu'il peut etre shattered par h
            cantshatter=False
            
            #Si on ne peut pas continuer car on a trouvé un A de longeur vc+1 shattered par h on sort
            if not contsearch:
                break
            
            #on convertit A en list numpy pour faciliter les calculs
            A=np.array(list(A))
            
            # Génération de tout les labels possible (2**len(x) labels) pour chaque point de données du sous-ensemble (il faut label contient au mois un 0 et un 1 (2 classes differantes))
            vecteur_zero=np.zeros(len(A)).astype(int)
            
            # Pour chaque A on suppose au debut que ce A peut etre shattered par h et on ne doit pas continuer  à chercher un autre combinaison A qui peut etre shattered par h
            contsearch=False
            
            #on itère sur les 2^(vc+1) possibilités de labelisation de A càd les (vc+1) uplet (0,0, ... , 0) jusqu'à (1,1, ... ,1) pour vérifier si len(Ha)=2^(vc+1) 
            for num in range(1,2**m-1):
                
                #Si on ne peut pas continuer car on a trouvé un A de longeur vc+1 shattered par h on sort
                if(time.perf_counter()-start>timeSearch):
                    return (vc,time.perf_counter()-start)
                
                #cette variable est la vecteur binaire (0,1,...)
                vecteur_binaire=np.array(list(bin(num)[2:])).astype(int)
                
                #Dans ce cas on complet par des 0 à gauche 
                if len(vecteur_binaire)<m:
                    label=np.concatenate((vecteur_zero[0:m-len(vecteur_binaire)],vecteur_binaire))
                else:
                    label=vecteur_binaire
                
                #On entraine notre modèle sur l'echantillon A fixé au debut et sur les labels (1,0,...) construit à l'instant
                h.fit(A,label)
                #On genere une prediction à partir de l'echantillon d'entrainenement A
                y_pred = h.predict(A)
                #Si les predictions ne coicide pas à 100% avec les labels generer (càd que cette hypothèse h ne pe pas shatter A) On ne peut pas shatter A et on doit chercher un autre echantillon A de meme longueur qui peut etre shattered par h puis on sort de la boucle 
                if not np.array_equal(label, y_pred):
                    cantshatter=True
                    contsearch=True
                    break
                    
        #Enfin si on peut shatterer càd que le présent A est shattered par h donc vc reelle >= vc+1 ainsi on donne vc=vc+1 et on passe à l'iteration suivante
        if not (cantshatter):
            vc += 1
    
    return (vc,time.perf_counter()-start)




#Punr une même reproduction de résultats
np.random.seed(0)

# Génération aléatoire de notre jeu de données X
X = np.random.rand(20, 5)
print("X ",X)

# Création de la class d'hypothèse Perceptron
H=Perceptron
# Instanciation d'une hypothèse h perceptron
h =  H(tol=1e-3, random_state=0)
#Temps de calcul maximal en secondes
t=10

# Calcul de la dimension VC du perceptron pendant 10s secondes maximum
vc,timelaps = vc_dimension(h, X, t)
if timelaps >= t:
    print("La dimension VC de la class d'hypothèse perceptron sur l'échantillon S est au moins égale à: ", vc)
else:
    print("La dimension VC de la class d'hypothèse perceptron sur l'échantillon S est: ", vc)