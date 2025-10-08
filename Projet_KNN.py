import numpy as np

#%%Cette partie ouvre et lit le fichier "train.csv" pour le transformer en une matrice de valeurs
train = open("train.csv","r")
M = train.readlines()
val_train = []
for line in M[1:]: #Slicing de la liste M car on enlève la première ligne de texte
    line = line.strip().split(",") #Transforme la liste de string en une matrice où chaque élément est une liste des caractérisques en string de chaque point
    line = list(map(float,line)) #Transforme les caractéristiques de chaque ligne en float
    val_train.append(line)
train.close()

#On crée une matrice de valeurs sans l'ID
val_train_2 = []
for i in val_train:
    val_train_2.append(i[1:])

#%%Cette partie ouvre et lit le fichier "test.csv" pour le transformer en une matrice de valeurs
test = open("test.csv","r")
N = test.readlines()
val_test = []
for line in N[1:]: #Slicing de la liste N car on enlève la première ligne de texte
    line = line.strip().split(",") #Transforme la liste de string en une matrice où chaque élément est une liste des caractérisques en string de chaque point
    line = list(map(float,line)) #Transforme les caractéristiques de chaque ligne en float
    val_test.append(line)
test.close()

#On crée une matrice de valeurs sans l'ID
val_test_2 = []
for i in val_test:
    val_test_2.append(i[1:])

#%%Cette partie est l'alogrithme knn
def dist_euclidienne(P1,P2):
    dist = 0
    for a,b in zip(P1,P2): #le zip fait directement la comparaison en ne prenant pas en compte le lable de train
        dist += (a-b)**2
    return dist**(1/2) #retourne la somme de la racine carré des différences au carré

def dist_manhattan(P1,P2):
    dist = 0
    for a,b in zip(P1,P2): #le zip fait directement la comparaison en ne prenant pas en compte le lable de train
        dist += np.absolute(a-b)
    return dist #retourne La somme des valeurs absolues des différences

def KNN(P_ref,k):
    list_dist = [] #liste contenant des tuples avec la distance et l'indice du point avec lequel on compare
    for i in range (len(val_train_2)):
        dist = dist_manhattan(P_ref,val_train_2[i])
        list_dist.append((dist,i))
    list_dist = sorted(list_dist,key=lambda x:x[0]) #On range par ordre croissant
    top_k = list_dist[:k] #On prend les top k trouvés


    Classe = {} #dictionnaire qui compte le nombre d'apparitions d'une classe
    for val in top_k:
        distance = val[0]
        indice = val[-1]
        classe_val = val_train_2[indice][-1] #La classe du point dans train avec lequel on a trouvé la plus petite distance
        poids = 1/(distance + 1e-6) #On évite de diviser par 0 grâce à "1e-6" et Le poids que l'on rajoute est en fonction de la distance avec l'indice du point avec lequel on a comparé auparavant
        if not(classe_val in Classe): #Cette boucle permet de mettre le poids de la classe, c'est-à-dire l'importance de chaque classe
            Classe[classe_val] = poids
        else:
            Classe[classe_val] += poids

    classe = max(Classe,key=Classe.get) #On retoune la classe avec le plus gros poids
    return classe


#%%
envoi = open("envoi.csv","w")
envoi.write("Id, Label\n")
for indice,val in enumerate(val_test_2):
    envoi.write(f"{int(val_test[indice][0])},{int(KNN(val,3))}\n") #écrit "ID,classe" dans le fichier envoi.csv
envoi.close()
#%%sans poids
#knn=10 : 0.97073
#knn=9 : 0.96975
#knn=8 : 0.97463
#knn=7 : 0.97365
#knn=6 : 0.97463
#knn=5 : 0.97463
#knn=4 : 0.98146
#knn=3 : 0.97756
#knn=2 : 0.98731
#knn=1 : 0.98731

#%%avec poids
#knn=7 : 0.97658
#knn=4 : 0.98146

#%%manhattan
#knn=4 : 0.98341
#knn=7 : 0.98048
#knn=5 : 0.98243
#knn=3 : 0.98536