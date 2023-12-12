# Importation des bibliothèques nécessaires
import tempfile  # Utilisé pour générer des fichiers temporaires
import IPython  # Utilisé pour l'interaction avec Jupyter notebook
import math  # Utilisé pour les opérations mathématiques
import numpy as np  # Utilisé pour les opérations sur les tableaux
import matplotlib.pyplot as plt  # Utilisé pour la visualisation des données
from skimage import io  # Utilisé pour lire et écrire des images
from skimage import filters  # Utilisé pour appliquer des filtres aux images
import cv2  # Utilisé pour le traitement d'image
import time  # Utilisé pour le suivi du temps
import random  # Utilisé pour générer des nombres aléatoires
from tqdm import tqdm  # Utilisé pour afficher une barre de progression
import preprocessing_functions as pre  # Module personnalisé pour le prétraitement des images
import tests_functions as tests  # Module personnalisé pour tester les fonctions
import skimage.morphology as morpho #Utilisé pour la dilatation
import pandas as pd #Utilisé pour le calcul du dice coefficient

class Pixel:
    """
    Classe représentant un pixel dans une image.
    """
    def __init__(self, x, y, value):
        """
        Initialise un pixel avec ses coordonnées (x, y) et sa valeur.
        """
        self.x = x
        self.y = y
        self.value = value

class Region:
    """
    Classe représentant une région dans une image.
    """
    def __init__(self, pixel):
        """
        Initialise une région avec un pixel.
        """
        self.pixels = [pixel]  # Liste des pixels de la région
        self.statistics = self.calculate_statistics()  # Propriétés statistiques de la région

    def calculate_statistics(self):
        """
        Calcule les statistiques de la région, par exemple la moyenne et l'écart type des valeurs de pixel.
        """
        RedValues = [pixel.value[0] for pixel in self.pixels]
        GreenValues = [pixel.value[1] for pixel in self.pixels] 
        BlueValues = [pixel.value[2] for pixel in self.pixels]
        # Calcul des moyennes selon chaque canal
        meanR=np.mean(RedValues)
        meanG=np.mean(GreenValues)
        meanB=np.mean(BlueValues)
        mean=[meanR,meanG,meanB]
        # Calcul des écarts types selon chaque canal
        stdR=np.std(RedValues)
        stdG=np.std(GreenValues)
        stdB=np.std(BlueValues)
        std_dev=[stdR,stdG,stdB]
        return {'mean': mean, 'std_dev': std_dev}

def calculate_similarity(region1, region2,choixcanal):
    """
    Calcule la similarité entre deux régions en comparant leurs statistiques.
    La similarité est calculée comme la différence absolue entre les moyennes des valeurs de pixel de chaque région.
    Le canal à utiliser pour le calcul de la similarité est spécifié par `choixcanal`.

    Args:
        region1 (Region): La première région à comparer.
        region2 (Region): La deuxième région à comparer.
        choixcanal (int): Le canal à utiliser pour le calcul de la similarité. 
                          1 pour le canal rouge, 2 pour le canal vert, 3 pour le canal bleu, 4 pour tous les canaux.

    Returns:
        float: La similarité entre les deux régions.
    """

    # Calcule la similarité entre deux régions en comparant leurs statistiques, par exemple en utilisant la différence des moyennes
    mean_diffR = abs(region1.statistics['mean'][0] - region2.statistics['mean'][0])
    mean_diffG = abs(region1.statistics['mean'][1] - region2.statistics['mean'][1])
    mean_diffB = abs(region1.statistics['mean'][2] - region2.statistics['mean'][2])
    if choixcanal==0: #On choisit le canal rouge
        return mean_diffR
    if choixcanal==1: #On choisit le canal vert
        return mean_diffG
    if choixcanal==2: #On choisit le canal bleu
        return mean_diffB
    if choixcanal==3: #On choisit les 3 canaux
        return mean_diffR+mean_diffG+mean_diffB/3

def creer_image_regions_simple(regions,regions_updated,canal):
    """
    Crée une image à partir des régions spécifiées, en utilisant la moyenne des valeurs de pixel de chaque région.
    Seul le canal spécifié est utilisé pour créer l'image.

    Args:
        regions (list of list of Region): Les régions à utiliser pour créer l'image.
        regions_updated (list of list of tuple): Les coordonnées mises à jour des régions.
        canal (int): Le canal à utiliser pour créer l'image.

    Returns:
        np.array: L'image créée à partir des régions.
    """
    image_regions=np.zeros((len(regions),len(regions[0]),3))
    for x in range(len(regions)):
        for y in range(len(regions[0])):
            image_regions[x][y]=[0,0,0]
            image_regions[x][y][canal]=regions[regions_updated[x][y][0]][regions_updated[x][y][1]].statistics['mean'].astype(int)
    return image_regions.astype(int)

def creer_image_regions_3canaux(regions,regions_updated):
    """
    Crée une image à partir des régions spécifiées, en utilisant la moyenne des valeurs de pixel de chaque région.
    Tous les canaux sont utilisés pour créer l'image. On peut choisir si l'on veut tous les canaux, ou seulement 1.

    Args:
        regions (list of list of Region): Les régions à utiliser pour créer l'image.
        regions_updated (list of list of tuple): Les coordonnées mises à jour des régions.

    Returns:
        np.array: L'image créée à partir des régions.
    """

    image_regions=np.zeros((len(regions),len(regions[0]),3))
    for x in range(len(regions)):
        for y in range(len(regions[0])):
            image_regions[x][y]=regions[regions_updated[x][y][0]][regions_updated[x][y][1]].statistics['mean']
    return image_regions.astype(int)

def SRM_segmentation_simple(image, seuil_similarity, canal=2,suivi_temps=False,suivi_image=False):
    """
    Effectue une segmentation SRM (Statistical Region Merging) sur l'image spécifiée.
    Chaque pixel de l'image est initialement considéré comme une région distincte.
    Les régions sont ensuite fusionnées en fonction de leur similarité, jusqu'à ce que la similarité minimale dépasse le seuil spécifié.

    Args:
        image (np.array): L'image à segmenter.
        seuil_similarity (float): Le seuil de similarité pour la fusion des régions.
        canal (int, optional): Le canal à utiliser pour le calcul de la similarité. Par défaut à 2 (canal vert).
        suivi_temps (bool, optional): Si vrai, affiche le temps restant estimé. Par défaut à False.
        suivi_image (bool, optional): Si vrai, affiche une image intermédiaire à chaque tiers de la progression. Par défaut à False.

    Returns:
        tuple: Un tuple contenant les régions après segmentation, les coordonnées mises à jour des régions, et le canal utilisé.
    """
    # Initialisation : chaque pixel est une région
    rows, cols = image.shape[0],image.shape[1]          
    regions = [[Region(Pixel(x, y, image[x,y])) for y in range(cols)] for x in range(rows)]
    regions_updated=[[(x,y)for y in range(cols)] for x in range(rows)]
    nb_regions_ini = len(regions)*len(regions[0])
    print("Nombre de régions au début: ", nb_regions_ini)
    # Fusion des régions
    n=0
    start_time = time.time()
    while True:
        k=0
        min_similarity = float('inf')
        regions_to_merge = None
        merge_occurred = False  # Variable pour suivre si une fusion a eu lieu dans cette itération
        # Recherche des paires de régions les plus similaires
        for x in range(rows):
            for y in range(cols):   
                if x < rows - 1 and regions_updated[x][y]!=regions_updated[x+1][y]:
                    x_region=regions_updated[x][y][0]
                    y_region=regions_updated[x][y][1]
                    x_xplus1y_region=regions_updated[x+1][y][0]
                    y_xplus1y_region=regions_updated[x+1][y][1]
                    similarity = calculate_similarity(regions[x_region][y_region], regions[x_xplus1y_region][y_xplus1y_region],canal)
                    if similarity < min_similarity:
                        min_similarity = similarity
                        regions_to_merge = ((x, y), (x + 1, y))
                        
                if y < cols - 1 and regions_updated[x][y]!=regions_updated[x][y+1]:
                    x_region=regions_updated[x][y][0]
                    y_region=regions_updated[x][y][1]
                    x_xyplus1_region=regions_updated[x][y+1][0]
                    y_xyplus1_region=regions_updated[x][y+1][1]
                    similarity = calculate_similarity(regions[x_region][y_region], regions[x_xyplus1_region][y_xyplus1_region],canal)
                    if similarity < min_similarity:
                        min_similarity = similarity
                        regions_to_merge = ((x, y), (x, y + 1))
                    
        # Vérification si la fusion est nécessaire
        if min_similarity <= seuil_similarity:
            (x1, y1), (x2, y2) = regions_to_merge
            x1_region=regions_updated[x1][y1][0]
            y1_region=regions_updated[x1][y1][1]
            x2_region=regions_updated[x2][y2][0]
            y2_region=regions_updated[x2][y2][1]
            # Fusion des deux régions
            merged_pixel_values = regions[x1_region][y1_region].pixels + regions[x2_region][y2_region].pixels
            # Create a new Region with the merged pixel values
            merged_region = Region(Pixel(x1_region, y1_region, merged_pixel_values[0].value))
            merged_region.pixels = merged_pixel_values
            # mise à jour de regions_updated
            for i in regions[x2_region][y2_region].pixels:
                regions_updated[i.x][i.y]=(x1_region,y1_region)
            # Suppression des anciennes régions
            regions[x1_region][y1_region] = merged_region
            regions[x2_region][y2_region] = None  # Marquer la région comme fusionnée
            regions[x1_region][y1_region].statistics = regions[x1_region][y1_region].calculate_statistics()  # Recalculer les statistiques de la région fusionnée
            merge_occurred = True
            
        else:
            break  # Arrêt lorsque la similarité minimale dépasse le seuil
        if not merge_occurred:
            break  # Arrêt lorsque aucune fusion n'a eu lieu dans cette itération
        # Calcul du temps restant
        n+=1
        if suivi_temps:
            if n%nb_regions_ini//20==0:
                elapsed_time = time.time() - start_time
                remaining_regions = sum([1 for row in regions for region in row if region is not None])
                time_per_merge = elapsed_time / (rows * cols - remaining_regions)
                remaining_time = remaining_regions * time_per_merge
                remaining_minutes = remaining_time // 60
                print(f"Temps restant : {remaining_minutes:.2f} minutes")
        if suivi_image:    
            if n%nb_regions_ini//3==0:
                #Affichage d'une image intermediaire
                image_region=creer_image_regions(regions,regions_updated)
                plt.figure(n)
                k+=1
                plt.imshow(image_region)
                plt.show()


    #Calcul du nombre de régions restantes
    nb_regions_fin=0
    for i in regions:
        for j in i:
            if j is not None:
                nb_regions_fin+=1
    print("Nombre de régions restantes: ", nb_regions_fin)
    return regions,regions_updated,canal


###########################################
def SRM_segmentation_3canaux(image, seuil_similarity,choixcanal,suivi_temps=False,suivi_image=False):
    
    """
    Performs SRM (Statistical Region Merging) segmentation on a 3-channel image.

    Parameters:
    image (numpy.ndarray): The input image to be segmented.
    seuil_similarity (float): The similarity threshold for region merging.
    choixcanal (int): The channel choice for the segmentation.
    suivi_temps (bool, optional): If True, the function will print the remaining time for the segmentation process. Defaults to False.
    suivi_image (bool, optional): If True, the function will display intermediate images during the segmentation process. Defaults to False.

    Returns:
    tuple: A tuple containing two lists. The first list is the final regions after segmentation. The second list is the updated regions after each merge operation.

    """
    # Initialisation : chaque pixel est une région
    rows, cols = image.shape[0],image.shape[1]          
    regions = [[Region(Pixel(x, y, image[x,y])) for y in range(cols)] for x in range(rows)]
    regions_updated=[[(x,y)for y in range(cols)] for x in range(rows)]
    nb_regions_ini = len(regions)*len(regions[0])
    # Fusion des régions
    n=0
    start_time = time.time()
    while True:
        k=0
        min_similarity = float('inf')
        regions_to_merge = None
        merge_occurred = False  # Variable pour suivre si une fusion a eu lieu dans cette itération
        # Recherche des paires de régions les plus similaires
        for x in range(rows):
            for y in range(cols):   
                if x < rows - 1 and regions_updated[x][y]!=regions_updated[x+1][y]:
                    x_region=regions_updated[x][y][0]
                    y_region=regions_updated[x][y][1]
                    x_xplus1y_region=regions_updated[x+1][y][0]
                    y_xplus1y_region=regions_updated[x+1][y][1]
                    similarity = calculate_similarity(regions[x_region][y_region], regions[x_xplus1y_region][y_xplus1y_region],choixcanal)
                    if similarity < min_similarity:
                        min_similarity = similarity
                        regions_to_merge = ((x, y), (x + 1, y))
                        
                if y < cols - 1 and regions_updated[x][y]!=regions_updated[x][y+1]:
                    x_region=regions_updated[x][y][0]
                    y_region=regions_updated[x][y][1]
                    x_xyplus1_region=regions_updated[x][y+1][0]
                    y_xyplus1_region=regions_updated[x][y+1][1]
                    similarity = calculate_similarity(regions[x_region][y_region], regions[x_xyplus1_region][y_xyplus1_region],choixcanal)
                    if similarity < min_similarity:
                        min_similarity = similarity
                        regions_to_merge = ((x, y), (x, y + 1))
                    
        # Vérification si la fusion est nécessaire
        if min_similarity <= seuil_similarity:
            (x1, y1), (x2, y2) = regions_to_merge
            x1_region=regions_updated[x1][y1][0]
            y1_region=regions_updated[x1][y1][1]
            x2_region=regions_updated[x2][y2][0]
            y2_region=regions_updated[x2][y2][1]
            # Fusion des deux régions
            merged_pixel_values = regions[x1_region][y1_region].pixels + regions[x2_region][y2_region].pixels
            # Create a new Region with the merged pixel values
            merged_region = Region(Pixel(x1_region, y1_region, merged_pixel_values[0].value))
            merged_region.pixels = merged_pixel_values
            # mise à jour de regions_updated
            for i in regions[x2_region][y2_region].pixels:
                regions_updated[i.x][i.y]=(x1_region,y1_region)
            # Suppression des anciennes régions
            regions[x1_region][y1_region] = merged_region
            regions[x2_region][y2_region] = None  # Marquer la région comme fusionnée
            regions[x1_region][y1_region].statistics = regions[x1_region][y1_region].calculate_statistics()  # Recalculer les statistiques de la région fusionnée
            merge_occurred = True
            
        else:
            break  # Arrêt lorsque la similarité minimale dépasse le seuil
        if not merge_occurred:
            break  # Arrêt lorsque aucune fusion n'a eu lieu dans cette itération
        # Calcul du temps restant
        n+=1
        if suivi_temps:
            if n%nb_regions_ini//20==0:
                elapsed_time = time.time() - start_time
                remaining_regions = sum([1 for row in regions for region in row if region is not None])
                time_per_merge = elapsed_time / (rows * cols - remaining_regions)
                remaining_time = remaining_regions * time_per_merge
                remaining_minutes = remaining_time // 60
                print(f"Temps restant : {remaining_minutes:.2f} minutes")
        if suivi_image:    
            if n%nb_regions_ini//3==0:
                #Affichage d'une image intermediaire
                image_region=creer_image_regions_3canaux(regions,regions_updated)
                plt.figure(n)
                k+=1
                plt.imshow(image_region)
                plt.show()

    coord_region,occurence_region,max_occurence_index=pre.detection_plus_grosse_region(regions,regions_updated)
    occurence_region.pop(max_occurence_index)
    coord_region.pop(max_occurence_index)
    max_occurence_index=np.argmax(occurence_region)
    if occurence_region[max_occurence_index]<50:
        print("Pas de région supérieur à 50 pixels, diminution du seuil de similarité de 20")
        seuil_similarity-=20
        SRM_segmentation_3canaux(image, seuil_similarity,choixcanal,suivi_temps=False,suivi_image=False)

    #Calcul du nombre de régions restantes
    nb_regions_fin=0
    for i in regions:
        for j in i:
            if j is not None:
                nb_regions_fin+=1
    print("Nombre de régions restantes: ", nb_regions_fin)
    return regions,regions_updated

def SRM_segmentation_finish(regions,regions_updated,choixcanal,seuil_similarity):
    
    """
    Performs SRM (Statistical Region Merging) segmentation on a 3-channel image.

    Parameters:
    image (numpy.ndarray): The input image to be segmented.
    seuil_similarity (float): The similarity threshold for region merging.
    choixcanal (int): The channel choice for the segmentation.
    suivi_temps (bool, optional): If True, the function will print the remaining time for the segmentation process. Defaults to False.
    suivi_image (bool, optional): If True, the function will display intermediate images during the segmentation process. Defaults to False.

    Returns:
    tuple: A tuple containing two lists. The first list is the final regions after segmentation. The second list is the updated regions after each merge operation.

    """
    # Initialisation : chaque pixel est une région
    rows, cols = len(regions),len(regions[0])          
    
    # Fusion des régions
    n=0
    while True:
        k=0
        min_similarity = float('inf')
        regions_to_merge = None
        merge_occurred = False  # Variable pour suivre si une fusion a eu lieu dans cette itération
        # Recherche des paires de régions les plus similaires
        for x in range(rows):
            for y in range(cols):   
                if x < rows - 1 and regions_updated[x][y]!=regions_updated[x+1][y]:
                    x_region=regions_updated[x][y][0]
                    y_region=regions_updated[x][y][1]
                    x_xplus1y_region=regions_updated[x+1][y][0]
                    y_xplus1y_region=regions_updated[x+1][y][1]
                    similarity = calculate_similarity(regions[x_region][y_region], regions[x_xplus1y_region][y_xplus1y_region],choixcanal)
                    if similarity < min_similarity:
                        min_similarity = similarity
                        regions_to_merge = ((x, y), (x + 1, y))
                        
                if y < cols - 1 and regions_updated[x][y]!=regions_updated[x][y+1]:
                    x_region=regions_updated[x][y][0]
                    y_region=regions_updated[x][y][1]
                    x_xyplus1_region=regions_updated[x][y+1][0]
                    y_xyplus1_region=regions_updated[x][y+1][1]
                    similarity = calculate_similarity(regions[x_region][y_region], regions[x_xyplus1_region][y_xyplus1_region],choixcanal)
                    if similarity < min_similarity:
                        min_similarity = similarity
                        regions_to_merge = ((x, y), (x, y + 1))
                    
        # Vérification si la fusion est nécessaire
        if min_similarity <= seuil_similarity:
            (x1, y1), (x2, y2) = regions_to_merge
            x1_region=regions_updated[x1][y1][0]
            y1_region=regions_updated[x1][y1][1]
            x2_region=regions_updated[x2][y2][0]
            y2_region=regions_updated[x2][y2][1]
            # Fusion des deux régions
            merged_pixel_values = regions[x1_region][y1_region].pixels + regions[x2_region][y2_region].pixels
            # Create a new Region with the merged pixel values
            merged_region = Region(Pixel(x1_region, y1_region, merged_pixel_values[0].value))
            merged_region.pixels = merged_pixel_values
            # mise à jour de regions_updated
            for i in regions[x2_region][y2_region].pixels:
                regions_updated[i.x][i.y]=(x1_region,y1_region)
            # Suppression des anciennes régions
            regions[x1_region][y1_region] = merged_region
            regions[x2_region][y2_region] = None  # Marquer la région comme fusionnée
            regions[x1_region][y1_region].statistics = regions[x1_region][y1_region].calculate_statistics()  # Recalculer les statistiques de la région fusionnée
            merge_occurred = True
            
        else:
            break  # Arrêt lorsque la similarité minimale dépasse le seuil
        if not merge_occurred:
            break  # Arrêt lorsque aucune fusion n'a eu lieu dans cette itération
        # Calcul du temps restant
        


    #Calcul du nombre de régions restantes
    nb_regions_fin=0
    for i in regions:
        for j in i:
            if j is not None:
                nb_regions_fin+=1
    print("Nombre de régions restantes: ", nb_regions_fin)
    return regions,regions_updated



def optimisation_seuil_similarity(image,inf_seuil_similarity,sup_seuil_similarity,pas, canal=2):
    """
    Optimizes the similarity threshold for SRM segmentation on a given image.

    Parameters:
    image (numpy.ndarray): The input image to be segmented.
    inf_seuil_similarity (float): The lower bound of the similarity threshold for region merging.
    sup_seuil_similarity (float): The upper bound of the similarity threshold for region merging.
    pas (int): The step size for the similarity threshold in the optimization process.
    canal (int, optional): The channel choice for the segmentation. Defaults to 2.

    Returns:
    None. The function displays a plot of the segmented images for each similarity threshold.

    """
    images_regions = []
    for seuil_similarity in tqdm(range(inf_seuil_similarity, sup_seuil_similarity, pas)):
        regions,regions_updated = SRM_segmentation_simple(image, seuil_similarity, canal)
        image_region = creer_image_regions(regions,regions_updated)
        images_regions.append(image_region)

    # Afficher les images des régions sur un même graphique
    fig, axs = plt.subplots(2, 5, figsize=(20, 8))
    axs = axs.ravel()
    for i, image_region in enumerate(images_regions):
        axs[i].imshow(image_region)
        axs[i].set_title(f"Seuil de similarité : {i*10}")
        axs[i].axis('off')
    
    plt.show()

#########################

def binary_image(regions,regions_updated,canal):
    seuil=50
    liste_images=[]

    def count_and_merge_regions_diff_color(seuil_diff_region=15):
        liste_moyenne=[]
        for x in range(len(regions)):
                for y in range(len(regions[0])):
                    if regions[x][y] != None:
                        k=regions[x][y].statistics['mean']
                        liste_moyenne.append((k,(x,y)))
        for i in range(len(liste_moyenne)):
            for j in range (len(liste_moyenne)):
                if i!=j:
                    if np.abs(liste_moyenne[i][0][0]-liste_moyenne[j][0][0])<seuil_diff_region and np.abs(liste_moyenne[i][0][1]-liste_moyenne[j][0][1])<seuil_diff_region and np.abs(liste_moyenne[i][0][2]-liste_moyenne[j][0][2])<seuil_diff_region:
                        region_couleur_à_modifier=regions[liste_moyenne[j][1][0]][liste_moyenne[j][1][1]]
                        region_couleur_à_garder=regions[liste_moyenne[i][1][0]][liste_moyenne[i][1][1]]
                        for p in region_couleur_à_modifier.pixels:
                            p.value=region_couleur_à_garder.statistics['mean']
                        regions[liste_moyenne[j][1][0]][liste_moyenne[j][1][1]].statistics=regions[liste_moyenne[j][1][0]][liste_moyenne[j][1][1]].calculate_statistics()    
        for i in liste_moyenne:
            for j in liste_moyenne:
                if i!=j:
                    if i[0]==j[0]:
                        liste_moyenne.remove(j)
        return liste_moyenne
    print("début liste moyenne")
    
    while len(count_and_merge_regions_diff_color(15))>2 and seuil<130:

        regions,regions_updated=SRM_segmentation_finish(regions,regions_updated,canal,seuil)
        liste_moyenne=count_and_merge_regions_diff_color(15)
        
        
        coord_region,occurence_region,max_occurence_index=pre.detection_plus_grosse_region(regions,regions_updated)
        #On enlève la plus grosse région qui est le fond
        coord_region.pop(max_occurence_index)
        occurence_region.pop(max_occurence_index)
        #on cherche toutes les regiosn qui ont plus de 500 pixels
        liste_index=[]
        for i in range(len(occurence_region)):
            if occurence_region[i]>100:
                liste_index.append(i)
        #On fait une liste avec leurs coordonnées
        liste_coord=[]
        for i in liste_index:
            liste_coord.append(coord_region[i])
        #On cherche la région qui a le plus de pixel
        max_occurence_index=np.argmax(occurence_region)
        coord_region_max=coord_region[max_occurence_index]
        x=coord_region_max[0]
        y=coord_region_max[1]
        meanRGB=regions[x][y].statistics['mean']
        for i in liste_coord:
            for j in regions[i[0]][i[1]].pixels:
                j.value=meanRGB
            regions[i[0]][i[1]].statistics=regions[i[0]][i[1]].calculate_statistics()
        image_current=creer_image_regions_3canaux(regions,regions_updated)
        liste_images.append(image_current)
        seuil+=10
    
    if seuil==130:
        seuil_taille=30
        liste_index=[]
        for i in range(len(occurence_region)):
            if occurence_region[i]<=seuil_taille:
                liste_index.append(i)
        #On fait une liste avec leurs coordonnées
        liste_coord=[]
        for i in liste_index:
            liste_coord.append(coord_region[i])
        for i in liste_coord:
            for j in regions[i[0]][i[1]].pixels:
                j.value=[0,255,255]
            regions[i[0]][i[1]].statistics=regions[i[0]][i[1]].calculate_statistics()
    
    #BINARISATION
    
    image_obtenue=creer_image_regions_3canaux(regions,regions_updated)
    image_regions_binaire=cv2.cvtColor(image_obtenue.astype(np.uint8),cv2.COLOR_RGB2GRAY)

    
    
    seuil=image_regions_binaire[0][0]+1
    image_regions_binaire_final=image_regions_binaire<seuil
    if np.all(image_regions_binaire_final == True) or np.all(image_regions_binaire_final == False):
        seuil=image_regions_binaire[0][0]-1
        image_regions_binaire_final=image_regions_binaire<seuil

    #determiner les coordonnées d'un pixel appartenant à la région finale
    coord_region,occurence_region,max_occurence_index=pre.detection_plus_grosse_region(regions,regions_updated)
    max_occurence_index=np.argmax(occurence_region)
    coord_region.pop(max_occurence_index)
    occurence_region.pop(max_occurence_index)
    max_occurence_index=np.argmax(occurence_region)
    coord_region_max=coord_region[max_occurence_index]
    if image_regions_binaire_final[coord_region_max[0]][coord_region_max[1]]==False:
        #inverser le blanc et le noir
        image_regions_binaire_final=np.invert(image_regions_binaire_final)

    

    #for i in range(len(liste_images)):
    #    plt.figure(i)
    #    plt.imshow(liste_images[i])
    #plt.show()
    #plt.figure("image binaire")
    #plt.title("image binaire")
    #plt.imshow(image_regions_binaire_final,cmap='gray')
    return image_regions_binaire_final

def dilatation(disk_size,image):
    image_dilatation=morpho.binary_dilation(image,morpho.disk(disk_size))
    return image_dilatation