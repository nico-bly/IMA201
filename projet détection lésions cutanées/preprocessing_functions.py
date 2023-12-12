import tempfile
import IPython
import math
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage import filters
import cv2
import time
import random
import SRM_segmentation_functions as SRM
import tests_functions as tests


def remove_black_frame(im):
    # Calcul de la luminosité
    im_lightness = np.zeros((im.shape[0], im.shape[1]))
    for row in range(im.shape[0]):
        for column in range(im.shape[1]):
            with np.errstate(over='ignore'):
                luminosity = (max(im[row][column]) + min(im[row][column])) // 2
            if luminosity < 60:
                im_lightness[row][column] = 0
            else:
                im_lightness[row][column] = 1
    # Suppression du cadre noir - Numérisation de haut en bas
    seuil = 0
    row = 0
    while row < im.shape[0] and seuil < 0.6:
        S = np.sum(im_lightness[row])
        seuil = S / im.shape[1]
        row += 1

    # Suppression du cadre noir - Numérisation de gauche à droite
    seuil = 0
    column = 0
    while column < im.shape[1] and seuil < 0.6:
        col_lightness=[]
        for i in range(im.shape[0]):
            col_lightness.append(im_lightness[i][column])
        S = np.sum(col_lightness)
        seuil = S / im.shape[1]
        column += 1
    
    # Suppression du cadre noir - Numérisation de droite à gauche
    seuil = 0
    column_reverse = im.shape[1] - 1
    while seuil < 0.6 and column_reverse > 0:
        col_lightness=[]
        for i in range(im.shape[0]):
            col_lightness.append(im_lightness[i][column_reverse])
        seuil = S / im.shape[1]
        column_reverse -= 1

    #Suppresion du cadre noir -Numérisation de bas en haut
    seuil = 0
    row_reverse = im.shape[0] - 1
    while seuil < 0.6 and row_reverse > 0:
        S = np.sum(im_lightness[row_reverse])
        seuil = S / im.shape[1]
        row_reverse -= 1
        


    if not (row < im.shape[0] and column < im.shape[1] and column_reverse > 0):
        return im
    # Suppression du cadre noir - Réduction de l'image
    im_cropped = im[row:, column:column_reverse]
    
    return im_cropped

#Filtre median
def median_filter(im):
    # Calcul de la taille du filtre
    M=im.shape[0]
    N=im.shape[1]
    kernel_size=int(np.floor(5*np.sqrt((M/768)*(N/512))))
    if kernel_size%2==0:
        kernel_size+=1
    image_filtree = cv2.medianBlur(im, kernel_size)
    return image_filtree

# Fonction de redimensionnement de l'image
def resize_image(im,size):
    # Réduire la taille de l'image
    im_optimal_size = cv2.resize(im, (size[0],size[1]))
    return im_optimal_size

#Fonction de pré-processing complet
def preprocessing(im, size):
    # Suppression du cadre noir
    im_cropped = remove_black_frame(im)
    # Redimensionnement
    image_optimal_size = resize_image(im_cropped, size)
    # Filtrage médian
    image_filtree = median_filter(image_optimal_size)
    
    return image_filtree


def black_frame_removal2(im,couleur_peau):
    im_cropped=remove_black_frame(im)
    seuil=50
    #image greyscale
    im_grey = cv2.cvtColor(im_cropped, cv2.COLOR_BGR2GRAY)
    for row in range (im_cropped.shape[0]):
        for column in range (im_cropped.shape[1]):
            if im_grey[row][column]<seuil:
                im_grey[row][column]=0
            else:
                im_grey[row][column]=255
    for row in range (im_grey.shape[0]):
        for column in range(im_grey.shape[1]):
            #coin1
            if row<250 and column<250:
                if im_grey[row][column]==0:
                    im_cropped[row][column]=couleur_peau
            #coin2
            if row<250 and column>im_grey.shape[1]-250:
                if im_grey[row][column]==0:
                    im_cropped[row][column]=couleur_peau
            #coin3
            if row>im_grey.shape[0]-250 and column<250:
                if im_grey[row][column]==0:
                    im_cropped[row][column]=couleur_peau
            #coin4
            if row>im_grey.shape[0]-250 and column>im_grey.shape[1]-250:
                if im_grey[row][column]==0:
                    im_cropped[row][column]=couleur_peau
    return im_cropped


####################
def detection_plus_grosse_region(regions,regions_updated):
    # Initialisation des listes pour stocker les coordonnées des régions et leurs occurrences
    coord_region=[]
    occurence_region=[]
    
    # Parcours des régions mises à jour
    for x in range(len(regions_updated)):
        for y in range(len(regions_updated[0])):
            # Si la région à (x, y) n'est pas déjà dans coord_region, on l'ajoute
            if regions_updated[x][y] not in coord_region:
                coord_region.append(regions_updated[x][y])
                # Initialisation de l'occurrence de cette région à 0
                occurence_region.append(0)
    
    # Nouveau parcours des régions mises à jour
    for x in range(len(regions_updated)):
        for y in range(len(regions_updated[0])):
            # Parcours de la liste coord_region
            for i in range(len(coord_region)):
                # Si la région à (x, y) correspond à la i-ème région dans coord_region
                if regions_updated[x][y]==coord_region[i]:
                    # Incrémentation de l'occurrence de la i-ème région
                    occurence_region[i]+=1
    
    # Recherche de l'index de la région avec l'occurrence maximale
    max_occurence_index=np.argmax(occurence_region)
    
    # Retourne la liste des coordonnées des régions, leurs occurrences, et l'index de la région avec l'occurrence maximale
    return coord_region,occurence_region,max_occurence_index


def remplacement_fond(regions,regions_updated):
    #ETAPE 1
    #On remplace la plus grosse région par du fond en cyan, couleur qui n'a pas de composante rouge
    cyan_color=[0,255,255]
    #on cherche la région qui a le plus de pixel
    coord_region,occurence_region,max_occurence_index=detection_plus_grosse_region(regions,regions_updated)
    coord_region_max=coord_region[max_occurence_index]
    x=coord_region_max[0]
    y=coord_region_max[1]
    coord_region_coin=[regions_updated[0][0],regions_updated[1][0],regions_updated[0][1],regions_updated[1][1],(x,y)]
    #on remplace les pixels des régions du fond par du cyan
    for coord in coord_region_coin:
        x=coord[0]
        y=coord[1]
        for j in regions[x][y].pixels:
            j.value=cyan_color
        regions[x][y].statistics=regions[x][y].calculate_statistics()
    
    #ETAPE 2
    #On remplace les régions qui sont de taille 1 ou 2 par un pixel de fond
    liste_index=[]
    for i in range(len(occurence_region)):
        if occurence_region[i]<=30:
            liste_index.append(i)
    #On fait une liste avec leurs coordonnées
    liste_coord=[]
    for i in liste_index:
        liste_coord.append(coord_region[i])
    for i in liste_coord:
        for j in regions[i[0]][i[1]].pixels:
            j.value=cyan_color
        regions[i[0]][i[1]].statistics=regions[i[0]][i[1]].calculate_statistics()

    #ETAPE 3
    #On remplace les régions de petites tailles et qui sont vraiment sur le côté de l'image
    #on prend les régions d'occurence inférieur à 100:
    for i in range(len(occurence_region)):
        if occurence_region[i]<100:
            x,y=coord_region[i]
            listeX=[]
            listeY=[]
            liste_coord=[]
            for i in range(len(regions_updated)):
                for j in range(len(regions_updated[0])):
                    if regions_updated[i][j]==(x,y):
                        listeX.append(i)
                        listeY.append(j)
                        liste_coord.append((i,j))
            if np.max(listeX)<20 or np.min(listeX)>130:
                for j in regions[x][y].pixels:
                    j.value=cyan_color
                regions[x][y].statistics=regions[x][y].calculate_statistics()
            if np.max(listeY)<20 or np.min(listeY)>80:
                for j in regions[x][y].pixels:
                    j.value=cyan_color
                regions[x][y].statistics=regions[x][y].calculate_statistics()

    return regions



