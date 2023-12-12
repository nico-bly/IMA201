import math
import numpy as np
import copy
import cv2
import matplotlib.pyplot as plt
import skimage as sk
import scipy.signal

from scipy.spatial import cKDTree
from PIL import Image
from scipy import ndimage
from scipy import signal
from scipy.interpolate import interp1d
from skimage import io, morphology, filters, measure
from skimage.color import rgb2gray


def resize_and_pad(img, target_size):
    #Pour utiliser les memes hyper paramètres à toutes les images, on va les redimensionner
    # sans modifier les proportions car on fait du padding avec des 0
    #Input: img = image à red, target_size = (height,width)
    # target = ( height,width)
    aspect = img.shape[1] / float(img.shape[0])
    
    if aspect > 1:
        # Landscape orientation - wide image
        res = (target_size[1], int(target_size[1] / aspect))
    elif aspect < 1:
        # Portrait orientation - tall image
        res = (int(target_size[0] * aspect), target_size[0])
    else:
        # Square image
        res = target_size

    # Resize the image
    img_resized = cv2.resize(img, res)

    # Calculate padding
    pad_x = max(0, target_size[1] - img_resized.shape[1])
    pad_y = max(0, target_size[0] - img_resized.shape[0])

    # Calculate the padding values for top, bottom, left, and right
    top = pad_y // 2
    bottom = pad_y - top
    left = pad_x // 2
    right = pad_x - left

    # Apply padding with a constant value of 0
    img_padded = cv2.copyMakeBorder(img_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0] if len(img.shape) == 3 else [0])

    return img_padded


# On crée également une fonction qui permet d'améliorer le contraste des images

def histogram_stretching(img):
    #Input: image = image dont on va faire un etirement d'histogramme
    #Output: stretched_image = image après etirement d'histogramme

    #Code trouvé sur internet
# Split the image into color channels
    r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]

    hist_r = np.zeros(256)
    hist_g = np.zeros(256)
    hist_b = np.zeros(256)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            hist_r[r[i,j]] += 1
            hist_g[g[i,j]] += 1
            hist_b[b[i,j]] += 1

# Stretch the contrast for each channel
    min_r, max_r = np.min(r), np.max(r)
    min_g, max_g = np.min(g), np.max(g)
    min_b, max_b = np.min(b), np.max(b)

    re_stretch = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    gr_stretch = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    bl_stretch = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            re_stretch[i,j] = int((r[i,j] - min_r) * 255 / (max_r - min_r))
            gr_stretch[i,j] = int((g[i,j] - min_g) * 255 / (max_g - min_g))
            bl_stretch[i,j] = int((b[i,j] - min_b) * 255 / (max_b - min_b))

    # Merge the channels back together
    img_stretch = cv2.merge((re_stretch, gr_stretch, bl_stretch))

    return img_stretch
#########################################################################################################
# Creation d'un premier masqe autour de la tache centrale, pour appliquer la méthode Dull Razor de facon différente autour de la tache et sur le reste de l'image
# On va utiliser un masque binaire pour cela, avec des 1 pour la tache et des 0 pour le reste de l'image

def mask_lesion(image_original):
    #Input: image_original = image à traiter
    #Output: closest_center_mask = masque binaire de l'image, avec des 1 pour la tache et des 0 pour le reste de l'image
    image = rgb2gray(image_original)
    threshold = filters.threshold_otsu(image)
    first_mask = image > threshold
    # On inverse le masque binaire pour avoir des 1 pour la tache et des 0 pour le reste de l'image
    inverted = 1 - first_mask

    # Operation d'ouverture pour déconnecter les lignes connectées à la tache centrale
    opened = morphology.opening(inverted, morphology.disk(5))  # Adjust the size of the disk as needed

    # Chaque région connectée est étiquetée avec un numéro différent
    labels = measure.label(opened)

    # On prend les propriétés(area, centroid, bounding box) de chaque région étiquetée
    properties = measure.regionprops(labels)

    # centre de l'image
    center = np.array(opened.shape) / 2

    # Trouver les régions proches du centre de l'image, avec la distance euclidienne
    if properties:
        closest_center_region = min(
            properties,
            key=lambda prop: np.sqrt((prop.centroid[0] - center[0])**2 + (prop.centroid[1] - center[1])**2)
        )
    else:
        closest_center_region = None
        
    # On fait un masque avec les pixels de la région la plus proche du centre de l'image
    if closest_center_region is not None:
        closest_center_mask = labels == closest_center_region.label
    else:
        closest_center_mask = np.zeros_like(opened, dtype=bool)
    disk = morphology.disk(10)

    #Pour grossir cette région, on effectue une dilatation
    closest_center_mask_dilated = morphology.dilation(closest_center_mask, disk)
    
    return closest_center_mask_dilated


##############################################################################################################
# Création des éléments structurants
def create_matrices_diago(length, width):
    #Input: length = taille de l'élément structurant, width = largeur de l'élément structurant
    #Output: S45 = élément structurant diagonal

    S45 = np.zeros((length, length))
    for e in range(width):
        for l in range(length):
            if l-e >= 0:
                S45[l, l-e] = 1
    S45[0, 0] = 0
    S45[1,0] = 0
    S45[length-1, length-1] = 0

    return S45

def create_matrices_horizontal(length,width):
    #Input: length = taille de l'élément structurant, width = largeur de l'élément structurant
    #Output: S0 = élément structurant horizontal, une ligne de 1 au milieu

    S0 =  np.zeros((length, length))
    for e in range(width):
        S0[e+length//2, :] = 1
    return S0

def create_matrix_angle(length, angle_degrees):
    #Input: length = taille de l'élément structurant, angle_degrees = angle de l'élément structurant
    #Output: matrix = élément structurant diagonal, cela permet de créer des lignes de 1 d'une certaine taille
    # et d'un certain angle, cela marche bien pour 30°, 20°, 40° 

    matrix = np.zeros((length,length), dtype=int)
    slope = np.tan(np.radians(angle_degrees))
    # Loop through each row and column to determine if a pixel should be "on"
    for row in range(length):
        for col in range(length):
            # Calculate the corresponding column for the given row that aligns with the line
            aligned_col = int(slope * (row - length // 2) + length // 2)

        # Set the pixel to 1 if it's within a certain range of the aligned column
            if col == aligned_col or col == aligned_col + 1:
                matrix[row, col] = 1
    return matrix

##############################################################################################################

#Fonction maximum de plusieurs arrays
def maximum_arrays(List_array):
    #Input: List_array = liste d'arrays
    #Output: max_array = array maximum de la liste

    n = len(List_array)
    for e in range(n):
        if e == 0:
            max_array = List_array[e]
        else:
            max_array = np.maximum(max_array,List_array[e])
    return max_array

#Opération de fermeture telle que définie dans l'article

def Greyscale_closing_one_channel(Color_channel,element_structurant):
    # The first element bust be the diagonal element
    #Input: Color_channel = canal sur lequel on va faire l'opération de fermeture, element_structurant = liste des éléments structurants
    #Output: max_closing = image après opération de fermeture

    element_structurant_extended = element_structurant
    for e in range(1,len(element_structurant)):
        element_structurant_extended.append(np.transpose(element_structurant[e]))
  
    #Operation de fermeture avec chacun des canaux
    Closing_directions_array = []
    for e in range(len(element_structurant_extended)):
        Closing_directions_array.append(morphology.closing(Color_channel, element_structurant_extended[e]))
    
    #On prend le maximum des opérations de fermeture
    max_closing = maximum_arrays(Closing_directions_array)    

    return np.abs(Color_channel - max_closing)


#binarisation du mask, avec un certain seuil
def binary_mask(Greyscale_closed_picture,Threshold):
    #Input: Greyscale_closed_picture = image après opération de fermeture, Threshold = seuil de binarisation
    #Output: Binary_mask = masque binaire de l'image
    Binary_mask = np.zeros_like(Greyscale_closed_picture, dtype=np.uint8) 
    Binary_mask[Greyscale_closed_picture > Threshold] = 1
    return Binary_mask


##############################################################################################
#n est le nombre de colonnes et m le nombre de lignes
# i indexe les lignes et j indexe les colonnes
#Sachant que l'on va regarder si la distance en pixel maximale est moins que 30, on arrête la boucle while si cette distance est dépassée pour réduire les calculs
#Ces fonctions renvoient la longueur de la ligne, et les coordonnées où elle s'arrête pour être considérée comme de la peau ou lorsque la longueur atteint la valeur maximale que nous avons fixée
#Un dessin dans le notebook permet de visualiser les directions que nous avons choisi

max_length = 35

def ligne_n(Binary_mask, i, j):
    (m, n) = Binary_mask.shape[0:2]
    lgth = 0
    x = j
    y = i
    while y >= 0 and Binary_mask[y][j] == 0 and lgth < max_length:
        lgth += 1
        y -= 1
    return lgth , y , x

def ligne_ne(Binary_mask, i, j):
    (m, n) = Binary_mask.shape[0:2]
    lgth = 0
    y = i
    x = j
    while y >= 0 and x < n and Binary_mask[y][x] == 0  and lgth < max_length:
        lgth += np.sqrt(2)
        y -= 1
        x += 1
    return np.round(lgth) , y , x

def ligne_e(Binary_mask, i, j):
    (m, n) = Binary_mask.shape[0:2]
    lgth = 0
    x = j
    y = i
    while x < n and Binary_mask[i][x] == 0  and lgth < max_length:
        lgth += 1
        x += 1
    return lgth , y , x

def ligne_se(Binary_mask, i, j):
    (m, n) = Binary_mask.shape[0:2]
    lgth = 0
    y = i
    x = j
    while x < n and y < m and Binary_mask[y][x] == 0  and lgth < max_length:
        lgth += np.sqrt(2)
        y += 1
        x += 1
    return np.round(lgth), y, x

def ligne_s(Binary_mask, i, j):
    (m, n) = Binary_mask.shape[0:2]
    lgth = 0
    x = j
    y = i
    while y < m and Binary_mask[y][j] == 0  and lgth < max_length:
        lgth += 1
        y += 1
    return lgth, y , x

def ligne_so(Binary_mask, i, j):
    (m, n) = Binary_mask.shape[0:2]
    lgth = 0
    x = j
    y = i
    while y < m and x >= 0 and Binary_mask[y][x] == 0  and lgth < max_length:
        lgth += np.sqrt(2)
        x -= 1
        y += 1
    return np.round(lgth), y , x

def ligne_o(Binary_mask, i, j):
    (m, n) = Binary_mask.shape[0:2]
    lgth = 0
    x = j
    y = i
    while x >= 0 and Binary_mask[i][x] == 0  and lgth < max_length:
        lgth += 1
        x -= 1
    return lgth, y , x

def ligne_no(Binary_mask, i, j):
    (m, n) = Binary_mask.shape[0:2]
    lgth = 0
    x = j
    y = i
    while x >= 0 and y >= 0 and Binary_mask[y][x] == 0  and lgth < max_length:
        lgth += np.sqrt(2)
        y -= 1
        x -= 1
    return np.round(lgth), y , x

## on rajoute des lignes +-30° et +-60° par rapport aux axes Nord, Sud, Est, Ouest

def ligne_ne_30(Binary_mask, i, j):
    (m, n) = Binary_mask.shape[0:2]
    lgth = 0
    x = j
    y= i
    while y >= 0 and x < n and Binary_mask[y][x] == 0  and lgth < max_length:
        x +=2
        y -=1
        lgth += np.sqrt(5)

    return np.round(lgth), y , x

def ligne_ne_60(Binary_mask, i, j):
    (m, n) = Binary_mask.shape[0:2]
    lgth = 0
    x = j
    y= i
    while y >= 0 and x < n and Binary_mask[y][x] == 0  and lgth < max_length:
        x +=1
        y -=2
        lgth += np.sqrt(5)

    return np.round(lgth), y , x

def ligne_se_30(Binary_mask, i, j):
    (m, n) = Binary_mask.shape[0:2]
    lgth = 0
    x = j
    y= i
    while y < m and x < n and Binary_mask[y][x] == 0  and lgth < max_length:
        x +=2
        y +=1
        lgth += np.sqrt(5)
    
    return np.round(lgth), y , x

def ligne_se_60(Binary_mask, i, j):
    (m, n) = Binary_mask.shape[0:2]
    lgth = 0
    x = j
    y= i
    while y < m and x < n and Binary_mask[y][x] == 0  and lgth < max_length:
        x +=1
        y +=2
        lgth += np.sqrt(5)
    
    return np.round(lgth), y , x

def ligne_so_30(Binary_mask, i, j):
    (m, n) = Binary_mask.shape[0:2]
    lgth = 0
    x = j
    y= i
    while y < m and x >= 0 and Binary_mask[y][x] == 0  and lgth < max_length:
        x -=2
        y +=1
        lgth += np.sqrt(5)
    
    return np.round(lgth), y , x

def ligne_so_60(Binary_mask, i, j):
    (m, n) = Binary_mask.shape[0:2]
    lgth = 0
    x = j
    y= i
    while y < m and x >= 0 and Binary_mask[y][x] == 0  and lgth < max_length:
        x -=1
        y +=2
        lgth += np.sqrt(5)
    
    return np.round(lgth), y , x

def ligne_no_30(Binary_mask, i, j):
    (m, n) = Binary_mask.shape[0:2]
    lgth = 0
    x = j 
    y= i
    while y >= 0 and x >= 0 and Binary_mask[y][x] == 0  and lgth < max_length:
        x -=2
        y -=1
        lgth += np.sqrt(5)

    return np.round(lgth), y , x
    
def ligne_no_60(Binary_mask, i, j):
    (m, n) = Binary_mask.shape[0:2]
    lgth = 0
    x = j 
    y= i
    while y >= 0 and x >= 0 and Binary_mask[y][x] == 0  and lgth < max_length:
        x -=1
        y -=2
        lgth += np.sqrt(5)
    
    return np.round(lgth), y , x

## On ajoute des lignes  +-20° les axes Nord, Sud, Est, Ouest

def ligne_ne_20(Binary_mask, i, j):
    (m,n) = Binary_mask.shape[0:2]
    lgth = 0
    x = j
    y = i
    while y >= 0 and x < n and Binary_mask[y][x] == 0  and lgth < max_length:
        x += 4
        y -= 1
        lgth += np.sqrt(17)

    return np.round(lgth), y , x

def ligne_ne_80(Binary_mask, i, j):
    (m,n) = Binary_mask.shape[0:2]
    lgth = 0
    x = j
    y = i
    while y >= 0 and x < n and Binary_mask[y][x] == 0  and lgth < max_length:
        x += 1
        y -= 4
        lgth += np.sqrt(17)

    return np.round(lgth), y , x

def ligne_se_20(Binary_mask, i, j):
    (m,n) = Binary_mask.shape[0:2]
    lgth = 0
    x = j
    y = i
    while y < m and x < n and Binary_mask[y][x] == 0  and lgth < max_length:
        x += 4
        y += 1
        lgth += np.sqrt(17)

    return np.round(lgth), y , x

def ligne_se_80(Binary_mask, i, j):
    (m,n) = Binary_mask.shape[0:2]
    lgth = 0
    x = j
    y = i
    while y < m and x < n and Binary_mask[y][x] == 0  and lgth < max_length:
        x += 1
        y += 4
        lgth += np.sqrt(17)

    return np.round(lgth), y , x

def ligne_so_20(Binary_mask, i, j):
    (m,n) = Binary_mask.shape[0:2]
    lgth = 0
    x = j
    y = i
    while y < m and x >= 0 and Binary_mask[y][x] == 0  and lgth < max_length:
        x -= 4
        y += 1
        lgth += np.sqrt(17)

    return np.round(lgth), y , x

def ligne_so_80(Binary_mask, i, j):
    (m,n) = Binary_mask.shape[0:2]
    lgth = 0
    x = j
    y = i
    while y < m and x >= 0 and Binary_mask[y][x] == 0  and lgth < max_length:
        x -= 1
        y += 4
        lgth += np.sqrt(17)

    return np.round(lgth), y , x

def ligne_no_20(Binary_mask, i, j):
    (m,n) = Binary_mask.shape[0:2]
    lgth = 0
    x = j
    y = i
    while y >= 0 and x >= 0 and Binary_mask[y][x] == 0  and lgth < max_length:
        x -= 4
        y -= 1
        lgth += np.sqrt(17)

    return np.round(lgth), y , x

def ligne_no_80(Binary_mask, i, j):
    (m,n) = Binary_mask.shape[0:2]
    lgth = 0
    x = j
    y = i
    while y >= 0 and x >= 0 and Binary_mask[y][x] == 0  and lgth < max_length:
        x -= 1
        y -= 4
        lgth += np.sqrt(17)

    return np.round(lgth), y , x


#vérification des pixels

def hair_pixel_verification_distances(Binary_mask_original,lesion_mask,seuil_hair_max,nombres_directions_peau,taille_max_poil_autre):
    '''
    Input: Binary_mask_original : masque binaire dans lequel les pixels des cheveux sont marqués comme étant à 0.
    seuil_hair_max : La longueur maximale d'un cheveu pour être considéré comme un cheveu
    nombres_directions_peau : nombre minimal de directions dont la longueur doit être inférieure à « taille_max_poil_autre » pour qu'un pixel soit considéré comme une peau
    taille_max_poil_autre : longueur maximale pour qu'une direction soit considérée comme de la peau 

    Output: Binary_mask : masque binaire mis à jour dans lequel les pixels des cheveux sont marqués comme 0 et les pixels de la peau sont marqués comme 1 
    distances : Un tableau 4D où distances[i,j,k, :] contient la longueur de la ligne dans la direction k à partir du pixel (i,j), et les coordonnées où elle s'arrête pour être considérée comme un cheveu ou lorsque la longueur atteint la valeur maximale que nous avons fixée 
    '''
    Binary_mask = copy.deepcopy(Binary_mask_original)
    m, n = Binary_mask.shape[0:2]
    # Create an array to store distances
    distances = np.zeros((m, n, 24, 3), dtype=int)
    #The element distances[i,j,k,:] contains the length of the line in the direction k, and the coordinates where it stops to be considered a hair or when the length reaches the maximum value we fixed
    for i in range(m):
        for j in range(n):
            if Binary_mask[i][j]==0:
                # Calculate distances for all 8 directions and store them in 'distances' array
                distances[i, j, 0,:] = ligne_n(Binary_mask, i, j)
                distances[i, j, 1,:] = ligne_ne(Binary_mask, i, j)
                distances[i, j, 2,:] = ligne_e(Binary_mask, i, j)
                distances[i, j, 3,:] = ligne_se(Binary_mask, i, j)
                distances[i, j, 4,:] = ligne_s(Binary_mask, i, j)
                distances[i, j, 5,:] = ligne_so(Binary_mask, i, j)
                distances[i, j, 6,:] = ligne_o(Binary_mask, i, j)
                distances[i, j, 7,:] = ligne_no(Binary_mask, i, j)
                distances[i, j, 8,:] = ligne_ne_30(Binary_mask, i, j)
                distances[i, j, 9,:] = ligne_ne_60(Binary_mask, i, j)
                distances[i, j, 10,:] = ligne_se_30(Binary_mask, i, j)
                distances[i, j, 11,:] = ligne_se_60(Binary_mask, i, j)
                distances[i, j, 12,:] = ligne_so_30(Binary_mask, i, j)
                distances[i, j, 13,:] = ligne_so_60(Binary_mask, i, j)
                distances[i, j, 14,:] = ligne_no_30(Binary_mask, i, j)
                distances[i, j, 15,:] = ligne_no_60(Binary_mask, i, j)
                distances[i, j, 16,:] = ligne_ne_20(Binary_mask, i, j)
                distances[i, j, 17,:] = ligne_ne_80(Binary_mask, i, j)
                distances[i, j, 18,:] = ligne_se_20(Binary_mask, i, j)
                distances[i, j, 19,:] = ligne_se_80(Binary_mask, i, j)
                distances[i, j, 20,:] = ligne_so_20(Binary_mask, i, j)
                distances[i, j, 21,:] = ligne_so_80(Binary_mask, i, j)
                distances[i, j, 22,:] = ligne_no_20(Binary_mask, i, j)
                distances[i, j, 23,:] = ligne_no_80(Binary_mask, i, j)

    # Pour les pixels valant 1 dans le lesion_mask, on effectue la vérification des pixels avec la méthode hair_pixel_verification
    # Pour les autre, on garde le masque binaire tel quel pour enlever au mieux les poils en dehors de la tache
  
    for i in range(m):
        for j in range(n):
            # On ne modifie que les pixels qui sont dans la tache centrale
            if Binary_mask[i][j] == 0 and lesion_mask[i][j] == 1:
                max_length = np.max(distances[i, j,:,0])
                other_lengths = distances[i, j, distances[i, j,:,0] < taille_max_poil_autre]
                if max_length > seuil_hair_max and len(other_lengths) >= nombres_directions_peau :
                    Binary_mask[i][j] = 0
                else:
                    Binary_mask[i][j] = 1
            #Pour les autres, on utilise des critères moins stricts
            elif Binary_mask[i][j] == 0 and lesion_mask[i][j] == 0:
                max_length = np.max(distances[i, j,:,0])
                other_lengths = distances[i, j, distances[i, j,:,0] < taille_max_poil_autre*2]
                if max_length > seuil_hair_max//2 and len(other_lengths) >= nombres_directions_peau-3:
                    Binary_mask[i][j] = 0
                else:
                    Binary_mask[i][j] = 1

    return Binary_mask, distances
 

##  Cette fonction reprend les fonctions définies au dessus  et permet d'avoir le masque binaire vérifié
def hair_detection(image, element_structurant, seuil_binarisation, Parameters_hair_verification):
    #Input: image = image à traiter, element_structurant = liste des éléments structurants, seuil_binarisation = seuil de binarisation
    # Parameters_hair_verification = liste de 3 paramètres: seuil_hair_max,nombres_directions_peau,taille_max_poil_autre
    #Output: Binary_mask = masque binaire de l'image, Binary_mask_verified = masque binaire vérifié et distances, le tableau tel que définit dans la fonction du dessus
    max_length_hair = Parameters_hair_verification[0]
    nombres_directions_peau = Parameters_hair_verification[1]
    taille_max_poil_autre = Parameters_hair_verification[2]

    # Split the image into its RGB channels
    red_channel = image[:, :, 0]
    green_channel = image[:, :, 1]
    blue_channel = image[:, :, 2]

    # Apply grayscale closing to each channel
    Gr = Greyscale_closing_one_channel(red_channel, element_structurant)
    Gv = Greyscale_closing_one_channel(green_channel, element_structurant)
    Gb = Greyscale_closing_one_channel(blue_channel, element_structurant)

    # Binarize the masks
    Binary_mask_red = binary_mask(Gr, seuil_binarisation)
    Binary_mask_green = binary_mask(Gv, seuil_binarisation)
    Binary_mask_blue = binary_mask(Gb, seuil_binarisation)

    # Combine the masks
    Binary_mask = np.logical_or.reduce([Binary_mask_red, Binary_mask_green, Binary_mask_blue]).astype(np.uint8)
    (m, n) = Binary_mask.shape[0:2]

    lesion_mask = mask_lesion(image)

    # Hair pixel verification pour les pixels du masque de la lésion
    Binary_mask_verified, distances = hair_pixel_verification_distances(Binary_mask,lesion_mask,max_length_hair,nombres_directions_peau,taille_max_poil_autre)
    
    return Binary_mask, Binary_mask_verified, distances


# Grace aux distances calculées avec hair_detection, peut utiliser ces valeurs pour retrouver les pixels blancs les plus proches dans le masque binaire
# et les remplacer par la moyenne des pixels non cheveux les plus proches
def find_nearest_non_hair_pixels(binary_mask_verified, distances, i, j):
    #Input: binary_mask_verified = masque binaire vérifié, distances = tableau tel que définit dans la fonction du dessus, i = ligne, j = colonne
    #Output: min1_coords = coordonnées du pixel non cheveux le plus proche, min2_coords = coordonnées du deuxième pixel non cheveux le plus proche  

    #Si le pixel que l'on étudie n'est pas noir, cela n'est pas normal, on renvoie (-1,-1) car cette fonction ne va être appelée que si 
    # le pixel est noir normalement
    if binary_mask_verified[i, j] != 0:
        return (-1, -1)

    min1_index = np.argmin(distances[i, j, :, 0])
    min1_coords = (distances[i, j, min1_index, 1], distances[i, j, min1_index, 2])

    distances_without_min1 = distances[i, j, distances[i, j, :, 0] != distances[i, j, min1_index, 0], :]
    if distances_without_min1.size == 0:
        return (min1_coords, min1_coords)

    min2_index = np.argmin(distances_without_min1[:, 0])
    min2_coords = (distances_without_min1[min2_index, 1], distances_without_min1[min2_index, 2])
    
    return (min1_coords, min2_coords)

# Fonction qui trouve le pixel noir le plus proche dans le masque binaire vérifié dilaté trouvée sur Internet
def find_nearest_black_pixel(B1, B2, position_B1):
    # Find the coordinates of black pixels in B2
    black_pixels_B2 = np.argwhere(B2 == 0)

    # Build a k-d tree from black_pixels_B2
    kdtree = cKDTree(black_pixels_B2)

    # Find the nearest black pixel in B2 to the given position in B1
    distance, nearest_index = kdtree.query(position_B1)

    # Get the coordinates of the nearest pixel in B2
    nearest_pixel_B2 = black_pixels_B2[nearest_index]

    return nearest_pixel_B2

# Fonction qui remplace les pixels de cheveux par la moyenne des pixels non cheveux les plus proches
# Pour cela on parcoure le masque binaire vérifié dilaté, si un pixel noir de ce masque est également noir dans le masque binaire vérifié, alors on peut retrouver les plus proches voisins grace au tableau distances
# Si le pixel correspondant dans le masque binaire vérifié est blanc, alors on cherche le pixel noir le plus proche dans le masque binaire vérifié. Le pixel aura la couleur de peau attrbuée à ce pixel noir le plus proche

def hair_replacement_dilated(image, Binary_mask_verified, Binary_mask_verified_dilated, distances):
    #Input: image = image à traiter, Binary_mask_verified = masque binaire vérifié, Binary_mask_verified_dilated = masque binaire vérifié dilaté, distances = tableau tel que définit dans une fonction au dessus
    #Output: new_image = image avec les pixels de cheveux remplacés par la moyenne des pixels non cheveux les plus proches

    new_image = copy.deepcopy(image)
    m, n = Binary_mask_verified.shape[0:2]
    for i in range(m):
        for j in range(n):
            if Binary_mask_verified_dilated[i][j] == 0:
                
                if Binary_mask_verified[i][j] == 0:

                    # Find the nearest non-hair pixels
                    min1, min2 = find_nearest_non_hair_pixels(Binary_mask_verified,distances, i, j)
                    # Check if there are at least two non-hair pixels
                    condition1 = min1[0] <300 and min1[1] <400
                    condition2 = min2[0] <300 and min2[1] <400
                    if min1 != (-1, -1) and min2 != (-1, -1) and condition1 and condition2:
                        #Taking the average of both
                        new_image[i][j][0] = np.mean([image[min1[0]][min1[1]][0], image[min2[0]][min2[1]][0]])
                        new_image[i][j][1] = np.mean([image[min1[0]][min1[1]][1], image[min2[0]][min2[1]][1]])
                        new_image[i][j][2] = np.mean([image[min1[0]][min1[1]][2], image[min2[0]][min2[1]][2]])

                else:
                    nearest_hair = find_nearest_black_pixel(Binary_mask_verified_dilated, Binary_mask_verified, (i, j))
                    i_nearest_hair = nearest_hair[0]
                    j_nearest_hair = nearest_hair[1]
                    min1, min2 = find_nearest_non_hair_pixels(Binary_mask_verified,distances, i_nearest_hair, j_nearest_hair)

                    condition1 = min1[0] <300 and min1[1] <400
                    condition2 = min2[0] <300 and min2[1] <400
                    if min1 != (-1, -1) and min2 != (-1, -1) and condition1 and condition2:
                        #Taking the average of both
                        new_image[i][j][0] = np.mean([image[min1[0]][min1[1]][0], image[min2[0]][min2[1]][0]])
                        new_image[i][j][1] = np.mean([image[min1[0]][min1[1]][1], image[min2[0]][min2[1]][1]])
                        new_image[i][j][2] = np.mean([image[min1[0]][min1[1]][2], image[min2[0]][min2[1]][2]])

    return new_image
##############################################################################################################