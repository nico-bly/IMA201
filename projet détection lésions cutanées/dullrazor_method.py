import functions_IMA as functions
import cv2
from skimage import morphology, filters
import numpy as np

def dullrazor(image):
    #Input: image à laquelle on va potentiellement appliquer le dullrazor
    #Output: 
    # filtered_image avec le dullrazor appliqué si il y a des poils dans l'image, sinon l'image originale
    # Binary_mask: masque binaire de l'image originale
    # binary_mask_verified_dilated: masque binaire de l'image originale ayant subi une vérification des poils

    # Changement de taille
    target_size = (300,400)
    ima_std = functions.resize_and_pad(image, target_size)

    # Etirement d'histogramme
    ima_std = functions.histogram_stretching(ima_std)
    
    S20_0 = functions.create_matrices_horizontal(10, 2)
    S20_45 = functions.create_matrices_diago(10, 2)
    S20_30 = functions.create_matrix_angle(10, 30)
    S20_minus20 = functions.create_matrix_angle(10, -20)
    S20_20 = functions.create_matrix_angle(10, 20)
    S20_minus40 = functions.create_matrix_angle(10, -40)
    S20_minus20 = functions.create_matrix_angle(10, -20)
    S5_0 = functions.create_matrices_horizontal(5, 1)
    S5_45 = functions.create_matrices_diago(5, 1)

    taille_grand_poil = 17
    nombres_directions_min = 20
    taille_max_autres_poils = 10
    Parametres_verfifications_poils = [taille_grand_poil, nombres_directions_min, taille_max_autres_poils]

    elements_structurants = [S20_45, S20_0,  S20_30,S5_0,S5_45]

    Binary_mask, Binary_mask_verified, distances = functions.hair_detection(ima_std, elements_structurants, 240, Parametres_verfifications_poils)    
    
    disk = morphology.disk(1)
    binary_mask_verified_dilated = cv2.erode(Binary_mask_verified,kernel= disk,iterations= 1)
      
    Binary_count = np.count_nonzero(Binary_mask ==0)+1
    Binary_verified_count = np.count_nonzero(Binary_mask_verified ==0)

    # On effectue le remplacement des pixels si on a assez de pixels noirs dans le masque vérifié
    # par rapport au nombre de pixels noirs dans le masque originel
    # Si le rapport est plus petit qu'un certain seuil, on effectue pas les changements car cela veut dire
    # qu'il n'y pas de poils dans l'image
   
    final_image =  functions.hair_replacement_dilated(ima_std,Binary_mask_verified,binary_mask_verified_dilated,distances)
    filtered_image = cv2.medianBlur(final_image, 1)

    return filtered_image, Binary_mask, binary_mask_verified_dilated 