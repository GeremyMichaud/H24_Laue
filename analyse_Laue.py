import os
import glob
import cv2
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.signal as sc
from scipy.optimize import curve_fit

palette = sns.color_palette("colorblind")

def process_image(image_bg, thresh_factor):
    """
    Processus de traitement d'une image pour détecter les contours 
    dans la densité spectrale.
    
    Args:
        image: np.ndarray, l'image en niveaux de gris
        thresh_factor: float, facteur de seuillage pour déterminer 
        le seuil de binarisation
        
    Returns:
        list, une liste de contours des régions d'intérêt fermées
    """
    img8 = np.clip((image_bg/65535.0) * 255.0, 0, 255).astype(np.uint8)

    threshold = thresh_factor * np.max(img8)
    regions_of_interest = (img8 > threshold).astype(np.uint8)

    kernel = np.ones((5, 5), np.uint8)
    regions_of_interest_closed = cv2.morphologyEx(regions_of_interest,
                                                cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(regions_of_interest_closed,
                                cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    filtered_contours = []
    for contour in contours:
        _, _, w, h = cv2.boundingRect(contour)
        if w > 10 and h > 10:
            filtered_contours.append(contour)

    return filtered_contours

def draw_all_contours(image, image_bg, name, factor=0.02):
    """
    Dessine les contours des régions d'intérêt sur l'image de densité spectrale.
    
    Args:
        image: np.ndarray, l'image en niveaux de gris
        name: string,  nom du fichier à sauvegarder
        factor: float, facteur de seuillage pour déterminer le seuil de binarisation
    """
    cnt = process_image(image_bg, factor)
    image = np.clip((image/65535.0) * 255.0, 0, 255).astype(np.uint8)
    colored_spectrum = cv2.cvtColor(image.astype(np.uint8)
                                    , cv2.COLOR_GRAY2BGR)

    for i, contour in enumerate(cnt):
        cv2.drawContours(colored_spectrum, [contour], -1, (0, 0, 255), 2)
        contour_center = contour.mean(axis=0).astype(int)[0]
        cv2.putText(colored_spectrum, str(i+1), tuple(contour_center),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 2)

    directory = os.path.join("output", "01_all_contours")
    if not os.path.exists(directory):
            os.makedirs(directory)

    cv2.imwrite(os.path.join(directory, name + ".png"), colored_spectrum)

def remove_contours(image_bg, contours_to_remove, factor=0.02):
    """
    Remove specified contours from the list.
    
    Args:
        image: np.ndarray, the grayscale image
        contours_to_remove: list, list of contour indices to remove
        factor: float, thresholding factor to determine the binarization threshold
    """
    bad_cnt = process_image(image_bg, factor)
    good_cnt = [contour for i, contour in enumerate(bad_cnt)
                if i+1 not in contours_to_remove]

    return good_cnt

def draw_good_contours(image, name, contours):
    """
    Draw only good contours on an image and save it.

    Args:
        image (np.ndarray): Input grayscale image.
        name (str): Name of the output image file.
        contours (list): List of contours to draw on the image.
    """
    image = np.clip((image/65535.0) * 255.0, 0, 255).astype(np.uint8)
    colored_spectrum = cv2.cvtColor(image.astype(np.uint8)
                                    , cv2.COLOR_GRAY2BGR)

    for i, contour in enumerate(contours):
        cv2.drawContours(colored_spectrum, [contour], -1, (0, 0, 255), 2)
        contour_center = contour.mean(axis=0).astype(int)[0]
        cv2.putText(colored_spectrum, str(i+1), tuple(contour_center)
                    , cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 2)

    directory = os.path.join("output", "02_contours_removed")
    os.makedirs(directory, exist_ok=True)
    cv2.imwrite(os.path.join(directory, name + ".png"), colored_spectrum)

def gaussian(x, a, x0, sigma):
    """
    Gaussian function.
    
    Args:
        x (array-like): Input data.
        a (float): Amplitude of the Gaussian curve.
        x0 (float): Mean of the Gaussian curve.
        sigma (float): Standard deviation of the Gaussian curve.
        
    Returns:
        array-like: Values of the Gaussian function evaluated at x.
    """
    return a * np.exp(-(x - x0)**2 / (2 * sigma**2))

def find_contour_centroids(contours, image):
    """
    Find centroids of contours in an image.

    Args:
        contours (list): List of contours.
        image (np.ndarray): Grayscale image.

    Returns:
        tuple: A tuple containing two lists. The first list contains
        tuples of centroid coordinates (x, y) for each contour.
            The second list contains tuples of centroid uncertainties
            (x_uncertainty, y_uncertainty) for each contour.
    """
    centroids = []
    centroids_uncertainty = []
    for i, contour in enumerate(contours):
        x_coords = contour[:, 0, 0]
        y_coords = contour[:, 0, 1]

        min_x_coord = np.min(x_coords)
        max_x_coord = np.max(x_coords)
        min_y_coord = np.min(y_coords)
        max_y_coord = np.max(y_coords)

        pixel_values = image[min_y_coord:max_y_coord, min_x_coord:max_x_coord]

        x_mean_intensity = np.mean(pixel_values, axis=0)
        y_mean_intensity = np.mean(pixel_values, axis=1)

        x_data = np.arange(len(x_mean_intensity))
        y_data = np.arange(len(y_mean_intensity))
        try:
            popt_x, cov_x = curve_fit(gaussian, x_data, x_mean_intensity, 
                                    p0=[np.max(x_mean_intensity), len(x_data) / 2, 10])
            popt_y, cov_y = curve_fit(gaussian, y_data, y_mean_intensity, 
                                    p0=[np.max(y_mean_intensity), len(y_data) / 2, 10])

            x_center = int(popt_x[1])
            x_center_pix = min_x_coord + x_center
            x_center_uncertainty = np.sqrt(np.diag(cov_x)[1])
            y_center = int(popt_y[1])
            y_center_pix = min_y_coord + y_center
            y_center_uncertainty = np.sqrt(np.diag(cov_y)[1])

            centroids.append((x_center_pix, y_center_pix))
            centroids_uncertainty.append((x_center_uncertainty, y_center_uncertainty))

        except RuntimeError:
            x_center = int(len(x_coords) // 2)
            x_center_pix = min_x_coord + x_center
            x_center_uncertainty = np.sqrt(len(x_coords)/2)
            y_center = int(len(y_coords) // 2)
            y_center_pix = min_y_coord + y_center
            y_center_uncertainty = np.sqrt(len(y_coords)/2)

            centroids.append((x_center_pix, y_center_pix))
            centroids_uncertainty.append((x_center_uncertainty, y_center_uncertainty))

            print(f"Error occured during curve fitting for contour {i+1}. Skipping contour.")
            continue

    return centroids, centroids_uncertainty

def draw_point(image, name, centroids):
    """
    Draws points at the centroids in the image and saves it as an image 
    file with the specified name.
    
    Parameters:
    image (numpy.ndarray): The image to draw on.
    name (str): The base name of the output image file.
    centroids (list): The list of centroids to draw.
    """
    # Create a copy of the image to draw on
    rescaled_spectrum = cv2.normalize(np.log(image), None, 0, 255, 
                                    cv2.NORM_MINMAX)
    colored_spectrum = cv2.cvtColor(rescaled_spectrum.astype(np.uint8), 
                                    cv2.COLOR_GRAY2BGR)

    # Iterate over the centroids and draw dots and numerotation for each one
    for i, centroid in enumerate(centroids):
        # Convert the centroid coordinates to integers
        x, y = map(int, centroid)

        # Draw a dot at the centroid
        cv2.circle(colored_spectrum, (x, y), 5, (255, 255, 255), -1)

        # Define the font and size for the numerotation
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1

        # Draw the numerotation on the image
        cv2.putText(colored_spectrum, str(i+1), (x, y), font, font_scale,
                    (0, 0, 0), font_thickness, cv2.LINE_AA)
    
    out_dir = os.path.join("output", "08_miller")
    os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(os.path.join(out_dir, name + ".png"), colored_spectrum)

def find_pairs(image, centroids):
    """
    Trouve les paires de centroïdes symétriques.


    Args:
        image (np.ndarray): L'image.
        centroids (list): Liste des coordonnées des centroïdes à rechercher.

    Returns:
        list: Liste de paires de centroïdes symétriques.
    """
    pos_pairs = []
    centroids2match = centroids.copy()

    for c in centroids2match:
        y, x = c

        center_y = int(image.shape[0] / 2) - 110 <= y <= int(image.shape[0] / 2) + 110
        center_x = int(image.shape[1] / 2) - 110 <= x <= int(image.shape[1] / 2) + 110
        if center_y and center_x:
            center = (y, x)
            centroids2match.remove((y, x))
            break

    for c in centroids2match:
        y, x = c
    
        y_prime = 2 * (center[0] - y) + y
        x_prime = 2 * (center[1] - x) + x

        for c_prime in centroids2match:
            y_prime_centroid, x_prime_centroid = c_prime
            if abs(y_prime_centroid - y_prime) <= 50 and abs(x_prime_centroid
                                                            - x_prime) <= 50:
                pos_pairs.append((c, c_prime))
                centroids2match.remove(c_prime)
                break

    return pos_pairs

def draw_pairs(image, name, pairs):
    """
    Draw pairs of points.

    Args:
        image (np.ndarray): Input grayscale image.
        name (str): Name of the output image file.
        pairs (list): List of pairs of points to draw on the image.
    """
    image = np.clip((image/65535.0) * 255.0, 0, 255).astype(np.uint8)
    colored_spectrum = cv2.cvtColor(image.astype(np.uint8),
                                    cv2.COLOR_GRAY2BGR)

    for i, (start_point, end_point) in enumerate(pairs):
        color = tuple(np.random.randint(0, 256, 3).tolist())
        cv2.line(colored_spectrum, start_point, end_point, color,
                1, cv2.LINE_AA)
        for point in [start_point, end_point]:
            cv2.putText(colored_spectrum, str(i + 1), point,
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    out_dir = os.path.join("output", "03_pairs")
    os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(os.path.join(out_dir, name + ".png"), colored_spectrum)

def print_d_spacings(d_spacings, d_spacings_uncert, d_spacings_res,
                    d_spacings_res_uncert, name):
    """
    Print d-spacings and uncertainties.

    Args:
        d_spacings (list): List of mean d-spacings.
        d_spacings_uncert (list): List of uncertainties in mean d-spacings (in pixels).
        d_spacings_res (list): List of mean d-spacings in picometers.
        d_spacings_res_uncert (list): List of uncertainties in mean d-spacings (in picometers).
        name (str): Name of the sample or dataset.
    """
    print("\n\tTable of D-Spacing of {}".format(name))
    print("Index |  D-Spacing (pixels)  |  D-Spacing (picometers)")
    print("------------------------------------------------------")
    for i, (d, uncert, d_res, res_uncert) in enumerate(zip(d_spacings, 
            d_spacings_uncert, d_spacings_res, d_spacings_res_uncert), 1):
        spacing_str = f"{d:.3f}"
        incertitude_str = f"{uncert:.3f}"
        spacing_res_str = f"{d_res:.3f}"
        incertitude_res_str = f"{res_uncert:.3f}"
        index_str = str(i).rjust(5, "0")
        print(f"{index_str} |     {spacing_str} ± {incertitude_str}    |     {spacing_res_str} ± {incertitude_res_str}")

if __name__ == "__main__":
    images_path = glob.glob(os.path.join("data", "recontrasted", "*png"))
    images_dic = {os.path.splitext(os.path.basename(chemin_image))[0]: cv2.imread(chemin_image, cv2.IMREAD_UNCHANGED) for chemin_image in images_path}

    images_bg_path = glob.glob(os.path.join("data", "bgless", "*png"))
    images_bg_dic = {os.path.splitext(os.path.basename(chemin_image))[0]: cv2.imread(chemin_image, cv2.IMREAD_UNCHANGED) for chemin_image in images_bg_path}

    lif_name, nacl_name, si_name = images_dic.keys()
    lif_img, nacl_img, si_img = images_dic.values()
    lif_bg_img, nacl_bg_img, si_bg_img = images_bg_dic.values()

    draw_all_contours(lif_img, lif_bg_img, lif_name, factor=0.31)
    draw_all_contours(nacl_img, nacl_bg_img, nacl_name, factor=0.05)
    draw_all_contours(si_img, si_bg_img, si_name, factor=0.07)

    contour2remove_lif = [2, 7, 10, 13, 14, 18, 19, 22, 23, 25, 27, 32]
    contour2remove_nacl = [17]
    contour2remove_si = [13, 14, 17, 26, 27, 33, 34, 35, 38, 47]
    contours_lif = remove_contours(lif_bg_img, contour2remove_lif, factor=0.31)
    contours_nacl = remove_contours(nacl_bg_img, contour2remove_nacl, factor=0.05)
    contours_si = remove_contours(si_bg_img, contour2remove_si, factor=0.07)
    draw_good_contours(lif_img, lif_name, contours_lif)
    draw_good_contours(nacl_img, nacl_name, contours_nacl)
    draw_good_contours(si_img, si_name, contours_si)

    centroids_lif, uncert_lif = find_contour_centroids(contours_lif, lif_img)
    centroids_nacl, uncert_nacl = find_contour_centroids(contours_nacl, nacl_img)
    centroids_si, uncert_si = find_contour_centroids(contours_si, si_img)

    pos_pairs_lif = find_pairs(lif_img, centroids_lif)
    pos_pairs_nacl = find_pairs(nacl_img, centroids_nacl)
    pos_pairs_si = find_pairs(si_img, centroids_si)
    draw_pairs(lif_img, lif_name, pos_pairs_lif)
    draw_pairs(nacl_img, nacl_name, pos_pairs_nacl)
    draw_pairs(si_img, si_name, pos_pairs_si)

    """pix_5, pm_5 = calculate_d_spacings(signals_5)
    pix_6, pm_6 = calculate_d_spacings(signals_6)
    pix_7, pm_7 = calculate_d_spacings(signals_7)
    print_d_spacings(pix_5[0], pix_5[1], pm_5[0], pm_5[1], image_5_name)
    print_d_spacings(pix_6[0], pix_6[1], pm_6[0], pm_6[1], image_6_name)
    print_d_spacings(pix_7[0], pix_7[1], pm_7[0], pm_7[1], image_7_name)"""
