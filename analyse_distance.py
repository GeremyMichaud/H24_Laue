import os
import glob
import cv2
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import linregress
import matplotlib.pyplot as plt
import openpyxl
from openpyxl.utils.dataframe import dataframe_to_rows
import seaborn as sns


palette = sns.color_palette("bright")

def process_image(image, thresh_factor):
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
    img8 = image.astype(np.uint8)

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

def draw_all_contours(image, name, factor=0.02):
    """
    Dessine les contours des régions d'intérêt sur l'image de densité spectrale.
    
    Args:
        image: np.ndarray, l'image en niveaux de gris
        name: string,  nom du fichier à sauvegarder
        factor: float, facteur de seuillage pour déterminer le seuil de binarisation
    """
    cnt = process_image(image, factor)
    image = image.astype(np.uint8)
    colored_spectrum = cv2.cvtColor(image.astype(np.uint8)
                                    , cv2.COLOR_GRAY2BGR)

    for i, contour in enumerate(cnt):
        cv2.drawContours(colored_spectrum, [contour], -1, (0, 0, 255), 2)
        contour_center = contour.mean(axis=0).astype(int)[0]
        cv2.putText(colored_spectrum, str(i+1), tuple(contour_center),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 2)

    directory = os.path.join("output", "11_distance", "01_all_contours")
    if not os.path.exists(directory):
            os.makedirs(directory)

    cv2.imwrite(os.path.join(directory, name + ".png"), colored_spectrum)

def remove_contours(image, good_contours, factor=0.02):
    """
    Remove specified contours from the list.
    
    Args:
        image: np.ndarray, the grayscale image
        contours_to_remove: list, list of contour indices to remove
        factor: float, thresholding factor to determine the binarization threshold
    """
    bad_cnt = process_image(image, factor)
    good_cnt = [contour for i, contour in enumerate(bad_cnt)
                if i+1 in good_contours]

    return good_cnt

def draw_good_contours(image, name, contours):
    """
    Draw only good contours on an image and save it.

    Args:
        image (np.ndarray): Input grayscale image.
        name (str): Name of the output image file.
        contours (list): List of contours to draw on the image.
    """
    image = image.astype(np.uint8)
    colored_spectrum = cv2.cvtColor(image.astype(np.uint8)
                                    , cv2.COLOR_GRAY2BGR)

    for i, contour in enumerate(contours):
        cv2.drawContours(colored_spectrum, [contour], -1, (0, 0, 255), 2)
        contour_center = contour.mean(axis=0).astype(int)[0]
        cv2.putText(colored_spectrum, str(i+1), tuple(contour_center)
                    , cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 2)

    directory = os.path.join("output", "11_distance", "02_contours_removed")
    os.makedirs(directory, exist_ok=True)
    cv2.imwrite(os.path.join(directory, name + ".png"), colored_spectrum)

def draw_points(image, name, contours):
    """
    Draws points at the centroids in the image and saves it as an image 
    file with the specified name.
    
    Parameters:
    image (numpy.ndarray): The image to draw on.
    name (str): The base name of the output image file.
    centroids (list): The list of centroids to draw.
    """
    image = image.astype(np.uint8)
    colored_spectrum = cv2.cvtColor(image.astype(np.uint8), 
                                    cv2.COLOR_GRAY2BGR)

    for i, contour in enumerate(contours):
        M = cv2.moments(contour)

        # Calculate the center of the contour
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

        cv2.circle(colored_spectrum, (cx, cy), 10, (0, 0, 255), -1)

        text = str(i+1)
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]

        text_x = cx - text_size[0] // 2
        text_y = cy + 40

        cv2.putText(colored_spectrum, text, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    out_dir = os.path.join("output", "11_distance", "03_points")
    os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(os.path.join(out_dir, name + ".png"), colored_spectrum)

def intensity(image, contours):
    image = image.astype(np.uint8)

    mean_contour = []

    for contour in contours:
        # Create an empty mask image
        mask = np.zeros_like(image)

        # Draw the contour on the mask
        cv2.drawContours(mask, [contour], -1, (255), -1)

        # Extract the region of interest using the mask
        roi = cv2.bitwise_and(image, image, mask=mask)

        # Calculate the mean intensity of pixels in the extracted region
        mean_intensity = np.mean(roi[mask != 0])
        area = cv2.contourArea(contour)

        mean_contour.append(mean_intensity / area)

    return np.mean(mean_contour), np.std(mean_contour)

def distance2center(image, contours):
    resolution = 49.5E-6
    center_x = len(image[0]) // 2
    center_y = len(image) // 2

    xQs = []
    yQs = []

    for contour in contours:
        M = cv2.moments(contour)

        # Calculate the center of the contour
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        
        xQ = np.abs(cx - center_x) * resolution
        yQ = np.abs(cy - center_y) * resolution

        xQs.append(xQ)
        yQs.append(yQ)

    xQ_mean = np.mean(xQs)
    xQ_err = np.std(xQs)
    yQ_mean = np.mean(yQs)
    yQ_err = np.std(xQs)

    distance = np.sqrt(xQ_mean**2 + yQ_mean**2) * 1000
    distance_err = np.sqrt((xQ_mean / np.sqrt(xQ_mean**2 + yQ_mean**2) * xQ_err)**2 + (yQ_mean / np.sqrt(xQ_mean**2 + yQ_mean**2) * yQ_err)**2) * 1000

    return distance, distance_err

def model_r2(x, fct, intercept):
    return fct / (x ** 2) + intercept

def plot_intensity(dict_intensity):
    x_values = np.array(list(dict_intensity.keys()), dtype=float)
    x_err = 1.
    y_values = np.array([value[0] for value in dict_intensity.values()])
    y_err = np.array([value[1] for value in dict_intensity.values()])

    popt, _ = curve_fit(model_r2, x_values, y_values)

    # Calculate R^2 coefficient
    residuals = y_values - model_r2(x_values, *popt)
    TSS = np.sum((y_values - np.mean(y_values)) ** 2)
    RSS = np.sum(residuals ** 2)
    R_squared = 1 - (RSS / TSS)

    x_fit = np.linspace(10, 35, 1000)

    plt.errorbar(x_values, y_values, xerr=x_err, yerr=y_err, color=palette[2], fmt="o", capsize=2, label="Données")

    plt.plot(x_fit, model_r2(x_fit, *popt), color=palette[9], linestyle="-", label=f"Ajustement $\\frac{{1}}{{r^2}}$ ($R^2=${R_squared:0.2f})")

    plt.legend(fontsize=11)
    plt.ylabel("Intensité relative [-]", fontsize=16)
    plt.xlabel(f"Distance cristal-écran ($r$) [mm]", fontsize=16)
    plt.minorticks_on()
    plt.xlim(13, 32)
    plt.ylim(0.15, 0.32)
    plt.tick_params(axis="both", which="both", direction="in", top=True, right=True, labelsize=14)

    out_dir = os.path.join("output", "11_distance", "04_r2")
    os.makedirs(out_dir, exist_ok=True)

    plt.savefig(os.path.join(out_dir, "r2.png"), transparent=True, bbox_inches="tight")
    plt.close()

def plot_distance(dict_distance):
    x_values = np.array(list(dict_distance.keys()), dtype=float)
    x_err = 1.
    y_values = np.array([value[0] for value in dict_distance.values()])
    y_err = np.array([value[1] for value in dict_distance.values()])

    #popt, _ = curve_fit(model_r2, x_values, y_values)

    # Calculate R^2 coefficient
    #residuals = y_values - model_r2(x_values, *popt)
    #TSS = np.sum((y_values - np.mean(y_values)) ** 2)
    #RSS = np.sum(residuals ** 2)
    #R_squared = 1 - (RSS / TSS)

    slope, intercept, r_value, _, _ = linregress(x_values, y_values)
    x_fit = np.linspace(10, 35, 1000)

    plt.errorbar(x_values, y_values, xerr=x_err, yerr=y_err, color=palette[4], fmt="o", capsize=2, label="Données")
    plt.plot(x_fit, slope * x_fit + intercept, color=palette[6], linestyle="-", label=f"Régression linéaire ($R^2={r_value**2:.2f}$)")

    plt.legend(fontsize=11)
    plt.ylabel("Distance point-centre de l'image [mm]", fontsize=16)
    plt.xlabel(f"Distance cristal-écran [mm]", fontsize=16)
    plt.minorticks_on()
    plt.xlim(13, 32)
    plt.ylim(8, 27)
    plt.tick_params(axis="both", which="both", direction="in", top=True, right=True, labelsize=14)

    out_dir = os.path.join("output", "11_distance", "05_distance")
    os.makedirs(out_dir, exist_ok=True)

    plt.savefig(os.path.join(out_dir, "distance.png"), transparent=True, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    images_path = glob.glob(os.path.join("data", "distance", "*png"))
    images_dic = {os.path.splitext(os.path.basename(chemin_image))[0]: cv2.imread(chemin_image, cv2.IMREAD_UNCHANGED) for chemin_image in images_path}

    name15, name20, name25, name27, name30 = images_dic.keys()
    img15, img20, img25, img27, img30 = images_dic.values()

    #draw_all_contours(img15, name15, factor=0.15)
    #draw_all_contours(img20, name20, factor=0.17)
    #draw_all_contours(img25, name25, factor=0.18)
    #draw_all_contours(img27, name27, factor=0.15)
    #draw_all_contours(img30, name30, factor=0.20)

    good_contours_15 = [2, 3, 9, 10]
    good_contours_20 = [4, 5, 19, 20]
    good_contours_25 = [4, 5, 17, 18]
    good_contours_27 = [4, 5, 17, 18]
    good_contours_30 = [4, 5, 29, 30]

    contours_15 = remove_contours(img15, good_contours_15, factor=0.15)
    contours_20 = remove_contours(img20, good_contours_20, factor=0.17)
    contours_25 = remove_contours(img25, good_contours_25, factor=0.18)
    contours_27 = remove_contours(img27, good_contours_27, factor=0.15)
    contours_30 = remove_contours(img30, good_contours_30, factor=0.20)

    #draw_good_contours(img15, name15, contours_15)
    #draw_good_contours(img20, name20, contours_20)
    #draw_good_contours(img25, name25, contours_25)
    #draw_good_contours(img27, name27, contours_27)
    #draw_good_contours(img30, name30, contours_30)

    #draw_points(img15, name15, contours_15)
    #draw_points(img20, name20, contours_20)
    #draw_points(img25, name25, contours_25)
    #draw_points(img27, name27, contours_27)
    #draw_points(img30, name30, contours_30)

    intensity_15 = intensity(img15, contours_15)
    intensity_20 = intensity(img20, contours_20)
    intensity_25 = intensity(img25, contours_25)
    intensity_27 = intensity(img27, contours_27)
    intensity_30 = intensity(img30, contours_30)

    dict_intensity = {15:intensity_15, 20:intensity_20, 25:intensity_25, 27:intensity_27, 30:intensity_30}
    #plot_intensity(dict_intensity)

    distance15 = distance2center(img15, contours_15)
    distance20 = distance2center(img20, contours_20)
    distance25 = distance2center(img25, contours_25)
    distance27 = distance2center(img27, contours_27)
    distance30 = distance2center(img30, contours_30)

    dict_distance = {15:distance15, 20:distance20, 25:distance25, 27:distance27, 30:distance30}
    #plot_distance(dict_distance)