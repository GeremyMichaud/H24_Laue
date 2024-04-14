import os
import numpy as np
from scipy.signal import find_peaks, peak_widths
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import linregress

palette = sns.color_palette("bright")


def read_txt_file(file_path):
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
            values_list = [float(line.strip()) for line in lines]
        return values_list
    except FileNotFoundError:
        print("File not found. Please check the file path.")
        return []

def indices_max(valeurs: np.array,
                                hauteur_minimum: int = None,
                                distance_minimum: int = None):
    peaks, _ = find_peaks(valeurs, height=hauteur_minimum, distance=distance_minimum)
    widths, _, _, _ = peak_widths(valeurs, peaks, rel_height=0.5)
    return peaks, widths

def gaussian(x, amp, cen, wid):
    return amp * np.exp(-(x - cen)**2 / (2 * wid**2))

def fit_gaussian_to_peak(x, y, peak_index, peak_width):
    peak_start = int(peak_index - peak_width)
    peak_end = int(peak_index + peak_width)
    x_peak = x[peak_start:peak_end]
    y_peak = y[peak_start:peak_end]

    amp_guess = np.max(y_peak)
    cen_guess = x_peak[np.argmax(y_peak)]
    wid_guess = peak_width

    amp_guess = max(amp_guess, 0.1)
    wid_guess = max(wid_guess, 0.1)

    popt, pcov = curve_fit(gaussian, x_peak, y_peak, p0=[amp_guess, cen_guess, wid_guess])

    return popt, pcov

names = ["NaCl", "LiF"]
dfs = {}

for name in names:
    file_path = os.path.join("data", "bragg", name + ".txt")
    data_list = read_txt_file(file_path)
    data_list = np.array(data_list)
    x = np.arange(len(data_list))
    peaks, widths = indices_max(data_list, hauteur_minimum=35, distance_minimum=8)
    if name == "NaCl":
        index_to_remove = [2, 3, 4, 5, 7, 9, 10, 11, 12]
        angle = np.linspace(4, 24, len(x))
    if name == "LiF":
        index_to_remove = [2, 3, 4, 5, 6,7 ,8 , 10, 12, 13, 14]
        angle = np.linspace(4, 34, len(x))
    mask = np.ones(len(peaks), dtype=bool)
    mask[index_to_remove] = False
    peaks = peaks[mask]
    widths = widths[mask]
    peaks_center = []
    peaks_error = []

    plt.plot(angle, data_list)
    for i, peak in enumerate(peaks):
        popt, pcov = fit_gaussian_to_peak(x, data_list, peak, widths[i])
        perr = np.sqrt(np.diag(pcov))
        amp_fit, cen_fit, wid_fit = popt
        cen_fit = cen_fit * 0.1 + 4
        wid_fit = wid_fit * 0.1

        error = wid_fit + perr[1] * 0.1
        peak_angle = peak * 0.1 + 4
        x_gauss = np.linspace(peak_angle-widths[i]*0.1, peak_angle+widths[i]*0.1, 1000)
        plt.plot(x_gauss, gaussian(x_gauss, amp_fit, cen_fit, wid_fit))
        plt.ylabel("Taux de comptage [-]", fontsize=16)
        plt.xlabel(r'Angle de la cible [$\degree$]', fontsize=16)
        plt.minorticks_on()
        plt.tick_params(axis="both", which="both", direction="in", top=True, right=True, labelsize=14)
        if name == "NaCl":
            plt.xlim(4, 24)
            peaks_center.append(cen_fit)
            peaks_error.append(error)
        if name == "LiF":
            plt.xlim(4, 34)
            peaks_center.append(peak_angle)
            peaks_error.append(error)

    df = pd.DataFrame({'peaks_center': peaks_center, 'peaks_error': peaks_error})
    dfs[name] = df

    out_dir = os.path.join("output", "09_a0_fit")
    os.makedirs(out_dir, exist_ok=True)

    plt.savefig(os.path.join(out_dir, "a0_fit_" + name), transparent=True)
    plt.close()

excel_file_path = f"bragg.xlsx"
write = False
if write:
    with pd.ExcelWriter(excel_file_path) as writer:
        for name, df in dfs.items():
            df.to_excel(writer, sheet_name=name, index=False)

df_nacl = pd.read_excel(excel_file_path, sheet_name="NaCl")
sintheta_nacl = df_nacl["sintheta"]
sinerr_nacl = df_nacl["sinerr"]
err_nacl = df_nacl["err"]
nlambda_nacl = df_nacl["nlambda"]
slope_nacl, intercept_nacl, rvalue_nacl, stderr_nacl, _ = linregress(sintheta_nacl, nlambda_nacl)

df_lif = pd.read_excel(excel_file_path, sheet_name="LiF")
sintheta_lif = df_lif["sintheta"]
sinerr_lif = df_lif["sinerr"]
err_lif = df_lif["err"]
nlambda_lif = df_lif["nlambda"]
slope_lif, intercept_lif, rvalue_lif, stderr_lif, _ = linregress(sintheta_lif, nlambda_lif)

x_range = np.linspace(0, 0.55, 1000)
rsquared_lif = rvalue_lif**2
rsquared_nacl = rvalue_nacl**2

plt.errorbar(sintheta_nacl, nlambda_nacl, xerr=sinerr_nacl, fmt='o', markersize=5, color=palette[3],
            elinewidth=0.8, capsize=3, label='Données NaCl')
plt.errorbar(sintheta_lif, nlambda_lif, xerr=sinerr_lif, fmt='s', markersize=5, color=palette[0],
            elinewidth=0.8, capsize=3, label='Données LiF')
plt.plot(x_range, slope_nacl * x_range + intercept_nacl, color=palette[1], linestyle='--', alpha=0.4, label=f"Régression NaCl ($R^2=${rsquared_nacl:0.2f})")
plt.plot(x_range, slope_lif * x_range + intercept_lif, color=palette[0], linestyle='-', alpha=0.4, label=f"Régression LiF ($R^2=${rsquared_lif:0.2f})")

plt.ylabel(r'$n\lambda$   [pm]', fontsize=16)
plt.xlabel(r'$\sin\theta$   [-]', fontsize=16)
plt.minorticks_on()
plt.xlim(0.0001, 0.55)
plt.ylim(0, 255)
plt.tick_params(axis="both", which="both", direction="in", top=True, right=True, labelsize=14)
plt.legend(fontsize=11)

out_dir = os.path.join("output", "10_a0_slope")
os.makedirs(out_dir, exist_ok=True)

plt.savefig(os.path.join(out_dir, "bragg_slope_"), transparent=True)
plt.close()
nacl_error = stderr_nacl + np.mean(err_nacl)
lif_error = stderr_lif + np.mean(err_lif)
print(f"a0 NaCl : {slope_nacl:.2f} ± {nacl_error:.2f}")
print(f"a0 LiF : {slope_lif:.2f} ± {lif_error:.2f}")
