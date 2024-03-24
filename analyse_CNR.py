import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import optimize
import os


palette = sns.color_palette("bright")

excel_file_path = 'snr.xlsx'
try:
    df = pd.read_excel(excel_file_path, sheet_name='CNR')
    n_images_values = df['n_images'].tolist()
    cnr_values = df['CNR'].tolist()

except FileNotFoundError:
    print(f"Error: File '{excel_file_path}' not found.")
except KeyError:
    print("Error: Sheet 'CNR' or columns 'n_images' and 'CNR' not found.")
except Exception as e:
    print("An error occurred:", e)

def model_CNR(x, fct, origin):
    return np.sqrt(x) * fct + origin

def r_squared(y, y_pred):
    ss_residual = np.sum((y - y_pred) ** 2)
    ss_total = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - (ss_residual / ss_total)
    return r2

params, params_covariance = optimize.curve_fit(model_CNR, n_images_values, cnr_values, p0=[1, 1])

predicted_cnr_values = model_CNR(np.array(n_images_values), params[0], params[1])
r_squared_value = r_squared(cnr_values, predicted_cnr_values)
x_range = np.linspace(0, 60, 1000)

plt.scatter(n_images_values, cnr_values, color=palette[3], label="Données CNR")
plt.plot(x_range, model_CNR(np.array(x_range), params[0], params[1]),
        linestyle='--', color=palette[3], alpha=0.4, label=f'Ajustement $\sqrt{{N}}$ ($R^2=${r_squared_value:0.2f})')
plt.ylabel("CNR [-]", fontsize=16)
plt.xlabel("Nombre d'images moyennées ($N$) [images]", fontsize=16)
plt.minorticks_on()
plt.tick_params(axis="both", which="both", direction="in", top=True, right=True, labelsize=14)
plt.xlim(-1, 55)
plt.legend(fontsize=11)

out_dir = os.path.join("output", "08_CNR")
os.makedirs(out_dir, exist_ok=True)

plt.savefig(os.path.join(out_dir, "CNR"), transparent=True, bbox_inches="tight", dpi=500)
plt.close()
