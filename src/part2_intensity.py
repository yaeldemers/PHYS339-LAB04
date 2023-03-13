import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as colors

import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from uncertainties import ufloat, unumpy
from uncertainties.umath import *  # sin(), etc.

# Apply the default theme
sns.set_theme()

# Data was taken manually
input_value = [200, 
390,
580,
770,
960,
1150,
1160,
1170,
1180,
1190,
1200,
1201,
1202,
1203,
1204,
1205,
1206,
1207,
1208,
1209,
1210,
1220,
1230,
1240,
1260,
1280,
1300,
1320,
1340,
1530,
1720,
1910,
2000,
3000,
4000]
measured_voltage = [0.24,
0.462,
0.69,
0.91,
1.137,
1.363,
1.376,
1.387,
1.4,
1.412,
1.424,
1.425,
1.427,
1.428,
1.43,
1.43,
1.432,
1.432,
1.434,
1.435,
1.436,
1.447,
1.46,
1.47,
1.493,
1.515,
1.54,
1.566,
1.59,
1.682,
1.682,
1.682,
1.683,
1.683,
1.684]
mean_intensity = [0.035,
0.093,
0.369,
2.171,
5.557,
14.765,
16.129,
18.029,
21.108,
26.556,
44.05,
48.761,
43.456,
52.05,
58.708,
70.625,
83.546,
92.876,
102.926,
113.557,
141.392,
261.835,
366.061,
437.251,
495.739,
520.829,
542.261,
560.378,
580.85,
635.678,
635.203,
634.717,
634.219,
633.588,
633.792]
std_intensity = [0.454,
0.67,
0.863,
1.059,
1.462,
1.282,
1.331,
1.915,
2.497,
4.372,
16.21,
20.189,
14.913,
22.519,
27.393,
38.169,
48.014,
56.218,
64.211,
72.872,
92.166,
89.151,
49.827,
28.085,
8.427,
4.542,
3.566,
3.249,
3.111,
3.341,
3.633,
3.726,
3.471,
3.747,
3.393]

# Setting up data and error
voltage = unumpy.uarray(measured_voltage[:20], 0.0005)
intensity = unumpy.uarray(mean_intensity[:20], std_intensity[:20])
resistor = ufloat(100, 10)

# convert ADC to int voltage
intensity_w_error = (intensity*5)/1023

# Convert voltage to current
current_w_error = voltage / resistor

# Separating values and errors for plotting purposes
intensity = unumpy.nominal_values(intensity_w_error)
intensity_error = unumpy.std_devs(intensity_w_error)

current = unumpy.nominal_values(current_w_error)
current_error = unumpy.std_devs(current_w_error)

# Linear fit definition
def lin(x, a, b):
   return  a*x+b

# Iterating over the fit configurations to find best fits using Mean Squared Error
best_config = 0
best_MSE = float('inf')

for i in range(5, 15, 1):
    current_left, current_right = current[:i+1], current[i:]
    intensity_left, intensity_right = intensity[:i+1], intensity[i:]
    
    popt_l, pcov_l = curve_fit(lin, current_left, intensity_left, sigma=intensity_error[:i+1])
    popt_r, pcov_r = curve_fit(lin, current_right, intensity_right, sigma=intensity_error[i:])

    MSE_left = np.sum((intensity_left-lin(current_left, popt_l[0], popt_l[1]))**2)/len(intensity_left)
    MSE_right = np.sum((intensity_right-lin(current_right, popt_r[0], popt_r[1]))**2)/len(intensity_right)

    if best_MSE > MSE_left + MSE_right:
        best_MSE = MSE_left + MSE_right
        best_config = i

# Fit lines for plotting purposes
popt_l, pcov_l = curve_fit(lin, current[:best_config+1], intensity[:best_config+1], sigma=intensity_error[:best_config+1])
popt_r, pcov_r = curve_fit(lin, current[best_config:], intensity[best_config:], sigma=intensity_error[best_config:])

# CAREFUL THIS IS WAY TOO HIGH
threshold_error = (np.sqrt(np.diag(pcov_l)) + np.sqrt(np.diag(pcov_r)))/2

expected_left = lin(np.linspace(0, 0.015, 10), popt_l[0], popt_l[1])
expected_right = lin(np.linspace(0, 0.015, 10), popt_r[0], popt_r[1])

# Plotting
plt.scatter(current, intensity, s=10)

plt.errorbar(current, intensity, xerr=current_error, yerr=intensity_error, alpha=0.5, fmt="none")
plt.plot(np.linspace(0, 0.015, 10), expected_left, linewidth=1, color='red')
plt.plot(np.linspace(0, 0.015, 10), expected_right, linewidth=1, color='red')
plt.xlabel("Current (A)") #(RZ 100 ohm)
plt.ylabel("Intensity $(V)$") #10bit need to convert to voltage
plt.scatter(0.0141845, 0.0653828, color="black", label = r"$I_{threshold} = (0.0141845 \pm 0.0048225)$ A")
plt.ylim(-0.05,0.6)
plt.legend()
plt.savefig("fig_part2.png", format="png", dpi=1200)
