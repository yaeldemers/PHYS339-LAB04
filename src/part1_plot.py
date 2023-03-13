import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as colors

#import laserClass2022
import numpy as np
#import lmfit
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from uncertainties import ufloat, unumpy
from uncertainties.umath import *  # sin(), etc.

# Apply the default theme
sns.set_theme()

x = np.linspace(1,360, 360)

errorColor = colors.to_rgba([0, 0.3, 0.5], 0.5)

stepCountsCal, adcValuesnp = np.load("data_part1.npy")

np.save("data_part1.npy", [stepCountsCal, adcValuesnp])

plt.plot(stepCountsCal, adcValuesnp)
plt.close()

sliced_step = np.zeros((360, 10))
sliced_adc = np.zeros((360, 10))

for i in range(10):
    #sliced_step[:,i] = stepCountsCal[i*360:(i+1)*360]
    sliced_step[:,i] = x
    sliced_adc[:,i] = adcValuesnp[i*360:(i+1)*360]
    
    

# Handling conversion and error propagation
stats_step = [np.mean(sliced_step,axis=1) , np.std(sliced_step,axis=1)]
stats_adc = unumpy.uarray(np.mean(sliced_adc,axis=1) , np.std(sliced_adc,axis=1))

temp_voltage = stats_adc * 5 / 1023

# Separating values and errors for plotting purposes
stats_voltage = [unumpy.nominal_values(temp_voltage), unumpy.std_devs(temp_voltage)]

# Fitting
def func(x, a, b, c):
   return  a*np.cos((np.pi*x)/180+b)**2 + c

popt, pcov = curve_fit(func, stats_step[0], stats_voltage[0], maxfev=5000) 
perr = np.sqrt(np.diag(pcov))

expected_out = func(x, popt[0], popt[1], popt[2])

# Plotting
fig, (ax1, ax2) = plt.subplots(2, sharex=True)
fig.supylabel("Intensity (V)")
ax1.plot(x, expected_out, linewidth=1, color="black", alpha=0.75)
ax1.scatter(stats_step[0], stats_voltage[0], s=1)
plt.axhline(y=0, linewidth=1, color="black", alpha=0.75)
ax2.errorbar(x,stats_voltage[0]-expected_out, yerr=2*stats_voltage[1], alpha=0.35, fmt="none")
ax2.scatter(x, stats_voltage[0]-expected_out, s=1)
ax2.set_xlabel(r"Rotation angle $(\theta)$")
plt.savefig("fig_part1.png", format="png", dpi=1200)