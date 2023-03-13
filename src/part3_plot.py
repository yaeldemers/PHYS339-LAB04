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

x = np.linspace(1,180, 180)

errorColor = colors.to_rgba([0, 0.3, 0.5], 0.4)

stepCountsCal, adcValuesnp = np.load("data_part3-p.npy")

sliced_step = np.zeros((180, 20))
sliced_adc = np.zeros((180, 20))

for i in range(20):
    #sliced_step[:,i] = stepCountsCal[i*360:(i+1)*360]
    sliced_step[:,i] = x
    sliced_adc[:,i] = adcValuesnp[i*180:(i+1)*180]
    
stats_step = [np.mean(sliced_step,axis=1) , np.std(sliced_step,axis=1)]
stats_adc = [np.mean(sliced_adc, axis=1), np.std(sliced_adc, axis=1)]

step_clean = stats_step[0][82:172]-82
adc_clean = stats_adc[0][82:172]/np.max(stats_adc[0])

voltage = unumpy.uarray(stats_adc[0][82:172], stats_adc[1][82:172])
voltage_clean = voltage / np.max(stats_adc)

# Separating values and errors for plotting purposes
voltage_errors = unumpy.std_devs(voltage_clean)

def func(x, a):
   #x_prime = np.arcsin(a*np.sin((np.pi*x)/180))
   x_prime=a
   return  1 - (((np.tan((np.pi*x)/180-x_prime))**2)/((np.tan((np.pi*x)/180+x_prime))**2))

def T(x, n1, n2):
    angle = np.arcsin((n1/n2) * np.sin((np.pi*x)/180))
    alpha = np.cos(angle)/np.cos((np.pi*x)/180)
    beta = n1/n2
    
    return (alpha*beta)*np.power(2/(alpha+beta), 2)

#popt, pcov = curve_fit(T, step_clean, adc_clean, p0=[0, 90], maxfev = 50000) 
#perr = np.sqrt(np.diag(pcov))
#expected_out = func(x, popt[0])

plt.scatter(step_clean, adc_clean, s=10)
#plt.plot(x, expected_out, linewidth=1, color='red')
plt.xlabel(r"Rotation Angle $(\theta)$") #(RZ 100 ohm)
plt.ylabel("Intensity $(V)$") #10bit need to convert to voltage
plt.errorbar(step_clean, adc_clean, xerr=0.5, yerr=voltage_errors, ecolor=errorColor, fmt="none")
plt.title("P-Polarized transmission curve")
plt.axvline(x=62, color='r', linestyle='--', label=r"$\theta_B=62 \pm 0.5$")
plt.xlim(0,90)
plt.ylim(0,1.25)
plt.legend()
plt.savefig("fig_part3.png", format="png", dpi=1200)

brewster_angle = ufloat(62, 0.5) #np.where(adc_clean==1)
n2 = tan(brewster_angle*np.pi/180) * 1.0003