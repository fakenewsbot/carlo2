#Sergey Kurennoy
"""
Only used for generating the picture of the 
family of Johnson's S_U distributions for 
different shape parameters a 
(with b fixed, because it is nearly the same in all fits)
"""

import numpy as np
from scipy.stats import johnsonsu
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 1)

b = 1.0 #same for each, only a varied

a = 1.0
mean, var, skew, kurt = johnsonsu.stats(a,b, moments='mvsk')
print("For shape = ",a,b," mean, var, skew, kurt:\n  ",mean, var, skew, kurt)
x = np.linspace(johnsonsu.ppf(0.0155, a,b),johnsonsu.ppf(0.9975, a,b), 400)
ax.plot(x, johnsonsu.pdf(x, a,b),
        'r-', lw=2, alpha=0.6, label='a=1.0 b=1.0')

a = 1.5
mean, var, skew, kurt = johnsonsu.stats(a,b, moments='mvsk')
print("For shape = ",a,b," mean, var, skew, kurt:\n  ",mean, var, skew, kurt)
x = np.linspace(johnsonsu.ppf(0.048, a,b),johnsonsu.ppf(0.9995, a,b), 300)
ax.plot(x, johnsonsu.pdf(x, a,b),
        'g-', lw=2, alpha=0.6, label='a=1.5 b=1.0')

a = 2.0
mean, var, skew, kurt = johnsonsu.stats(a,b, moments='mvsk')
print("For shape = ",a,b," mean, var, skew, kurt:\n  ",mean, var, skew, kurt)
x = np.linspace(johnsonsu.ppf(0.12, a,b),johnsonsu.ppf(0.9999, a,b), 200)
ax.plot(x, johnsonsu.pdf(x, a,b),
        'k-', lw=2, alpha=0.6, label='a=2.0 b=1.0')

a = 2.5
mean, var, skew, kurt = johnsonsu.stats(a,b, moments='mvsk')
print("For shape = ",a,b," mean, var, skew, kurt:\n  ",mean, var, skew, kurt)
x = np.linspace(johnsonsu.ppf(0.245, a,b),johnsonsu.ppf(0.99999, a,b), 100)
ax.plot(x, johnsonsu.pdf(x, a,b),
        'b-', lw=2, alpha=0.6, label='a=2.5 b=1.0')

a = 3.0
mean, var, skew, kurt = johnsonsu.stats(a,b, moments='mvsk')
print("For shape = ",a,b," mean, var, skew, kurt:\n  ",mean, var, skew, kurt)
x = np.linspace(johnsonsu.ppf(0.42, a,b),johnsonsu.ppf(0.999999, a,b), 100)
ax.plot(x, johnsonsu.pdf(x, a,b),
        'm-', lw=2, alpha=0.6, label='a=3.0 b=1.0')

#ax.hist(r, density=True, histtype='stepfilled', alpha=0.2)
ax.legend(loc='upper left', prop={'size': 24}, frameon=False)
plt.show()



