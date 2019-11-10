#Sergey Kurennoy
"""
Performs the Johnson inverse hyperbolic sine transformation on the 
sample data, to obtain a normal-like associated distribution.
Use arcsinh(w) = ln( w + √(w^2 + 1) )
 or sinh(v) = (exp(v) - exp(-v) )/2
"""

from scipy.stats import johnsonsu, norm, uniform, levy_stable
from scipy.stats import skew, kurtosis
import numpy as np

"""
#check against a generated true Johnson S_U distribution
points = 1000000

a = 2.53727
b = 1.10494
loc = -4577.17
sc = 3357.76

Xjsu = johnsonsu.rvs(a, b, loc, sc, size=points, random_state=None)
print(johnsonsu.fit(Xjsu))
"""

Xjsu = np.array(q1)

#reverse Johnson TRANSFORMATION
Xjsu00 = (Xjsu -loc) / sc #remove scale, location
arcsinhX = np.log(Xjsu00 + (Xjsu00**2 + 1.)**0.5)
z = a + b * arcsinhX #remove a, b shape parameters: ~normally distributed
print(norm.fit(z))
u = norm.cdf(z) #uniformly distributed
print(uniform.fit(u))


#Normal fit:
[mean, var] = norm.fit(z)
print("Normal fit parameters:", mean, var)

#Calculate chi-square test statistic:
k = 32 #number of bins in histogram
#use numpy histogram for the expected value
observed, hist_binedges = np.histogram(z, bins=k)
#use the cumulative density function (c.d.f.)
#of the distribution for the expected value:
cdf = norm.cdf(hist_binedges, mean, var)
expected = len(z) * np.diff(cdf)
#use scipy.stats chisquare function, where 
#ddof is the adjustment to the k-1 degrees of freedom, 
#which is the number of distribution parameters
chisqnorm, pvalnorm = st.chisquare(observed, expected, ddof=2)
print("Chi-square for normal =", chisqnorm)
print("Normal skew =", skew(z), ";\nexcess kurtosis =", kurtosis(z))


#Levy-stable fit
[al, be, de, ga] = levy_stable._fitstart(z)
print("Levy stable fit parameters:", al, be, de, ga)

#Calculate chi-square test statistic:
k = 34 #number of bins in histogram
#use numpy histogram for the expected value
observed, hist_binedges = np.histogram(z, bins=k)
#use the cumulative density function (c.d.f.)
#of the distribution for the expected value:
cdf = levy_stable.cdf(hist_binedges, al, be, de, ga)
expected = len(z) * np.diff(cdf)
#use scipy.stats chisquare function, where 
#ddof is the adjustment to the k-1 degrees of freedom, 
#which is the number of distribution parameters
chisqlevy, pvallevy = st.chisquare(observed, expected, ddof=4)
print("Chi-square for levy stable =", chisqlevy)


#Plotting
matplotlib.style.use('ggplot')

data = pd.Series(z)

#determines left and right limits of fit displayed (adjust numbers for more/less)
x = np.linspace(norm.ppf(0.001, mean, var),
                norm.ppf(0.999, mean, var), 1000)
y = norm.pdf(x, mean, var)
pdf = pd.Series(y, x)

x = np.linspace(levy_stable.ppf(0.005, al, be, de, ga),
                levy_stable.ppf(0.99, al, be, de, ga), 1000)
y = levy_stable.pdf(x, al, be, de, ga)
pdflevy = pd.Series(y, x)

#plot of histogram and fit probability density function
plt.figure(figsize=(10,6), dpi=100)

ax = pdf.plot(lw=2, label="normal fit: μ=" + str("%.8f" % mean) + 
              " σ^2=" + str("%.8f" % var), legend=True)
pdflevy.plot(lw=2, label="levy stable fit: a=" + str("%.2f" % al) + 
             " b=" + str("%.2f" % be), legend=True, ax=ax)
data.plot(kind='hist', bins=200, alpha=0.5, 
          label='a+b*arcsinh((data-loc)/sc), δ=' + str("%.2f" % delta) + 
          " γ=" + str("%.2f" % gamma), density=True, legend=True, ax=ax)


#Uniform fit:
#Fit the sample to Johnson's S_U distribution, 
#with shape parameters a, b, as well as location and shape parameters
[lo1, sc1] = uniform.fit(u)
print("Uniform fit parameters:", lo1, sc1)

#Calculate chi-square test statistic:
k = 32 #number of bins in histogram
#use numpy histogram for the expected value
observed, hist_binedges = np.histogram(u, bins=k)
#use the cumulative density function (c.d.f.)
#of the distribution for the expected value:
cdf = uniform.cdf(hist_binedges, lo1, sc1)
expected = len(u) * np.diff(cdf)
#use scipy.stats chisquare function, where 
#ddof is the adjustment to the k-1 degrees of freedom, 
#which is the number of distribution parameters
chisquni, pvaluni = st.chisquare(observed, expected, ddof=2)
print("Chi-square for uniform =", chisquni)

print("Uniform skew =", skew(u), ";\nexcess kurtosis =", kurtosis(u))


datau = pd.Series(u)

#determines left and right limits of fit displayed (adjust numbers for more/less)
x = np.linspace(uniform.ppf(0.0007, lo1, sc1),
                uniform.ppf(0.999, lo1, sc1), 1000)
y = uniform.pdf(x, mean, var)
pdf = pd.Series(y, x)

#plot of histogram and fit probability density function
plt.figure(figsize=(10,6), dpi=100)

ax = pdf.plot(lw=2, label="uniform fit: loc=" + str("%.4f" % lo1) + 
              " sc=" + str("%.4f" % sc1), legend=True)
datau.plot(kind='hist', bins=200, alpha=0.5, 
          label='Φ(a+b*arcsinh((data-loc)/sc)) δ=' + str("%.2f" % delta) + 
          " γ=" + str("%.2f" % gamma), density=True, legend=True, ax=ax)

