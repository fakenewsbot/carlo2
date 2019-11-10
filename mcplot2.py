#Sergey Kurennoy
"""
Produces the plot of the sample histogram with fitting distribution.
Assumes that gammalist, deltalist, samples, fitresult are all populated.
"""
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import johnsonsu
#FIRST run mctask


numplots = len(deltalist)*len(gammalist)

#Prompt for case index
gdindex = int(input("Which plot (1 to "+str(numplots)+")? ")) - 1

#Determine gamma and delta (reverse 2D array gammalist x deltalist)
gamma = gammalist[gdindex % len(gammalist)]
delta = deltalist[(gdindex - (gdindex % len(gammalist))) // len(gammalist)]
print("(δ,γ)=", delta, gamma)


#Recall remembered sample and fit results
q1in = samples[gdindex]
#q1in = q1r10
#[[afull, bfull, locfull, scfull], chisqfull] = fitresult[gdindex]

#q1in = q1milld1g1 #million points for delta=gamma=1
#delta, gamma = 1., 1.

#Remove about <1% of outliers below cutoff value 
#(histogram plotting and distribution-fitting
#is very difficult with a "fat tail" that is 
#sparcely populated by the Monte Carlo simulation. 
#Even with very big sample sizes, most of the 
#long tail bins are empty)
q1 = []
cutoff = -4.5-150*gamma-75*delta  #-150000. for delta=gamma=1.
countoutliers = 0
for i in range(len(q1in)):
    if q1in[i] < cutoff:
        countoutliers += 1
    else:
        q1.append(q1in[i])
print("Cut off ", (countoutliers * 100.) / len(q1in), 
      "% below cutoff=", cutoff)


#Redo fit after cutoff:
#Fit the sample to Johnson's S_U distribution, 
#with shape parameters a, b, as well as location and shape parameters
[a, b, loc, sc] = johnsonsu.fit(q1)
print("Johnson S_U fit parameters:", a, b, loc, sc)

#Calculate chi-square test statistic:
k = 34 #number of bins in histogram
#use numpy histogram for the expected value
observed, hist_binedges = np.histogram(q1, bins=k)
#use the cumulative density function (c.d.f.)
#of the distribution for the expected value:
cdf = johnsonsu.cdf(hist_binedges, a, b, loc, sc)
expected = len(q1) * np.diff(cdf)
#use scipy.stats chisquare function, where 
#ddof is the adjustment to the k-1 degrees of freedom, 
#which is the number of distribution parameters
chisq, pval = st.chisquare(observed, expected, ddof=4)
print("Chi-square = ", chisq)


#Plotting
matplotlib.style.use('ggplot')

data = pd.Series(q1)
#determines left and right limits of fit displayed (adjust numbers for more/less)
x = np.linspace(johnsonsu.ppf(0.05, a, b, loc, sc),
                johnsonsu.ppf(0.9999, a, b, loc, sc), 2000)
#x = np.linspace(-2.5, 0.5, 1201)
#x = np.linspace(-2.0, 1.0, 1201) #shift up for delta=0.06 gamma = 0.01
y = johnsonsu.pdf(x, a, b, loc, sc)
pdf = pd.Series(y, x)

#find probability density function maximum location numerically
pdfmax = x[np.where(y==np.amax(y))[0][0]]

#plot of histogram and fit probability density function
plt.figure(figsize=(8,5), dpi=100)

ax = pdf.plot(lw=2, label="fit a=" + str("%.3f" % a) + 
              " b=" + str("%.3f" % b) + 
              " max=" + str("%.3f" % pdfmax) + 
              " (χ2=" + str("%.0f" % chisq) + ")", legend=True)
data.plot(kind='hist', bins=500, alpha=0.5, 
          label='data δ=' + str("%.2f" % delta) + 
          " γ=" + str("%.2f" % gamma), density=True, legend=True, ax=ax)

#remember new fit results, with cutoff and maximum
#fitresult[gdindex] = [[a, b, loc, sc], chisq, pdfmax]
#fitresult.append([[a, b, loc, sc], chisq, pdfmax])

