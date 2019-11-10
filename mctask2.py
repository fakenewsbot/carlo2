#Sergey Kurennoy
"""
Generates samples of 1% percentile overlapping 10-day proportional returns 
based on a stable distribution for log-price difference. Fits the sample to 
a Johnson's S_U distribution, and remembers the results.
Plots are produced in mcplot.py, which recalls remembered data.
"""
import numpy as np
import scipy.stats as st
from scipy.stats import levy_stable, johnsonsu
import matplotlib.pyplot as plt
import time

#given length of time series
N = 750

#initialize arrays of returns: only N-9 overlapping 10-day returns
r1 = np.zeros(N)
r10 = np.zeros(N-9)

"""
Routine generates and returns one sample of 
the time series of 10-day overlapping returns
""" 
def generate_series(N, alpha, beta, delta, gamma):
    #Start by generating N random variables X distributed 
    #according to the Levy alpha-stable distribution.
    #Presume that the one-day returns are distributed according to 
    #to said distribution:  r^1_i := X
    #return1 = np.ones(N)
    #return1 = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 
    #                    0.5, 0.6, 0.7, 0.8, 0.9, 
    #                    1.0, 1.1, 1.2, 1.3])
    return1 = levy_stable.rvs(alpha, beta, delta, gamma, 
                              size=N, random_state=None)
    
    #Avoid generating the time series for prices P_i directly.
    #Use the equations given for fractional daily returns:
    #  r^1_i := ( P_{i+1} - P_i ) / P_i
    #    = ( P_{i+1} / P_i ) - 1
    #  r^10_i := ( P_{i+10) - P_i ) / P_i
    #    = ( P_{i+10} / P_i ) - 1
    #      ( P_{i+10)   P_{i+9}        P_{i+1} )
    #    = | -------- * ------- * ... *------- | - 1   
    #      ( P_{i+9}    P_{i+8}          P_i   )
    #    = ( r^1_{i+9} + 1 )*( r^1_{i+8} + 1 )*...*( r^1_i + 1 ) - 1
    #to compute the overlapping ten-day returns.
    #This requires calculating N-9 products of 10 consecutive values 
    #of the array ( r^1_i + 1 )_{i=0}^{N-1} of one-day returns plus 1 
    return1plusone = np.zeros(N)
    return1plusone = return1 + 1.
    """
    #Way 0 in the most direct way,
    #gives exactly the same result as way 1, but slower 4 times
    return10plusone = np.zeros(N-9)
    for i in range(N-9):
        product = np.prod(return1plusone[i:(i + 10)])
        return10plusone[i] = product
    """
    
    #Way 1 using numpy array element-with-element multiplication:
    #   [     1      ,     1     , ... ,       1       ,       1       ]
    # * [ r^1_1 + 1  , r^1_2 + 1 , ... , r^1_{N-10} + 1, r^1_{N-9} + 1 ]
    # * [ r^1_2 + 1  , r^1_3 + 1 , ... , r^1_{N-9} + 1 , r^1_{N-8} + 1 ]
    #  ...
    # * [ r^1_10 + 1 , r^1_11 + 1, ... , r^1_{N-1} + 1 ,   r^1_N + 1   ]
    # = [ r^10_1 + 1 , r^10_2 + 1, ... ,r^10_{N-10} + 1, r^10_{N-9} + 1]
    return10plusone = np.ones(N-9)
    for k in range(10):
        return10plusone *= return1plusone[k:(N-9 + k)]
    
    """
    #Way 2 using a running product
    #about 50% slower than way 1 (counterintuitively), 
    #and also entails a "randomly walking" error that
    #stays bounded (< 10^-14) unless one-day returns get near -1
    return10plusone = np.zeros(N-9)
    runningproduct = np.prod(return1plusone[0:10])
    #print("start with first product", runningproduct)
    return10plusone[0] = runningproduct
    for i in range(1, N-9):
        runningproduct = runningproduct * return1plusone[i+9] / return1plusone[i-1]
        #print("multiply by",return1plusone[k+9],
        #      "and divide by", return1plusone[k-1], 
        #      "equals", runningproduct)
        return10plusone[i] = runningproduct
    """
    return10 = return10plusone - 1.
    return return10


def drawplots(arrays):
    #Used for looking at time series
    fig = plt.figure(figsize=(10,8), dpi=100)
    
    nplots = len(arrays)
    for k in range(len(arrays)):
        plt.subplot(nplots,1,k+1)
        plt.plot(arrays[k])



#Generate single time series
#Given parameters of Levy alpha-stable distribution,
#where delta=location parameter comes before gamma=scale parameter
#to coincide with the order in the scipy.ststs.levy_stable specification:
#  rvs(alpha, beta, loc=0, scale=1, size=1, random_state=None)
alpha, beta, delta, gamma = 1.7, 0.0, 1.0, 1.0
r10 = generate_series(N, alpha, beta, delta, gamma)
drawplots([r10])
print("q1%=", np.quantile(r10, 0.01))


#Generate a single sample
alpha, beta, delta, gamma = 1.7, 0.0, 1.0, 1.0
samplesize = 1000000
q1r10 = np.zeros(samplesize)
#Collect each sample from generated independent time series
for j in range(samplesize):
    if ((j+1) % 24000 == 0):
        print((j+1)*100.0/samplesize,"% done")
    r10 = generate_series(N, alpha, beta, delta, gamma)
    #Find 1% quantile for the ten-day return timeseries
    q1r10[j] = np.quantile(r10, 0.01)


#ERROR SCALING: vary the sample size and look at the 
#empirical errors in the mean of the resulting 1% quantile returns, 
#compare to the expected error scaling ~1/√N
alpha, beta, delta, gamma = 1.7, 0.0, 1.0, 1.0

trials = 26 #26 for 6.0 biggest exponent 
exponents = np.linspace(1.0, 6.0, trials)

sizelist = [] #list of sample sizes = 10^exponents
for i in range(trials):
    sizelist.append(int(round(10**exponents[i])))

meanlist = np.zeros(trials)
for i in range(trials):
    samplesize = sizelist[i]
    print("Sample with", samplesize, "points")
    
    start = time.time() #time each case
    
    q1r10 = np.zeros(samplesize)
    #Collect each sample from generated independent time series
    for j in range(samplesize):
        if ((j+1) % 24000 == 0):
            print((j+1)*100.0/samplesize,"% done")
        r10 = generate_series(N, alpha, beta, delta, gamma)
        
        #Find 1% quantile for ten-day return time series
        q1r10[j] = np.quantile(r10, 0.01)
    meanlist[i] = np.mean(q1r10)
    
    end = time.time()
    print("Time:", end - start, "s")

#Calculate error relative to the last trial result (most points: 10^6)
#throw away last element in error (= 0)
error = np.abs((meanlist - meanlist[-1]) / meanlist[-1])[:-1]

#Plot scatter of errors and theory in log-log plot
fig, ax = plt.subplots(1, 1)
plt.xscale('log')
plt.xlabel("sample size N")
plt.yscale('log')
plt.ylabel("proportional error")
plt.title("log-log error in mean of samples")

#Central limit theorem predicts that error 
#scales as M^(-1/2) for sample size M
theoryworst = 4. * np.array(sizelist)**-0.5
theory = 1.5 * np.array(sizelist)**-0.5
#for 1% quantile returns, expected error for M=100 
#could be up to ~1=100%, but most likely much less.
#With a factor of 10., maximum error bound 10/sqrt(100) = 1 at M=100;
#with a factor of 5., the line corresponds to most empirical results.
ax.plot(sizelist, theoryworst, 
        'b--', lw=2, alpha=0.6, label="bound: max error ≈ 4/√M")
ax.plot(sizelist, theory, 
        'k-', lw=2, alpha=0.6, label="prediction: expected error ≈ 1.5/√M")
ax.plot(sizelist[:-1], error,  
        'ro', alpha=0.6, label="empirical error δ=" + str('%.2f' % delta) + 
          " γ=" + str('%.2f' % gamma))
ax.legend(loc='upper right', prop={'size': 20})
plt.show()


#Generate a large set of SMALL DELTA/GAMMA SAMPLES
#(delta and gamma to be varied)
deltalist = [-0.02, 0.0, 0.02, 0.04, 0.06]
gammalist = [0.01, 0.03, 0.05, 0.07]

#Number of independent time series in each sample
samplesize = 160000
#expected error is then about 5/√N = 0.0125 (1.25%)
#which, in the worst case is better than about 10/√N = 0.025 (2.5%)

#collection of samples and fit parameters for different delta&gamma cases:
samples = []
fitresult = []

for delta in deltalist:
    for gamma in gammalist:
        print("Time series via params (α,β,δ,γ)=", alpha, beta, delta, gamma)
        
        start = time.time() #time each case
        
        q1r10 = np.zeros(samplesize)
        #Collect each sample from generated independent time series
        for j in range(samplesize):
            if ((j+1) % 24000 == 0):
                print((j+1)*100.0/samplesize,"% done")
            r10 = generate_series(N, alpha, beta, delta, gamma)
            #Find 1% quantile for the ten-day return timeseries
            q1r10[j] = np.quantile(r10, 0.01)
        #remember sample
        samples.append(q1r10)
 
        #Fit the sample to Johnson's S_U distribution, 
        #with shape parameters a, b, as well as location and shape parameters
        [a, b, loc, sc] = johnsonsu.fit(q1r10)
        print("Johnson S_U fit parameters:", a, b, loc, sc)
        
        #Calculate chi-square test statistic:
        k = 34 #number of bins in histogram
        #use numpy histogram for the expected value
        observed, hist_binedges = np.histogram(q1r10, bins=k)
        #use the cumulative density function (c.d.f.)
        #of the distribution for the expected value
        cdf = johnsonsu.cdf(hist_binedges, a, b, loc, sc)
        expected = len(q1r10) * np.diff(cdf)
        #use scipy.stats chisquare function, where 
        #ddof is the adjustment to the k-1 degrees of freedom, 
        #equal to the number of distribution parameters
        chisq, pval = st.chisquare(observed, expected, ddof=4)
        print("Chi-square = ", chisq)
        
        #remember fit results
        fitresult.append([[a, b, loc, sc], chisq])
        
        end = time.time()
        print("Time:", end - start, "s")



