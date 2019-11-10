"""
Borrowed from 
https://stackoverflow.com/questions/20699821/find-and-draw-regression-plane-to-a-set-of-points/20700063#20700063

Used to find a linear model for the parameters a and b of the fitting distributions,
as they depend on delta and gamma parameters of the time series in the samples.
"""

import numpy as np
#from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import scipy.optimize
import functools

def plane(x, y, params):
    a = params[0]
    b = params[1]
    c = params[2]
    z = a*x + b*y + c
    return z

def error(params, points):
    result = 0
    for (x,y,z) in points:
        plane_z = plane(x, y, params)
        diff = abs(plane_z - z)
        result += diff**2
    return result


#enter data saved in fitresult
points = []
for i in range(len(deltalist)):
    #print("i=",i)
    for j in range(len(gammalist)):
        #print("j=", j, "index=", i*len(gammalist) + j)
        #print("at (δ,γ)=(" + str("%.2f" % deltalist[i]) + 
        #  " γ=" + str("%.2f" % gammalist[j]) + "): exp =", 
        #  fitresult[i*len(gammalist) + j][0][0])
        points.append([deltalist[i], gammalist[j], 
                    fitresult[i*len(gammalist) + j][0][0]]) #parameter a
#                    fitresult[i*len(gammalist) + j][0][1]]) #parameter b


fun = functools.partial(error, points=points)
params0 = [0., 0., 0.]
res = scipy.optimize.minimize(fun, params0)

a = res.x[0]
b = res.x[1]
c = res.x[2]

xs, ys, zs = zip(*points)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(xs, ys, zs)

ax.set_xlim(-0.02,0.06)
ax.set_ylim(0.01,0.07)
#ax.set_zlim(-1.,0.4) #for pdfmax
ax.set_zlim(1.5,3.) #for a
#ax.set_zlim(0.7,1.) #for b

plt.show()

#display percent errors
for i in range(len(deltalist)):
    #print("i=",i)
    for j in range(len(gammalist)):
        #print("j=", j, "index=", i*len(gammalist) + j)
        pred = a*deltalist[i] + b*gammalist[j] + c
        exp = fitresult[i*len(gammalist) + j][0][0]
        err = 100.*(pred - exp)/exp
        print("at (δ,γ)=(" + str("%.2f" % deltalist[i]) + 
          " γ=" + str("%.2f" % gammalist[j]) + "): max =", 
          str("%.3f" % exp), 
          "pred =", str("%.3f" % pred), 
          "err% =", str("%.1f" % err))

