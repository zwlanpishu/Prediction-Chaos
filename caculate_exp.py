import math
import numpy as np
import matplotlib.pyplot as plt
import operator 
from math import log

def d(series,i,j):
    return abs(series[i]-series[j])
 
f=open('test_for_lyapunovexp.txt', 'r')
series=[float(i) for i in f.read().split()]
f.close()
N=len(series)
eps=0.01
dlist=[[] for i in range(N)]
n=0 #number of nearby pairs found
for i in range(N):
    for j in range(i+1,N):
        if d(series,i,j) < eps:
            n+=1
            print(n)
            for k in range(min(N-i,N-j)):
                dlist[k].append(log(d(series,i+k,j+k)))

f = open('lyapunov.txt','w')
for i in range(len(dlist)):
    if len(dlist[i]):
        print(sum(dlist[i])/len(dlist[i]), file = f)
f.close()

a = np.array([1, 1, 1])
b = np.array([2, 2, 2])
c = np.dot(a, b)
print(c)