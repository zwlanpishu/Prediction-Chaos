import math
import numpy as np 
import matplotlib.pyplot as plt 
from math import log

 
def d(series,i,j):
    return abs(series[i]-series[j])
 
# 载入时间序列数据，该数据只有1列，且无序列号、字段名等其他信息 
file_data = open('timeseries.txt', 'r')
time_series = [float(i) for i in file_data.read().split()]
file_data.close()

N = len(time_series)
eps = 1
dlist = [[] for i in range(N)]
num_nearby = 0

for i in range(N):
    for j in range(i+1,N):
        if d(time_series, i, j) < eps:
            num_nearby += 1
            print("the num of pairs found : " + str(num_nearby))
            for k in range(min(N - i, N - j)):
                value_nonzero = d(time_series, i + k, j + k) + 0.0001
                dlist[k].append((value_nonzero))
  
file_output = open('lyapunov.txt','w')

for i in range(len(dlist)) :
    if len(dlist[i]) :
        print(i, sum(dlist[i])/len(dlist[i]), file = file_output)

file_output.close()


def distance_euclid(a, b) : 
    """
    compute the distance between two vectors a and b
    """

    assert(a.shape == b.shape)
    distance = np.sqrt(np.sum(np.square(a - b)))
    assert(distance.shape == ())
    return distance

def angle_between(a, b) : 
    """
    compute the angle between two vectors a and b
    """

    lenth_a = np.sqrt(np.sum(np.square(a)))
    lenth_b = np.sqrt(np.sum(np.square(b)))
    mul_ab = np.sum(a * b)
    cos_angle = mul_ab / (lenth_a * lenth_b)
    angle = math.acos(cos_angle)

    return angle

def lyapunov_max(Y, dim, tau_delay, epsilon = 1, gama = 0.78) :
    """
    compute the max lyapunov exponent of an reconstructed phase space
    """

    num = Y.shape[0]
    log_list = []

    i = 0
    for j in range(i + 1, num) : 
        dist = distance_euclid(Y[i], Y[j])
        angle = angle_between(Y[i], Y[j])

        if (dist < epsilon) and (angle < gama) :
            dist_nearest = dist 
            
            for k in range(1, min(num - i, num - j)) :
                dist = distance_euclid(Y[i + k], Y[j + k])
                dist_evolution = dist
                i = i + k
                if (dist >= epsilon) :
                    break
            log_list.append(math.log(dist_evolution / dist_nearest))

    lyapunov_max = sum(log_list) / len(log_list)
    return lyapunov_max


Y = np.array([[1, 2, 3, 4], 
              [1, 2, 3, 3.5],
              [4, 5, 6, 7],
              [2, 2, 3, 8]])

lyapunov = lyapunov_max(Y, 4, 1)
print(lyapunov)