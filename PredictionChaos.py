import math
import numpy as np
import matplotlib.pyplot as plt
import operator 
from math import log


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
    
    if (a == b).all() : 
        angle = 0
    elif (a == -b).all() : 
        angle = 3.14
    else :
        lenth_a = np.sqrt(np.sum(np.square(a)))
        lenth_b = np.sqrt(np.sum(np.square(b)))
        mul_ab = np.dot(a, b)
        cos_angle = mul_ab / (lenth_a * lenth_b)

        if cos_angle < -1 : 
            angle = -3.14
        elif cos_angle > 1 :
            angle = 0
        else : 
            angle = math.acos(cos_angle)
    return angle

def find_near_point(Y, base_point, former, num, epsilon = 1, gama = 0.78) : 

    near_point = -1
    near_dist = -1
    for cmp_point in range(base_point + 1, num) : 
        dist = distance_euclid(Y[base_point], Y[cmp_point])
        angle = angle_between(Y[former] - Y[base_point], Y[cmp_point] - Y[base_point])

        if (dist < epsilon) and (angle < gama) : 
            near_point = cmp_point
            near_dist = dist
            break

    return near_point, near_dist

def find_near_first(Y, base_point, num, epsilon = 1) : 
    
    near_first = -1
    near_dist_first = -1
    for cmp_point in range(base_point + 1, num) : 
        dist = distance_euclid(Y[base_point], Y[cmp_point])
        if (dist < epsilon) : 
            near_first = cmp_point
            near_dist_first = dist
            break

    return near_first, near_dist_first

def lyapunov_max(Y, tau_delay = 1, epsilon = 0.1, gama = 0.78) :
    """
    compute the max lyapunov exponent of an reconstructed phase space
    """

    num = Y.shape[0]
    dim = Y.shape[1]   
    log_list = []
    base_point = 0
    status = 0

    while(base_point < num) :
        if (status == 0) :  
            near_first, near_dist_first = find_near_first(Y, base_point, num, epsilon)
            if (near_first == -1) : 
                base_point = base_point + 1
            else : 
                for k in range(1, min(num - base_point, num - near_first)) : 
                    dist = distance_euclid(Y[base_point + k], Y[near_first + k])
                    if (dist >= epsilon) :
                        break

                base_point = base_point + k
                former = near_first + k
                dist_evolution = dist
                status = 1
                log_list.append(math.log(dist_evolution / near_dist_first))
                        
        else : 
            near_point, near_dist = find_near_point(Y, base_point, former, num, epsilon, gama)
            if (near_point == -1) : 
                break
            else : 
                for k in range(1, min(num - base_point, num - near_point)) : 
                    dist = distance_euclid(Y[base_point + k], Y[near_point + k])
                    if (dist >= epsilon) :
                        break
                base_point = base_point + k
                former = near_point + k
                dist_evolution = dist
                log_list.append(math.log(dist_evolution / near_dist))
    
    lyapunov_max = sum(log_list) / len(log_list)
    return lyapunov_max

def self_correlation(time_series, tau) : 
    """
    compute the self_correlation value of the time_series
    the time_series is an array of shape (N,)
    """

    num = time_series.shape[0]
    assert(tau < num)
    self_correlation = 0

    for i in range(0, num - tau) :
        self_correlation += (time_series[i] * time_series[i + tau]) / (num - tau)

    return self_correlation

def get_delay_tau(time_series) : 
    """
    get the autocorrelation values and write them in the txt
    the index of the lines minus one equals the corresponding tau value
    """

    num = time_series.shape[0]
    init_correlation = self_correlation(time_series, 0)
    ratio = 1 - np.exp(-1)

    f = open("tau_delay.txt", "w")
    for tau in range(0, num) : 
        correlation = self_correlation(time_series, tau)
        print(correlation, file = f)
    f.close()

    return init_correlation * ratio

def relevance_integral(time_series, embed_m, tau, r = 0.001) : 
    num = time_series.shape[0]
    num_reconstructed = num - (embed_m - 1) * tau

    # initialize the phase space
    cnt = 0
    Y_phase = np.zeros((num_reconstructed, embed_m))

    # reconstruct the phase space
    for i in range(num_reconstructed) : 
        for j in range(embed_m) : 
            Y_phase[i][j] = time_series[i + j * tau]

    # compute the relevance_integral
    for i in range(num_reconstructed) : 
        for j in range(num_reconstructed) : 
            dist = distance_euclid(Y_phase[i], Y_phase[j])
            if (dist <= r) :
                cnt += 1

    relevance_integral = cnt / (num_reconstructed * num_reconstructed)
    return Y_phase, relevance_integral

def get_the_dm(time_series, tau = 1, r = 0.01) :
    """
    increase the value of embed_m and record the changes of dm
    the value of embed_m should range between [2, (N - 1) / tau + 1)
    """

    num = time_series.shape[0]
    upper_embed_m = min(math.floor((num - 1) / tau) + 1, 100)
    embed_m_list = []
    dm_list = []

    f = open("record_dm.txt", "w")
    for embed_m in range(2, upper_embed_m) : 
        _, relevance = relevance_integral(time_series, embed_m, tau, r)
        dm = math.log(relevance) / math.log(r)
        embed_m_list.append(embed_m)
        dm_list.append(dm_list)
        print(dm, file = f)
    f.close()
    
    return embed_m_list, dm_list

def get_ref_point(Y_phase, margin = 1) : 
    """
    get the reference points which are most similar to current ones
    """

    num = Y_phase.shape[0]
    embed_m = Y_phase.shape[1]
    unit = ()
    units_ref = []
    
    for i in range(num - 1) :
        dist_similar  = distance_euclid(Y_phase[i], Y_phase[num - 1])
        angle_similar = angle_between(Y_phase[i], Y_phase[num - 1])
        unit = (i, dist_similar, angle_similar)
        units_ref.append(unit)
    
    # ensure the least-square has solution
    num_ref = embed_m + 1 + margin
    if num_ref > (num - 1) : 
        print("the time series is too short")
        return -1
    
    units_ref = sorted(units_ref, key = operator.itemgetter(2))
    units_ref = units_ref[0 : num_ref]
    units_ref = sorted(units_ref, key = operator.itemgetter(1))
    units_ref = units_ref[0 : embed_m + 1]

    return units_ref

def get_ref_weight(units_ref) : 
    """
    compute the corresponding weight of ref points
    """

    num_ref = len(units_ref)
    p_weight = []

    for i in range(num_ref) : 
        temp = 0.5 * units_ref[i][1] + 0.5 * units_ref[i][2]
        p_weight.append([units_ref[i][0], temp])
    
    p_weight = sorted(p_weight, key = operator.itemgetter(1))
    p_weight_min = p_weight[0][1]
    sum_p_weight = 0
    for i in range(num_ref) :
        sum_p_weight += math.exp(-(p_weight[i][1] - p_weight_min))

    for i in range(num_ref) : 
        p_weight[i][1] = math.exp(-(p_weight[i][1] - p_weight_min)) / sum_p_weight

    return p_weight

def get_XY_wls(Y_phase, p_weight) : 
    """
    get the data matrix X, Y for the wls
    """

    num = Y_phase.shape[0]
    embed_dim = Y_phase.shape[1]
    num_ref = len(p_weight)

    # for the purpose of convenience, actually the num_ref = embed_dim + 1
    X_data = np.ones((num_ref, embed_dim + 1))
    Y_data = np.ones((num_ref, embed_dim))
    Weight = np.zeros((num_ref, num_ref))

    for i in range(num_ref) :
        p_index =  p_weight[i][0]
        X_data[i, 0 : embed_dim] = Y_phase[p_index]
        Y_data[i, 0 : embed_dim] = Y_phase[p_index + 1]
        Weight[i][i] = p_weight[i][1]
    
    return X_data, Y_data, Weight

def wls_estimate(X, Y, W) : 
    """
    implement the weighted least square for the equation XP = Y
    with the weight matrix W
    """

    detMat = np.dot(np.dot(X.T, W), X)
    if (np.linalg.det(detMat) == 0) : 
        print("Error, the matrix can not be inversed")
        return -1
    
    detMat_inv = np.linalg.inv(detMat)
    P = np.dot(np.dot(np.dot(detMat_inv, X.T), W), Y)

    return P

def prediction_phase_space(Y_phase, units_ref) : 
    """
    predict the next point of the Y_phase and return the next number in time series
    """

    p_weight = get_ref_weight(units_ref)
    
    # use the weighted least square to estimate
    # get the X and Y according to the form of wls
    X_data, Y_data, Weight = get_XY_wls(Y_phase, p_weight)

    # get the parameters matrix P
    Para = wls_estimate(X_data, Y_data, Weight)

    # use parameters matrix P to predict the next point
    Y_prev = np.ones((1, Y_phase.shape[1] + 1))
    Y_prev[0, 0: Y_phase.shape[1]] = Y_phase[Y_phase.shape[0] - 1]
    Y_predict = np.dot(Y_prev, Para)

    # return next_point
    return Y_predict

def load_data() : 
    """
    the data in the txt need to be one column like Pm2_data.txt
    """
    data = []
    f = open("test_ampli.txt", "r")
    data_str = f.readlines()
    f.close()

    for i in range(len(data_str)) : 
        temp = data_str[i].rstrip("\n")
        temp = float(temp)
        data.append(temp)

    time_series = np.array(data)

    min_value = np.min(time_series)
    max_value = np.max(time_series)
    time_series = (time_series - min_value) / (max_value - min_value)
    return time_series, min_value, max_value

def reconstruct(time_series, embed_m, tau) :
    num = time_series.shape[0]
    num_reconstructed = num - (embed_m - 1) * tau

    # initialize the phase space
    Y_phase = np.zeros((num_reconstructed, embed_m))

    # reconstruct the phase space
    for i in range(num_reconstructed) : 
        for j in range(embed_m) : 
            Y_phase[i][j] = time_series[i + j * tau]

    return Y_phase

def batches_generate(time_series) : 
    """
    generate the batches for times_series
    """
    batches = []
    min_max = []
    num = len(time_series)

    for i in range(num - 200 + 1) :
        temp_batch = time_series[i : i + 200]
        min_value = np.min(temp_batch)
        max_value = np.max(temp_batch)
        #temp_batch = (temp_batch - min_value) / (max_value - min_value)
        min_max.append((min_value, max_value))
        batches.append(temp_batch)

    return batches, min_max

### The steps of prediction ###

# 1. load the data of time series
time_series, min_val, max_val = load_data()
print("The min value is %f and the max value is %f" % (min_val, max_val))

# 2. generate the batches of data, each batch has 200 elments
#batches, min_max = batches_generate(time_series)
#print(len(batches))

# 2. confirm the value of tau 
# when the value in tau_delay.txt is most near to the init_condition
# tau is the corresponding iteration value
# for Pm2_data.txt, tau = 2

init_condition = get_delay_tau(time_series)
print("The initial value of self-correlation is: " + str(init_condition))

# 3. confirm the value of embed dimension
# according to the record_dm.txt, chose m = 20 as the embed dimension
#embed_m_list, dm_list = get_the_dm(time_series, tau = 15, r = 0.2)

# 4. reconstruct the phase space
Y_phase = reconstruct(time_series, embed_m = 6, tau = 15)

# 5. get the reference points
units_ref = get_ref_point(Y_phase, margin = 3)

# 6. predict the next point
Y_predict = prediction_phase_space(Y_phase, units_ref)
print(Y_predict[0][-1] * (max_val-min_val) + min_val)

# 7. get the max lyapunov exponent
#ly_exp = lyapunov_max(Y_phase, tau_delay = 1, epsilon = 0.01)
#print(ly_exp)







"""
LLT testcase
"""
"""
Y_phase = np.array([[1, 2],
                   [1, 3],
                   [2, 1],
                   [2, 2],
                   [2, 2.5]])    
print(Y_phase)

units_ref = get_ref_point(Y_phase)
print(units_ref)
p_weight = get_ref_weight(units_ref)
print(p_weight)

X_data, Y_data, Weight = get_XY_wls(Y_phase, p_weight)
print(X_data)
print(Y_data)
print(Weight)

P = wls_estimate(X_data, Y_data, Weight)

Y_predict = prediction_phase_space(Y_phase, units_ref)

print(Y_predict)
"""