import numpy as np 
a_list = [1, 2, 3]
b_list = [1, 2]
a = np.array(a_list).reshape(1, -1)
b = np.array(b_list).reshape(1, -1)

for i in range(5) : 
    i = i + 3
    print(i)
