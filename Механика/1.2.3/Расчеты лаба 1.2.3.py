import numpy as np

T1 = (1/20)*np.array([74.12, 74.01, 74.37, 73.38, 73.85])
T2 = (1/20)*np.array([84.44, 84.49, 84.38, 84.16, 84.16])
T3 = (1/20)*np.array([76.01, 75.76, 75.70, 76.34, 75.98])

sr_znach1 = sum(T1)/5
sigma_sl_1 = ((1/20)*sum([(i - sr_znach1)**2 for i in T1]))**0.5
print((sigma_sl_1**2 + 0.001*2)**0.5, sr_znach1)

sr_znach2 = sum(T2)/5
sigma_sl_2 = ((1/20)*sum([(i - sr_znach2)**2 for i in T2]))**0.5
print((sigma_sl_2**2 + 0.001*2)**0.5, sr_znach2)

sr_znach3 = sum(T3)/5
sigma_sl_3 = ((1/20)*sum([(i - sr_znach3)**2 for i in T3]))**0.5
print((sigma_sl_3**2 + 0.001*2)**0.5, sr_znach3)


