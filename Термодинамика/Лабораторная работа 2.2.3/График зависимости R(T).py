import numpy as np
import matplotlib.pyplot as plt

T1 = 23
I_1 = (10**(-3))*np.array([10.16, 19.627, 29.5, 40.53, 49.8, 60.88]) #мА
U_n1 = np.array([0.207, 0.4, 0.6, 0.83, 1.026, 1.26])#В
R_n1 = U_n1/I_1
Q1 = U_n1*I_1

t1 = np.polyfit(Q1, R_n1, 1)
f1 = np.poly1d(t1)


T2 = 30
I_2 = (10**(-3))*np.array([10.15, 19.08, 31.56, 40.73, 50, 60.47]) #мА
U_n2 = np.array([0.212, 0.398, 0.66, 0.86, 1.057, 1.28])#В
R_n2 = U_n2/I_2
Q2 = U_n2*I_2

t2 = np.polyfit(Q2, R_n2, 1)
f2 = np.poly1d(t2)


T3 = 40
I_3 = (10**(-3))*np.array([10.14, 19.04, 29.33, 40.345, 50.08, 60.04]) #мА
U_n3 = np.array([0.218, 0.41, 0.634, 0.876, 1.092, 1.317])#В
R_n3 = U_n3/I_3
Q3 = U_n3*I_3

t3 = np.polyfit(Q3, R_n3, 1)
f3 = np.poly1d(t3)


T4 = 50
I_4 = (10**(-3))*np.array([10.124, 21.84, 31.288, 39.895, 50.53, 60.272]) #мА
U_n4 = np.array([0.225, 0.487, 0.699, 0.894, 1.138, 1.365])#В
R_n4 = U_n4/I_4
Q4 = U_n4*I_4

t4 = np.polyfit(Q4, R_n4, 1)
f4 = np.poly1d(t4)


T6 = 60
I_6 = (10**(-3))*np.array([10.107, 20.726, 29.048, 39.637, 50.703, 60.512]) #мА
U_n6 = np.array([0.232, 0.476, 0.669, 0.916, 1.178, 1.414])#В
R_n6 = U_n6/I_6
Q6 = U_n6*I_6

t6 = np.polyfit(Q6, R_n6, 1)
f6 = np.poly1d(t6)


T7 = 70
I_7 = (10**(-3))*np.array([10.09, 20.657, 30.972, 39.383, 50.877, 61.702]) #мА
U_n7 = np.array([0.239, 0.49, 0.736, 0.939, 1.218, 1.486])#В
R_n7 = U_n7/I_7
Q7 = U_n7*I_7

t7 = np.polyfit(Q7, R_n7, 1)
f7 = np.poly1d(t7)

R = np.array([f1(0), f2(0), f3(0), f4(0), f6(0), f7(0)])
T = np.array([T1, T2, T3, T4, T6, T7])
sr_T = sum(T)/6
sr_R = sum(R)/6
sr_TR = sum(T*R)/6
sr_T_kv = sum(T**2)/6
sr_kv_T = sr_T**2
sr_R_kv = sum(R**2)/6
sr_kv_R = sr_R**2
k = (sr_TR - sr_T*sr_R)/(sr_T_kv - sr_kv_T)
sigma_angle = (1/(6**0.5))*(((sr_R_kv - sr_kv_R)/(sr_T_kv - sr_kv_T) - k**2)**0.5)
print(k, sigma_angle)

fig, ax = plt.subplots()

t = np.polyfit(T, R, 1)
f = np.poly1d(t)

plt.scatter(T, R, c="r", zorder = 1)
plt.plot(T, f(T),
         linestyle='-', alpha = 1,
         color = "b", ms=5, zorder = 0)
plt.grid(linewidth = 0.5)

plt.ylabel('R, Ом', fontsize=10)
plt.xlabel('T, ℃', fontsize=10)
plt.savefig("График зависимости сопротивления от температуры", dpi=600)

plt.show()
