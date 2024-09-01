import numpy as np
import matplotlib.pyplot as plt

d_s = np.array([2.2, 2.3, 2.9, 3.1, 3.4, 3.5, 3.7, 3.7])
d_mv = np.array([12, 14.8, 16.4, 19.2, 26, 29.8, 35.2, 42.4])*10
lamd = np.array([210, 221, 228, 231, 244, 255, 266, 276])/100
d_T = (d_mv)/(40*5)
print(d_T)
A = 14.67*(0.5*lamd**2 + 1/lamd - 1.5)/3
print(A)

pogr_x = lamd*(((0.5/(lamd*100))**2 + (0.5/100)**2)**0.5)
pogr_A = A*((0.019**2 + (((lamd-1/(lamd**2))*pogr_x)/(0.5*lamd**2 + 1/lamd))**2)**0.5)
pogr_dT = d_T*0.02

print(pogr_A[7]/A[7])

q = A
k = d_T

sr_q = sum(q)/8
sr_k = sum(k)/8
sr_k_kv = sum(k**2)/8
sr_kv_k = sr_k**2
sr_qk = sum(q*k)/8
sr_q_kv1 = sum(q**2)/8
sr_kv_q = sr_q**2
k1 = (sr_qk - sr_q*sr_k)/(sr_q_kv1 - sr_kv_q)
sigma_angle_1 = (1/(8**0.5))*(((sr_k_kv - sr_kv_k)/(sr_q_kv1 - sr_kv_q) - k1**2)**0.5)

print("Поргешность k: ", sigma_angle_1, k1)

fig, ax = plt.subplots()

t = np.polyfit(A, d_T,  1)
f = np.poly1d(t)
print(t)

#plt.scatter(A, d_T, c="g", zorder = 1,)
plt.plot(A, f(A),
         linestyle='-', alpha = 1,
         color = "g", ms=3, zorder = 0, label = 'k = 0.21 ± 0.01')
plt.errorbar(A, d_T, xerr=pogr_A, yerr=pogr_dT, c="green", fmt='.', linewidth=2, capsize=3)
plt.grid(linewidth = 0.5)
plt.xlabel('А, Дж', fontsize=10)
plt.ylabel('dT, K', fontsize=10)
plt.legend()
plt.savefig("График зависимости dT от A", dpi=600)

plt.show()
