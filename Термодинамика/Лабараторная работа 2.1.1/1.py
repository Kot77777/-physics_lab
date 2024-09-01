#объемный расход: dv_dt
import numpy as np
import matplotlib.pyplot as plt

dv_dt1 = 0.158
I1 = np.array([125.77, 177.32, 251.2, 270.1])#мА
U1 = np.array([3.711, 5.237, 7.192, 7.736])#В
e1 = np.array([0.068, 0.143, 0.293, 0.336])#мВ
N1 = I1*U1
print(N1/200)
dT1 = np.array([2, 5, 8, 10])
sr_N1_kv = sum(N1**2)/4
sr_dTN1 = sum(dT1*N1)/4
k1 = sr_dTN1/sr_N1_kv
sr_dT1_kv = sum(dT1**2)/4
sigma1 = 0.5*((abs(sr_dT1_kv/sr_N1_kv - k1**2))**0.5)
print(k1, sigma1)

dv_dt2 = 0.06
I2 = np.array([0, 75.12, 102.18, 117.92, 126.13])#мА
U2 = np.array([0, 2.214, 3.014,3.48, 3.722])#В
e2 = np.array([0.055, 0.103, 0.124, 0.142])#мВ
N2 = I2*U2
print(N2/78)
dT2 = np.array([0, 2, 4, 6, 8])
sr_N2_kv = sum(N2[1:]**2)/4
sr_dTN2 = sum(dT2[1:]*N2[1:])/4
k2 = sr_dTN2/sr_N2_kv
sr_dT2_kv = sum(dT2[1:]**2)/4
sigma2 = 0.5*((abs(sr_dT2_kv/sr_N2_kv - k2**2))**0.5)
print(k2, sigma2)

fig, ax = plt.subplots()

t = np.polyfit(N1, dT1,  1)
f = np.poly1d(t)

plt.scatter(N1, dT1, c="g", zorder = 1, label = 'q_max')
plt.plot(np.arange(0, 2250, 100), f(np.arange(0, 2250, 100)),
         linestyle='-', alpha = 1,
         color = "g", ms=3, zorder = 0)

t2 = np.polyfit(N2, dT2, 1)
f2 = np.poly1d(t2)

plt.scatter(N2[1:], dT2[1:], c="orange", zorder = 1, label = 'q_1')
plt.plot(np.arange(0, 501, 100), f2(np.arange(0, 501, 100)),
         linestyle='-', alpha = 1,
         color = "orange", ms=3, zorder = 0)
plt.grid(linewidth = 0.5)

plt.xlabel('N, мВт', fontsize=10)
plt.ylabel('dT, K', fontsize=10)
plt.legend()
plt.savefig("График зависимости N от dT", dpi=600)

plt.show()