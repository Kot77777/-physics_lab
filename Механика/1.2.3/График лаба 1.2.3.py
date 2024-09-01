import numpy as np
import matplotlib.pyplot as plt

T = np.array([3.116, 3.124, 3.144, 3.196, 3.315, 3.472, 3.648, 3.861, 4.209])
I = (0.442*2.5094*(T**2))
print(I)
h_kv = (0.1*np.array([0, 5, 10, 20, 30, 40, 50, 60, 75]))**2
print(sum(T)/9)
sr_Ih_kv = sum(I*h_kv)/9
sr_I = sum(I)/9
sr_h_kv = sum(h_kv)/9
sr_h_kv_kv = sum(h_kv**2)/9
kv_sr_h_kv = sr_h_kv**2
sr_I_kv = sum(I**2)/9
kv_sr_I = sr_I**2
sr_I_sr_h_kv = sr_I*sr_h_kv

k = (sr_Ih_kv - sr_I_sr_h_kv)/(sr_h_kv_kv - kv_sr_h_kv)
b = sr_I - k*sr_h_kv
print(k, b, sr_I)

sigma_sl_k = (1/3)*(((sr_I_kv - kv_sr_I)/(sr_h_kv_kv - kv_sr_h_kv) - k**2)**0.5)
sigma_sl_b = sigma_sl_k*((sr_h_kv_kv - kv_sr_h_kv)**0.5)
print(sigma_sl_k, sigma_sl_b)

fig, ax = plt.subplots()

t1 = np.polyfit(h_kv, I, 1)
f1 = np.poly1d(t1)

plt.scatter(h_kv, I, c = "r")
plt.plot(h_kv, f1(h_kv), label = 'y = 0,16x + 10,79',
         linestyle='-', mfc = 'r',mec = 'r', alpha = 1,
         color = 'blue', ms=5)
plt.errorbar(h_kv, I, xerr=0, yerr=0.13, fmt='|', color='black')
plt.xlim([0, 60])
plt.grid(linewidth = 0.5)
plt.xlabel('h^2, cм', fontsize=10)
plt.ylabel('I*10^(-3), кг*м^2', fontsize=10)

plt.legend()
plt.show()