import numpy as np
import matplotlib.pyplot as plt

f_izm = np.array([
                102.4, 305.2, 470.2, 343, 519.7,
                694.4, 872.4, 996.9, 1171.6, 1349.6,
                1552.4, 1809.5, 2012.3, 1039.5, 1204.4
                ])

f = ((294.7 + f_izm) / 1000) * 9.81
lamda = np.array([
                 174, 181, 186, 182, 188,
                 194, 202, 208, 217, 227,
                 246, 264, 280, 212, 223
                 ]) / 100

pogr_x = lamda*(((0.5/(lamda*100))**2 + (0.5/100)**2)**0.5)

lamda2 = lamda - 1/(lamda**2)

q = lamda2
k = f

sr_q = sum(q)/15
sr_k = sum(k)/15
sr_k_kv = sum(k**2)/15
sr_kv_k = sr_k**2
sr_qk = sum(q*k)/15
sr_q_kv1 = sum(q**2)/15
sr_kv_q = sr_q**2
k1 = (sr_qk - sr_q*sr_k)/(sr_q_kv1 - sr_kv_q)
sigma_angle_1 = (1/(15**0.5))*(((sr_k_kv - sr_kv_k)/(sr_q_kv1 - sr_kv_q) - k1**2)**0.5)
#sigma_R01 = sigma_angle_1*(sr_q_kv1 - sr_kv_q)**0.5
print("Поргешность k: ", sigma_angle_1, k1)

t = np.polyfit(lamda, f,  1)
fun = np.poly1d(t)
print(t)

t2 = np.polyfit(lamda2, f,  1)
fun2 = np.poly1d(t2)
print(t2)

fig, ax = plt.subplots()

#plt.scatter(lamda, f, c="g", zorder = 1,)
plt.plot(lamda, fun(lamda),
          linestyle='-', alpha = 1,
          color = "g", ms=3, zorder = 0, label = 'f(λ) - №1 (k = 17.40 ± 0.82)')

#plt.scatter(lamda2, f, c="b", zorder = 1,)
plt.plot(lamda2, fun2(lamda2),
          linestyle='-', alpha = 1,
          color = "b", ms=3, zorder = 0, label = 'f(λ - 1/λ^2) - №2 (k = 14.67 ± 0.56)')

plt.errorbar(lamda, f, xerr=pogr_x, c="green", fmt='.', linewidth=2, capsize=6)
plt.errorbar(lamda2, f, xerr=pogr_x, c="b", fmt='.', linewidth=2, capsize=6)
plt.grid(linewidth = 0.5)
#plt.xlim(1.25, 3)
plt.ylabel('f, H', fontsize=10)
plt.legend()
plt.savefig("График зависимости f от функции λ", dpi=600)
plt.show()