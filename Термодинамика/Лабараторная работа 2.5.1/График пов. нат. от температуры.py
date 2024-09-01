import numpy as np
import matplotlib.pyplot as plt

T = np.array([25, 30, 35, 40, 45, 50, 55, 60])
P = np.array([219, 218, 218, 216, 215, 214, 211, 210])
dP = P*0.2*9.8 - 160.7
print(dP)
pov = dP*1.08/4
print(pov)
#sigma_pov = pov*((((0.54/pov)**2) + (0.05/1.08)**2)**0.5)

sr_T = sum(T)/6
sr_pov = sum(pov)/6
sr_pov_kv = sum(pov**2)/6
sr_kv_pov = sr_pov**2
sr_T_pov = sum(T*pov)/6
sr_T_kv1 = sum(T**2)/6
sr_kv_T = sr_T**2
k1 = (sr_T_pov - sr_T*sr_pov)/(sr_T_kv1 - sr_kv_T)
sigma_angle_1 = (1/(8**0.5))*(abs((sr_pov_kv - sr_kv_pov)/(sr_T_kv1 - sr_kv_T) - k1**2)**0.5)
print(k1, sigma_angle_1)
q = T*k1
U_F = pov + q


fig, ax = plt.subplots()

t = np.polyfit(T, pov,  1)
f = np.poly1d(t)
plt.scatter(T, pov, c="g", zorder = 1)
plt.plot(T, f(T),
         linestyle='-', alpha = 1,
         color = "orange", ms=3, zorder = 0)

plt.grid(linewidth = 0.5)

#plt.errorbar(T, pov, yerr=sigma_pov, c="b", fmt='o', linewidth=2, capsize=6)
plt.xlabel('T, ℃', fontsize=10)
plt.ylabel('\u03C3, Н/м', fontsize=10)
plt.savefig("График зависимости \u03C3 от T", dpi=600)

fig1, ax1 = plt.subplots()

t = np.polyfit(T, q,  1)
f = np.poly1d(t)

t1 = np.polyfit(T, U_F,  1)
f1 = np.poly1d(t1)

plt.scatter(T, q, c="r", zorder = 1)
plt.plot(T, f(T),
         linestyle='-', alpha = 1,
         color = "g", ms=3, zorder = 0, label = 'q(T)')

plt.scatter(T, U_F, c="orange", zorder = 1)
plt.plot(T, f1(T),
         linestyle='-', alpha = 1,
         color = "b", ms=3, zorder = 0, label = 'U/F(T)')

plt.grid(linewidth = 0.5)

#plt.errorbar(T, pov, yerr=sigma_pov, c="b", fmt='o', linewidth=2, capsize=6)
plt.xlabel('T, ℃', fontsize=10)
plt.ylabel('\u03C3, Н/м', fontsize=10)
plt.legend()
plt.savefig("Графики зависимостей \u03C3 от T", dpi=600)

plt.show()