import numpy as np
import matplotlib.pyplot as plt

k = np.log(np.array([0.025, 0.0311, 0.026, 0.028, 0.0276, 0.031]))
T = np.log(np.array([23, 30, 40, 50, 60, 70]))

k_t = np.log(np.array([0.02485, 0.02553, 0.02621, 0.02836, 0.03026]))
T_t = np.log(np.array([17, 27, 37, 67, 97]))

sr_T = sum(T)/6
sr_R = sum(k)/6
sr_TR = sum(T*k)/6
sr_T_kv = sum(T**2)/6
sr_kv_T = sr_T**2
sr_R_kv = sum(k**2)/6
sr_kv_R = sr_R**2
k1 = (sr_TR - sr_T*sr_R)/(sr_T_kv - sr_kv_T)
sigma_angle = (1/(6**0.5))*(((sr_R_kv - sr_kv_R)/(sr_T_kv - sr_kv_T) - k1**2)**0.5)
print(k1, sigma_angle)

fig, ax = plt.subplots()

t = np.polyfit(T, k, 1)
f = np.poly1d(t)
print(f)

t1 = np.polyfit(T_t, k_t, 1)
f1 = np.poly1d(t1)
print(f1)

#ax.scatter(T, k, c="r", zorder = 1)
ax.plot(T_t, f1(T_t), linestyle='--', alpha = 1,
         color = 'black', ms=5, zorder = 0, label = "Теоретический график")

ax.scatter(T, k, c="r", zorder = 1)
ax.plot(T, f(T), linestyle='-', alpha = 1,
         color = 'orange', ms=5, zorder = 0, label = "Экспериментальный график")
ax.set_ylim(-4, -3)
ax.grid(linewidth = 0.5)

plt.ylabel('ln(k), Вт/(м*с)', fontsize=10)
plt.xlabel('ln(T), ℃', fontsize=10)
plt.legend()
plt.savefig("График зависимости ln(k) от ln(T)", dpi=600)

plt.show()