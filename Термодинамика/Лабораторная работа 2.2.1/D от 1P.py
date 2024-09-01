import numpy as np
import matplotlib.pyplot as plt

D = 2*np.array([6.16, 2.797, 1.972, 1.544])
P_ob = 1/np.array([40, 80, 120, 160])

sr_P_ob = sum(P_ob)/4
sr_D = sum(D)/4
sr_D_kv = sum(D**2)/4
sr_kv_D = sr_D**2
sr_P_obD = sum(P_ob*D)/4
sr_P_ob_kv = sum(P_ob**2)/4
sr_kv_P_ob = sr_P_ob**2
k = sr_P_obD/sr_P_ob_kv
sigma_angle = (1/(4**0.5))*(((sr_D_kv)/(sr_P_ob_kv) - k**2)**0.5)
print("k", k, sigma_angle)

fig, ax = plt.subplots()

t = np.polyfit(P_ob, D,  1)
f = np.poly1d(t)
print(f)

plt.scatter(P_ob, D, c="g", zorder = 1,)
plt.plot(P_ob, f(P_ob),
         linestyle='-', alpha = 1,
         color = "g", ms=3, zorder = 0)
plt.grid(linewidth = 0.5)

plt.xlabel('D, см^2/c', fontsize=10)
plt.ylabel('q, торр^(-1)', fontsize=10)
plt.savefig("График зависимости D от q", dpi=600)

plt.show()