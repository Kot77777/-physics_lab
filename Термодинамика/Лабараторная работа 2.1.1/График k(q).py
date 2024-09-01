import numpy as np
import matplotlib.pyplot as plt

k = np.array([1/4.699, 1/15.22])
q = np.array([0.2, 0.078])

sr_q = sum(q)/6
sr_k = sum(k)/6
sr_k_kv = sum(k**2)/6
sr_kv_k = sr_k**2
sr_qk = sum(q*k)/6
sr_q_kv1 = sum(q**2)/6
sr_kv_q = sr_q**2
k1 = (sr_qk - sr_q*sr_k)/(sr_q_kv1 - sr_kv_q)
sigma_angle_1 = (1/(6**0.5))*(((sr_k_kv - sr_kv_k)/(sr_q_kv1 - sr_kv_q) - k1**2)**0.5)
sigma_R01 = sigma_angle_1*(sr_q_kv1 - sr_kv_q)**0.5


fig, ax = plt.subplots()

t = np.polyfit(q, k,  1)
f = np.poly1d(t)

print("Погрешности: ", sigma_angle_1, sigma_R01)
print(k1, -f(0))

plt.scatter(q, k, c="g", zorder = 1,)
plt.plot(q, f(q),
         linestyle='-', alpha = 1,
         color = "g", ms=3, zorder = 0)
plt.grid(linewidth = 0.5)

plt.xlabel('1/k, Вт/К', fontsize=10)
plt.ylabel('q, г/c', fontsize=10)
plt.savefig("График зависимости 1k от q", dpi=600)

plt.show()