import numpy as np
import matplotlib.pyplot as plt

d_mv1 = np.array([35, 33.6, 32, 30, 28.4, 28.4, 27.6, 25.2, 24.2])
s1 = np.array([0, 31.6, 44.6, 56.4, 69.8, 75.8, 84.8, 93.6, 99.8])
d_T1 = (d_mv1)/(40*5)
ln_T1 = np.log(d_T1)

d_mv2 = np.array([30.1, 28.6, 27, 27, 25.2, 24.4, 24.4, 20.2, 20.4, 19, 18.8, 18.2])
s2 = np.array([-43.2, -38, -32.8, -26.8, -20.2, -14.6, -3.6, 18.2, 34.6, 47.6, 56.6, 63.8]) + 43.2
d_T2 = (d_mv2)/(40*5)
ln_T2 = np.log(d_T2)

d_mv3 = np.array([25.4, 20.2, 19.2, 17.6, 18.4, 16.6, 15.4, 13.8])
s3 = np.array([-64.4, -23, -11.2, -0.6, 8.4, 18, 51.6, 66.6]) + 64.4
d_T3 = (d_mv3)/(40*5)
ln_T3 = np.log(d_T3)

q = s3
k = ln_T3

sr_q = sum(q)/8
sr_k = sum(k)/8
sr_k_kv = sum(k**2)/8
sr_kv_k = sr_k**2
sr_qk = sum(q*k)/8
sr_q_kv1 = sum(q**2)/8
sr_kv_q = sr_q**2
k1 = (sr_qk - sr_q*sr_k)/(sr_q_kv1 - sr_kv_q)
sigma_angle_1 = (1/(8**0.5))*(((sr_k_kv - sr_kv_k)/(sr_q_kv1 - sr_kv_q) - k1**2)**0.5)
sigma_R01 = sigma_angle_1*(sr_q_kv1 - sr_kv_q)**0.5
print("Поргешность k: ", sigma_angle_1, sigma_R01)

fig, ax = plt.subplots()

t1 = np.polyfit(s1, ln_T1,  1)
f1 = np.poly1d(t1)
print(t1)

t2 = np.polyfit(s2, ln_T2,  1)
f2 = np.poly1d(t2)
print(t2)

t3 = np.polyfit(s3, ln_T3,  1)
f3 = np.poly1d(t3)
print(t3)

#plt.scatter(s1, ln_T1, c="g", zorder = 1,)
plt.plot(s1, f1(s1),
         linestyle='-', alpha = 1,
         color = "g", ms=3, zorder = 0, label = '\u03BB = 2.44 (b = -1.70 ± 0.01)')
plt.errorbar(s1, ln_T1, yerr=0.02, c="g", fmt='.', linewidth=2, capsize=6)

#plt.scatter(s2, ln_T2, c="orange", zorder = 1,)
plt.plot(s2, f2(s2),
         linestyle='-', alpha = 1,
         color = "orange", ms=3, zorder = 0, label = '\u03BB = 2.66 (b = -1.94 ± 0.01)')
plt.errorbar(s2, ln_T2, yerr=0.02, c="orange", fmt='.', linewidth=2, capsize=6)

#plt.scatter(s3, ln_T3, c="b", zorder = 1,)
plt.plot(s3, f3(s3),
         linestyle='-', alpha = 1,
         color = "b", ms=3, zorder = 0, label = '\u03BB = 2.76 (b = -2.09 ± 0.01)')
plt.errorbar(s3, ln_T3, yerr=0.02, c="b", fmt='.', linewidth=2, capsize=6)

plt.grid(linewidth = 0.5)
plt.ylabel('ln(dT/T0)', fontsize=10)
plt.xlabel('t, c', fontsize=10)

print(f1(0), f2(0), f3(0))
plt.legend()
plt.savefig("График зависимости ln(dT) от t", dpi=600)
plt.show()