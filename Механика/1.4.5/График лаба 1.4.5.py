import numpy as np
import matplotlib.pyplot as plt

n = np.array(range(0, 9))
nu_T1 = np.array([0, 134.6, 270, 404.4, 541, 677.3, 815, 951.2, 1092])
nu_T2 = np.array([0, 154, 306, 462, 614, 772, 925, 1083, 1237])
nu_T3 = np.array([0, 178.6, 358, 537, 718, 898, 1078, 1257, 1439])
nu_T4 = np.array([0, 200.2, 400, 601, 801, 1003, 1205, 1407, 1610])
nu_T5 = np.array([0, 219.5, 440, 659, 882, 1100, 1325, 1541, 1768])

size = [10, 10, 10, 10, 10, 10, 10, 10, 10]
plt.scatter(n, nu_T1, size, c="black")
plt.scatter(n, nu_T2, size, c='black')
plt.scatter(n, nu_T3, size, c='black')
plt.scatter(n, nu_T4, size, c='black')
plt.scatter(n, nu_T5, size, c='black')

t1 = np.polyfit(n, nu_T1, 1)
f1 = np.poly1d(t1)
plt.plot(n, f1(n), label = 'T1 = 10,27 Н; k1 = 135,9 \u00B1 0,77', alpha=0.5)

t2 = np.polyfit(n, nu_T2, 1)
f2 = np.poly1d(t2)
plt.plot(n, f2(n), label = 'T2 = 13,5 Н; k2 = 154,4 \u00B1 0,27' , alpha=0.5)

t3 = np.polyfit(n, nu_T3, 1)
f3 = np.poly1d(t3)
plt.plot(n, f3(n), label = 'T3 = 18,3 Н; k3 = 179,6 \u00B1 0,19', alpha=0.5)

t4 = np.polyfit(n, nu_T4, 1)
f4 = np.poly1d(t4)
plt.plot(n, f4(n), label = 'T4 = 23 Н; k4 = 200,9 \u00B1 0,26', alpha=0.5)

t5 = np.polyfit(n, nu_T5, 1)
f5 = np.poly1d(t5)
plt.plot(n, f5(n), label = 'T5 = 27,7 Н; k5 = 220,5 \u00B1 0,33', alpha=0.5)

plt.xlabel('n', fontsize=10, fontweight='bold')
plt.ylabel('v, Гц', fontsize=10, fontweight='bold')

plt.grid(linewidth = 0.5)
plt.legend()
plt.show()