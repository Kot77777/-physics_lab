import numpy as np
import matplotlib.pyplot as plt
from docutils.nodes import label

data_phi = np.array([0, np.pi / 12, np.pi / 6, np.pi / 4, np.pi / 3, np.pi * 5 / 12, np.pi / 2])

data1_weight = np.array([23.5, 22.75, 22.45, 23.2, 22.55, 22.3, 22.9])
data1_S = np.array([47, 45.5, 44.9, 46.4, 45.1, 44.6, 45.8])
data1_P = np.array([237, 217, 207, 190, 189, 171, 175])
data1_sigma = data1_P / data1_S

data2_weight = np.array([23.1, 22.8, 23.4, 22.3, 23.8, 21.8, 23.3])
data2_S = np.array([46.2, 45.6, 46.8, 44.6, 47.6, 43.6, 46.6])
data2_P = np.array([188, 162, 155, 127, 106, 98, 110])
data2_sigma = data2_P / data2_S

data3_weight = np.array([22.1, 23.6, 23.1, 22.7, 21.9, 21.1, 23.2])
data3_S = np.array([44.2, 47.2, 46.2, 45.4, 43.8, 44.2, 46.4])
data3_P = np.array([183, 178, 144, 110, 102, 99, 111])
data3_sigma = data3_P / data3_S

data_sr_weight = (data1_weight + data2_weight + data3_weight) / 3
data_sr_S = (data1_S + data2_S + data3_S) / 3
data_sr_P = (data1_P + data2_P + data3_P) / 3
data_sr_sigma = data_sr_P / data_sr_S
print(data_sr_sigma/data_sr_sigma[0])

hi_1 = 0.645
b_1 = 1.6
data1_sigma_norm = data1_sigma / data1_sigma[0]
data_sr_sigma_norm = data_sr_sigma / data_sr_sigma[0]

phi = np.arange(0, np.pi / 2, 0.01)
phi_pogr = np.arange(0, np.pi / 2 + np.pi / 12, np.pi / 12)
sigma = hi_1 / ((hi_1**2 * (np.cos(phi))**4 + (np.sin(phi))**4 + b_1 * (np.sin(phi))**2 * (np.cos(phi))**2)**0.5)
sigma_pogr = hi_1 / ((hi_1**2 * (np.cos(phi_pogr))**4 + (np.sin(phi_pogr))**4 + b_1 * (np.sin(phi_pogr))**2 * (np.cos(phi_pogr))**2)**0.5)
total = 0
for i in range(7):
    total = total + (data_sr_sigma_norm[i] - sigma_pogr[i])**2
sl = ((1/42) * total)**0.5
print(((1/42) * total)**0.5)

print(sigma_pogr)
#b_1 = ((1 / data1_sigma_norm)**2 - (np.cos(phi))**4 - (np.sin(phi))**4) / ((np.sin(phi) * np.cos(phi))**2)
#phi = np.pi / 2

plt.xlim(-0.1, 1.7)
plt.ylim(0.55, 1.1)
plt.plot(phi, sigma)

plt.plot(phi, sigma, 'r', label = 'Теоретический график')
plt.errorbar(data_phi, data1_sigma / data1_sigma[0], yerr = 2*sl, fmt = 'D', color = 'b', capsize=5, label = 'Эксперементальный график')
plt.errorbar(data_phi, data_sr_sigma / data_sr_sigma[0], yerr = 2*sl, fmt = 'o', color = 'orange', capsize=5, label = 'Эксперементальный график из средних')

plt.grid(which='both', color='gray', linestyle='--', linewidth=0.5)
plt.minorticks_on()
plt.grid(which='minor', color='lightgray', linestyle=':', linewidth=0.5)

plt.xlabel('\u03C6')
plt.ylabel('σ(\u03C6)/σ(0)')
plt.legend()

plt.savefig('Графики', dpi=300)
plt.show()

