import numpy as np
import matplotlib.pyplot as plt

df1 = np.array([234, 458, 670, 1100, 1315])

df2 = np.array([252, 465, 681, 1117, 1335])

df3 = np.array([255, 471, 690, 1134, 1356])

df4 = np.array([259, 480, 703, 1154, 1379])

df5 = np.array([266, 490, 716, 1174, 1402])
k = np.array([1, 2, 3, 5, 6])

sr_k = sum(k)/5
sr_df1 = sum(df1)/5
sr_df1_kv = sum(df1**2)/5
sr_kv_df1 = sr_df1**2
sr_kdf1 = sum(k*df1)/5
sr_k_kv = sum(k**2)/5
sr_kv_k = sr_k**2
k1 = (sr_kdf1 - sr_k*sr_df1)/(sr_k_kv - sr_kv_k)
sigma_angle_1 = (1/(5**0.5))*(((sr_df1_kv - sr_kv_df1)/(sr_k_kv - sr_kv_k) - k1**2)**0.5)
print("1", k1, sigma_angle_1)

sr_df2 = sum(df2)/5
sr_df2_kv = sum(df2**2)/5
sr_kv_df2 = sr_df2**2
sr_kdf2 = sum(k*df2)/5
k2 = (sr_kdf2 - sr_k*sr_df2)/(sr_k_kv - sr_kv_k)
sigma_angle_2 = (1/(5**0.5))*(((sr_df2_kv - sr_kv_df2)/(sr_k_kv - sr_kv_k) - k2**2)**0.5)
print("2", k2, sigma_angle_2)

sr_df3 = sum(df3)/5
sr_df3_kv = sum(df3**2)/5
sr_kv_df3 = sr_df3**2
sr_kdf3 = sum(k*df3)/5
k3 = (sr_kdf3 - sr_k*sr_df3)/(sr_k_kv - sr_kv_k)
sigma_angle_3 = (1/(5**0.5))*(((sr_df3_kv - sr_kv_df3)/(sr_k_kv - sr_kv_k) - k3**2)**0.5)
print("3", k3, sigma_angle_3)

sr_df4 = sum(df4)/5
sr_df4_kv = sum(df4**2)/5
sr_kv_df4 = sr_df4**2
sr_kdf4 = sum(k*df4)/5
k4 = (sr_kdf4 - sr_k*sr_df4)/(sr_k_kv - sr_kv_k)
sigma_angle_4 = (1/(5**0.5))*(((sr_df4_kv - sr_kv_df4)/(sr_k_kv - sr_kv_k) - k4**2)**0.5)
print("4", k4, sigma_angle_4)

sr_df5 = sum(df5)/5
sr_df5_kv = sum(df5**2)/5
sr_kv_df5 = sr_df5**2
sr_kdf5 = sum(k*df5)/5
k5 = (sr_kdf5 - sr_k*sr_df5)/(sr_k_kv - sr_kv_k)
sigma_angle_5 = (1/(5**0.5))*(((sr_df5_kv - sr_kv_df5)/(sr_k_kv - sr_kv_k) - k5**2)**0.5)
print("5", k5, sigma_angle_5)
c_2L = np.array([k1, k2, k3, k4, k5])
sigma_c_2L = np.array([sigma_angle_1, sigma_angle_2, sigma_angle_3, sigma_angle_4, sigma_angle_5])
sigma_c = c_2L*2*0.8*(((sigma_c_2L/c_2L)**2 - (1/800)**2)**0.5)
print(c_2L*2*0.8)
print(sigma_c)

fig, ax = plt.subplots()
plt.gcf().set_size_inches(15, 7.5)
t1 = np.polyfit(k, df1,  1)
f1 = np.poly1d(t1)

t2 = np.polyfit(k, df2,  1)
f2 = np.poly1d(t2)

t3 = np.polyfit(k, df3,  1)
f3 = np.poly1d(t3)

t4 = np.polyfit(k, df4,  1)
f4 = np.poly1d(t4)

t5 = np.polyfit(k, df5,  1)
f5 = np.poly1d(t5)

plt.plot(k, f1(k), label = 'T1 = 21.6℃',
         linestyle='-', alpha = 1,
         color = 'purple', ms=5, zorder = 0, marker = "*")
plt.grid(linewidth = 0.5)

plt.plot(k, f2(k), label = 'T2 = 30℃',
         linestyle='-', alpha = 1,
         color = 'blue', ms=5, zorder = 0, marker = "*")
plt.grid(linewidth = 0.5)

plt.plot(k, f3(k), label = 'T3 = 40℃',
         linestyle='-', alpha = 1,
         color = 'green', ms=5, zorder = 0, marker = "*")
plt.grid(linewidth = 0.5)

plt.plot(k, f4(k), label = 'T4 = 50℃',
         linestyle='-', alpha = 1,
         color = 'orange', ms=5, zorder = 0, marker = "*")
plt.grid(linewidth = 0.5)

plt.plot(k, f5(k), label = 'T5 = 60℃',
         linestyle='-', alpha = 1,
         color = 'red', ms=5, zorder = 0, marker = "*")
plt.grid(linewidth = 0.5)
plt.legend()
plt.grid(linewidth = 0.5)

plt.ylabel('f(k+1) - f(1), Гц', fontsize=10)
plt.xlabel('k', fontsize=10)
plt.savefig("График зависимости частоты от номера резонанса", dpi=600)


fig, ax1 = plt.subplots()
plt.gcf().set_size_inches(7.5, 6)
c = (c_2L*2*0.8)**2
T_k = np.array([21.6, 30, 40, 50, 60]) + 273

sr_T_k = sum(T_k)/5
sr_c = sum(c)/5
sr_c_kv = sum(c**2)/5
sr_kv_c = sr_c**2
sr_T_kc = sum(T_k*c)/5
sr_T_k_kv = sum(T_k**2)/5
sr_kv_T_k = sr_T_k**2
k = sr_T_kc/sr_T_k_kv
sigma_angle = (1/(5**0.5))*(((sr_c_kv)/(sr_T_k_kv) - k**2)**0.5)
print("k", k, sigma_angle)

t8 = np.polyfit(T_k, c,  1)
f8 = np.poly1d(t8)
print(f8)

plt.scatter(T_k, c)
plt.plot(T_k, f8(T_k),
         linestyle='-', alpha = 1,
         color = 'red', ms=5, zorder = 0)
plt.grid(linewidth = 0.5)
plt.grid(linewidth = 0.5)

plt.ylabel('c^2, (м/с)^2', fontsize=10)
plt.xlabel('T, К', fontsize=10)
plt.savefig("График зависимости квадрата скорости от температуры", dpi=600)

plt.show()