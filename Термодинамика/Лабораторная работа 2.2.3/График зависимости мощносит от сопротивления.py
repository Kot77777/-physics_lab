import numpy as np
import matplotlib.pyplot as plt

T1 = 23
I_1 = (10**(-3))*np.array([10.16, 19.627, 29.5, 40.53, 49.8, 60.88]) #мА
U_n1 = np.array([0.207, 0.4, 0.6, 0.83, 1.026, 1.26])#В
R_n1 = U_n1/I_1
Q1 = U_n1*I_1

print(Q1)
print(R_n1)

fig1, ax = plt.subplots()

t1 = np.polyfit(Q1, R_n1, 1)
f1 = np.poly1d(t1)

print("t1 = ", t1)
print("f1 = ", f1)

plt.scatter(Q1, R_n1, c="r", zorder = 1)
plt.plot(Q1, f1(Q1), label = 'T1 = 23℃',
         linestyle='-', alpha = 1,
         color = 'blue', ms=5, zorder = 0)
plt.grid(linewidth = 0.5)

sr_Q1 = sum(Q1)/6
sr_R_n1 = sum(R_n1)/6
sr_R_n1_kv = sum(R_n1**2)/6
sr_kv_R_n1 = sr_R_n1**2
sr_QR1 = sum(Q1*R_n1)/6
sr_Q_kv1 = sum(Q1**2)/6
sr_kv_Q1 = sr_Q1**2
k1 = (sr_QR1 - sr_Q1*sr_R_n1)/(sr_Q_kv1 - sr_kv_Q1)
sigma_angle_1 = (1/(6**0.5))*(((sr_R_n1_kv - sr_kv_R_n1)/(sr_Q_kv1 - sr_kv_Q1) - k1**2)**0.5)
sigma_R01 = sigma_angle_1*(sr_Q_kv1 - sr_kv_Q1)**0.5
print("Погрешности: ", sigma_angle_1, sigma_R01)
print(k1, f1(0))
print()


T2 = 30
I_2 = (10**(-3))*np.array([10.15, 19.08, 31.56, 40.73, 50, 60.47]) #мА
U_n2 = np.array([0.212, 0.398, 0.66, 0.86, 1.057, 1.28])#В
R_n2 = U_n2/I_2
Q2 = U_n2*I_2

print(Q2)
print(R_n2)

t2 = np.polyfit(Q2, R_n2, 1)
f2 = np.poly1d(t2)

plt.scatter(Q2, R_n2, c="r", zorder = 1)
plt.plot(Q2, f2(Q2), label = 'T2 = 30℃',
         linestyle='-', alpha = 1,
         color = 'r', ms=5, zorder = 0)
plt.grid(linewidth = 0.5)

sr_Q2 = sum(Q2)/6
sr_R_n2 = sum(R_n2)/6
sr_R_n2_kv = sum(R_n2**2)/6
sr_kv_R_n2 = sr_R_n2**2
sr_QR2 = sum(Q2*R_n2)/6
sr_Q_kv2 = sum(Q2**2)/6
sr_kv_Q2 = sr_Q2**2
k2 = (sr_QR2 - sr_Q2*sr_R_n2)/(sr_Q_kv2 - sr_kv_Q2)
sigma_angle_2 = (1/(6**0.5))*(((sr_R_n2_kv - sr_kv_R_n2)/(sr_Q_kv2 - sr_kv_Q2) - k2**2)**0.5)
sigma_R02 = sigma_angle_2*(sr_Q_kv2 - sr_kv_Q2)**0.5
print("Погрешности: ", sigma_angle_2, sigma_R02)
print(k2, f2(0))
print()

T3 = 40
I_3 = (10**(-3))*np.array([10.14, 19.04, 29.33, 40.345, 50.08, 60.04]) #мА
U_n3 = np.array([0.218, 0.41, 0.634, 0.876, 1.092, 1.317])#В
R_n3 = U_n3/I_3
Q3 = U_n3*I_3

print(Q3)
print(R_n3)

t3 = np.polyfit(Q3, R_n3, 1)
f3 = np.poly1d(t3)

plt.scatter(Q3, R_n3, c="r", zorder = 1)
plt.plot(Q3, f3(Q3), label = 'T3 = 40℃',
         linestyle='-', alpha = 1,
         color = 'green', ms=5, zorder = 0)
plt.grid(linewidth = 0.5)

sr_Q3 = sum(Q3)/6
sr_R_n3 = sum(R_n3)/6
sr_R_n3_kv = sum(R_n3**2)/6
sr_kv_R_n3 = sr_R_n3**2
sr_QR3 = sum(Q3*R_n3)/6
sr_Q_kv3 = sum(Q3**2)/6
sr_kv_Q3 = sr_Q3**2
k3 = (sr_QR3 - sr_Q3*sr_R_n3)/(sr_Q_kv3 - sr_kv_Q3)
sigma_angle_3 = (1/(6**0.5))*(((sr_R_n3_kv - sr_kv_R_n3)/(sr_Q_kv3 - sr_kv_Q3) - k3**2)**0.5)
sigma_R03 = sigma_angle_2*(sr_Q_kv3 - sr_kv_Q3)**0.5
print("Погрешности: ", sigma_angle_3, sigma_R03)
print(k3, f3(0))
print()

T4 = 50
I_4 = (10**(-3))*np.array([10.124, 21.84, 31.288, 39.895, 50.53, 60.272]) #мА
U_n4 = np.array([0.225, 0.487, 0.699, 0.894, 1.138, 1.365])#В
R_n4 = U_n4/I_4
Q4 = U_n4*I_4

print(Q4)
print(R_n4)

t4 = np.polyfit(Q4, R_n4, 1)
f4 = np.poly1d(t4)

plt.scatter(Q4, R_n4, c="r", zorder = 1)
plt.plot(Q4, f4(Q4), label = 'T4 = 50℃',
         linestyle='-', alpha = 1,
         color = 'pink', ms=5, zorder = 0)
plt.grid(linewidth = 0.5)

sr_Q4 = sum(Q4)/6
sr_R_n4 = sum(R_n4)/6
sr_R_n4_kv = sum(R_n4**2)/6
sr_kv_R_n4 = sr_R_n4**2
sr_QR4 = sum(Q4*R_n4)/6
sr_Q_kv4 = sum(Q4**2)/6
sr_kv_Q4 = sr_Q4**2
k4 = (sr_QR4 - sr_Q4*sr_R_n4)/(sr_Q_kv4 - sr_kv_Q4)
sigma_angle_4 = (1/(6**0.5))*(((sr_R_n4_kv - sr_kv_R_n4)/(sr_Q_kv4 - sr_kv_Q4) - k4**2)**0.5)
sigma_R04 = sigma_angle_4*(sr_Q_kv4 - sr_kv_Q4)**0.5
print("Погрешности: ", sigma_angle_4, sigma_R04)
print(k4, f4(0))
print()

T6 = 60
I_6 = (10**(-3))*np.array([10.107, 20.726, 29.048, 39.637, 50.703, 60.512]) #мА
U_n6 = np.array([0.232, 0.476, 0.669, 0.916, 1.178, 1.414])#В
R_n6 = U_n6/I_6
Q6 = U_n6*I_6

print(Q6)
print(R_n6)

t6 = np.polyfit(Q6, R_n6, 1)
f6 = np.poly1d(t6)

plt.scatter(Q6, R_n6, c="r", zorder = 1)
plt.plot(Q6, f6(Q6), label = 'T5 = 60℃',
         linestyle='-', alpha = 1,
         color = 'y', ms=5, zorder = 0)
plt.grid(linewidth = 0.5)

sr_Q6 = sum(Q6)/6
sr_R_n6 = sum(R_n6)/6
sr_R_n6_kv = sum(R_n6**2)/6
sr_kv_R_n6 = sr_R_n6**2
sr_QR6 = sum(Q6*R_n6)/6
sr_Q_kv6 = sum(Q6**2)/6
sr_kv_Q6 = sr_Q6**2
k6 = (sr_QR6 - sr_Q6*sr_R_n6)/(sr_Q_kv6 - sr_kv_Q6)
sigma_angle_6 = (1/(6**0.5))*(((sr_R_n6_kv - sr_kv_R_n6)/(sr_Q_kv6 - sr_kv_Q6) - k6**2)**0.5)
sigma_R06 = sigma_angle_6*(sr_Q_kv6 - sr_kv_Q6)**0.5
print("Погрешности: ", sigma_angle_6, sigma_R06)
print(k6, f6(0))
print()

T7 = 70
I_7 = (10**(-3))*np.array([10.09, 20.657, 30.972, 39.383, 50.877, 61.702]) #мА
U_n7 = np.array([0.239, 0.49, 0.736, 0.939, 1.218, 1.486])#В
R_n7 = U_n7/I_7
Q7 = U_n7*I_7

print(Q7)
print(R_n7)

t7 = np.polyfit(Q7, R_n7, 1)
f7 = np.poly1d(t7)

plt.scatter(Q7, R_n7, c="r", zorder = 1)
plt.plot(Q7, f7(Q7), label = 'T6 = 70℃',
         linestyle='-', alpha = 1,
         color = 'orange', ms=5, zorder = 0)
plt.xlim(0, 0.1)
plt.grid(linewidth = 0.5)

sr_Q7 = sum(Q7)/6
sr_R_n7 = sum(R_n7)/6
sr_R_n7_kv = sum(R_n7**2)/6
sr_kv_R_n7 = sr_R_n7**2
sr_QR7 = sum(Q7*R_n7)/6
sr_Q_kv7 = sum(Q7**2)/6
sr_kv_Q7 = sr_Q7**2
k7 = (sr_QR7 - sr_Q7*sr_R_n7)/(sr_Q_kv7 - sr_kv_Q7)
sigma_angle_7 = (1/(6**0.5))*(((sr_R_n7_kv - sr_kv_R_n7)/(sr_Q_kv7 - sr_kv_Q7) - k7**2)**0.5)
sigma_R07 = sigma_angle_7*(sr_Q_kv7 - sr_kv_Q7)**0.5
print("Погрешности: ", sigma_angle_7, sigma_R07)
print(k7, f7(0))
print()

major_ticks = np.arange(0, 0.1, 0.01)
plt.xticks(major_ticks)
plt.xlabel('Q, Вт', fontsize=10)
plt.ylabel('R_н, Ом', fontsize=10)
plt.legend()
plt.savefig("График зависимости мощности от сопротивления", dpi=600)

plt.show()