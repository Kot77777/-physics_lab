import numpy as np
import matplotlib.pyplot as plt

t_l1 = []
V_l1 = []
with open('40.csv', 'r', encoding='utf8') as f:
    for line in f:
        if(not(line[0].isalpha())):
            s = line.rstrip().split(sep=',')
            t_l1.append(float(s[0]))
            V_l1.append(float(s[1]))
f.close()

t_l2 = []
V_l2 = []
with open('80.csv', 'r', encoding='utf8') as f:
    for line in f:
        if(not(line[0].isalpha())):
            s = line.rstrip().split(sep=',')
            t_l2.append(float(s[0]))
            V_l2.append(float(s[1]))
f.close()

t_l3 = []
V_l3 = []
with open('120.csv', 'r', encoding='utf8') as f:
    for line in f:
        if(not(line[0].isalpha())):
            s = line.rstrip().split(sep=',')
            t_l3.append(float(s[0]))
            V_l3.append(float(s[1]))
f.close()

t_l4 = []
V_l4 = []
with open('160.csv', 'r', encoding='utf8') as f:
    for line in f:
        if(not(line[0].isalpha())):
            s = line.rstrip().split(sep=',')
            t_l4.append(float(s[0]))
            V_l4.append(float(s[1]))
f.close()


t1 = np.array(t_l1)
V1 = np.log(np.array(V_l1))

# sr_t1 = sum(t1)/len(t1)
# sr_V1 = sum(V1)/len(t1)
# sr_V1_kv = sum(V1**2)/len(t1)
# sr_kv_V1 = sr_V1**2
# sr_tV1 = sum(t1*V1)/len(t1)
# sr_t1_kv = sum(t1**2)/len(t1)
# sr_kv_t1 = sr_t1**2
# k1 = (sr_tV1 - sr_t1*sr_V1)/(sr_t1 - sr_kv_t1)
# sigma_angle_1 = (1/(len(t1)**0.5))*(((sr_V1_kv - sr_kv_V1)/(sr_t1_kv - sr_kv_t1) - k1**2)**0.5)
# print("1", k1, sigma_angle_1)

fig, ax1 = plt.subplots()
l1 = np.polyfit(t1, V1,  1)
f1 = np.poly1d(l1)
print(f1, 1/l1[0])

plt.plot(t1, V1, label = 'P = 40торр', alpha = 1, color = 'blue')

plt.grid(linewidth = 0.5)

plt.xlabel('t, c', fontsize=10)
plt.ylabel('ln(U), mV', fontsize=10)
plt.legend()
plt.savefig("Графики зависимостей ln(U) от T при 40торр", dpi=600)
######################################################
t2 = np.array(t_l2)
V2 = np.log(np.array(V_l2))

fig, ax2 = plt.subplots()
l2 = np.polyfit(t2, V2,  1)
f2 = np.poly1d(l2)
print(f2, 1/l2[0])

plt.plot(t2, V2, label = 'P = 80торр', alpha = 1, color = 'blue')

plt.grid(linewidth = 0.5)

plt.xlabel('t, c', fontsize=10)
plt.ylabel('ln(U), mV', fontsize=10)
plt.legend()
plt.savefig("Графики зависимостей ln(U) от T при 80торр", dpi=600)
######################################################
t3 = np.array(t_l3)
V3 = np.log(np.array(V_l3))
l3 = np.polyfit(t3, V3,  1)
f3 = np.poly1d(l3)
print(f3, 1/l3[0])

fig, ax3 = plt.subplots()

plt.plot(t3, V3, label = 'P = 120торр', alpha = 1, color = 'blue')

plt.grid(linewidth = 0.5)

plt.xlabel('t, c', fontsize=10)
plt.ylabel('ln(U), mV', fontsize=10)
plt.legend()
plt.savefig("Графики зависимостей ln(U) от T при 120торр", dpi=600)
######################################################
t4 = np.array(t_l4)
V4 = np.log(np.array(V_l4))

fig, ax4 = plt.subplots()
l4 = np.polyfit(t4, V4,  1)
f4 = np.poly1d(l4)
print(f4, 1/l4[0])

plt.plot(t4, V4, label = 'P = 160торр', alpha = 1, color = 'blue')

plt.grid(linewidth = 0.5)

plt.xlabel('t, c', fontsize=10)
plt.ylabel('ln(U), mV', fontsize=10)
plt.legend()
plt.savefig("Графики зависимостей ln(U) от T при 160торр", dpi=600)

plt.show()
