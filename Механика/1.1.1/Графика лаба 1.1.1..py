import numpy as np
import matplotlib.pyplot as plt

V_l1 = []
I_l1 = []
line1 = []
with open('l1.txt', 'r', encoding='utf8') as f:
    for line in f:
        s = line.rstrip().split(sep=' ')
        V_l1.append(float(s[0]))
        I_l1.append(float(s[1]))
f.close()
print(V_l1, I_l1)

V_l2 = []
I_l2 = []
line2 = []
with open('l2.txt', 'r', encoding='utf8') as f:
    for line in f:
        s = line.rstrip().split(sep=' ')
        V_l2.append(float(s[0]))
        I_l2.append(float(s[1]))
f.close()
print(V_l2, I_l2)

V_l3 = []
I_l3 = []
line3 = []
with open('l3.txt', 'r', encoding='utf8') as f:
    for line in f:
        s = line.rstrip().split(sep=' ')
        V_l3.append(float(s[0]))
        I_l3.append(float(s[1]))
f.close()
print(V_l3, I_l3)


fig, ax = plt.subplots()

ax.set_xbound(0,160)
ax.set_ybound(0,650)
ax.set_box_aspect(1)

t1 = np.polyfit(I_l1, V_l1, 1)
f1 = np.poly1d(t1)
print(f1)
plt.plot(I_l1, f1(I_l1), '*', label = 'l1 = 20 см; k1 = 2,104',
         linestyle='-', mfc = 'r',mec = 'r', alpha = 1,
         color = 'black', linewidth=0.5, ms=5)
plt.errorbar(I_l1, f1(I_l1), xerr=0, yerr=7, fmt='|', color='black')

t2 = np.polyfit(I_l2, V_l2, 1)
f2 = np.poly1d(t2)
plt.plot(I_l2, f2(I_l2), '*',label = 'l2 = 30 см; k2 = 3,223',
         linestyle='-', mfc = 'g',mec = 'g', alpha = 0.5,
         color = 'black', linewidth=0.5, ms=5)
plt.errorbar(I_l2, f2(I_l2), xerr=0, yerr=7, fmt='|', color='black')

t3 = np.polyfit(I_l3, V_l3, 1)
f3 = np.poly1d(t3)
print(f3)
plt.plot(I_l3, f3(I_l3), '*',label = 'l3 = 50 см; k3 = 5,304',
         linestyle='-', mfc = 'b',mec = 'b', alpha = 0.5,
         color = 'black', linewidth=0.5, ms = 5)
plt.errorbar(I_l3, f3(I_l3), xerr=0, yerr=7, fmt='|', color='black')

plt.xticks(np.arange(min(I_l1), max(I_l2)+1, 10))
plt.yticks(np.arange(min(V_l1), max(V_l3)+1, 50))
plt.grid(linewidth = 0.5)
plt.xlabel('I, мА', fontsize=10, fontweight='bold')
plt.ylabel('V, мВ', fontsize=10, fontweight='bold')

plt.legend()
plt.show()
