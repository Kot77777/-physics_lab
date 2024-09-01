import matplotlib.pyplot as plt
import numpy as np
sp = []
st = []
with open('gistogramma_raschety.txt', 'r', encoding='utf8') as f:
    for line in f:
        s = line.rstrip().split(sep=' ')
        st.append(" & ".join(s))
        s1 = list(map(int, s))
        sp.append(s1)
f.close()
total = 0
for i in range(20):
    print(sp[i])
    total += sum(sp[i])
print(total)
print(sp)
k=1
s2 = []
s3 = []
sp5 = []
for i in range(0, 20, 2):
    for j in range(0, 10):
        s2.append(sp[i][k] + sp[i][k-1])
        sp5.append(sp[i][k] + sp[i][k-1])
        k+=2
        if (k==11):
            k = 1
            i+=1
    s3.append(s2.copy())
    s2.clear()
for i in range(20):
    print(i*10, " & ", st[i], "\\", "\\", sep="")
    print('\hline')
st2 = []
for line in s3:
    st2.append(" & ".join(list(map(str, line))))
for i in range(10):
    print(i * 10, " & ", st2[i] , "\\", "\\", sep="")
    print('\hline')
for i in set(sp5):
    print(sp5.count(i), end=" ")
print("\n")
print(np.array(sp5))
print(set(sp5))
print(sum(sp5))
total2 = 0
for i in range(len(sp5)):
    total2+=(sp5[i] - 49.01)**2
print(total2)

fig, ax1 = plt.subplots()
fig.set_figheight(10)
fig.set_figwidth(5)

values_10 = np.array([i for i in range(3, 26)])
counts_10 = np.array([1, 0, 5, 13, 20, 18, 29, 45, 42,
                      47, 42, 35, 25, 23, 29, 10, 8, 4,
                      1, 1, 0, 0, 1])
data_10 = []
for i in range(len(values_10)):
    data_10 += [values_10[i]] * counts_10[i]
data_10 = np.array(data_10)

color = 'tab:red'
ax1.set_xlabel('$n_i$, число импульсов', fontsize=15, color=color)
ax1.set_ylabel(r'$\omega$, доля случаев', fontsize=15)
ax1.hist(data_10, np.arange(0, 31), alpha=0.5, density=True, label='10с', color=color)
ax1.tick_params(axis='x', labelcolor=color,  labelsize=15)
ax1.tick_params(axis='y', labelcolor='#000000',  labelsize=15)
plt.xticks(np.arange(0, 31, 1))
plt.yticks(np.arange(0, 0.13, 0.01))
plt.legend(loc='upper left', fontsize=30)
plt.grid(linewidth = 0.5)

color = 'tab:blue'
ax2 = ax1.twiny()
ax2.set_xlabel('$n_i$, число импульсов', fontsize=15, color=color)

ax2.hist(np.array(sp5), np.arange(0, 121), alpha=0.5, density=True, label='40с')
ax2.tick_params(axis='x', labelcolor=color, labelsize=12)
plt.xticks(np.arange(0, 121, 4))
fig.tight_layout()
plt.legend(loc='upper right', fontsize=30)

plt.show()