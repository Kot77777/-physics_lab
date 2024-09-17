import numpy as np
import matplotlib.pyplot as plt
from pogr_koef import slych_pogr

rho = 898
l = 0.725e-2
n = 1.85e-5
g = 9.81
h = 0.25e-3
dh = h/4
dt = dt0 = 0.4

def calculate_solution(time_up, time_down, U):
    q = []
    err_q_ar = []

    for i in range(len(time_up)):
        q.append(9*3.14*(l/U)*((2/rho/g)**0.5)*((n*h)**1.5)*((time_up[i] + time_down[i])/(time_down[i]**1.5)/time_up[i]))
        dU = U * 0.05

        err_q_ar.append(q[i] * (((dU/U)**2 + 2.25*(h**0.5)*dh**2 + (dt*(time_down[i] / time_up[i] / (time_down[i] + time_up[i])))**2 +
                        ((dt0*(3*time_up[i] + time_down[i])/4/time_down[i])/(time_down[i]+time_up[i])))**0.5))

    err_d_np = np.array(err_q_ar)
    err_d_sr = (sum(err_d_np**2)) ** 0.5

    q_sr = sum(q)/len(q)

    return [q_sr, err_d_sr/q_sr]

time_up = [[6.02, 4.30],
           [3.81, 4.74],
           [7.18, 9.70, 7.63],
           [6.09],
           [6.85, 5.10],
           [7.85, 6.60, 6.01],
           [10.03, 11.49, 12.12, 11.57, 10.34, 12.15]]

time_down = [[25.77, 28.17],
             [18.41, 28.19],
             [19.36, 18.75, 16.63],
             [19.56],
             [21.18, 21.11],
             [41.20, 40.32, 50.94],
             [15.92, 18.44, 18.10, 17.00, 16.85, 20.67]]

U = [500, 500, 300, 300, 300, 250, 250]

q_arr = []
err_arr = []
for i in range(len(time_up)):
    solut = calculate_solution(time_up[i], time_down[i], U[i])
    q_arr.append(solut[0])
    err_arr.append(solut[1])

print(q_arr)
print(err_arr)

# Добавляем полупрозрачные области для значений, кратных заряду электрона
electron_charge = 1.6e-19
plt.axhspan(0, electron_charge, color='blue', alpha=0.3)
plt.axhspan(electron_charge, electron_charge * 2, color='green', alpha=0.3)


# Устанавливаем пределы оси Y
plt.ylim(0, 4e-19)  # Увеличиваем область отображения по оси Y


plt.errorbar(q_arr, q_arr, yerr=np.array(err_arr)*np.array(q_arr), fmt='o', color='black', capsize=5, linestyle='None', label='Заряд капли')

print(np.array(err_arr)*np.array(q_arr))
# Настраиваем сетку
plt.grid(True)
plt.grid(which='both', color='gray', linestyle='--', linewidth=0.5)
plt.minorticks_on()
plt.grid(which='minor', color='lightgray', linestyle=':', linewidth=0.5)
plt.gca().set_axisbelow(True)

# Добавляем заголовок и метки осей
plt.title('Заряд капли')
plt.xlabel('q, Кл')
plt.ylabel('q, Кл')
plt.legend()

plt.savefig("График заряда капель", dpi=600)

# Показываем график
plt.show()

