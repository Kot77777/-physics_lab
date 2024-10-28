import numpy as np
import matplotlib.pyplot as plt
from pogr_koef import slych_pogr, solution

def student_k(arr, S):
    n = 6
    confidence_level = 0.95
    t_crit = 2.57
    sigma = t_crit * S / np.sqrt(n)
    return [np.mean(arr) - sigma, np.mean(arr) + sigma]

# Данные измерений
class Resonance:
    U_c = np.array([6.983, 6.219, 5.384, 4.9165, 4.5675, 3.8382])  # B

    U_E = np.array([0.2796, 0.2796, 0.2796, 0.2797, 0.2796, 0.2796])  # В

    f = np.array([32.1, 27.73, 23.170, 21.071, 19.377, 15.757]) * 1e3  # Гц
    print(len(U_c) == len(U_E) == len(f))

    w0 = 2 * np.pi * f

class AFC3:
    # C3 data
    U_min = 0.6 * Resonance.U_c[2]
    f = np.array([22.60, 22.65, 22.70, 22.75, 22.8, 22.90, 23.03, 23.09, 23.120,
                  23.27, 23.36, 23.45, 23.54, 23.63, 23.720, 23.810, 23.9, 23.99]) * 1e3  # кГц

    U_c = np.array([3.7455, 3.919, 4.0991, 4.2832, 4.4677, 4.8180, 5.232, 5.327, 5.356,
                    5.307, 5.085, 4.867, 4.6108, 4.3402, 4.0686, 3.8077, 3.5598, 3.33])  # В

    U_E = np.array([2.2793, 0.2792, 0.2792, 0.2793, 0.2792, 0.2793, 0.2794, 0.2794, 0.2794,
                    0.2791, 0.2791, 0.2792, 0.2792, 0.2793, 0.2793, 0.2793, 0.2794, 0.2797])  # B
    print(len(U_c) == len(U_E) == len(f))

class AFC5:
    # C5 data
    U_min = 0.6 * Resonance.U_c[4]
    f = np.array([18.65, 18.72, 18.79, 18.86, 18.93, 19.00, 19.07, 19.14, 19.21,
                  19.48, 19.57, 19.66, 19.75, 19.84, 19.93, 20.02, 20.11, 20.2]) * 1e3

    U_c = np.array([2.775, 2.9507, 3.1426, 3.3494, 3.5673, 3.7901, 4.0083, 4.2082, 4.3741,
                    4.4944, 4.351, 4.15, 3.9171, 3.6730, 3.4307, 3.1986, 2.9817, 2.7810])

    U_E = np.array([0.2794, 0.2794, 0.2794, 0.2794, 0.2795, 0.2796, 0.2796, 0.2797, 0.2797,
                    0.2794, 0.2794, 0.2795, 0.2795, 0.2796, 0.2796, 0.2797, 0.2797, 0.2798])
    print(len(U_c) == len(U_E) == len(f))


class PhaseDif3:
    dt = np.array([2.3, 2.5, 2.8, 3.1, 3.5, 4.2, 5.1, 6.6, 8.6, 10.9,
                   12.9, 14.5, 15.5, 16.5, 17, 17.3, 17.6, 18.0, 18.2]) * 1e-6

    f = np.array([21.4, 21.6, 21.8, 22, 22.2, 22.4, 22.6, 22.8, 23, 23.2,
                  23.4, 23.6, 23.8, 24.0, 24.2, 24.4, 24.6, 24.8, 25]) * 1e3
    T = 1 / f
    print(len(f) == len(dt))


class PhaseDif5:
    dt = np.array([2.9, 3.1, 3.5, 4.0, 4.5, 5.5, 6.5, 7.9,
                   13, 15, 16.5, 17.6, 18.6, 19.4, 20, 20.5]) * 1e-6

    f = np.array([17.977, 18.120, 18.27, 18.42, 18.57, 18.72, 18.87, 19.02,
                  19.4, 19.55, 19.7, 19.85, 20, 20.15, 20.3, 20.45]) * 1e3
    T = 1 / f
    print(len(f) == len(dt))

class Data:
    C_arr = np.array([24.8, 33.2, 47.6, 57.5, 68.0, 102.8]) * 1e-9

    R = 3.5

    L_arr = 1 / (C_arr * Resonance.w0 ** 2)
    L_mean = np.mean(L_arr)
    sem_L = np.std(L_arr, ddof=1) / np.sqrt(len(L_arr))
    L_confidence_interval = student_k(L_arr, sem_L)

    Q_arr = Resonance.U_c / Resonance.U_E

    rho_arr = np.sqrt(L_arr / C_arr)

    R_sum_arr = rho_arr / Q_arr

    tg_delta = 1e-3
    R_s_max = tg_delta * R_sum_arr

    R_L_arr = R_sum_arr - R - R_s_max
    R_L_mean = np.mean(R_L_arr)
    sem_R_L = np.std(L_arr, ddof=1) / np.sqrt(len(R_L_arr))
    R_L_confidence_interval = student_k(R_L_arr, sem_R_L)

    I_res = Resonance.U_E / R_sum_arr

#===============================================================#
#==========================График АЧХ===========================#

sluch_U_AFC3 = slych_pogr(AFC3.U_c)
sluch_U_AFC5 = slych_pogr(AFC5.U_c)

#plt.figure(1)

plt.figure(figsize=(15, 10))

#plt.plot(AFC3.f / 1000, AFC3.U_c, alpha = 0.5)
plt.errorbar(AFC3.f / 1000, AFC3.U_c, yerr = sluch_U_AFC3, fmt = 'o', color = 'blue', capsize=5, label = 'C3 = 47.6нФ')

#plt.plot(AFC5.f / 1000, AFC5.U_c, alpha = 0.5)
plt.errorbar(AFC5.f / 1000, AFC5.U_c, yerr = sluch_U_AFC5, fmt = 'D', color = 'orange', capsize=5, label = 'C5 = 19.377нФ')


plt.legend()
plt.grid(True)
plt.grid(which='both', color='gray', linestyle='--', linewidth=0.5)
plt.minorticks_on()
plt.grid(which='minor', color='lightgray', linestyle=':', linewidth=0.5)
plt.gca().set_axisbelow(True)
plt.xlabel('f, кГц')
plt.ylabel('U_c, В')
plt.savefig('График АЧХ', dpi=300)
plt.show()

plt.figure(figsize=(12, 8))
x_3 = AFC3.f / Resonance.f[2]
y_3 = AFC3.U_c / Resonance.U_c[2]

x_5 = AFC5.f / Resonance.f[4]
y_5 = AFC5.U_c / Resonance.U_c[4]

#plt.plot(x_3, y_3, alpha = 0.5)
line1 = plt.errorbar(x_3, y_3, yerr = y_3 * (sluch_U_AFC3 / (sum(AFC3.U_c) / len(AFC3.U_c))) * 2**0.5, fmt = 'o', color = 'blue', capsize=5, label = 'C3 = 47.6нФ')

#plt.plot(x_5, y_5, alpha = 0.5)
line2 = plt.errorbar(x_5, y_5, yerr = y_5 * (sluch_U_AFC5 / (sum(AFC5.U_c) / len(AFC5.U_c))) * 2**0.5, fmt = 'D', color = 'orange', capsize=5, label = 'C5 = 19.377нФ')

plt.grid(True)
plt.grid(which='both', color='gray', linestyle='--', linewidth=0.5)
plt.minorticks_on()
plt.grid(which='minor', color='lightgray', linestyle=':', linewidth=0.5)
plt.gca().set_axisbelow(True)
plt.xlabel('f/f0')
plt.ylabel('U_c/U_c0')



#===============================================================#
#==================Вычисление Q через АЧХ=======================#
def between_dot(arr, a):
    if a < min(arr) or a > max(arr):
        return []
    answer = []
    for i in range(len(arr) - 1):
        if (arr[i] < a and arr[i + 1] > a) or (arr[i] > a and arr[i + 1] < a):
            answer.append(i)
    return answer

y_C3_1 = y_3[(between_dot(y_3, 0.707))[0]]
y_C3_2 = y_3[between_dot(y_3, 0.707)[0] + 1]
x_C3_1 = x_3[between_dot(y_3, 0.707)[0]]
x_C3_2 = x_3[between_dot(y_3, 0.707)[0] + 1]
y_C3_11 = y_3[between_dot(y_3, 0.707)[1]]
y_C3_22 = y_3[between_dot(y_3, 0.707)[1] + 1]
x_C3_11 = x_3[between_dot(y_3, 0.707)[1]]
x_C3_22 = x_3[between_dot(y_3, 0.707)[1] + 1]
f1_C3 = ((0.707 - y_C3_1) / (y_C3_2 - y_C3_1)) * (x_C3_2 - x_C3_1) + x_C3_1
f2_C3 = ((0.707 - y_C3_11) / (y_C3_22 - y_C3_11)) * (x_C3_22 - x_C3_11) + x_C3_11

y_C5_1 = y_5[(between_dot(y_5, 0.707))[0]]
y_C5_2 = y_5[between_dot(y_5, 0.707)[0] + 1]
x_C5_1 = x_5[between_dot(y_5, 0.707)[0]]
x_C5_2 = x_5[between_dot(y_5, 0.707)[0] + 1]
y_C5_11 = y_5[between_dot(y_5, 0.707)[1]]
y_C5_22 = y_5[between_dot(y_5, 0.707)[1] + 1]
x_C5_11 = x_5[between_dot(y_5, 0.707)[1]]
x_C5_22 = x_5[between_dot(y_5, 0.707)[1] + 1]
f1_C5 = ((0.707 - y_C5_1) / (y_C5_2 - y_C5_1)) * (x_C5_2 - x_C5_1) + x_C5_1
f2_C5 = ((0.707 - y_C5_11) / (y_C5_22 - y_C5_11)) * (x_C5_22 - x_C5_11) + x_C5_11


plt.annotate('', xy=(f1_C3, 0.707), xytext=(f2_C3, 0.707),
             arrowprops=dict(arrowstyle='<->', color='green', lw=2))

plt.annotate('', xy=(f1_C5, 0.7), xytext=(f2_C5, 0.7),
             arrowprops=dict(arrowstyle='<->', color='red', lw=2))
arrow_sin = plt.Line2D([0], [0], color='green', lw=2, marker='>', markersize=10)
arrow_cos = plt.Line2D([0], [0], color='red', lw=2, marker='<', markersize=10)
plt.text(1, 0.72, 'dw_c3', fontsize=12, color = 'g')
plt.text(1, 0.68, 'dw_c5', fontsize=12, color = 'r')
# Настраиваем легенду с использованием стрелочек
plt.legend(handles=[line1, line2, arrow_sin, arrow_cos], labels=['C3 = 47.6нФ', 'C5 = 19.377нФ', 'dw_C3', 'dw_C5'])
plt.savefig('График АЧХ в относительных единицах', dpi=300)
plt.show()

dw_C3 = (f2_C3 * Resonance.f[2] - f1_C3 * Resonance.f[2])
dw_C5 = (f2_C5 * Resonance.f[4]- f1_C5 * Resonance.f[4])
print('dw', dw_C3, dw_C5)
Q_C3 = Resonance.f[2] / dw_C3
Q_C5 = Resonance.f[4] / dw_C5

dQ_C3 = 2*3.14*(x_C3_22 - x_C3_2) - (x_C3_11 - x_C3_1) / dw_C3
dQ_C5 = 2*3.14*(x_C5_22 - x_C5_2) - (x_C5_11 - x_C5_1) / dw_C5

print("Значения добротности по АЧХ: ", Q_C3, Q_C5)
print("Погрешнотсть добротноти по АЧХ: ", dQ_C3, dQ_C5)

#===============================================================#
#=======================График ФЧХ + approx=====================#

y_F_C3 = 2 * PhaseDif3.dt / PhaseDif3.T
x_F_C3 = PhaseDif3.f / Resonance.f[2]
y_F_C3_sluch = slych_pogr(y_F_C3)

y_F_C5 = 2 *  PhaseDif5.dt / PhaseDif5.T
x_F_C5 = PhaseDif5.f / Resonance.f[4]
y_F_C5_sluch = slych_pogr(y_F_C5)

x_app_C3 = x_F_C3[7:12]
y_app_C3 = y_F_C3[7:12]

y_app_C5 = y_F_C5[7:11]
x_app_C5 = x_F_C5[7:11]

t_C3 = np.polyfit(x_app_C3, y_app_C3,  1)
f_C3 = np.poly1d(t_C3)
pogr_C3 = solution(x_app_C3, y_app_C3)

t_C5 = np.polyfit(x_app_C5, y_app_C5,  1)
f_C5 = np.poly1d(t_C5)
pogr_C5 = solution(x_app_C5, y_app_C5)

plt.figure(figsize=(15, 10))

plt.plot(np.arange(0.96, 1.04, 0.01), f_C3(np.arange(0.96, 1.04, 0.01)), color = 'green', label = 'k_C3 = 11.30 ± 0.25')
plt.plot(np.arange(0.96, 1.04, 0.01), f_C5(np.arange(0.96, 1.04, 0.01)), color = 'red', linestyle = '--',
                                                                                                label = 'k_C5 = 10.09 ± 0.21')

plt.errorbar(x_F_C3, y_F_C3, yerr = y_F_C3_sluch, fmt = 'o', color = 'blue', capsize=5, label = 'C3 = 47.6нФ')
plt.errorbar(x_F_C5, y_F_C5, yerr = y_F_C5_sluch, fmt = 'D', color = 'orange', capsize=5, label = 'C5 = 19.377нФ')

plt.legend()
plt.grid(True)
plt.grid(which='both', color='gray', linestyle='--', linewidth=0.5)
plt.minorticks_on()
plt.grid(which='minor', color='lightgray', linestyle=':', linewidth=0.5)
plt.gca().set_axisbelow(True)
plt.xlabel('f/f0')
plt.ylabel('\u03C6/\u03C0')
plt.savefig('График ФЧХ в относительных единицах', dpi=300)

plt.show()

Q_F_C3 = 0.5 * np.pi * pogr_C3[0]
Q_F_C5 = 0.5 * np.pi * pogr_C5[0]

dQ_F_C3 = pogr_C3[2]
dQ_F_C5 = pogr_C5[2]

print("Значения добротности по ФЧХ: ", Q_F_C3, Q_F_C5)
print("Погрешнотсть добротноти по ФЧХ: ", dQ_F_C3, dQ_F_C5)

print(Data.L_arr * 10 ** 6)
print(Data.Q_arr)
print(Data.rho_arr)

def calculate_chi_squared(x, y, y_errors, coefficients):
    """
    Функция для расчета значения хи-квадрат и хи-квадрат на степень свободы.

    Аргументы:
    x — массив значений x
    y — массив значений y
    y_errors — массив погрешностей для y
    coefficients — коэффициенты аппроксимирующей функции

    Возвращает:
    chi_squared — значение хи-квадрат
    chi_squared_per_dof — хи-квадрат на степень свободы
    """
    # Вычисление значений аппроксимации
    y_fit = np.polyval(coefficients, x)

    # Расчет хи-квадрат
    chi_squared = np.sum(((y - y_fit) / y_errors) ** 2)

    # Число степеней свободы
    degrees_of_freedom = len(x) - len(coefficients)  # N - p, где p - число параметров

    # Хи-квадрат на степень свободы
    chi_squared_per_dof = chi_squared / degrees_of_freedom

    return [chi_squared, chi_squared_per_dof]

def random_error_L(L_arr, w0_arr):
    return 2 * L_arr * 0.01


def random_error_R_L(U_E, U_c, L, c, w0):
    sigma_L = random_error_L(L, w0)
    sigma_U_E = U_E * 0.03
    sigma_U_c = U_c * 0.03
    k = 1 - 1e-3

    dR_L_dU_E = (1 / U_c) * np.sqrt(L / c) * k
    dR_L_dU_c = -(U_E / U_c ** 2) * np.sqrt(L / c) * k
    dR_L_dL = (1 / 2) * (U_E / U_c) * (1 / np.sqrt(L * c)) * k

    sigma_R_L = np.sqrt((dR_L_dU_E * sigma_U_E) ** 2 + (dR_L_dU_c * sigma_U_c) ** 2 + (dR_L_dL * sigma_L) ** 2)
    return sigma_R_L

# Данные
E = Resonance.U_E[-1]
U_R = Data.I_res[-1] * Data.R
U_C = Resonance.U_c[-1]
U_L = Resonance.U_c[-1]
I = Data.I_res[-1]

# Углы фаз (в радианах)
psi_i = 0  # ток и ЭДС в фазе
psi_c = np.pi / 2 - np.arctanh(Data.tg_delta)
psi_l = - np.pi / 2 + Data.R_L_arr[-1] / Data.rho_arr[-1]
print(psi_l)

# Построение векторов
plt.figure(figsize=(16, 16))
ax = plt.gca()

# Вектор E (направлен вдоль оси X)
ax.quiver(0, 0.02, E, 0.008, angles='xy', scale_units='xy', scale=1, color='r', width=0.005, label="E (ЭДС), В")
plt.text(E / 2, 0.1, 'E', color = 'r', fontsize=15)
# Вектор тока I (в фазе с E)
ax.quiver(0, 0, I, 0, angles='xy', scale_units='xy', scale=1, color='b', width=0.005, label="I (Ток), А", alpha=0.8)
plt.text(I/2, 0.1, 'I', color = 'b', fontsize=15)

# Вектор напряжения на R (U_R, в фазе с током и ЭДС)
ax.quiver(0, 0, U_R, 0, angles='xy', scale_units='xy', scale=1, color='g', width=0.003, label="U_R (на сопротивлении), В", alpha=0.8)
plt.text(U_R/2, -0.3, 'U_R', color = 'g', fontsize=15)

# Вектор напряжения на L (U_L, угол psi_l), начинается в начале координат
ax.quiver(0, 0, U_L * np.cos(psi_l), U_L * np.sin(psi_l), angles='xy', scale_units='xy', scale=1, color='m', width=0.005, label="U_L (на катушке), В")
plt.text(0.08, -2, 'U_L', color = 'm', fontsize=15)

# Вектор напряжения на C (U_C, угол psi_c), начинается в начале координат
ax.quiver(0, 0, U_C * np.cos(psi_c), U_C * np.sin(psi_c), angles='xy', scale_units='xy', scale=1, color='c', width=0.005, label="U_C (на конденсаторе), В")
plt.text(0.01, 2, 'U_C', color = 'c', fontsize=15)

# Проекция U_L на ось X (от конца U_R)
ax.quiver(U_R, 0, U_L * np.cos(psi_l), 0, angles='xy', scale_units='xy', scale=1, color='orange', alpha=0.8, width=0.003, label="U_L на активном сопротивлении, В")
plt.text(U_R + U_L * np.cos(psi_l) / 2, -0.3, 'U_L_акт', color = 'orange', fontsize=15)

# Проекция U_C на ось X (от конца U_R + U_L)
ax.quiver(U_R + U_L * np.cos(psi_l), 0, U_C * np.cos(psi_c), 0, angles='xy', scale_units='xy', scale=1, color='c', linestyle='dashed', alpha=0.8, width=0.003)

# Настройки графика
plt.xlim(-1 / 3, 1 / 3)  # масштаб по оси X
plt.ylim(-4, 4)  # масштаб по оси Y

# Добавление более мелкой сетки
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.minorticks_on()
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='gray')

# Оси X и Y
ax.axhline(0, color='black', linewidth=0.5)
ax.axvline(0, color='black', linewidth=0.5)

# Легенда
plt.legend()

# Подписи осей
plt.title("Векторная диаграмма в резонансе с проекциями на ось X")
plt.xlabel("Re")
plt.ylabel("Im")
plt.savefig('Векторная диаграмма в резонансе', dpi=300)
plt.show()

plt.figure(figsize=(15, 10))

# График
plt.errorbar(Resonance.f / 1e3, Data.R_L_arr, yerr=random_error_R_L(Resonance.U_E, Resonance.U_c, Data.L_arr, Data.C_arr, Resonance.w0),
             xerr=Resonance.f * 0.01 / 1e3, fmt=".",
             label='Измеренные значения $R_L$', color='red')
plt.plot(Resonance.f / 1e3, [Data.R_L_mean] * len(Resonance.f),
         label=f'Среднее значение $R_L$ ({Data.R_L_mean:.1f} Ом)')



coefficients, cov_matrix = np.polyfit(Resonance.f / 1e3, Data.R_L_arr, 1, cov=True)
polynomial = np.poly1d(coefficients)
x_linspace = np.linspace(min(Resonance.f / 1e3), max(Resonance.f / 1e3), 100)
y_approx = polynomial(x_linspace)
chi_squared = calculate_chi_squared(Resonance.f / 1e3, Data.R_L_arr,
                                    random_error_R_L(Resonance.U_E, Resonance.U_c, Data.L_arr, Data.C_arr, Resonance.w0),
                                    coefficients)
plt.plot(x_linspace, y_approx, label=f'МНК, chi-squared = {chi_squared[0]:.2f}')

plt.title("Зависимость напряжения на катушке от резонансной частоты")
plt.ylabel('R_L, Ом')
plt.xlabel('f_0, кГц')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.minorticks_on()
plt.legend()
plt.savefig('График зависимости сопротивления на катушке от резонансной частоты', dpi=300)
plt.show()

print(slych_pogr(Data.L_arr))

