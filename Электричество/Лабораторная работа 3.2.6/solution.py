import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)
pd.options.display.float_format = '{:.3f}'.format


def analyze_data(x_arr, y_arr, y_errors):
    coefficients, cov_matrix = np.polyfit(x_arr, y_arr, 1, cov=True)
    polynomial = np.poly1d(coefficients)
    x_linspace = np.linspace(min(x_arr), max(x_arr), 100)
    y_approx = polynomial(x_linspace)

    y_fit = np.polyval(coefficients, x_arr)
    chi_squared = np.sum(((y_arr - y_fit) / y_errors) ** 2)


    slope_error = np.sqrt(cov_matrix[0][0])
    intercept_error = np.sqrt(cov_matrix[1][1])

    return x_linspace, y_approx, coefficients[0], coefficients[1], slope_error, intercept_error, chi_squared


class Data:
    a = 1.115  # м
    R0 = 500  # Ом
    R2 = 10 * 10 ** 3  # Ом
    c = 2 * 10 ** (-6)  # Фр
    dU_0 = 0.02
    dx = 0.01
    da = 0.02
    dx_n = 0.01
    dx_n1 = 0.01


class DynamicConst:
    df = pd.DataFrame(
        {"R, Ом": np.array([565, 665, 865, 1000, 2000, 3000, 4500, 6000, 7500, 9000, 1500, 15000, 20000, 30000, 45000]),
         "delta_x, м": np.array([25, 22.6, 19.05, 16.1, 9.6, 6.9, 4.9, 3.8, 3.1, 2.6, 12.6, 1.6, 1.2, 0.8, 0.5]) / 100})

    df.sort_values(by="R, Ом", inplace=True)

    U_0 = 66 * 3 / 150  # В
    R1_R2 = 1 / 5000

    I = U_0 * R1_R2 / ((df["R, Ом"]) + Data.R0)
    x_linspace, y_approx, coefficients0, coefficients1, slope_error, intercept_error, chi_squared = analyze_data(df["delta_x, м"], I, len(I) * [Data.dx])
    k = (y_approx[-1] - y_approx[0]) / (x_linspace[-1] - x_linspace[0])

    C_I = 2 * Data.a * k
    S_I = 1 / C_I
    pogr_C_I = C_I * (((slope_error / k)**2 + (Data.da / Data.a)**2)**0.5)
    pogr_S_I = pogr_C_I / C_I * S_I

    def plot(self):
        plt.title("Зависимость смещения зайчика гальванометра от силы тока")
        plt.ylabel("I, A")
        plt.xlabel("delta_x, м")
        plt.plot(self.x_linspace, self.y_approx, color = 'green', label = f'k = (10.06 ± 0.14)e-7 А/м')
        plt.errorbar(self.df["delta_x, м"], self.I, xerr=0.005, yerr = Data.dU_0 * self.I / self.U_0, color="brown", fmt=".", alpha=0.8)
        plt.scatter([0], [0], color="brown", alpha=0.8, s=5)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.minorticks_on()
        plt.legend()
        plt.savefig('График зависимости смещения зайчика гальванометра от силы тока', dpi=300)
        plt.show()


class CritResistance:
    R_max = 9600  # Ом
    decr_max = np.log(18.9 / 14.6)
    T = 5.7  # c

    # Таблица значений с погрешностями
    df = pd.DataFrame({"R, Ом": np.array(
                           [3 * R_max, 4 * R_max, 5 * R_max, 5.5 * R_max, 6 * R_max, 7 * R_max, 8 * R_max, 9 * R_max,
                            10 * R_max]),
                       "R1_R2": np.array([500.0, 500.0, 300.0, 300.0, 300.0, 300.0, 300.0, 300.0, 300.0]),
                       "x_n": np.array([5.9, 5.6, 9.4, 8.7, 8.3, 7.5, 6.8, 6.1, 5.7]),
                       "x_n1": np.array([0.7, 1.1, 2.6, 2.2, 2.7, 2.8, 2.7, 2.6, 2.6])})

    # Вычисление логарифмических декрементов и их погрешностей
    df["Log_decr"] = np.log(df["x_n"] / df["x_n1"])
    df["Delta_Log_decr"] = np.sqrt((1 / df["x_n"] * Data.dx_n) ** 2 + (1 / df["x_n1"] * Data.dx_n1) ** 2)

    Theta = df["Log_decr"]

    term = (2 * np.pi / Theta) ** 2 + 1
    R_crit = (df["R, Ом"] + Data.R0) / np.sqrt(term) - Data.R0
    df["R_crit"] = R_crit

    term = (2 * np.pi / Theta) ** 2 + 1
    partial_Theta = (df["R, Ом"] + Data.R0) * (2 * np.pi) ** 2 / (Theta ** 3 * term ** (3 / 2))
    Delta_R_crit = partial_Theta * df["Delta_Log_decr"]
    df["Delta_R_crit"] = Delta_R_crit

    R_crit_sr = sum(R_crit)/len(R_crit)
    Delta_R_crit_sr = ((sum(Delta_R_crit) / len(Delta_R_crit))**2 + np.std(R_crit)**2)**0.5


    print(df)



class BallisticConst:
    l_max = 19.7 * 10 ** (-2)  # м
    R1_R2 = 1 / 30

    df = pd.DataFrame({"R, Ом": np.array([2200, 2700, 3500, 4500, 1500, 2000]),
                       "delta_x, м": np.array([4.5, 4.8, 5.5, 6.6, 3.4, 4.0])})
    df.sort_values(by="delta_x, м", inplace=True)
    df["delta_x, м"] /= 100

    x_linspace, y_approx, slope, intercept, slope_error, intercept_error, chi_squared = analyze_data(
        df["delta_x, м"], df["R, Ом"] + Data.R0, len(df["R, Ом"]) * [Data.dx])

    k = (y_approx[-1] - y_approx[0]) / (x_linspace[-1] - x_linspace[0])

    def plot(self):
        plt.title("Зависимость смещения зайчика гальванометра от R + R0")
        plt.xlabel("R + R_0, Ом")
        plt.ylabel("delta_x, м")
        plt.plot(self.y_approx, self.x_linspace, color='red')
        plt.errorbar((self.df["R, Ом"] + Data.R0), self.df["delta_x, м"],  xerr=0, yerr=0.0025, color="brown", fmt=".",
                     alpha=0.8)
        plt.grid(True , which='both', linestyle='--', linewidth=0.5)
        plt.minorticks_on()
        plt.savefig('График зависимости смещения зайчика гальванометра от R + R0', dpi=300)
        plt.show()

    def R_crit_finder(self):
        x_max = 0.197
        x_max_crit = x_max / np.e * (1 + 0.25 * CritResistance.decr_max)
        print(x_max_crit)
        RplusR0 = self.slope * x_max_crit + self.intercept
        print(RplusR0 - Data.R0)


BallisticConst().plot()
BallisticConst().R_crit_finder()
print(CritResistance().Delta_R_crit_sr)