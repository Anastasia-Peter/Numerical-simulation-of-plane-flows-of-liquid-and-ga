
import numpy as np

from scipy.linalg import solve


with open('input_prandtl_1.txt', 'r') as input:
    L = float(input.readline())
    H = float(input.readline())
    NI = int(input.readline())
    NJ = int(input.readline())
    U_0 = float(input.readline())
    mu = float(input.readline())
    ro = float(input.readline())
    eps = float(input.readline())
    S_max = int(input.readline())

nu = mu / ro

Re = U_0 * L / nu

# сетка
X = np.zeros(NI)
Y = np.zeros(NJ)
# шаг сетки
dx = L / (NI - 1)
dy = H / (NJ - 1)


X[0] = 0
for i in range(1, NI - 1):
    X[i] = X[i - 1] + dx
X[NI - 1] = L

Y[0] = 0
for i in range(1, NJ - 1):
    Y[i] = Y[i - 1] + dy
Y[NJ - 1] = H

# Создаем матрицы для Продольной(U_n) и Вертикальной(V_n) скоростей
U_n = np.zeros((NI, NJ))
V_n = np.zeros((NI, NJ))

# Оределяем функцию, которая вводит начальные условия.
def initial_conditions(U_n, V_n, NI, NJ):
    for i in range(0, NI):  
        for j in range(0, NJ):
            U_n[i][j] = U_0
            V_n[i][j] = 10 ** (-6)


initial_conditions(U_n, V_n, NI, NJ)

# Определяем функцию граничных условий
def boundary_conditions(U_n, NI, NJ):
    for i in range(0, NI):
        U_n[i][0] = 0
        U_n[i][NJ-1] = U_0 
        V_n[i][0] = 0


boundary_conditions(U_n, NI, NJ)

# Метод запаздывающих коеффициентов используется для решения задачи; массивы для коэффициентов
A = np.zeros(NJ)
B = np.zeros(NJ)
C = np.zeros(NJ)
D = np.zeros(NJ)

matrix = np.zeros((NJ, NJ))

U_n_next = np.zeros(NJ)
V_n_next = np.zeros(NJ)

# Определяем функцию, которая решает трехдиагональную матрицу методом прогонки, на самом деле можно её не использовать, а использовать solve ( так и сделаем далее), показано чисто для примера как можно сделать ещё. В этом плюс Питона !
def matrix_solver(A, B):

    k1 = -A[0, 1]
    m1 = B[0]
    k2 = -A[A.shape[0] - 1, A.shape[1] - 2] 
    m2 = B[B.shape[0] - 1]
    alfa = k1
    beta = m1

    c = 2
    a = 0
    b = 1
    alf = [alfa]
    bet = [beta]
    for i in range(1, A.shape[0] - 1):
        beta = (B[i] - A[i, a] * beta) / (A[i, a] * alfa + A[i, b])
        alfa = -A[i, c] / (A[i, a] * alfa + A[i, b])
        a += 1
        b += 1
        c += 1
        alf.append(alfa)
        bet.append(beta)

    y = (k2 * beta + m2) / (1 - k2 * alfa)
    res = [y]
    for i in range(len(alf) - 1, -1, -1):
        y = alf[i] * y + bet[i]
        res.append(y)

    result = []
    for i in reversed(res):
        result.append(i)
    return result


# численное решение
for i in range(1, NI):  # цикл по узлам X координаты

    for j in range(NJ):  # начальное приближение
        U_n_next[j] = U_n[i - 1][j]
        V_n_next[j] = V_n[i - 1][j]
# Начальная ошибка
    error_U = 1
    error_V = 1

    for s in range(S_max):  # цикл итераций

        # коэф-ты
        A[0] = 0
        B[0] = 1
        C[0] = 0
        D[0] = 0
# Считаем коэффициенты...
        for j in range(1, NJ - 1):
            A[j] = - V_n[i][j - 1] / (2 * dy) - nu / dy ** 2
            B[j] = U_n[i][j] / dx + 2 * nu / dy ** 2
            C[j] = V_n[i][j + 1] / (2 * dy) - nu / dy ** 2
            D[j] = U_n[i - 1][j] * U_n[i - 1][j] / dx
# Коэффы на границах не забываем....
        A[NJ - 1] = 0
        B[NJ - 1] = 1
        C[NJ - 1] = 0
        D[NJ - 1] = U_0

        # Матрица
        for j in range(NJ):
            matrix[j][j] = B[j]
        for j in range(0, NJ - 1):
            matrix[j + 1][j] = A[j + 1]
            matrix[j][j + 1] = C[j]

        # решение трехдиагональной матрицы. Вот этот момент, где мы используем функцию solve из пакета, вместо от руки написанного matrix.solver()
        U_n_next = solve(matrix, D)
        #U_n_next = np.array(matrix_solver(matrix, D)) #Можно и вторым способом

        # ур-е неразрывности
        V_n_next[0] = 0
        for j in range(1, NJ):
            V_n_next[j] = V_n_next[j - 1] - dy / 2 * ((U_n_next[j] - U_n[i - 1][j]) / dx + (U_n_next[j - 1] - U_n[i - 1][j - 1]) / dx)

        # невязки
        max_U = np.max(np.abs(U_n_next))
        max_V = np.max(np.abs(V_n_next))
        
        # из невязок находим ошибки
        error_U = np.max((np.abs(U_n_next - U_n[i])) / max_U)
        error_V = np.max((np.abs(V_n_next - V_n[i])) / max_V)

        for j in range(NJ):
            U_n[i][j] = U_n_next[j]  # сохраняем результат

        for j in range(NJ):
            V_n[i][j] = V_n_next[j]  # сохраняем результат

        print('X ', X[i], 'Iteration ', s, 'Error U ', error_U, 'Error V ', error_V, 'Re = ', Re)


tau = np.zeros(NI)  # трение на пластине
coef_f = np.zeros(NI)  # коэф трения на пластине
Re_x = np.zeros(NI)  # местное число Рейнольдса
for i in range(NI):
    tau[i] = mu * (U_n[i][1] - U_n[i][0]) / dy
    coef_f[i] = 2 * tau[i] / (ro * U_0 ** 2)
    Re_x[i] = U_0 * X[i] / nu

file = open('data_output_prandtl_1.plt', 'w')
file.write('VARIABLES = "X" "Re_x" "Tau" "Coef_f"' + '\n')
for i in range(NI):
    file.write('{0:.9f} {1:.9f} {2:.9f} {3:.9f}\n'.format(X[i], Re_x[i], tau[i], coef_f[i]))
file.close()

fi = open('result_prandtl_1.plt', 'w')
fi.write('VARIABLES = "X" "Y" "U" "V"' + '\n')
fi.write('ZONE i={0}, j={1}\n'.format(NI, NJ))
for j in range(NJ):
    for i in range(NI):
        fi.write('{0:.9f} {1:.9f} {2:.9f} {3:.9f}\n'.format(X[i], Y[j], U_n[i][j], V_n[i][j]))
fi.close()

with open(r'U_in_x=0_5.txt', 'w') as file:
    file.write('Y U \n')
    for i in range(NI):
        if round(X[i], 2) == 0.5:
            for j in range(NJ):
                file.write('{0:.9f} {1:.9f}\n'.format(Y[j], U_n[i][j]))

with open(r'U_in_x=1.txt', 'w') as file:
    file.write('Y U \n')
    for i in range(NI):
        if round(X[i], 2) == 1.0:
            for j in range(NJ):
                file.write('{0:.9f} {1:.9f}\n'.format(Y[j], U_n[i][j]))

with open(r'U_in_x=1_5.txt', 'w') as file:
    file.write('Y U \n')
    for i in range(NI):
        if round(X[i], 2) == 1.5:
            for j in range(NJ):
                file.write('{0:.9f} {1:.9f}\n'.format(Y[j], U_n[i][j]))
