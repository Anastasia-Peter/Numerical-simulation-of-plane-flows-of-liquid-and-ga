import numpy as np

from numba import jit


with open('input_nav.txt', 'r') as file:
     L = float(file.readline())
     H = float(file.readline())
     u0 = float(file.readline())
     nu = float(file.readline())
     ni = int(file.readline())
     nj = int(file.readline())
     eps = float(file.readline())
     max_s = int(file.readline())
     cfl = float(file.readline())
     

x_node = np.zeros((ni + 1, nj + 1))
y_node = np.zeros((ni + 1, nj + 1))
x_cell = np.zeros((ni + 1, nj + 1))
y_cell = np.zeros((ni + 1, nj + 1))
p = np.zeros((ni + 1, nj + 1))
u = np.zeros((ni + 1, nj + 1))
v = np.zeros((ni + 1, nj + 1))
u_old = np.zeros((ni + 1, nj + 1))
v_old = np.zeros((ni + 1, nj + 1))
p_old = np.zeros((ni + 1, nj + 1))
residual_u = np.zeros((ni + 1, nj + 1))
residual_v = np.zeros((ni + 1, nj + 1))
residual_p = np.zeros((ni + 1, nj + 1))
cf = np.zeros(ni)
reses = np.zeros((max_s, 4))
out = np.zeros((1, 4))

# шаг сетки
dx = L / (ni - 1)
dy = H / (nj - 1)

for i in range(1, ni + 1):
    for j in range(1, nj + 1):
        x_node[i, j] = (i - 1) * dx
        y_node[i, j] = (j - 1) * dy

for j in range(1, nj + 1):
    x_cell[0, j] = - dx / 2
    y_cell[0, j] = y_node[1, j] + dy / 2

for i in range(1, ni + 1):
    x_cell[i, 0] = x_node[i, 1] + dx / 2
    y_cell[i, 0] = - dy / 2

for i in range(1, ni + 1):
    for j in range(1, nj + 1):
        x_cell[i, j] = x_node[i, j] + dx / 2
        y_cell[i, j] = y_node[i, j] + dy / 2

#Ускоряем функцию применения ГУ. 
@jit(nopython=True)
def boundary_conditions(u, v, p, u0, ni, nj):
    for j in range(nj + 1):
        # левая граница, НУ
        u[0][j] = u0
        v[0][j] = 0.0
        p[0][j] = p[1][j]
        # правая граница
        u[ni][j] = u[ni - 1][j]
        v[ni][j] = v[ni - 1][j]
        p[ni][j] = 0.0
    for i in range(ni + 1):
        # нижняя граница, стенка
        u[i][0] = - u[i][1]
        v[i][0] = - v[i][1]
        p[i][0] = p[i][1]
        # верхняя граница
        u[i][nj] = u[i][nj - 1]
        v[i][nj] = v[i][nj - 1]
        p[i][nj] = p[i][nj - 1]

#Ускоряем функцию применения вычисления НС.
@jit(nopython=True)
def navier(ni, nj, u0, nu, dx, dy, eps, max_s, u, v, p, u_old, v_old, p_old, residual_u, residual_v, residual_p, reses, out):
    for i in range(ni + 1):
        for j in range(nj + 1):
            u[i][j] = u0
    s = 0
    residual_u_max = 1.0
    residual_v_max = 1.0
    residual_p_max = 1.0
    boundary_conditions(u, v, p, u0, ni, nj)
    a = 1.0 / (u0**2)
    dt = cfl * min(0.5 * dx**2 / nu, 0.5 * dy**2 / nu, dx/u0)

    while s <= max_s and (residual_u_max >= eps or residual_v_max >= eps or residual_p_max >= eps):

        boundary_conditions(u, v, p, u0, ni, nj)
        for i in range(1, ni + 1):
            for j in range(1, nj + 1):
                u_old[i][j] = u[i][j]
                v_old[i][j] = v[i][j]
                p_old[i][j] = p[i][j]

        for i in range(1, ni):
            for j in range(1, nj):
                u_per1 = (u[i][j] + u[i + 1][j])/2
                if u_per1 >= 0.0:
                    u_right = u[i][j]
                    v_right = v[i][j]
                    p_right = p[i + 1][j]
                else:
                    u_right = u[i + 1][j]
                    v_right = v[i + 1][j]
                    p_right = p[i][j]

                u_per2 = (u[i - 1][j] + u[i][j])/2
                if u_per2 >= 0.0:
                    u_left = u[i - 1][j]
                    v_left = v[i - 1][j]
                    p_left = p[i][j]
                else:
                    u_left = u[i][j]
                    v_left = v[i][j]
                    p_left = p[i - 1][j]

                v_per1 = (v[i][j] + v[i][j + 1])/2
                if v_per1 >= 0.0:
                    u_up = u[i][j]
                    v_up = v[i][j]
                    p_up = p[i][j + 1]
                else:
                    u_up = u[i][j + 1]
                    v_up = v[i][j + 1]
                    p_up = p[i][j]

                v_per2 = (v[i][j - 1] + v[i][j])/2
                if v_per2 >= 0.0:
                    u_down = u[i][j - 1]
                    v_down = v[i][j - 1]
                    p_down = p[i][j]
                else:
                    u_down = u[i][j]
                    v_down = v[i][j]
                    p_down = p[i][j - 1]

                # ур-е неразрывности
                conv1_p = (u_right - u_left) / dx
                conv2_p = (v_up - v_down) / dy
                p[i][j] = p[i][j] + dt / a * (- conv1_p - conv2_p)

                # ур-е движ-я в проекции на ось X
                conv1_u = (u_per1 * u_right - u_per2 * u_left) / dx
                conv2_u = (v_per1 * u_up - v_per2 * u_down) / dy
                p1 = (p_right - p_left) / dx
                diff1_u = nu / (dx**2) * (u[i + 1][j] - 2 * u[i][j] + u[i - 1][j])
                diff2_u = nu / (dy**2) * (u[i][j + 1] - 2 * u[i][j] + u[i][j - 1])
                u[i][j] = u[i][j] + dt * (- conv1_u - conv2_u - p1 + diff1_u + diff2_u)

                # ур-е движ-я в проекции на ось Y
                conv1_v = (u_per1 * v_right - u_per2 * v_left) / dx
                conv2_v = (v_per1 * v_up - v_per2 * v_down) / dy
                p1 = (p_up - p_down) / dy
                diff1_v = nu / (dx**2) * (v[i + 1][j] - 2 * v[i][j] + v[i - 1][j])
                diff2_v = nu / (dy**2) * (v[i][j + 1] - 2 * v[i][j] + v[i][j - 1])
                v[i][j] = v[i][j] + dt * (- conv1_v - conv2_v - p1 + diff1_v + diff2_v)

                residual_p[i][j] = abs(p[i][j] - p_old[i][j]) * a / dt
                residual_u[i][j] = abs(u[i][j] - u_old[i][j]) / dt
                residual_v[i][j] = abs(v[i][j] - v_old[i][j]) / dt

        residual_u_max = 0
        residual_v_max = 0
        residual_p_max = 0

        for i in range(2, ni):
            for j in range(2, nj):
                if residual_u[i][j] >= residual_u_max:
                    residual_u_max = residual_u[i][j]
                if residual_v[i][j] >= residual_v_max:
                    residual_v_max = residual_v[i][j]
                if residual_p[i][j] >= residual_p_max:
                    residual_p_max = residual_p[i][j]
        boundary_conditions(u, v, p, u0, ni, nj)
        out = [s, residual_u_max, residual_v_max, residual_p_max]
        reses[s, :] = out
        s += 1
        print('iter = ', s, 'res_u = ', residual_u_max, 'res_v = ', residual_v_max, 'res_p = ', residual_p_max, 'Re = ', u0 * L / nu)


#Вызываем функцию НС
navier(ni, nj, u0, nu, dx, dy, eps, max_s, u, v, p, u_old, v_old, p_old, residual_u, residual_v, residual_p, reses, out)

tau = np.zeros(ni)  # трение на пластине
coef_f = np.zeros(ni)  # коэф трения на пластине
Re_x = np.zeros(ni)  # местное число Рейнольдса
for i in range(ni):
    tau[i] = nu * 1000 * (u[i][2] - u[i][1]) / dy
    coef_f[i] = 2 * tau[i] / (1000 * u0 ** 2)
    Re_x[i] = u0 * x_node[i + 1][1] / nu

with open(r'data_output_navier_stokes_1.plt', 'w') as file:
    file.write('VARIABLES = "X" "Re_x" "Tau" "Coef_f"' + '\n')
    for i in range(ni):
        file.write('{0:.9f} {1:.9f} {2:.9f} {3:.9f}\n'.format(x_node[i + 1][1], Re_x[i], tau[i], coef_f[i]))

with open(r'residuals_navier_stokes_1.plt', 'w') as file:
    file.write('VARIABLES = "Iteration" "Residual U" "Residual V" "Residual P"' + '\n')
    for i in range(1, max_s):
        file.write((' '.join(map(str, reses[i, :]))) + '\n')

with open('data_navier_stokes_1.plt', 'w') as file:
    file.write('VARIABLES="X", "Y", "U", "V", "P"\n')
    file.write(f'ZONE I={ni}, J={nj}, DATAPACKING=BLOCK, VARLOCATION=([3-20]=CELLCENTERED)\n')
    for j in range(1, nj + 1):
        file.write((' '.join(map(str, x_node[1:ni + 1, j]))) + '\n')
    for j in range(1, nj + 1):
        file.write((' '.join(map(str, y_node[1:ni + 1, j]))) + '\n')
    for j in range(1, nj):
        file.write((' '.join(map(str, u[1:ni, j]))) + '\n')
    for j in range(1, nj):
        file.write((' '.join(map(str, v[1:ni, j]))) + '\n')
    for j in range(1, nj):
        file.write((' '.join(map(str, p[1:ni, j]))) + '\n')
