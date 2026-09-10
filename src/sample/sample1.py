#Listo

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import fbsm
import time

T = 1000 # Tiempo total de simulación
# Parámetros reales
mu_h = 0.002273    # tasa de mortalidad natural en humanos
mu_v = 0.0323       # tasa de natalidad natural en vectores
mu_D = 0.04        # tasa de mortalidad del reservorio D
gamma = 0.32       # tasa de recuperación en humanos
alpha = 0.01       # tasa de mortalidad del vector causada por el reservorio D
eta = 0.1          # tasa de contacto entre el vector y el reservorio D
b = 0.5            # tasa de transmisión
beta_h = 0.45       # coeficiente de transmisión humano
beta_v = 0.45       # coeficiente de transmisión vector
a = 1               # peso de S_h(T)
c = 0.5             # peso de integral de u^2(t)
d = 0.95             # peso para la estabilización

S_h0, I_h0 = 100, 10  # Humanos
S_v0, I_v0, D0 = 100, 10, 10   # Vectores y reservorio
y0 = [S_h0, I_h0, S_v0, I_v0, D0]

N_h = S_h0 + I_h0    # Población total de humanos
N_v = S_v0 + I_v0    # Población total de vectores

R0 = (b/N_h)*np.sqrt((beta_h*beta_v*S_h0*mu_D)/((gamma+mu_h)*mu_v*alpha*eta))

mosquitos = mu_D/(alpha*eta)
libelulas = mu_v / alpha


print(f"El valor de R0 es: {R0:.2f}")
print(f"Los puntos de equilibrio del sistema Lotka-Volterra son:\n - Mosquitos: {mosquitos}\n - Libélulas: {libelulas}")

# Función de control inicial
def u0(t):
    return 0.0

def efe(t, x, u = u0):
    S_h, I_h, S_v, I_v, D = x
    dS_h = mu_h * N_h - b * (beta_h / N_h) * S_h * I_v - mu_h * S_h
    dI_h = b * (beta_h / N_h) * S_h * I_v - (gamma + mu_h) * I_h
    dS_v = mu_v * (S_v + I_v) - b * (beta_v / N_h) * S_v * I_h - alpha * D * S_v
    dI_v = b * (beta_v / N_h) * S_v * I_h - alpha * D * I_v
    dD = eta * alpha * (S_v + I_v) * D - mu_D * D + u(t)
    return [dS_h, dI_h, dS_v, dI_v, dD]

# Resolución del sistema en el intervalo de tiempo
#T = 30 # Tiempo total de simulación
T2 = np.linspace(0, T, 200)  # Puntos de evaluación de la solución
sol = solve_ivp(efe, [0, T], y0, t_eval=T2)


# CONTROL
u_max = 10  # Límite máximo de control
x0 = [S_h0, I_h0, S_v0, I_v0, D0]  # Condiciones iniciales para el control


# Función para el sistema adjunto
def pe(t, x, W):
    W1, W2, W3, W4, W5 = W
    p1, p2, p3, p4, p5 = x
    dp1_dt = (p1 - p2) * b * beta_h * W4(t) / N_h + p1 * mu_h
    dp2_dt = p2 * (mu_h + gamma) + (p3 - p4) * b * beta_v * W3(t) / N_h
    dp3_dt = p3 * (-mu_v + alpha * W5(t)) + (p3 - p4) * b * beta_v * W2(t) / N_h - p5 * eta * alpha * W5(t)
    dp4_dt = (p1 - p2) * b * beta_h * W1(t) / N_h - mu_v * p3 + p4 * alpha * W5(t) - p5 * eta * alpha * W5(t)
    dp5_dt = p3 * alpha * W3(t) + p4 * alpha * W4(t) - p5 * alpha * eta * (W3(t) + W4(t)) + p5 * mu_D
    return [dp1_dt, dp2_dt, dp3_dt, dp4_dt, dp5_dt]

# Definimos una nueva función de control basada en la solución
def u_f(t, previous_u_t, control_params: fbsm.ControlParams):
    [p1, p2, p3, p4, p5] = control_params.p_arr
    return d * previous_u_t + (1 - d) * max(0.0, min(u_max, p5(t) / 2 * c))


solver = fbsm.Solver()
solver.set_initial_control_function(u0)
solver.set_direct_system(fbsm.DirectSystem(efe, [0, T], x0, T2))
solver.set_adj_system(fbsm.AdjSystem(pe, [T, 0], [a, 0, 0, 0, 0], np.linspace(T, 0, 2 * T)))
solver.set_control_function(u_f)
#solver.set_terminating_condition(self,__some__terminating__condition__)
solver.set_max_iterations(80)
#solver.set_min_iterations(200)
start_time = time.time()
result = solver.solve()

sol_c = result.direct_sol
sol_adj = result.adj_sol
W = result.w_arr
P = result.p_arr
total_its = result.iterations
last_control = result.last_control

elapsed_time = time.time() - start_time
print("Total iterations: ", total_its)
print("Elapsed time: ", elapsed_time)


TZF = np.flip(sol_adj.t)
# Gráfico : Diagrama de fase (S_v vs D)
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

axes[0].plot(sol.y[2]+sol.y[3], sol.y[4], label="Nv vs D", color="b")
axes[0].set_xlabel("S_v + I_v")
axes[0].set_ylabel("D")
axes[0].set_title("Phase Diagram Lotka-Volterra sin control")
axes[0].legend()
axes[0].grid()

axes[1].plot(sol_c.y[2]+sol_c.y[3],sol_c.y[4], label="Nv vs D", color="b")
axes[1].set_xlabel("S_v + I_v")
axes[1].set_ylabel("D")
axes[1].set_title("Phase Diagram Lotka-Volterra con control")
axes[1].legend()
axes[1].grid()

# Ajustar espacio entre subplots
plt.tight_layout()
plt.show()

# Crear una malla fina de puntos en el rango de TZF
t_plot = np.linspace(TZF[0], TZF[-1], 1000)

# Evaluar las funciones interpoladas en los puntos de la malla
P1, P2, P3, P4, P5 = P
p1_vals = P1(t_plot)
p2_vals = P2(t_plot)
p3_vals = P3(t_plot)
p4_vals = P4(t_plot)
p5_vals = P5(t_plot)

# Crear la figura y las subtramas para visualizar las soluciones


#Diferencia
# Aseguramos que ambas soluciones estén evaluadas en los mismos tiempos
W1, W2, W3, W4, W5 = W
S_h_no_control = np.interp(sol_c.t, sol.t, sol.y[0])  # Interpolamos para igualar los tiempos
S_h_control = W1(sol_c.t)  # S_h con control

# Diferencia punto a punto
delta_S_h = - S_h_no_control + S_h_control


# Crear una figura con subplots de 2 columnas y 3 filas
fig, axes = plt.subplots(4, 2, figsize=(16, 18))  # 3 filas, 2 columnas

# Gráfico 1: Dinámica del sistema epidemiológico
axes[0, 0].plot(sol.t, sol.y[0], label="S_h (Susceptible humans)")
axes[0, 0].plot(sol.t, sol.y[1], label="I_h (Infected humans)")
axes[0, 0].plot(sol.t, sol.y[2], label="S_v (Susceptible vectors)")
axes[0, 0].plot(sol.t, sol.y[3], label="I_v (Infected vectors)")
axes[0, 0].plot(sol.t, sol.y[4], label="D (Predator)")
axes[0, 0].set_xlabel("Time (days)")
axes[0, 0].set_ylabel("Population")
axes[0, 0].set_title("Dynamics of the Epidemiological System")
axes[0, 0].legend()
axes[0, 0].grid()

# Gráfico 2: Dinámica del sistema epidemiológico bajo control
axes[0, 1].plot(sol.t, W1(sol_c.t), label='S_h (Susceptible humans)')
axes[0, 1].plot(sol.t, W2(sol_c.t), label='I_h (Infected humans)')
axes[0, 1].plot(sol.t, W3(sol_c.t), label='S_v (Susceptible vectors)')
axes[0, 1].plot(sol.t, W4(sol_c.t), label='I_v (Infected vectors)')
axes[0, 1].plot(sol.t, W5(sol_c.t), label='D (Predator)')
axes[0, 1].set_xlabel('Time (days)')
axes[0, 1].set_ylabel('Population')
axes[0, 1].set_title('Dynamics of the Epidemiological System under Control', fontsize=10)
axes[0, 1].legend()
axes[0, 1].grid()

#Grafico 3

axes[1, 0].plot(t_plot, p1_vals, label='$P_1(t)$')
axes[1, 0].plot(t_plot, p2_vals, label='$P_2(t)$')
axes[1, 0].plot(t_plot, p3_vals, label='$P_3(t)$')
axes[1, 0].plot(t_plot, p4_vals, label='$P_4(t)$')
axes[1, 0].plot(t_plot, p5_vals, label='$P_5(t)$')
axes[1, 0].set_title('Adjoint system $P_1(t)$ to $P_5(t)$')
axes[1, 0].set_xlabel('Time')
axes[1, 0].set_ylabel('Value of $P_i(t)$')
axes[1, 0].legend()
axes[1, 0].grid()


# Gráfico 4: Variable de control
axes[1, 1].plot(sol.t, last_control(sol.t), label="u (Control)", color="blue")
axes[1, 1].set_xlabel("Time")
axes[1, 1].set_ylabel("Control Value")
axes[1, 1].set_title("Control Variable Over Time")
axes[1, 1].legend()
axes[1, 1].grid()

# Gráfico 5: Comparación con y sin control
axes[2, 0].plot(sol_c.t, W1(sol_c.t), label='S_h (Susceptible Humans-Control)')
axes[2, 0].plot(sol.t, sol.y[0], label="S_h (Susceptible Humans Without Control)")
axes[2, 0].set_title('Comparison of S_h With and Without Control')
axes[2, 0].set_xlabel('Time (days)')
axes[2, 0].set_ylabel('Population')
axes[2, 0].legend()
axes[2, 0].grid()

# Gráfico 6: Comparación con y sin control
axes[2, 1].plot(sol_c.t, W2(sol_c.t), label='I_h (Infected Humans-Control)')
axes[2, 1].plot(sol.t, sol.y[1], label="I_h (Infected Humans Without Control)")
axes[2, 1].set_title('Comparison of I_h With and Without Control')
axes[2, 1].set_xlabel('Time (days)')
axes[2, 1].set_ylabel('Population')
axes[2, 1].legend()
axes[2, 1].grid()

# Gráfico 7: Diferencia entre S_h con y sin control
S_h_no_control = np.interp(sol_c.t, sol.t, sol.y[0])  # Interpolamos para igualar los tiempos
S_h_control = W1(sol_c.t)  # S_h con control
delta_S_h = - S_h_no_control + S_h_control  # Diferencia punto a punto


axes[3, 0].plot(sol_c.t, delta_S_h, label=r"$\Delta S_h$ (Difference in Susceptible Humans)", color="red")
axes[3, 0].axhline(0, color="black", linestyle="--", linewidth=1, label="Zero Difference")  # Línea de referencia
axes[3, 0].set_title('Difference Between $S_h$ Without Control and $S_h$ With Control')
axes[3, 0].set_xlabel('Time (days)')
axes[3, 0].set_ylabel('Difference in Population')
axes[3, 0].legend()
axes[3, 0].grid()


# Ajustar espacio entre subplots
plt.tight_layout()
plt.show()


percentage_error = np.abs(S_h_no_control - S_h_control) / np.abs(S_h_control) * 100

# Calculate the average percentage error
average_error = np.mean(percentage_error)
print(f"Average error: {average_error:.2f}%")

# Plot the percentage error
plt.plot(sol_c.t, percentage_error, label='Percentage error', color='r')
plt.xlabel('Time')
plt.ylabel('Percentage error (%)')
plt.title('Percentage error between S_h_no_control and S_h_control')
plt.legend()
plt.grid(True)
plt.show()