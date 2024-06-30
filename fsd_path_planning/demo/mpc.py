# #this is a code for mpc 
# import casadi as ca
# import numpy as np

# # Define parameters
# N = 10  # Prediction horizon
# dt = 0.1  # Time step
# L = 2.9  # Wheelbase
# initial_state = np.array([0, 0, 0, 10])  # [x, y, psi, v]

# # Reference trajectory
# x_ref = np.linspace(0, 10, N)
# y_ref = np.linspace(0, 5, N)
# psi_ref = np.zeros(N)
# v_ref = 10 * np.ones(N)

# # Define CasADi variables
# X = ca.MX.sym('X', 4, N+1)  # State variables: [x, y, psi, v]
# U = ca.MX.sym('U', 2, N)   # Control inputs: [delta, a]

# # Objective and constraints
# obj = 0  # Objective function
# g = []  # Constraints

# # Initial condition constraint
# g.append(X[:,0] - initial_state)

# # Dynamics and cost
# for i in range(N):
#     # System dynamics
#     x_next = X[0,i] + X[3,i] * ca.cos(X[2,i]) * dt
#     y_next = X[1,i] + X[3,i] * ca.sin(X[2,i]) * dt
#     psi_next = X[2,i] + X[3,i] / L * ca.tan(U[0,i]) * dt
#     v_next = X[3,i] + U[1,i] * dt
    
#     # Update objective
#     obj += ca.sumsqr(X[0,i+1] - x_ref[i]) + ca.sumsqr(X[1,i+1] - y_ref[i])
#     obj += ca.sumsqr(X[2,i+1] - psi_ref[i]) + ca.sumsqr(X[3,i+1] - v_ref[i])
    
#     # Update constraints
#     g.append(X[:,i+1] - ca.vertcat(x_next, y_next, psi_next, v_next))

# # NLP setup
# opts = {"ipopt.print_level": 0, "print_time": 0}
# nlp = {'x': ca.vertcat(X.reshape((-1, 1)), U.reshape((-1, 1))), 'f': obj, 'g': ca.vertcat(*g)}
# solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

# # Solve NLP
# sol = solver(x0=np.zeros((X.size()[0]*(N+1) + U.size()[0]*N, 1)), lbg=0, ubg=0, lbx=-ca.inf, ubx=ca.inf)
# u_opt = np.array(sol['x'][-2*N:]).reshape((N, 2))

# print("Optimal inputs (steering angle, acceleration):", u_opt)

# this code above works, heres another version.#

# import casadi as ca
# import numpy as np
# import matplotlib.pyplot as plt

# # Define parameters
# N = 20  # Prediction horizon
# dt = 0.2  # Time step
# L = 2.9  # Wheelbase
# initial_state = np.array([0, 0, 0, 0])  # [y, psi, vy, r] where r is yaw rate

# # Reference trajectory for a right turn
# turn_rate = 0.05  # Rate of turn, radians per step
# psi_ref = np.linspace(0, turn_rate*N, N)
# y_ref = np.linspace(0, 5, N)  # Assuming constant lateral displacement for simplicity
# vy_ref = np.zeros(N)  # No lateral velocity reference for simplicity
# r_ref = np.full(N, turn_rate/dt)  # Constant yaw rate

# # Define CasADi variables
# X = ca.MX.sym('X', 4, N+1)  # State variables: [y, psi, vy, r]
# U = ca.MX.sym('U', 2, N)   # Control inputs: [delta, a]

# # Objective function and constraints
# obj = 0  # Objective function
# g = []  # Constraints
# g.append(X[:,0] - initial_state)

# # Vehicle dynamics model
# for i in range(N):
#     # Simplified vehicle dynamics for lateral control
#     y_next = X[0,i] + X[2,i] * dt
#     psi_next = X[1,i] + X[3,i] * dt
#     vy_next = X[2,i] + (U[1,i] * ca.sin(U[0,i]) - X[3,i] * L / 2.0) * dt
#     r_next = X[3,i] + (U[1,i] * ca.cos(U[0,i]) / L) * dt

#     # Update objective
#     obj += ca.sumsqr(X[0,i+1] - y_ref[i]) + ca.sumsqr(X[1,i+1] - psi_ref[i])
#     obj += ca.sumsqr(X[2,i+1] - vy_ref[i]) + ca.sumsqr(X[3,i+1] - r_ref[i])

#     # Update constraints
#     g.append(X[:,i+1] - ca.vertcat(y_next, psi_next, vy_next, r_next))

# # NLP setup
# opts = {"ipopt.print_level": 0, "print_time": 0}
# nlp = {'x': ca.vertcat(X.reshape((-1, 1)), U.reshape((-1, 1))), 'f': obj, 'g': ca.vertcat(*g)}
# solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

# # Solve NLP
# sol = solver(x0=np.zeros((X.size1()*(N+1) + U.size1()*N, 1)), lbg=0, ubg=0, lbx=-ca.inf, ubx=ca.inf)
# u_opt = np.array(sol['x'][-2*N:]).reshape((N, 2))

# # Extract optimized trajectory
# x_opt = np.array(sol['x'][:4*(N+1)]).reshape((4, N+1))

# # Plotting
# plt.figure(figsize=(10, 5))
# plt.subplot(1, 2, 1)
# plt.plot(x_opt[0, :], x_opt[1, :], 'b-', label='Optimized Path')
# plt.xlabel('Lateral Position (m)')
# plt.ylabel('Yaw Angle (rad)')
# plt.legend()
# plt.title('Vehicle Path')

# plt.subplot(1, 2, 2)
# plt.plot(np.arange(N), u_opt[:, 0], 'g-', label='Steering Angle')
# plt.plot(np.arange(N), u_opt[:, 1], 'r-', label='Acceleration')
# plt.xlabel('Time Step')
# plt.ylabel('Control Inputs')
# plt.legend()
# plt.title('Control Inputs Over Time')

# plt.tight_layout()
# plt.show()


#another version#

import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

# Parameters
N = 10  # Prediction horizon
dt = 0.1  # Time step
L = 2.9  # Wheelbase
m = 100 #mass of the car
#importing stuff two array#


# Initial state [x, y, phi, vx, vy,r(yaw rate),Delta(searing angle),T (throtale)]
initial_state = np.array([0, 0, 0, 0, 0,0])  # Starting with some initial forward velocity
u_0 = np.array([0,1]).T #col vector of 0,1 that we insert on the first step a full press on the throtale. its delta stering angle and delta reaction



# Reference trajectory (straight line for simplicity)
x_ref = np.linspace(0, 10, N)
y_ref = np.zeros(N)  # Stay in the lane
psi_ref = np.zeros(N)  # No change in orientation
vx_ref = 10 * np.ones(N)  # Constant velocity
vy_ref = np.zeros(N)  # No lateral velocity

# Define CasADi variables
X = ca.MX.sym('X', 8, N+1)  # State variables: [x, y, psi, vx, vy]
U = ca.MX.sym('U', 2, N)   # Control inputs: [delta, accl_reaction]

# Objective function and constraints
obj = 0  # Objective function
g = []  # Constraints
g.append(X[:,0] - initial_state)

# Dynamics model
for i in range(N):
    # Extract states for readability
    x, y, phi, vx, vy, r,delta,driver_react = X[0,i], X[1,i], X[2,i], X[3,i], X[4,i],X[5,i],X[6,i],X[7,i]
    delta, ax = U[0,i], U[1,i]
    #state vector aka X_dot
    x_dot[i] = X[2,i]*np.cos(X[2,i])-X[3,i]*np.sin(X[2,i])
    y_dot[i] = X[2,i]*np.sin(X[2,i])+X[3,i]*np.cos(X[2,i])
    phi_dot[i] = r
    a_x[i] = (F_X-F_Fy*np.sin(X[6,i])+m*X[4,i]*r)*1/m
    a_y[i] = (F_Ry+F_Fy.np.cos(X[6,i])-m*X[3,i]*r)*1/m
    r_dot[i] = (F_Fy*lf*np.cos(X[6,i])-F_Ry*lr + Tau_vec)*1/Iz
    Delta_dot[i] = U[0,i]
    T_dot = U[1,i]



    # Update equations based on a simplified bicycle model
    x_next = x + vx * ca.cos(psi) * dt
    y_next = y + vx * ca.sin(psi) * dt
    psi_next = psi + vx / L * ca.tan(delta) * dt
    vx_next = vx + ax * dt
    vy_next = vy  # Assuming no direct control over vy, simplified for this example

    # Update objective (tracking and control effort)
    obj += ca.sumsqr(x_next - x_ref[i]) + ca.sumsqr(y_next - y_ref[i])
    obj += ca.sumsqr(psi_next - psi_ref[i]) + ca.sumsqr(vx_next - vx_ref[i])
    obj += ca.sumsqr(vy_next - vy_ref[i])
    obj += ca.sumsqr(delta) + ca.sumsqr(ax)

    # Dynamics constraints
    g.append(X[:,i+1] - ca.vertcat(x_next, y_next, psi_next, vx_next, vy_next))

# NLP setup
opts = {"ipopt.print_level": 0, "print_time": 0}
nlp = {'x': ca.vertcat(X.reshape((-1, 1)), U.reshape((-1, 1))), 'f': obj, 'g': ca.vertcat(*g)}
solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

# Solve NLP
sol = solver(x0=np.zeros((X.size1()*(N+1) + U.size1()*N, 1)), lbg=0, ubg=0, lbx=-ca.inf, ubx=ca.inf)
u_opt = np.array(sol['x'][-2*N:]).reshape((N, 2))

# Extract optimized trajectory for plotting
x_opt = np.array(sol['x'][:5*(N+1)]).reshape((5, N+1))

# Plotting
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(x_opt[0, :], x_opt[1, :], 'b-', label='Optimized Path')
plt.xlabel('Longitudinal Position (m)')
plt.ylabel('Lateral Position (m)')
plt.legend()
plt.title('Vehicle Path')

plt.subplot(1, 2, 2)
plt.plot(np.arange(N), u_opt[:, 0], 'g-', label='Steering Angle (rad)')
plt.plot(np.arange(N), u_opt[:, 1], 'r-', label='Acceleration (m/s²)')
plt.xlabel('Time Step')
plt.ylabel('Control Inputs')
plt.legend()
plt.title('Control Inputs Over Time')

plt.tight_layout()
plt.show()