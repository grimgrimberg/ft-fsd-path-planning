from __future__ import annotations
import casadi as ca
#this is a try to mix both up#

from typing import Any, List, Optional, Union

import numpy as np

from fsd_path_planning.calculate_path.core_calculate_path import PathCalculationInput
from fsd_path_planning.cone_matching.core_cone_matching import ConeMatchingInput
from fsd_path_planning.config import (
    create_default_cone_matching_with_non_monotonic_matches,
    create_default_pathing,
    create_default_sorting,
)
from fsd_path_planning.skidpad.skidpad_path_data import BASE_SKIDPAD_PATH
from fsd_path_planning.skidpad.skidpad_relocalizer import SkidpadRelocalizer
from fsd_path_planning.sorting_cones.core_cone_sorting import ConeSortingInput
from fsd_path_planning.types import FloatArray, IntArray
from fsd_path_planning.utils.cone_types import ConeTypes
from fsd_path_planning.utils.math_utils import (
    angle_from_2d_vector,
    unit_2d_vector_from_angle,
)
from fsd_path_planning.calculate_path.core_calculate_path import do_all_mpc_parameter_calculations
from fsd_path_planning.utils.mission_types import MissionTypes
from fsd_path_planning.utils.utils import Timer
# MPC Setup
N = 40  # Prediction horizon
dt = 0.05  # Time step

# Define the parameters
m = ca.SX.sym('m')  # mass of the vehicle
I_z = ca.SX.sym('I_z')  # moment of inertia about the vertical axis
l_f = ca.SX.sym('l_f')  # distance from the center of mass to the front axle
l_r = ca.SX.sym('l_r')  # distance from the center of mass to the rear axle
F_rx = ca.SX.sym('F_rx')  # external force applied to the vehicle in the x direction
F_ry = ca.SX.sym('F_ry')  # external force applied to the vehicle in the y direction
T = ca.SX.sym('T')  # torque applied by the driver command

#Defenition of paramaters with values

m = 250  # mass of the vehicle (not final)
I_z = 1700 #(this is a guess)  # moment of inertia about the vertical axis
l_f = 0.835  # distance from the center of mass to the front axle
l_r = 0.705  # distance from the center of mass to the rear axle
#L TOTAL = LF+LR = 1.54.

# Define the state vector components
X = ca.SX.sym('X')
Y = ca.SX.sym('Y')
psi = ca.SX.sym('psi') #Heading angle
v_x = ca.SX.sym('v_x')
v_y = ca.SX.sym('v_y')
r = ca.SX.sym('r') #yaw rate
delta = ca.SX.sym('delta') #stearing angle
T = ca.SX.sym('T')

# Tire and vehicle parameters
D_r = ca.SX.sym('D_r')  # Peak lateral force - rear
D_f = ca.SX.sym('D_f')  # Peak lateral force - front
C_r = ca.SX.sym('C_r')  # Shape factor - rear
C_f = ca.SX.sym('C_f')  # Shape factor - front
B_r = ca.SX.sym('B_r')  # Stiffness factor - rear
B_f = ca.SX.sym('B_f')  # Stiffness factor - front
C_m1 = ca.SX.sym('C_m1')  # Scaling factor for driver command
C_r0 = ca.SX.sym('C_r0')  # Constant rolling resistance force
C_r2 = ca.SX.sym('C_r2')  # Aerodynamic drag factor

# Define slip angles
alpha_r = ca.atan2(v_y - l_r * r, v_x)
alpha_f = ca.atan2(v_y + l_f * r, v_x) - delta

# Lateral forces using a simplified Pacejka tire model
F_ry = D_r * ca.sin(C_r * ca.atan(B_r * alpha_r))
F_fy = D_f * ca.sin(C_f * ca.atan(B_f * alpha_f))

# Longitudinal force
F_x = C_m1 * T - C_r0 - C_r2 * v_x**2

# Now, combine these forces into a vector (for usage in the model)
forces = ca.vertcat(F_x, F_ry, F_fy)

# Print out the forces (for demonstration purposes; remove in actual use)
print("Forces:", forces)


# Combine them into a state vector
state_vector = ca.vertcat(X, Y, psi, v_x, v_y, r, delta, T)

# Define initial conditions (all zeros)
initial_conditions = ca.DM.zeros(state_vector.size())

# Define the inputs to the system
delta_change = ca.SX.sym('delta_change')  # Change in steering angle
T_command = ca.SX.sym('T_command')  # Driver command (throttle/brake)

# Define the input vector components
delta_dot = ca.SX.sym('delta_dot')  # rate of change of steering angle
T_dot = ca.SX.sym('T_dot')  # rate of change of driver command

# Set the initial input values
initial_inputs = ca.DM([0, 1])  # 0 for steering angle change, 1 for driver command

# The nominal model equations
x_dot = ca.vertcat(
    v_x * ca.cos(psi) - v_y * ca.sin(psi),
    v_x * ca.sin(psi) + v_y * ca.cos(psi),
    r,
    (1/m) * (F_rx - F_ry * ca.sin(delta) + m * v_y * r),
    (1/m) * (F_ry + F_rx * ca.cos(delta) - m * v_x * r),
    (1/I_z) * (F_ry * l_f * ca.cos(delta) - F_rx * l_r + T * r * v_x),
    delta_dot,
    T_dot
)

# Print out the state derivatives (for demonstration purposes; remove in actual use)
print("State Derivatives:", x_dot)
# Display the initial conditions and initial inputs (for demonstration purposes; remove in actual use)
print("Initial State Vector:", initial_conditions)
print("Initial Inputs:", initial_inputs)
