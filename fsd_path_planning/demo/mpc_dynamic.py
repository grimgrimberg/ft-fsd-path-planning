from __future__ import annotations
import casadi as ca
#this is a try to mix both up#
from fsd_path_planning.demo.json_demo import app
from typing import Any, List, Optional, Union
import json
import os
import numpy as np
from pathlib import Path

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
from fsd_path_planning.calculate_path.core_calculate_path import CalculatePath
#from fsd_path_planning.calculate_path.core_calculate_path import CalculatePath.do_all_mpc_parameter_calculations ,CalculatePath.create_path_for_mpc_from_path_update
from fsd_path_planning.utils.mission_types import MissionTypes
from fsd_path_planning.utils.utils import Timer

# MPC Setup
N = 40  # Prediction horizon
dt = 0.05  # Time step
#do_all_mpc_parameter_calculations()

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
#constraints for steering
Max_Delta = 0.418879 # 0.418879 radians = 24 degrees
Min_Delta = -Max_Delta
Max_T = 1 # 1 max Torque (full gas)
Min_T = -Max_T

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

#constraints 
# Define maximum and minimum limits for the steering angle and throttle
max_delta = 0.418879  # 24 degrees in radians
min_delta = -max_delta
max_T = 1  # Maximum throttle
min_T = -1  # Minimum throttle (or braking force)

# Assuming `U` is your control input vector where U[0] is delta (steering angle) and U[1] is T (throttle)
constraints = []
U = ca.SX.sym('U', 2)  # Control input vector
constraints.append(U[0] - max_delta)  # Steering angle should not exceed max_delta
constraints.append(min_delta - U[0])  # Steering angle should be above min_delta
constraints.append(U[1] - max_T)  # Throttle should not exceed max_T
constraints.append(min_T - U[1])  # Throttle should be above min_T

max_v = 100  # Maximum velocity, e.g., 360 km/h in m/s
min_v = 0  # Minimum velocity

# Assuming `v_x` is the longitudinal velocity of the vehicle
v_x = ca.SX.sym('v_x')
constraints.append(v_x - max_v)  # Velocity should not exceed max_v
constraints.append(min_v - v_x)  # Velocity should be above min_v


# The nominal model equations
x_dot = ca.vertcat(
    v_x * ca.cos(psi) - v_y * ca.sin(psi),
    v_x * ca.sin(psi) + v_y * ca.cos(psi),
    r,
    (1/m) * (F_rx - F_ry * ca.sin(delta) + m * v_y * r),#A_X
    (1/m) * (F_ry + F_rx * ca.cos(delta) - m * v_x * r),
    (1/I_z) * (F_ry * l_f * ca.cos(delta) - F_rx * l_r + T * r * v_x),
    delta_dot,
    T_dot
)

Next_X = ca.vertcat(state_vector + x_dot * dt) #next state is this state + state derivatives * dt.
print ("this is the next state",Next_X)

# Print out the state derivatives (for demonstration purposes; remove in actual use)
print("State Derivatives:", x_dot)
# Display the initial conditions and initial inputs (for demonstration purposes; remove in actual use)
print("Initial State Vector:", initial_conditions)
print("Initial Inputs:", initial_inputs)

def load_data_json(
    data_path: Path,
    remove_color_info: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[list[np.ndarray]]]:
    """
    Load data from a JSON file and return the positions, directions, and cone observations.

    Args:
        data_path (Path): The path to the JSON data file.
        remove_color_info (bool, optional): Whether to remove color information. Defaults to False.

    Returns:
        tuple[np.ndarray, np.ndarray, list[list[np.ndarray]]]: A tuple containing the positions (np.ndarray), 
        directions (np.ndarray), and cone observations (list[list[np.ndarray]]).
    """
    # extract data
    data_path = get_filename()
    data = json.loads(data_path.read_text())[:]

    positions = np.array([d["car_position"] for d in data])
    directions = np.array([d["car_direction"] for d in data])
    cone_observations = [
        [np.array(c).reshape(-1, 2) for c in d["slam_cones"]] for d in data
    ]

    if remove_color_info:
        cones_observations_all_unknown = []
        for cones in cone_observations:
            new_observation = [np.zeros((0, 2)) for _ in ConeTypes]
            new_observation[ConeTypes.UNKNOWN] = np.row_stack(
                [c.reshape(-1, 2) for c in cones]
            )
            cones_observations_all_unknown.append(new_observation)

        cone_observations = cones_observations_all_unknown.copy()

    return positions, directions, cone_observations

def get_filename(data_path: Path | None) -> Path:
    """
    Function to get the filename from the given data path.

    :param data_path: The path to the file. If None, a default file path is used.
    :type data_path: Path | None
    :return: The path to the file.
    :rtype: Path
    """
    if data_path is None:
        data_path = Path(__file__).parent / "fsg_19_2_laps.json"

    return data_path

if __name__ == "__main__":
    #data_path = []
    data_path = get_filename(None)
    positions, directions, cone_observations = load_data_json(data_path,bool=False)

