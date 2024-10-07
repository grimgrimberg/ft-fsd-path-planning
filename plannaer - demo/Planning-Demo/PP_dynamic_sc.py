from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from pathlib import Path
from typing import Optional
import pandas as pd
from fsd_path_planning import ConeTypes, MissionTypes, PathPlanner
from fsd_path_planning.utils.utils import Timer
from fsd_path_planning.demo.json_demo import load_data_json,get_filename,select_mission_by_filename,PathPlanner


try:
    import matplotlib.animation
    import matplotlib.pyplot as plt
    import typer
except ImportError:
    print(
        "\n\nThis demo requires matplotlib and typer to be installed. You can install"
        " them with by using the [demo] extra.\n\n"
    )
    raise

try:
    from tqdm import tqdm
except ImportError:
    print("You can get a progress bar by installing tqdm: pip install tqdm")
    tqdm = lambda x, total=None: x


try:
    app = typer.Typer(pretty_exceptions_enable=False)
except TypeError:
    app = typer.Typer()

# Vehicle parameters
vehicle_params = {
    'mass': 250.0,            # Mass of the vehicle in kg
    'I_z': 120.0,             # Yaw moment of inertia in kg*m^2 (estimate)
    'wheelbase': 1.5,         # Distance between front and rear axles in meters
    'lf': 0.75,               # Distance from CG to front axle
    'lr': 0.75,               # Distance from CG to rear axle
    'max_steering_angle': np.radians(30),  # Maximum steering angle in radians
    'max_acceleration': 5.0,  # Maximum acceleration in m/s^2
    'max_deceleration': -5.0, # Maximum deceleration in m/s^2
    # Pacejka tire model coefficients (example values)
    'B': 10.0,
    'C': 1.9,
    'D': 0.7 * 250 * 9.81 / 2,  # Peak force per tire (70% of half vehicle weight)
    'E': 0.97
}

data_path: Optional[Path] = typer.Option(None, "--data-path", "-i"),
data_rate: float = 10,
remove_color_info: bool = False,
show_runtime_histogram: bool = False,
output_path: Optional[Path] = typer.Option(None, "--output-path", "-o"),
"""
    A function to generate a main animation based on given parameters and data. The function takes in various parameters such as data_path, data_rate, remove_color_info, show_runtime_histogram, and output_path. It then performs several operations such as loading data, warming up the JIT compiler, running the planner, calculating paths, and generating an animation. The function also saves the animation to the specified output path.
    """
#data_path = get_filename(data_path)
data_path = Path('C:/Users/yuval/ft-fsd-path-planning/plannaer - demo/Planning-Demo/fsg_19_2_laps.json')

#mission = select_mission_by_filename(data_path.name)

planner = PathPlanner(MissionTypes.trackdrive)

positions, directions, cone_observations = load_data_json(
        data_path, remove_color_info=remove_color_info
    )


    # run planner once to "warm up" the JIT compiler / load all cached jit functions
try:
        planner.calculate_path_in_global_frame(
            cone_observations[0], positions[0], directions[0]
        )
except Exception:
        print("Error during warmup")
        raise

results = []
timer = Timer(noprint=True)

for i, (position, direction, cones) in tqdm(
    enumerate(zip(positions, directions, cone_observations)),
    total=len(positions),
    desc="Calculating paths",
    ):
        try:
            with timer:
                out = planner.calculate_path_in_global_frame(
                    cones,
                    position,
                    direction,
                    return_intermediate_results=True,
                )
        except KeyboardInterrupt:
            print(f"Interrupted by user on frame {i}")
            break
        except Exception:
            print(f"Error at frame {i}")
            raise
        results.append(out)

# Import your planner and load data (assuming these functions are defined)
# from fsd_path_planning import PathPlanner, MissionTypes, ConeTypes
# path_planner = PathPlanner(MissionTypes.trackdrive)
# global_cones, car_position, car_direction = load_data()
# path = path_planner.calculate_path_in_global_frame(global_cones, car_position, car_direction)

# For the sake of this example, let's create a sample path
# Replace this with your actual path from the planner
num_points = 1000
t = np.linspace(0, 50, num_points)
#path = np.zeros((num_points, 4))
#path[:, 0] = t  # Spline parameter
#path[:, 1] = t  # x-coordinate (straight line for simplicity)
#path[:, 2] = np.sin(t / 5) * 10  # y-coordinate (wavy path)
#path[:, 3] = np.gradient(np.arctan2(np.gradient(path[:,2]), np.gradient(path[:,1])), t)  # Curvature

# Initial vehicle state
state = np.array([
    position[0], 
    position[1], 
    np.arctan2(direction[1], direction[0]), 
    0.1,   # v_x: Small initial speed to avoid division by zero
    0.0,   # v_y
    0.0    # omega
])


# Pacejka tire model function
def pacejka_tire_model(alpha, params):
    B, C, D, E = params['B'], params['C'], params['D'], params['E']
    F = D * np.sin(C * np.arctan(B * alpha - E * (B * alpha - np.arctan(B * alpha))))
    return F

# Dynamic bicycle model function
def dynamic_bicycle_model(state, steering_angle, acceleration_command, vehicle_params, dt):
    x, y, psi, v_x, v_y, omega = state
    m = vehicle_params['mass']
    I_z = vehicle_params['I_z']
    lf = vehicle_params['lf']
    lr = vehicle_params['lr']
    
    # Avoid division by zero
    if v_x < 0.1:
        v_x = 0.1

    # Compute slip angles
    alpha_f = steering_angle - np.arctan2(v_y + lf * omega, v_x)
    alpha_r = -np.arctan2(v_y - lr * omega, v_x)
    
    # Tire forces
    F_yf = 2 * pacejka_tire_model(alpha_f, vehicle_params)  # Front lateral force
    F_yr = 2 * pacejka_tire_model(alpha_r, vehicle_params)  # Rear lateral force

    # Longitudinal forces (assuming rear-wheel drive)
    F_xf = 0.0  # No longitudinal force at the front wheels
    F_xr = m * acceleration_command  # Rear longitudinal force
    
    # Equations of motion
    v_x_dot = (F_xf * np.cos(steering_angle) - F_yf * np.sin(steering_angle) + F_xr) / m + v_y * omega
    v_y_dot = (F_xf * np.sin(steering_angle) + F_yf * np.cos(steering_angle) + F_yr) / m - v_x * omega
    omega_dot = (lf * (F_yf * np.cos(steering_angle) + F_xf * np.sin(steering_angle)) - lr * F_yr) / I_z
    
    # Update velocities
    v_x += v_x_dot * dt
    v_y += v_y_dot * dt
    omega += omega_dot * dt
    
    # Update positions
    x += (v_x * np.cos(psi) - v_y * np.sin(psi)) * dt
    y += (v_x * np.sin(psi) + v_y * np.cos(psi)) * dt
    psi += omega * dt
    
    # Return updated state
    return np.array([x, y, psi, v_x, v_y, omega])

# Helper functions for the pure pursuit controller
def find_closest_point(x, y, out):
    distances = np.sqrt((out[:, 1] - x)**2 + (out[:, 2] - y)**2)
    closest_idx = np.argmin(distances)
    return closest_idx

def find_lookahead_point(closest_idx, path, lookahead_distance, x, y):
    accumulated_distance = 0.0
    for i in range(closest_idx, len(path) - 1):
        dx = path[i + 1, 1] - path[i, 1]
        dy = path[i + 1, 2] - path[i, 2]
        segment_length = np.sqrt(dx**2 + dy**2)
        accumulated_distance += segment_length
        if accumulated_distance >= lookahead_distance:
            return path[i + 1, 1:3], i + 1  # Return (x, y) of lookahead point and its index
    return path[-1, 1:3], len(path) - 1

# Pure pursuit control function
def pure_pursuit_control(state, path, lookahead_distance, vehicle_params):
    x, y, psi = state[0], state[1], state[2]
    wheelbase = vehicle_params['wheelbase']

    # Find the closest point on the path
    closest_idx = find_closest_point(x, y, path)

    # Find the lookahead point
    lookahead_point, lookahead_idx = find_lookahead_point(closest_idx, path, lookahead_distance, x, y)

    # Transform lookahead point to vehicle coordinates
    dx = lookahead_point[0] - x
    dy = lookahead_point[1] - y

    # Rotate to vehicle coordinate frame
    local_x = np.cos(-psi) * dx - np.sin(-psi) * dy
    local_y = np.sin(-psi) * dx + np.cos(-psi) * dy

    # Calculate the steering angle
    if local_x == 0:
        curvature = 0.0
    else:
        curvature = (2 * local_y) / (local_x**2 + local_y**2)
    steering_angle = np.arctan(curvature * wheelbase)

    # Clamp the steering angle to the vehicle's limits
    max_steering = vehicle_params['max_steering_angle']
    steering_angle = np.clip(steering_angle, -max_steering, max_steering)

    return steering_angle, lookahead_idx

# Compute desired speeds from path curvature
def compute_desired_speed(path, max_speed, min_speed, scaling_factor):
    #curvatures = np.abs(path[:, 3])  # Get the curvature column
    curvatures = out[0][3]
    desired_speeds = np.zeros(len(curvatures))
    for i, curvature in enumerate(curvatures):
        if curvature == 0:
            desired_speeds[i] = max_speed
        else:
            desired_speed = np.sqrt(scaling_factor / curvature)
            desired_speeds[i] = np.clip(desired_speed, min_speed, max_speed)
    return desired_speeds

# PID controller class
class PIDController:
    def __init__(self, Kp, Ki, Kd, dt, output_limits=(None, None)):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.dt = dt
        self.integral = 0.0
        self.prev_error = 0.0
        self.min_output, self.max_output = output_limits

    def compute(self, setpoint, measurement):
        error = setpoint - measurement
        self.integral += error * self.dt
        derivative = (error - self.prev_error) / self.dt

        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative

        # Clamp the output to output limits
        if self.max_output is not None:
            output = min(self.max_output, output)
        if self.min_output is not None:
            output = max(self.min_output, output)

        self.prev_error = error
        return output

# Simulation parameters
dt = 0.01                # Time step in seconds
simulation_time = 50.0   # Total simulation time in seconds
time_steps = int(simulation_time / dt)

# Generate desired speeds from path curvature
max_speed = 20.0  # Maximum speed in m/s
min_speed = 5.0   # Minimum speed in m/s
scaling_factor = 2.0  # Tuning parameter for speed adjustment
desired_speeds = compute_desired_speed(out, max_speed, min_speed, scaling_factor)

# Initialize controllers
lookahead_distance = 5.0  # meters
Kp = 1.0
Ki = 0.1
Kd = 0.01
pid_controller = PIDController(Kp, Ki, Kd, dt, output_limits=(vehicle_params['max_deceleration'], vehicle_params['max_acceleration']))

# Data storage for plotting and analysis
x_history = []
y_history = []
yaw_history = []
velocity_history = []
steering_history = []
acceleration_history = []
time_history = []

# Simulation loop
for step in range(time_steps):
    current_time = step * dt

    # Get control inputs
    steering_angle, lookahead_idx = pure_pursuit_control(state, out, lookahead_distance, vehicle_params)
    desired_speed = desired_speeds[lookahead_idx]
    current_speed = state[3]  # v_x
    acceleration_command = pid_controller.compute(desired_speed, current_speed)

    # Clamp acceleration command to vehicle limits
    acceleration_command = np.clip(acceleration_command, vehicle_params['max_deceleration'], vehicle_params['max_acceleration'])

    # Update vehicle state
    state = dynamic_bicycle_model(state, steering_angle, acceleration_command, vehicle_params, dt)

    # Store data
    x_history.append(state[0])
    y_history.append(state[1])
    yaw_history.append(state[2])
    velocity_history.append(state[3])
    steering_history.append(steering_angle)
    acceleration_history.append(acceleration_command)
    time_history.append(current_time)

    # Check for end of path
    if lookahead_idx >= len(out) - 1:
        print("Reached the end of the path.")
        break

# Plotting the path and vehicle trajectory
plt.figure(figsize=(10, 6))
plt.plot(out[:, 1], out[:, 2], 'r--', label='Reference Path')
plt.plot(x_history, y_history, 'b-', label='Vehicle Trajectory')
plt.xlabel('X Position (m)')
plt.ylabel('Y Position (m)')
plt.title('Vehicle Path Tracking')
plt.legend()
plt.axis('equal')
plt.grid(True)
plt.show()

# Plotting speed profile and vehicle speed
plt.figure(figsize=(10, 6))
plt.plot(time_history, velocity_history, 'b-', label='Vehicle Speed')
desired_speed_history = [desired_speeds[find_closest_point(x, y, out)] for x, y in zip(x_history, y_history)]
plt.plot(time_history, desired_speed_history, 'r--', label='Desired Speed')
plt.xlabel('Time (s)')
plt.ylabel('Speed (m/s)')
plt.title('Speed Profile')
plt.legend()
plt.grid(True)
plt.show()
