# full_pipeline_dynamic_bicycle.py

import numpy as np
import math
import matplotlib.pyplot as plt
from fsd_path_planning import PathPlanner, MissionTypes, ConeTypes
from fsd_path_planning.utils.math_utils import (
    rotate,
    unit_2d_vector_from_angle
)
from dynamic_bicycle_model import DynamicBicycleModel
from stanley_controller import StanleyController
from simple_pid import PID

# Vehicle parameters
m = 250  # Vehicle mass [kg]
I_z = 1700  # Moment of inertia [kg*m^2]
l_f = 0.835  # Distance from the center of mass to the front axle [m]
l_r = 0.705  # Distance from the center of mass to the rear axle [m]
c_f = 16000  # Cornering stiffness front [N/rad]
c_r = 17000  # Cornering stiffness rear [N/rad]
mu = 1.0  # Coefficient of friction

dt = 0.05  # Time step [s]

# PID Controller parameters
kp = 2.0
ki = 0.2
kd = 0.1

# Visualization parameters
show_animation = True

class State:
    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0, beta=0.0, r=0.0):
        self.x = x
        self.y = y
        self.yaw = yaw  # Vehicle heading
        self.v = v  # Velocity
        self.beta = beta  # Slip angle
        self.r = r  # Yaw rate
        self.update_positions()

    def update(self, a, delta):
        # Dynamic bicycle model integration step
        dynamic_model = DynamicBicycleModel(m, I_z, l_f, l_r, c_f, c_r, mu)
        x_next = dynamic_model.predict_next_state([self.x, self.y, self.yaw, self.v, self.r, self.beta], [delta, a], dt)

        # Unpack the updated state
        self.x, self.y, self.yaw, self.v, self.r, self.beta = x_next
        self.update_positions()

    def update_positions(self):
        """
        Update the positions of both the rear and front axles of the vehicle based on the current state.
        """
        self.rear_x = self.x - (l_f * math.cos(self.yaw))
        self.rear_y = self.y - (l_f * math.sin(self.yaw))
        self.front_x = self.x + (l_r * math.cos(self.yaw))
        self.front_y = self.y + (l_r * math.sin(self.yaw))

    def calc_distance(self, point_x, point_y, use_front=True):
        """
        Calculate the distance between the vehicle and a point.

        Args:
            point_x: x coordinate of the point.
            point_y: y coordinate of the point.
            use_front: Boolean indicating whether to use the front or rear axle for calculation.

        Returns:
            Distance between the vehicle and the point.
        """
        if use_front:
            dx = self.front_x - point_x
            dy = self.front_y - point_y
        else:
            dx = self.rear_x - point_x
            dy = self.rear_y - point_y
        return math.hypot(dx, dy)


class TargetCourse:
    def __init__(self, cx, cy):
        self.cx = cx
        self.cy = cy
        self.old_nearest_point_index = None

    def search_target_index(self, state):
        if self.old_nearest_point_index is None:
            # Search nearest point index based on front axle position
            dx = [state.front_x - icx for icx in self.cx]
            dy = [state.front_y - icy for icy in self.cy]
            d = np.hypot(dx, dy)
            ind = np.argmin(d)
            self.old_nearest_point_index = ind
        else:
            ind = self.old_nearest_point_index
            distance_this_index = state.calc_distance(self.cx[ind], self.cy[ind])
            while True:
                if (ind + 1) >= len(self.cx):
                    break
                distance_next_index = state.calc_distance(self.cx[ind + 1], self.cy[ind + 1])
                if distance_this_index < distance_next_index:
                    break
                ind += 1
                distance_this_index = distance_next_index
            self.old_nearest_point_index = ind

        return ind

class States:
    def __init__(self):
        self.x = []
        self.y = []
        self.yaw = []
        self.v = []
        self.t = []

    def append(self, t, state):
        self.x.append(state.x)
        self.y.append(state.y)
        self.yaw.append(state.yaw)
        self.v.append(state.v)
        self.t.append(t)

# Load cones function

def load_cones():
    """
    Simulate or load cone positions.

    Returns:
        cones_by_type, car_position, car_direction
    """
    phi_inner = np.arange(0, np.pi / 2, np.pi / 20)
    phi_outer = np.arange(0, np.pi / 2, np.pi / 25)

    points_inner = unit_2d_vector_from_angle(phi_inner) * 9
    points_outer = unit_2d_vector_from_angle(phi_outer) * 12

    center = np.mean((points_inner[:2] + points_outer[:2]) / 2, axis=0)
    points_inner -= center
    points_outer -= center

    rotated_points_inner = rotate(points_inner, -np.pi / 2)
    rotated_points_outer = rotate(points_outer, -np.pi / 2)
    cones_left_raw = rotated_points_inner
    cones_right_raw = rotated_points_outer

    rng = np.random.default_rng(0)
    rng.shuffle(cones_left_raw)
    rng.shuffle(cones_right_raw)

    car_position = np.array([0.0, 0.0])
    car_direction = np.array([1.0, 0.0])
    mask_is_left = np.ones(len(cones_left_raw), dtype=bool)
    mask_is_right = np.ones(len(cones_right_raw), dtype=bool)

    # for demonstration purposes, we will only keep the color of the first 4 cones
    # on each side
    mask_is_left[np.argsort(np.linalg.norm(cones_left_raw, axis=1))[20:]] = False
    mask_is_right[np.argsort(np.linalg.norm(cones_right_raw, axis=1))[20:]] = False

    cones_left = cones_left_raw[mask_is_left]
    cones_right = cones_right_raw[mask_is_right]
    cones_unknown = np.row_stack(
        [cones_left_raw[~mask_is_left], cones_right_raw[~mask_is_right]]
    )

    # Initialize cones by type and add debug statements
    cones_by_type = [np.zeros((0, 2)) for _ in range(5)]
    cones_by_type[ConeTypes.LEFT] = cones_left
    cones_by_type[ConeTypes.RIGHT] = cones_right
    cones_by_type[ConeTypes.UNKNOWN] = cones_unknown

    # Debug print statements to ensure proper initialization
    print("Cones by type:")
    for i, cones in enumerate(cones_by_type):
        print(f"Type {i}: {len(cones)} cones")

    return cones_by_type, car_position, car_direction

# Plot cones function
def plot_cones(cones_by_type):
    cones_left = cones_by_type[ConeTypes.LEFT]
    cones_right = cones_by_type[ConeTypes.RIGHT]
    plt.plot(cones_left[:, 0], cones_left[:, 1], "ob", label="Left Cones")
    plt.plot(cones_right[:, 0], cones_right[:, 1], "oy", label="Right Cones")

# Plot car function
def plot_car(x, y, yaw, steer=0.0, truckcolor="-k"):
    LENGTH = 4.5  # [m]
    WIDTH = 2.0  # [m]
    BACKTOWHEEL = 1.0  # [m]
    WHEEL_LEN = 0.3  # [m]
    WHEEL_WIDTH = 0.2  # [m]
    TREAD = 0.7  # [m]
    WB = 2.5  # [m]

    outline = np.array([[-BACKTOWHEEL, (LENGTH - BACKTOWHEEL), (LENGTH - BACKTOWHEEL), -BACKTOWHEEL, -BACKTOWHEEL],
                        [WIDTH / 2, WIDTH / 2, - WIDTH / 2, -WIDTH / 2, WIDTH / 2]])

    fr_wheel = np.array([[WHEEL_LEN, -WHEEL_LEN, -WHEEL_LEN, WHEEL_LEN, WHEEL_LEN],
                         [-WHEEL_WIDTH - TREAD, -WHEEL_WIDTH - TREAD, WHEEL_WIDTH - TREAD, WHEEL_WIDTH - TREAD,
                          -WHEEL_WIDTH - TREAD]])

    rr_wheel = fr_wheel.copy()
    fl_wheel = fr_wheel.copy()
    rl_wheel = fr_wheel.copy()
    fl_wheel[1, :] *= -1
    rl_wheel[1, :] *= -1

    Rot1 = np.array([[math.cos(yaw), math.sin(yaw)], [-math.sin(yaw), math.cos(yaw)]])
    Rot2 = np.array([[math.cos(steer), math.sin(steer)], [-math.sin(steer), math.cos(steer)]])

    fr_wheel = (fr_wheel.T @ Rot2).T
    fl_wheel = (fl_wheel.T @ Rot2).T
    fr_wheel[0, :] += WB
    fl_wheel[0, :] += WB

    fr_wheel = (fr_wheel.T @ Rot1).T
    fl_wheel = (fl_wheel.T @ Rot1).T
    outline = (outline.T @ Rot1).T
    rr_wheel = (rr_wheel.T @ Rot1).T
    rl_wheel = (rl_wheel.T @ Rot1).T

    outline[0, :] += x
    outline[1, :] += y
    fr_wheel[0, :] += x
    fr_wheel[1, :] += y
    rr_wheel[0, :] += x
    rr_wheel[1, :] += y
    fl_wheel[0, :] += x
    fl_wheel[1, :] += y
    rl_wheel[0, :] += x
    rl_wheel[1, :] += y

    plt.plot(outline[0, :], outline[1, :], truckcolor)
    plt.plot(fr_wheel[0, :], fr_wheel[1, :], truckcolor)
    plt.plot(rr_wheel[0, :], rr_wheel[1, :], truckcolor)
    plt.plot(fl_wheel[0, :], fl_wheel[1, :], truckcolor)
    plt.plot(rl_wheel[0, :], rl_wheel[1, :], truckcolor)

# Main function
def main():
    # Initialize path planner
    path_planner = PathPlanner(MissionTypes.trackdrive)
    cones_by_type, car_position, car_direction = load_cones()

    # Generate path using the path planner
    path = path_planner.calculate_path_in_global_frame(
        cones_by_type, car_position, car_direction
    )

    # Extract x and y coordinates from the path
    cx = path[:, 1]
    cy = path[:, 2]

    # Initialize the target course with the generated path
    target_course = TargetCourse(cx, cy)

    # Initial state
    state = State(
        x=car_position[0],
        y=car_position[1],
        yaw=np.arctan2(car_direction[1], car_direction[0]),
        v=0.0  # Initial velocity is set to 0
    )

    target_speed = 15.0 / 3.6  # Target speed [m/s]
    T = 500.0  # Max simulation time [s]

    # Apply an initial acceleration command to get moving
    initial_acceleration = 1.0  # Initial acceleration [m/s^2]
    state.update(initial_acceleration, 0.0)

    # Initialize simulation variables
    time = 0.0
    states = States()
    states.append(time, state)

    # Find the initial target index
    target_ind = target_course.search_target_index(state)
    lastIndex = len(cx) - 1

    # Initialize PID Controller for speed
    pid_controller = PID(kp, ki, kd, setpoint=target_speed)
    pid_controller.windup_guard = 2.0

    # Define maximum limits
    MAX_ACCEL = 1.5  # Maximum acceleration [m/s^2]
    MAX_SPEED = 10.0 / 3.6  # Maximum speed [m/s]
    MAX_STEER = np.deg2rad(20)  # Maximum steering angle [rad]

    # Start the main loop
    while T >= time and lastIndex > target_ind:
        # Calculate control input for acceleration
        ai = pid_controller(target_speed - state.v)
        ai = max(min(ai, MAX_ACCEL), -MAX_ACCEL)  # Limit acceleration

        # Calculate steering using a basic heading alignment for now
        target_yaw = math.atan2(cy[target_ind] - state.y, cx[target_ind] - state.x)
        steering_error = target_yaw - state.yaw
        steering_error = (steering_error + math.pi) % (2 * math.pi) - math.pi
        di = di = 0.8 * steering_error  # Simple proportional steering control
        di = max(min(di, MAX_STEER), -MAX_STEER)  # Limit steering angle

        # Update vehicle state
        state.update(ai, di)

        # Limit the speed of the vehicle
        state.v = min(state.v, MAX_SPEED)

        # Refine target index update to prevent sticking at the current point
        if state.calc_distance(target_course.cx[target_ind], target_course.cy[target_ind]) < 1.0 and target_ind < len(cx) - 1:
            target_ind += 1

        # Update time and states
        time += dt
        states.append(time, state)

        # Debug print for controller outputs
        print(f"Time: {time:.2f}, Acceleration (ai): {ai:.2f}, Steering Angle (di): {di:.2f}, Target Index: {target_ind}")
        print(f"Vehicle State -> x: {state.x:.2f}, y: {state.y:.2f}, yaw: {state.yaw:.2f}, velocity: {state.v:.2f}")

        # Visualization
        if show_animation:
            plt.cla()
            plt.gcf().canvas.mpl_connect(
                'key_release_event',
                lambda event: [exit(0) if event.key == 'escape' else None])
            plt.plot(cx, cy, "r--", label="Planned Path")
            plt.plot(states.x, states.y, "-b", label="Vehicle Path")
            plt.plot(cx[target_ind], cy[target_ind], "xg", label="Target")
            plot_cones(cones_by_type)
            plot_car(state.x, state.y, state.yaw, steer=di)
            plt.axis("equal")
            plt.grid(True)
            plt.title(f"Speed [km/h]: {state.v * 3.6:.2f}")
            plt.pause(0.001)

    # Final plotting
    plt.figure()
    plt.plot(cx, cy, "r--", label="Planned Path")
    plt.plot(states.x, states.y, "-b", label="Vehicle Path")
    plt.legend()
    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")
    plt.axis("equal")
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()
