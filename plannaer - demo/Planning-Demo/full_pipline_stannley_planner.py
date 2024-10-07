# full_pipeline.py

import numpy as np
import math
import matplotlib.pyplot as plt
from fsd_path_planning import PathPlanner, MissionTypes, ConeTypes
import numba
from fsd_path_planning.utils.math_utils import (
    my_njit,
    norm_of_last_axis,
    normalize_last_axis,
    rotate,
    unit_2d_vector_from_angle
)
from stanley_controller import StanleyController
from simple_pid import PID

# Vehicle parameters
m = 250  # Vehicle mass
I_z = 1700  # Moment of inertia (guess)
l_f = 0.835  # Distance from the center of mass to the front axle
l_r = 0.705  # Distance from the center of mass to the rear axle
WB = l_f + l_r  # Wheelbase [m]

# Controller parameters
k = 1.0      # Look-forward gain for Pure Pursuit (not used in Stanley Controller)
Lfc = 8    # Look-ahead distance for Pure Pursuit (not used in Stanley Controller)
Kp = 0.2     # Speed proportional gain
dt = 0.05     # Time step [s]

# PID Controller parameters
kp = 2.0  # Increase proportional gain
ki = 0.2
kd = 0.1

# Visualization parameters
show_animation = True

class State:
    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v
        self.update_positions()

    def update(self, a, delta):
        self.x += self.v * math.cos(self.yaw) * dt
        self.y += self.v * math.sin(self.yaw) * dt
        self.yaw += self.v / WB * math.tan(delta) * dt
        self.v += a * dt
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
        """
        Initialize the target course.

        Args:
            cx: x coordinates of the course.
            cy: y coordinates of the course.
        """
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
            distance_this_index = state.calc_distance(self.cx[ind], self.cy[ind], use_front=True)
            while True:
                if (ind + 1) >= len(self.cx):
                    break
                distance_next_index = state.calc_distance(self.cx[ind + 1], self.cy[ind + 1], use_front=True)
                if distance_this_index < distance_next_index:
                    break
                ind += 1
                distance_this_index = distance_next_index
            self.old_nearest_point_index = ind

        Lf = k * state.v + Lfc  # Update look-ahead distance

        # Search look ahead target point index based on front axle position
        while Lf > state.calc_distance(self.cx[ind], self.cy[ind], use_front=True):
            if (ind + 1) >= len(self.cx):
                break
            ind += 1

        return ind, Lf

class States:
    def __init__(self):
        """
        Initialize a container to store vehicle states over time.
        """
        self.x = []
        self.y = []
        self.yaw = []
        self.v = []
        self.t = []

    def append(self, t, state):
        """
        Append a state to the container.

        Args:
            t: Time.
            state: State object representing the vehicle state.
        """
        self.x.append(state.x)
        self.y.append(state.y)
        self.yaw.append(state.yaw)
        self.v.append(state.v)
        self.t.append(t)

def proportional_control(target, current):
    """
    Calculate acceleration based on proportional control.

    Args:
        target: Target speed.
        current: Current speed.

    Returns:
        Acceleration.
    """
    a = Kp * (target - current)
    return a

def pid_control(target, current, pid_controller):
    """
    Calculate acceleration based on PID control.

    Args:
        target: Target speed.
        current: Current speed.
        pid_controller: PID controller instance.

    Returns:
        Acceleration.
    """
    return pid_controller(target - current)

def plot_car(x, y, yaw, steer=0.0, truckcolor="-k"):
    """
    Plot the vehicle.

    Args:
        x: x coordinate of the vehicle.
        y: y coordinate of the vehicle.
        yaw: Orientation angle of the vehicle.
        steer: Steering angle of the vehicle.
        truckcolor: Color of the vehicle.
    """
    LENGTH = 4.5  # [m]
    WIDTH = 2.0  # [m]
    BACKTOWHEEL = 1.0  # [m]
    WHEEL_LEN = 0.3  # [m]
    WHEEL_WIDTH = 0.2  # [m]
    TREAD = 0.7  # [m]
    WB2 = 2.5  # [m]

    outline = np.array([[-BACKTOWHEEL, (LENGTH - BACKTOWHEEL), (LENGTH - BACKTOWHEEL),
                         -BACKTOWHEEL, -BACKTOWHEEL],
                        [WIDTH / 2, WIDTH / 2, - WIDTH / 2, -WIDTH / 2, WIDTH / 2]])

    fr_wheel = np.array([[WHEEL_LEN, -WHEEL_LEN, -WHEEL_LEN, WHEEL_LEN, WHEEL_LEN],
                         [-WHEEL_WIDTH - TREAD, -WHEEL_WIDTH - TREAD,
                          WHEEL_WIDTH - TREAD, WHEEL_WIDTH - TREAD, -WHEEL_WIDTH - TREAD]])

    rr_wheel = fr_wheel.copy()
    fl_wheel = fr_wheel.copy()
    rl_wheel = fr_wheel.copy()
    fl_wheel[1, :] *= -1
    rl_wheel[1, :] *= -1

    Rot1 = np.array([[math.cos(yaw), math.sin(yaw)],
                     [-math.sin(yaw), math.cos(yaw)]])
    Rot2 = np.array([[math.cos(steer), math.sin(steer)],
                     [-math.sin(steer), math.cos(steer)]])

    fr_wheel = (fr_wheel.T @ Rot2).T
    fl_wheel = (fl_wheel.T @ Rot2).T
    fr_wheel[0, :] += WB2
    fl_wheel[0, :] += WB2

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

def plot_cones(cones_by_type):
    """
    Plot the cones on the track.

    Args:
        cones_by_type: List of arrays containing cone positions by type.
    """
    cone_colors = ['gray', 'blue', 'yellow', 'green', 'red']
    labels = ['Unknown', 'Right', 'Left', 'Start Finish Area', 'Start Finish Line']

    for i, cones in enumerate(cones_by_type):
        if len(cones) > 0:
            plt.scatter(cones[:, 0], cones[:, 1], c=cone_colors[i], label=labels[i], s=20)

def load_cones():
    """
    Simulate or load cone positions.

    Returns:
        global_cones, car_position, car_direction
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
    for i, c in enumerate(ConeTypes):
        print(c, f"= {i}")

        cones_by_type = [np.zeros((0, 2)) for _ in range(5)]
        cones_by_type[ConeTypes.LEFT] = cones_left
        cones_by_type[ConeTypes.RIGHT] = cones_right
        cones_by_type[ConeTypes.UNKNOWN] = cones_unknown
    return cones_by_type, car_position, car_direction

def main():
    # Initialize path planner
    path_planner = PathPlanner(MissionTypes.trackdrive)
    global_cones, car_position, car_direction = load_cones()

    # Generate path using the path planner
    path = path_planner.calculate_path_in_global_frame(
        global_cones, car_position, car_direction
    )

    # Extract x and y coordinates from the path
    # The path is an Mx4 array: [spline parameter, x, y, curvature]
    cx = path[:, 1]
    cy = path[:, 2]

    # Initialize the target course with the generated path
    target_course = TargetCourse(cx, cy)

    # Initial state
    state = State(
        x=car_position[0],
        y=car_position[1],
        yaw=np.arctan2(car_direction[1], car_direction[0]),
        v=0.00001  # Set an initial velocity to get the vehicle moving
    )

    target_speed = 15.0 / 3.6  # [m/s] Target speed
    T = 500.0  # Max simulation time

    # Initialize simulation variables
    time = 0.0
    states = States()
    states.append(time, state)

    # Find the initial target index
    target_ind, _ = target_course.search_target_index(state)
    lastIndex = len(cx) - 1

    # Initialize Stanley Controller
    stanley_controller = StanleyController(k=0.8)  # Slightly reduced gain for smoother steering

    # Initialize PID Controller for speed
    pid_controller = PID(1.5, 0.1, 0.05, setpoint=target_speed)
    #pid_controller.windup_guard = 2.0  # Limit on the integral term to avoid windup

    # Define maximum limits
    MAX_ACCEL = 1.5  # Reduce maximum acceleration for stability [m/s^2]
    MAX_SPEED = 15.0 / 3.6  # Reduce maximum speed to 10 km/h for more control [m/s]
    MAX_STEER = np.deg2rad(20)  # Reduce maximum steering angle [rad]

    # Start the main loop
    while T >= time and lastIndex > target_ind:
        # Calculate control input for acceleration
        ai = pid_control(MAX_SPEED, state.v, pid_controller)
        #ai = max(min(ai, MAX_ACCEL), -MAX_ACCEL)  # Limit acceleration

        # Calculate steering using Stanley Controller
        di, target_ind = stanley_controller.control(state, target_course, target_ind, use_front=True)
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
            plot_car(state.x, state.y, state.yaw, steer=di)
            plot_cones(global_cones)
            plt.axis("equal")
            plt.grid(True)
            plt.title(f"Speed [km/h]: {state.v * 3.6:.2f}")
            plt.pause(0.001)

    # Final plotting
    plt.figure()
    plt.plot(cx, cy, "r--", label="Planned Path")
    plt.plot(states.x, states.y, "-b", label="Vehicle Path")
    plot_cones(global_cones)
    plt.legend()
    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")
    plt.axis("equal")
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    main()