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

# Vehicle parameters
m = 250  # Vehicle mass
I_z = 1700  # Moment of inertia (guess)
l_f = 0.835  # Distance from the center of mass to the front axle
l_r = 0.705  # Distance from the center of mass to the rear axle
WB = l_f + l_r  # Wheelbase [m]

# Pure Pursuit parameters
k = 0.5      # Look-forward gain
Lfc = 4.0    # Look-ahead distance
Kp = 0.2     # Speed proportional gain
dt = 0.1     # Time step [s]

# Visualization parameters
show_animation = True

class State:
    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0):
        """
        Initialize the state of the vehicle.

        Args:
            x: Initial x position.
            y: Initial y position.
            yaw: Initial yaw angle (orientation).
            v: Initial velocity.
        """
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v
        self.rear_x = self.x - (l_f * math.cos(self.yaw))
        self.rear_y = self.y - (l_f * math.sin(self.yaw))

    def update(self, a, delta):
        """
        Update the state of the vehicle based on control inputs.

        Args:
            a: Acceleration.
            delta: Steering angle.
        """
        self.x += self.v * math.cos(self.yaw) * dt
        self.y += self.v * math.sin(self.yaw) * dt
        self.yaw += self.v / WB * math.tan(delta) * dt
        self.v += a * dt
        self.rear_x = self.x - (l_f * math.cos(self.yaw))
        self.rear_y = self.y - (l_f * math.sin(self.yaw))

    def calc_distance(self, point_x, point_y):
        """
        Calculate the distance between the rear of the vehicle and a point.

        Args:
            point_x: x coordinate of the point.
            point_y: y coordinate of the point.

        Returns:
            Distance between the vehicle's rear and the point.
        """
        dx = self.rear_x - point_x
        dy = self.rear_y - point_y
        return math.hypot(dx, dy)

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
        """
        Search for the target index on the course.

        Args:
            state: State object representing the vehicle state.

        Returns:
            Target index and look-ahead distance.
        """
        if self.old_nearest_point_index is None:
            # Search nearest point index
            dx = [state.rear_x - icx for icx in self.cx]
            dy = [state.rear_y - icy for icy in self.cy]
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

        Lf = k * state.v + Lfc  # Update look-ahead distance

        # Search look ahead target point index
        while Lf > state.calc_distance(self.cx[ind], self.cy[ind]):
            if (ind + 1) >= len(self.cx):
                break
            ind += 1

        return ind, Lf

def pure_pursuit_steer_control(state, trajectory, pind):
    """
    Calculate steering angle using pure pursuit algorithm.

    Args:
        state: State object representing the vehicle state.
        trajectory: Target trajectory.
        pind: Previous target index.

    Returns:
        Steering angle and updated target index.
    """
    ind, Lf = trajectory.search_target_index(state)

    if pind >= ind:
        ind = pind

    if ind < len(trajectory.cx):
        tx = trajectory.cx[ind]
        ty = trajectory.cy[ind]
    else:  # Toward goal
        tx = trajectory.cx[-1]
        ty = trajectory.cy[-1]
        ind = len(trajectory.cx) - 1

    alpha = math.atan2(ty - state.rear_y, tx - state.rear_x) - state.yaw

    delta = math.atan2(2.0 * WB * math.sin(alpha) / Lf, 1.0)

    return delta, ind

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
    return cones_by_type,car_position,car_direction

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
        v=0.0
    )

    target_speed = 15.0 / 3.6  # [m/s] Target speed
    T = 500.0  # Max simulation time

    # Initialize simulation variables
    time = 0.0
    states = States()
    states.append(time, state)
    target_ind, _ = target_course.search_target_index(state)
    lastIndex = len(cx) - 1

    while T >= time and lastIndex > target_ind:
        # Calculate control input
        ai = proportional_control(target_speed, state.v)
        di, target_ind = pure_pursuit_steer_control(state, target_course, target_ind)

        # Update state
        state.update(ai, di)
        time += dt
        states.append(time, state)
        deg_rad = unit_2d_vector_from_angle(di)
        print("this is the accalaration",ai, "and this is the deg and the idx",deg_rad,target_ind)


        if show_animation:
            plt.cla()
            plt.gcf().canvas.mpl_connect(
                'key_release_event',
                lambda event: [exit(0) if event.key == 'escape' else None])
            plt.plot(cx, cy, "r--", label="Planned Path")
            plt.plot(states.x, states.y, "-b", label="Vehicle Path")
            plt.plot(cx[target_ind], cy[target_ind], "xg", label="Target")
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
