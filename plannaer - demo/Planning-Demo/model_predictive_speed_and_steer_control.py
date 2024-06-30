"""

Path tracking simulation with iterative linear model predictive control for speed and steer control


"""
import matplotlib.pyplot as plt
import cvxpy
import math
import numpy as np
import sys
import pathlib
import csv
import pandas as pd
import os
from utils.angle import angle_mod

sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))
from PathPlanning.CubicSpline import cubic_spline_planner


# Define constants and parameters
NX = 4  # State dimension: x, y, v, yaw
NU = 2  # Control input dimension: acceleration, steering angle
T = 5  # Horizon length

# MPC parameters
R = np.diag([0.01, 0.01])  # Input cost matrix
Rd = np.diag([0.01, 1.0])  # Input difference cost matrix
Q = np.diag([1.0, 1.0, 0.5, 0.5])  # State cost matrix
Qf = Q  # Final state cost matrix
GOAL_DIS = 1.5  # Goal distance
STOP_SPEED = 0.5 / 3.6  # Stop speed
MAX_TIME = 70.0  # Max simulation time

# Iterative parameters
MAX_ITER = 3  # Max iteration
DU_TH = 0.1  # Iteration finish parameter

TARGET_SPEED = 10.0 / 3.6  # Target speed [m/s]
N_IND_SEARCH = 10  # Search index number
DT = 0.3  # Time tick [s]

# Vehicle parameters
LENGTH = 4.5  # Length of the vehicle [m]
WIDTH = 2.0  # Width of the vehicle [m]
BACKTOWHEEL = 1.0  # Distance from back to wheel [m]
WHEEL_LEN = 0.3  # Length of the wheel [m]
WHEEL_WIDTH = 0.2  # Width of the wheel [m]
TREAD = 0.7  # Tread of the vehicle [m]
WB = 2.5  # Wheelbase of the vehicle [m]

MAX_STEER = np.deg2rad(45.0)  # Maximum steering angle [rad]
MAX_DSTEER = np.deg2rad(30.0)  # Maximum steering speed [rad/s]
MAX_SPEED = 55.0 / 3.6  # Maximum speed [m/s]
MIN_SPEED = -20.0 / 3.6  # Minimum speed [m/s]
MAX_ACCEL = 1.0  # Maximum acceleration [m/s^2]

show_animation = True


class State:
    """
    vehicle state class
    """

    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v
        self.predelta = None


def pi_2_pi(angle):
    """
    Map an angle to the range [-pi, pi].
    """
    return angle_mod(angle)


def get_linear_model_matrix(v, phi, delta):
    """
    Calculate the linear model matrices.

    Args:
        v: Velocity.
        phi: Yaw angle.
        delta: Steering angle.

    Returns:
        A tuple containing matrices A, B, and C.
    """
    A = np.zeros((NX, NX))
    A[0, 0] = 1.0
    A[1, 1] = 1.0
    A[2, 2] = 1.0
    A[3, 3] = 1.0
    A[0, 2] = DT * math.cos(phi)
    A[0, 3] = - DT * v * math.sin(phi)
    A[1, 2] = DT * math.sin(phi)
    A[1, 3] = DT * v * math.cos(phi)
    A[3, 2] = DT * math.tan(delta) / WB

    B = np.zeros((NX, NU))
    B[2, 0] = DT
    B[3, 1] = DT * v / (WB * math.cos(delta) ** 2)

    C = np.zeros(NX)
    C[0] = DT * v * math.sin(phi) * phi
    C[1] = - DT * v * math.cos(phi) * phi
    C[3] = - DT * v * delta / (WB * math.cos(delta) ** 2)

    return A, B, C


def plot_car(x, y, yaw, steer=0.0, truckcolor="-k"):  # pragma: no cover
    """
    Plot the vehicle.

    Args:
        x: x coordinate of the vehicle.
        y: y coordinate of the vehicle.
        yaw: Orientation angle of the vehicle.
        steer: Steering angle of the vehicle.
        truckcolor: Color of the vehicle.
    """
    outline = np.array([[-BACKTOWHEEL, (LENGTH - BACKTOWHEEL), (LENGTH - BACKTOWHEEL), -BACKTOWHEEL, -BACKTOWHEEL],
                        [WIDTH / 2, WIDTH / 2, - WIDTH / 2, -WIDTH / 2, WIDTH / 2]])

    fr_wheel = np.array([[WHEEL_LEN, -WHEEL_LEN, -WHEEL_LEN, WHEEL_LEN, WHEEL_LEN],
                         [-WHEEL_WIDTH - TREAD, -WHEEL_WIDTH - TREAD, WHEEL_WIDTH - TREAD, WHEEL_WIDTH - TREAD,
                          -WHEEL_WIDTH - TREAD]])

    rr_wheel = np.copy(fr_wheel)

    fl_wheel = np.copy(fr_wheel)
    fl_wheel[1, :] *= -1
    rl_wheel = np.copy(rr_wheel)
    rl_wheel[1, :] *= -1

    Rot1 = np.array([[math.cos(yaw), math.sin(yaw)],
                     [-math.sin(yaw), math.cos(yaw)]])
    Rot2 = np.array([[math.cos(steer), math.sin(steer)],
                     [-math.sin(steer), math.cos(steer)]])

    fr_wheel = (fr_wheel.T.dot(Rot2)).T
    fl_wheel = (fl_wheel.T.dot(Rot2)).T
    fr_wheel[0, :] += WB
    fl_wheel[0, :] += WB

    fr_wheel = (fr_wheel.T.dot(Rot1)).T
    fl_wheel = (fl_wheel.T.dot(Rot1)).T

    outline = (outline.T.dot(Rot1)).T
    rr_wheel = (rr_wheel.T.dot(Rot1)).T
    rl_wheel = (rl_wheel.T.dot(Rot1)).T

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

    plt.plot(np.array(outline[0, :]).flatten(),
             np.array(outline[1, :]).flatten(), truckcolor)
    plt.plot(np.array(fr_wheel[0, :]).flatten(),
             np.array(fr_wheel[1, :]).flatten(), truckcolor)
    plt.plot(np.array(rr_wheel[0, :]).flatten(),
             np.array(rr_wheel[1, :]).flatten(), truckcolor)
    plt.plot(np.array(fl_wheel[0, :]).flatten(),
             np.array(fl_wheel[1, :]).flatten(), truckcolor)
    plt.plot(np.array(rl_wheel[0, :]).flatten(),
             np.array(rl_wheel[1, :]).flatten(), truckcolor)
    # plt.plot(x, y, "*")


def update_state(state, a, delta):
    """
    Update the vehicle state based on control inputs.

    Args:
        state: Current state of the vehicle.
        a: Acceleration.
        delta: Steering angle.

    Returns:
        State: Updated state of the vehicle.
    """
    # input check
    if delta >= MAX_STEER:
        delta = MAX_STEER
    elif delta <= -MAX_STEER:
        delta = -MAX_STEER

    state.x = state.x + state.v * math.cos(state.yaw) * DT
    state.y = state.y + state.v * math.sin(state.yaw) * DT
    state.yaw = state.yaw + state.v / WB * math.tan(delta) * DT
    state.v = state.v + a * DT

    if state.v > MAX_SPEED:
        state.v = MAX_SPEED
    elif state.v < MIN_SPEED:
        state.v = MIN_SPEED

    return state


def get_nparray_from_matrix(x):
    """
    Convert a matrix to a numpy array.
    """
    return np.array(x).flatten()


def calc_nearest_index(state, cx, cy, cyaw, pind):
    """
    Calculate the index of the nearest point on the path to the vehicle.

    Args:
        state: Current state of the vehicle.
        cx: List of x-coordinates of the path.
        cy : List of y-coordinates of the path.
        cyaw : List of yaw angles of the path.
        pind: Previous index.

    Returns:
        A tuple containing the index and distance to the nearest point.
    """
    dx = [state.x - icx for icx in cx[pind:(pind + N_IND_SEARCH)]]
    dy = [state.y - icy for icy in cy[pind:(pind + N_IND_SEARCH)]]

    d = [idx ** 2 + idy ** 2 for (idx, idy) in zip(dx, dy)]

    mind = min(d)

    ind = d.index(mind) + pind

    mind = math.sqrt(mind)

    dxl = cx[ind] - state.x
    dyl = cy[ind] - state.y

    angle = pi_2_pi(cyaw[ind] - math.atan2(dyl, dxl))
    if angle < 0:
        mind *= -1

    return ind, mind


def predict_motion(x0, oa, od, xref):
    """
    Predict the vehicle motion based on linear model predictive control.

    Args:
        x0: Initial state of the vehicle.
        oa: List of acceleration inputs.
        od: List of steering inputs.
        xref: Reference trajectory.

    Returns:
        Predicted vehicle motion.
    """
    xbar = xref * 0.0
    for i, _ in enumerate(x0):
        xbar[i, 0] = x0[i]

    state = State(x=x0[0], y=x0[1], yaw=x0[3], v=x0[2])
    for (ai, di, i) in zip(oa, od, range(1, T + 1)):
        state = update_state(state, ai, di)
        xbar[0, i] = state.x
        xbar[1, i] = state.y
        xbar[2, i] = state.v
        xbar[3, i] = state.yaw

    return xbar


def iterative_linear_mpc_control(xref, x0, dref, oa, od):
    """
    MPC control with updating operational point iteratively

    Args:
        xref: Reference trajectory.
        x0: Initial state of the vehicle.
        dref: Reference steering angle.
        oa: List of previous acceleration inputs.
        od: List of previous steering inputs.

    Returns:
        A tuple containing the acceleration inputs, steering inputs,
       and updated vehicle motion.
    """
    ox, oy, oyaw, ov = None, None, None, None

    if oa is None or od is None:
        oa = [0.0] * T
        od = [0.0] * T

    for i in range(MAX_ITER):
        xbar = predict_motion(x0, oa, od, xref)
        poa, pod = oa[:], od[:]
        oa, od, ox, oy, oyaw, ov = linear_mpc_control(xref, xbar, x0, dref)
        du = sum(abs(oa - poa)) + sum(abs(od - pod))  # calc u change value
        if du <= DU_TH:
            break
    else:
        print("Iterative is max iter")

    return oa, od, ox, oy, oyaw, ov


def linear_mpc_control(xref, xbar, x0, dref):
    """
    Perform linear model predictive control.

    Args:
        xref: Reference trajectory.
        xbar: Operational point trajectory.
        x0: Initial state of the vehicle.
        dref: Reference steering angle.

    Returns:
        A tuple containing the acceleration inputs, steering inputs,
       and updated vehicle motion.
    """

    x = cvxpy.Variable((NX, T + 1))
    u = cvxpy.Variable((NU, T))

    cost = 0.0
    constraints = []

    for t in range(T):
        cost += cvxpy.quad_form(u[:, t], R)

        if t != 0:
            cost += cvxpy.quad_form(xref[:, t] - x[:, t], Q)

        A, B, C = get_linear_model_matrix(
            xbar[2, t], xbar[3, t], dref[0, t])
        constraints += [x[:, t + 1] == A @ x[:, t] + B @ u[:, t] + C]

        if t < (T - 1):
            cost += cvxpy.quad_form(u[:, t + 1] - u[:, t], Rd)
            constraints += [cvxpy.abs(u[1, t + 1] - u[1, t]) <=
                            MAX_DSTEER * DT]

    cost += cvxpy.quad_form(xref[:, T] - x[:, T], Qf)

    constraints += [x[:, 0] == x0]
    constraints += [x[2, :] <= MAX_SPEED]
    constraints += [x[2, :] >= MIN_SPEED]
    constraints += [cvxpy.abs(u[0, :]) <= MAX_ACCEL]
    constraints += [cvxpy.abs(u[1, :]) <= MAX_STEER]

    prob = cvxpy.Problem(cvxpy.Minimize(cost), constraints)
    prob.solve(solver=cvxpy.ECOS, verbose=False)

    if prob.status == cvxpy.OPTIMAL or prob.status == cvxpy.OPTIMAL_INACCURATE:
        ox = get_nparray_from_matrix(x.value[0, :])
        oy = get_nparray_from_matrix(x.value[1, :])
        ov = get_nparray_from_matrix(x.value[2, :])
        oyaw = get_nparray_from_matrix(x.value[3, :])
        oa = get_nparray_from_matrix(u.value[0, :])
        odelta = get_nparray_from_matrix(u.value[1, :])

    else:
        print("Error: Cannot solve mpc..")
        oa, odelta, ox, oy, oyaw, ov = None, None, None, None, None, None

    return oa, odelta, ox, oy, oyaw, ov


def calc_ref_trajectory(state, cx, cy, cyaw, ck, sp, dl, pind):
    """
    Calculate the reference trajectory.

    Args:
        state: Current state of the vehicle.
        cx: List of x-coordinates of the path.
        cy: List of y-coordinates of the path.
        cyaw: List of yaw angles of the path.
        ck : List of curvatures of the path.
        sp: Speed profile.
        dl: Course tick.
        pind: Previous index.

    Returns:
        A tuple containing the reference trajectory, current index,
       and reference steering angle.
    """
    xref = np.zeros((NX, T + 1))
    dref = np.zeros((1, T + 1))
    ncourse = len(cx)

    ind, _ = calc_nearest_index(state, cx, cy, cyaw, pind)

    if pind >= ind:
        ind = pind

    xref[0, 0] = cx[ind]
    xref[1, 0] = cy[ind]
    xref[2, 0] = sp[ind]
    xref[3, 0] = cyaw[ind]
    dref[0, 0] = 0.0  # steer operational point should be 0

    travel = 0.0

    for i in range(T + 1):
        travel += abs(state.v) * DT
        dind = int(round(travel / dl))

        if (ind + dind) < ncourse:
            xref[0, i] = cx[ind + dind]
            xref[1, i] = cy[ind + dind]
            xref[2, i] = sp[ind + dind]
            xref[3, i] = cyaw[ind + dind]
            dref[0, i] = 0.0
        else:
            xref[0, i] = cx[ncourse - 1]
            xref[1, i] = cy[ncourse - 1]
            xref[2, i] = sp[ncourse - 1]
            xref[3, i] = cyaw[ncourse - 1]
            dref[0, i] = 0.0

    return xref, ind, dref


def check_goal(state, goal, tind, nind):
    """
    Check if the vehicle has reached the goal.

    Args:
        state: Current state of the vehicle.
        goal: Goal position [x, y].
        tind : Target index.
        nind: Next index.

    Returns:
        True if the goal is reached, False otherwise.
    """
    # check goal
    dx = state.x - goal[0]
    dy = state.y - goal[1]
    d = math.hypot(dx, dy)

    isgoal = (d <= GOAL_DIS)

    if abs(tind - nind) >= 5:
        isgoal = False

    isstop = (abs(state.v) <= STOP_SPEED)

    if isgoal and isstop:
        return True

    return False


def do_simulation(cx, cy, cyaw, ck, sp, dl, initial_state, cones):
    """
    Perform simulation of the vehicle trajectory.

    Args:
        cx: List of x-coordinates of the path.
        cy: List of y-coordinates of the path.
        cyaw: List of yaw angles of the path.
        ck: List of curvatures of the path.
        sp: Speed profile.
        dl: Course tick.
        initial_state: Initial state of the vehicle.
        cones: Dataframe containing cone positions and colors.

    Returns:
        A tuple containing lists of time, x-coordinate, y-coordinate, yaw angle,
        velocity, steering angle, and acceleration.
    """

    goal = [cx[-1], cy[-1]]

    state = initial_state

    # initial yaw compensation
    if state.yaw - cyaw[0] >= math.pi:
        state.yaw -= math.pi * 2.0
    elif state.yaw - cyaw[0] <= -math.pi:
        state.yaw += math.pi * 2.0

    time = 0.0
    x = [state.x]
    y = [state.y]
    yaw = [state.yaw]
    v = [state.v]
    t = [0.0]
    d = [0.0]
    a = [0.0]
    target_ind, _ = calc_nearest_index(state, cx, cy, cyaw, 0)

    odelta, oa = None, None

    cyaw = smooth_yaw(cyaw)

    while MAX_TIME >= time:
        xref, target_ind, dref = calc_ref_trajectory(
            state, cx, cy, cyaw, ck, sp, dl, target_ind)

        x0 = [state.x, state.y, state.v, state.yaw]  # current state

        oa, odelta, ox, oy, oyaw, ov = iterative_linear_mpc_control(
            xref, x0, dref, oa, odelta)

        di, ai = 0.0, 0.0
        if odelta is not None:
            di, ai = odelta[0], oa[0]
            state = update_state(state, ai, di)

        time = time + DT

        x.append(state.x)
        y.append(state.y)
        yaw.append(state.yaw)
        v.append(state.v)
        t.append(time)
        d.append(di)
        a.append(ai)

        if check_goal(state, goal, target_ind, len(cx)):
            print("Goal")
            break

        if show_animation:  # pragma: no cover
            plt.cla()
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect('key_release_event',
                                         lambda event: [exit(0) if event.key == 'escape' else None])
            plt.plot(cx, cy, "tab:red", label="course")
            plt.plot(x, y, "-k", label="trajectory")
            plt.plot(xref[0, :], xref[1, :], "orange", label="xref")
            plt.plot(cx[target_ind], cy[target_ind], "b", label="target")
            if ox is not None:
                plt.plot(ox, oy, "-g", label="MPC")
            plt.scatter(cones['x'], cones['y'], c=cones['color'].map({'Y': 'gold', 'B': 'tab:blue'}), s=5)
            plot_car(state.x, state.y, state.yaw, steer=di)
            plt.axis("equal")
            # plt.grid(True)
            plt.title("Time[s]:" + str(round(time, 2))
                      + ", Speed[km/h]:" + str(round(state.v * 3.6, 2)))
            plt.pause(0.0001)

    return t, x, y, yaw, v, d, a


def calc_speed_profile(cx, cy, cyaw, target_speed):
    """
    Calculate the speed profile along the path.

    Args:
        cx: List of x-coordinates of the path.
        cy: List of y-coordinates of the path.
        cyaw: List of yaw angles of the path.
        target_speed: Target speed.

    Returns:
        Speed profile along the path.
    """
    speed_profile = [target_speed] * len(cx)
    direction = 1.0  # forward

    # Set stop point
    for i in range(len(cx) - 1):
        dx = cx[i + 1] - cx[i]
        dy = cy[i + 1] - cy[i]

        move_direction = math.atan2(dy, dx)

        if dx != 0.0 and dy != 0.0:
            dangle = abs(pi_2_pi(move_direction - cyaw[i]))
            if dangle >= math.pi / 4.0:
                direction = -1.0
            else:
                direction = 1.0

        if direction != 1.0:
            speed_profile[i] = - target_speed
        else:
            speed_profile[i] = target_speed

    speed_profile[-1] = 0.0

    return speed_profile


def smooth_yaw(yaw):
    """
    Smooth the yaw angles to avoid discontinuities.

    Args:
        yaw: List of yaw angles.

    Returns:
        Smoothed yaw angles.
    """
    for i in range(len(yaw) - 1):
        dyaw = yaw[i + 1] - yaw[i]

        while dyaw >= math.pi / 2.0:
            yaw[i + 1] -= math.pi * 2.0
            dyaw = yaw[i + 1] - yaw[i]

        while dyaw <= -math.pi / 2.0:
            yaw[i + 1] += math.pi * 2.0
            dyaw = yaw[i + 1] - yaw[i]

    return yaw


def read_csv_points(file_path):
    """
    Read x, y points from a CSV file.

    Args:
        file_path: Path to the CSV file.

    Returns:
        A tuple containing lists of x and y points.
    """
    x_points = []
    y_points = []
    counter = 0
    with open(file_path, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)  # Skip header if present
        for row in csvreader:
            counter += 1
            if counter % 2 == 0:
                x_points.append(float(row[0]))
                y_points.append(float(row[1]))
    return x_points, y_points


def load_and_concatenate_data(yellow_file, blue_file):
    """
    Load cone positions from CSV files and concatenate them.

    Args:
        yellow_file: Path to the CSV file containing yellow cone positions.
        blue_file: Path to the CSV file containing blue cone positions.

    Returns:
        Concatenated dataframe containing cone positions and colors.
    """
    yellow_cone_df = pd.read_csv(yellow_file)
    blue_cone_df = pd.read_csv(blue_file)

    concatenated_df = pd.concat([
        pd.concat([yellow_cone_df.iloc[i:i + 1].assign(color='Y'),
                   blue_cone_df.iloc[i:i + 1].assign(color='B')])
        for i in range(min(len(yellow_cone_df), len(blue_cone_df)))
    ], ignore_index=True)

    return concatenated_df


def main():
    print(__file__ + " start!!")
    
    csv_filename = 'midpoint.csv'
    track_points_x, track_points_y = read_csv_points(csv_filename)

    yellow_file = 'C:\Users\yuval\ft-fsd-path-planning\plannaer - demo\Planning-Demo\yellow_cone_position.csv'
    blue_file = 'C:\Users\yuval\ft-fsd-path-planning\plannaer - demo\Planning-Demo\blue_cone_position.csv'
    cones = load_and_concatenate_data(yellow_file, blue_file)

    dl = 1.0  # course tick
    cx, cy, cyaw, ck, s = cubic_spline_planner.calc_spline_course(track_points_x, track_points_y, ds=dl)

    sp = calc_speed_profile(cx, cy, cyaw, TARGET_SPEED)

    initial_state = State(x=cx[0], y=cy[0], yaw=cyaw[0], v=0.0)

    t, x, y, yaw, v, d, a = do_simulation(
        cx, cy, cyaw, ck, sp, dl, initial_state, cones)

    if show_animation:  # pragma: no cover
        plt.close("all")
        plt.subplots()
        plt.plot(cx, cy, "-r", label="spline")
        plt.plot(x, y, "-g", label="tracking")
        plt.grid(True)
        plt.axis("equal")
        plt.xlabel("x[m]")
        plt.ylabel("y[m]")
        plt.legend()

        plt.subplots()
        plt.plot(t, v, "-r", label="speed")
        plt.grid(True)
        plt.xlabel("Time [s]")
        plt.ylabel("Speed [kmh]")

        plt.show()


if __name__ == '__main__':
    main()
