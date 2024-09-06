import numpy as np
import math
import matplotlib.pyplot as plt
import csv
import pandas as pd

# Vehicle parameters
m = 250  # mass of the vehicle [kg]
I_z = 1700  # moment of inertia about the vertical axis [kg.m^2]
l_f = 0.835  # distance from the center of mass to the front axle [m]
l_r = 0.705  # distance from the center of mass to the rear axle [m]
C_f = 1600.0  # cornering stiffness front [N/rad]
C_r = 1700.0  # cornering stiffness rear [N/rad]
WB = l_f + l_r  # Wheelbase of the vehicle [m]

# Parameters
k = 0.8  # look forward gain
Lfc = 4  # [m] look-ahead distance
Kp = 0.2  # speed proportional gain
dt = 0.1  # [s] time tick
max_speed = 17 / 3.6  # [m/s] max speed (~15 km/h)

# Display parameters
show_animation = True

class State:
    def __init__(self, x=0.0, y=0.0, yaw=0.0, v_x=0, v_y=0.0, r=0.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v_x = v_x  # Longitudinal velocity
        self.v_y = v_y  # Lateral velocity
        self.r = r  # Yaw rate
        self.max_power = 80000  # Peak power of 80 kW

    def update(self, throttle, delta):
        # Calculate the power supplied by the motors based on throttle input (0 to 1)
        # Cap the power at the peak value (80 kW)
        power_available = throttle * self.max_power

        # Prevent division by zero
        if self.v_x == 0.0:
            self.v_x = 1e-3

        # Convert power to force (P = F * v_x => F = P / v_x)
        F_x = power_available / self.v_x
        a_x = F_x / m  # Longitudinal acceleration

        # Dynamic Bicycle Model for Lateral Forces
        F_yf = 2 * C_f * (delta - (self.v_y + l_f * self.r) / self.v_x)
        F_yr = 2 * C_r * (- (self.v_y - l_r * self.r) / self.v_x)

        # Calculate the lateral acceleration
        a_y = (F_yf + F_yr) / m

        # Update velocities
        self.v_x += a_x * dt  # Update longitudinal velocity
        self.v_y += a_y * dt - self.v_x * self.r * dt  # Update lateral velocity

        # Update yaw rate
        self.r += (l_f * F_yf - l_r * F_yr) / I_z * dt  # Update yaw based on lateral forces

        # Update position in global coordinates
        self.x += (self.v_x * np.cos(self.yaw) - self.v_y * np.sin(self.yaw)) * dt
        self.y += (self.v_x * np.sin(self.yaw) + self.v_y * np.cos(self.yaw)) * dt
        self.yaw += self.r * dt  # Update yaw based on yaw rate

        # Cap the speed to avoid runaway acceleration
        if self.v_x > max_speed:
            self.v_x = max_speed

    def calc_distance(self, point_x, point_y):
        dx = self.x - point_x
        dy = self.y - point_y
        return math.hypot(dx, dy)

class States:
    def __init__(self):
        self.x = []
        self.y = []
        self.yaw = []
        self.v_x = []
        self.v_y = []
        self.t = []

    def append(self, t, state):
        self.x.append(state.x)
        self.y.append(state.y)
        self.yaw.append(state.yaw)
        self.v_x.append(state.v_x)
        self.v_y.append(state.v_y)
        self.t.append(t)

def proportional_control(target, current):
    return Kp * (target - current)

class TargetCourse:
    def __init__(self, cx, cy):
        self.cx = cx
        self.cy = cy
        self.old_nearest_point_index = None

    def search_target_index(self, state):
        if self.old_nearest_point_index is None:
            dx = [state.x - icx for icx in self.cx]
            dy = [state.y - icy for icy in self.cy]
            d = np.hypot(dx, dy)
            ind = np.argmin(d)
            self.old_nearest_point_index = ind
        else:
            ind = self.old_nearest_point_index
            distance_this_index = state.calc_distance(self.cx[ind], self.cy[ind])
            while True:
                distance_next_index = state.calc_distance(self.cx[ind + 1], self.cy[ind + 1])
                if distance_this_index < distance_next_index:
                    break
                ind = ind + 1 if (ind + 1) < len(self.cx) else ind
                distance_this_index = distance_next_index
            self.old_nearest_point_index = ind

        Lf = k * state.v_x + Lfc  # update look-ahead distance

        while Lf > state.calc_distance(self.cx[ind], self.cy[ind]):
            if (ind + 1) >= len(self.cx):
                break
            ind += 1

        return ind, Lf

def pure_pursuit_steer_control(state, trajectory, pind):
    ind, Lf = trajectory.search_target_index(state)
    if pind >= ind:
        ind = pind
    if ind < len(trajectory.cx):
        tx = trajectory.cx[ind]
        ty = trajectory.cy[ind]
    else:
        tx = trajectory.cx[-1]
        ty = trajectory.cy[-1]
        ind = len(trajectory.cx) - 1

    alpha = math.atan2(ty - state.y, tx - state.x) - state.yaw
    delta = math.atan2(2.0 * WB * math.sin(alpha) / Lf, 1.0)

    # Limit the steering angle to avoid excessive turns
    max_delta = np.radians(30)  # Limit steering to 30 degrees
    delta = np.clip(delta, -max_delta, max_delta)

    return delta, ind


def plot_car(x, y, yaw, steer=0.0, truckcolor="-k"):
    LENGTH = 4.5  # [m]
    WIDTH = 2.0  # [m]
    BACKTOWHEEL = 1.0  # [m]
    WHEEL_LEN = 0.3  # [m]
    WHEEL_WIDTH = 0.2  # [m]
    TREAD = 0.7  # [m]
    WB2 = 2.5  # [m]

    outline = np.array([[-BACKTOWHEEL, (LENGTH - BACKTOWHEEL), (LENGTH - BACKTOWHEEL), -BACKTOWHEEL, -BACKTOWHEEL],
                        [WIDTH / 2, WIDTH / 2, - WIDTH / 2, -WIDTH / 2, WIDTH / 2]])

    fr_wheel = np.array([[WHEEL_LEN, -WHEEL_LEN, -WHEEL_LEN, WHEEL_LEN, WHEEL_LEN],
                         [-WHEEL_WIDTH - TREAD, -WHEEL_WIDTH - TREAD, WHEEL_WIDTH - TREAD, WHEEL_WIDTH - TREAD, -WHEEL_WIDTH - TREAD]])

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
    fr_wheel[0, :] += WB2
    fl_wheel[0, :] += WB2

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

def read_csv_points(filename):
    x = []
    y = []
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header line
        for row in reader:
            x.append(float(row[0]))
            y.append(float(row[1]))
    return x, y

def load_and_concatenate_data(yellow_file, blue_file):
    yellow_cone_df = pd.read_csv(yellow_file)
    blue_cone_df = pd.read_csv(blue_file)

    concatenated_df = pd.concat([
        pd.concat([yellow_cone_df.iloc[i:i+1].assign(color='Y'),
                   blue_cone_df.iloc[i:i+1].assign(color='B')])
        for i in range(min(len(yellow_cone_df), len(blue_cone_df)))
    ], ignore_index=True)

    return concatenated_df

def calculate_curvature(cx, cy, target_ind):
    if target_ind > 0 and target_ind < len(cx) - 1:
        dx1 = cx[target_ind + 1] - cx[target_ind]
        dy1 = cy[target_ind + 1] - cy[target_ind]
        dx2 = cx[target_ind] - cx[target_ind - 1]
        dy2 = cy[target_ind] - cy[target_ind - 1]

        angle1 = math.atan2(dy1, dx1)
        angle2 = math.atan2(dy2, dx2)
        
        curvature = abs(angle2 - angle1)
    else:
        curvature = 0  # Assume straight line at the ends of the path
    
    return curvature

def calculate_dynamic_lookahead(state, curvature, base_Lf=4.0):
    # Adjust look-ahead distance based on speed and curvature
    speed_factor = k * state.v_x
    curvature_factor = 1.0 / (1.0 + curvature * 5)  # Adjusted scaling factor for curvature
    Lf = base_Lf + speed_factor * curvature_factor
    return Lf


def dynamic_speed_control(cx, cy, state, target_ind, v_max, a_y_max=9.81):
    curvature = calculate_curvature(cx, cy, target_ind)
    # More conservative speed calculation
    v_target = min(v_max, math.sqrt(a_y_max / (curvature + 0.1))) if curvature > 0 else v_max
    ai = Kp * (v_target - state.v_x)
    return ai, v_target


class TargetCourse:
    def __init__(self, cx, cy):
        self.cx = cx
        self.cy = cy
        self.old_nearest_point_index = None

    def search_target_index(self, state):
        if self.old_nearest_point_index is None:
            dx = [state.x - icx for icx in self.cx]
            dy = [state.y - icy for icy in self.cy]
            d = np.hypot(dx, dy)
            ind = np.argmin(d)
            self.old_nearest_point_index = ind
        else:
            ind = self.old_nearest_point_index
            distance_this_index = state.calc_distance(self.cx[ind], self.cy[ind])
            while True:
                distance_next_index = state.calc_distance(self.cx[ind + 1], self.cy[ind + 1])
                if distance_this_index < distance_next_index:
                    break
                ind = ind + 1 if (ind + 1) < len(self.cx) else ind
                distance_this_index = distance_next_index
            self.old_nearest_point_index = ind

        curvature = calculate_curvature(self.cx, self.cy, ind)  # Call the standalone function
        Lf = calculate_dynamic_lookahead(state, curvature)

        while Lf > state.calc_distance(self.cx[ind], self.cy[ind]):
            if (ind + 1) >= len(self.cx):
                break
            ind += 1

        return ind, Lf

def calculate_cost(state, cx, cy, target_ind, track_width=3.0):
    lane_deviation = state.calc_distance(cx[target_ind], cy[target_ind])
    lane_cost = lane_deviation ** 2

    target_speed = 15.0 / 3.6
    speed_cost = (state.v_x - target_speed) ** 2

    left_edge = np.array([cx[target_ind] - track_width / 2, cy[target_ind]])
    right_edge = np.array([cx[target_ind] + track_width / 2, cy[target_ind]])
    dist_to_left_edge = state.calc_distance(left_edge[0], left_edge[1])
    dist_to_right_edge = state.calc_distance(right_edge[0], right_edge[1])
    safety_cost = max(0, track_width / 2 - min(dist_to_left_edge, dist_to_right_edge)) ** 2

    # Adjusted weights to balance objectives
    total_cost = 1.0 * lane_cost + 0.05 * speed_cost + 5.0 * safety_cost
    return total_cost

def optimize_control(state, target_course, cx, cy, v_max, previous_steering=0):
    best_cost = float('inf')
    best_throttle = 0.0
    best_steering = 0.0
    steering_damping = 0.9  # Damping factor for steering input

    for throttle in np.linspace(0, 1, 10):
        for delta in np.linspace(-np.radians(30), np.radians(30), 10):
            temp_state = State(state.x, state.y, state.yaw, state.v_x, state.v_y, state.r)
            temp_state.update(throttle, delta)
            target_ind, _ = target_course.search_target_index(temp_state)

            cost = calculate_cost(temp_state, cx, cy, target_ind)
            if cost < best_cost:
                best_cost = cost
                best_throttle = throttle
                best_steering = delta

    # Apply damping to the steering input
    best_steering = steering_damping * previous_steering + (1 - steering_damping) * best_steering
    return best_throttle, best_steering

def main():
    csv_filename = 'centerline_track.csv'
    cx, cy = read_csv_points(csv_filename)

    yellow_file = 'yellow_cones_track.csv'
    blue_file = 'blue_cones_track.csv'
    cones = load_and_concatenate_data(yellow_file, blue_file)

    v_max = 17 / 3.6  # [m/s] Max speed
    T = 200.0  # max simulation time

    state = State(x=60.786, y=0, yaw=0.168*math.pi, v_x=1.0, v_y=0.0, r=0.0)

    lastIndex = len(cx) - 1
    time = 0.0
    states = States()
    states.append(time, state)
    target_course = TargetCourse(cx, cy)
    target_ind, _ = target_course.search_target_index(state)

    previous_steering = 0.0

    while T >= time and lastIndex > target_ind:
        throttle, steering = optimize_control(state, target_course, cx, cy, v_max, previous_steering)
        state.update(throttle, steering)
        previous_steering = steering  # Store the previous steering for damping
        time += dt
        states.append(time, state)
        
        if show_animation:
            plt.cla()
            plt.gcf().canvas.mpl_connect(
                'key_release_event',
                lambda event: [exit(0) if event.key == 'escape' else None])
            plt.plot(cx, cy, "tab:red", label="course")
            plt.plot(states.x, states.y, "-k", label="trajectory")
            plt.plot(cx[target_ind], cy[target_ind], "xg", label="target")
            plt.scatter(cones['X'], cones['Y'], c=cones['color'].map({'Y': 'gold', 'B': 'tab:blue'}), s=5)
            plot_car(state.x, state.y, state.yaw, steer=steering)
            plt.axis("equal")
            plt.title("Speed[km/h]:" + str(state.v_x * 3.6)[:4])
            plt.pause(0.001)

    if show_animation:
        plt.cla()
        plt.plot(cx, cy, ".r", label="course")
        plt.plot(states.x, states.y, "-b", label="trajectory")
        plt.legend()
        plt.xlabel("x[m]")
        plt.ylabel("y[m]")
        plt.axis("equal")
        plt.grid(True)

        plt.subplots(1)
        plt.plot(states.t, [iv * 3.6 for iv in states.v_x], "-r")
        plt.xlabel("Time[s]")
        plt.ylabel("Speed[km/h]")
        plt.grid(True)
        plt.show()

if __name__ == '__main__':
    print("Pure pursuit path tracking simulation with dynamic bicycle model start")
    main()