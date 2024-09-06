import numpy as np
import math
import matplotlib.pyplot as plt
import csv
import pandas as pd

# Vehicle parameters
m = 250  # mass of the vehicle (not final)
I_z = 1700  # moment of inertia about the vertical axis
l_f = 0.835  # distance from the center of mass to the front axle
l_r = 0.705  # distance from the center of mass to the rear axle
WB = 1.54  # [m] wheel base of vehicle

# Parameters
k = 0.5  # look forward gain
Lfc = 5  # [m] look-ahead distance
Kp = 0.2  # speed proportional gain
dt = 0.5  # [s] time tick

# Display parameters
show_animation = True

class State:
    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v
        self.rear_x = self.x - ((l_f) * math.cos(self.yaw))
        self.rear_y = self.y - ((l_f) * math.sin(self.yaw))

    def update(self, a, delta):
        self.x += self.v * math.cos(self.yaw) * dt
        self.y += self.v * math.sin(self.yaw) * dt
        self.yaw += self.v / WB * math.tan(delta) * dt
        self.v += a * dt
        self.rear_x = self.x - ((l_f) * math.cos(self.yaw))
        self.rear_y = self.y - ((l_f) * math.sin(self.yaw))

    def calc_distance(self, point_x, point_y):
        dx = self.rear_x - point_x
        dy = self.rear_y - point_y
        return math.hypot(dx, dy)

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

def proportional_control(target, current):
    return Kp * (target - current)

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
class TargetCourse:
    def __init__(self, cx, cy):
        self.cx = cx
        self.cy = cy
        self.old_nearest_point_index = None

    def search_target_index(self, state):
        if self.old_nearest_point_index is None:
            dx = [state.rear_x - icx for icx in self.cx]
            dy = [state.rear_y - icy for icy in self.cy]
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

    alpha = math.atan2(ty - state.rear_y, tx - state.rear_x) - state.yaw
    delta = math.atan2(2.0 * WB * math.sin(alpha) / Lf, 1.0)

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



def dynamic_alpha(state, curvature, a_y_max=9.81):
    lateral_accel = (state.v ** 2) * curvature
    base_alpha = 10.0

    if lateral_accel > a_y_max:
        alpha = base_alpha * 2
    else:
        alpha = base_alpha * (lateral_accel / a_y_max)

    return alpha

def dynamic_speed_control(cx, cy, state, target_ind, v_max, a_y_max=9.81):
    curvature = calculate_curvature(cx, cy, target_ind)
    alpha = dynamic_alpha(state, curvature, a_y_max)
    v_target = max(v_max - alpha * curvature, 5.0 / 3.6)  # Ensure minimum speed
    ai = Kp * (v_target - state.v)
    return ai, v_target

def calculate_dynamic_lookahead(state, curvature, base_Lf=4.0):
    # Increase look-ahead distance with speed, but decrease it with tighter curves
    speed_factor = k * state.v
    curvature_factor = 1.0 / (1.0 + curvature * 10)  # The 10 here is an arbitrary scaling factor
    Lf = base_Lf + speed_factor * curvature_factor
    return Lf

def main():
    csv_filename = 'centerline_track.csv'
    cx, cy = read_csv_points(csv_filename)

    yellow_file = 'yellow_cones_track.csv'
    blue_file = 'blue_cones_track.csv'
    cones = load_and_concatenate_data(yellow_file, blue_file)

    v_max = 17.0 / 3.6  # [m/s] Max speed
    T = 200.0  # max simulation time

    state = State(x=60.786, y=0, yaw=0.168*math.pi, v=0.0)

    lastIndex = len(cx) - 1
    time = 0.0
    states = States()
    states.append(time, state)
    target_course = TargetCourse(cx, cy)
    target_ind, _ = target_course.search_target_index(state)

    while T >= time and lastIndex > target_ind:
        ai, v_target = dynamic_speed_control(cx, cy, state, target_ind, v_max)
        di, target_ind = pure_pursuit_steer_control(state, target_course, target_ind)
        state.update(ai, di)
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
            plot_car(state.x, state.y, state.yaw, steer=di)
            plt.axis("equal")
            plt.title("Speed[km/h]:" + str(state.v * 3.6)[:4])
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
        plt.plot(states.t, [iv * 3.6 for iv in states.v], "-r")
        plt.xlabel("Time[s]")
        plt.ylabel("Speed[km/h]")
        plt.grid(True)
        plt.show()

if __name__ == '__main__':
    print("Pure pursuit path tracking simulation start")
    main()
