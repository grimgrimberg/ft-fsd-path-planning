# stanley_controller.py
import math
import numpy as np

class StanleyController:
    def __init__(self, k=0.5, k_soft=1e-6):
        """
        Initialize the Stanley Controller.

        Args:
            k: Gain for the lateral error term.
            k_soft: Small constant to avoid division by zero for velocity.
        """
        self.k = k
        self.k_soft = k_soft

    def control(self, state, trajectory, pind):
        """
        Calculate steering angle using Stanley control.

        Args:
            state: State object representing the vehicle state.
            trajectory: Target trajectory.
            pind: Previous target index.

        Returns:
            Steering angle and updated target index.
        """
        ind = self.find_target_index(state, trajectory)

        # Calculate heading error
        tx, ty = trajectory.cx[ind], trajectory.cy[ind]
        target_yaw = math.atan2(ty - state.front_y, tx - state.front_x)
        heading_error = target_yaw - state.yaw
        heading_error = self.normalize_angle(heading_error)

        # Calculate cross track error (use front axle position)
        dx = state.front_x - tx
        dy = state.front_y - ty
        cross_track_error = dy * math.cos(target_yaw) - dx * math.sin(target_yaw)

        # Stanley control law
        delta = heading_error + math.atan2(self.k * cross_track_error, state.v + self.k_soft)
        return delta, ind

    def find_target_index(self, state, trajectory):
        """
        Find the target index on the trajectory for Stanley control.

        Args:
            state: State object representing the vehicle state.
            trajectory: Target trajectory.

        Returns:
            Updated target index.
        """
        # Search nearest point index based on front axle position
        dx = [state.front_x - icx for icx in trajectory.cx]
        dy = [state.front_y - icy for icy in trajectory.cy]
        d = np.hypot(dx, dy)
        ind = np.argmin(d)

        # Refine target index
        distance_this_index = state.calc_distance(trajectory.cx[ind], trajectory.cy[ind], use_front=True)
        while True:
            if (ind + 1) >= len(trajectory.cx):
                break
            distance_next_index = state.calc_distance(trajectory.cx[ind + 1], trajectory.cy[ind + 1], use_front=True)
            if distance_this_index < distance_next_index:
                break
            ind += 1
            distance_this_index = distance_next_index

        return ind

    def normalize_angle(self, angle):
        """
        Normalize an angle to the range [-pi, pi].

        Args:
            angle: Angle to normalize.

        Returns:
            Normalized angle.
        """
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle
