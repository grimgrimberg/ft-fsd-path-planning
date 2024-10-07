# dynamic_bicycle_model.py
import numpy as np

class DynamicBicycleModel:
    def __init__(self, m, I_z, l_f, l_r, c_f, c_r, mu):
        """
        Initialize the dynamic bicycle model.

        Args:
            m: Vehicle mass [kg]
            I_z: Yaw moment of inertia [kg*m^2]
            l_f: Distance from the center of mass to the front axle [m]
            l_r: Distance from the center of mass to the rear axle [m]
            c_f: Cornering stiffness of the front tires [N/rad]
            c_r: Cornering stiffness of the rear tires [N/rad]
            mu: Friction coefficient between tires and road
        """
        self.m = m
        self.I_z = I_z
        self.l_f = l_f
        self.l_r = l_r
        self.c_f = c_f
        self.c_r = c_r
        self.mu = mu

    def state_equations(self, x, u, dt):
        """
        Compute the next state of the vehicle given the current state and control inputs.

        Args:
            x: Current state [x_position, y_position, yaw, velocity, yaw_rate, slip_angle]
            u: Control input [steering_angle, acceleration]
            dt: Time step [s]

        Returns:
            x_next: Next state of the vehicle
        """
        x_pos, y_pos, yaw, v, r, beta = x
        delta, a = u

        # Vehicle parameters
        m = self.m
        I_z = self.I_z
        l_f = self.l_f
        l_r = self.l_r
        c_f = self.c_f
        c_r = self.c_r
        mu = self.mu

        # Adding a small epsilon to avoid division by zero
        epsilon = 1e-5
        v = max(v, epsilon)

        # Lateral forces (considering the friction limit using mu)
        F_yf = -c_f * (beta + (l_f * r) / v - delta)
        F_yr = -c_r * (beta - (l_r * r) / v)

        # Limiting the lateral forces by friction (Pacejka-like limitation)
        F_yf = np.clip(F_yf, -mu * m * 9.81, mu * m * 9.81)
        F_yr = np.clip(F_yr, -mu * m * 9.81, mu * m * 9.81)

        # State update equations
        x_pos_next = x_pos + v * np.cos(yaw + beta) * dt
        y_pos_next = y_pos + v * np.sin(yaw + beta) * dt
        yaw_next = yaw + r * dt
        v_next = v + a * dt
        r_next = r + (l_f * F_yf - l_r * F_yr) / I_z * dt
        beta_next = beta + (F_yr / (m * v) - r) * dt

        x_next = [x_pos_next, y_pos_next, yaw_next, v_next, r_next, beta_next]
        return x_next

    def predict_next_state(self, current_state, control_input, dt):
        """
        Predict the next state using the dynamic bicycle model.

        Args:
            current_state: Current state of the vehicle [x, y, yaw, velocity, yaw_rate, slip_angle]
            control_input: Control inputs [steering_angle, acceleration]
            dt: Time step [s]

        Returns:
            next_state: Predicted next state of the vehicle
        """
        next_state = self.state_equations(current_state, control_input, dt)
        return next_state

# Example usage
if __name__ == "__main__":
    # Vehicle parameters
    m = 1500  # mass [kg]
    I_z = 3000  # yaw moment of inertia [kg*m^2]
    l_f = 1.2  # distance from CoG to front axle [m]
    l_r = 1.6  # distance from CoG to rear axle [m]
    c_f = 15000  # cornering stiffness front [N/rad]
    c_r = 15000  # cornering stiffness rear [N/rad]
    mu = 0.9  # friction coefficient

    # Initialize the model
    model = DynamicBicycleModel(m, I_z, l_f, l_r, c_f, c_r, mu)

    # Initial state [x, y, yaw, velocity, yaw_rate, slip_angle]
    current_state = [0, 0, 0, 5, 0, 0]

    # Control input [steering_angle, acceleration]
    control_input = [0.05, 1.0]

    # Time step
    dt = 0.1

    # Predict the next state
    next_state = model.predict_next_state(current_state, control_input, dt)
    print("Next state:", next_state)