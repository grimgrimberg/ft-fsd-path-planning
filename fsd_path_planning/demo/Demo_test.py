# # from __future__ import annotations

# # import json
# # from pathlib import Path
# # from typing import Optional

# # import numpy as np

# # from fsd_path_planning import ConeTypes, MissionTypes, PathPlanner
# # from fsd_path_planning.utils.utils import Timer

# # try:
# #     import matplotlib.animation
# #     import matplotlib.pyplot as plt
# #     import typer
# # except ImportError:
# #     print(
# #         "\n\nThis demo requires matplotlib and typer to be installed. You can install"
# #         " them with by using the [demo] extra.\n\n"
# #     )
# #     raise

# # try:
# #     from tqdm import tqdm
# # except ImportError:
# #     print("You can get a progress bar by installing tqdm: pip install tqdm")
# #     tqdm = lambda x, total=None: x


# # try:
# #     app = typer.Typer(pretty_exceptions_enable=False)
# # except TypeError:
# #     app = typer.Typer()


# # def select_mission_by_filename(filename: str) -> MissionTypes:
# #     is_skidpad = "skidpad" in filename

# #     if is_skidpad:
# #         print(
# #             'The filename contains "skidpad", so we assume that the mission is skidpad.'
# #         )
# #         return MissionTypes.skidpad

# #     return MissionTypes.trackdrive


# # def get_filename(data_path: Path | None) -> Path:
# #     if data_path is None:
# #         data_path = Path(__file__).parent / "fss_19_4_laps.json"

# #     return data_path


# # @app.command()
# # def main(
# #     data_path: Optional[Path] = typer.Option(None, "--data-path", "-i"),
# #     data_rate: float = 10,
# #     remove_color_info: bool = False,
# #     show_runtime_histogram: bool = False,
# #     output_path: Optional[Path] = typer.Option(None, "--output-path", "-o"),
# # ) -> None:
# #     data_path = get_filename(data_path)

# #     mission = select_mission_by_filename(data_path.name)

# #     planner = PathPlanner(mission)

# #     positions, directions, cone_observations = load_data_json(
# #         data_path, remove_color_info=remove_color_info
# #     )

# #     if not numba_cache_files_exist():
# #         print(
# #             """
# # It looks like this is the first time you are running this demo. It will take around a 
# # minute to compile the numba functions. If you want to estimate the runtime of the
# # planner, you should run the demo one more time after it is finished.
# # """
# #         )

# #     # run planner once to "warm up" the JIT compiler / load all cached jit functions
# #     try:
# #         planner.calculate_path_in_global_frame(
# #             cone_observations[0], positions[0], directions[0]
# #         )
# #     except Exception:
# #         print("Error during warmup")
# #         raise

# #     results = []
# #     timer = Timer(noprint=True)

# #     for i, (position, direction, cones) in tqdm(
# #         enumerate(zip(positions, directions, cone_observations)),
# #         total=len(positions),
# #         desc="Calculating paths",
# #     ):
# #         try:
# #             with timer:
# #                 out = planner.calculate_path_in_global_frame(
# #                     cones,
# #                     position,
# #                     direction,
# #                     return_intermediate_results=True,
# #                 )
# #         except KeyboardInterrupt:
# #             print(f"Interrupted by user on frame {i}")
# #             break
# #         except Exception:
# #             print(f"Error at frame {i}")
# #             raise
# #         results.append(out)

# #         if timer.intervals[-1] > 0.1:
# #             print(f"Frame {i} took {timer.intervals[-1]:.4f} seconds")

# #     if show_runtime_histogram:
# #         # skip the first few frames, because they include "warmup time"
# #         plt.hist(timer.intervals[10:])
# #         plt.show()

# #     fig, ax = plt.subplots(figsize=(10, 10))
# #     ax.set_aspect("equal")
# #     # plot animation
# #     frames = []
# #     for i in tqdm(range(len(results)), desc="Generating animation"):
# #         co = cone_observations[i]
# #         (yellow_cones,) = plt.plot(*co[ConeTypes.YELLOW].T, "yo")
# #         (blue_cones,) = plt.plot(*co[ConeTypes.BLUE].T, "bo")
# #         (unknown_cones,) = plt.plot(*co[ConeTypes.UNKNOWN].T, "ko")
# #         (orange_small_cones,) = plt.plot(*co[ConeTypes.ORANGE_SMALL].T, "o", c="orange")
# #         (orange_big_cones,) = plt.plot(
# #             *co[ConeTypes.ORANGE_BIG].T, "o", c="darkorange", markersize=10
# #         )
# #         (yellow_cones_sorted,) = plt.plot(*results[i][2].T, "y-")
# #         (blue_cones_sorted,) = plt.plot(*results[i][1].T, "b-")
# #         (path,) = plt.plot(*results[i][0][:, 1:3].T, "r-")
# #         (position,) = plt.plot(*positions[i], "go")
# #         (direction,) = plt.plot(
# #             *np.array([positions[i], positions[i] + directions[i]]).T, "g-"
# #         )
# #         title = plt.text(
# #             0.5,
# #             1.01,
# #             f"Frame {i}",
# #             ha="center",
# #             va="bottom",
# #             transform=ax.transAxes,
# #             fontsize="large",
# #         )
# #         frames.append(
# #             [
# #                 yellow_cones,
# #                 blue_cones,
# #                 unknown_cones,
# #                 orange_small_cones,
# #                 orange_big_cones,
# #                 yellow_cones_sorted,
# #                 blue_cones_sorted,
# #                 path,
# #                 position,
# #                 direction,
# #                 title,
# #             ]
# #         )

# #     anim = matplotlib.animation.ArtistAnimation(
# #         fig, frames, interval=1 / data_rate * 1000, blit=True, repeat_delay=1000
# #     )

# #     if output_path is not None:
# #         absolute_path_str = str(output_path.absolute())
# #         typer.echo(f"Saving animation to {absolute_path_str}")
# #         anim.save(absolute_path_str, fps=data_rate)

# #     plt.show()


# # def numba_cache_files_exist() -> bool:
# #     package_file = Path(__file__).parent.parent
# #     try:
# #         next(package_file.glob("**/*.nbc"))
# #     except StopIteration:
# #         return False

# #     return True


# # def load_data_json(
# #     data_path: Path,
# #     remove_color_info: bool = False,
# # ) -> tuple[np.ndarray, np.ndarray, list[list[np.ndarray]]]:
# #     # extract data
# #     data = json.loads(data_path.read_text())[:]

# #     positions = np.array([d["car_position"] for d in data])
# #     directions = np.array([d["car_direction"] for d in data])
# #     cone_observations = [
# #         [np.array(c).reshape(-1, 2) for c in d["slam_cones"]] for d in data
# #     ]

# #     if remove_color_info:
# #         cones_observations_all_unknown = []
# #         for cones in cone_observations:
# #             new_observation = [np.zeros((0, 2)) for _ in ConeTypes]
# #             new_observation[ConeTypes.UNKNOWN] = np.row_stack(
# #                 [c.reshape(-1, 2) for c in cones]
# #             )
# #             cones_observations_all_unknown.append(new_observation)

# #         cone_observations = cones_observations_all_unknown.copy()

# #     return positions, directions, cone_observations


# # if __name__ == "__main__":
# #     app()
# import casadi as ca
# import numpy as np

# # Define state variables
# phi = ca.SX.sym('phi')  # Roll angle
# delta = ca.SX.sym('delta')  # Steer angle
# Vx = ca.SX.sym('Vx')  # Longitudinal velocity
# Vy = ca.SX.sym('Vy')  # Lateral velocity
# omega = ca.SX.sym('omega')  # Yaw rate
# psi = ca.SX.sym('psi')  # Pitch angle
# z = ca.SX.sym('z')  # Vertical displacement
# theta = ca.SX.sym('theta')  # Roll rate
# z_dot = ca.SX.sym('z_dot')  # Vertical velocity
# psi_dot = ca.SX.sym('psi_dot')  # Pitch rate

# # Define control inputs
# u_steer = ca.SX.sym('u_steer')  # Steering input
# u_drive = ca.SX.sym('u_drive')  # Driving/braking input

# # Define parameters (placeholders for actual values)
# m = ca.SX.sym('m')  # Mass
# Ixx = ca.SX.sym('Ixx')  # Moment of inertia around x-axis
# Iyy = ca.SX.sym('Iyy')  # Moment of inertia around y-axis
# Izz = ca.SX.sym('Izz')  # Moment of inertia around z-axis
# g = 9.81  # Gravity

# # Dynamics equations (placeholders for detailed modeling)
# x_dot = ca.vertcat(
#     Vx * ca.cos(psi) - Vy * ca.sin(psi),  # dx/dt
#     Vx * ca.sin(psi) + Vy * ca.cos(psi),  # dy/dt
#     omega,  # dpsi/dt
#     u_drive / m,  # dVx/dt, assuming direct influence of u_drive
#     u_steer - Vx * omega,  # dVy/dt, simplified lateral dynamics
#     theta,  # dphi/dt
#     z_dot,  # dz/dt
#     psi_dot  # dpsi_dot/dt
# )

# # Placeholder function for dynamics, to be replaced with actual equations
# dynamics = ca.Function('dynamics', [phi, delta, Vx, Vy, omega, psi, z, theta, z_dot, psi_dot, u_steer, u_drive], [x_dot])

# # Time horizon and discretization
# T = 10.0  # Total simulation time
# N = 100  # Number of steps
# dt = T/N  # Time step

# # Initial conditions and control inputs
# x0 = [0, 0, 5, 0, 0, 0, 0, 0, 0, 0]  # Example initial state
# U = [0.1, 0]  # Example control inputs (steer, drive)

# # Integrating the dynamics
# X = [x0]
# for i in range(N):
#     # Integrate for one time step
#     x_next = X[-1] + dt * dynamics(X[-1] + U)
#     X.append(x_next)


import numpy as np
import matplotlib.pyplot as plt

# Step 1: Define waypoints for a square path
waypoints = np.array([[0, 0], [10, 0], [10, 10], [0, 10]])

# Step 2: Interpolate between waypoints to generate a smooth path
# For simplicity, this example uses linear interpolation.
from scipy.interpolate import interp1d

# Assume equal distance between waypoints and normalize it
distance = np.linspace(0, 1, len(waypoints))

# Interpolate for both x and y
interpolation_function_x = interp1d(distance, waypoints[:, 0], kind='linear')
interpolation_function_y = interp1d(distance, waypoints[:, 1], kind='linear')

# Generate a finer distance vector
fine_distance = np.linspace(0, 1, 100)

# Generate the smooth path
smooth_path_x = interpolation_function_x(fine_distance)
smooth_path_y = interpolation_function_y(fine_distance)

# Step 3: Calculate heading (psi) for each point on the path
# For a simple linear path, psi can be calculated as the arctangent of the derivative of the path.
# Here we'll use a simple finite difference method to approximate this.
psi = np.arctan2(np.diff(smooth_path_y, prepend=smooth_path_y[0]), np.diff(smooth_path_x, prepend=smooth_path_x[0]))

# Now, psi, smooth_path_x, and smooth_path_y can be used as reference trajectory inputs to the MPC

# Plot the original waypoints and the interpolated path for visualization
plt.plot(waypoints[:, 0], waypoints[:, 1], 'o', label='Waypoints')
plt.plot(smooth_path_x, smooth_path_y, label='Interpolated Path')
plt.quiver(smooth_path_x, smooth_path_y, np.cos(psi), np.sin(psi), scale=20, label='Heading')
plt.legend()
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Reference Trajectory with Heading')
plt.axis('equal')
plt.grid(True)
plt.show()