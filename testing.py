import numpy as np
import matplotlib.pyplot as plt
from fsd_path_planning import PathPlanner, MissionTypes, ConeTypes

def generate_oval_track(num_points=50, track_length=100, track_width=20, track_curve_radius=30):
    """Generate an oval track with specified parameters."""
    # Calculate the number of points for straight and curved segments
    num_straight = num_points // 4
    num_curved = num_points // 4
    
    # Straight section lengths
    straight_length = track_length / 2 - track_curve_radius * np.pi / 2

    # Generate straight sections
    straight_left = np.array([[x, -track_width / 2] for x in np.linspace(0, straight_length, num=num_straight)])
    straight_right = np.array([[x, track_width / 2] for x in np.linspace(0, straight_length, num=num_straight)])
    
    # Generate curved sections
    theta = np.linspace(-np.pi / 2, np.pi / 2, num=num_curved)
    curve_left = np.column_stack((track_curve_radius * np.cos(theta) + straight_length, track_curve_radius * np.sin(theta) - track_width / 2))
    curve_right = np.column_stack((track_curve_radius * np.cos(theta) + straight_length, track_curve_radius * np.sin(theta) + track_width / 2))
    
    # Assemble the left and right sides of the track
    left_side = np.vstack((straight_left, curve_left, np.flipud(straight_left) + [track_length / 2, 0], np.flipud(curve_left)))
    right_side = np.vstack((straight_right, curve_right, np.flipud(straight_right) + [track_length / 2, 0], np.flipud(curve_right)))
    
    # Adjust to center the track around the origin
    left_side -= [track_length / 4, 0]
    right_side -= [track_length / 4, 0]

    return left_side, right_side

def load_data():
    """Load track data for testing with the path planner."""
    left_cones, right_cones = generate_oval_track()
    global_cones = [np.array([]),  # UNKNOWN cones
                    right_cones,   # RIGHT/YELLOW cones
                    left_cones,    # LEFT/BLUE cones
                    np.array([]),  # START_FINISH_AREA/ORANGE_SMALL cones, empty for simplicity
                    np.array([[0, 0]])]  # START_FINISH_LINE/ORANGE_BIG cones, simplified as a single point
    car_position = np.array([0, 0])  # Example starting position
    car_direction = np.array([1.0, 0.0])  # Example starting direction, pointing right
    return global_cones, car_position, car_direction

# Plotting function remains the same
def visualize_path_with_cones(path, global_cones):
    """Visualize the calculated path along with the cones on the track."""
    plt.figure(figsize=(12, 6))
    colors = ['gray', 'yellow', 'blue', 'orange', 'red']
    labels = ['Unknown', 'Right/Yellow', 'Left/Blue', 'Start/Finish Area', 'Start/Finish Line']
    for i, cones in enumerate(global_cones):
        if len(cones) > 0:
            plt.scatter(cones[:, 0], cones[:, 1], c=colors[i], label=labels[i])
    plt.plot(path[:, 1], path[:, 2], 'g--', label='Calculated Path')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.legend()
    plt.axis('equal')
    plt.title('Path Planning on Oval Track')
    plt.grid(True)
    plt.show()
# Main execution
if __name__ == "__main__":
    try:
        path_planner = PathPlanner(MissionTypes.trackdrive)
        global_cones, car_position, car_direction = load_data()

        # Validate the data before passing it to the path planner
        assert global_cones is not None and len(global_cones) > 0, "Global cones data cannot be empty."
        assert car_position.size == 2, "Car position must be a 2D point."
        assert car_direction.size == 2, "Car direction must be a 2D vector."

        path = path_planner.calculate_path_in_global_frame(global_cones, car_position, car_direction)
        visualize_path_with_cones(path, global_cones)
    except ZeroDivisionError:
        print("Caught division by zero error in path planning calculation.")
    except AssertionError as e:
        print(f"Data validation error: {e}")

# Integrate with the path planner and visualize

# Main execution
if __name__ == "__main__":
    try:
        path_planner = PathPlanner(MissionTypes.trackdrive)
        global_cones, car_position, car_direction = load_data()

        # Validate the data before passing it to the path planner
        assert global_cones is not None and len(global_cones) > 0, "Global cones data cannot be empty."
        assert car_position.size == 2, "Car position must be a 2D point."
        assert car_direction.size == 2, "Car direction must be a 2D vector."

        path = path_planner.calculate_path_in_global_frame(global_cones, car_position, car_direction)
        visualize_path_with_cones(path, global_cones)
    except ZeroDivisionError:
        print("Caught division by zero error in path planning calculation.")
    except AssertionError as e:
        print(f"Data validation error: {e}")

