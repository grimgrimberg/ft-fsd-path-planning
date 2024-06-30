"""

Path planning simulation with Delaunay Triangulation.

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from scipy.interpolate import interp1d
from matplotlib.animation import FuncAnimation


# Function to load and concatenate data from two CSV files
def load_and_concatenate_data(yellow_file, blue_file):
    yellow_cone_df = pd.read_csv(yellow_file)
    blue_cone_df = pd.read_csv(blue_file)

    concatenated_df = pd.concat([
        pd.concat([yellow_cone_df.iloc[i:i+1].assign(color='Y'),
                   blue_cone_df.iloc[i:i+1].assign(color='B')])
        for i in range(min(len(yellow_cone_df), len(blue_cone_df)))
    ], ignore_index=True)

    return concatenated_df

# Function to perform Delaunay triangulation on a subset of points
def perform_triangulation(points_subset):
    triangulation = Delaunay(points_subset[['x', 'y']])
    return triangulation

# Function to filter out valid simplices (triangles) from the triangulation
def filter_valid_simplices(triangulation, points_subset):
    valid_simplices = []
    for simplex in triangulation.simplices:
        simplex_colors = points_subset.iloc[simplex]['color'].values
        if len(set(simplex_colors)) != 1:
            valid_simplices.append(simplex)
    sorted_valid_simplices = sorted(valid_simplices, key=lambda x: tuple(sorted(x)))
    triangulation.simplices = np.array(sorted_valid_simplices)
    return triangulation

# Function to find midpoints of edges between points of different colors
def find_midpoints(triangulation, points_subset):
    midpoints = []
    for simplex in triangulation.simplices:
        simplex_colors = points_subset.iloc[simplex]['color'].values
        if len(set(simplex_colors)) != 1:
            idx1, idx2, idx3 = simplex
            for idx_pair in [(idx1, idx2), (idx1, idx3), (idx2, idx3)]:
                color1, color2 = points_subset.iloc[idx_pair[0]]['color'], points_subset.iloc[idx_pair[1]]['color']
                if color1 != color2:
                    point1 = points_subset.iloc[idx_pair[0]][['x', 'y']]
                    point2 = points_subset.iloc[idx_pair[1]][['x', 'y']]
                    midpoint = (point1 + point2) / 2
                    midpoints.append(midpoint)
    return pd.DataFrame(midpoints, columns=['x', 'y'])

# Function to interpolate midpoints to smooth the path
def interpolate_midpoints(midpoints):
    interp_x = interp1d(np.arange(len(midpoints)), midpoints['x'], kind='linear')
    interp_y = interp1d(np.arange(len(midpoints)), midpoints['y'], kind='linear')
    num_interpolated_points = 100
    x_interpolated = interp_x(np.linspace(0, len(midpoints) - 1, num_interpolated_points))
    y_interpolated = interp_y(np.linspace(0, len(midpoints) - 1, num_interpolated_points))
    return pd.DataFrame({'x': x_interpolated, 'y': y_interpolated})

# Function to plot the results
def plot_results(ax, df, midpoints_list=None, interpolated_midpoints_list=None, triangulation_list=None):
    ax.scatter(df['x'], df['y'], c=df['color'].map({'Y': 'yellow', 'B': 'blue'}))
    if midpoints_list is not None:
        for midpoints_df in midpoints_list:
            ax.scatter(midpoints_df['x'], midpoints_df['y'], color='red', marker='x')
    if interpolated_midpoints_list is not None:
        for interpolated_midpoints_df in interpolated_midpoints_list:
            ax.plot(interpolated_midpoints_df['x'], interpolated_midpoints_df['y'], color='green')
    if triangulation_list is not None:
        for triangulation in triangulation_list:
            ax.triplot(triangulation.points[:, 0], triangulation.points[:, 1], triangulation.simplices, '-k')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_title('Path Planning Demo')

# Function to update the plot for each frame of the animation
def update(frame, ax, map_df, inter, triangulation_list=None, midpoints_list=None,
           interpolated_midpoints_list=None):
    ax.clear()
    points_subset = map_df.iloc[frame: frame + inter + 2]
    triangulation = perform_triangulation(points_subset)
    filter_triangulation = filter_valid_simplices(triangulation, points_subset)
    midpoints = find_midpoints(filter_triangulation, points_subset)
    interpolated_midpoints = interpolate_midpoints(midpoints)

    if triangulation_list is not None:
        triangulation_list.append(filter_triangulation)
    if midpoints_list is not None:
        midpoints_list.append(midpoints)
    if interpolated_midpoints_list is not None:
        interpolated_midpoints_list.append(interpolated_midpoints)
    plot_results(ax, map_df, midpoints_list, interpolated_midpoints_list, triangulation_list)

# Function to handle key events (in this case, closing the plot when 'escape' is pressed)
def on_key(event):
    if event.key == 'escape':
        plt.close()

# Main function to orchestrate the animation
def main():
    yellow_file = 'yellow_cone_position.csv'
    blue_file = 'blue_cone_position.csv'
    map_df = load_and_concatenate_data(yellow_file, blue_file)

    fig, ax = plt.subplots()
    fig.canvas.manager.full_screen_toggle()  # Toggle full-screen mode

    triangulation_list = []
    midpoints_list = []
    interpolated_midpoints_list = []
    inter = 10  # Number of frames between each interpolated frame

    ani = FuncAnimation(fig, update, fargs=(
        ax, map_df, inter, triangulation_list, midpoints_list, interpolated_midpoints_list),
                        frames=range(0, len(map_df), inter), repeat=False, interval=500)  # Animation object

    fig.canvas.mpl_connect('key_press_event', on_key)  # Connect key press event to the function
    plt.show()  # Show the plot

if __name__ == "__main__":
    main()
