"""

Path planning simulation with Delaunay Triangulation.

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from scipy.interpolate import interp1d


# Function to load and concatenate data from yellow and blue cone files
def load_and_concatenate_data(yellow_file, blue_file):
    yellow_cone_df = pd.read_csv(yellow_file)
    blue_cone_df = pd.read_csv(blue_file)

    concatenated_df = pd.concat([
        pd.concat([yellow_cone_df.iloc[i:i+1].assign(color='Y'),
                   blue_cone_df.iloc[i:i+1].assign(color='B')])
        for i in range(min(len(yellow_cone_df), len(blue_cone_df)))
    ], ignore_index=True)

    return concatenated_df

# Function to plot the initial data
def plot_data(df):
    plt.scatter(df['x'], df['y'], c=df['color'].map({'Y': 'yellow', 'B': 'blue'}), s=10)
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.title('MAP')
    plt.show()

# Function to perform Delaunay triangulation on a subset of points
def perform_triangulation(points_subset):
    triangulation = Delaunay(points_subset[['x', 'y']])
    return triangulation

# Function to filter out valid simplices from the triangulation
def filter_valid_simplices(triangulation, points_subset):
    valid_simplices = []
    for simplex in triangulation.simplices:
        simplex_colors = points_subset.iloc[simplex]['color'].values
        if len(set(simplex_colors)) != 1:
            valid_simplices.append(simplex)
    sorted_valid_simplices = sorted(valid_simplices, key=lambda x: tuple(sorted(x)))
    triangulation.simplices = np.array(sorted_valid_simplices)
    return triangulation

# Function to find midpoints of valid simplices
def find_midpoints(triangulation, points_subset):
    midpoints = []
    for simplex in triangulation.simplices:
        simplex_colors = points_subset.iloc[simplex]['color'].values
        if len(set(simplex_colors)) != 1:
            idx1, idx2, idx3 = simplex
            for idx_pair in [(idx1, idx3), (idx2, idx3), (idx1, idx2)]:
                color1, color2 = points_subset.iloc[idx_pair[0]]['color'],points_subset.iloc[idx_pair[1]]['color']
                if color1 != color2:
                    point1 = points_subset.iloc[idx_pair[0]][['x', 'y']]
                    point2 = points_subset.iloc[idx_pair[1]][['x', 'y']]
                    midpoint = (point1 + point2) / 2
                    midpoints.append(midpoint)
    return pd.DataFrame(midpoints, columns=['x', 'y'])

# Function to interpolate midpoints
def interpolate_midpoints(midpoints):
    interp_x = interp1d(np.arange(len(midpoints)), midpoints['x'], kind='linear')
    interp_y = interp1d(np.arange(len(midpoints)), midpoints['y'], kind='linear')
    num_interpolated_points = 100
    x_interpolated = interp_x(np.linspace(0, len(midpoints) - 1, num_interpolated_points))
    y_interpolated = interp_y(np.linspace(0, len(midpoints) - 1, num_interpolated_points))
    return pd.DataFrame({'x': x_interpolated, 'y': y_interpolated})

# Function to plot results
def plot_results(df, midpoints_list=None, interpolated_midpoints_list=None, triangulation_list=None):
    for df in df:
        plt.scatter(df['x'], df['y'], c=df['color'].map({'Y': 'yellow', 'B': 'blue'}))
    if midpoints_list is not None:
        for midpoints_df in midpoints_list:
            plt.scatter(midpoints_df['x'], midpoints_df['y'], color='red', marker='x')
    if interpolated_midpoints_list is not None:
        for interpolated_midpoints_df in interpolated_midpoints_list:
            plt.plot(interpolated_midpoints_df['x'], interpolated_midpoints_df['y'], color='green')
    if triangulation_list is not None:
        for triangulation in triangulation_list:
            plt.triplot(triangulation.points[:, 0], triangulation.points[:, 1], triangulation.simplices, '-k')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.title('Path Planning Demo')
    plt.waitforbuttonpress()
    if plt.get_current_fig_manager().toolbar:
        plt.connect('key_press_event', lambda event: plt.close() if event.key == ' ' else None)
    else:
        plt.connect('key_press_event', lambda event: plt.close() if event.key == ' ' else None)
        plt.show()


# Main function
def main():
    yellow_file = 'yellow_cone_position.csv'
    blue_file = 'blue_cone_position.csv'
    map_df = load_and_concatenate_data(yellow_file, blue_file)
    plot_data(map_df)

    # Variables to store results
    midpoints_total = []
    interpolated_midpoints_total = []
    triangulation_total = []
    points_total = []
    inter = 10  # Interval for selecting subsets of points

    # Loop through the data in chunks
    for i in range(0, map_df.shape[0] - inter, inter):
        points_subset = map_df.iloc[i: i + inter + 2]
        triangulation = perform_triangulation(points_subset)
        filter_triangulation = filter_valid_simplices(triangulation, points_subset)
        midpoints = find_midpoints(filter_triangulation, points_subset)
        interpolated_midpoints = interpolate_midpoints(midpoints)

        # Store results
        points_total.append(points_subset)
        triangulation_total.append(filter_triangulation)
        midpoints_total.append(midpoints)
        interpolated_midpoints_total.append(interpolated_midpoints)

        # Plot the final results
        plot_results(points_total, midpoints_list=midpoints_total[:-1], interpolated_midpoints_list=interpolated_midpoints_total[:-1], triangulation_list=triangulation_total[:-1])
        plot_results(points_total, midpoints_list=midpoints_total[:-1], interpolated_midpoints_list=interpolated_midpoints_total[:-1], triangulation_list=triangulation_total)
        plot_results(points_total, midpoints_list=midpoints_total, interpolated_midpoints_list=interpolated_midpoints_total[:-1], triangulation_list=triangulation_total)
        plot_results(points_total, midpoints_list=midpoints_total, interpolated_midpoints_list=interpolated_midpoints_total, triangulation_list=triangulation_total)

if __name__ == "__main__":
    main()

