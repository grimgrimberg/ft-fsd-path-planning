from __future__ import annotations
import os

import json
from pathlib import Path
from typing import Optional
import pandas as pd

import numpy as np

from fsd_path_planning import ConeTypes, MissionTypes, PathPlanner
from fsd_path_planning.utils.utils import Timer

try:
    import matplotlib.animation
    import matplotlib.pyplot as plt
    import typer
except ImportError:
    print(
        "\n\nThis demo requires matplotlib and typer to be installed. You can install"
        " them with by using the [demo] extra.\n\n"
    )
    raise

try:
    from tqdm import tqdm
except ImportError:
    print("You can get a progress bar by installing tqdm: pip install tqdm")
    tqdm = lambda x, total=None: x


try:
    app = typer.Typer(pretty_exceptions_enable=False)
except TypeError:
    app = typer.Typer()

def output2csv(out,results):
    dfs=[]
    # Example values - replace these with your actual data parameters
    total_length_of_array = len(results)  # Total length of your data array
    num_measurements_per_timestep = 40 # Number of measurements in each timestep

# Calculate the total number of timesteps
    # total_timesteps = (total_length_of_array - 1) // num_measurements_per_timestep
    total_timesteps = (total_length_of_array)
# Define your starting timestamp and frequency interval
    start_time = '00:00:00'
    freq = '1s'  # 'T' for minute. Use 'S' for second, 'H' for hour, etc., as needed

# Generate the timestep data
# Pandas date_range can create a range of datetime values at a specified frequency
    # timesteps = pd.date_range(start=start_time, periods=total_timesteps, freq=freq)
    # if not out :
    #     return
    # print(out[0])
    # print("this is the full results",results[0:])
    # print("this is results")
    # print(results)
    # print(results[0][:])
    # print(type(results.))
    # results_arr = np.array(results,dtype='object')
    # print('this is the shape of results_arr ',np.shape.results_arr)
    # print("this is results as array " , results_arr[:,0])
    # print(a)
    # print("this is out")
    # print(out)
    for index, element in enumerate(results):
        df2 = pd.DataFrame(element[0],columns=['spline','x','y','kappa'])
        # filename = f'measurement_{index}.csv'
        dfs.append(df2)
        combined_df = pd.concat(dfs, ignore_index=True)
        # timestep_df = pd.date_range(start=start_time, periods=total_timesteps, freq=freq)
        # Add a 'Timestamp' column to the DataFrame
        # If the number of rows in 'df' doesn't match the length of 'timesteps', adjust accordingly
        # df['Timestamp'] = pd.NaT  # Initialize column with 'Not a Time' (NaT)
        # df.loc[:len(timesteps)-1, 'Timestamp'] = timesteps
        # combined_df = combined_df[['Timestamp','spline','x','y','kappa']]

    # Export the combined DataFrame to a CSV file
    combined_df.to_csv('combined_measurements_output.csv', index=False,columns=['spline','x','y','kappa'])
    
    # Export the DataFrame to a CSV file without the index
    # df2.to_csv(filename, index=False)
    
    df=pd.DataFrame(out[0],columns=['spline','x','y','kappa'])
    # df2 = pd.DataFrame(results_arr[:,0],columns=['spline','x','y','kappa'])
    csv_filename = 'trakoutput.csv'
    csv_filename2 = 'fulltrackout.csv'
    df.to_csv(csv_filename, index=False)
    df2.to_csv(csv_filename2, index=False)


    print("CSV file created using Method 2")
    # file_name = 'out.csv'
    #     # Attempt to get the directory of the current script
    # #directory = os.path.dirname(os.path.abspath(__file__))
    #     # Fallback to the current working directory if __file__ is not defined
    # directory = os.getcwd()
    
    # # Construct the full file path
    # full_path = os.path.join(directory, file_name)
    
    # # Use numpy.savetxt to write the array to a CSV file
    # np.savetxt(full_path, out, delimiter=',', fmt='%s')
    # print(f"File saved to: {full_path}")

    

def select_mission_by_filename(filename: str) -> MissionTypes:
    is_skidpad = "skidpad" in filename

    if is_skidpad:
        print(
            'The filename contains "skidpad", so we assume that the mission is skidpad.'
        )
        return MissionTypes.skidpad

    return MissionTypes.trackdrive


def get_filename(data_path: Path | None) -> Path:
    if data_path is None:
        data_path = Path(__file__).parent / "fsg_19_2_laps.json"

    return data_path


@app.command()
def main(
    data_path: Optional[Path] = typer.Option(None, "--data-path", "-i"),
    data_rate: float = 10,
    remove_color_info: bool = False,
    show_runtime_histogram: bool = False,
    output_path: Optional[Path] = typer.Option(None, "--output-path", "-o"),
) -> None:
    """
    A function to generate a main animation based on given parameters and data. The function takes in various parameters such as data_path, data_rate, remove_color_info, show_runtime_histogram, and output_path. It then performs several operations such as loading data, warming up the JIT compiler, running the planner, calculating paths, and generating an animation. The function also saves the animation to the specified output path.
    """
    data_path = get_filename(data_path)

    mission = select_mission_by_filename(data_path.name)

    planner = PathPlanner(mission)

    positions, directions, cone_observations = load_data_json(
        data_path, remove_color_info=remove_color_info
    )

    if not numba_cache_files_exist():
        print(
            """
It looks like this is the first time you are running this demo. It will take around a 
minute to compile the numba functions. If you want to estimate the runtime of the
planner, you should run the demo one more time after it is finished.
"""
        )

    # run planner once to "warm up" the JIT compiler / load all cached jit functions
    try:
        planner.calculate_path_in_global_frame(
            cone_observations[0], positions[0], directions[0]
        )
    except Exception:
        print("Error during warmup")
        raise

    results = []
    timer = Timer(noprint=True)

    for i, (position, direction, cones) in tqdm(
        enumerate(zip(positions, directions, cone_observations)),
        total=len(positions),
        desc="Calculating paths",
    ):
        try:
            with timer:
                out = planner.calculate_path_in_global_frame(
                    cones,
                    position,
                    direction,
                    return_intermediate_results=True,
                )
        except KeyboardInterrupt:
            print(f"Interrupted by user on frame {i}")
            break
        except Exception:
            print(f"Error at frame {i}")
            raise
        results.append(out)

        if timer.intervals[-1] > 0.1:
            print(f"Frame {i} took {timer.intervals[-1]:.4f} seconds")

    if show_runtime_histogram:
        # skip the first few frames, because they include "warmup time"
        plt.hist(timer.intervals[10:])
        plt.show()

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect("equal")
    # plot animation
    frames = []
    for i in tqdm(range(len(results)), desc="Generating animation"):
        co = cone_observations[i]
        (yellow_cones,) = plt.plot(*co[ConeTypes.YELLOW].T, "yo")
        (blue_cones,) = plt.plot(*co[ConeTypes.BLUE].T, "bo")
        (unknown_cones,) = plt.plot(*co[ConeTypes.UNKNOWN].T, "ko")
        (orange_small_cones,) = plt.plot(*co[ConeTypes.ORANGE_SMALL].T, "o", c="orange")
        (orange_big_cones,) = plt.plot(
            *co[ConeTypes.ORANGE_BIG].T, "o", c="darkorange", markersize=10
        )
        (yellow_cones_sorted,) = plt.plot(*results[i][2].T, "y-")
        (blue_cones_sorted,) = plt.plot(*results[i][1].T, "b-")
        (path,) = plt.plot(*results[i][0][:, 1:3].T, "r-")
        (position,) = plt.plot(*positions[i], "go")
        (direction,) = plt.plot(
            *np.array([positions[i], positions[i] + directions[i]]).T, "g-"
        )
        title = plt.text(
            0.5,
            1.01,
            f"Frame {i}",
            ha="center",
            va="bottom",
            transform=ax.transAxes,
            fontsize="large",
        )
        frames.append(
            [
                yellow_cones,
                blue_cones,
                unknown_cones,
                orange_small_cones,
                orange_big_cones,
                yellow_cones_sorted,
                blue_cones_sorted,
                path,
                position,
                direction,
                title,
            ]
        )

    anim = matplotlib.animation.ArtistAnimation(
        fig, frames, interval=1 / data_rate * 1000, blit=True, repeat_delay=1000
    )

    if output_path is not None:
        absolute_path_str = str(output_path.absolute())
        typer.echo(f"Saving animation to {absolute_path_str}")
        anim.save(absolute_path_str, fps=data_rate)

    plt.show()
    output2csv(out,results)
    


def numba_cache_files_exist() -> bool:
    """
    Check if any '.nbc' files exist in the package directory.
    Returns True if files exist, False otherwise.
    """
    package_file = Path(__file__).parent.parent
    try:
        next(package_file.glob("**/*.nbc"))
    except StopIteration:
        return False

    return True


def load_data_json(
    data_path: Path,
    remove_color_info: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[list[np.ndarray]]]:
    """
    Load data from a JSON file and return the positions, directions, and cone observations.

    Args:
        data_path (Path): The path to the JSON data file.
        remove_color_info (bool, optional): Whether to remove color information. Defaults to False.

    Returns:
        tuple[np.ndarray, np.ndarray, list[list[np.ndarray]]]: A tuple containing the positions (np.ndarray), 
        directions (np.ndarray), and cone observations (list[list[np.ndarray]]).
    """
    # extract data
    data = json.loads(data_path.read_text())[:]

    positions = np.array([d["car_position"] for d in data])
    directions = np.array([d["car_direction"] for d in data])
    cone_observations = [
        [np.array(c).reshape(-1, 2) for c in d["slam_cones"]] for d in data
    ]

    if remove_color_info:
        cones_observations_all_unknown = []
        for cones in cone_observations:
            new_observation = [np.zeros((0, 2)) for _ in ConeTypes]
            new_observation[ConeTypes.UNKNOWN] = np.row_stack(
                [c.reshape(-1, 2) for c in cones]
            )
            cones_observations_all_unknown.append(new_observation)

        cone_observations = cones_observations_all_unknown.copy()

    return positions, directions, cone_observations


if __name__ == "__main__":
    app()

