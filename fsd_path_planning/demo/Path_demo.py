from __future__ import annotations
from fsd_path_planning import PathPlanner, MissionTypes, ConeTypes

import os

import json
from pathlib import Path
from typing import Optional
import pandas as pd
import typer
import numpy as np

from fsd_path_planning import ConeTypes, MissionTypes, PathPlanner
from fsd_path_planning.utils.utils import Timer

from typing import List

def flatten_list(nested_list: List[List[np.ndarray]]) -> List[np.ndarray]:
    """
    Flatten a nested list of NumPy arrays.

    Parameters:
    nested_list (List[List[np.ndarray]]): A nested list of NumPy arrays.

    Returns:
    List[np.ndarray]: A flattened list of NumPy arrays.
    """
    return [item for sublist in nested_list for item in sublist]
def get_filename(data_path: Path | None) -> Path:
    if data_path is None:
        data_path = Path(__file__).parent / "fsg_19_2_laps.json"

    return data_path


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
    data_path = get_filename(data_path)
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


def select_mission_by_filename(filename: str) -> MissionTypes:
    is_skidpad = "skidpad" in filename

    if is_skidpad:
        print(
            'The filename contains "skidpad", so we assume that the mission is skidpad.'
        )
        return MissionTypes.skidpad

    return MissionTypes.trackdrive

def main(
    data_path: Optional[Path] = typer.Option(None, "--data-path", "-i"),
    data_rate: float = 10,
    remove_color_info: bool = False,
    show_runtime_histogram: bool = False,
    output_path: Optional[Path] = typer.Option(None, "--output-path", "-o"),
) -> None:

    data_path = get_filename(data_path)
    #mission = select_mission_by_filename(data_path.name)
    path_planner = PathPlanner(MissionTypes.trackdrive)
    # you have to load/get the data, this is just an example

    car_position, car_direction, global_cones = load_data_json(None,remove_color_info=remove_color_info)

    # tqdm = lambda x, total=None: x
    # global_cones is a sequence that contains 5 numpy arrays with shape (N, 2),
    # where N is the number of cones of that type
    results = []
    # ConeTypes is an enum that contains the following values:
    # ConeTypes.UNKNOWN which maps to index 0
    # ConeTypes.RIGHT/ConeTypes.YELLOW which maps to index 1
    # ConeTypes.LEFT/ConeTypes.BLUE which maps to index 2
    # ConeTypes.START_FINISH_AREA/ConeTypes.ORANGE_SMALL which maps to index 3
    # ConeTypes.START_FINISH_LINE/ConeTypes.ORANGE_BIG which maps to index 4

    # car_position is a 2D numpy array with shape (2,)
    # car_direction is a 2D numpy array with shape (2,) representing the car's direction vector
    # car_direction can also be a float representing the car's direction in radians
    # timer = Timer(noprint=True)
    # print(global_cones)
    # for i, (position, direction, cones) in tqdm(
    #         enumerate(zip(car_position, car_direction, global_cones)),
    #         total=len(car_position),
    #         desc="Calculating paths",
    # ):
    #     try:
    #         with timer:
    #             out = path_planner.calculate_path_in_global_frame(
    #                 cones,
    #                 position,
    #                 direction,
    #                 return_intermediate_results=True,
    #             )
    #     except KeyboardInterrupt:
    #         print(f"Interrupted by user on frame {i}")
    #         break
    #     except Exception:
    #         print(f"Error at frame {i}")
    #         raise
    #     results.append(out)
    cones = flatten_list(global_cones)

    path = path_planner.calculate_path_in_global_frame(cones, car_position, car_direction)

    # path is a Mx4 numpy array, where M is the number of points in the path
    # the columns represent the spline parameter (distance along path), x, y and path curvature
    print(cones)
    return


if __name__ == "__main__":
    main()
