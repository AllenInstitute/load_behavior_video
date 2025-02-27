import os
import json
import cv2
import numpy as np
from tqdm import tqdm
from pathlib import Path
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

VIDEO_SUFFIXES = ('.mp4', '.avi', '.wmv', '.mov')

def load_camera_json(json_path: str) -> dict:
    """
    Load camera metadata from a JSON file.

    Args:
        json_path (str): Path to the JSON file.

    Returns:
        dict: Metadata dictionary extracted from the 'RecordingReport' field in the JSON file.
    """
    with open(json_path, 'r') as file:
        metadata = json.load(file)
    return metadata.get('RecordingReport', {})


def load_metadata_file(root_dir: str) -> dict:
    """
    Load the metadata file from the specified directory.

    Args:
        root_dir (str): Directory where the metadata file is located.

    Returns:
        dict: Loaded metadata dictionary if found, otherwise None.
    """
    metadata_file = 'metadata.nd.json'
    file_path = os.path.join(root_dir, metadata_file)
    
    if os.path.exists(file_path):
        print(f"Found metadata file at: {file_path}")
        with open(file_path, 'r') as file:
            metadata = json.load(file)
        return metadata
    
    print(f"Metadata file {metadata_file} not found in {root_dir}")
    return None

def get_video_paths(directory: Path ) --> list[str]:
    return [
        str(p) for p in DATA_PATH.rglob(VIDEO_FILE_GLOB_PATTERN)
    ]


def get_video_paths_old(directory: Path = Path(/root/capsule/data), subselect: str = None) -> list:
    """
    Retrieve video file paths from the specified directory, optionally filtering by a subdirectory.

    Args:
        directory (str): The directory to search for video files.
        subselect (str): Optional subdirectory name to filter the search.

    Returns:
        list: A list of paths to video files.
    """
    video_paths = []
    trial_videos_added = set()  # Track unique trial directories

    for root, _, files in os.walk(directory):
        if subselect and subselect not in root:
            continue  # Skip directories that don't match the subselect
        
        trial_video_added = False
        for file in tqdm(files, desc=f"Searching for videos in {root}"):
            if file.lower().endswith(('.mp4', '.avi')):
                full_path = os.path.join(root, file)
                
                # Ensure only one trial video per directory
                if 'trial' in file.lower():
                    if not trial_video_added and root not in trial_videos_added:
                        video_paths.append(full_path)
                        trial_video_added = True
                        trial_videos_added.add(root)
                else:
                    print(f"Found video file: {full_path}")
                    video_paths.append(full_path)

    return video_paths


def process_chunk(start: int, chunk_size: int, frame_shape: tuple, video_path: str) -> np.ndarray:
    """
    Process a chunk of video frames, converting each to grayscale and resizing.

    Args:
        start (int): Starting frame index.
        chunk_size (int): Number of frames to load in the chunk.
        frame_shape (tuple): Target frame shape as (height, width).
        video_path (str): Path to the video file.

    Returns:
        np.ndarray: A stack of processed grayscale frames in the specified chunk.

    Raises:
        ValueError: If no frames are returned for the specified chunk.
    """
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)  # Set starting frame position
    chunk = []

    for _ in range(chunk_size):
        ret, frame = cap.read()
        if not ret:
            break  # Exit loop if there are no more frames
        
        # Convert to grayscale and resize to target shape
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.resize(gray_frame, (frame_shape[1], frame_shape[0]))
        chunk.append(gray_frame)

    cap.release()
    
    if not chunk:
        raise ValueError(f"No frames found in chunk starting at {start}. Check the video length.")
    
    return np.stack(chunk)


def get_results_folder() -> str:
    """
    Get the results folder path.

    Returns:
        str: Path to the results folder.
    """
    return Path('/results/')


def object_to_dict(obj):
    if hasattr(obj, "__dict__"):
        return {key: object_to_dict(value) for key, value in vars(obj).items()}
    elif isinstance(obj, list):
        return [object_to_dict(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: object_to_dict(value) for key, value in obj.items()}
    else:
        return obj


def get_zarr_path(self, path_to: str = 'gray_frames') -> str:
    """
    Construct the path for Zarr storage based on mouse and camera metadata.

    Args:
        path_to (str): Type of frames to be saved ('gray_frames' or 'motion_energy_frames').

    Returns:
        str: Full path to the Zarr storage file.
    """
    zarr_folder = f"{self.mouse_id}_{self.data_asset_name}_{self.camera_label}_frames"
    zarr_path = os.path.join(get_results_folder(), zarr_folder)
    
    # Create directory if it doesn't exist
    os.makedirs(zarr_path, exist_ok=True)
    
    filename = 'processed_frames_zarr' if path_to == 'gray_frames' else 'motion_energy_frames.zarr'
    return os.path.join(zarr_path, filename)





## EXTRA 
def create_metadata_dataframe(video_path: str) -> pd.DataFrame:
    """
    Loads metadata from a file and converts it into a Pandas DataFrame.

    Args:
        video_path (str): Path to the video file.

    Returns:
        pd.DataFrame: A DataFrame containing session type and data asset ID.

    Raises:
        ValueError: If metadata is missing required fields.
        FileNotFoundError: If the metadata file cannot be loaded.
    """
    try:
        # Load metadata
        metadata = load_metadata_file(video_path.split('behavior-videos')[0])

        # Extract relevant fields
        session_type = metadata.get('session', {}).get('session_type')
        data_asset_id = metadata.get('_id')
        data_asset_name = metadata.get('name')

        # Ensure required fields are present
        if session_type is None or data_asset_id is None:
            raise ValueError("Missing required fields: 'session_type' or '_id' in metadata.")

        # Create DataFrame
        df = pd.DataFrame({'Session Type': [session_type], 'Data Asset ID': [data_asset_id], 'Data Asset Name': [data_asset_name]})

        logger.info(f"Created DataFrame with session type: {session_type} and data asset name: {data_asset_name}")

        return df

    except FileNotFoundError as e:
        logger.error(f"Metadata file not found for video path: {video_path}")
        raise e
    except Exception as e:
        logger.error(f"Error loading metadata: {e}")
        raise e


