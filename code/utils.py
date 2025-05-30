import os
import json
import cv2
import numpy as np
from pathlib import Path
import glob
#from comb.processing.sync import sync_utilities
#from comb import data_file_keys

#VIDEO_SUFFIXES = ('.mp4', '.avi', '.wmv', '.mov')

def show_cropped_frame(frame_rgb, frame_shape, initial_crop):
    """
    Handles drawing the rectangle, displaying the frame, and user interaction for confirming crop.
    
    Parameters:
    - frame_rgb: RGB image frame.
    - frame_shape: Tuple containing frame dimensions.
    - initial_crop: Tuple (y, x, height, width).
    
    Returns:
    - Final crop coordinates (y, x, height, width).
    """
    import matplotlib.pyplot as plt
    frame_height, frame_width, _ = frame_shape
    total_pixels = frame_height * frame_width
    y, x, h, w = initial_crop

    while True:
        # Draw rectangle
        frame_copy = frame_rgb.copy()
        cv2.rectangle(frame_copy, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Red rectangle
        
        # Plot the frame
        plt.imshow(frame_copy)
        plt.title("Frame with Rectangle")
        plt.axis('off')
        plt.show()
        
        # Calculate pixels in the area
        area_pixels = h * w
        area_percentage = (area_pixels / total_pixels) * 100
        
        print(f"Area pixels: {area_pixels}")
        print(f"Area percentage of total frame: {area_percentage:.2f}%")
        
        # Ask for user input
        user_input = input("Does the crop look correct? (y/n): ").strip().lower()
        if user_input == 'y':
            print("Crop confirmed.")
            return (y, x, h, w), frame_copy
        else:
            # Get new crop values
            print("Enter new crop coordinates:")
            y = int(input(f"Enter new y (0 to {frame_height - 1}): "))
            x = int(input(f"Enter new x (0 to {frame_width - 1}): "))
            h = int(input(f"Enter new height (1 to {frame_height - y}): "))
            w = int(input(f"Enter new width (1 to {frame_width - x}): "))

            frame_copy = frame_rgb.copy()
            cv2.rectangle(frame_copy, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Red rectangle
        
            # Plot the frame
            plt.imshow(frame_copy)
            plt.title("Frame with Rectangle")
            plt.axis('off')
            plt.show()


def get_sync_file(video_path: str) -> Path:
    """
    Recursively search for a single '*_sync.h5' file under the given root directory.

    Args:
        video_dir (str): The root directory to search from.

    Returns:
        Path: The path to the found sync file.

    Raises:
        FileNotFoundError: If no matching file is found.
        RuntimeError: If more than one matching file is found.
    """
    root_dir = Path(video_path).parent.parent
    pattern = str(Path(root_dir) / "**" / "*_sync.h5")
    matches = list(glob.glob(pattern, recursive=True))

    if len(matches) == 0:
        raise FileNotFoundError(f"No '*_sync.h5' file found under {root_dir}")
    elif len(matches) > 1:
        raise RuntimeError(f"Multiple '*_sync.h5' files found under {root_dir}: {matches}")
    
    return matches[0]

# def load_camera_json(json_path: str) -> dict:
#     """
#     Load camera metadata from a JSON file.

#     Args:
#         json_path (str): Path to the JSON file.

#     Returns:
#         dict: Metadata dictionary extracted from the 'RecordingReport' field in the JSON file.
#     """
#     with open(json_path, 'r') as file:
#         metadata = json.load(file)
#     return metadata.get('RecordingReport', {})


def extract_camera_label(file_path: str) -> str:
    """
    Extracts the camera label (e.g., "Behavior") from the given file path.

    Args:
        file_path (str): Path to the video file.

    Returns:
        str: The extracted camera label.
    """
    filename = Path(file_path).stem  # Extract the filename without extension
    try: # this is to extract camera label from filename
        parts = filename.split('_')
        # The camera label is typically the second last element
        if len(parts) >= 2:
            return parts[-2]
        else:
            path_parts = Path(file_path).parts
            idx = path_parts.index("behavior_videos")
            return path_parts[idx + 1]
    except: # this is looking for camera name in the folder name
        print(f"Unexpected filename format: {filename}")
        return "unknown"


def load_session_metadata_file(root_dir: str) -> dict:
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


def construct_results_folder(self) -> str:
    """
    Construct a results folder name based on metadata fields.

    Returns:
        str: Folder name for results.
    """
    try:
        return f"{self.data_asset_name}_{self.camera_label}_motion_energy"
    except KeyError as e:
        raise KeyError(f"Missing required metadata field: {e}")


def object_to_dict(obj):
    if hasattr(obj, "__dict__"):
        meta_dict = {key: object_to_dict(value) for key, value in vars(obj).items()}
    elif isinstance(obj, list):
        meta_dict = [object_to_dict(item) for item in obj]
    elif isinstance(obj, dict):
        meta_dict = {key: object_to_dict(value) for key, value in obj.items()}
    else:
        meta_dict = obj

    # Convert Path to str for json serialization
    if isinstance(meta_dict, dict):
        return {k: str(v) if isinstance(v, Path) else v for k, v in meta_dict.items()}
    elif isinstance(meta_dict, list):
        return [str(v) if isinstance(v, Path) else v for v in meta_dict]
    elif isinstance(meta_dict, Path):
        return str(meta_dict)
    else:
        return meta_dict

