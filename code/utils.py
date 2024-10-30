import os
from tqdm import tqdm
import json
import cv2


def load_camera_json(json_path: str) -> dict:
    """
    Loads the camera metadata from a JSON file.

    Args:
        json_path (str): Path to the JSON file.

    Returns:
        dict: Metadata dictionary from the 'RecordingReport' field in the JSON file.
    """
    with open(json_path, 'r') as file:
        metadata = json.load(file)
    return metadata['RecordingReport']


def load_metadata_file(root_dir: str) -> dict:
    """
    Loads the metadata file from the root directory.

    Args:
        root_dir (str): The directory where the metadata file is expected.

    Returns:
        dict: Loaded metadata dictionary or None if the file is not found.
    """
    filenames = os.listdir(root_dir)
    if 'metadata.nd.json' in filenames:
        file_path = os.path.join(root_dir, 'metadata.nd.json')
        print(f"Found file at: {file_path}")
        with open(file_path, 'r') as file:
            metadata = json.load(file)
        return metadata
    
    print("File not found")
    return None


def get_video_paths(directory: str = '/root/capsule/data', subselect: str = None) -> list:
    """
    Retrieves video file paths from the specified directory, optionally filtering by a subdirectory.

    Args:
        directory (str): The directory to search for video files.
        subselect (str): An optional subdirectory name to filter the video search.

    Returns:
        list: A list of paths to video files.
    """
    video_paths = []
    trial_videos_added = set()

    for root, _, files in os.walk(directory):
        # Skip non-matching directories if subselect is specified
        if subselect and subselect not in root:
            continue

        trial_video_added = False
        for file in tqdm(files, desc=f"Processing files in {root}"):
            if file.lower().endswith(('.mp4', '.avi')):
                full_path = os.path.join(root, file)
                # Check if the file is a 'trial' video
                if 'trial' in file.lower():
                    if not trial_video_added and root not in trial_videos_added:
                        video_paths.append(full_path)
                        trial_video_added = True
                        trial_videos_added.add(root)
                else:
                    print(f'Found video file: {full_path}')
                    video_paths.append(full_path)

    return video_paths

def process_chunk(self, start):
    cap = self.cap
    start_frame, stop_frame = self.start_frame, self.stop_frame
    chunk_size = self.chunk_size
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)  # Set starting frame
    chunk = []
    for i in range(chunk_size):
        current_frame = start + i
        if current_frame >= stop_frame:
            break
        ret, frame = cap.read()
        if not ret:
            break  # Stop if no frame is returned (end of video)
        if self.gray:
            # Convert to grayscale
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        chunk.append(gray_frame)
    cap.release()
    if len(chunk) == 0:
        raise ValueError(f"No frames found in chunk starting at {start}. Check the video length.")
    return np.stack(chunk)

def get_results_folder():
    return '/root/capsule/results'

def get_zarr_paths(self, path_to = 'gray_frames')
    zarr_folder = self.mouse_id + '_' +self.camera_label + '_' + self.data_asset_id
    if path_to == 'gray_frames':
        filename = 'processed_frames.zarr'
        filepath = os.path.join(get_results_folder(), zarr_folder, filename)
    elif path_to = 'motion_energy_frames': 
        filename = 'motion_energy_frames.zarr'
        filepath = os.path.join(get_results_folder(), zarr_folder, filename)
    return filepath
    
    

#_________________________________________________________
# old code
# def load_frame_parallel(args):
#     video_path, frame_idx, gray = args
#     cap = cv2.VideoCapture(video_path)
#     cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
#     ret, frame = cap.read()
#     if ret and gray:
#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     cap.release()
#     return frame if ret else None

# def parse_video_path(video_path):

#     # List of known project names and video types
#     project_names = ['multiplane', 'ecephys', 'single-plane', 'unknown']
#     video_types = ['face','eye', 'bottom', 'side', 'unknown']
    
#     # Extract the base filename
#     filename = os.path.basename(path)
    
#     # Extract the relevant part of the directory path
#     parts = path.split('/')
    
#     # Extract the part containing the project and mouse ID (assuming it's the 5th part)
#     project_mouse_id = parts[4].split('_')
    
#     # Determine project name
#     project = next((p for p in project_names if p in project_mouse_id[0]), 'unknown')
    
#     # Extract mouse ID (second part of the project_mouse_id)
#     mouse_id = project_mouse_id[1]
    
#     # Check for specific behavior video type patterns
#     if '.behavior.' in lower_filename or 'behavior_' in lower_filename:
#         video_type = 'behavior'
#     else:
#         # Extract video type from the filename (case-insensitive) if no special case for 'behavior'
#         video_type = next((v for v in video_types if v in lower_filename), 'unknown')

#     return project, mouse_id, video_type
