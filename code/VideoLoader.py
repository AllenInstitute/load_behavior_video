import cv2
import random
import numpy as np
import os
import pickle
from tqdm import tqdm
from multiprocessing import Pool
import utils  # local module

DATA_PATH = '/root/capsule/data/'
OUTPUT_PATH = '/root/capsule/results/'

class VideoLoader:
    """
    Class for loading a video, extracting frames, and managing metadata.

    Attributes:
        video_path (str): Path to the video file.
        fps (int): Frames per second for the video.
        cap (cv2.VideoCapture): OpenCV video capture object.
        width (int): Width of the video frames.
        height (int): Height of the video frames.
        total_frames (int): Total number of frames in the video.
        video_info (dict): Metadata from a JSON file corresponding to the video.
        mouse_id (str): Subject ID extracted from metadata.
        rig_id (str): Rig ID extracted from metadata.
        project (str): Project name extracted from metadata.
        timestamps (numpy.ndarray): Array of timestamps for each frame.
        loaded_frames (dict): Dictionary of loaded frames and related data.
    """

    def __init__(self, video_path: str, fps: int = None):
        """Initialize the VideoLoader with a video file and optional FPS."""
        self.video_path = video_path
        self.root_dir = video_path.split('behavior-videos')[0]
        self.cap = cv2.VideoCapture(self.video_path)
        self.chunk_size = 100

        # Set FPS either from input or video metadata
        self.fps = fps if fps is not None else int(self.cap.get(cv2.CAP_PROP_FPS))
        
        # Extract video details
        self._get_details()

    def _get_details(self):
        """Extracts and stores video details such as width, height, and total frames."""
        if not self.cap.isOpened():
            raise ValueError("Error opening video file. Please check the file path.")
        print(f'Using FPS: {self.fps}')
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        return self

    def _get_metadata(self):
        """Loads metadata from a corresponding JSON file and the root directory."""
        json_path = self.video_path.replace('mp4', 'json')
        self.video_info = utils.load_camera_json(json_path)  # Saving full dict
        self.camera_label = self.video_info['CameraLabel']

        metadata = utils.load_metadata_file(self.root_dir)
        self.data_asset_id = metadata['_id']
        self.mouse_id = metadata['subject']['subject_id']
        self.rig_id = metadata['session']['rig_id']
        self.project = metadata['data_description']['platform']['abbreviation']

        return self

    def _get_timestamps(self):
        """Generates timestamps for each frame based on video metadata and FPS."""
        fps = self.fps
        time_start = self.video_info['TimeStart']
        duration = self.video_info['Duration']
        frames_recorded = self.video_info['FramesRecorded']

        # Convert duration to total seconds
        h, m, s = map(int, duration.split(':'))
        total_duration_seconds = h * 3600 + m * 60 + s

        # Calculate the time interval between frames
        time_interval = 1 / fps

        # Create an array of timestamps in seconds
        timestamps = np.arange(0, total_duration_seconds, time_interval)
        self.timestamps = timestamps[:frames_recorded]
        
        return self

    def _load_video(self, start_sec: float = None, stop_sec: float = None, 
                    start_frame: int = None, stop_frame: int = None,
                    gray: bool = True):
        """
        Loads frames from the video within specified frame or time ranges.

        Args:
            start_sec (float): Start time in seconds to load frames from.
            stop_sec (float): Stop time in seconds to load frames until.
            start_frame (int): Start frame number to load.
            stop_frame (int): Stop frame number to load.
            gray (bool): Whether to convert frames to grayscale.

        Returns:
            VideoLoader: The instance of VideoLoader with loaded frames.
        """
        frames = []
        
        # Convert seconds to frames if provided
        if start_sec is not None:
            start_frame = int(start_sec * self.fps)
        if stop_sec is not None:
            stop_frame = int(stop_sec * self.fps)
        
        # If no specific frames, load full video
        if start_frame is None and stop_frame is None:
            start_frame, stop_frame = 1, self.total_frames

        self.'gray': gray,
        self.'start_frame': start_frame,
        self.'stop_frame': stop_frame}

        chunks = []
        for start in range(start_frame, stop_frame, chunk_size):
            delayed_chunk = dask.delayed(process_chunk)(start, chunk_size, frame_shape, start_frame, stop_frame)
            chunks.append(delayed_chunk)
            
        self.frames_zarr_path = ''

        dask_chunks = [
        da.from_delayed(chunk, shape=(min(chunk_size, stop_frame - start), *frame_shape), dtype='f4') 
        for start, chunk in zip(range(start_frame, stop_frame, chunk_size), chunks)]
            dask_array = da.concatenate(dask_chunks, axis=0)



        return self

    def _save(self, save_dir: str, suffix: str = ''):
        """Saves the loaded frames and metadata to a specified directory."""
        if not hasattr(self, 'mouse_id') or not hasattr(self, 'project'):
            self._get_metadata()

        fullpath = os.path.join(save_dir, f'{self.mouse_id}')
        if os.path.exists(fullpath) is False:
            os.makedirs(fullpath, exist_ok=True)

        if not suffix:
            print('suffix is empty, consider adding a date or other unique character to the filename')
        
        videoname = os.path.basename(self.video_path).replace('.mp4', '').replace('.avi', '')
        filename = f"{self.mouse_id}_{self.project}_{videoname}_{suffix}.pkl"
        fullfile = os.path.join(fullpath, filename)
        
        #cv2 object cannot be pickled
        del self.cap

        # Save the object using pickle
        with open(fullfile, 'wb') as f:
            pickle.dump(self, f)
        
        # Get the size of the saved file
        file_size = os.path.getsize(fullfile)
        
        # Convert file size to kilobytes (KB)
        file_size_kb = file_size / 1024
        
        print(f"Object saved as {fullfile}")
        print(f"File size: {file_size_kb:.2f} KB")

    def _process(self, start_sec: float = None, stop_sec: float = None,
                 gray: bool = True, save_dir: str = OUTPUT_PATH, suffix: str = ''):
        """
        Process the video by loading frames, generating timestamps, and saving the results.

        Args:
            start_sec (float): Start time in seconds for frame loading.
            stop_sec (float): Stop time in seconds for frame loading.
            gray (bool): Whether to convert frames to grayscale.
            save_dir (str): Directory to save the processed results.
            suffix (str): Optional identifying info for filename.
        """
        # Ensure metadata is loaded
        if not hasattr(self, 'video_info'):
            self._get_metadata()

        # Load video frames
        self._load_video(start_sec=start_sec, stop_sec=stop_sec, gray=gray)

        # Generate timestamps if they don't exist
        if not hasattr(self, 'timestamps'):
            self._get_timestamps()

        # Save processed frames and metadata
        self._save(save_dir, suffix=suffix)
