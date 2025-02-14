import cv2
import numpy as np
import dask.array as da
import dask
import zarr
import utils  # local module for metadata and video processing utilities
import json

DATA_PATH = '/root/capsule/data/'
OUTPUT_PATH = '/root/capsule/results/'

class VideoLoader:
    """
    Class for loading a video, extracting frames, managing metadata, and saving frames to Zarr format.

    Attributes:
        video_path (str): Path to the video file.
        fps (int): Frames per second of the video.
        cap (cv2.VideoCapture): OpenCV video capture object.
        width (int): Width of the video frames.
        height (int): Height of the video frames.
        total_frames (int): Total number of frames in the video.
        video_info (dict): Metadata from a JSON file corresponding to the video.
        mouse_id (str): Subject ID extracted from metadata.
        rig_id (str): Rig ID extracted from metadata.
        project (str): Project name extracted from metadata.
        timestamps (numpy.ndarray): Array of timestamps for each frame.
        frames_zarr_path (str): Path to save frames in Zarr format.
        chunk_size (int): Number of frames to process in each chunk.
    """

    def __init__(self, video_path: str, fps: int = None):
        """Initialize the VideoLoader with a video file path and optional FPS."""
        self.video_path = video_path
        self.cap = cv2.VideoCapture(self.video_path)
        self.fps = fps or int(self.cap.get(cv2.CAP_PROP_FPS))
        self.chunk_size = 100
        self._get_video_details()

    def _get_video_details(self):
        """Extracts and stores basic video details such as width, height, and total frames."""
        if not self.cap.isOpened():
            raise ValueError(f"Error opening video file at {self.video_path}.")
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def _get_metadata(self):
    #  Need to add session information
        """Loads metadata from associated JSON and metadata files."""
        json_path = self.video_path.replace('mp4', 'json')
        self.video_info = utils.load_camera_json(json_path)
        self.camera_label = self.video_info['CameraLabel']
        metadata = utils.load_metadata_file(self.video_path.split('behavior-videos')[0])
        self.data_asset_id = metadata['_id']
        self.mouse_id = metadata['subject']['subject_id']
        self.rig_id = metadata['session']['rig_id']
        self.session_type = metadata['session']['session_type']
        self.project = metadata['data_description']['platform']['abbreviation']

    def _get_timestamps(self):
        """Generates timestamps for each frame using video metadata and FPS."""
        duration_str = self.video_info['Duration']
        hours, minutes, seconds = map(int, duration_str.split(':'))
        total_seconds = hours * 3600 + minutes * 60 + seconds
        
        time_interval = 1 / self.fps
        self.timestamps = list(np.arange(0, total_seconds, time_interval)[:self.video_info['FramesRecorded']])

    def _load_video(self, start_sec: float = None, stop_sec: float = None, gray: bool = True):
        """
        Loads frames within specified time range, converts them to grayscale if specified, and saves as Zarr.
        
        Args:
            start_sec (float, optional): Start time in seconds for loading frames.
            stop_sec (float, optional): End time in seconds for loading frames.
            gray (bool): Convert frames to grayscale if True.
        
        Returns:
            self: Instance of VideoLoader with loaded frames.
        """
        start_frame = int(start_sec * self.fps) if start_sec else 0
        stop_frame = int(stop_sec * self.fps) if stop_sec else self.total_frames
        frame_shape = (self.height, self.width)
        
        chunks = [
            dask.delayed(utils.process_chunk)(start=start, chunk_size=self.chunk_size,
                                              frame_shape=frame_shape, video_path=self.video_path)
            for start in range(start_frame, stop_frame, self.chunk_size)
        ]
        
        # Create Dask array from chunks and save to Zarr
        dask_chunks = [
            da.from_delayed(chunk, shape=(min(self.chunk_size, stop_frame - start), *frame_shape), dtype='f4')
            for start, chunk in zip(range(start_frame, stop_frame, self.chunk_size), chunks)
        ]
        dask_array = da.concatenate(dask_chunks, axis=0)

        # save files
        self.frames_zarr_path = utils.get_zarr_path(self, path_to='gray_frames')
        zarr_store = zarr.DirectoryStore(self.frames_zarr_path)

        # Create the Zarr group at the root (root_group) and save the array in the 'data' path (subgroup)
        root_group = zarr.group(store=zarr_store, overwrite=True)
        dask_array.to_zarr(zarr_store, component='data', overwrite=True)

        del self.cap # remove cap to be able to save json
        meta_dict = utils.object_to_dict(self)
        root_group.attrs['metadata'] = json.dumps(meta_dict)

        print(f'Saved frames and metadata in {self.frames_zarr_path}')

        return self


    def _process(self, start_sec: float = None, stop_sec: float = None, gray: bool = True, save_dir: str = OUTPUT_PATH):
        """
        Processes the video by loading frames, generating timestamps, and saving them in Zarr format.
        
        Args:
            start_sec (float, optional): Start time in seconds for frame loading.
            stop_sec (float, optional): Stop time in seconds for frame loading.
            gray (bool): Whether to convert frames to grayscale.
            save_dir (str): Directory to save processed results.
        """
        if not hasattr(self, 'video_info'):
            self._get_metadata()
        

        if not hasattr(self, 'timestamps'):
            self._get_timestamps()
        
        self._load_video(start_sec=start_sec, stop_sec=stop_sec, gray=gray)


        
