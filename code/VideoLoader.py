import cv2
import numpy as np
import utils  # local module for metadata and video processing utilities
import json
import os
from pathlib import Path


RESULTS = Path("/results")
check_crop = False #will ask for input

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

    def __init__(self, video_path: Path, sync_path: Path, crop_region: tuple, fps: int = None):
        """Initialize the VideoLoader with a video file path and optional FPS."""
        self.video_path = video_path
        self.video_name = Path(video_path).stem
        self.sync_path = sync_path
        self.fps = fps # a hack for videos where metadata json file or video for some reason contains wrong fps
        self.crop_region = crop_region
        self._get_video_details()


    def _get_video_details(self):
        """Extracts and stores basic video details such as width, height, and total frames."""
        cap = cv2.VideoCapture(self.video_path)
        if self.fps is None:
            self.fps = int(cap.get(cv2.CAP_PROP_FPS))

        self.og_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.og_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))-1 #first frame is meta


    def _get_metadata(self):
    #  Need to add session information
        """Loads metadata from associated JSON and metadata files."""
        metadata = utils.load_session_metadata_file(self.video_path.split('behavior-videos')[0])
        self.camera_label = utils.extract_camera_label(self.video_path)
        self.data_asset_id = metadata['_id']
        self.data_asset_name = metadata['name']
        self.mouse_id = metadata['subject']['subject_id']
        self.rig_id = metadata['session']['rig_id']
        self.session_type = metadata['session']['session_type']
        self.project = metadata['data_description']['platform']['abbreviation']

    
    def _check_crop(self, frame_number = 100):
        """
        Loads the video, extracts the 100th frame, and calls the crop_frame function.
        
        Parameters:
        - video_path: Path to the MP4 video file.
        - initial_crop: Initial crop coordinates (y, x, height, width).
        """
        cap = cv2.VideoCapture(self.video_path)
        
        if not cap.isOpened():
            print("Error opening video file.")
            return
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        
        if not ret:
            print(f"Error reading frame {frame_number}.")
            cap.release()
            return
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_shape = frame.shape
        
        final_crop, example_frame = utils.show_cropped_frame(frame_rgb, frame_shape, self.crop_region)

        self.crop_region = final_crop
        self.example_frame = example_frame
        print(f"Final crop coordinates: {final_crop}")
        cap.release()

    def _process_video(self):
        """
        Loads the video, drops the first frame (metadata), converts frames to grayscale,
        applies cropping, and saves the processed frames in MP4 format.

        Parameters:
        - output_path: Path to save the processed video (default: 'output.mp4').
        """
        # Prepare for writing output
        y, x, h, w = self.crop_region

        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video file: {self.video_path}")

        # Read and drop the first frame (metadata)
        ret, _ = cap.read()
        if not ret:
            raise RuntimeError("Failed to read the first frame (metadata).")

        results_folder = utils.construct_results_folder(self)
        results_path = Path(RESULTS, results_folder)
        os.makedirs(results_path, exist_ok=True)
        filename = self.video_name + "_processed.mp4"
        output_path = Path(results_path, filename)
        print(output_path)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 format
        out = cv2.VideoWriter(output_path, fourcc, self.fps, (w, h), isColor=False)  # Grayscale frames

        processed_frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break  # End of video
            
            # Convert to grayscale
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Apply crop
            cropped_frame = gray_frame[y:y+h, x:x+w]
            
            # Write frame
            out.write(cropped_frame)

            processed_frame_count += 1

        cap.release()
        out.release()

        print(f"Processed {processed_frame_count} frames (excluding metadata). Saved to {output_path}")

    def _save(self):
        meta_dict = utils.object_to_dict(self)
        results_folder = utils.construct_results_folder(self)
        results_path = Path(RESULTS, results_folder)
        filename = self.video_name + "_metadata.json"
        metadata_path =  Path(results_path, filename)
        with metadata_path.open('w') as f:
            json.dump(meta_dict, f, indent=4)

    def process_and_save_video(self):
        self._get_metadata()
        if check_crop:
            self._check_crop()
        self._process_video()
        self._save()

# timestamps will be processed with COMB repo, apart from video QC and motion energy analysis.
    
class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
