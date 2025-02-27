import os
from tqdm import tqdm
import utils
import time  # Added for timing
from VideoLoader import VideoLoader  # Correct import
from pathlib import Path

DATA_PATH = Path("/data/")#'/root/capsule/data'
tag = 'face'
def run():
    def analyze_and_save_videos(data_folder):
        # Ensure results directory exists
        
        # Get video paths from utils
        video_paths = utils.get_video_paths(directory=data_folder)

        start_time = time.time()  # Start the timer

        # Process each video
        for video_path in video_paths:
            # Initialize VideoLoader object with the video path
            if tag in video_path.lower():
                print(f'Processing {video_path}')
                loader = VideoLoader(video_path=video_path)
            
                # Process the video (load frames, metadata, timestamps, and save results)
                loader._process(gray=True, start_sec = None, stop_sec = None)
            else:
                print(f'processing only {tag} videos for now')

        end_time = time.time()  # End the timer
        duration = end_time - start_time
        print(f"Total time taken: {duration:.2f} seconds")

    # Example usage with subselect
    analyze_and_save_videos(data_folder=DATA_PATH)

if __name__ == "__main__":
    run()
