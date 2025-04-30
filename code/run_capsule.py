import os
from tqdm import tqdm
import utils
import time  # Added for timing
from VideoLoader import VideoLoader  # Correct import
from pathlib import Path

DATA_PATH = Path("/data")
tag = 'Face'
VIDEO_FILE_GLOB_PATTERN = "*"+tag+"*.mp4"

# Get video paths 
video_paths = [str(p) for p in DATA_PATH.rglob(VIDEO_FILE_GLOB_PATTERN)]
Print('Found {len(video_paths)} {tag} videos') 

def run():
    # Process each video
    for video_path in video_paths:
        start_time = time.time()  # Start the timer
        print(f'Processing {video_path}')
        loader = VideoLoader(video_path=video_path)
        loader.process_and_save_video()

        end_time = time.time()  # End the timer
        duration = end_time - start_time
        print(f"Total time taken: {duration:.2f} seconds")

    # Example usage with subselect
    analyze_and_save_videos(DATA_PATH=DATA_PATH)

if __name__ == "__main__":
    run()
