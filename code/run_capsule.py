
from tqdm import tqdm
import utils
import time  # Added for timing
from VideoLoader import VideoLoader  # Correct import
from pathlib import Path

DATA_PATH = Path("/data")
tag = 'Nose'
VIDEO_FILE_GLOB_PATTERN = "*"+tag+"*.mp4"
crop_region = (100, 100, 250, 370) #y,x,height,width

# Get video paths 
video_paths = [str(p) for p in DATA_PATH.rglob(VIDEO_FILE_GLOB_PATTERN)]
print(f'Found {len(video_paths)} {tag} raw videos') 

def run():
    # Process each video
    for video_path in tqdm(video_paths):
        start_time = time.time()  # Start the timer
        print(f'Processing {video_path}')
        sync_path = utils.get_sync_file(video_path)
        loader = VideoLoader(video_path=video_path, sync_path = sync_path, crop_region = crop_region)
        loader.process_and_save_video()

        end_time = time.time()  # End the timer
        duration = end_time - start_time
        print(f"Total time taken in load behavior video: {duration:.2f} seconds")

if __name__ == "__main__":
    run()
