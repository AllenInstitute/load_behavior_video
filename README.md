# Video Analysis Pipeline  

This repository provides the first step in behavior video QC pipeline to process and analyze videos using the `VideoLoader` class. The script loads video frames, extracts metadata and timestamps, and saves video frames and metadata in as a zarr group to a specified output directory.

## Features  
- Processes only videos containing a specified tag (default: `"face"`).  
- Loads video frames and extracts metadata.  
- Supports grayscale processing.  
- Saves results in an output directory.  
- Tracks processing time.  

## Prerequisites  

Ensure you have the following dependencies installed:  
```bash
pip install tqdm
```
Additionally, make sure utils.py and VideoLoader.py are available in your project.

## Parameters

| Parameter      | Description                                    | Default Value           |
|--------------|--------------------------------|-------------------------|
| `DATA_PATH`  | Path to the directory containing videos. | `/root/capsule/data` |
| `OUTPUT_PATH` | Path to save processed video results. | `/root/capsule/results` |
| `tag`        | Filter for video files containing this tag. | `'face'` |
| `subselect`  | Optional filter for selecting specific videos. | `'multiplane'` |

### How It Works  

1. The script scans the `DATA_PATH` directory for videos matching the `subselect` filter.  
2. It initializes a `VideoLoader` object for each video containing the `tag` in its filename.  
3. It processes the video in grayscale and saves results to `OUTPUT_PATH`.  
4. It prints the processing time upon completion.  
5. The output is saved as a zarr group with fields "data" and "metadata".

### Example Output  

```plaintext
Processing /root/capsule/data/video1_face.mp4
Processing /root/capsule/data/video2_face.mp4
processing only face videos for now
Total time taken: 15.32 seconds
