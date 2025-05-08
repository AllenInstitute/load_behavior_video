# VideoLoader: Preprocessing Pipeline for Experimental Video Data

This module provides a robust pipeline to preprocess experimental videos—particularly those from behavior, face, or eye-tracking cameras used in neuroscience experiments. It handles metadata loading, grayscale conversion, optional cropping, and saves processed videos in a consistent format for downstream analysis. While sync path is saved, the timestamp alignment should be done in the further analysis steps. 

---

## 🎯 Purpose

The `VideoLoader` class:
- Loads raw `.mp4` video files
- Optionally visualizes and adjusts cropping regions
- Converts frames to grayscale and applies cropping
- Drops metadata-containing first frame
- Saves processed video in `.mp4` format
- Exports structured metadata in JSON format

---

## 📦 Key Features

- ✅ Handles and validates video metadata
- ✅ Interactive crop preview for manual verification
- ✅ Saves output video in grayscale MP4 format
- ✅ Structured results saved per session, with consistent naming
- ✅ Compatible with motion energy analysis and COMB sync processing

---

## 🧠 Class Overview: `VideoLoader`

### Constructor
```python
VideoLoader(video_path: Path, sync_path: Path, crop_region: tuple, fps: int = None)
```
### Attributes
video_path: Path to input video file

sync_path: Corresponding sync file (*_sync.h5)

crop_region: Tuple (y, x, height, width)

fps: Override frames per second (optional)

### Main Methods
`process_and_save_video():` End-to-end processing

Calls `_get_metadata(), _check_crop(), _process_video(), and _save()`


## 🛠 Dependencies
Python 3.7+, OpenCV (cv2), NumPy, tqdm

`utils.py` module with the following functions:

* `load_camera_json(json_path)`
* `load_session_metadata_file(parent_path)`
* `construct_results_folder(self)`
* `object_to_dict(obj)`
* `show_cropped_frame(frame_rgb, shape, crop_region)`
* `get_sync_file(video_path)`

