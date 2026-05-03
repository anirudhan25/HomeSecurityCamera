# Home Security Camera

Flask backend for a Raspberry Pi Zero W MJPEG camera stream with a modular CV processing pipeline and a live dashboard.

## What it does

- Reads an MJPEG stream from the Pi in a background thread with auto-reconnect
- Applies a chain of CV processors (grayscale, edge detection, motion detection, face detection, timestamp)
- Serves raw and processed feeds via `/feed/raw` and `/feed/processed`
- Dashboard at `/` lets you toggle processors on/off and save snapshots
- REST API at `/api/` for status, processor control, and snapshots

## Setup

```bash
pip install flask opencv-python numpy
python backend.py --stream-url http://<pi-address>:8080/stream
```

## CV Processors

| Name | Description |
|---|---|
| `grayscale` | Converts to grayscale |
| `edge_detection` | Canny edge detection |
| `gaussian_blur` | Gaussian blur (15×15) |
| `motion_detection` | MOG2 background subtraction with bounding boxes |
| `face_detection` | Haar cascade face detection |
| `timestamp` | Overlays current timestamp (on by default) |

Add your own by registering a `(np.ndarray) -> np.ndarray` callable via `pipeline.register()`.
