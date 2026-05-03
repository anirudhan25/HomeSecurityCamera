import argparse, time, threading, logging
from datetime import datetime
from collections import deque
from typing import Callable

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template_string

# IP ADDRESS: 81.106.34.95:2222/video_feed?password=home-cam  (running in tmux)
# usage: python app.py --stream-url http://home-cam.local:8080/stream

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class FrameGrabber:
    def __init__(self, stream_url: str, reconnect_delay: float = 3.0):
        self.stream_url = stream_url
        self.reconnect_delay = reconnect_delay
        self._frame: np.ndarray | None = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None
        self._fps = 0.0
        self._frame_count = 0

    @property
    def frame(self) -> np.ndarray | None:
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    @property
    def fps(self) -> float:
        return round(self._fps, 1)

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        logger.info(f"grabber started: {self.stream_url}")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)

    def _capture_loop(self):
        while self._running:
            cap = cv2.VideoCapture(self.stream_url)
            if not cap.isOpened():
                logger.warning(f"cannot open stream, retrying in {self.reconnect_delay}s...")
                time.sleep(self.reconnect_delay)
                continue

            logger.info("stream connected")
            t_start, count = time.monotonic(), 0

            while self._running:
                ret, frame = cap.read()
                if not ret:
                    logger.warning("stream dropped")
                    break

                with self._lock:
                    self._frame = frame
                    self._frame_count += 1

                count += 1
                elapsed = time.monotonic() - t_start
                if elapsed >= 1.0:
                    self._fps = count / elapsed
                    count, t_start = 0, time.monotonic()

            cap.release()
            if self._running:
                logger.info(f"reconnecting in {self.reconnect_delay}s...")
                time.sleep(self.reconnect_delay)


class CVPipeline:
    def __init__(self):
        self._processors: dict[str, Callable[[np.ndarray], np.ndarray]] = {}
        self._active: set[str] = set()

    def register(self, name: str, fn: Callable[[np.ndarray], np.ndarray]):
        self._processors[name] = fn

    def enable(self, name: str):
        if name in self._processors:
            self._active.add(name)

    def disable(self, name: str):
        self._active.discard(name)

    def toggle(self, name: str) -> bool:
        if name in self._active:
            self.disable(name)
            return False
        self.enable(name)
        return True

    @property
    def available(self) -> dict[str, bool]:
        return {name: name in self._active for name in self._processors}

    def process(self, frame: np.ndarray) -> np.ndarray:
        for name, fn in self._processors.items():
            if name in self._active:
                try:
                    frame = fn(frame)
                except Exception as e:
                    logger.error(f"processor '{name}' failed: {e}")
        return frame


def grayscale(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)


def edge_detection(frame: np.ndarray) -> np.ndarray:
    edges = cv2.Canny(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 50, 150)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)


def gaussian_blur(frame: np.ndarray) -> np.ndarray:
    return cv2.GaussianBlur(frame, (15, 15), 0)


def motion_detector(threshold: float = 25.0, min_area: int = 500):
    bg_sub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=threshold, detectShadows=True)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def _process(frame: np.ndarray) -> np.ndarray:
        mask = bg_sub.apply(frame)
        _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)  # shadow pixels = 127 in MOG2
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        output = frame.copy()
        motion = any(cv2.contourArea(c) >= min_area for c in contours)

        for cnt in contours:
            if cv2.contourArea(cnt) < min_area:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if motion:
            cv2.putText(output, "MOTION DETECTED", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        return output

    return _process


def face_detector():
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    def _process(frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        output = frame.copy()
        for (x, y, w, h) in faces:
            cv2.rectangle(output, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(output, "Face", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        return output

    return _process


def timestamp_overlay(frame: np.ndarray) -> np.ndarray:
    output = frame.copy()
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(output, ts, (10, frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return output


class SnapshotBuffer:
    def __init__(self, maxlen: int = 30):
        self._buffer: deque[tuple[float, np.ndarray]] = deque(maxlen=maxlen)

    def push(self, frame: np.ndarray):
        self._buffer.append((time.time(), frame))

    def latest(self) -> np.ndarray | None:
        return self._buffer[-1][1] if self._buffer else None

    def save(self, path: str) -> str | None:
        frame = self.latest()
        if frame is None:
            return None
        filepath = f"{path}/snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(filepath, frame)
        return filepath


DASHBOARD_HTML = """
<!DOCTYPE html>
<html><head>
    <title>Home-Cam</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: system-ui, sans-serif; background: #0f1117; color: #e0e0e0; }
        .container { max-width: 1100px; margin: 0 auto; padding: 20px; }
        h1 { font-size: 1.4rem; margin-bottom: 16px; color: #8bc6fc; }
        .grid { display: grid; grid-template-columns: 1fr 300px; gap: 16px; }
        .stream-box { background: #1a1d27; border-radius: 8px; overflow: hidden; }
        .stream-box img { width: 100%; display: block; }
        .panel { background: #1a1d27; border-radius: 8px; padding: 16px; }
        .panel h2 { font-size: 1rem; margin-bottom: 12px; color: #8bc6fc; }
        .toggle { display: flex; justify-content: space-between; align-items: center;
                   padding: 8px 0; border-bottom: 1px solid #2a2d37; cursor: pointer; }
        .toggle:hover { color: #8bc6fc; }
        .dot { width: 10px; height: 10px; border-radius: 50%; }
        .dot.on { background: #4ade80; }
        .dot.off { background: #555; }
        .btn { background: #2563eb; color: white; border: none; padding: 8px 16px;
               border-radius: 6px; cursor: pointer; margin-top: 12px; width: 100%; }
        .btn:hover { background: #1d4ed8; }
        #stats { font-size: 0.85rem; color: #888; margin-top: 12px; }
    </style>
</head><body>
    <div class="container">
        <h1>&#128247; Home-Cam Dashboard</h1>
        <div class="grid">
            <div class="stream-box">
                <img id="stream" src="/feed/processed" alt="Stream" />
            </div>
            <div class="panel">
                <h2>CV Processors</h2>
                <div id="processors"></div>
                <button class="btn" onclick="snapshot()">&#128248; Save Snapshot</button>
                <div id="stats"></div>
            </div>
        </div>
    </div>
    <script>
        async function loadProcessors() {
            const res = await fetch('/api/processors');
            const data = await res.json();
            const el = document.getElementById('processors');
            el.innerHTML = '';
            for (const [name, active] of Object.entries(data)) {
                const div = document.createElement('div');
                div.className = 'toggle';
                const nameSpan = document.createElement('span');
                nameSpan.textContent = name;
                const dot = document.createElement('span');
                dot.className = 'dot ' + (active ? 'on' : 'off');
                div.appendChild(nameSpan);
                div.appendChild(dot);
                div.onclick = async () => { await fetch('/api/processors/' + name + '/toggle', {method: 'POST'}); loadProcessors(); };
                el.appendChild(div);
            }
        }
        async function snapshot() {
            const res = await fetch('/api/snapshot', {method: 'POST'});
            const data = await res.json();
            alert(data.message || data.error);
        }
        async function updateStats() {
            const res = await fetch('/api/status');
            const data = await res.json();
            document.getElementById('stats').textContent = 'FPS: ' + data.fps + ' | Frames: ' + data.frame_count;
        }
        loadProcessors();
        setInterval(updateStats, 2000);
    </script>
</body></html>
"""


def create_app(grabber: FrameGrabber, pipeline: CVPipeline, snapshots: SnapshotBuffer) -> Flask:
    app = Flask(__name__)

    @app.route("/")
    def index():
        return render_template_string(DASHBOARD_HTML)

    def _generate_feed(process: bool = False):
        while True:
            frame = grabber.frame
            if frame is None:
                time.sleep(0.05)
                continue
            if process:
                frame = pipeline.process(frame)
                snapshots.push(frame)
            _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n"
            time.sleep(0.03)  # ~30 fps cap

    @app.route("/feed/raw")
    def feed_raw():
        return Response(_generate_feed(process=False), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.route("/feed/processed")
    def feed_processed():
        return Response(_generate_feed(process=True), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.route("/api/processors", methods=["GET"])
    def get_processors():
        return jsonify(pipeline.available)

    @app.route("/api/processors/<name>/toggle", methods=["POST"])
    def toggle_processor(name: str):
        if name not in pipeline.available:
            return jsonify({"error": f"unknown processor: {name}"}), 404
        return jsonify({"name": name, "active": pipeline.toggle(name)})

    @app.route("/api/snapshot", methods=["POST"])
    def take_snapshot():
        path = snapshots.save(".")
        if path:
            return jsonify({"message": f"saved to {path}"})
        return jsonify({"error": "no frame available"}), 503

    @app.route("/api/status", methods=["GET"])
    def status():
        return jsonify({
            "fps": grabber.fps,
            "frame_count": grabber._frame_count,
            "stream_url": grabber.stream_url,
            "processors": pipeline.available,
        })

    return app


def main():
    parser = argparse.ArgumentParser(description="home-cam cv backend")
    parser.add_argument("--stream-url", default="http://81.106.34.95:2222/video_feed?password=home-cam")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    grabber = FrameGrabber(args.stream_url)

    pipeline = CVPipeline()
    pipeline.register("grayscale", grayscale)
    pipeline.register("edge_detection", edge_detection)
    pipeline.register("gaussian_blur", gaussian_blur)
    pipeline.register("motion_detection", motion_detector(threshold=25.0, min_area=500))
    pipeline.register("face_detection", face_detector())
    pipeline.register("timestamp", timestamp_overlay)
    pipeline.enable("timestamp")

    snapshots = SnapshotBuffer(maxlen=60)
    grabber.start()

    app = create_app(grabber, pipeline, snapshots)
    logger.info(f"http://localhost:{args.port}")
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
