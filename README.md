# rPPG Vitals Monitor

Camera-based contactless heart rate, respiratory rate, and HRV from a standard webcam — no wearables, no contact.

*Note: Live cloud deployment on free tiers (like Render) is not recommended due to ML model RAM constraints. See local/Docker instructions below.*

---

## Quick Start

**Local CLI (native pipeline, best accuracy):**

```bash
git clone [https://github.com/JoshPola96/rppg-monitor.git](https://github.com/JoshPola96/rppg-monitor.git)
cd rppg-monitor

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python local_run.py --mode webcam
python local_run.py --mode file --file path/to/video.mp4
```

**Browser app (Docker, headless-safe):**

```bash
docker build -t rppg-monitor .
docker run --rm --init -p 8080:8000 rppg-monitor
# → http://localhost:8080

# With live reload during development:
docker run --rm --init -p 8080:8000 -v ${PWD}:/app rppg-monitor \
  uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

> The `--init` flag is required on Linux and Windows. It wraps the process in `tini` so `Ctrl+C` terminates the container immediately and releases the port rather than waiting for a forced kill timeout.

---

## What It Does

Point a camera at your face. The system detects the subtle rhythmic colour changes in skin — the green channel brightens and dims as blood pulses through surface vessels — and extracts:

* **Heart rate (BPM)** — per sliding window and SQI-weighted aggregate over the full session
* **Respiratory rate** — extracted from the BVP waveform's low-frequency modulation
* **HRV:** IBI, SDNN, RMSSD, pNN50, LF/HF ratio

Two input paths are supported: live webcam stream (WebSocket) or file upload (POST).

---

## Camera and Positioning Requirements

Signal quality depends almost entirely on two physical inputs the software cannot control: the light hitting your face, and your distance from the lens. The model extracts pulse from the green channel of skin pixels in the forehead and upper cheek regions. Anything that masks, shadows, or noisily modulates that channel will degrade SQI.

**What works:**

* Face the light source directly. A window in front of you, a lamp aimed at your face, or a bright monitor — all work. A window or bright surface behind you creates silhouette conditions that suppress the signal.
* Sit close enough that your face fills roughly a third to half of the frame. The Haar cascade needs a minimum 80×80 pixel face detection box; once detected, the pipeline crops to 128×128 for inference.
* Sit upright, chin level, face forward. The forehead and cheeks are the primary ROI. When the head tilts down the forehead foreshortens in the 2D frame and the brow casts shadows across the cheek — both effects reduce the pixel area the model can use.
* Stay still. Slow postural sway is tolerated; talking, laughing, or a hand crossing the frame will be caught by the motion gate and those frames will be dropped.

**What the UI does and does not tell you:**
The SQI bar, dynamic tracking bounding box, and face-frame colour indicator give real-time signal quality feedback: green means the model has a clean lock, amber is marginal, red means the current window is too noisy to trust. Specific drop reasons (blurry, motion, no face detected, lighting out of range) are surfaced as status messages when the quality gate rejects frames.

Hardcoded posture-correction overlays were considered and explicitly dropped. A guide that says "chin up" relative to a fixed pixel position would give instructions that are entirely inconsistent with the actual SQI — it cannot know whether the problem is head angle, lighting, distance, or webcam exposure drift. The tracking box dynamically projects the backend Haar cascade boundary onto the live video feed so you always know what the model is looking at, while the SQI bar and colour feedback convey real signal quality.

---

## Two Systems: What They Are and Why Both Exist

This project has two distinct implementations that share the same model weights but differ in how they interact with the camera and the inference pipeline.

### System 1 — `local_run.py` (Native Pipeline)

Uses open-rppg's built-in real-time pipeline directly:

```text
Camera device
    │
    ▼
model.video_capture(0)        ← opens /dev/video0 or DirectShow source
    │
model.preview                 ← generator: yields (frame_rgb, box) per camera frame
    │                            library handles: face detection, ROI extraction,
    │                            signal buffering, bandpass filter, detrending,
    │                            quality checks
    ▼
model.hr(start=-WINDOW_SIZE)  ← pulls metrics from the last N seconds of
                                 buffered signal on a timed interval
```

The library is doing everything internally. You are just reading its output. This is the gold standard: the same preprocessing that the model was trained against, applied to raw camera frames.
File mode uses `model.process_video(path)` — a single call that handles the complete pipeline over a video file and returns one result dict for the entire clip. 

*Limitation: `model.video_capture(0)` opens a physical camera device on the machine running the code. In a headless Docker container there is no camera device to open. This mode only runs locally.*

### System 2 — `app.py` (Browser-based, Docker-safe)

The browser is the capture device. The backend is the inference engine. The library's native pipeline is replicated manually in Python:

```text
Browser camera (getUserMedia)
    │
    │  JPEG frames over WebSocket (~30 fps)
    ▼
FastAPI /ws/stream
    │
    ├── cv2.imdecode (JPEG → BGR)
    ├── Haar cascade face detection (replicates library's internal detection)
    ├── Pad + crop → 128×128 RGB (replicates library's ROI extraction)
    ├── FrameQualityMonitor (replicates library's quality checks)
    │     blur (Laplacian variance), brightness, inter-frame motion
    ├── deque(maxlen=600) rolling buffer (replicates library's signal buffer)
    │
    ├── Every STEP_FRAMES: np.stack → process_faces_tensor(tensor, fps) (Async)
    │     ↑ pre-cropped tensor path — skips library's internal detection
    │     ↑ note: "Tensor mode, video quality check disabled" fires here;
    │       that is expected — our FrameQualityMonitor is the replacement
    │
    ├── Median filter (5-window, kills single-inference spikes)
    ├── EMA smoothing (α=0.70 for HR, α=0.30 for RR)
    └── JSON → WebSocket → browser
```

For file upload (`/analyze`), `process_video_file_sync` reads frames one by one with OpenCV, extracts 128×128 face crops, accumulates chunks of `WINDOW_FRAMES` crops, and calls `process_faces_tensor` per chunk. It never allocates a full-video tensor — critical for staying within strict memory limits:

* **Full frame buffer at 640×480:** `600 × 640 × 480 × 3 = 552 MB` (OOM on free tiers)
* **Face crop buffer at 128×128:** `600 × 128 × 128 × 3 = 37 MB` (Safe)

---

## Architecture

```text
Browser (Live)    JPEG canvas frames ──► WS  /ws/stream
Browser (Upload)  <input type=file>  ──► POST /analyze

                                                │
                              ┌─────────────────▼────────────────┐
                              │            FastAPI               │
                              │                                  │
                              │  /analyze                        │
                              │  cv2.VideoCapture (frame-by-frame)│
                              │  extract_face → 128×128 crops    │
                              │  adaptive window (≥5s, ≤20s)     │
                              │  ThreadPool: process_faces_tensor│
                              │  SQI-weighted aggregate          │
                              │                                  │
                              │  /ws/stream  [O(1) mode]         │
                              │  JPEG → cv2.imdecode             │
                              │  Haar cascade + 10% pad crop     │
                              │  FrameQualityMonitor gate        │
                              │  deque(maxlen=600) rolling buffer│
                              │  np.stack → process_faces_tensor │
                              │  (Non-blocking Task)             │
                              │  median(5) + EMA(α=0.70)         │
                              │  → JSON back over WebSocket      │
                              │                                  │
                              │  FacePhys.rlap                   │
                              │  warm in memory, threading.Lock  │
                              └──────────────────────────────────┘
```

`local_run.py` (local only):

```text
Camera device ──► model.video_capture(0) ──► model.preview ──► model.hr(start=-20)
```

The model is loaded once at startup and held in memory behind a `threading.Lock`. Inference runs in a `ThreadPoolExecutor(max_workers=2)` via `asyncio.run_in_executor`, keeping the async event loop free during CPU-bound inference.

---

## Key Configuration Values

| Parameter | Value | Rationale |
| :--- | :--- | :--- |
| **WINDOW_FRAMES** | 600 (20 s) | Narrower FFT bins — at 5 s, frequency resolution is 0.2 Hz, wide enough for adjacent BPM values to bleed into each other. 20 s gives 0.05 Hz resolution. |
| **STEP_FRAMES** | 36 (~1.2 s) | Inference fires once per 1.2 s of new frames. Matches the `UPDATE_INTERVAL` used by `local_run.py`. |
| **EMA_ALPHA** | 0.70 | Higher alpha = faster response. Earlier value of 0.15 meant a real HR change took ~30 s to appear in the UI. |
| **SQI_THRESHOLD** | 0.35 | Gate for accepting an inference result. The native pipeline pre-cleans signal more aggressively, so the threshold can be lower than typical rPPG deployments. |
| **BLUR_THRESHOLD** | 30.0 | Laplacian variance on the 128×128 face crop. Earlier value of 80.0 was rejecting clean frames. |
| **MOTION_THRESHOLD** | 25.0 | Mean absolute inter-frame diff on face grayscale. Earlier value of 8.0 was dropping frames during normal breathing movement. |
| **JPEG quality** | 0.5 | At quality 1.0, frames were ~410 KB each → ~12 MB/s WebSocket throughput → TCP congestion caused frames to arrive in bursts. Quality 0.5 gives ~27 KB per frame (~800 KB/s), uniform arrival cadence. |

---

## The Window Constraint (5s vs 20s)

A 5-second processing window is often asked for in real-time specifications, but it is an honest constraint worth understanding:

| Metric | 5 s window | 20 s window | Why |
| :--- | :--- | :--- | :--- |
| **Heart Rate** | ✅ Estimable, volatile | ✅ Stable | 5–6 beats visible at rest. FFT bins are wide (0.2 Hz) at 5 s, narrow (0.05 Hz) at 20 s. |
| **Respiratory Rate** | ⚠️ Unreliable | ✅ Usable | One breath is 3–5 s. A single cycle is not enough for frequency analysis. |
| **HRV SDNN / RMSSD** | ⚠️ Indicative | ⚠️ Indicative | Clinically meaningful HRV requires ≥1–2 min of clean signal. |
| **LF/HF ratio** | ❌ Meaningless | ❌ Questionable | LF band = 0.04–0.15 Hz; one LF wave period is 7–25 s — physically longer than a 5 s window and marginal at 20 s. |

The system runs with a 20-second window by default. All metrics are produced and surfaced. HR is the most trustworthy output at this window size. Respiratory rate and HRV values are real computations from the model's BVP signal, not fabricated — but they should be read as indicative rather than clinical.

---

## Signal Quality — The Compounding Variables

* **Webcam auto-exposure is the dominant noise source.** The rPPG algorithm detects periodic green-channel fluctuations caused by blood flow through skin. A webcam's automatic exposure system adjusts overall frame brightness continuously in response to head movement, background content, and even what is displayed on a nearby monitor. To the algorithm, a hardware brightness shift is indistinguishable from a blood volume pulse. This is the primary ceiling in typical indoor setups.
* **JPEG quality vs. arrival cadence.** This was counterintuitive. Lower-quality JPEG improves signal stability because uniform frame arrival matters more than per-pixel fidelity. The rPPG signal exists in the colour domain across time — timing regularity of frames is a prerequisite for FFT accuracy.
* **Frame timing: requestVideoFrameCallback vs. setInterval.** `setInterval(fn, 33)` drifts against the hardware camera clock. `requestVideoFrameCallback` fires at the exact moment the camera delivers a new hardware frame. The difference is relevant because the FFT assumes uniformly-spaced temporal samples. Frame-rate jitter introduces frequency-domain noise that shows up as SQI degradation.
* **Targeted Quality Gates.** The quality gate thresholds are applied to the face crop, not the full frame. This is deliberate. A threshold applied to the full frame will fire on background lighting changes (someone turning on a lamp across the room) that do not actually affect the face-ROI signal. By applying blur, brightness, and motion checks to the 128×128 crop, background noise cannot trigger false rejections.

---

## File Upload: Video Encoding Matters

Standard consumer video uses inter-frame compression (H.264/H.265). Only key frames (I-frames) store complete pixel data; subsequent frames store motion deltas. rPPG depends on detecting ~1% green-channel variation across frames — compression algorithms routinely classify this as noise and discard it. The open-rppg library logs a warning when it detects non-key frames: `OPEN-RPPG:WARNING - Detected non-key frames, this will damage the rPPG signal.`

For best results with file upload, transcode the video to an all-intra stream before uploading:

```bash
# Install ffmpeg if not available:
pip install ffmpeg-downloader
ffdl install

# Transcode (forces every frame to be a keyframe at constant 30 fps):
ffmpeg -i input.mp4 \
       -c:v libx264 \
       -x264-params "keyint=1" \
       -r 30 \
       -pix_fmt yuv420p \
       input_fixed.mp4
```

On Windows, if ffmpeg is not on PATH after installation:

```powershell
# Add to session PATH:
$env:Path += ";C:\Users\<user>\AppData\Local\ffmpegio\ffmpeg-downloader\ffmpeg\bin"

# Or call directly with full path:
& "C:\Users\<user>\AppData\Local\ffmpegio\ffmpeg-downloader\ffmpeg\bin\ffmpeg.exe" `
    -i input.mp4 -c:v libx264 -x264-params "keyint=1" -r 30 -pix_fmt yuv420p input_fixed.mp4
```

A well-encoded I-frame-only video will eliminate the non-key frames warning and typically raise SQI from the 18–40% range to 60–80%+. The browser live stream path does not have this problem because it captures raw canvas frames before any compression is applied.

---

## SQI-Weighted Aggregation

A plain average across chunks does not hold up when a single window has a lighting change or motion burst. The Signal Quality Index measures how clean the extracted BVP signal is for that window. Every metric is weighted by it:

```text
final_bpm = Σ(bpm_i × sqi_i) / Σ(sqi_i)
```

Chunks below `SQI = 0.35` are excluded entirely. For short videos (shorter than the 20-second ideal window), the window size shrinks adaptively down to a 5-second minimum (150 frames) so that something useful is returned rather than an error.

---

## Sample Output

**File upload — 60-second video:**

```json
{
  "chunks": [
    {
      "chunk_index": 0,
      "time_start_s": 0.0,
      "time_end_s": 20.0,
      "bpm_fft": 74.3,
      "bpm_peak": 73.8,
      "sqi": 0.7841,
      "respiratory_rate": 15.2,
      "hrv_ibi": 807.6,
      "hrv_sdnn": 42.1,
      "hrv_rmssd": 38.4,
      "hrv_pnn50": 21.3,
      "hrv_lf_hf": 1.42,
      "processing_time_ms": 1340.0
    }
  ],
  "aggregate": {
    "final_bpm": 74.5,
    "final_rr": 15.1,
    "avg_sqi": 0.7214,
    "bpm_std": 1.82,
    "agg_hrv_sdnn": 40.7,
    "agg_hrv_rmssd": 37.2,
    "agg_hrv_lf_hf": 1.38,
    "chunks_total": 3,
    "chunks_valid": 3,
    "total_time_ms": 14820.0,
    "message": "OK"
  }
}
```

**Live stream — WebSocket emits this every ~1.2 s of new frames:**

```json
{
  "status": "ok",
  "bpm": 74.5,
  "raw_bpm": 75.1,
  "bpm_peak": 73.9,
  "sqi": 0.7841,
  "rr": 15.2,
  "hrv_ibi": 807.6,
  "hrv_sdnn": 42.1,
  "hrv_rmssd": 38.4,
  "hrv_pnn50": 21.3,
  "hrv_lf_hf": 1.38,
  "buffered_frames": 600,
  "box": [245, 120, 160, 160]
}
```

**WebSocket status values:**

| Status | Meaning |
| :--- | :--- |
| `buffering` | Filling the 20-second initial window. buffered and target fields show progress. |
| `low_signal` | Frame dropped by quality gate. reason: blurry / motion / no_face / lighting. |
| `ok` | Full metrics payload — SQI gate passed. |
| `no_signal` | Model returned no HR estimate this window. |
| `error` | Inference failure. Check detail field. |

---

## Browser vs. Native: Empirical Comparison (Lab vs. Wild)

This comparison is a perfect "lab vs. wild" case study. While the core logic is successfully replicated in the browser architecture, empirical data reveals exactly where the web architecture pays a "tax" compared to the native implementation.

| Metric | Browser (Custom via app.py) | Native (Direct via local_run.py) | Observation |
| :--- | :--- | :--- | :--- |
| **Live HR Accuracy** | ~61.1 – 62.7 BPM | ~61.8 BPM (Avg) | Parity. The core math is identical. |
| **File HR Accuracy** | 73.9 BPM | 73.3 BPM | Parity. < 1 BPM difference on the same file. |
| **SQI (Quality)** | ~53% – 64% | ~82.8% | Native Wins. Browser compression hurts signal. |
| **File Process Time** | 13.46 seconds | 2.54 seconds | Native Wins. Massive overhead in the web loop. |
| **Inference Latency** | ~1800ms | ~1100ms (est) | Native Wins. Web involves JPEG/Base64 overhead. |

1. **The "Compression Tax" (SQI Gap):** Native SQI (~82.8%) is significantly higher than Browser SQI (~53-64%). In `index.html`, frames are sent as `image/jpeg` at 0.5 quality to save bandwidth. rPPG relies on detecting minute color changes (often in the 8-bit noise floor). JPEG compression "smooths" these colors to save space, effectively deleting the very signal the model is looking for.
2. **The Loop Overhead (Processing Time):** Native `model.process_video` finished in 2.54s, while the Browser `POST /analyze` took 13.46s. The Native library uses a highly optimized pre-compiled C++/CUDA pipeline pulling frames directly into a buffer. The browser backend has to save the file, open it with OpenCV, run a Python loop, execute a Haar Cascade on every single frame, and use `np.stack` to create tensors. This "Python tax" and per-frame detection adds nearly 11 seconds of latency for a 12-second video.
3. **Heart Rate Consistency:** The good news is that the FFT (Fast Fourier Transform) results are incredibly stable across both. Both versions landed within 0.6 BPM of each other on static files, and within ~1 BPM on live webcams. The custom face-cropping and tensor-stacking logic is mathematically sound.

**Conclusion:** Native is the "Gold Standard" for accuracy and speed. However, for a remote monitoring tool, a 1.8s delay and a 15% drop in SQI is a fair trade for the ability to run in a web browser. Note that HRV metrics (SDNN/RMSSD) are much more sensitive to the JPEG compression "jittering" the peak detection, making web HRV less reliable than native and the browser based implementation can be improved with further iterations.

---

## Failure Cases

* **Lighting changes:** A bright flash or window glare oversaturates the green channel. The `FrameQualityMonitor` catches abrupt brightness changes, but a sustained change (someone turning a lamp on in the background) will depress SQI for several windows until the EMA adapts. Front-facing diffuse light eliminates this class of failure.
* **Webcam auto-exposure:** The primary ceiling in normal indoor use. See the signal quality section above. A ring light or facing a bright window reduces this significantly. Sitting in a dark room with only a monitor as the light source is the worst-case scenario — the monitor content change triggers continuous auto-exposure adjustment.
* **Motion:** Slow head movement and normal breathing are tolerated. Talking, laughing, or a hand in front of the face will degrade SQI for that window. The motion gate catches abrupt movement and drops those frames from the buffer.
* **Short windows and HRV:** SDNN, RMSSD, and LF/HF are produced from the model's BVP signal analysis. They are real values from the algorithm, not estimates — but at 20-second windows they should be read as indicative. Clinical HRV analysis uses 5-minute recordings as a minimum.
* **Variable frame rate (file upload):** The pipeline reads actual source FPS from OpenCV (`CAP_PROP_FPS`) and sizes windows accordingly, so FFT frequency mapping stays accurate regardless of recording FPS. For live streaming, 30 fps is assumed from the camera — a production system would timestamp each frame individually.
* **Short uploads:** If a video is shorter than the 20-second ideal window, the window shrinks adaptively to fit. Absolute floor is 5 seconds (150 frames). Below that, inference is refused with a descriptive error rather than returning a number that looks credible but is based on too little signal.

---

## Running Locally: Full Command Reference

```bash
# Clone and set up environment
git clone [https://github.com/JoshPola96/rppg-monitor.git](https://github.com/JoshPola96/rppg-monitor.git)
cd rppg-monitor

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# ── Native CLI ───────────────────────────────────────────────────────────────

# Live webcam (default model: FacePhys.rlap)
python local_run.py --mode webcam

# Live webcam with alternate model
python local_run.py --mode webcam --model PhysMamba.pure

# Single video file analysis
python local_run.py --mode file --file path/to/video.mp4

# File analysis with alternate model
python local_run.py --mode file --file path/to/video.mp4 --model EfficientPhys

# Press Q in the preview window to exit cleanly.
# Ctrl+C also works. 

# ── Browser App (local, no Docker) ──────────────────────────────────────────

uvicorn app:app --reload --port 8000
# → http://localhost:8000

# ── Browser App (Docker) ─────────────────────────────────────────────────────

# Build image
docker build -t rppg-monitor .

# Run (production mode)
docker run --rm --init -p 8080:8000 rppg-monitor

# Run with live reload (development — mounts local directory)
docker run --rm --init -p 8080:8000 -v ${PWD}:/app rppg-monitor \
  uvicorn app:app --host 0.0.0.0 --port 8000 --reload

# Windows PowerShell (use ${PWD} or full path):
docker run --rm --init -p 8080:8000 -v ${PWD}:/app rppg-monitor `
  uvicorn app:app --host 0.0.0.0 --port 8000 --reload

# Check container health
curl http://localhost:8080/health

# Stop container (Ctrl+C, or from another terminal):
docker ps                          # find CONTAINER_ID
docker stop <CONTAINER_ID>
```

> The `--init` flag is critical. Without it, Docker runs uvicorn as PID 1 with no signal handler. `Ctrl+C` sends SIGTERM to the container, which is ignored by a PID-1 process without explicit handling. The container then waits 10 seconds before sending SIGKILL. `--init` inserts `tini` as PID 1, which correctly propagates SIGTERM to uvicorn and exits immediately. The Dockerfile's `CMD` uses `exec` form (not `sh -c` shell form) for the same reason — shell-wrapped processes do not receive signals.

---

## Deployment (Informational Note on Render)

If deploying to a cloud platform like Render, a standard web service with a Docker runtime works well. Render will auto-deploy upon GitHub push. The Dockerfile pre-bakes model weights at build time so there is no cold-start download in production:

```dockerfile
RUN python -c "import rppg; rppg.Model('FacePhys.rlap')" || true
```

The `|| true` prevents the build from failing if the model download has a transient network error — it will re-download at first request instead.

**Note:** Cloud providers' free tiers spin down inactive services. A WebSocket handshake against a sleeping instance will fail at the TLS layer before the app receives it. Either set the service to always-on, or implement reconnect logic on the frontend (the `ws.onclose` handler in `index.html` calls `stopStream()` on unexpected disconnect — for auto-reconnect, that handler would instead call `startStream()` after a short delay).

---

## API

### `POST /analyze`

Upload a video for chunk-level and aggregate analysis.

* **Request:** `multipart/form-data`, field `file`. Accepts `.mp4 .webm .mov .avi .mkv`. Minimum 5 seconds.
* **Response:**
  ```json
  {
    "chunks":    ChunkMetrics[],
    "aggregate": AggregateResult
  }
  
```
* **Error codes:** 400 (file too small), 415 (unsupported format), 422 (no signal extracted or video too short)

### `WS /ws/stream`

Send JPEG data-URLs from a canvas element. Receive JSON per inference window. See status values table above.

### `GET /health`

```json
{
  "status": "ok",
  "model": "FacePhys.rlap",
  "inference": "process_faces_tensor",
  "target_fps": 30,
  "window_s": 20.0,
  "face_size": 128
}
```

---

## Stack

* **Model:** open-rppg — `FacePhys.rlap`, a state-space model trained on the RLAP benchmark
* **Backend:** FastAPI + uvicorn, Python 3.11
* **Face detection:** OpenCV Haar cascade (frontal face, bundled with opencv-python-headless)
* **Media decoding:** OpenCV VideoCapture (file upload), base64 JPEG over WebSocket (live stream)
* **Frontend:** plain HTML/JS — no framework, no build step
* **Deployment:** Docker

---

## The Engineering Journey

What follows is an account of the actual development sequence, including the wrong turns.

### Version 1 — The Bug That Ate the First Implementation

Initial architecture: `MediaRecorder` with a 250 ms timeslice, blobs sent over WebSocket, decoded with PyAV on the backend, frames accumulated, inference every 5 seconds.
*This produced exactly one reading and then nothing.*

The problem: browser `MediaRecorder` with timeslice only writes the WebM EBML container header into the very first blob. Every blob after that is a raw VP8 media segment — technically an incomplete file. PyAV requires the container header to initialise its demuxer. It cannot parse a bare media segment in isolation, and it returns an empty frame list without throwing an error. The buffer never filled past the first window. The logs looked normal. The silence was the only signal something was wrong.
Fix at that stage: accumulate all bytes from session start into one growing buffer and pass the entire thing to PyAV on each decode. This works because blob zero always contains the EBML header. It produced valid results.

### Version 2 — The Performance Cliff

Once byte-accumulation was stable, a new problem appeared: inference gaps that were 1 second at session start were 3+ seconds by minute one. The loop was falling behind real-time.
Cause: `decode_video_bytes(full_accumulated_buffer)` is not constant-time. It grows linearly with session length. At 50 seconds in, the server was decoding 50 seconds of video just to extract the last few seconds of frames. The PyAV call was the bottleneck, not the model inference.
This is the $O(n)$ problem: accumulated bytes × decode cost × inference frequency = a wall that arrives around the 30–40 second mark for typical bitrates.

### Version 3 — O(1) Frame Mode (Current Architecture)

The architectural shift: stop treating frames as a transport problem and treat them as a data problem.
Instead of accumulating compressed video fragments and repeatedly decoding them, extract raw frames on the frontend and send them individually. The deque on the backend automatically discards frames older than the window. Memory and compute are flat regardless of session length.

* **Frontend:** `requestVideoFrameCallback` draws each hardware-synchronised camera frame onto a hidden canvas and sends it as a base64 JPEG.
* **Backend:** `deque(maxlen=600)` discards the oldest frame on each new arrival. When inference fires, `np.stack(frame_buffer)` builds the tensor from exactly the most recent 600 frames.

### The Async Inference Backpressure Trap (Bug Fix)

In the live stream, a severe bug was discovered where the WebSocket stream automatically closed after a few seconds. The cause was an inference speed (~1400-1800ms) and camera stream accumulation step (~1200ms) mismatch. Because they were running sequentially in the async handler, frames piled up in memory, backpressure built, and Python eventually killed the connection.

The fix was to decouple frame ingestion from inference. Inference is now offloaded to a non-blocking `asyncio.create_task`. If an inference cycle is still running when the next 1.2s step arrives, the app simply skips the UI update cycle and continues receiving frames rather than queuing a blocking operation, keeping memory perfectly flat.

### Signal Quality — The Compounding Variables

After the architecture was stable, SQI was still fluctuating between 0.19 and 0.74 in the same session under apparently unchanged conditions. Working through the variables one by one:

* **JPEG quality** mattered in the opposite direction from what was expected. Initial quality was set to 1.0 on the theory that micro-colour changes need maximum fidelity. High-quality JPEGs at 640×480 are ~410 KB per frame. At 30 fps that is ~12 MB/s over the WebSocket. TCP congestion caused 2–3 frames to arrive simultaneously, then a gap. To the rPPG model, irregular frame timing looks like heartbeat events in the frequency domain. Dropping quality to 0.5 reduced frame size to ~27 KB (~800 KB/s) and arrival cadence became uniform. SQI stabilised.
* **Frame timing.** `setInterval` at 33 ms introduces drift against the hardware camera clock under any CPU load. `requestVideoFrameCallback` fires at the exact moment the camera delivers a new hardware frame. The difference is not visible to the eye but is meaningful to an FFT expecting uniformly-spaced temporal samples.
* **Quality gate thresholds** needed empirical calibration. Initial blur threshold of 80.0 (Laplacian variance) was rejecting clean frames — the face crop at 128×128 has a naturally lower Laplacian variance than a full HD frame. Initial motion threshold of 8.0 was dropping frames during normal breathing movement. Final values (blur 30.0, motion 25.0) gate genuinely bad frames while passing normal slight movement.

### Bounded Face Tracking UI

To give users confidence that the system hasn't frozen when inference lags or drops frames, a dynamic tracking box was added to the frontend. It projects the server-side Haar cascade boundary onto the live video feed. This ensures the bounding box continuously tracks the face even when the signal quality drops to amber or red, providing immediate visual feedback.

### Webcam Mode — The Exit Crash

The `RuntimeError: cannot join current thread` that fires on exit from the native webcam mode is a thread-cleanup bug in the `open-rppg` library itself. When exiting the `model.video_capture(0)` block, the context manager attempts to join the library's internal capture threads. However, a background thread (`Thread-3`) tries to `join()` itself during cleanup.

Because this happens inside a separate background thread, it bypasses the main thread's `try...except` block entirely and prints an ugly stack trace directly to `stderr` right after the session summary prints. To get a completely clean exit, this can be fixed by intercepting unhandled thread exceptions globally using `threading.excepthook` to selectively mute this specific exception.

### Bugs Resolved Across the Session

* **`Object of type float32 is not JSON serializable`** — FastAPI HTTP endpoints go through Pydantic, which handles NumPy types natively. The WebSocket `send_json` uses the stdlib encoder, which does not. The `make_serializable` utility recursively walks the payload dict and converts all NumPy primitives before the encoder sees them.
* **`Unexpected token 'N' … NaN is not valid JSON`** — `heartpy` returns `float('nan')` (standard Python NaN) when it fails to find peaks in a noisy signal. The stdlib encoder writes `NaN` which is valid JavaScript but not valid JSON. `make_serializable` now checks `math.isnan` and `math.isinf` on standard Python floats and substitutes `0.0`.
* **`'<' not supported between instances of 'NoneType' and 'float'`** — `result.get("SQI", 0.0)` does not return the default when the model explicitly sets `SQI: None`. Python's `dict.get` only uses the default for absent keys, not for explicitly-set `None` values. The fix is `result.get("SQI") or 0.0`, which evaluates `None` as falsy.
* **`unsupported format string passed to NoneType.__format__`** — `heartpy` occasionally returns `{"hr": None}` on complete signal failure. The metrics formatter was calling `:.1f` on that value. Fixed by guarding every extraction.
* **Duplicate frame append** — An incomplete merge of a reviewed patch resulted in `frame_buffer.append(face_rgb)` being called twice per frame. The buffer looked full but contained only half the unique time coverage, giving the model a signal that pulsed at half the real frequency.
* **Dockerfile CMD used `sh -c` wrapper** — `sh -c "uvicorn app:app ..."` is the parent process. SIGTERM goes to `sh`, which does not propagate it to uvicorn. The container waited 10 seconds for the forced SIGKILL on every shutdown. Replaced with `exec` form.
* **The 422 Unprocessable Entity on file upload** — After switching to a 20-second window (`WINDOW_FRAMES=600`), short test videos (under 20 seconds) produced zero chunks because the pipeline required a full 600-frame window before running inference. Fixed with adaptive windowing.

### Video Encoding — The I-frame Discovery

Early file upload tests returned SQI values in the 18–21% range with a consistent library warning: `OPEN-RPPG:WARNING - Detected non-key frames, this will damage the rPPG signal.`
Standard H.264/H.265 video uses inter-frame compression. Only I-frames (keyframes) contain complete pixel data; P-frames and B-frames store motion vectors and deltas. The rPPG signal exists as a ~1% variation in the green channel across frames. Inter-frame compression routinely classifies this as noise and discards it. Transcoding to an all-intra stream forces every frame to be a complete image, raising SQI to 60–80%+. The live stream path avoids this completely because Canvas JPEG frames sent over WebSocket are individually complete images with no inter-frame compression.

### How AI Was Used

LLMs (Claude and Gemini) were used throughout as a combination of research layer, code scaffolding generator, and debugging thought partner.

* **What AI is genuinely good at here:** Structural scaffolding — FastAPI app skeleton, Pydantic schemas, WebSocket endpoint boilerplate, Dockerfile, frontend JS — all generated fast and iterated from there. When reasoning through design tradeoffs — O(1) deque vs. byte accumulation, EMA vs. median smoothing — AI holds context well and stress-tests ideas quickly.
* **What AI cannot do:** Run the code. Every bug in this build was found in a terminal or a browser log. AI reviewed the same code containing the `dict.get` / explicit `None` crash and did not catch it — because catching it requires executing the inference path against a model that actually returns `{"SQI": None}`. Once it saw the error text, it identified the root cause immediately — that is pattern matching on known error strings, not understanding the execution path.
* **The 5-Second Window Calibration:** Working with a 5-second window acts as an excellent signal processing stress test. At 5 seconds you get 5–6 heartbeats at resting rate. FFT frequency resolution at 5 seconds is 0.2 Hz — wide enough that adjacent BPM values bleed into each other. Respiratory rate requires at least one full breath cycle (3–5 s) to measure a frequency. LF/HF ratio requires the low-frequency band (0.04–0.15 Hz), whose period is 7–25 seconds. When asked to extract all these metrics from 5-second chunks, AI generated the code, the schemas, and the aggregation logic without flagging the math logic. Recognising that respiratory and HRV figures from a 5-second window are not reliable required understanding the signal processing — not generating code for it.

Approximate split: AI generated the initial structure and most of the boilerplate (~70% of lines written). Architectural decisions, all debugging against live logs, all threshold calibration (blur, brightness, motion, SQI gate, EMA alpha) against a real face and webcam, and decisions about what to discard from AI suggestions when they conflicted with what the logs were showing — that was manual.

---

## What Would Come Next

* **Frame-level timestamping.** The current pipeline assumes frames arrive at exactly `TARGET_FPS`. Attaching a monotonic timestamp to each frame and using it to build the FFT time axis would make frequency mapping accurate under variable camera frame rates without assuming uniform delivery.
* **Stateful inference.** Each window is currently independent. A model that carries BVP signal history across windows would give more stable HRV estimates and a faster time to first reliable reading — the 20-second warmup is a direct consequence of the stateless window approach.
* **Concurrency.** `ThreadPoolExecutor(max_workers=2)` is a single-server constraint. Proper multi-user deployment separates the FastAPI ingest layer from compute and pushes inference to a worker queue (Celery, RQ, or similar).
* **Auto-exposure suppression.** `getUserMedia` constraints include `exposureMode: 'manual'` and `exposureTime` settings. Browser and hardware support is currently too inconsistent to rely on, but as webcam driver support improves this becomes the single highest-impact quality improvement available.

*Joshua Peter Polaprayil — AI/ML Engineer May 2026*