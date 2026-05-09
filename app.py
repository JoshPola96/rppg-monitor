# rppg_monitor/app.py

"""
rPPG Vitals Monitor
Full-stack integration of the open-rppg library with dual-path inference:
  - /analyze   : file upload → face-crop chunks → process_faces_tensor → AnalysisResponse
  - /ws/stream : WebSocket live stream → O(1) face-crop deque → process_faces_tensor
"""

import asyncio
import os
import time
import tempfile
import math
import logging
import warnings
import base64
from collections import deque
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import mediapipe as mp

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import rppg
import uvicorn

# ─────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("rppg")


# ─────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────
TARGET_FPS       = 30    # Webcam target frame rate
WINDOW_FRAMES    = 600   # 20-second window (20 * 30fps) — much more stable FFT resolution
STEP_FRAMES      = 36    # Inference every ~1.2s
SQI_THRESHOLD    = 0.30  # Signal quality gate
MIN_VALID_CHUNKS = 2     # Minimum chunks for a valid aggregate (file upload)
EMA_ALPHA        = 0.70  # Higher alpha = faster response

# Face extraction — matches process_faces_tensor expected input (T, 128, 128, 3)
FACE_SIZE = 128

# Frame-level quality thresholds — applied to the face crop, not the full frame
BLUR_THRESHOLD   = 30.0   # Laplacian variance; lower = blurrier
BRIGHTNESS_MIN   = 40.0   # Mean pixel floor
BRIGHTNESS_MAX   = 220.0  # Mean pixel ceiling
MOTION_THRESHOLD = 25.0   # Mean absolute diff between consecutive face grays

# ─────────────────────────────────────────────────
# Face detector — MediaPipe BlazeFace
# ─────────────────────────────────────────────────
_mp_detector = mp.solutions.face_detection.FaceDetection(
    model_selection=0,           # 0 = short-range (<2m), tuned for webcam distance
    min_detection_confidence=0.5
)

# ─────────────────────────────────────────────────
# Global model — warm, single instance
# ─────────────────────────────────────────────────
cv_model: rppg.Model | None = None
model_lock = Lock()
executor   = ThreadPoolExecutor(max_workers=2)


# ─────────────────────────────────────────────────
# Lifespan
# ─────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global cv_model
    logger.info("Loading FacePhys.rlap model …")
    cv_model = rppg.Model("FacePhys.rlap")
    logger.info("Model ready — process_faces_tensor (pre-cropped 128x128 input)")
    yield
    logger.info("Shutting down.")


# ─────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────
app = FastAPI(
    title="rPPG Vitals Monitor",
    description="Camera-based contactless vital sign measurement prototype.",
    version="1.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")


# ─────────────────────────────────────────────────
# Schemas
# ─────────────────────────────────────────────────
class ChunkMetrics(BaseModel):
    chunk_index:        int
    time_start_s:       float
    time_end_s:         float
    bpm_fft:            float
    bpm_peak:           float
    sqi:                float
    respiratory_rate:   float
    hrv_ibi:            float
    hrv_sdnn:           float
    hrv_rmssd:          float
    hrv_pnn50:          float
    hrv_lf_hf:          float
    processing_time_ms: float


class AggregateResult(BaseModel):
    final_bpm:     float
    final_rr:      float
    avg_sqi:       float
    bpm_std:       float
    agg_hrv_sdnn:  float
    agg_hrv_rmssd: float
    agg_hrv_lf_hf: float
    chunks_total:  int
    chunks_valid:  int
    total_time_ms: float
    message:       str


class AnalysisResponse(BaseModel):
    chunks:    list[ChunkMetrics]
    aggregate: AggregateResult


# ─────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────
class EMA:
    """Exponential Moving Average for live BPM display smoothing."""
    def __init__(self, alpha: float = EMA_ALPHA):
        self.alpha = float(alpha)
        self.v: float | None = None

    def update(self, x: float) -> float:
        x_val = float(x)
        self.v = x_val if self.v is None else self.alpha * x_val + (1 - self.alpha) * self.v
        return float(round(self.v, 1))

    def reset(self):
        self.v = None


class FrameQualityMonitor:
    """
    Per-frame gate replacing the library's internal quality pipeline.

    process_faces_tensor skips the library's own quality checks (the "Tensor mode,
    video quality check disabled" warning fires on every call). This class is the
    manual replacement: blur, exposure, and motion checks applied to the 128x128
    face crop, not the full frame, so background lighting shifts cannot contaminate
    the signal path.
    """
    def __init__(self):
        self.prev_gray: np.ndarray | None = None
        self._brightness_history: deque = deque(maxlen=15)  # ~18s at 1.2s step rate

    def brightness_drift(self, brightness: float) -> bool:
        """
        True if face brightness has ranged >15 units over the last 15 readings.
        Signature of webcam auto-exposure adjusting — indistinguishable from
        blood volume pulse changes and the primary SQI destabiliser indoors.
        Operates on face-crop brightness, not full-frame, to avoid false fires
        from background lighting changes.
        """
        self._brightness_history.append(brightness)
        if len(self._brightness_history) < 8:
            return False
        return (max(self._brightness_history) - min(self._brightness_history)) > 15.0

    def check(self, face_rgb: np.ndarray) -> tuple[bool, str, float]:
        """Motion → Brightness → Blur (cheapest first). Applied to face crop only."""
        gray = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2GRAY)

        # 1. Motion — absdiff, cheapest, most common rejection cause
        if self.prev_gray is not None:
            motion = float(np.mean(cv2.absdiff(self.prev_gray, gray)))
            if motion > MOTION_THRESHOLD:
                self.prev_gray = gray   # update so we don't double-reject recovery frame
                return False, "motion", float(np.mean(gray))

        self.prev_gray = gray

        # 2. Brightness — single mean pass
        brightness = float(np.mean(gray))
        if not (BRIGHTNESS_MIN < brightness < BRIGHTNESS_MAX):
            return False, "lighting", brightness

        # 3. Blur — Laplacian convolution, most expensive
        if cv2.Laplacian(gray, cv2.CV_64F).var() < BLUR_THRESHOLD:
            return False, "blurry", brightness

        return True, "ok", brightness

# Module-level singletons (single-user prototype — one stream at a time)
quality_monitor     = FrameQualityMonitor()
hr_smoothing_buffer = deque(maxlen=7)  # Median filter — kills single-window spikes


def make_serializable(obj):
    """
    Recursively convert NumPy types and NaN/Inf to JSON-safe Python primitives.
    Required for WebSocket send_json which uses the stdlib encoder (unlike
    FastAPI HTTP endpoints which go through Pydantic and handle NumPy natively).
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return 0.0 if (np.isnan(obj) or np.isinf(obj)) else float(obj)
    elif isinstance(obj, float):
        return 0.0 if (math.isnan(obj) or math.isinf(obj)) else obj
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_serializable(i) for i in obj]
    return obj

def _clamp_hrv(hrv: dict) -> dict:
    """
    Clamp heartpy HRV output to physiologically plausible ranges.
    heartpy returns extreme outliers on marginal signal (IBI=8000ms, SDNN=800ms)
    that pass the SQI gate but corrupt the weighted aggregate.
    Ranges from literature: (40<bpm<150), (400<ibi<2000), (rmssd<100).
    Values outside range are zeroed — parse_result already handles 0.0 gracefully.
    """

    def safe(key, lo, hi):
        v = hrv.get(key)
        if v is None: return 0.0
        f = float(v)
        return f if lo <= f <= hi else 0.0
    
    # 1. Extract raw Hz (e.g., 0.25 Hz)
    raw_br_hz = float(hrv.get("breathingrate") or 0.0)
    
    # 2. Convert to Breaths Per Minute
    br_bpm = raw_br_hz * 60.0
    
    # 3. heartpy search band: 0.1-0.4 Hz = 6-24 br/m (library ceiling)
    raw_br_hz = float(hrv.get("breathingrate") or 0.0)
    br_bpm = raw_br_hz * 60.0    
    clamped_br = br_bpm if 4.0 <= br_bpm <= 24.0 else 0.0 # reduced to 4 to accomodate for shorter videos with less stable respiratory estimates

    return {
        "bpm":           safe("bpm",           40,   180),
        "ibi":           safe("ibi",            400,  2000),
        "sdnn":          safe("sdnn",           0,    200),
        "rmssd":         safe("rmssd",          0,    150),
        "pnn50":         safe("pnn50",          0,    100),
        "LF/HF":         safe("LF/HF",          0,    15),
        "breathingrate": clamped_br,
    }

def extract_face(
    frame_bgr: np.ndarray,
    last_box: tuple | None = None,
) -> tuple[np.ndarray | None, tuple | None]:
    """
    BlazeFace detection → largest face → 10% pad crop → 128×128 RGB.
    Falls back to last_box on miss (same contract as previous Haar version).
    MediaPipe expects RGB input; returns relative bounding box coordinates.
    """
    h, w = frame_bgr.shape[:2]
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    res = _mp_detector.process(frame_rgb)

    box = None
    if res.detections:
        best = max(res.detections, key=lambda d: d.score[0])
        bb   = best.location_data.relative_bounding_box
        x    = max(0, int(bb.xmin * w))
        y    = max(0, int(bb.ymin * h))
        bw   = int(bb.width  * w)
        bh   = int(bb.height * h)
        if bw > 0 and bh > 0:
            box = (x, y, bw, bh)
            logger.debug(f"[FaceDetect] BlazeFace score={best.score[0]:.2f} box={box}")
    
    if box is None:
        if last_box is not None:
            box = last_box
            logger.debug("[FaceDetect] No detection, reusing last_box")
        else:
            return None, None

    x, y, bw, bh = box
    pad_x, pad_y = int(bw * 0.10), int(bh * 0.10)
    x1 = max(0, x - pad_x);    y1 = max(0, y - pad_y)
    x2 = min(w, x + bw + pad_x); y2 = min(h, y + bh + pad_y)

    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return None, None

    face_rgb     = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    face_resized = cv2.resize(face_rgb, (FACE_SIZE, FACE_SIZE))
    return face_resized, (x, y, bw, bh)


def run_model(tensor: np.ndarray, fps: float) -> dict:
    """
    Thread-safe inference via process_faces_tensor.

    We use process_faces_tensor rather than process_video_tensor because:
    - process_video_tensor: expects full frames, runs internal face detection on
      every call, triggers "Tensor mode, video quality check disabled" (the library
      knows full frames may contain non-face content and disables its own checks).
    - process_faces_tensor: expects pre-cropped (T, H, W, 3) face tensors, skips
      internal detection. Our Haar cascade pipeline feeds it exactly this format
      with the background already eliminated.

    RuntimeWarnings from heartpy are suppressed — they fire normally when SQI is
    low and heartpy finds no valid peaks in the BVP waveform.
    """
    if cv_model is None:
        raise RuntimeError("Model not loaded.")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        with model_lock:
            result = cv_model.process_faces_tensor(tensor, fps=fps)
            if result:
                result["hrv"] = _clamp_hrv(result.get("hrv") or {})
                logger.debug(
                    f"[Model] SQI={float(result.get('SQI') or 0):.3f} | "
                    f"HR={result.get('hr')} | HRV keys: {list(result['hrv'].keys())}"
                )
            return result        

def parse_result(
    result: dict,
    elapsed_ms: float,
    idx: int,
    t_start: float,
    fps: float = TARGET_FPS,
    win_frames: int = WINDOW_FRAMES,
) -> ChunkMetrics:
    """
    Unpack all model output fields into a typed schema.

    fps and win_frames are used to calculate time_end_s correctly — the adaptive
    windowing path may use a win_frames value smaller than WINDOW_FRAMES for short
    videos, so hardcoding WINDOW_FRAMES here would produce incorrect timestamps.
    """
    hrv   = result.get("hrv") or {}
    chunk = ChunkMetrics(
        chunk_index        = idx,
        time_start_s       = round(t_start, 1),
        time_end_s         = round(t_start + (win_frames / fps), 1),
        bpm_fft            = round(float(result.get("hr") or 0.0), 1),
        bpm_peak           = round(float(hrv.get("bpm") or 0.0), 1),
        sqi                = round(float(result.get("SQI") or 0.0), 4),
        respiratory_rate   = round(float(hrv.get("breathingrate") or 0.0), 1),
        hrv_ibi            = round(float(hrv.get("ibi") or 0.0), 1),
        hrv_sdnn           = round(float(hrv.get("sdnn") or 0.0), 2),
        hrv_rmssd          = round(float(hrv.get("rmssd") or 0.0), 2),
        hrv_pnn50          = round(float(hrv.get("pnn50") or 0.0), 2),
        hrv_lf_hf          = round(float(hrv.get("LF/HF") or 0.0), 4),
        processing_time_ms = round(elapsed_ms, 1),
    )
    logger.info(
        f"[Chunk {idx}] {chunk.time_start_s:.0f}–{chunk.time_end_s:.0f}s | "
        f"BPM {chunk.bpm_fft} | SQI {chunk.sqi:.3f} | RR {chunk.respiratory_rate} | "
        f"SDNN {chunk.hrv_sdnn} ms | RMSSD {chunk.hrv_rmssd} ms | "
        f"LF/HF {chunk.hrv_lf_hf} | {elapsed_ms:.0f}ms"
    )
    return chunk


def aggregate_chunks(chunks: list[ChunkMetrics], total_ms: float) -> AggregateResult:
    """SQI-weighted aggregation. Chunks below threshold are excluded entirely."""
    valid = [c for c in chunks if c.sqi >= SQI_THRESHOLD]
    logger.info(
        f"[Aggregate] {len(valid)}/{len(chunks)} chunks passed SQI≥{SQI_THRESHOLD}"
    )

    # Adaptive gate — if the video was short enough to produce only 1–2 chunks,
    # don't require MIN_VALID_CHUNKS chunks to return a result.
    required_chunks = min(MIN_VALID_CHUNKS, len(chunks))

    if len(valid) < required_chunks or len(valid) == 0:
        return AggregateResult(
            final_bpm=0, final_rr=0, avg_sqi=0, bpm_std=0,
            agg_hrv_sdnn=0, agg_hrv_rmssd=0, agg_hrv_lf_hf=0,
            chunks_total=len(chunks), chunks_valid=len(valid),
            total_time_ms=round(total_ms, 1),
            message=f"Insufficient valid data. {len(valid)}/{len(chunks)} chunks passed SQI gate."
        )

    total_w = sum(c.sqi for c in valid)
    def wavg(attr: str) -> float:
        return sum(getattr(c, attr) * c.sqi for c in valid) / total_w

    bpms = [c.bpm_fft for c in valid]
    agg  = AggregateResult(
        final_bpm     = round(wavg("bpm_fft"), 1),
        final_rr      = round(wavg("respiratory_rate"), 1),
        avg_sqi       = round(total_w / len(valid), 4),
        bpm_std       = round(float(np.std(bpms)), 2),
        agg_hrv_sdnn  = round(wavg("hrv_sdnn"), 2),
        agg_hrv_rmssd = round(wavg("hrv_rmssd"), 2),
        agg_hrv_lf_hf = round(wavg("hrv_lf_hf"), 4),
        chunks_total  = len(chunks),
        chunks_valid  = len(valid),
        total_time_ms = round(total_ms, 1),
        message       = "OK",
    )
    logger.info(
        f"[Aggregate] BPM={agg.final_bpm} | RR={agg.final_rr} | "
        f"SQI={agg.avg_sqi} | SDNN={agg.agg_hrv_sdnn} | "
        f"RMSSD={agg.agg_hrv_rmssd} | LF/HF={agg.agg_hrv_lf_hf} | "
        f"total={total_ms:.0f}ms"
    )
    return agg


def process_video_file_sync(path: str) -> tuple[list[ChunkMetrics], float]:
    """
    Memory-safe streaming processor for the file upload path.

    Reads frames one-by-one (never allocates a full-video tensor in RAM),
    extracts 128x128 face crops, accumulates chunks of win_frames crops,
    runs process_faces_tensor per chunk, clears the chunk tensor immediately.

    Full frame buffer at 640x480: 600 x 640 x 480 x 3 = 552 MB.
    Face crop buffer at 128x128:  600 x 128 x 128 x 3 =  37 MB.
    This ~15x reduction is what keeps the pipeline within typical server limits.
    """
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = TARGET_FPS

    # ── Adaptive window ───────────────────────────────────────────────────────
    # If the video is shorter than the ideal 20s window, shrink to fit.
    # Floor is 5 seconds (150 frames) — below that, FFT resolution is too poor.
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_video_frames < WINDOW_FRAMES:
        win_frames = max(150, total_video_frames - 1)
        logger.info(
            f"[FileProcess] Video shorter than 20s window. "
            f"Adapting to {win_frames} frames ({win_frames/fps:.1f}s)."
        )
    else:
        win_frames = WINDOW_FRAMES

    if win_frames < 150:
        cap.release()
        raise ValueError(
            f"Video is too short ({total_video_frames} frames at {fps:.1f} fps). "
            "Need at least 5 seconds."
        )

    logger.info(
        f"[FileProcess] {fps:.1f} fps | chunk_size={win_frames} frames "
        f"| window={win_frames/fps:.1f}s"   # uses actual win_frames, not WINDOW_FRAMES
    )

    chunk_results: list[ChunkMetrics] = []
    chunk_buffer  = deque(maxlen=win_frames)   
    frames_since_infer = 0
    chunk_index   = 0
    CHUNK_STRIDE  = win_frames // 2            # 50% overlap — 300 frames = 10s stride
    last_box     = None
    total_frames = 0
    dropped      = 0

    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            total_frames += 1

            face_rgb, box = extract_face(frame_bgr, last_box)
            if box is not None:
                last_box = box

            if face_rgb is None:
                dropped += 1
                logger.debug(f"[FileProcess] Frame {total_frames}: no face, skipping")
                continue

            chunk_buffer.append(face_rgb)
            frames_since_infer += 1

            if len(chunk_buffer) == win_frames and frames_since_infer >= CHUNK_STRIDE:
                tensor    = np.stack(chunk_buffer).astype(np.uint8)
                t_start_s = (chunk_index * CHUNK_STRIDE) / fps
                t0        = time.time()
                try:
                    result     = run_model(tensor, fps)
                    elapsed_ms = (time.time() - t0) * 1000
                    if result and "hr" in result:
                        chunk_results.append(
                            parse_result(result, elapsed_ms, chunk_index, t_start_s, fps, win_frames)
                        )
                except Exception as e:
                    logger.warning(f"[FileProcess] Chunk {chunk_index} failed: {e}")
                del tensor
                frames_since_infer = 0
                chunk_index += 1

    finally:
        cap.release()

    logger.info(
        f"[FileProcess] Complete — {total_frames} frames | {dropped} dropped | "
        f"{chunk_index} chunks | {len(chunk_results)} with results"
    )
    return chunk_results, float(fps)


def process_incoming_frame(
    base64_str: str,
    last_box: tuple | None,
) -> tuple[np.ndarray | None, tuple | None, int, float]:
    """
    CPU-bound JPEG decode + face extraction, offloaded from the async event loop.
    Returns (face_rgb_128x128, box, decoded_bytes, face_brightness).
    Brightness is measured on the face crop so background exposure shifts
    do not trigger the lighting quality gate.
    """
    try:
        img_bytes   = base64.b64decode(base64_str)
        decoded_len = len(img_bytes)

        np_arr    = np.frombuffer(img_bytes, np.uint8)
        frame_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame_bgr is None:
            return None, None, decoded_len, 0.0

        face_rgb, box = extract_face(frame_bgr, last_box)

        if face_rgb is None:
            return None, None, decoded_len, 0.0

        mean_brightness = float(cv2.cvtColor(face_rgb, cv2.COLOR_RGB2GRAY).mean())
        logger.debug(
            f"[IncomingFrame] {decoded_len}B | brightness={mean_brightness:.1f} | box={box}"
        )
        return face_rgb, box, decoded_len, mean_brightness

    except Exception as e:
        logger.debug(f"[IncomingFrame] exception: {e}")
        return None, None, 0, 0.0


# ─────────────────────────────────────────────────
# File Upload Endpoint
# ─────────────────────────────────────────────────
@app.post("/analyze", response_model=AnalysisResponse, summary="Analyze uploaded video")
async def analyze_video(file: UploadFile = File(...)):
    """
    Streaming face-crop pipeline for file upload.

    Processes video frame-by-frame (never holds a full-file tensor in RAM),
    extracts 128x128 face crops, builds non-overlapping chunks, runs inference
    per chunk, then SQI-weighted aggregates across all valid chunks.
    """
    t0  = time.time()
    ext = (os.path.splitext(file.filename or "upload.mp4")[1].lower()) or ".mp4"

    if ext not in {".mp4", ".webm", ".mov", ".avi", ".mkv"}:
        raise HTTPException(415, f"Unsupported file type: {ext}")

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    file_mb = os.path.getsize(tmp_path) / 1024 / 1024
    logger.info(f"[Upload] {file.filename} ({file_mb:.1f} MB)")

    try:
        if os.path.getsize(tmp_path) < 50_000:
            raise HTTPException(400, "File too small — ensure the video is at least 5 seconds.")

        loop = asyncio.get_running_loop()

        try:
            chunk_results, fps = await loop.run_in_executor(
                executor, process_video_file_sync, tmp_path
            )
        except ValueError as e:
            raise HTTPException(422, str(e))

        if not chunk_results:
            raise HTTPException(
                422,
                "No signal extracted. Ensure good frontal lighting, minimal movement, "
                "and a clearly visible face throughout the video."
            )

        total_ms = (time.time() - t0) * 1000
        agg = aggregate_chunks(chunk_results, total_ms)
        return AnalysisResponse(chunks=chunk_results, aggregate=agg)

    finally:
        os.unlink(tmp_path)


# ─────────────────────────────────────────────────
# WebSocket Streaming Endpoint — O(1) Face-Crop Mode
# ─────────────────────────────────────────────────
@app.websocket("/ws/stream")
async def stream_endpoint(ws: WebSocket):
    """
    O(1) sliding-window inference over a live webcam feed.

    Frame flow per received message:
      1. Decode base64 JPEG → cv2.imdecode (thread pool)
      2. BlazeFace (MediaPipe) → largest face → 128x128 RGB crop
      3. FrameQualityMonitor gate (blur, brightness, motion — on face crop)
      4. Append to deque(maxlen=WINDOW_FRAMES) — oldest frame auto-discarded
      5. Every STEP_FRAMES: np.stack → process_faces_tensor
      6. Median-smooth HR, EMA for display, send JSON

    Buffer memory: WINDOW_FRAMES × 128 × 128 × 3 = ~37 MB, constant throughout session.

    NOTE: model.video_capture() cannot be used because it opens a camera device
    directly on the server, which does not exist in a headless container. The
    canvas-JPEG WebSocket approach replicates its function: the browser is the
    capture device, the backend is the inference engine.
    """
    await ws.accept()
    logger.info("[Stream] Client connected — O(1) face-crop mode")

    frame_buffer            = deque(maxlen=WINDOW_FRAMES)
    frames_since_last_infer = 0
    ema_bpm                 = EMA(alpha=EMA_ALPHA)   # HR: fast response
    ema_rr                  = EMA(alpha=0.30)         # RR: slower — respiratory cycle is longer
    loop                    = asyncio.get_running_loop()
    last_box: tuple | None  = None
    inference_task = None

    try:
        while True:
            data = await ws.receive_text()

            if not data.startswith("data:image/jpeg;base64,"):
                continue

            # ── 1. Decode + face crop ────────────────────────────────────────
            face_rgb, box, decoded_len, brightness = await loop.run_in_executor(
                executor, process_incoming_frame, data.split(",")[1], last_box
            )

            if box is not None:
                last_box = box

            if face_rgb is None:
                await ws.send_json({
                    "status": "low_signal", "reason": "no_face", "sqi": 0.0
                })
                continue

            # ── 2. Quality gate (on face crop only) ──────────────────────────
            is_valid, reason, brightness = quality_monitor.check(face_rgb)
            exposure_drifting = quality_monitor.brightness_drift(brightness)   # ← use brightness, not captured_brightness
            if not is_valid:
                await ws.send_json({
                    "status": "low_signal", "reason": reason, "sqi": 0.0
                })
                continue

            # ── 3. Rolling buffer ────────────────────────────────────────────
            frame_buffer.append(face_rgb)
            frames_since_last_infer += 1

            # ── 4. Warmup ────────────────────────────────────────────────────
            if len(frame_buffer) < WINDOW_FRAMES:
                if frames_since_last_infer % 15 == 0:
                    buffered = len(frame_buffer)
                    logger.debug(f"[Stream] Buffering {buffered}/{WINDOW_FRAMES}")
                    await ws.send_json({
                        "status":    "buffering",
                        "buffered":  buffered,
                        "target":    WINDOW_FRAMES,
                        "box":       last_box 
                    })
                continue

            # ── 5. Sliding-step gate ─────────────────────────────────────────
            if frames_since_last_infer < STEP_FRAMES:
                continue
            frames_since_last_infer = 0

            # ── 6. Infer (Non-Blocking) ──────────────────────────────────────
            if inference_task is None or inference_task.done():
                
                current_tensor = np.stack(frame_buffer).astype(np.uint8)
                captured_brightness = brightness 
                captured_buffered_len = len(frame_buffer)
                captured_box = last_box
                captured_drift        = exposure_drifting    
                
                async def infer_and_send(tensor, bright, buf_len, box, drift): 
                    try:
                        t0 = time.time()
                        
                        result = await loop.run_in_executor(
                            executor, run_model, tensor, float(TARGET_FPS)
                        )
                        
                        inf_ms = (time.time() - t0) * 1000
                        speed_fps = WINDOW_FRAMES / (inf_ms / 1000.0) if inf_ms > 0 else 0
                        
                        sqi    = result.get("SQI") or 0.0
                        hrv    = result.get("hrv") or {}
                        raw_hr = result.get("hr")

                        logger.info(
                            f"[Stream] {inf_ms:.0f}ms | {speed_fps:.0f} FPS | "
                            f"SQI={float(sqi):.2f} | HR={float(raw_hr) if raw_hr is not None else 0.0:.1f} | "
                            f"face_brightness={bright:.1f}"
                        )

                        # Median HR smoothing
                        if sqi > SQI_THRESHOLD and raw_hr is not None:
                            new_hr = float(raw_hr)
                            _prev  = list(hr_smoothing_buffer)
                            prev_median = float(np.median(_prev)) if _prev else None

                            # Consistency gate — reject if >15 BPM jump from last stable median
                            if prev_median is not None and abs(new_hr - prev_median) > 15.0:
                                logger.debug(f"[Stream] HR spike rejected: {new_hr:.1f} vs {prev_median:.1f} BPM")
                                hr_num = prev_median   # hold last stable value, don't pollute buffer
                            else:
                                hr_smoothing_buffer.append(new_hr)
                                hr_num = float(np.median(hr_smoothing_buffer))
                        else:
                            hr_num = float(hr_smoothing_buffer[-1]) if hr_smoothing_buffer else 0.0

                        # Signal gate
                        if not result or raw_hr is None:
                            await ws.send_json({"status": "no_signal", "box": box})
                            return

                        if sqi < SQI_THRESHOLD:
                            # Send the box back even on low signal so the UI keeps tracking
                            await ws.send_json(
                                make_serializable({"status": "low_signal", "sqi": round(float(sqi), 4), "box": box})
                            )
                            return

                        # Send payload
                        payload = {
                            "status":          "ok",
                            "bpm":             ema_bpm.update(hr_num),
                            "raw_bpm":         round(hr_num, 1),
                            "bpm_peak":        round(float(hrv.get("bpm") or 0.0), 1),
                            "sqi":             round(float(sqi), 4),
                            "rr":              ema_rr.update(float(hrv.get("breathingrate") or 0.0)),
                            "hrv_ibi":         round(float(hrv.get("ibi") or 0.0), 1),
                            "hrv_sdnn":        round(float(hrv.get("sdnn") or 0.0), 2),
                            "hrv_rmssd":       round(float(hrv.get("rmssd") or 0.0), 2),
                            "hrv_pnn50":       round(float(hrv.get("pnn50") or 0.0), 2),
                            "hrv_lf_hf":       round(float(hrv.get("LF/HF") or 0.0), 4),
                            "buffered_frames": buf_len,
                            "box":             box,                                  
                        }
                        if drift:
                            payload["warning"] = "exposure_drift"
                        await ws.send_json(make_serializable(payload))

                    except RuntimeError as e:
                        if "Unexpected ASGI message" in str(e) or "websocket.close" in str(e):
                            # The user disconnected while the ML model was thinking. 
                            # Silently discard the result.
                            pass 
                        else:
                            logger.error(f"[Stream] Async inference RuntimeError: {e}")

                    except Exception as e:
                        logger.error(f"[Stream] Async inference error: {e}")

                inference_task = asyncio.create_task(
                    infer_and_send(current_tensor, captured_brightness, captured_buffered_len, captured_box, captured_drift)
                )
            
            else:
                logger.debug("[Stream] Inference lagging, skipping UI update cycle.")

    except WebSocketDisconnect:
        logger.info("[Stream] Client disconnected")
    except Exception as e:
        logger.error(f"[Stream] Fatal error: {e}")

# ─────────────────────────────────────────────────
# Utility Routes
# ─────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {
        "status":     "ok",
        "model":      "FacePhys.rlap",
        "inference":  "process_faces_tensor",
        "target_fps": TARGET_FPS,
        "window_s":   round(WINDOW_FRAMES / TARGET_FPS, 1),
        "face_size":  FACE_SIZE,
    }


@app.get("/")
async def root():
    return FileResponse("static/index.html")


# ─────────────────────────────────────────────────
# Entry Point (local dev — production uses CMD in Dockerfile)
# ─────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)