# rPPG_bpm/app.py

"""
Wise AI — rPPG Vitals Monitor
Full-stack integration of the open-rppg library with dual-path inference:
  - /analyze   : file upload → chunk-level + aggregate metrics
  - /ws/stream : WebSocket live streaming → O(1) sliding-window inference
"""

import asyncio
import io
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

import av
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
logger = logging.getLogger("wise-rppg")


# ─────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────
TARGET_FPS       = 30    # Webcam target frame rate
WINDOW_FRAMES    = 150   # 5-second inference window (150 / 30 fps)
STEP_FRAMES      = 15    # Inference every 0.5 s (15 / 30 fps)
SQI_THRESHOLD    = 0.40  # Signal quality gate — below this, readings are unreliable
MIN_VALID_CHUNKS = 2      # Minimum chunks required for a valid aggregate (file upload)
EMA_ALPHA        = 0.15  # EMA smoother weight for live BPM display

# ─────────────────────────────────────────────────
# Quality Thresholds (Frame-level gate, live stream only)
# ─────────────────────────────────────────────────
BLUR_THRESHOLD    = 30.0   # Laplacian variance — lower = blurrier
BRIGHTNESS_MIN    = 40.0   # Mean pixel value floor
BRIGHTNESS_MAX    = 220.0  # Mean pixel value ceiling
MOTION_THRESHOLD  = 25.0   # Mean absolute frame diff

# ─────────────────────────────────────────────────
# Global model — warm, single instance
# ─────────────────────────────────────────────────
cv_model: rppg.Model | None = None
model_lock = Lock()
executor   = ThreadPoolExecutor(max_workers=2)


# ─────────────────────────────────────────────────
# Lifespan — model loaded once on startup
# ─────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global cv_model
    logger.info("Loading FacePhys.rlap model …")
    cv_model = rppg.Model("FacePhys.rlap")
    logger.info("Model ready.")
    yield
    logger.info("Shutting down.")


# ─────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────
app = FastAPI(
    title="Wise AI — rPPG Vitals Monitor",
    description="Camera-based contactless vital sign measurement prototype.",
    version="1.0.0",
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
    """Exponential Moving Average — smooths live BPM display."""
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
    Per-frame gate that drops visually bad frames before they enter the buffer.
    Catches blur, under/over-exposure, and sudden motion — each of which looks
    like a heartbeat spike to the rPPG model.
    """
    def __init__(self):
        self.prev_gray = None

    def check(self, frame_rgb: np.ndarray) -> tuple[bool, str, float]:
        """Returns (is_valid, reason, brightness)."""
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)

        brightness = float(np.mean(gray))
        if not (BRIGHTNESS_MIN < brightness < BRIGHTNESS_MAX):
            return False, "lighting", brightness

        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        if blur_score < BLUR_THRESHOLD:
            return False, "blurry", brightness

        if self.prev_gray is not None:
            diff = cv2.absdiff(self.prev_gray, gray)
            if float(np.mean(diff)) > MOTION_THRESHOLD:
                self.prev_gray = gray
                return False, "motion", brightness

        self.prev_gray = gray
        return True, "ok", brightness


# Module-level singletons (single-user prototype — one stream at a time)
quality_monitor    = FrameQualityMonitor()
hr_smoothing_buffer = deque(maxlen=5)  # median filter to suppress inference spikes


def make_serializable(obj):
    """
    Recursively convert NumPy primitives and NaN/Inf into JSON-safe Python types.
    WebSocket send_json uses the stdlib json encoder which doesn't know NumPy types
    (unlike FastAPI's Pydantic layer used by HTTP endpoints).
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        if np.isnan(obj) or np.isinf(obj):
            return 0.0
        return float(obj)
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return 0.0
        return obj
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_serializable(i) for i in obj]
    return obj


def run_model(tensor: np.ndarray, fps: float) -> dict:
    """Thread-safe model inference. RuntimeWarnings from heartpy on noisy data are suppressed."""
    if cv_model is None:
        raise RuntimeError("Model not loaded — startup may have failed.")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        with model_lock:
            return cv_model.process_video_tensor(tensor, fps=fps)


def parse_result(result: dict, elapsed_ms: float, idx: int, t_start: float) -> ChunkMetrics:
    """Unpack every metric the library provides into a typed schema."""
    hrv = result.get("hrv") or {}
    
    return ChunkMetrics(
        chunk_index        = idx,
        time_start_s       = round(t_start, 1),
        time_end_s         = round(t_start + (WINDOW_FRAMES / TARGET_FPS), 1),
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

def aggregate_chunks(chunks: list[ChunkMetrics], total_ms: float) -> AggregateResult:
    """SQI-weighted aggregation — down-weights noisy windows automatically."""
    valid = [c for c in chunks if c.sqi >= SQI_THRESHOLD]

    if len(valid) < MIN_VALID_CHUNKS:
        return AggregateResult(
            final_bpm=0, final_rr=0, avg_sqi=0, bpm_std=0,
            agg_hrv_sdnn=0, agg_hrv_rmssd=0, agg_hrv_lf_hf=0,
            chunks_total=len(chunks), chunks_valid=len(valid),
            total_time_ms=round(total_ms, 1),
            message="Insufficient signal quality — check lighting and face framing.",
        )

    total_w = sum(c.sqi for c in valid)

    def wavg(attr: str) -> float:
        return sum(getattr(c, attr) * c.sqi for c in valid) / total_w

    bpms = [c.bpm_fft for c in valid]

    return AggregateResult(
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

def process_video_file_sync(path: str) -> tuple[list[ChunkMetrics], float]:
    """
    Reads the video file chunk-by-chunk to keep memory usage flat.
    Yields parsed chunk metrics and actual FPS.
    """
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = TARGET_FPS

    win_frames = int(fps * (WINDOW_FRAMES / TARGET_FPS))
    if win_frames < 30:
        cap.release()
        raise ValueError("Video FPS too low for reliable rPPG analysis.")

    chunk_results = []
    current_chunk_frames = []
    chunk_index = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            
            current_chunk_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # When we hit exactly the window size, run inference and clear memory
            if len(current_chunk_frames) == win_frames:
                window_tensor = np.stack(current_chunk_frames).astype(np.uint8)
                t_start_s = chunk_index * (win_frames / fps)
                t_infer_start = time.time()
                
                try:
                    # Run model synchronously (we are already in a ThreadPool)
                    result = run_model(window_tensor, fps)
                    elapsed_ms = (time.time() - t_infer_start) * 1000
                    
                    if result and "hr" in result:
                        parsed = parse_result(result, elapsed_ms, chunk_index, t_start_s)
                        chunk_results.append(parsed)
                        logger.info(
                            f"  [Chunk {chunk_index}] {parsed.time_start_s:.0f}–{parsed.time_end_s:.0f}s | "
                            f"BPM {parsed.bpm_fft:.1f} | SQI {parsed.sqi:.3f} | {elapsed_ms:.0f} ms"
                        )
                except Exception as e:
                    logger.warning(f"Chunk {chunk_index} failed: {e}")
                
                # CLEAR memory for the next chunk
                current_chunk_frames.clear() 
                chunk_index += 1

    finally:
        cap.release()

    return chunk_results, float(fps)


def process_incoming_frame(base64_str: str) -> tuple[np.ndarray | None, int, float]:
    """
    CPU-bound image decoding offloaded from the async event loop.
    Returns (rgb_frame, decoded_byte_len, mean_brightness).
    """
    try:
        img_bytes = base64.b64decode(base64_str)
        decoded_len = len(img_bytes)

        np_arr    = np.frombuffer(img_bytes, np.uint8)
        frame_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame_bgr is None:
            return None, decoded_len, 0.0

        frame_rgb       = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mean_brightness = float(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).mean())

        return frame_rgb, decoded_len, mean_brightness
    except Exception:
        return None, 0, 0.0


# ─────────────────────────────────────────────────
# File Upload Endpoint
# ─────────────────────────────────────────────────
@app.post("/analyze", response_model=AnalysisResponse, summary="Analyze uploaded video")
async def analyze_video(file: UploadFile = File(...)):
    """
    Accepts any face video. Uses a memory-safe streaming generator to prevent OOM 
    crashes on large files, processes chunks, and aggregates results.
    """
    t0  = time.time()
    ext = os.path.splitext(file.filename or "upload.mp4")[1].lower() or ".mp4"

    if ext not in {".mp4", ".webm", ".mov", ".avi", ".mkv"}:
        raise HTTPException(415, f"Unsupported file type: {ext}")

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        if os.path.getsize(tmp_path) < 50_000:
            raise HTTPException(400, "File too small — ensure the video is at least 5 seconds.")

        loop = asyncio.get_running_loop()

        try:
            # Offload the entire disk-read and processing loop to the thread executor
            chunk_results, fps = await loop.run_in_executor(
                executor, process_video_file_sync, tmp_path
            )
        except ValueError as ve:
             raise HTTPException(422, str(ve))

        if not chunk_results:
            raise HTTPException(422, "No valid signal extracted. Ensure good lighting, no movement, and a clear face.")

        total_ms = (time.time() - t0) * 1000
        agg = aggregate_chunks(chunk_results, total_ms)

        logger.info(f"Analysis complete: {agg.final_bpm} BPM | SQI {agg.avg_sqi:.3f} | {total_ms:.0f} ms")
        return AnalysisResponse(chunks=chunk_results, aggregate=agg)

    finally:
        os.unlink(tmp_path)

# ─────────────────────────────────────────────────
# WebSocket Streaming Endpoint — O(1) Frame Mode
# ─────────────────────────────────────────────────
@app.websocket("/ws/stream")
async def stream_endpoint(ws: WebSocket):
    """
    O(1) sliding-window inference over a live webcam feed.

    The frontend sends individual JPEG frames (base64 data-URL) captured via
    requestVideoFrameCallback, which hardware-syncs to the camera clock.
    The backend maintains a deque(maxlen=WINDOW_FRAMES) that automatically
    discards the oldest frame on each new arrival — no byte accumulation,
    no growing decode cost, flat memory throughout the session.

    This replaces the earlier PyAV WebM fragment accumulation approach, which
    was O(n) in decode cost: at 50s the server was decoding 50s of video just
    to extract the last 150 frames, causing the inference loop to fall behind
    real-time.
    """
    await ws.accept()
    logger.info("WebSocket client connected (O(1) Frame Mode)")

    frame_buffer             = deque(maxlen=WINDOW_FRAMES)
    frames_since_last_infer  = 0
    ema_bpm                  = EMA()
    ema_rr                   = EMA()
    loop                     = asyncio.get_running_loop()

    try:
        while True:
            data = await ws.receive_text()

            if not data.startswith("data:image/jpeg;base64,"):
                continue

            # ── 1. Decode frame (off the event loop) ──
            frame_rgb, decoded_len, mean_brightness = await loop.run_in_executor(
                executor, process_incoming_frame, data.split(",")[1]
            )

            logger.debug(f"Frame | bytes: {decoded_len} | brightness: {mean_brightness:.1f}")

            if frame_rgb is None:
                continue

            # ── 2. Frame-quality gate ──
            is_valid, reason, _ = quality_monitor.check(frame_rgb)
            if not is_valid:
                await ws.send_json({"status": "low_signal", "reason": reason, "sqi": 0.0})
                continue

            # ── 3. Append to rolling window ──
            frame_buffer.append(frame_rgb)
            frames_since_last_infer += 1

            # ── 4. Buffer warmup ──
            if len(frame_buffer) < WINDOW_FRAMES:
                if frames_since_last_infer % 15 == 0:
                    await ws.send_json({
                        "status":   "buffering",
                        "buffered": len(frame_buffer),
                        "target":   WINDOW_FRAMES,
                    })
                continue

            # ── 5. Sliding-step gate ──
            if frames_since_last_infer < STEP_FRAMES:
                continue

            frames_since_last_infer = 0

            # ── 6. Build tensor and run inference ──
            t0 = time.time()
            current_tensor = np.stack(frame_buffer).astype(np.uint8)

            try:
                result = await loop.run_in_executor(
                    executor, run_model, current_tensor, float(TARGET_FPS)
                )
            except Exception as e:
                logger.warning(f"Stream inference error: {e}")
                await ws.send_json({"status": "error", "detail": str(e)})
                continue

            inf_ms    = (time.time() - t0) * 1000
            speed_fps = WINDOW_FRAMES / (inf_ms / 1000.0) if inf_ms > 0 else 0

            # ── 7. Safe extraction (model may return explicit None for SQI/hr) ──
            sqi    = result.get("SQI") or 0.0   # `or 0.0` handles explicit None
            hrv    = result.get("hrv") or {}
            raw_hr = result.get("hr")

            logger.info(
                f"[Metrics] Infer: {inf_ms:.0f}ms | Speed: {speed_fps:.0f} FPS | "
                f"SQI: {float(sqi):.2f} | HR: {float(raw_hr) if raw_hr is not None else 0.0:.1f}"
            )

            # ── 8. HR smoothing (median filter kills inference spikes) ──
            if sqi > SQI_THRESHOLD and raw_hr is not None:
                hr_smoothing_buffer.append(float(raw_hr))
                hr_num = float(np.median(hr_smoothing_buffer))
            else:
                hr_num = float(hr_smoothing_buffer[-1]) if hr_smoothing_buffer else 0.0

            # ── 9. Signal quality gate ──
            if not result or raw_hr is None:
                await ws.send_json({"status": "no_signal"})
                continue

            if sqi < SQI_THRESHOLD:
                await ws.send_json(make_serializable({"status": "low_signal", "sqi": round(float(sqi), 4)}))
                continue

            # ── 10. Build and send payload ──
            payload = {
                "status":          "ok",
                "bpm":             ema_bpm.update(hr_num),   # EMA of median-smoothed HR
                "raw_bpm":         round(hr_num, 1),          # median-smoothed (pre-EMA)
                "bpm_peak":        round(float(hrv.get("bpm") or 0.0), 1),
                "sqi":             round(float(sqi), 4),
                "rr":              ema_rr.update(float(hrv.get("breathingrate") or 0.0)),
                "hrv_ibi":         round(float(hrv.get("ibi") or 0.0), 1),
                "hrv_sdnn":        round(float(hrv.get("sdnn") or 0.0), 2),
                "hrv_rmssd":       round(float(hrv.get("rmssd") or 0.0), 2),
                "hrv_pnn50":       round(float(hrv.get("pnn50") or 0.0), 2),
                "hrv_lf_hf":       round(float(hrv.get("LF/HF") or 0.0), 4),
                "buffered_frames": len(frame_buffer),
            }

            await ws.send_json(make_serializable(payload))

            await ws.send_json(make_serializable(payload))

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"Stream fatal error: {e}")


# ─────────────────────────────────────────────────
# Utility Routes
# ─────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {
        "status":    "ok",
        "model":     "FacePhys.rlap",
        "target_fps": TARGET_FPS,
        "window_s":  round(WINDOW_FRAMES / TARGET_FPS, 1),
    }


@app.get("/")
async def root():
    return FileResponse("static/index.html")


# ─────────────────────────────────────────────────
# Entry Point (local dev only — production uses CMD in Dockerfile)
# ─────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)