# rPPG Vitals Monitor

> **Scope** · Timeboxed technical assessment, built to a brief over ~5 days against a research rPPG library from the company that develops the technology. Not maintained since.

> [!WARNING]
> **Not a medical device.** This is an engineering project exploring remote
> photoplethysmography. Its outputs are estimates from video, sensitive to lighting,
> motion and skin tone, and are not validated against clinical reference standards.
> Do not use it for diagnosis, monitoring a health condition, or any decision about
> medical care. Consult a qualified clinician and use approved equipment.

Camera-based contactless heart rate, respiratory rate, and HRV from a standard webcam — no wearables, no contact.

*Note: Live cloud deployment on free tiers (like Render) is not recommended due to ML model RAM constraints. See local/Docker instructions below.*

---

## Quick Start

**Local CLI (native pipeline, best accuracy):**

```bash
git clone https://github.com/JoshPola96/rppg-monitor.git
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
* **Respiratory rate** — derived from RSA modulation of the BVP waveform; indicative, not clinical
* **HRV:** IBI, SDNN, RMSSD, pNN50, LF/HF ratio

Two input paths are supported: live webcam stream (WebSocket) or file upload (POST).

---

## Camera and Positioning Requirements

Signal quality depends almost entirely on two physical inputs the software cannot control: the light hitting your face, and your distance from the lens. The model extracts pulse from the green channel of skin pixels across the forehead and upper cheek regions. Anything that masks, shadows, or noisily modulates that channel will degrade SQI.

**What works:**

* Face the light source directly. A window in front of you, a lamp aimed at your face, or a bright monitor all work. A window or bright surface behind you creates silhouette conditions that suppress the signal.
* Sit close enough that your face fills roughly a third to half of the frame. BlazeFace needs a minimum detection confidence of 0.5; once detected, the pipeline crops to 128×128 for inference.
* Sit upright, chin level, face forward. The forehead and cheeks are the primary ROI. When the head tilts down the forehead foreshortens in the 2D frame and the brow casts shadows across the cheek — both effects reduce the pixel area the model can use.
* Stay still. Slow postural sway is tolerated; talking, laughing, or a hand crossing the frame will be caught by the motion gate and those frames will be dropped.

**What the UI does and does not tell you:**
The SQI bar, dynamic tracking bounding box, and face-frame colour indicator give real-time signal quality feedback: green means the model has a clean lock, amber is marginal, red means the current window is too noisy to trust. Specific drop reasons (blurry, motion, no face detected, lighting out of range) are surfaced as status messages when the quality gate rejects frames. An amber banner fires when the system detects webcam auto-exposure drifting — the single biggest SQI destabiliser in typical indoor conditions.

Hardcoded posture-correction overlays were considered and explicitly dropped. A guide that says "chin up" relative to a fixed pixel position would give instructions entirely inconsistent with the actual SQI — it cannot know whether the problem is head angle, lighting, distance, or exposure drift. The tracking box dynamically projects the server-side BlazeFace detection boundary onto the live video feed so you always know what the model is looking at, while the SQI bar and colour feedback convey real signal quality.

---

## Two Systems: What They Are and Why Both Exist

This project has two distinct implementations that share the same model weights but differ fundamentally in how they interact with the camera and the inference pipeline.

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
    │  JPEG frames over WebSocket (~30 fps, quality=0.85)
    ▼
FastAPI /ws/stream
    │
    ├── cv2.imdecode (JPEG → BGR)
    ├── MediaPipe BlazeFace detection → largest face → 10% pad crop → 128×128 RGB
    ├── FrameQualityMonitor gate (motion → brightness → blur, on face crop only)
    │     motion (absdiff), brightness (mean), blur (Laplacian variance)
    │     auto-exposure drift detection (brightness range over 15-reading window)
    ├── deque(maxlen=600) rolling buffer (replicates library's signal buffer)
    │
    ├── Every STEP_FRAMES: np.stack → process_faces_tensor(tensor, fps) (Async)
    │     ↑ pre-cropped tensor path — skips library's internal detection
    │     ↑ note: "Tensor mode, video quality check disabled" fires here;
    │       that is expected — our FrameQualityMonitor is the replacement
    │
    ├── HR temporal consistency gate (rejects >15 BPM jumps as physiologically impossible)
    ├── Median filter (7-window, kills single-inference spikes)
    ├── EMA smoothing (α=0.70 for HR, α=0.30 for RR)
    ├── HRV sanity clamp (_clamp_hrv — rejects heartpy outliers outside physiological bounds)
    └── JSON → WebSocket → browser
```

For file upload (`/analyze`), `process_video_file_sync` reads frames one by one with OpenCV, extracts 128×128 face crops, accumulates a sliding deque with **50% stride overlap**, and calls `process_faces_tensor` per chunk. It never allocates a full-video tensor — critical for staying within strict memory limits:

* **Full frame buffer at 640×480:** `600 × 640 × 480 × 3 = 552 MB` (OOM on free tiers)
* **Face crop buffer at 128×128:** `600 × 128 × 128 × 3 = 37 MB` (Safe)

The 50% stride (300 frames = 10 seconds) means a 60-second video produces 5–6 overlapping inference windows instead of 3 non-overlapping ones, giving the SQI-weighted aggregate substantially more signal to work with.

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
                              │  BlazeFace → 128×128 crops       │
                              │  50% overlap sliding window      │
                              │  adaptive window (≥5s, ≤20s)     │
                              │  ThreadPool: process_faces_tensor│
                              │  HRV sanity clamp                │
                              │  SQI-weighted aggregate          │
                              │                                  │
                              │  /ws/stream  [O(1) mode]         │
                              │  JPEG → cv2.imdecode             │
                              │  BlazeFace + 10% pad crop        │
                              │  FrameQualityMonitor gate        │
                              │  auto-exposure drift detection   │
                              │  deque(maxlen=600) rolling buffer│
                              │  np.stack → process_faces_tensor │
                              │  (Non-blocking Task)             │
                              │  HR consistency gate             │
                              │  median(7) + EMA(α=0.70)         │
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
| **WINDOW_FRAMES** | 600 (20 s) | Narrower FFT bins — at 5 s, frequency resolution is 0.2 Hz, wide enough for adjacent BPM values to bleed into each other. 20 s gives 0.05 Hz resolution. Validated against literature (npj Biosensing 2024). |
| **STEP_FRAMES** | 36 (~1.2 s) | Inference fires once per 1.2 s of new frames. Matches the `UPDATE_INTERVAL` used by `local_run.py`. |
| **CHUNK_STRIDE** | 300 (50% overlap) | File upload produces 5–6 overlapping windows instead of 3 non-overlapping. Dramatically improves aggregate robustness. |
| **EMA_ALPHA (HR)** | 0.70 | Higher alpha = faster response. Earlier value of 0.15 meant a real HR change took ~30 s to appear in the UI. |
| **EMA_ALPHA (RR)** | 0.30 | Slower smoothing — the respiratory cycle is longer, so RR needs more lag-time to not chase noise. |
| **SQI_THRESHOLD** | 0.30 | Literature-validated (npj Biosensing 2024 established the optimal threshold at 0.293). The original 0.35 was rejecting valid windows unnecessarily, especially under the JPEG compression path. |
| **BLUR_THRESHOLD** | 30.0 | Laplacian variance on the 128×128 face crop. Earlier value of 80.0 was rejecting clean frames — the small crop has naturally lower variance than a full HD frame. |
| **MOTION_THRESHOLD** | 25.0 | Mean absolute inter-frame diff on face grayscale. Earlier value of 8.0 was dropping frames during normal breathing movement. |
| **JPEG quality** | 0.85 | Empirically validated. Q=0.5 caused a 5–10 BPM systematic underestimate vs native pipeline. Q=0.85 closes this gap to ~1–3 BPM. PNG is unusable — browser JPEG encoding latency floors capture below 25 fps. See full history in the Engineering Journey. |
| **HR_CONSISTENCY_GATE** | 15 BPM | A >15 BPM change in 1.2 s is physiologically impossible at rest. Readings that exceed this delta are rejected and the previous stable value is held. |
| **HRV_CLAMP** | see `_clamp_hrv` | heartpy returns extreme outliers on marginal signal (IBI=8000 ms, SDNN=800 ms). Hard bounds: bpm 40–180, ibi 400–2000 ms, sdnn 0–200 ms, rmssd 0–150 ms, pnn50 0–100%, LF/HF 0–15, breathingrate 6–24 br/m. |
| **SQI_DISPLAY_ALPHA** | 0.25 (frontend) | Client-side EMA on the displayed SQI bar only — no inference impact. Stops the bar visually jumping between 38% and 72% on each cycle, which was a UX problem even when the underlying signal was stable. |

---

## The Window Constraint (5s vs 20s)

A 5-second processing window is often asked for in real-time specifications, but it is an honest constraint worth understanding:

| Metric | 5 s window | 20 s window | Why |
| :--- | :--- | :--- | :--- |
| **Heart Rate** | ✅ Estimable, volatile | ✅ Stable | 5–6 beats visible at rest. FFT bins are wide (0.2 Hz) at 5 s, narrow (0.05 Hz) at 20 s. |
| **Respiratory Rate** | ⚠️ Unreliable | ⚠️ Indicative | RSA-derived. Needs ≥60 s of clean IBI for reliable extraction. Face-only rPPG limits respiratory signal amplitude further. |
| **HRV SDNN / RMSSD** | ⚠️ Indicative | ⚠️ Indicative | Clinically meaningful HRV requires ≥1–2 min of clean signal. |
| **LF/HF ratio** | ❌ Meaningless | ❌ Questionable | LF band = 0.04–0.15 Hz; one LF wave period is 7–25 s — physically longer than a 5 s window and marginal at 20 s. |

The system runs with a 20-second window by default. HR is the most trustworthy output at this window size. All other metrics are real computations from the model's BVP signal — not fabricated — but they should be read as indicative rather than clinical.

---

## Respiratory Rate — An Honest Assessment

The respiratory rate figure in this system deserves a dedicated explanation because it is frequently misunderstood.

**How it's derived:** heartpy extracts breathing rate from **Respiratory Sinus Arrhythmia (RSA)** — the natural modulation of your heart rate interval caused by the breathing cycle. It does not detect chest movement. The face has essentially no chest signal. This is an indirect method.

**What heartpy actually returns:** The `breathingrate` key is in **Hz**, not breaths per minute. The documentation example explicitly shows `breathing rate is: 0.16109544905356424 Hz`. Every value must be multiplied by 60 before display. This is a non-obvious API detail that caused incorrect readings until discovered empirically.

**The search band ceiling:** heartpy's `calc_breathing` function uses a default bandpass filter of `[0.1, 0.4]` Hz — meaning it can only detect breathing between 6 and 24 breaths per minute. Breathing faster than 24 br/m (0.4 Hz) is outside the library's detection range regardless of signal quality.

**Why readings are typically low:** RSA-derived breathing rate from a 20-second window of face rPPG is inherently noisy. Clean RSA extraction needs: (a) a high-quality IBI series, which requires a clean BVP signal, which face rPPG delivers at 60–76% SQI rather than the 95%+ of a contact PPG; (b) at minimum 60 seconds of signal, not 20.

**The honest ceiling:** A reading of 10–14 br/m in relaxed conditions is plausible from this system. Values outside 6–24 br/m are clamped to zero. Treat the figure as a rough indication, not a measurement. It is displayed with an explicit `est.` label and a caveat in the UI for this reason. Improving it without a dedicated respiratory sensor or a full-body camera view is not achievable within this architecture.

---

## Signal Quality — The Compounding Variables

* **Webcam auto-exposure is the dominant noise source.** The rPPG algorithm detects periodic green-channel fluctuations caused by blood flow through skin. A webcam's automatic exposure system adjusts overall frame brightness continuously in response to head movement, background content, and even what is displayed on a nearby monitor. To the algorithm, a hardware brightness shift is indistinguishable from a blood volume pulse. This is the primary ceiling in typical indoor setups. The system now detects sustained brightness drift (>15 units over 18 seconds) on the face crop and surfaces a warning banner — it cannot fix it, but at least the user knows why SQI has dropped.
* **JPEG quality vs. arrival cadence — the counterintuitive history.** The rPPG signal exists in the colour domain across time. Timing regularity of frames is a prerequisite for FFT accuracy. At Q=1.0, high-quality JPEGs at 640×480 are ~410 KB each, producing ~12 MB/s WebSocket throughput that causes TCP congestion and bursty frame delivery. To the FFT, bursty delivery looks like heartbeat events. Dropping quality to Q=0.5 stabilised arrival cadence but introduced a 5–10 BPM systematic underestimate from green-channel quantization degrading the signal. The current Q=0.85 (~80 KB/frame, ~2.4 MB/s) was validated empirically: it closes the HR accuracy gap to ~1–3 BPM while maintaining uniform frame cadence on a local connection. The Q=0.5 → Q=0.85 change is a real finding, not a tuning guess.
* **Frame timing: `requestVideoFrameCallback` vs. `setInterval`.** `setInterval(fn, 33)` drifts against the hardware camera clock under any CPU load. `requestVideoFrameCallback` fires at the exact moment the camera delivers a new hardware frame. The difference is not visible to the eye but is meaningful to an FFT expecting uniformly-spaced temporal samples. Frame-rate jitter introduces frequency-domain noise that shows up as SQI degradation. The system measures actual capture rate and warns when it falls below 22 fps.
* **Quality gate ordering matters.** The `FrameQualityMonitor` runs checks in cheapest-first order: motion (absdiff, O(n)) → brightness (mean, O(n)) → blur (Laplacian convolution, more expensive). The motion check also updates the reference frame even on rejection, so the first recovery frame after a motion burst is not double-penalised against a stale pre-motion reference.
* **Targeted quality gates.** All checks are applied to the 128×128 face crop, not the full frame. A threshold on the full frame fires on background lighting changes (someone turning a lamp on across the room) that don't affect the face-ROI signal at all. Crop-level gating eliminates this class of false rejection.

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

A well-encoded I-frame-only video will eliminate the non-key frames warning and typically raise SQI from the 18–40% range to 60–80%+. The browser live stream path does not have this problem because canvas JPEG frames sent over WebSocket are individually complete images with no inter-frame compression.

---

## SQI-Weighted Aggregation

A plain average across chunks does not hold up when a single window has a lighting change or motion burst. The Signal Quality Index measures how clean the extracted BVP signal is for that window. Every metric is weighted by it:

```text
final_bpm = Σ(bpm_i × sqi_i) / Σ(sqi_i)
```

Chunks below `SQI = 0.30` are excluded entirely. The threshold of 0.30 is grounded in literature: npj Biosensing (2024) established the empirical optimal at 0.293 across multiple datasets using leave-one-out cross-validation. The previous value of 0.35 was rejecting 15–20% of valid windows unnecessarily.

For short videos (shorter than the 20-second ideal window), the window size shrinks adaptively down to a 5-second minimum (150 frames) so that something useful is returned rather than an error.

---

## HRV Sanity Clamping

heartpy occasionally returns physiologically impossible values on marginal signal: IBI of 8000 ms, SDNN of 800 ms, RMSSD above 200 ms. These values pass the SQI gate but corrupt the weighted aggregate if not filtered. The `_clamp_hrv` function applies hard physiological bounds before any value reaches the response payload:

| Field | Valid range | Literature basis |
| :--- | :--- | :--- |
| bpm (peak) | 40–180 | Normal human HR range |
| ibi | 400–2000 ms | 30–150 BPM in IBI space |
| sdnn | 0–200 ms | Clinical HRV norms |
| rmssd | 0–150 ms | PPG HRV literature |
| pnn50 | 0–100% | By definition |
| LF/HF | 0–15 | Physiological HRV range |
| breathingrate | 6–24 br/m | heartpy search band (0.1–0.4 Hz) |

Values outside these bounds are zeroed rather than clamped to the boundary — a zeroed value is honest about the failure; a clamped boundary value looks plausible but is fabricated.

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
      "respiratory_rate": 12.6,
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
    "final_rr": 12.4,
    "avg_sqi": 0.7214,
    "bpm_std": 1.82,
    "agg_hrv_sdnn": 40.7,
    "agg_hrv_rmssd": 37.2,
    "agg_hrv_lf_hf": 1.38,
    "chunks_total": 5,
    "chunks_valid": 5,
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
  "rr": 12.4,
  "hrv_ibi": 807.6,
  "hrv_sdnn": 42.1,
  "hrv_rmssd": 38.4,
  "hrv_pnn50": 21.3,
  "hrv_lf_hf": 1.38,
  "buffered_frames": 600,
  "box": [245, 120, 160, 160],
  "warning": "exposure_drift"
}
```

The optional `warning` field is present only when auto-exposure drift is detected. The frontend surfaces this as an amber banner without interrupting the session.

**WebSocket status values:**

| Status | Meaning |
| :--- | :--- |
| `buffering` | Filling the 20-second initial window. `buffered` and `target` fields show progress. |
| `low_signal` | Frame dropped by quality gate. `reason`: blurry / motion / no_face / lighting. |
| `ok` | Full metrics payload — SQI gate passed. |
| `no_signal` | Model returned no HR estimate this window. |
| `error` | Inference failure. Check `detail` field. |

---

## Browser vs. Native: Empirical Comparison (Lab vs. Wild)

This project maintains two distinct implementations that share core model weights but differ fundamentally in how the optical signal reaches the inference engine. The table below documents real-world measured values—not theoretical maximums—captured in controlled indoor conditions.

> [!IMPORTANT]
> **Iteration Note**: The results documented below are from initial integration testing across platforms. While they demonstrate functional parity, they have not yet undergone rigorous multi-subject longitudinal iteration. They serve as a baseline for current system performance as of May 2026.

| Metric | Browser (`app.py`) @ Q=0.85 | Native (`local_run.py`) | Observation |
| --- | --- | --- | --- |
| **Live HR Accuracy** | ~63.5–72.9 BPM | ~62.8–63.4 BPM | **Near-parity.** The Browser path shows slightly higher volatility during initial buffer stabilization. |
| **File HR Accuracy** | ~70.0–72.5 BPM | 73.3 BPM | **~3 BPM gap.** Attributed to MediaPipe vs. library-internal crop region logic. |
| **SQI (Quality)** | 58% – 72% | 63% – 80% | Native maintains a higher ceiling, but Browser is now competitive after the Q=0.85 optimization. |
| **Resp. Rate (br/m)** | 10.2 (est.) | 8.9 – 12.6 (est.) | **Successful.** RSA-based derivation is indicative; requires >1 min of high SQI signal for clinical-grade stability. |
| **HRV (SDNN)** | ~70.8 ms | ~74.3 ms | **High Correlation.** Validates that the JPEG-over-WebSocket path preserves timing intervals (~5% delta). |
| **Processing Time** | ~1050 ms (chunk) | ~2.52 s (full file) | Browser chunking allows near-real-time feedback despite Python web architectural overhead. |
| **Inference Latency** | ~1750–1830 ms | ~1100 ms (est.) | Browser involves Base64 and JPEG decoding overhead (~700ms delta). |

---

### Comparison & Contrast: The Two Systems

While both systems utilize the same `FacePhys.rlap` weights, their performance characteristics diverge based on their interaction with hardware and transport protocols.

#### 1. Signal Fidelity & The "Compression Penalty"

* **Native (`local_run.py`)**: The "Gold Standard." It provides direct access to raw camera frames (zero JPEG artifacts) and utilizes hardware-timed polling. It produces the "smoothest" signal, frequently hitting the **80% SQI** mark. It remains the reference tool for validating new models.
* **Browser (`app.py`)**: The "Docker-Safe" implementation. Initially, this path showed a 5–10 BPM systematic underestimate at low JPEG qualities (Q=0.5).
* **The Fix**: Increasing to **Q=0.85** closed the HR accuracy gap to within 1–3 BPM of the native pipeline while maintaining a stable 30 FPS arrival rate.

#### 2. Inference Latency & Overhead

* **Native**: Significantly faster. By bypassing the network and image-encoding stack entirely, it achieves a latency of **~1100ms**.
* **Browser**: Measures higher due to the mechanical necessity of the web architecture. The server must perform Base64 decoding and JPEG decompression on every frame before the tensor can be constructed, resulting in latencies of **~1800ms**.

#### 3. Detection Reliability

* **Native**: Uses the library’s internal detector. It provides slightly more stable Region of Interest (ROI) tracking for the FFT but is less resilient to extreme angles.
* **Browser**: Uses **MediaPipe BlazeFace**. This has proven highly robust against head tilts, partial occlusions, and low-light "silhouetting" during test runs.

#### 4. Environmental Sensitivity

Both systems now successfully detect and flag **Auto-exposure drift**. In recent test runs, the systems correctly flagged a drift of 18 units, coinciding with a marginal drop in SQI. This confirms that hardware-level exposure hunting remains the primary signal destabilizer for rPPG, regardless of the software implementation.

---

### Engineering Insights

**Why the HR gap closed at Q=0.85:**
The rPPG signal is a subtle ~1% variation in the spatial mean of the green channel across the 128×128 face crop. JPEG quantization at Q=0.5 introduces signal-dependent noise into the chroma planes that does not average out; instead, it creates low-frequency artifacts that alias with the cardiac band and shift the FFT peak. At Q=0.85, the quantization step is small enough that spatial averaging over 16,384 pixels drives the residual noise below the signal floor.

**Why a 3 BPM file gap remains:**
When using the same source file, compression is no longer a variable. The residual gap is architectural: **MediaPipe BlazeFace** crops a slightly different face boundary than the library's internal detector. Since the model was tuned against the library's specific training distribution, the slight variation in ROI contents results in a minor but measurable delta in the final peak detection.

---

## Failure Cases

* **Lighting changes:** A bright flash or window glare oversaturates the green channel. The `FrameQualityMonitor` catches abrupt brightness changes, but a sustained change (someone turning a lamp on in the background) will depress SQI for several windows until the EMA adapts. The auto-exposure drift detector will surface a warning banner during this period. Front-facing diffuse light eliminates this class of failure.
* **Webcam auto-exposure:** The primary ceiling in normal indoor use. A ring light or facing a bright window reduces this significantly. Sitting in a dark room with only a monitor as the light source is the worst-case scenario — the monitor content change triggers continuous auto-exposure adjustment, making the green channel variation indistinguishable from the pulse signal.
* **Motion:** Slow head movement and normal breathing are tolerated (MOTION_THRESHOLD=25.0 was calibrated against breathing artefact). Talking, laughing, or a hand in front of the face will degrade SQI for that window. The motion gate updates its reference frame even on rejection, so the recovery frame after a burst is not double-penalised.
* **Short windows and HRV:** SDNN, RMSSD, and LF/HF are produced from the model's BVP signal analysis — real computations, not estimates. But at 20-second windows they should be read as indicative. Clinical HRV analysis uses 5-minute recordings as a minimum. pNN50 values near zero are expected and correct at this window length.
* **Respiratory rate ceiling:** heartpy's search band is 0.1–0.4 Hz (6–24 br/m). Faster breathing is outside the library's detection capability. Values are clamped to this range; out-of-range results are zeroed.
* **HR spikes:** The temporal consistency gate rejects any single-window HR reading that differs by more than 15 BPM from the previous stable median. At a 1.2-second step interval, a 15 BPM change is physiologically impossible at rest. Artefact spikes are held at the last valid reading without polluting the smoothing buffer.
* **Variable frame rate (file upload):** The pipeline reads actual source FPS from OpenCV (`CAP_PROP_FPS`) and sizes windows accordingly, so FFT frequency mapping stays accurate regardless of recording FPS. For live streaming, 30 fps is assumed; the FPS monitor warns below 22 fps.
* **Short uploads:** If a video is shorter than the 20-second ideal window, the window shrinks adaptively to fit. Absolute floor is 5 seconds (150 frames). Below that, inference is refused with a descriptive error rather than returning a number that looks credible but is based on too little signal.

---

## Running Locally: Full Command Reference

```bash
# Clone and set up environment
git clone https://github.com/JoshPola96/rppg-monitor.git
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

**RAM constraint:** The FacePhys.rlap model plus the MediaPipe BlazeFace detector and a full 20-second face-crop buffer sits at approximately 600–800 MB resident. Render's free tier (512 MB) will OOM on startup. The 512 MB Render instance tier is the practical minimum.

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

Send JPEG data-URLs (quality=0.85) from a canvas element. Receive JSON per inference window. See status values table above.

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
* **Face detection:** MediaPipe BlazeFace (short-range model, `model_selection=0`) — replaced Haar cascade for better accuracy under angle, partial occlusion, and low contrast
* **Signal processing:** heartpy — RSA-based HRV and respiratory rate extraction from BVP waveform
* **Media decoding:** OpenCV VideoCapture (file upload), base64 JPEG over WebSocket (live stream)
* **Frontend:** plain HTML/JS — no framework, no build step
* **Deployment:** Docker

---

## The Engineering Journey

What follows is an account of the actual development sequence, including the wrong turns, the research-backed iterations, and the empirical validation that closed them out.

### Version 1 — The Bug That Ate the First Implementation

Initial architecture: `MediaRecorder` with a 250 ms timeslice, blobs sent over WebSocket, decoded with PyAV on the backend, frames accumulated, inference every 5 seconds.
*This produced exactly one reading and then nothing.*

The problem: browser `MediaRecorder` with timeslice only writes the WebM EBML container header into the very first blob. Every blob after that is a raw VP8 media segment — technically an incomplete file. PyAV requires the container header to initialise its demuxer. It cannot parse a bare media segment in isolation, and it returns an empty frame list without throwing an error. The buffer never filled past the first window. The logs looked normal. The silence was the only signal something was wrong.
Fix at that stage: accumulate all bytes from session start into one growing buffer and pass the entire thing to PyAV on each decode. This works because blob zero always contains the EBML header. It produced valid results.

### Version 2 — The Performance Cliff

Once byte-accumulation was stable, a new problem appeared: inference gaps that were 1 second at session start were 3+ seconds by minute one. The loop was falling behind real-time.
Cause: `decode_video_bytes(full_accumulated_buffer)` is not constant-time. It grows linearly with session length. At 50 seconds in, the server was decoding 50 seconds of video just to extract the last few seconds of frames. The PyAV call was the bottleneck, not the model inference.
This is the O(n) problem: accumulated bytes × decode cost × inference frequency = a wall that arrives around the 30–40 second mark for typical bitrates.

### Version 3 — O(1) Frame Mode (Current Architecture)

The architectural shift: stop treating frames as a transport problem and treat them as a data problem.
Instead of accumulating compressed video fragments and repeatedly decoding them, extract raw frames on the frontend and send them individually. The deque on the backend automatically discards frames older than the window. Memory and compute are flat regardless of session length.

* **Frontend:** `requestVideoFrameCallback` draws each hardware-synchronised camera frame onto a hidden canvas and sends it as a base64 JPEG.
* **Backend:** `deque(maxlen=600)` discards the oldest frame on each new arrival. When inference fires, `np.stack(frame_buffer)` builds the tensor from exactly the most recent 600 frames.

### The Async Inference Backpressure Trap

In the live stream, a severe bug caused the WebSocket to close automatically after a few seconds. The cause was a mismatch between inference speed (~1400–1800 ms) and the camera stream accumulation step (~1200 ms). Running sequentially in the async handler, frames piled up in memory, backpressure built, and Python eventually killed the connection.

The fix: decouple frame ingestion from inference with `asyncio.create_task`. If an inference cycle is still running when the next 1.2 s step arrives, the step is skipped and frame ingestion continues uninterrupted. Memory stays flat.

### Signal Quality — The Compounding Variables

After the architecture was stable, SQI was still fluctuating between 0.19 and 0.74 in the same session under apparently unchanged conditions. Working through the variables one by one:

* **JPEG quality** mattered in the opposite direction from expected. See the full analysis in the Signal Quality section above. The counterintuitive conclusion: timing regularity at Q=0.5 outweighed per-pixel fidelity until it didn't — the 5–10 BPM HR underestimate only became measurable once native pipeline comparison data existed.
* **Frame timing.** `setInterval` vs `requestVideoFrameCallback` — documented above.
* **Quality gate thresholds** needed empirical calibration. Initial blur threshold of 80.0 was rejecting clean frames. Initial motion threshold of 8.0 was dropping frames during normal breathing movement.

### The Haar Cascade → MediaPipe BlazeFace Migration

The original face detector was OpenCV's Haar cascade frontal face classifier (2001-era technology). Its failure modes are exactly the worst-case rPPG scenarios: head angle beyond 15°, low contrast lighting, partial occlusion. A 2023 ICECET comparative study (Contactless Camera-Based Heart Rate and Respiratory Rate) tested MediaPipe Face Mesh, Haar Cascade, MTCNN, and Dlib for rPPG HR measurements and recommended MediaPipe as the most suitable for rPPG applications due to accuracy, speed, and ROI extraction quality.

The migration was a drop-in replacement for `extract_face` — same function signature, same fallback-to-`last_box` behaviour on missed frames, same 10% padding. The detector is initialised at module level with `model_selection=0` (short-range, optimised for webcam distances under 2 m).

### The Non-Overlapping Chunk Problem

The file upload path originally cleared the frame buffer after every `WINDOW_FRAMES` frames — producing 3 non-overlapping chunks from a 60-second video. A single motion burst or lighting change in chunk 2 degraded that entire chunk, dragging the SQI-weighted aggregate down with it.

rPPG literature consistently uses a 50% stride (half-window overlap) for temporal averaging. The fix replaces the static `list.clear()` with a `deque(maxlen=win_frames)` and a `frames_since_infer` counter. When the deque is full and a stride has elapsed, inference fires without clearing — the oldest frames naturally fall off as new ones arrive. A 60-second video now produces 5–6 overlapping windows, giving the aggregate substantially more data and robustness against per-chunk noise events.

### The JPEG Quality Calibration

The Q=0.5 → Q=0.85 change is not a guess. Here is what the data showed:

| JPEG Quality | Web live HR | Native HR | Gap | SQI peak |
| :--- | :--- | :--- | :--- | :--- |
| 0.5 | 62–68 BPM | 72–73 BPM | 5–10 BPM | ~70% |
| 0.85 | 73–76 BPM (settling) | 72–73 BPM | ~1–3 BPM | 76% |

The green channel spatial mean across 16,384 pixels is a low-frequency signal that should, in theory, average out JPEG quantization noise. In practice, Q=0.5's aggressive DCT quantization tables introduce signal-dependent artefacts at frequencies close to the cardiac band that do not average out — they shift the FFT peak. Raising to Q=0.85 pushes the quantization step below the signal floor. PNG was tested (fully lossless) and failed: the browser's JPEG encoder at Q=0.85 runs in ~2 ms per frame; PNG compression at 640×480 runs in ~20 ms, which hard-floors effective capture rate below 25 fps regardless of bandwidth.

### The SQI Threshold — Literature-Backed Reduction

The original SQI gate of 0.35 was conservative. A 2024 paper in npj Biosensing established the empirically optimal threshold at **0.293** across multiple datasets using leave-one-out cross-validation. The previous 0.35 was rejecting 15–20% of valid windows in the browser path, particularly during the first several inference cycles when the sliding buffer is fresh and signal has not yet stabilised. Reducing to 0.30 (slightly above the empirical optimal, to account for webcam noise above lab conditions) recovered these windows without meaningfully increasing noise.

### heartpy Breathing Rate — The Hz Discovery

heartpy's `breathingrate` key is documented in Hz, not breaths per minute. The library's own documentation example shows `breathing rate is: 0.16109544905356424 Hz`. Every display value must be multiplied by 60. This was not discovered from documentation but from a combination of empirical readings (values of 0.2–0.3 that made no sense as br/m) and tracing the heartpy source, where `measures['breathingrate'] = frq[np.argmax(psd)]` assigns the frequency in Hz directly. Both `app.py` and `local_run.py` now apply the conversion and clamp to 6–24 br/m before display.

### The HRV Sanity Clamp

heartpy returns legitimate NaN or extreme values when signal is marginal. Before the `_clamp_hrv` filter, IBI values of 8000 ms, SDNN values of 800 ms, and RMSSD values above 300 ms were occasionally passing the SQI gate and corrupting the weighted aggregate. The filter applies hard physiological bounds derived from clinical literature. Values outside bounds are zeroed (not clamped to boundary — a zero is honestly missing data; a boundary value looks plausible but isn't).

### The HR Temporal Consistency Gate

The 7-window median filter kills single-inference spikes. But a 15 BPM jump between two consecutive medians at 1.2-second intervals is physiologically impossible at rest — heart rate simply cannot change that fast. The consistency gate checks before appending to the smoothing buffer: if the new reading deviates from the previous median by more than 15 BPM, it is rejected and the last stable value is held. The spike never enters the buffer, so the next reading is compared against the correct baseline.

### The Warmup Blind Period

The native pipeline previously logged `[Warmup] 11.3s / 20s` and printed nothing to the terminal HUD, which looked indistinguishable from a frozen process. The warmup display now renders a progress bar (`[████████░░░░░░░░░░░░] 40%`) and explicitly shows the buffer stabilising. The FPS counter also now resets after warmup, fixing a false "1.6 fps" warning caused by dividing a single initialisation frame over the entire startup time.

### The Auto-Exposure Drift Detector

After architecture, quality gates, and SQI thresholds were all tuned, SQI was still fluctuating unexpectedly in sessions where nothing visible had changed. The culprit: webcam auto-exposure adjusting to background elements — a monitor changing content, a light outside a window, a person walking past. The face crop brightness was tracking these adjustments at 10–20 unit swings over 15–20 readings.

The fix is a rolling brightness history on the face crop (not full frame — background changes would trigger false alarms). When the range of the last 15 brightness readings exceeds 15 units, an `exposure_drift` warning is added to the payload. The frontend surfaces an amber banner. It does not drop frames, does not affect inference, and does not produce false positives from gradual lighting changes — only from the sustained, rapid adjustment characteristic of auto-exposure hunting.

### Bounded Face Tracking UI

To give users confidence that the system hasn't frozen when inference lags or drops frames, a dynamic tracking box projects the server-side BlazeFace detection boundary onto the live video feed with CSS transitions. The box colour reflects signal quality: green (SQI ≥ 55%), amber (28–55%), red (< 28%). It tracks continuously even when the signal drops to red — the face is still being detected even if the signal is too noisy to extract vitals from.

### Webcam Mode — The Exit Crash

The `RuntimeError: cannot join current thread` that fires on exit from the native webcam mode is a thread-cleanup bug in the open-rppg library itself. When exiting the `model.video_capture(0)` block, the context manager attempts to join the library's internal capture threads. However, a background thread (`Thread-3`) tries to `join()` itself during cleanup.

Because this happens inside a separate background thread, it bypasses the main thread's `try...except` block entirely and prints a stack trace to stderr right after the session summary. The bug is in the library, not the application. The `except RuntimeError` block checks for `"cannot join current thread"` in the error message and suppresses it selectively — all other RuntimeErrors propagate normally.

### Bugs Resolved Across the Build

* **`Object of type float32 is not JSON serializable`** — FastAPI HTTP endpoints go through Pydantic, which handles NumPy types natively. The WebSocket `send_json` uses the stdlib encoder, which does not. The `make_serializable` utility recursively walks the payload dict and converts all NumPy primitives before the encoder sees them.
* **`Unexpected token 'N' … NaN is not valid JSON`** — `heartpy` returns `float('nan')` when it fails to find peaks in a noisy signal. The stdlib encoder writes `NaN` which is valid JavaScript but not valid JSON. `make_serializable` now checks `math.isnan` and `math.isinf` on standard Python floats and substitutes `0.0`.
* **`'<' not supported between instances of 'NoneType' and 'float'`** — `result.get("SQI", 0.0)` does not return the default when the model explicitly sets `SQI: None`. Python's `dict.get` only uses the default for absent keys, not for explicitly-set `None` values. The fix is `result.get("SQI") or 0.0`, which evaluates `None` as falsy.
* **`unsupported format string passed to NoneType.__format__`** — `heartpy` occasionally returns `{"hr": None}` on complete signal failure. The metrics formatter was calling `:.1f` on that value. Fixed by guarding every extraction.
* **Duplicate frame append** — An incomplete merge resulted in `frame_buffer.append(face_rgb)` being called twice per frame. The buffer looked full but contained only half the unique time coverage, giving the model a signal that appeared to pulse at half the real frequency.
* **Dockerfile CMD used `sh -c` wrapper** — `sh -c "uvicorn app:app ..."` is the parent process. SIGTERM goes to `sh`, which does not propagate it to uvicorn. The container waited 10 seconds for the forced SIGKILL on every shutdown. Replaced with `exec` form.
* **`captured_brightness` used before assignment in `stream_endpoint`** — The exposure drift check was placed at the quality gate step, before `captured_brightness` was assigned and before `payload` existed. Fixed by computing an `exposure_drifting` boolean at the gate and passing it as a captured variable into `infer_and_send`, where it is applied to the payload after the dict is constructed.
* **`_clamp_hrv` result discarded** — The clamped HRV dict was computed inside `run_model` but assigned to a local variable used only for logging, then the original unclamped result was returned. Fixed by mutating `result["hrv"]` in place before returning.
* **HR spike gate checked after buffer append** — The spike was already inside the median before the gate fired, making the gate compare a spike-inclusive median against itself. Fixed by checking the previous median before appending, rejecting the spike without polluting the buffer.
* **`numpy` not imported in `local_run.py`** — The auto-exposure drift block uses `np.mean(face_gray)` but numpy was not imported. Added `import numpy as np`.
* **Hardcoded `0.35` threshold in `run_native_file`** — The `SQI_THRESHOLD` constant was defined but not used in the file mode logging. Fixed.
* **The 422 Unprocessable Entity on file upload** — After switching to a 20-second window, short test videos produced zero chunks. Fixed with adaptive windowing (shrinks to 5-second minimum).

### Video Encoding — The I-frame Discovery

Early file upload tests returned SQI values in the 18–21% range with a consistent library warning: `OPEN-RPPG:WARNING - Detected non-key frames, this will damage the rPPG signal.`
Standard H.264/H.265 video uses inter-frame compression. Only I-frames (keyframes) contain complete pixel data; P-frames and B-frames store motion vectors and deltas. The rPPG signal exists as a ~1% variation in the green channel across frames. Inter-frame compression routinely classifies this as noise and discards it. Transcoding to an all-intra stream forces every frame to be a complete image, raising SQI to 60–80%+. The live stream path avoids this completely because canvas JPEG frames sent over WebSocket are individually complete images with no inter-frame compression.

### How AI Was Used

LLMs (Claude and Gemini) were used throughout as a combination of research layer, code scaffolding generator, and debugging thought partner.

**What AI is genuinely good at here:** Structural scaffolding — FastAPI app skeleton, Pydantic schemas, WebSocket endpoint boilerplate, Dockerfile, frontend JS — generated fast and iterated from there. When reasoning through design tradeoffs — O(1) deque vs. byte accumulation, EMA vs. median smoothing, overlapping vs. non-overlapping chunks — AI holds context well and stress-tests ideas quickly. When given error text, it identifies root causes immediately. Cross-referencing research papers against implementation decisions (SQI threshold literature, heartpy band limits, rPPG window duration constraints) is genuinely useful.

**What AI cannot do:** Run the code. Every bug in this build was found in a terminal or a browser log. The `dict.get` / explicit `None` crash, the `captured_brightness` scope error, the `_clamp_hrv` result being discarded — all were present in code that had been reviewed by AI. Catching them required executing the inference path against a model that returns the problematic values. Once the error text was provided, root causes were identified immediately — pattern matching on known error strings, not understanding the execution path.

**The 5-Second Window Calibration:** AI generated the code, schemas, and aggregation logic for 5-second chunks without flagging that respiratory rate requires at least one full breath cycle (3–5 s) to measure a frequency, or that LF/HF requires 7–25 s of signal per LF wave period. Recognising that these figures from a 5-second window are not reliable required understanding the signal processing, not generating code for it. The heartpy Hz vs BPM unit issue was similarly missed until empirical data made the numbers obviously wrong.

**The JPEG Quality Paradox:** This finding is a prime example of the value of human-AI iteration. The AI initially advised against higher JPEG quality; while it acknowledged that higher fidelity would improve the signal, it reasoned that TCP congestion from the increased bitrate would destabilize frame timing. This theoretical deadlock was only resolved through empirical testing. Back-to-back sessions proved that while Q=0.5 stabilized cadence, it introduced a 5–10 BPM systematic error. Moving to Q=0.85 closed the accuracy gap without triggering the predicted congestion. PNG was also tested but abandoned due to high encoding latency. Ultimately, human judgment provided the override, and the terminal provided the validation..

Approximate split: AI generated the initial structure and most of the boilerplate (~70% of lines written). Architectural decisions, all debugging against live logs, all threshold calibration against a real face and webcam, all empirical validation, and decisions about what to discard from AI suggestions when they conflicted with what the logs were showing — that was manual.

---

## What Would Come Next

* **Frame-level timestamping.** The current pipeline assumes frames arrive at exactly `TARGET_FPS`. Attaching a monotonic timestamp to each frame and using it to build the FFT time axis would make frequency mapping accurate under variable camera frame rates without assuming uniform delivery — the single most impactful correctness improvement available without changing the model.
* **Stateful inference.** Each window is currently independent. A model that carries BVP signal history across windows would give more stable HRV estimates and a faster time to first reliable reading — the 20-second warmup is a direct consequence of the stateless window approach.
* **Concurrency.** `ThreadPoolExecutor(max_workers=2)` is a single-server constraint. Proper multi-user deployment separates the FastAPI ingest layer from compute and pushes inference to a worker queue (Celery, RQ, or similar).
* **Auto-exposure suppression.** `getUserMedia` constraints include `exposureMode: 'manual'` and `exposureTime` settings. Browser and hardware support is currently too inconsistent to rely on, but as webcam driver support improves this becomes the single highest-impact quality improvement available — it eliminates the dominant noise source rather than detecting and warning about it.
* **Respiratory rate via amplitude demodulation.** The current RSA-based approach derives breathing from heart rate variability, which is inherently limited by IBI quality and window length. A direct approach — amplitude demodulating the BVP envelope to extract the respiratory frequency — would not require chest movement and could be more accurate from a face-only signal. The BVP waveform is already available via `model.bvp()`.
* **Session export.** The BVP waveform and per-window metrics are all in memory during a session but not persisted. A simple session-end export (JSON or CSV) would allow offline analysis with dedicated HRV tools.

---

*Joshua Peter Polaprayil — AI/ML Engineer — May 2026*