# rPPG Vitals Monitor

A prototype that reads heart rate, respiratory rate, and HRV from a standard webcam — no wearables, no contact. Built as a take-home challenge for Wise AI.

**Live demo →** `https://wise-ai-rppg.onrender.com`

---

## What it does

You point a camera at your face for 60 seconds. The system processes your video in 5-second sliding windows, extracts your pulse signal from subtle skin colour changes (remote photoplethysmography), and gives you:

- **Heart rate (BPM)** — per chunk and aggregated over the full session, with SQI-weighted smoothing
- **Respiratory rate** — extracted from the BVP waveform (reliable on ≥ 15 s windows)
- **HRV:** SDNN, RMSSD, pNN50, LF/HF ratio, IBI

Two ways to use it. Upload a pre-recorded video, or open the live stream tab and let it run directly from your webcam.

---

## The 5-second window constraint (read this first)

The challenge specifies a 5-second processing window. It is important to be upfront about what signal processing can and cannot do in that time.

| Metric | 5 s window | Why |
|---|---|---|
| **Heart Rate** | ✅ Estimable, volatile | 5–6 heartbeats visible at 60–70 BPM — barely enough for FFT |
| **Respiratory Rate** | ⚠️ Unreliable | One breath takes 3–5 s; can't run frequency analysis on a single cycle |
| **HRV SDNN / RMSSD** | ⚠️ Indicative only | Need ≥ 1–2 min for stable estimates |
| **LF/HF ratio** | ❌ Not meaningful | LF band = 0.04–0.15 Hz; one LF wave takes 7–25 s — longer than the window |

The system produces all metrics from the model's output. The HR estimate is the most trustworthy one at 5-second resolution. Respiratory and HRV values are included because the model computes them, but they should be treated as indicative rather than clinical at this window size.

The live stream UI intentionally keeps the "deeper metrics" section always visible — for a production system, the HRV fields would be hidden behind a "gathering longer context…" state until at least 30 s of clean signal has accumulated.

---

## Sample output

**File upload — 60-second video, 12 chunks**

```json
{
  "chunks": [
    {
      "chunk_index": 0,
      "time_start_s": 0.0,
      "time_end_s": 5.0,
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
    "chunks_total": 12,
    "chunks_valid": 10,
    "total_time_ms": 14820.0,
    "message": "OK"
  }
}
```

**Live stream** — WebSocket emits this every ~0.5 s of new data:

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
  "buffered_frames": 150
}
```

---

## Stack

- **Model:** [open-rppg](https://github.com/KegangWangCCNU/open-rppg) — `FacePhys.rlap`, a state-space model trained on the RLAP benchmark
- **Backend:** FastAPI + uvicorn, Python 3.11
- **Media decoding:** OpenCV (file upload), base64 canvas frames (live stream)
- **Frontend:** plain HTML/JS — no framework, no build step
- **Deployment:** Docker → Render

---

## Architecture

```
Browser (Live)    JPEG canvas frames  ──► WS  /ws/stream
Browser (Upload)  <input type=file>   ──► POST /analyze
                                                │
                                   ┌────────────▼──────────────┐
                                   │         FastAPI            │
                                   │                            │
                                   │  /analyze                  │
                                   │  OpenCV decode once        │
                                   │  → (T,H,W,3) tensor        │
                                   │  Non-overlap windows       │
                                   │  ThreadPool: run_model     │
                                   │  → SQI-weighted aggregate  │
                                   │                            │
                                   │  /ws/stream  [O(1) mode]   │
                                   │  JPEG → cv2.imdecode       │
                                   │  FrameQualityMonitor gate  │
                                   │  deque(maxlen=150)         │
                                   │  np.stack → run_model      │
                                   │  Median + EMA smooth       │
                                   │  → stream JSON back        │
                                   │                            │
                                   │  FacePhys.rlap             │
                                   │  warm in memory, locked    │
                                   └────────────────────────────┘
```

The model is loaded once at startup and held in memory behind a `threading.Lock`. Inference runs in a `ThreadPoolExecutor` via `asyncio.run_in_executor`, so the event loop stays free while a chunk is processing.

---

## Why SQI-weighted aggregation

A plain average across 12 chunks doesn't hold up when even one window has a lighting change or motion. The Signal Quality Index the model returns measures how clean the extracted pulse signal is for that window. Every metric is weighted by it:

```
final_bpm = Σ(bpm_i × sqi_i) / Σ(sqi_i)
```

Chunks below `SQI = 0.40` are dropped entirely. If fewer than 3 windows survive that cut, the system says so rather than returning a number that looks credible but isn't.

---

## Performance

| Stage | Typical time (CPU) |
|---|---|
| OpenCV decode, 60 s video | 300–600 ms |
| Inference per 5 s window | 450–520 ms |
| Aggregation (12 chunks) | < 5 ms |
| Full 60 s file analysis | 14–22 s |
| Live stream update cadence | ~0.5 s per reading |
| Effective inference speed | ~290–330 FPS equivalent |

---

## Failure cases worth knowing about

**Lighting changes.** A bright flash or window glare can oversaturate the green channel for a frame or two. The `FrameQualityMonitor` gates these frames out before they enter the sliding buffer, but a sustained change (someone turning a light on) will still depress SQI for several windows until the EMA adapts.

**Webcam auto-exposure.** Most webcams continuously adjust brightness frame by frame. To the rPPG algorithm, a hardware brightness shift looks identical to a massive blood volume pulse. This is the primary driver of SQI fluctuation in normal indoor use. A ring light or sitting directly facing a bright window (not behind you) dramatically reduces this. Disabling auto-exposure via `getUserMedia` constraints is possible in theory but browser/hardware support is too inconsistent to rely on.

**Motion.** Slow head movement is fine — the model handles it. Talking, laughing, or a hand in front of your face will cause signal degradation in that window. The quality monitor's motion gate (mean absolute frame diff > 25.0) catches abrupt movement and drops those frames.

**Short windows and HRV.** See the 5-second window constraint table at the top. SDNN and RMSSD are real values derived from the model's BVP signal analysis. LF/HF is present in the output but should not be interpreted at this window size.

**Variable frame rate.** For file upload, we read actual source FPS from OpenCV (`CAP_PROP_FPS`) and size windows accordingly, so the FFT frequency mapping stays accurate. For live streaming, we assume the camera delivers at ~30 fps — a production system would timestamp each frame and adjust.

---

## Running locally

```bash
git clone https://github.com/your-username/wise-ai-rppg
cd wise-ai-rppg

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

uvicorn app:app --reload --port 8000
# → http://localhost:8000
```

**Docker (development — with auto-reload):**

```bash
docker build -t rppg-bpm .
docker run --rm --init -p 8080:8000 -v ${PWD}:/app rppg-bpm uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

The `--init` flag is important on Windows — it wraps the process in `tini` so `Ctrl+C` actually terminates the container and releases the port immediately.

---

## Deploying to Render

Push to GitHub. New Web Service → Docker runtime → connect repo. `render.yaml` is auto-detected, health check runs at `/health`.

The Dockerfile pre-bakes model weights at build time so there's no cold-start download in production:

```dockerfile
RUN python -c "import rppg; rppg.Model('FacePhys.rlap')" || true
```

The `CMD` uses `exec` (not `sh -c`) so Render's shutdown signal goes directly to uvicorn rather than being swallowed by the shell, enabling graceful draining.

One thing to watch: Render's free tier spins down inactive services. A WebSocket handshake against a sleeping instance will fail. Either set the service to always-on, or handle reconnect on the client.

---

## API

### `POST /analyze`

Upload a video file for chunk-level and aggregate analysis.

**Request:** `multipart/form-data`, field `file`. Accepts `.mp4 .webm .mov .avi .mkv`.  
**Response:** `{ chunks: ChunkMetrics[], aggregate: AggregateResult }`

### `WS /ws/stream`

Send JPEG data-URLs from a canvas element. Receive JSON per inference window.

| Status | Meaning |
|---|---|
| `buffering` | Filling the initial 150-frame window, not enough data yet |
| `low_signal` | Frame dropped by quality gate (`reason`: blurry / motion / lighting) or SQI below threshold |
| `ok` | Full metrics payload |
| `no_signal` | Model returned no HR estimate |
| `error` | Inference failure, check `detail` field |

### `GET /health`

```json
{ "status": "ok", "model": "FacePhys.rlap", "target_fps": 30, "window_s": 5.0 }
```

---

## The engineering journey

Wise AI asked how I used AI tools. The honest answer involves going down several wrong paths and having to reason my way out of them, so I'm writing it as a story.

### Version 1 — the bug that ate the first implementation

The initial approach: `MediaRecorder` with a 250 ms timeslice, send each blob over WebSocket, decode each blob with PyAV, accumulate frames, fire inference every 5 seconds. This produced exactly one reading and then nothing.

The problem took time to find. Browser `MediaRecorder` with `timeslice` only writes the WebM EBML container header into the very first blob. Every blob after that is a raw VP8 media segment — technically an incomplete file. PyAV can't parse it in isolation. It returns an empty frame list without throwing an error. The buffer never filled past the first window.

The fix at that stage: accumulate all bytes from the start of the session into one growing buffer and pass the entire thing to PyAV on each decode. You always have the EBML header because it was blob zero. This worked, but it was O(n) — at second 50 the server was decoding 50 seconds of video just to extract the last 150 frames.

### Version 2 — the performance cliff

Once the byte-accumulation approach was stable, a new problem appeared in the logs. Inference gaps that were 1 second at session start were 3+ seconds by minute one. The loop was falling behind real-time.

Cause: `decode_video_bytes(full_accumulated_buffer)` is not a constant-time operation. It grows linearly with session length. 250 ms × 30 fps × 50 s = a lot of bytes to hand PyAV on every step.

### Version 3 (current) — O(1) frame mode

The architectural shift: stop treating frames as a transport problem and treat them as a data problem. Instead of accumulating compressed video fragments and decoding them repeatedly, extract raw frames on the frontend and send them individually.

Frontend: `requestVideoFrameCallback` draws each hardware-synchronized camera frame onto a hidden canvas and sends it as a base64 JPEG over the WebSocket. Backend: `deque(maxlen=150)` automatically discards the oldest frame on each new arrival. When inference fires, `np.stack(frame_buffer)` builds the tensor from exactly 150 frames. Memory is flat. Decode cost is flat. This holds at any session length.

### Signal quality — the compounding variables

After the architecture was stable, SQI was still fluctuating between 0.19 and 0.74 in the same session under apparently unchanged conditions. Working through the variables:

**JPEG quality mattered more than expected.** Initially quality was set to 1.0 (lossless) on the theory that the micro-color changes needed maximum fidelity. Testing showed that high-quality JPEGs at 640×480 are ~410 KB per frame. At 30 fps that's ~12 MB/s over the WebSocket. TCP congestion caused 2–3 frames to arrive at the server simultaneously, then a gap. To the rPPG model, that timing irregularity looks like a heartbeat event. Dropping quality to 0.5 reduced frame size to ~27 KB (~800 KB/s) and the arrival cadence became uniform.

**Frame timing matters as much as frame content.** Using `setInterval` at 33 ms introduces drift against the hardware camera clock. `requestVideoFrameCallback` fires at the exact moment the camera delivers a new frame, eliminating that drift entirely.

**Webcam auto-exposure is the dominant noise source indoors.** The model extracts heart rate by detecting periodic colour fluctuations in the green channel of facial skin. A webcam's automatic exposure system is continuously adjusting overall frame brightness in response to head movements, background changes, and monitor content. To the algorithm, a hardware brightness shift is indistinguishable from a large blood volume pulse. This is the primary ceiling at typical indoor setups. The `FrameQualityMonitor` catches abrupt changes (sudden motion, blurring) but cannot compensate for gradual exposure drift, which the model interprets as signal noise.

**The quality monitor thresholds needed calibration.** Early values (blur threshold 80, motion threshold 8) were too aggressive — they were rejecting clean frames. The final values (blur 30, motion 25) gate genuinely bad frames while passing normal slight head movement.

### Bugs resolved across the session

Six iterations of concrete bugs were caught and fixed, mostly through reading the actual error rather than the expected error:

`Object of type float32 is not JSON serializable` — the WebSocket `send_json` uses the stdlib encoder, which doesn't know NumPy types. FastAPI's HTTP endpoints go through Pydantic, which does. The `make_serializable` recursive utility handles this by walking the entire payload dict and converting all NumPy primitives before the encoder sees them.

`Unexpected token 'N' … NaN is not valid JSON` — `heartpy` returns `float('nan')` (standard Python NaN) when it fails to find peaks in a noisy signal. The stdlib encoder writes `NaN` which is valid JavaScript but not valid JSON. `make_serializable` now also checks `math.isnan` and `math.isinf` on standard floats.

`'<' not supported between instances of 'NoneType' and 'float'` — the fatal crash visible at the end of the latest log session. `result.get("SQI", 0.0)` does not return the default when the model explicitly sets `SQI: None` — Python's `dict.get` only uses the default for absent keys. The fix is `result.get("SQI") or 0.0`, which evaluates `None` as falsy and substitutes zero.

`unsupported format string passed to NoneType.__format__` — `heartpy` occasionally returns `{"hr": None}` on a complete signal failure. The metrics logger was calling `:.1f` on that value. Fixed by guarding with `float(x) if x is not None else 0.0`.

The duplicate frame append (from an incomplete merge of a reviewed patch) was filling the deque with duplicate frames — the buffer looked full but contained only half the unique time coverage.

The Dockerfile `CMD` used `sh -c` wrapper, which swallowed SIGTERM on container shutdown and caused 10-second forced kills. Replaced with `exec uvicorn ...` so the process is PID 1 and handles signals directly.

### How AI was used — and where it hits its ceiling

I used Claude and Gemini throughout this build as a combination of junior researcher, knowledge retrieval layer, and code generator. I want to be specific about what that actually looked like rather than just saying "AI helped."

**What AI is genuinely good at here.** Drafting structural scaffolding fast — the FastAPI app skeleton, Pydantic schemas, WebSocket endpoint boilerplate, Dockerfile, frontend JS — all of that came out of AI-assisted generation and then got shaped through iteration. When I needed to understand the open-rppg library's `process_video_tensor` interface or PyAV's container format expectations, AI was a faster first-pass reference than reading raw source. For reasoning through design tradeoffs — O(1) frame deque vs byte accumulation, EMA vs median smoothing, why SQI-weighting is preferable to a plain average — AI is a capable thought partner. It holds the context of a conversation well and helps stress-test ideas.

**What AI cannot do.** Run the code. Every bug in this build was found in a terminal or a browser log, not in conversation. AI reviewed the same code that had the `dict.get` / explicit `None` crash and did not catch it — because catching it requires actually executing the inference path against a model that returns `{"SQI": None}` on a complete signal failure, then reading the traceback. AI consistently proposed fixes to symptoms rather than causes until the actual stack trace was pasted in. Once it saw the error text, it identified the root cause immediately — but that's pattern matching on known error strings, not understanding the execution path.

The failures that required the most judgment were all the same class of problem: assuming that a mental model of how something works matches how it actually behaves at runtime. The PyAV blob header issue looked fine on paper — of course you accumulate fragments and decode them. It failed silently. The `dict.get` default looked fine — of course it falls back to `0.0`. It didn't when the value was explicitly `None`. `setInterval` at 33 ms looks like 30 fps. It isn't when the system is under load. None of these were visible in code review, AI-assisted or otherwise.

**The 5-second window — a deliberate challenge.** The task specifies a 5-second processing window, and I believe this is intentional. It is a constraint that forces you to understand what the algorithm actually measures rather than just wiring up an API call. At 5 seconds you get 5–6 heartbeats at a resting rate. The FFT frequency resolution at 5 seconds is 0.2 Hz — wide enough that adjacent BPM values become hard to distinguish. Respiratory rate needs at least one full breath cycle (3–5 seconds), which means you can only ever sample a single cycle in the window — not enough for frequency analysis. LF/HF ratio requires the Low Frequency band (0.04–0.15 Hz), whose period is 7–25 seconds — physically longer than the window. When AI was asked to extract all these metrics from 5-second chunks it did so without flagging any of this. It generated the code, the schemas, the aggregation logic. Recognising that respiration and HRV figures from a 5-second window are not reliable in any clinical sense — and documenting that honestly rather than just displaying the numbers — was a judgment call that required understanding the signal processing, not generating code for it.

**What the iteration cycle actually looked like.** A rough split: AI generated the initial structure and most of the boilerplate (~70% of lines written), I made the architectural decisions, debugged every failure against live logs, calibrated every threshold (blur, brightness, motion, SQI gate, EMA alpha) empirically against my own face and webcam, and decided what to discard from AI suggestions when they conflicted with what the logs were showing. Several AI-suggested "fixes" during the signal quality phase were solving the wrong problem — proposing code changes when the actual issue was lighting and webcam hardware. Identifying that distinction required running the system, not reviewing it.

---

## What would come next

**Face crop on the client side.** A lightweight face detector (MediaPipe via WebAssembly) cropping to the forehead/cheek ROI before transmission would reduce frame payload size and improve signal quality by cutting background noise from the input.

**Frame-level timestamping.** The current pipeline assumes frames arrive at exactly TARGET_FPS. Attaching a monotonic timestamp to each frame and using it to build the time axis for FFT would make the frequency mapping accurate under variable camera frame rates.

**Stateful inference.** Each 5-second window is currently independent. A model that carries BVP signal history across windows would give more stable HRV estimates and a faster time to first reliable reading.

**Concurrency.** The current `ThreadPoolExecutor(max_workers=2)` is a single-server constraint. Proper multi-user deployment would separate the FastAPI ingest layer from compute and push inference to a worker queue.

---

*Joshua Peter Polaprayil — AI/ML Engineer*  
*May 2026*