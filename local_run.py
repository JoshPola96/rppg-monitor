# rppg_monitor/local_run.py

"""
Native rPPG Trace Monitor (Local CLI)

Uses the library's native real-time pipeline for maximum accuracy.
Two modes:
  --mode webcam : live camera via model.video_capture + model.preview (full native pipeline)
  --mode file   : single video via model.process_video (full native pipeline)

The native pipeline handles face detection, ROI extraction, quality checks,
bandpass filtering, and detrending internally — this is the gold standard
that the browser-based WebSocket approach replicates manually.

Usage:
  python local_trace.py --mode webcam
  python local_trace.py --mode file --file path/to/video.mp4
  python local_trace.py --mode webcam --model PhysMamba.pure
"""

import argparse
import logging
import time
from collections import deque as _deque 
import sys
import numpy as np

import cv2
import rppg

# ─────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("native")
logging.getLogger("rppg").setLevel(logging.INFO)


# ─────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────
WARMUP_TIME      = 20     # seconds before accepting readings (camera needs to stabilise)
WINDOW_SIZE      = 20     # seconds of recent signal to pass to model.hr()
UPDATE_INTERVAL  = 1.2    # seconds between HR/HRV extraction calls
SQI_THRESHOLD    = 0.30   # lower than web because native pipeline pre-cleans signal
EMA_ALPHA        = 0.70   # higher alpha = faster response (native signal is cleaner)


# ─────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────
def _fmt(v, decimals=1):
    """Safe formatter — returns '—' for None/0."""
    if v is None:
        return "—"
    try:
        return f"{float(v):.{decimals}f}"
    except (TypeError, ValueError):
        return str(v)


def _print_hud(val_hr, val_sqi, val_rr, val_sdnn, val_rmssd, val_pnn50, val_lf_hf, val_ibi, elapsed):
    """Single-line HUD to stdout — overwrites previous line."""
    line = (
        f"\r[{elapsed:6.1f}s] "
        f"HR: {_fmt(val_hr)} BPM  "
        f"SQI: {_fmt(val_sqi*100 if val_sqi else 0, 1)}%  "
        f"RR: {_fmt(val_rr)} br/m  "
        f"IBI: {_fmt(val_ibi, 0)} ms  "
        f"SDNN: {_fmt(val_sdnn)} ms  "
        f"RMSSD: {_fmt(val_rmssd)} ms  "
        f"pNN50: {_fmt(val_pnn50)}%  "
        f"LF/HF: {_fmt(val_lf_hf, 4)}"
    )
    sys.stdout.write(line)
    sys.stdout.flush()


def _print_result_block(result: dict, elapsed_s: float):
    """Full parameter dump for file mode."""
    hrv = result.get("hrv") or {}

    print("\n" + "=" * 60)
    print("  FULL PARAMETER TRACE")
    print("=" * 60)
    print(f"  {'HR (FFT)':<20}: {_fmt(result.get('hr'))} BPM")
    print(f"  {'SQI':<20}: {_fmt((result.get('SQI') or 0) * 100, 2)}%")
    print(f"  {'Latency':<20}: {_fmt(result.get('latency'), 1)} ms")
    print()
    print(f"  {'HRV bpm (peaks)':<20}: {_fmt(hrv.get('bpm'))} BPM")
    print(f"  {'IBI':<20}: {_fmt(hrv.get('ibi'), 1)} ms")
    print(f"  {'SDNN':<20}: {_fmt(hrv.get('sdnn'))} ms")
    print(f"  {'RMSSD':<20}: {_fmt(hrv.get('rmssd'))} ms")
    print(f"  {'pNN50':<20}: {_fmt(hrv.get('pnn50'))}%")
    print(f"  {'LF/HF':<20}: {_fmt(hrv.get('LF/HF'), 4)}")
    print(f"  {'Resp. Rate':<20}: {_fmt(hrv.get('breathingrate'))} br/m")
    print()
    print(f"  {'Wall time':<20}: {elapsed_s:.2f} s")
    print("=" * 60 + "\n")


# ─────────────────────────────────────────────────
# Webcam Mode — native pipeline
# ─────────────────────────────────────────────────
def run_native_webcam(model_path: str):
    """
    Uses model.video_capture() + model.preview for the full native pipeline.

    model.preview yields (frame, box) for every camera frame. The library handles
    face detection, ROI extraction, signal buffering, and quality checks internally.
    We call model.hr(start=-WINDOW_SIZE) on a timer to extract metrics from
    the last N seconds of buffered signal.

    This is the reference pipeline — the browser WebSocket approach replicates
    this behaviour manually because video_capture() requires direct camera access
    on the machine running the server, which does not exist in a headless container.

    Exit: press Q in the preview window, or Ctrl+C.
    The open-rppg library's context manager may internally raise
    RuntimeError('cannot join current thread') during thread cleanup on exit.
    This is caught and suppressed — it does not affect collected data.
    """
    logger.info(f"Initialising model: {model_path}")
    model = rppg.Model(model_path)

    # State
    val_hr = val_sqi = val_rr = val_sdnn = 0.0
    val_rmssd = val_pnn50 = val_lf_hf = val_ibi = 0.0
    stable_hr  = 0.0
    start_time = time.time()
    last_update = 0.0
    sqi_ema: float = 0.0
    SQI_DISPLAY_ALPHA = 0.25
    brightness_history: _deque = _deque(maxlen=15)
    exposure_warn_shown = False
    frame_count = 0
    fps_check_start = time.time()

    # For final summary
    history = []

    logger.info("Opening native camera context …")

    # ── Main capture loop ─────────────────────────────────────────────────────
    # The try/except/finally wraps the context manager so that:
    #   - KeyboardInterrupt (Ctrl+C) is caught and produces a clean shutdown message
    #   - RuntimeError("cannot join current thread") from the library's own thread
    #     cleanup is suppressed — it is a known benign side-effect of exiting the
    #     with block mid-stream and does not indicate a problem with the data
    #   - All other RuntimeErrors propagate normally
    #   - cv2.destroyAllWindows() always runs, even on unexpected exit
    try:
        with model.video_capture(0):
            logger.info("Native pipeline active. Press Q in the preview window or Ctrl+C to quit.")

            for frame, box in model.preview:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                now       = time.time()
                elapsed   = now - start_time
                frame_count += 1

                # Auto-exposure drift detection — face crop only
                if box is not None:
                    y1, y2 = box[0]; x1, x2 = box[1]
                    face_crop  = frame[y1:y2, x1:x2]   # frame is already RGB
                    face_gray  = cv2.cvtColor(face_crop, cv2.COLOR_RGB2GRAY)
                    face_brightness = float(np.mean(face_gray))
                    brightness_history.append(face_brightness)

                    if len(brightness_history) >= 8:
                        drift = max(brightness_history) - min(brightness_history)
                        if drift > 15.0 and not exposure_warn_shown:
                            print("\n⚠  Auto-exposure drifting — face brightness range "
                                f"{drift:.0f} units. Face a stable light source.")
                            exposure_warn_shown = True
                        elif drift <= 15.0:
                            exposure_warn_shown = False

                # ── Extract metrics on interval ───────────────────────────────
                if now - last_update > UPDATE_INTERVAL:
                    measured_fps = frame_count / (now - fps_check_start)
                    frame_count = 0; fps_check_start = now
                    if measured_fps < 22:
                        logger.warning(f"[FPS] Camera delivering {measured_fps:.1f} fps — below 25fps threshold")
                    if elapsed < WARMUP_TIME:
                        pct = int((elapsed / WARMUP_TIME) * 100)
                        bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
                        sys.stdout.write(f"\r[Warmup] [{bar}] {pct}% — stabilising signal buffer…")
                        sys.stdout.flush()
                        last_update = now
                        frame_count = 0
                        fps_check_start = now
                        continue                          # skip inference until buffer is full
                    else:
                        result      = model.hr(start=-WINDOW_SIZE)
                        last_update = now

                        if result:
                            sqi = result.get("SQI") or 0.0
                            hr  = result.get("hr")
                            hrv = result.get("hrv") or {}

                            if sqi > SQI_THRESHOLD and hr is not None:
                                new_hr = float(hr)
                                # Consistency gate — >15 BPM change in 1.2s is physiologically impossible
                                if stable_hr > 0.0 and abs(new_hr - stable_hr) > 15.0:
                                    logger.debug(f"[REJECT] HR spike: {new_hr:.1f} vs {stable_hr:.1f}")
                                else:
                                    stable_hr = (
                                        new_hr if stable_hr == 0.0
                                        else EMA_ALPHA * new_hr + (1 - EMA_ALPHA) * stable_hr
                                    )
                                    sqi_ema = (
                                        float(sqi) if sqi_ema == 0.0
                                        else SQI_DISPLAY_ALPHA * float(sqi) + (1 - SQI_DISPLAY_ALPHA) * sqi_ema
                                    )
                                    val_hr    = stable_hr
                                    val_sqi   = sqi_ema
                                    # Convert Hz to br/m and clamp (4-40)
                                    raw_br_hz = float(hrv.get("breathingrate") or 0.0)
                                    br_bpm = raw_br_hz * 60.0
                                    val_rr    = br_bpm if 6.0 <= br_bpm <= 24.0 else 0.0
                                    val_ibi   = float(hrv.get("ibi")           or 0.0)
                                    val_sdnn  = float(hrv.get("sdnn")          or 0.0)
                                    val_rmssd = float(hrv.get("rmssd")         or 0.0)
                                    val_pnn50 = float(hrv.get("pnn50")         or 0.0)
                                    val_lf_hf = float(hrv.get("LF/HF")         or 0.0)
                                    history.append(val_hr)
                            else:
                                logger.info(f"[REJECT] SQI={float(sqi):.2f} HR={hr}")

                # ── HUD on stdout ─────────────────────────────────────────────
                _print_hud(
                    val_hr, val_sqi, val_rr, val_sdnn, val_rmssd,
                    val_pnn50, val_lf_hf, val_ibi, elapsed
                )

                # ── OpenCV overlay ────────────────────────────────────────────
                if box is not None:
                    y1, y2 = box[0]
                    x1, x2 = box[1]
                    color  = (0, 255, 0) if val_sqi > SQI_THRESHOLD else (0, 165, 255)
                    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)

                overlay = frame_bgr.copy()
                cv2.rectangle(overlay, (10, 10), (320, 200), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.6, frame_bgr, 0.4, 0, frame_bgr)

                metrics = [
                    (f"HR:     {_fmt(val_hr)} BPM",    (0, 255, 0)),
                    (f"SQI:    {_fmt(val_sqi * 100, 1)}%", (255, 255, 255)),
                    (f"RR:     {_fmt(val_rr)} br/m",   (255, 200, 0)),
                    (f"SDNN:   {_fmt(val_sdnn)} ms",   (0, 165, 255)),
                    (f"RMSSD: {_fmt(val_rmssd)} ms",   (0, 200, 255)),
                    (f"LF/HF: {_fmt(val_lf_hf, 4)}",  (200, 200, 0)),
                ]
                y_off = 38
                for text, clr in metrics:
                    cv2.putText(frame_bgr, text, (18, y_off),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.62, clr, 2)
                    y_off += 28

                cv2.imshow("Native Trace", frame_bgr)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print()
                    logger.info("Exit requested by user")
                    break

    except KeyboardInterrupt:
        print()
        logger.info("Interrupted — shutting down.")
    except RuntimeError as e:
        if "cannot join current thread" not in str(e):
            raise
        # Library thread cleanup on context manager exit — benign, suppress it
    finally:
        cv2.destroyAllWindows()

    # ── Final Report ──────────────────────────────────────────────────────────
    if history:
        avg_hr = sum(history) / len(history)
        print("\n" + "=" * 60)
        print("   FINAL SESSION SUMMARY")
        print("=" * 60)
        print(f"   Averaged HR  : {avg_hr:.1f} BPM")
        print(f"   Last SQI     : {val_sqi * 100:.1f}%")
        print(f"   Last RR      : {_fmt(val_rr)} br/m")
        print(f"   Last SDNN    : {_fmt(val_sdnn)} ms")
        print(f"   Last RMSSD   : {_fmt(val_rmssd)} ms")
        print(f"   Last LF/HF   : {_fmt(val_lf_hf, 4)}")
        print(f"   Total Time   : {time.time() - start_time:.1f} s")
        print(f"   Readings     : {len(history)}")
        print(f"   Status       : Clean Shutdown")
        print("=" * 60 + "\n")
    else:
        logger.info("Pipeline shut down cleanly (no valid readings recorded).")


# ─────────────────────────────────────────────────
# File Mode — single-call native pipeline
# ─────────────────────────────────────────────────
def run_native_file(model_path: str, file_path: str):
    """
    Uses model.process_video() for a complete single-call analysis.

    process_video handles everything: frame extraction, face detection, quality
    checks, bandpass filtering, detrending, and signal extraction. It returns one
    result dict for the full video. This is the highest-accuracy mode since the
    model sees the entire signal at once rather than windowed chunks.

    For the web UI chunk breakdown we cannot use this path (it returns one result,
    not per-chunk metrics), but for validation and comparison it is the reference.
    """
    logger.info(f"Model:  {model_path}")
    logger.info(f"File:   {file_path}")

    model = rppg.Model(model_path)
    t0    = time.time()

    logger.info("Running native process_video …")
    result    = model.process_video(file_path)
    elapsed_s = time.time() - t0

    if not result:
        logger.error("process_video returned no result. Check the video file.")
        return

    sqi = result.get("SQI") or 0.0
    hr  = result.get("hr")
    hrv = result.get("hrv") or {}

    # Convert Hz to br/m and clamp (4-40)
    raw_br_hz = float(hrv.get("breathingrate") or 0.0)
    br_bpm = raw_br_hz * 60.0
    hrv["breathingrate"] = br_bpm if 4.0 <= br_bpm <= 40.0 else 0.0
    result["hrv"] = hrv  # Reassign so _print_result_block gets the updated value

    logger.info(
        f"[Result] HR={_fmt(hr)} | SQI={_fmt(float(sqi) * 100, 2)}% | "
        f"RR={_fmt(hrv.get('breathingrate'))} | SDNN={_fmt(hrv.get('sdnn'))} | "
        f"RMSSD={_fmt(hrv.get('rmssd'))} | pNN50={_fmt(hrv.get('pnn50'))} | "
        f"LF/HF={_fmt(hrv.get('LF/HF'), 4)} | IBI={_fmt(hrv.get('ibi'), 0)} ms"
    )

    _print_result_block(result, elapsed_s)

    if sqi > SQI_THRESHOLD and hr is not None:
        logger.info(f"[Stable HR] {float(hr):.1f} BPM (SQI gate passed)")
    else:
        logger.warning(
            f"[Stable HR] Rejected — SQI={float(sqi):.3f} below threshold {SQI_THRESHOLD} "
            "or HR is None. Improve lighting and keep face centred."
        )


# ─────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Native rPPG trace tool. "
                    "Uses open-rppg's native pipeline for maximum accuracy."
    )
    parser.add_argument(
        "--mode", choices=["webcam", "file"], default="webcam",
        help="webcam: live camera. file: process a video file."
    )
    parser.add_argument(
        "--file", type=str, default=None,
        help="Path to video file (required when --mode file)"
    )
    parser.add_argument(
        "--model", type=str, default="FacePhys.rlap",
        help="Model name (default: FacePhys.rlap). See model zoo in open-rppg docs."
    )
    args = parser.parse_args()

    if args.mode == "file":
        if not args.file:
            parser.error("--file is required when --mode file")
        run_native_file(args.model, args.file)
    else:
        run_native_webcam(args.model)