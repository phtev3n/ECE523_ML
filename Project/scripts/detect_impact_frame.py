from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import av
import cv2
import numpy as np

try:
    from scipy.signal import butter, filtfilt, find_peaks
    SCIPY_OK = True
except ImportError:
    SCIPY_OK = False


# ── Tuning constants ───────────────────────────────────────────────────────────

MOTION_W, MOTION_H = 160, 90
AUDIO_HP_CUTOFF_HZ = 1500
AUDIO_RMS_WINDOW_MS = 4
AUDIO_MIN_PEAK_SEP_S = 0.15
GUI_HALF_WIN = 40
GUI_MAX_PX = 900


# ── Metadata helpers (PyAV only) ──────────────────────────────────────────────

def _open_container(video_path: Path) -> av.container.InputContainer:
    try:
        return av.open(str(video_path))
    except Exception as e:
        sys.exit(f"Could not open video {video_path}: {e}")


def _video_stream(container: av.container.InputContainer) -> av.video.stream.VideoStream:
    for stream in container.streams:
        if stream.type == "video":
            return stream
    raise RuntimeError("No video stream found")


def _audio_stream(container: av.container.InputContainer):
    for stream in container.streams:
        if stream.type == "audio":
            return stream
    return None


def _read_tkhd_rotation(video_path: Path) -> int:
    """Read rotation from the MOV/MP4 tkhd transformation matrix.

    iPhone HEVC .MOV files store the display orientation in the track-header
    (tkhd) box as a 3×3 matrix of 16.16 fixed-point values.  PyAV does not
    expose this through its Python metadata API, so we parse the binary atoms
    directly.  Returns the clockwise rotation in degrees (0, 90, 180, 270).
    """
    import math
    import struct as _struct

    def _read_box(f):
        hdr = f.read(8)
        if len(hdr) < 8:
            return None, 0
        size = _struct.unpack(">I", hdr[:4])[0]
        name = hdr[4:8].decode("latin-1", errors="replace")
        if size == 1:          # extended 64-bit size
            ext = f.read(8)
            if len(ext) < 8:
                return None, 0
            size = _struct.unpack(">Q", ext)[0] - 16
        elif size == 0:        # extends to end of file
            size = -1
        else:
            size -= 8
        return name, size

    try:
        with open(video_path, "rb") as f:
            file_size = video_path.stat().st_size
            while f.tell() < file_size:
                box_start = f.tell()
                name, payload = _read_box(f)
                if name is None:
                    break
                if name == "moov":
                    moov_end = f.tell() + payload
                    while f.tell() < moov_end:
                        trak_start = f.tell()
                        tname, tpayload = _read_box(f)
                        if tname is None:
                            break
                        if tname == "trak":
                            trak_end = f.tell() + tpayload
                            while f.tell() < trak_end:
                                iname, ipayload = _read_box(f)
                                if iname is None:
                                    break
                                if iname == "tkhd":
                                    ver = _struct.unpack("B", f.read(1))[0]
                                    f.read(3)           # flags
                                    skip = 20 if ver == 0 else 32
                                    f.read(skip)        # timestamps + track_id + duration
                                    f.read(14)          # reserved, layer, alt_group, vol, reserved
                                    raw = f.read(36)    # 3×3 matrix (9 × int32)
                                    if len(raw) == 36:
                                        m = _struct.unpack(">9i", raw)
                                        a = m[0] / (1 << 16)   # cos θ × scale
                                        b = m[3] / (1 << 16)   # sin θ × scale
                                        angle = round(math.degrees(math.atan2(b, a)))
                                        return int(angle) % 360
                                    break
                                elif ipayload > 0:
                                    f.seek(ipayload, 1)
                                else:
                                    break
                            break  # use the first video trak found
                        elif tpayload > 0:
                            f.seek(tpayload, 1)
                        else:
                            break
                    break
                elif payload > 0:
                    f.seek(payload, 1)
                else:
                    break
    except Exception:
        pass
    return 0


def _stream_rotation(container: av.container.InputContainer, vstream) -> int:
    """Return the clockwise rotation (degrees) for this video.

    Tries in priority order:
      1. stream metadata 'rotate' tag (works for some formats / older devices)
      2. container format metadata 'rotate' tag
      3. AVStream side-data display matrix (PyAV 9+ only)
      4. Direct MOV tkhd atom parse  ← catches iPhone HEVC files where
         PyAV exposes no metadata at all
    """
    # 1 & 2: metadata tags
    sources = [
        vstream.metadata or {},
        getattr(getattr(container, "format", None), "metadata", None) or {},
    ]
    for meta in sources:
        for key in ("rotate", "Rotate", "ROTATE"):
            val = meta.get(key)
            if val is not None:
                try:
                    return int(float(str(val).strip())) % 360
                except (ValueError, TypeError):
                    pass

    # 3: side-data display matrix (9 little-endian int32s, 16.16 fixed-point)
    import math, struct
    try:
        for sd in getattr(vstream, "side_data", []):
            raw = bytes(sd)
            if len(raw) < 36:
                continue
            m = struct.unpack_from("<9i", raw)
            cos_a = m[0] / (1 << 16)
            sin_a = m[3] / (1 << 16)
            angle = round(math.degrees(math.atan2(sin_a, cos_a)))
            return int(angle) % 360
    except Exception:
        pass

    # 4: parse the MOV tkhd box directly (works for iPhone HEVC .MOV files)
    try:
        video_path = Path(container.name)
        return _read_tkhd_rotation(video_path)
    except Exception:
        pass

    return 0


def _orient_frame(img: np.ndarray, rotation: int) -> np.ndarray:
    """Apply clockwise rotation correction so the frame displays upright."""
    if rotation == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if rotation == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    if rotation == 270:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img


def detect_fps(video_path: Path) -> float:
    with _open_container(video_path) as container:
        stream = _video_stream(container)

        rate = stream.average_rate or stream.base_rate or stream.guessed_rate
        if rate is not None:
            try:
                raw = float(rate)
                for std in (24, 25, 30, 48, 50, 60, 120, 240):
                    if abs(raw - std) < 0.5:
                        return float(std)
                return raw
            except Exception:
                pass

        # Fallback from duration/time_base
        if stream.duration is not None and stream.time_base is not None and stream.frames:
            dur_s = float(stream.duration * stream.time_base)
            if dur_s > 0:
                return float(stream.frames) / dur_s

    return 60.0


def get_total_frames(video_path: Path, fps: float) -> int:
    with _open_container(video_path) as container:
        stream = _video_stream(container)

        if stream.frames:
            return int(stream.frames)

        if stream.duration is not None and stream.time_base is not None:
            dur_s = float(stream.duration * stream.time_base)
            if dur_s > 0:
                return int(round(dur_s * fps))

        if container.duration is not None:
            dur_s = float(container.duration) / 1e6
            if dur_s > 0:
                return int(round(dur_s * fps))

    return 0


# ── Signal 1: audio onset (PyAV decode) ───────────────────────────────────────

def compute_audio_onset(video_path: Path, fps: float) -> np.ndarray | None:
    """Decode mono audio and return per-video-frame onset strength array."""
    if not SCIPY_OK:
        print("    [audio] scipy not available — skipping audio signal")
        return None

    try:
        with _open_container(video_path) as container:
            astream = _audio_stream(container)
            if astream is None:
                print("    [audio] no audio track found")
                return None

            chunks = []
            sample_rate = None

            for frame in container.decode(astream):
                if sample_rate is None:
                    sample_rate = int(frame.sample_rate)

                arr = frame.to_ndarray()

                # Common shapes:
                # packed mono: (samples,)
                # packed stereo: (channels, samples) or (samples, channels) depending on format path
                # planar: often (channels, samples)
                arr = np.asarray(arr)

                if arr.ndim == 1:
                    mono = arr.astype(np.float32)
                elif arr.ndim == 2:
                    # choose channel axis heuristically
                    if arr.shape[0] <= 8:
                        mono = np.mean(arr.astype(np.float32), axis=0)
                    else:
                        mono = np.mean(arr.astype(np.float32), axis=1)
                else:
                    mono = arr.reshape(-1).astype(np.float32)

                chunks.append(mono)

            if not chunks or sample_rate is None:
                print("    [audio] could not decode audio")
                return None

            data = np.concatenate(chunks).astype(np.float32)

    except Exception as e:
        print(f"    [audio] decode failed: {e}")
        return None

    # Normalize to roughly [-1, 1] if integer-like scales are present
    peak = float(np.max(np.abs(data))) if data.size else 0.0
    if peak > 0:
        if peak > 32768:
            data = data / peak
        elif peak > 1.5:
            data = data / peak

    nyq = sample_rate / 2.0
    if AUDIO_HP_CUTOFF_HZ < nyq:
        b, a = butter(4, AUDIO_HP_CUTOFF_HZ / nyq, btype="high")
        data = filtfilt(b, a, data)

    win = max(1, int(sample_rate * AUDIO_RMS_WINDOW_MS / 1000.0))
    n_win = len(data) // win
    if n_win <= 0:
        return None

    trimmed = data[: n_win * win]
    blocks = trimmed.reshape(n_win, win)
    rms = np.sqrt(np.mean(blocks ** 2, axis=1)).astype(np.float32)

    onset = np.diff(rms, prepend=rms[0])
    onset = np.clip(onset, 0.0, None)

    win_times = np.arange(n_win, dtype=np.float32) * (win / sample_rate)
    total_s = len(data) / float(sample_rate)
    n_vid = int(total_s * fps) + 2

    per_frame = np.zeros(n_vid, dtype=np.float32)
    for t, o in zip(win_times, onset):
        fi = int(t * fps)
        if 0 <= fi < n_vid:
            per_frame[fi] = max(per_frame[fi], o)

    return per_frame


# ── Signal 2: frame-to-frame motion onset (PyAV decode) ───────────────────────

def compute_motion_onset(video_path: Path) -> np.ndarray:
    """Decode video frames at low resolution and return motion onset."""
    try:
        with _open_container(video_path) as container:
            vstream = _video_stream(container)

            mad_series: list[float] = []
            prev: np.ndarray | None = None

            for frame in container.decode(vstream):
                img = frame.to_ndarray(format="gray")
                small = cv2.resize(
                    img,
                    (MOTION_W, MOTION_H),
                    interpolation=cv2.INTER_AREA,
                ).astype(np.float32)

                if prev is not None:
                    mad_series.append(float(np.mean(np.abs(small - prev))))
                else:
                    mad_series.append(0.0)

                prev = small

    except Exception as e:
        sys.exit(f"Video decode failed for motion analysis on {video_path}: {e}")

    if not mad_series:
        return np.zeros(1, dtype=np.float32)

    mad = np.array(mad_series, dtype=np.float32)

    kernel = np.array([0.1, 0.2, 0.4, 0.2, 0.1], dtype=np.float32)
    smooth = np.convolve(mad, kernel, mode="same")

    onset = np.diff(smooth, prepend=smooth[0])
    onset = np.clip(onset, 0.0, None)
    return onset


# ── Candidate selection ────────────────────────────────────────────────────────

def _peak_frame(signal: np.ndarray, fps: float, margin_frac: float = 0.05) -> int:
    n = len(signal)
    if n <= 1:
        return 0

    lo = max(1, int(n * margin_frac))
    hi = min(n - 1, int(n * (1.0 - margin_frac)))
    search = signal[lo:hi]

    if len(search) == 0:
        return int(np.argmax(signal))

    if SCIPY_OK:
        min_dist = max(1, int(fps * AUDIO_MIN_PEAK_SEP_S))
        threshold = float(np.percentile(search, 90))
        peaks, props = find_peaks(search, height=threshold, distance=min_dist)
        if len(peaks):
            best = int(peaks[np.argmax(props["peak_heights"])])
            return best + lo

    return int(np.argmax(search)) + lo


def find_candidate(
    motion_onset: np.ndarray,
    audio_onset: np.ndarray | None,
    fps: float,
    total_frames: int,
) -> tuple[int, int | None, int | None]:
    n = total_frames if total_frames > 0 else len(motion_onset)
    n = min(n, len(motion_onset))

    motion_cand = _peak_frame(motion_onset[:n], fps)

    audio_cand: int | None = None
    if audio_onset is not None and len(audio_onset) > 0:
        audio_cand = _peak_frame(audio_onset[:n], fps)

    combined = audio_cand if audio_cand is not None else motion_cand
    return combined, audio_cand, motion_cand


# ── GUI frame extraction (PyAV seek/decode) ───────────────────────────────────

def _load_gui_frames(
    video_path: Path,
    center: int,
    half_win: int,
    fps: float,
    orient_override: int = -1,
) -> tuple[list[np.ndarray], int]:
    """Load a window of decoded frames around center using PyAV.

    Seeks using AV_TIME_BASE (microseconds) without a stream argument so the
    offset is correctly interpreted regardless of the stream's time_base.
    Falls back to decoding from the start of file if seek fails.
    """
    start = max(0, center - half_win)
    end = center + half_win
    frames: list[np.ndarray] = []
    actual_start = start

    with _open_container(video_path) as container:
        vstream = _video_stream(container)

        # Use stream time_base for pts → seconds conversion; fall back to 1/fps.
        tb: float = float(vstream.time_base) if vstream.time_base else (1.0 / fps)

        # Rotation: use manual override if given, else auto-detect from metadata.
        if orient_override >= 0:
            rotation = orient_override % 360
        else:
            rotation = _stream_rotation(container, vstream)
        print(f"    [GUI] rotation={rotation}°")

        # Seek using AV_TIME_BASE (microseconds) — NOT stream-relative units.
        target_s = max(0.0, (start - 2) / max(fps, 1e-6))   # seek 2 frames early
        try:
            container.seek(int(target_s * 1_000_000), backward=True, any_frame=False)
        except Exception:
            pass  # decode from wherever we are (beginning of file on failure)

        found_first = False
        for frame in container.decode(vstream):
            # Compute frame index from presentation timestamp.
            if frame.pts is not None:
                idx = int(round(float(frame.pts) * tb * fps))
            else:
                # No pts: estimate from decode order after first valid frame.
                idx = (frames[-1][1] + 1) if frames else 0  # type: ignore[index]

            if idx < start:
                continue
            if idx > end:
                break

            if not found_first:
                actual_start = idx
                found_first = True

            img = frame.to_ndarray(format="bgr24")
            img = _orient_frame(img, rotation)
            h, w = img.shape[:2]
            scale = GUI_MAX_PX / max(h, w)
            if scale < 1.0:
                img = cv2.resize(
                    img,
                    (int(w * scale), int(h * scale)),
                    interpolation=cv2.INTER_AREA,
                )
            frames.append(img)

            if len(frames) >= (2 * half_win + 1):
                break

    return frames, actual_start


# ── GUI ────────────────────────────────────────────────────────────────────────

_TIMELINE_H = 64
_INFO_H = 48

_COL_MOTION = (60, 200, 60)
_COL_AUDIO = (60, 60, 220)
_COL_CURSOR = (0, 255, 255)
_COL_AUTO = (0, 200, 255)
_COL_AUDIO_MARK = (80, 255, 80)
_COL_MOTION_MARK = (80, 255, 255)


def _draw_signal_bar(
    canvas: np.ndarray,
    signal: np.ndarray,
    sig_start: int,
    n_frames: int,
    x0: int, y0: int, w: int, h: int,
    color: tuple[int, int, int],
    label: str,
) -> None:
    seg = signal[sig_start:sig_start + n_frames]
    if len(seg) == 0:
        return

    max_v = float(np.max(seg)) + 1e-8
    bar_w = max(1, w // max(len(seg), 1))

    for i, v in enumerate(seg):
        bh = int((v / max_v) * (h - 2))
        bx = x0 + int(i / max(len(seg), 1) * w)
        cv2.rectangle(canvas, (bx, y0 + h - bh), (bx + bar_w, y0 + h), color, -1)

    cv2.putText(
        canvas, label, (x0 + 3, y0 + 12),
        cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1
    )


def run_gui(
    video_path: Path,
    candidate: int,
    audio_cand: int | None,
    motion_cand: int | None,
    motion_onset: np.ndarray,
    audio_onset: np.ndarray | None,
    fps: float,
    total_frames: int,
    orient_override: int = -1,
) -> int | None:
    print("    Loading GUI frames...", end=" ", flush=True)
    frames, start_frame = _load_gui_frames(
        video_path, candidate, GUI_HALF_WIN, fps, orient_override=orient_override,
    )
    if not frames:
        print("FAILED — falling back to auto candidate")
        return None
    print(f"loaded {len(frames)} frames (#{start_frame}–#{start_frame + len(frames) - 1})")

    fh, fw = frames[0].shape[:2]
    win_w = max(fw + 20, 720)
    win_h = fh + _INFO_H + _TIMELINE_H + 10

    win_name = f"Impact Confirmation — {video_path.name}"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, win_w, win_h)

    n = len(frames)
    offset = max(0, min(n - 1, candidate - start_frame))

    def _render() -> None:
        canvas = np.zeros((win_h, win_w, 3), dtype=np.uint8)
        cur_frame = start_frame + offset

        img = frames[offset]
        ifh, ifw = img.shape[:2]
        ix = (win_w - ifw) // 2
        canvas[5:5 + ifh, ix:ix + ifw] = img

        cv2.putText(canvas, f"#{cur_frame}", (ix + 4, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)

        iy = fh + 10
        cv2.putText(canvas, f"Frame: {cur_frame}", (10, iy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        if audio_cand is not None:
            col = _COL_AUDIO_MARK if cur_frame == audio_cand else (110, 110, 110)
            cv2.putText(canvas, f"Audio:{audio_cand}", (180, iy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 1)

        if motion_cand is not None:
            col = _COL_MOTION_MARK if cur_frame == motion_cand else (110, 110, 110)
            cv2.putText(canvas, f"Motion:{motion_cand}", (350, iy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 1)

        if cur_frame == candidate:
            cv2.putText(canvas, "[AUTO]", (520, iy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, _COL_AUTO, 1)

        ctrl_y = iy + 22
        cv2.putText(
            canvas,
            "← a / → d  step    A/D jump5    c=auto  u=audio  m=motion    Enter=confirm  q=skip",
            (10, ctrl_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1,
        )

        tl_y = fh + _INFO_H + 5
        tl_x = 5
        tl_w = win_w - 10
        cv2.rectangle(canvas, (tl_x, tl_y), (tl_x + tl_w, tl_y + _TIMELINE_H), (25, 25, 25), -1)

        half_tl = _TIMELINE_H // 2
        _draw_signal_bar(canvas, motion_onset, start_frame, n, tl_x, tl_y, tl_w, half_tl, _COL_MOTION, "Motion onset")
        if audio_onset is not None:
            _draw_signal_bar(canvas, audio_onset, start_frame, n, tl_x, tl_y + half_tl, tl_w, half_tl, _COL_AUDIO, "Audio onset")

        cx = tl_x + int(offset / max(n - 1, 1) * tl_w)
        cv2.line(canvas, (cx, tl_y), (cx, tl_y + _TIMELINE_H), _COL_CURSOR, 2)

        def _mark(frame_idx: int, color: tuple[int, int, int]) -> None:
            rel = frame_idx - start_frame
            if 0 <= rel < n:
                mx = tl_x + int(rel / max(n - 1, 1) * tl_w)
                cv2.line(canvas, (mx, tl_y), (mx, tl_y + _TIMELINE_H), color, 1)

        if audio_cand is not None:
            _mark(audio_cand, _COL_AUDIO_MARK)
        if motion_cand is not None:
            _mark(motion_cand, _COL_MOTION_MARK)
        _mark(candidate, _COL_AUTO)

        cv2.imshow(win_name, canvas)

    _render()

    confirmed: int | None = None
    while True:
        key = cv2.waitKey(0)
        if key == -1:
            continue
        k8 = key & 0xFF

        if k8 in (13, 10):
            confirmed = start_frame + offset
            break
        if k8 in (ord("q"), 27):
            break

        if k8 == ord("a") or key in (2424832, 65361):
            offset = max(0, offset - 1)
        elif k8 == ord("d") or key in (2555904, 65363):
            offset = min(n - 1, offset + 1)
        elif k8 == ord("A"):
            offset = max(0, offset - 5)
        elif k8 == ord("D"):
            offset = min(n - 1, offset + 5)
        elif k8 in (ord("c"), ord("h")):
            offset = max(0, min(n - 1, candidate - start_frame))
        elif k8 == ord("u") and audio_cand is not None:
            offset = max(0, min(n - 1, audio_cand - start_frame))
        elif k8 == ord("m") and motion_cand is not None:
            offset = max(0, min(n - 1, motion_cand - start_frame))

        _render()

    cv2.destroyWindow(win_name)
    return confirmed


# ── Per-video processing ───────────────────────────────────────────────────────

def process_video(video_path: Path, no_gui: bool, orient_override: int = -1) -> dict:
    print(f"\n{'─' * 64}")
    print(f"  {video_path.name}")

    fps = detect_fps(video_path)
    total = get_total_frames(video_path, fps)
    print(f"    fps={fps:.0f}  total_frames={total}")

    print("    Computing motion onset...", end=" ", flush=True)
    motion_onset = compute_motion_onset(video_path)
    print(f"{len(motion_onset)} frames")

    print("    Extracting audio onset...", end=" ", flush=True)
    audio_onset = compute_audio_onset(video_path, fps)
    if audio_onset is not None:
        peak = int(np.argmax(audio_onset))
        print(f"peak at frame {peak}")
    else:
        print("unavailable")

    combined, audio_cand, motion_cand = find_candidate(
        motion_onset, audio_onset, fps, total,
    )
    print(f"    Audio candidate : {audio_cand}")
    print(f"    Motion candidate: {motion_cand}")
    print(f"    Combined        : {combined}")

    impact_frame = combined
    user_confirmed = False

    if not no_gui:
        result = run_gui(
            video_path, combined, audio_cand, motion_cand,
            motion_onset, audio_onset, fps, total,
            orient_override=orient_override,
        )
        if result is not None:
            impact_frame = result
            user_confirmed = True
            print(f"    Confirmed       : {impact_frame}")
        else:
            print(f"    Skipped — using auto candidate: {impact_frame}")

    return {
        "video": str(video_path),
        "fps": fps,
        "total_frames": total,
        "impact_frame": impact_frame,
        "audio_candidate": audio_cand,
        "motion_candidate": motion_cand,
        "auto_candidate": combined,
        "user_confirmed": user_confirmed,
    }


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Auto-detect golf shot impact frame with optional GUI confirmation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--video", nargs="+", required=True, help="One or more .mov video files")
    parser.add_argument("--out", default="impact_frames.json", help="Output JSON path")
    parser.add_argument("--no_gui", action="store_true", help="Skip GUI")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output file")
    parser.add_argument(
        "--orient", type=int, default=-1,
        help="Override frame orientation: 0=none, 90=CW, 180=upside-down, 270=CCW. "
             "Use 180 if the GUI shows the video upside-down. Default: auto-detect from metadata.",
    )
    args = parser.parse_args()

    video_paths = [Path(v) for v in args.video]
    missing = [p for p in video_paths if not p.exists()]
    if missing:
        sys.exit("Video not found:\n" + "\n".join(f"  {p}" for p in missing))

    out_path = Path(args.out)
    results: list[dict] = []
    done_paths: set[str] = set()
    done_stems: set[str] = set()

    if args.resume and out_path.exists():
        with open(out_path) as f:
            results = json.load(f)
        # Match by resolved path AND by filename stem so the done-set survives
        # video directory moves (e.g. UofA_Docs → repos) without re-processing.
        done_paths = {str(Path(r["video"]).resolve()) for r in results}
        done_stems = {Path(r["video"]).stem.lower() for r in results}
        print(f"Resuming — {len(results)} video(s) already processed")

    for vp in video_paths:
        if str(vp.resolve()) in done_paths or vp.stem.lower() in done_stems:
            print(f"  Skipping (done): {vp.name}")
            continue

        r = process_video(vp, no_gui=args.no_gui, orient_override=args.orient)
        results.append(r)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)

    print(f"\n{'=' * 64}")
    print(f"Processed {len(results)} video(s)  →  {out_path}")
    confirmed = sum(1 for r in results if r.get('user_confirmed'))
    print(f"  {confirmed} confirmed by user / {len(results) - confirmed} auto-only")

    print("\nNext: extract frames for each shot")
    print("  python scripts/extract_frames.py \\")
    print("      --video VIDEO --impact_frame N --out_dir OUTDIR")


if __name__ == "__main__":
    main()