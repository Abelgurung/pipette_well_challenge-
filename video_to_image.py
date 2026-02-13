#!/usr/bin/env python3
"""
Extract frames from a video (or a folder of videos) into images.

Prefers ffmpeg (fast, no Python deps). Falls back to OpenCV if ffmpeg
is not available and opencv-python is installed.

Examples:
  python video_to_image.py /path/to/video.mp4
  python video_to_image.py /path/to/folder/with/videos
  python video_to_image.py video.mp4 --fps 2
  python video_to_image.py video.mp4 --every-n 10 --ext png
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert a video (or folder of videos) into per-frame images."
    )
    p.add_argument(
        "input_path",
        help="Path to an input video file OR a directory containing videos",
    )
    p.add_argument(
        "--out-root",
        default="vid_to_img",
        help='Root output directory (default: "vid_to_img"). Each video writes to <out-root>/<video_name>/',
    )
    p.add_argument(
        "--recursive",
        action="store_true",
        help="If input_path is a directory, search for videos recursively.",
    )
    p.add_argument(
        "--video-exts",
        default="mp4,mov,avi,mkv,webm,m4v,mpg,mpeg",
        help="Comma-separated list of video extensions to process when input_path is a directory.",
    )
    g = p.add_mutually_exclusive_group()
    g.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Extract at N frames per second (e.g., 2.5).",
    )
    g.add_argument(
        "--every-n",
        type=int,
        default=2,
        help="Extract every Nth frame (e.g., 10).",
    )
    p.add_argument(
        "--ext",
        default="jpg",
        choices=["jpg", "jpeg", "png", "webp"],
        help="Output image extension (default: jpg).",
    )
    p.add_argument(
        "--pattern",
        default="frame_%06d",
        help="Output filename pattern (printf-style, no extension). If no %%d is present, _%%06d will be appended. (default: frame_%%06d).",
    )
    p.add_argument(
        "--start-number",
        type=int,
        default=1,
        help="Starting index for extracted frames (default: 1).",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output images if present.",
    )
    return p.parse_args(argv)


def _normalize_pattern(pattern: str) -> str:
    # ffmpeg expects a sequence pattern (e.g. frame_%06d). If the user provides a
    # plain base name, make it a sequence.
    return pattern if "%" in pattern else f"{pattern}_%06d"


def _video_extensions(video_exts_csv: str) -> set[str]:
    exts: set[str] = set()
    for raw in video_exts_csv.split(","):
        e = raw.strip().lower()
        if not e:
            continue
        exts.add(e[1:] if e.startswith(".") else e)
    return exts


def _iter_video_files(input_path: Path, recursive: bool, video_exts: set[str]) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if not input_path.is_dir():
        return []

    pattern = "**/*" if recursive else "*"
    vids: list[Path] = []
    for p in input_path.glob(pattern):
        if not p.is_file():
            continue
        ext = p.suffix.lower().lstrip(".")
        if ext in video_exts:
            vids.append(p)
    vids.sort()
    return vids


def _out_dir_for_video(*, out_root: Path, input_dir: Path | None, video_path: Path) -> Path:
    # "same file name" => each video goes to <out_root>/<video_stem>/ (plus
    # subdirectories if recursive and input is a directory).
    if input_dir is not None:
        rel_parent = video_path.parent.relative_to(input_dir)
        out_dir = out_root / rel_parent / video_path.stem
    else:
        out_dir = out_root / video_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _run_ffmpeg(
    *,
    video_path: Path,
    out_dir: Path,
    ext: str,
    pattern: str,
    start_number: int,
    overwrite: bool,
    fps: float | None,
    every_n: int | None,
) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found")

    pattern = _normalize_pattern(pattern)
    out_pattern = str(out_dir / f"{pattern}.{ext}")

    cmd: list[str] = [ffmpeg]
    cmd += ["-hide_banner", "-loglevel", "error"]
    cmd += ["-y" if overwrite else "-n"]
    cmd += ["-i", str(video_path)]
    cmd += ["-start_number", str(start_number)]

    vf_parts: list[str] = []
    if fps is not None:
        vf_parts.append(f"fps={fps}")
    elif every_n is not None:
        # keep only frames where frame_number mod N == 0
        # Note: commas must be escaped inside filter strings for ffmpeg CLI.
        vf_parts.append(f"select=not(mod(n\\,{every_n}))")
        cmd += ["-vsync", "vfr"]

    if vf_parts:
        cmd += ["-vf", ",".join(vf_parts)]

    # Reasonable quality defaults.
    if ext in ("jpg", "jpeg"):
        cmd += ["-q:v", "2"]
    elif ext == "webp":
        cmd += ["-q:v", "75"]

    cmd += [out_pattern]

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg failed with exit code {e.returncode}") from e


def _run_opencv(
    *,
    video_path: Path,
    out_dir: Path,
    ext: str,
    pattern: str,
    start_number: int,
    overwrite: bool,
    fps: float | None,
    every_n: int | None,
) -> None:
    try:
        import cv2  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Neither ffmpeg nor OpenCV is available. Install ffmpeg or opencv-python."
        ) from e

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frame_idx = 0
    out_idx = start_number

    pattern = _normalize_pattern(pattern)

    # For --fps sampling, approximate by skipping frames based on source fps.
    step = None
    if fps is not None and src_fps > 0:
        step = max(1, int(round(src_fps / fps)))
    elif every_n is not None:
        step = max(1, every_n)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if step is None or (frame_idx % step == 0):
            # If pattern is printf-style (e.g. frame_%06d), format it. Otherwise,
            # append an index.
            if "%" in pattern and "d" in pattern:
                name = pattern % out_idx
            else:
                name = f"{pattern}_{out_idx:06d}"
            out_path = out_dir / f"{name}.{ext}"
            if out_path.exists() and not overwrite:
                raise RuntimeError(
                    f"Output file exists and --overwrite not set: {out_path}"
                )
            if not cv2.imwrite(str(out_path), frame):
                raise RuntimeError(f"Failed to write image: {out_path}")
            out_idx += 1

        frame_idx += 1

    cap.release()


def extract_frames(argv: list[str]) -> int:
    args = _parse_args(argv)
    input_path = Path(args.input_path).expanduser()
    if not input_path.exists():
        print(f"Error: input path not found: {input_path}", file=sys.stderr)
        return 2

    out_root = Path(args.out_root).expanduser()
    out_root.mkdir(parents=True, exist_ok=True)

    video_exts = _video_extensions(args.video_exts)
    videos = _iter_video_files(input_path, args.recursive, video_exts)
    if not videos:
        if input_path.is_dir():
            print(
                f"Error: no video files found in directory: {input_path}",
                file=sys.stderr,
            )
        else:
            print(f"Error: not a file or directory: {input_path}", file=sys.stderr)
        return 2

    input_dir = input_path if input_path.is_dir() else None
    pattern = args.pattern

    failures: list[tuple[Path, str]] = []
    for i, video_path in enumerate(videos, start=1):
        out_dir = _out_dir_for_video(
            out_root=out_root, input_dir=input_dir, video_path=video_path
        )
        print(f"[{i}/{len(videos)}] {video_path} -> {out_dir}")

        try:
            _run_ffmpeg(
                video_path=video_path,
                out_dir=out_dir,
                ext=args.ext,
                pattern=pattern,
                start_number=args.start_number,
                overwrite=args.overwrite,
                fps=args.fps,
                every_n=args.every_n,
            )
        except Exception as ffmpeg_err:
            # If ffmpeg missing, try OpenCV fallback. If ffmpeg existed but failed,
            # still try OpenCV (useful on some broken builds).
            try:
                _run_opencv(
                    video_path=video_path,
                    out_dir=out_dir,
                    ext=args.ext,
                    pattern=pattern,
                    start_number=args.start_number,
                    overwrite=args.overwrite,
                    fps=args.fps,
                    every_n=args.every_n,
                )
            except Exception as cv_err:
                failures.append((video_path, f"ffmpeg: {ffmpeg_err}; opencv: {cv_err}"))

    if failures:
        print("\nSome videos failed:", file=sys.stderr)
        for vp, msg in failures:
            print(f"- {vp}: {msg}", file=sys.stderr)
        return 1

    return 0


def main() -> None:
    raise SystemExit(extract_frames(sys.argv[1:]))


if __name__ == "__main__":
    main()
