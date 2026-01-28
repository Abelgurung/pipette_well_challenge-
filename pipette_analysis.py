from __future__ import annotations

import argparse
import base64
import json
import os
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

import dotenv  # type: ignore
dotenv.load_dotenv()

# Default video paths
DEFAULT_FPV_PATH = "pipette_well_dataset/pipette_well_dataset/Plate_10_clip_0003_FPV.mp4"
DEFAULT_TOPVIEW_PATH = "pipette_well_dataset/pipette_well_dataset/Plate_10_clip_0003_Topview.mp4"

# Well ID parsing pattern: matches A-H + optional separator + 1-12
WELL_ID_PATTERN = re.compile(r"\b([A-Ha-h])\s*[- ]?\s*(1[0-2]|[1-9])\b")

ANALYSIS_PROMPT_TEMPLATE = """You are analyzing videos of a pipetting action into a 96-well plate to determine which well(s) receive liquid.

## 96-WELL PLATE LAYOUT
- 8 rows labeled A-H (A at top, H at bottom)
- 12 columns labeled 1-12 (1 at left, 12 at right)
- A1 is TOP-LEFT corner, H12 is BOTTOM-RIGHT corner
- Wells are arranged in an 8x12 grid

## CAMERA VIEWS
- **Topview**: Bird's-eye view looking straight down at the plate. Best for determining X-Y position (row and column).
- **FPV (First-Person View)**: Angled view from the side/front. Best for seeing the pipette tips and confirming dispense action.

## HOW TO IDENTIFY THE TARGET WELL(S)
1. **Use Topview** to determine POSITION: Look at where the pipette tip(s) are positioned over the plate grid.
   - Count columns from the LEFT edge (column 1)
   - Count rows from the TOP edge (row A)
2. **Use FPV** to CONFIRM: See the pipette entering/near the well(s) and any visible dispense action.
3. **For multi-channel pipettes**: Count the number of tips visible - they dispense into multiple wells simultaneously.
   - 8-channel pipettes fill an entire COLUMN (e.g., A1-H1 or A5-H5)
   - 12-channel pipettes fill an entire ROW (e.g., A1-A12 or D1-D12)

## KEY VISUAL CUES
- The dispense happens when the pipette tip(s) are INSIDE or just above the well opening
- Look at the LOWEST point of the pipette motion - this is the dispense position
- Multi-channel pipettes have multiple tips in a straight line
- Single-channel pipettes have ONE tip
- Also pay attention to the thumb as it is used to trigger the pipette

Clip ID: {clip_id}

## YOUR RESPONSE
Return ONLY the well ID(s) where liquid is dispensed:
- Single well: A1
- Multiple wells in a row: A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12
- Multiple wells in a column: A1, B1, C1, D1, E1, F1, G1, H1
- Any pattern: List all wells that receive liquid

Think step-by-step: First identify single vs multi-channel, then determine row/column position.
"""


@dataclass(frozen=True)
class FrameExtractionConfig:
    """Configuration for video frame extraction."""
    stride: int = 3
    max_frames: Optional[int] = None
    start_frame: int = 0
    jpeg_quality: int = 85

    def __post_init__(self):
        if self.stride < 1:
            raise ValueError("stride must be >= 1")
        if self.max_frames is not None and self.max_frames < 1:
            raise ValueError("max_frames must be >= 1 (or None for no cap)")
        if self.start_frame < 0:
            raise ValueError("start_frame must be >= 0")


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for GPT model calls."""
    model: str = "gpt-5-mini"
    max_output_tokens: int = 4000
    reasoning_effort: str = "high"


@dataclass(frozen=True)
class EncodedFrame:
    """A single video frame encoded as a GPT-ready image input."""
    frame_index: int
    data_url: str
    width: Optional[int] = None
    height: Optional[int] = None

    def as_gpt_input_image(self) -> Dict[str, str]:
        return {"type": "input_image", "image_url": self.data_url}


@dataclass
class VideoFramePayload:
    """Container for extracted frames from paired videos."""
    fpv_frames: List[EncodedFrame] = field(default_factory=list)
    topview_frames: List[EncodedFrame] = field(default_factory=list)

    @property
    def fpv_images(self) -> List[Dict[str, str]]:
        return [f.as_gpt_input_image() for f in self.fpv_frames]

    @property
    def topview_images(self) -> List[Dict[str, str]]:
        return [f.as_gpt_input_image() for f in self.topview_frames]

    @property
    def fpv_frame_indices(self) -> List[int]:
        return [f.frame_index for f in self.fpv_frames]

    @property
    def topview_frame_indices(self) -> List[int]:
        return [f.frame_index for f in self.topview_frames]

    @property
    def num_synchronized_frames(self) -> int:
        return min(len(self.fpv_frames), len(self.topview_frames))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fpv_images": self.fpv_images,
            "topview_images": self.topview_images,
            "fpv_frame_indices": self.fpv_frame_indices,
            "topview_frame_indices": self.topview_frame_indices,
        }


# --- Lazy dependency loading ---

def _require_dependency(module_name: str, package_name: str, install_hint: str):
    """Generic dependency loader with helpful error messages."""
    try:
        import importlib
        return importlib.import_module(module_name)
    except ImportError as e:
        raise RuntimeError(
            f"{package_name} is required for this operation.\n"
            f"Install it with: pip install {install_hint}\n"
            f"Original import error: {e}"
        ) from e


def _require_cv2():
    return _require_dependency("cv2", "OpenCV", "opencv-python")


def _require_openai():
    module = _require_dependency("openai", "OpenAI Python SDK", "openai")
    return module.OpenAI


# --- Image encoding utilities ---

def _bgr_to_rgb(bgr_image, cv2):
    return cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)


def _encode_frame_as_jpeg_data_url(rgb_image, cv2, quality: int = 85) -> Tuple[str, int, int]:
    """Encode an RGB image into a JPEG base64 data URL. Returns (data_url, width, height)."""
    bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    ok, buf = cv2.imencode(".jpg", bgr, encode_params)
    if not ok:
        raise RuntimeError("Failed to JPEG-encode a frame with OpenCV.")
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    h, w = rgb_image.shape[:2]
    return f"data:image/jpeg;base64,{b64}", w, h


# --- Video frame extraction ---

def iter_encoded_frames(
    video_path: str,
    config: FrameExtractionConfig,
) -> Iterable[EncodedFrame]:
    """Load an .mp4 and yield encoded frames suitable for GPT image input."""
    cv2 = _require_cv2()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    try:
        if config.start_frame:
            cap.set(cv2.CAP_PROP_POS_FRAMES, config.start_frame)

        yielded = 0
        frame_index = int(cap.get(cv2.CAP_PROP_POS_FRAMES) or config.start_frame)

        while True:
            if config.max_frames is not None and yielded >= config.max_frames:
                break
            ok, frame_bgr = cap.read()
            if not ok:
                break

            if ((frame_index - config.start_frame) % config.stride) == 0:
                rgb = _bgr_to_rgb(frame_bgr, cv2)
                data_url, w, h = _encode_frame_as_jpeg_data_url(rgb, cv2, config.jpeg_quality)
                yield EncodedFrame(frame_index=frame_index, data_url=data_url, width=w, height=h)
                yielded += 1

            frame_index += 1
    finally:
        cap.release()


def extract_video_frames(
    fpv_path: str,
    topview_path: str,
    config: Optional[FrameExtractionConfig] = None,
) -> VideoFramePayload:
    """Extract frames from both FPV and Topview videos."""
    config = config or FrameExtractionConfig()
    return VideoFramePayload(
        fpv_frames=list(iter_encoded_frames(fpv_path, config)),
        topview_frames=list(iter_encoded_frames(topview_path, config)),
    )


# --- Clip ID utilities ---

def get_clip_id(video_path: str) -> str:
    """Extract clip ID from video path (e.g., 'Plate_1_clip_0001_FPV')."""
    return Path(video_path).stem


def get_base_clip_id(clip_id: str) -> str:
    """Normalize clip ID by removing viewpoint suffix (FPV/Topview)."""
    return re.sub(r"_(?:FPV|Topview)$", "", clip_id)


def get_unified_clip_id(fpv_path: str, topview_path: str) -> str:
    """Get a unified clip ID from paired video paths."""
    fpv_id = get_base_clip_id(get_clip_id(fpv_path))
    topview_id = get_base_clip_id(get_clip_id(topview_path))
    return fpv_id if fpv_id == topview_id else f"{fpv_id} / {topview_id}"


# --- Well ID parsing ---

def parse_well_ids(text: str) -> List[Dict[str, str]]:
    """Parse well IDs from free-form model output."""
    seen: set[tuple[str, str]] = set()
    wells: List[Dict[str, str]] = []
    for row, col in WELL_ID_PATTERN.findall(text):
        row_u = row.upper()
        col_s = str(int(col))
        key = (row_u, col_s)
        if key not in seen:
            seen.add(key)
            wells.append({"well_row": row_u, "well_column": col_s})
    return wells


# --- GPT content building ---

def _is_key_frame(frame_idx: int, total_frames: int) -> bool:
    """Determine if a frame is in the key middle portion of the video."""
    if total_frames <= 3:
        return False
    middle_start = total_frames // 3
    middle_end = (2 * total_frames) // 3
    return middle_start <= frame_idx <= middle_end


def build_gpt_content(
    payload: VideoFramePayload,
    clip_id: str,
) -> List[Dict[str, Any]]:
    """Build the content array for GPT input."""
    content: List[Dict[str, Any]] = [
        {"type": "input_text", "text": ANALYSIS_PROMPT_TEMPLATE.format(clip_id=clip_id)}
    ]

    num_frames = payload.num_synchronized_frames
    content.append({
        "type": "input_text",
        "text": (
            f"Below are {num_frames} synchronized frame pairs from the video.\n"
            "Each pair shows the SAME moment from both camera angles.\n"
            "Focus especially on the MIDDLE frames where the dispense action occurs.\n"
        ),
    })

    for i in range(num_frames):
        fpv_idx = payload.fpv_frame_indices[i] if i < len(payload.fpv_frame_indices) else "?"
        top_idx = payload.topview_frame_indices[i] if i < len(payload.topview_frame_indices) else "?"
        
        frame_marker = " [KEY FRAME - likely dispense action]" if _is_key_frame(i, num_frames) else ""
        
        content.extend([
            {"type": "input_text", "text": f"--- Frame pair {i+1}/{num_frames}{frame_marker} ---"},
            {"type": "input_text", "text": f"Topview (frame {top_idx}) - Use this to identify row/column position:"},
            payload.topview_images[i],
            {"type": "input_text", "text": f"FPV (frame {fpv_idx}) - Use this to confirm pipette action:"},
            payload.fpv_images[i],
        ])

    return content


# --- Response handling ---

def print_model_response_debug(resp, result_text: str) -> None:
    """Print debug information about the model response."""
    print("\n=== RAW MODEL RESPONSE OBJECT ===")
    if hasattr(resp, "model_dump_json"):
        try:
            print(resp.model_dump_json(indent=2))
        except TypeError:
            print(resp.model_dump_json())
    elif hasattr(resp, "model_dump"):
        print(json.dumps(resp.model_dump(), indent=2))
    else:
        print(resp)
    print("=== MODEL OUTPUT TEXT (resp.output_text) ===")
    print(result_text.strip() or "<empty>")
    print("=== END RAW MODEL RESPONSE ===\n")


def call_gpt_model(
    client,
    content: List[Dict[str, Any]],
    model_config: ModelConfig,
) -> str:
    """Call the GPT model and return the output text."""
    try:
        resp = client.responses.create(
            model=model_config.model,
            input=[{"role": "user", "content": content}],
            max_output_tokens=model_config.max_output_tokens,
            reasoning={"effort": model_config.reasoning_effort},
        )
    except TypeError:
        # Fallback for SDKs that don't support reasoning parameter
        resp = client.responses.create(
            model=model_config.model,
            input=[{"role": "user", "content": content}],
            max_output_tokens=model_config.max_output_tokens,
        )
    return resp, getattr(resp, "output_text", None) or ""


# --- Main prediction function ---

def predict_wells(
    fpv_video_path: str,
    topview_video_path: str,
    frame_config: Optional[FrameExtractionConfig] = None,
    model_config: Optional[ModelConfig] = None,
    include_raw_model_output: bool = False,
    print_response: bool = False,
) -> Dict[str, Any]:
    """
    Analyze pipette videos and predict target well(s).
    
    This is the main entry point for well prediction.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY environment variable.")

    frame_config = frame_config or FrameExtractionConfig()
    model_config = model_config or ModelConfig()

    # Extract frames from videos
    payload = extract_video_frames(fpv_video_path, topview_video_path, frame_config)
    
    # Build GPT content
    clip_id = get_unified_clip_id(fpv_video_path, topview_video_path)
    content = build_gpt_content(payload, clip_id)

    # Call model
    OpenAI = _require_openai()
    client = OpenAI(api_key=api_key)
    resp, result_text = call_gpt_model(client, content, model_config)

    if print_response:
        print_model_response_debug(resp, result_text)

    # Build output
    result: Dict[str, Any] = {
        "clip_id_FPV": get_clip_id(fpv_video_path),
        "clip_id_Topview": get_clip_id(topview_video_path),
        "wells_prediction": parse_well_ids(result_text),
    }
    if include_raw_model_output:
        result["raw_model_output"] = result_text.strip()
    return result


# --- Legacy API compatibility ---

def iter_encoded_frames_from_video(
    video_path: str,
    *,
    stride: int = 3,
    max_frames: Optional[int] = None,
    start_frame: int = 0,
    jpeg_quality: int = 85,
) -> Iterable[EncodedFrame]:
    """Legacy wrapper for iter_encoded_frames."""
    config = FrameExtractionConfig(
        stride=stride,
        max_frames=max_frames,
        start_frame=start_frame,
        jpeg_quality=jpeg_quality,
    )
    return iter_encoded_frames(video_path, config)


def build_gpt5_nano_image_inputs_for_videos(
    fpv_path: str,
    topview_path: str,
    *,
    stride: int = 3,
    max_frames_each: Optional[int] = None,
    start_frame: int = 0,
    jpeg_quality: int = 85,
) -> Dict[str, Any]:
    """Legacy wrapper for extract_video_frames."""
    config = FrameExtractionConfig(
        stride=stride,
        max_frames=max_frames_each,
        start_frame=start_frame,
        jpeg_quality=jpeg_quality,
    )
    payload = extract_video_frames(fpv_path, topview_path, config)
    return payload.to_dict()


def ask_gpt5_nano_for_well_prediction(
    *,
    fpv_video_path: str,
    topview_video_path: str,
    model: str = "gpt-5-mini",
    stride: int = 3,
    max_frames_each: Optional[int] = None,
    start_frame: int = 0,
    jpeg_quality: int = 85,
    max_output_tokens: int = 4000,
    reasoning_effort: str = "high",
    include_raw_model_output: bool = False,
    print_model_response: bool = False,
) -> Dict[str, Any]:
    """Legacy wrapper for predict_wells."""
    frame_config = FrameExtractionConfig(
        stride=stride,
        max_frames=max_frames_each,
        start_frame=start_frame,
        jpeg_quality=jpeg_quality,
    )
    model_config = ModelConfig(
        model=model,
        max_output_tokens=max_output_tokens,
        reasoning_effort=reasoning_effort,
    )
    return predict_wells(
        fpv_video_path,
        topview_video_path,
        frame_config=frame_config,
        model_config=model_config,
        include_raw_model_output=include_raw_model_output,
        print_response=print_model_response,
    )


def analyze_pipette_well(
    fpv_video_path: str,
    topview_video_path: str,
) -> Dict[str, Any]:
    """Convenience function for simple well analysis."""
    return predict_wells(fpv_video_path, topview_video_path)


# --- CLI ---

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze pipette videos to predict target wells."
    )
    
    # Video paths
    parser.add_argument("--fpv", default=DEFAULT_FPV_PATH, help="Path to FPV .mp4")
    parser.add_argument("--topview", default=DEFAULT_TOPVIEW_PATH, help="Path to Topview .mp4")
    
    # Frame extraction options
    parser.add_argument("--stride", type=int, default=3,
                        help="Take every Nth frame (default: 3)")
    parser.add_argument("--max-frames-each", type=int, default=0,
                        help="Max frames per video (0 = no cap)")
    parser.add_argument("--start-frame", type=int, default=0,
                        help="Start decoding at this frame index")
    parser.add_argument("--jpeg-quality", type=int, default=85,
                        help="JPEG quality 1-100 (default: 85)")
    
    # Model options
    parser.add_argument("--run-gpt", action="store_true",
                        help="Call the model and print JSON prediction")
    parser.add_argument("--model", default="gpt-5-mini",
                        help="Model name (default: gpt-5-mini)")
    parser.add_argument("--max-output-tokens", type=int, default=100_000,
                        help="Max tokens for model response")
    parser.add_argument("--reasoning-effort", default="high",
                        choices=["low", "medium", "high"],
                        help="Reasoning effort hint (default: high)")
    
    # Output options
    parser.add_argument("--include-raw", action="store_true",
                        help="Include raw model output in JSON")
    parser.add_argument("--print-model-response", action="store_true",
                        help="Print full raw SDK response")
    
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    
    max_frames = None if args.max_frames_each == 0 else args.max_frames_each
    frame_config = FrameExtractionConfig(
        stride=args.stride,
        max_frames=max_frames,
        start_frame=args.start_frame,
        jpeg_quality=args.jpeg_quality,
    )

    if args.run_gpt:
        model_config = ModelConfig(
            model=args.model,
            max_output_tokens=args.max_output_tokens,
            reasoning_effort=args.reasoning_effort,
        )
        prediction = predict_wells(
            args.fpv,
            args.topview,
            frame_config=frame_config,
            model_config=model_config,
            include_raw_model_output=args.include_raw,
            print_response=args.print_model_response,
        )
        print(json.dumps(prediction, indent=2))
        return 0

    # Preview mode: just show frame extraction info
    payload = extract_video_frames(args.fpv, args.topview, frame_config)
    print(
        f"Sampling mode: uniform (stride={args.stride})\n"
        f"FPV frames: {len(payload.fpv_frames)}, Topview frames: {len(payload.topview_frames)}\n"
        f"Frame indices (FPV): {payload.fpv_frame_indices}\n"
        f"Example fpv_images[0] keys: {list(payload.fpv_images[0].keys()) if payload.fpv_images else 'n/a'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
