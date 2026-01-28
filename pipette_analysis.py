from __future__ import annotations

import argparse
import base64
import json
import os
import re
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    import dotenv  # type: ignore

    dotenv.load_dotenv()
except Exception:
    # Optional dependency; environment variables can still come from the shell.
    pass
fpv_video_path = "pipette_well_dataset/pipette_well_dataset/Plate_10_clip_0003_FPV.mp4"
topview_video_path = "pipette_well_dataset/pipette_well_dataset/Plate_10_clip_0003_Topview.mp4"
# fpv_video_path = "pipette_well_dataset/pipette_well_dataset/Plate_1_clip_0001_FPV.mp4"
# topview_video_path = "pipette_well_dataset/pipette_well_dataset/Plate_1_clip_0001_Topview.mp4"


@dataclass(frozen=True)
class EncodedFrame:
    """A single video frame encoded as a GPT-ready image input."""

    frame_index: int
    data_url: str
    width: Optional[int] = None
    height: Optional[int] = None

    def as_gpt_input_image(self) -> Dict[str, str]:
        # OpenAI "image input" shape (Responses API-style content part)
        return {"type": "input_image", "image_url": self.data_url}


def _require_cv2():
    try:
        import cv2  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "OpenCV is required to decode .mp4 files in this project.\n"
            "Install it with: pip install opencv-python\n"
            f"Original import error: {e}"
        ) from e
    return cv2


def _require_openai():
    try:
        from openai import OpenAI  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "The OpenAI Python SDK is required to call gpt-5-mini.\n"
            "Install it with: pip install openai\n"
            f"Original import error: {e}"
        ) from e
    return OpenAI


def _bgr_to_rgb(bgr_image, cv2):
    return cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)


def _encode_jpeg_data_url_from_rgb(rgb_image, cv2, *, quality: int = 85) -> Tuple[str, int, int]:
    """
    Encode an RGB image (numpy array) into a JPEG base64 data URL.
    Returns (data_url, width, height).
    """
    # cv2.imencode expects BGR, so convert back for encoding to avoid Pillow dependency.
    bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    ok, buf = cv2.imencode(".jpg", bgr, encode_params)
    if not ok:  # pragma: no cover
        raise RuntimeError("Failed to JPEG-encode a frame with OpenCV.")
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    h, w = rgb_image.shape[:2]
    return f"data:image/jpeg;base64,{b64}", w, h


def iter_encoded_frames_from_video(
    video_path: str,
    *,
    stride: int = 3,
    max_frames: Optional[int] = None,
    start_frame: int = 0,
    jpeg_quality: int = 85,
) -> Iterable[EncodedFrame]:
    """
    Load an .mp4 and yield encoded frames suitable for GPT image input.

    - stride: take every Nth frame (after start_frame). Use 3 for every third frame.
    - max_frames: cap how many frames to return. Use None for no cap (all frames).
    """
    if stride < 1:
        raise ValueError("stride must be >= 1")
    if max_frames is not None and max_frames < 1:
        raise ValueError("max_frames must be >= 1 (or None for no cap)")
    if start_frame < 0:
        raise ValueError("start_frame must be >= 0")

    cv2 = _require_cv2()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    try:
        # Seek to start_frame if possible
        if start_frame:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        yielded = 0
        frame_index = int(cap.get(cv2.CAP_PROP_POS_FRAMES) or start_frame)

        while True:
            if max_frames is not None and yielded >= max_frames:
                break
            ok, frame_bgr = cap.read()
            if not ok:
                break

            # Note: CAP_PROP_POS_FRAMES is not always reliable across codecs, so track manually.
            take = ((frame_index - start_frame) % stride) == 0
            if take:
                rgb = _bgr_to_rgb(frame_bgr, cv2)
                data_url, w, h = _encode_jpeg_data_url_from_rgb(rgb, cv2, quality=jpeg_quality)
                yield EncodedFrame(frame_index=frame_index, data_url=data_url, width=w, height=h)
                yielded += 1

            frame_index += 1
    finally:
        cap.release()


def build_gpt5_nano_image_inputs_for_videos(
    fpv_path: str,
    topview_path: str,
    *,
    stride: int = 3,
    max_frames_each: Optional[int] = None,
    start_frame: int = 0,
    jpeg_quality: int = 85,
) -> Dict[str, Any]:
    """
    Produce a minimal, "ready to send" payload fragment consisting ONLY of images.

    Returns a dict:
      {
        "fpv_images": [ {"type":"input_image","image_url":"data:..."} , ... ],
        "topview_images": [ ... ]
      }
    """
    fpv_frames = list(
        iter_encoded_frames_from_video(
            fpv_path,
            stride=stride,
            max_frames=max_frames_each,
            start_frame=start_frame,
            jpeg_quality=jpeg_quality,
        )
    )
    top_frames = list(
        iter_encoded_frames_from_video(
            topview_path,
            stride=stride,
            max_frames=max_frames_each,
            start_frame=start_frame,
            jpeg_quality=jpeg_quality,
        )
    )

    return {
        "fpv_images": [f.as_gpt_input_image() for f in fpv_frames],
        "topview_images": [f.as_gpt_input_image() for f in top_frames],
        # Keeping indices is useful for debugging / alignment, but it's not required by GPT.
        "fpv_frame_indices": [f.frame_index for f in fpv_frames],
        "topview_frame_indices": [f.frame_index for f in top_frames],
    }


def _clip_id_from_path(video_path: str) -> str:
    # E.g. ".../Plate_1_clip_0001_FPV.mp4" -> "Plate_1_clip_0001_FPV"
    return Path(video_path).stem


def _base_clip_id(clip_id: str) -> str:
    """
    Normalize a clip id across viewpoints.

    Example:
      Plate_1_clip_0001_FPV -> Plate_1_clip_0001
      Plate_1_clip_0001_Topview -> Plate_1_clip_0001
    """
    return re.sub(r"_(?:FPV|Topview)$", "", clip_id)


def _parse_well_ids(text: str) -> List[Dict[str, str]]:
    """
    Parse well IDs from free-form model output.

    Accepts patterns like:
    - "A1"
    - "A 1"
    - "A-1"
    - "A1, A2, A3"
    - "row A column 1" (will usually still contain "A 1")
    """
    # Match A-H + optional separator + 1-12
    pattern = re.compile(r"\b([A-Ha-h])\s*[- ]?\s*(1[0-2]|[1-9])\b")
    seen: set[tuple[str, str]] = set()
    wells: List[Dict[str, str]] = []
    for row, col in pattern.findall(text):
        row_u = row.upper()
        col_s = str(int(col))  # normalize "01" -> "1" if it ever appears
        key = (row_u, col_s)
        if key in seen:
            continue
        seen.add(key)
        wells.append({"well_row": row_u, "well_column": col_s})
    return wells


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
    """
    Calls a GPT model with extracted frames and returns a JSON prediction that we build locally.

    Requires OPENAI_API_KEY in your environment.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY environment variable.")

    payload = build_gpt5_nano_image_inputs_for_videos(
        fpv_video_path,
        topview_video_path,
        stride=stride,
        max_frames_each=max_frames_each,
        start_frame=start_frame,
        jpeg_quality=jpeg_quality,
    )

    OpenAI = _require_openai()
    client = OpenAI(api_key=api_key)

    clip_id_fpv = _clip_id_from_path(fpv_video_path)
    clip_id_top = _clip_id_from_path(topview_video_path)
    base_id_fpv = _base_clip_id(clip_id_fpv)
    base_id_top = _base_clip_id(clip_id_top)
    base_id = base_id_fpv if base_id_fpv == base_id_top else f"{base_id_fpv} / {base_id_top}"

    prompt = (
        "You are analyzing videos of a pipetting action into a 96-well plate to determine which well(s) receive liquid.\n\n"
        
        "## 96-WELL PLATE LAYOUT\n"
        "- 8 rows labeled A-H (A at top, H at bottom)\n"
        "- 12 columns labeled 1-12 (1 at left, 12 at right)\n"
        "- A1 is TOP-LEFT corner, H12 is BOTTOM-RIGHT corner\n"
        "- Wells are arranged in an 8x12 grid\n\n"
        
        "## CAMERA VIEWS\n"
        "- **Topview**: Bird's-eye view looking straight down at the plate. Best for determining X-Y position (row and column).\n"
        "- **FPV (First-Person View)**: Angled view from the side/front. Best for seeing the pipette tips and confirming dispense action.\n\n"
        
        "## HOW TO IDENTIFY THE TARGET WELL(S)\n"
        "1. **Use Topview** to determine POSITION: Look at where the pipette tip(s) are positioned over the plate grid.\n"
        "   - Count columns from the LEFT edge (column 1)\n"
        "   - Count rows from the TOP edge (row A)\n"
        "2. **Use FPV** to CONFIRM: See the pipette entering/near the well(s) and any visible dispense action.\n"
        "3. **For multi-channel pipettes**: Count the number of tips visible - they dispense into multiple wells simultaneously.\n"
        "   - 8-channel pipettes fill an entire COLUMN (e.g., A1-H1 or A5-H5)\n"
        "   - 12-channel pipettes fill an entire ROW (e.g., A1-A12 or D1-D12)\n\n"
        
        "## KEY VISUAL CUES\n"
        "- The dispense happens when the pipette tip(s) are INSIDE or just above the well opening\n"
        "- Look at the LOWEST point of the pipette motion - this is the dispense position\n"
        "- Multi-channel pipettes have multiple tips in a straight line\n"
        "- Single-channel pipettes have ONE tip\n\n"
        
        f"Clip ID: {base_id}\n\n"
        
        "## YOUR RESPONSE\n"
        "Return ONLY the well ID(s) where liquid is dispensed:\n"
        "- Single well: A1\n"
        "- Multiple wells in a row: A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12\n"
        "- Multiple wells in a column: A1, B1, C1, D1, E1, F1, G1, H1\n"
        "- Any pattern: List all wells that receive liquid\n\n"
        "Think step-by-step: First identify single vs multi-channel, then determine row/column position.\n"
    )

    content: List[Dict[str, str]] = [{"type": "input_text", "text": prompt}]
    
    # Interleave FPV and Topview frames to show both views at each time point
    fpv_images = payload["fpv_images"]
    top_images = payload["topview_images"]
    fpv_indices = payload["fpv_frame_indices"]
    top_indices = payload["topview_frame_indices"]
    
    num_frames = min(len(fpv_images), len(top_images))
    
    content.append(
        {
            "type": "input_text",
            "text": (
                f"Below are {num_frames} synchronized frame pairs from the video.\n"
                "Each pair shows the SAME moment from both camera angles.\n"
                "Focus especially on the MIDDLE frames where the dispense action occurs.\n"
                "---"
            ),
        }
    )
    
    for i in range(num_frames):
        fpv_idx = fpv_indices[i] if i < len(fpv_indices) else "?"
        top_idx = top_indices[i] if i < len(top_indices) else "?"
        
        # Mark key frames (middle portion of video where action likely happens)
        frame_marker = ""
        if num_frames > 3:
            middle_start = num_frames // 3
            middle_end = (2 * num_frames) // 3
            if middle_start <= i <= middle_end:
                frame_marker = " [KEY FRAME - likely dispense action]"
        
        content.append({"type": "input_text", "text": f"--- Frame pair {i+1}/{num_frames}{frame_marker} ---"})
        content.append({"type": "input_text", "text": f"Topview (frame {top_idx}) - Use this to identify row/column position:"})
        content.append(top_images[i])
        content.append({"type": "input_text", "text": f"FPV (frame {fpv_idx}) - Use this to confirm pipette action:"})
        content.append(fpv_images[i])

    # Prefer explicit reasoning effort when supported by the SDK/model.
    # If the local OpenAI SDK doesn't support this parameter yet, fall back gracefully.
    try:
        resp = client.responses.create(
            model=model,
            input=[{"role": "user", "content": content}],
            max_output_tokens=max_output_tokens,
            reasoning={"effort": reasoning_effort},
        )
    except TypeError:
        resp = client.responses.create(
            model=model,
            input=[{"role": "user", "content": content}],
            max_output_tokens=max_output_tokens,
        )

    # The SDK exposes output text aggregated across message parts.
    result_text = getattr(resp, "output_text", None) or ""
    if print_model_response:
        # Best-effort printing of the raw response object across SDK versions.
        print("\n=== RAW MODEL RESPONSE OBJECT ===")
        if hasattr(resp, "model_dump_json"):
            try:
                print(resp.model_dump_json(indent=2))
            except TypeError:
                # Some SDK/pydantic combos don't support indent=...
                print(resp.model_dump_json())
        elif hasattr(resp, "model_dump"):
            print(json.dumps(resp.model_dump(), indent=2))
        else:
            print(resp)
        print("=== MODEL OUTPUT TEXT (resp.output_text) ===")
        print(result_text.strip() or "<empty>")
        print("=== END RAW MODEL RESPONSE ===\n")
    wells_prediction = _parse_well_ids(result_text)

    out: Dict[str, Any] = {
        "clip_id_FPV": clip_id_fpv,
        "clip_id_Topview": clip_id_top,
        "wells_prediction": wells_prediction,
    }
    if include_raw_model_output:
        out["raw_model_output"] = result_text.strip()
    return out


def analyze_pipette_well(
    fpv_video_path: str,
    topview_video_path: str,
) -> Dict[str, Any]:
    """
    Analyze a pipette-well clip pair and return the predicted well(s).

    This uses gpt-5-mini with frames extracted from each video.
    """
    return ask_gpt5_nano_for_well_prediction(
        fpv_video_path=fpv_video_path,
        topview_video_path=topview_video_path,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Split pipette videos into GPT-ready image inputs (and optionally call GPT).")
    parser.add_argument("--fpv", default=fpv_video_path, help="Path to FPV .mp4")
    parser.add_argument("--topview", default=topview_video_path, help="Path to Topview .mp4")
    parser.add_argument("--stride", type=int, default=3, help="Take every Nth frame (3 = every third frame)")
    parser.add_argument(
        "--max-frames-each",
        type=int,
        default=0,
        help="Max frames per video (0 = no cap; send all frames)",
    )
    parser.add_argument("--start-frame", type=int, default=0, help="Start decoding at this frame index")
    parser.add_argument("--jpeg-quality", type=int, default=85, help="JPEG quality (1-100)")
    parser.add_argument("--run-gpt", action="store_true", help="Call the model and print JSON prediction")
    parser.add_argument("--model", default="gpt-5-mini", help="Model name (default: gpt-5-mini)")
    parser.add_argument("--max-output-tokens", type=int, default=100_000, help="Max tokens for the model response")
    parser.add_argument(
        "--reasoning-effort",
        default="high",
        choices=["low", "medium", "high"],
        help="Reasoning effort hint for the model (default: high)",
    )
    parser.add_argument("--include-raw", action="store_true", help="Include raw model output in printed JSON")
    parser.add_argument(
        "--print-model-response",
        action="store_true",
        help="Print the full raw SDK response object and output text",
    )
    args = parser.parse_args()
    max_frames_each: Optional[int] = None if int(args.max_frames_each) == 0 else int(args.max_frames_each)

    if args.run_gpt:
        prediction = ask_gpt5_nano_for_well_prediction(
            fpv_video_path=args.fpv,
            topview_video_path=args.topview,
            model=args.model,
            stride=args.stride,
            max_frames_each=max_frames_each,
            start_frame=args.start_frame,
            jpeg_quality=args.jpeg_quality,
            max_output_tokens=args.max_output_tokens,
            reasoning_effort=args.reasoning_effort,
            include_raw_model_output=args.include_raw,
            print_model_response=args.print_model_response,
        )
        print(json.dumps(prediction, indent=2))
        return 0

    payload = build_gpt5_nano_image_inputs_for_videos(
        args.fpv,
        args.topview,
        stride=args.stride,
        max_frames_each=max_frames_each,
        start_frame=args.start_frame,
        jpeg_quality=args.jpeg_quality,
    )

    # Print counts and a tiny preview (not the full base64 strings).
    print(
        f"Sampling mode: uniform (stride={args.stride})\n"
        f"FPV frames: {len(payload['fpv_images'])}, Topview frames: {len(payload['topview_images'])}\n"
        f"Frame indices (FPV): {payload['fpv_frame_indices']}\n"
        f"Example fpv_images[0] keys: {list(payload['fpv_images'][0].keys()) if payload['fpv_images'] else 'n/a'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())