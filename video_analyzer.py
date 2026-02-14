import argparse
import glob
import json
import os
import re
from typing import Any, Optional

from transformers import pipeline


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-VL-8B-Thinking")
    parser.add_argument("--cache_dir", type=str, default="/work/hdd/bgea/${USER}/hyperhyper/models")

    # Batch mode (default): iterate vid_to_img for clip pairs.
    parser.add_argument(
        "--base_dir",
        type=str,
        default="vid_to_img",
        help="Folder containing per-clip *_FPV and *_Topview subfolders.",
    )
    parser.add_argument("--limit_clips", type=int, default=10, help="How many clip pairs to run (0 means no limit).")

    # Single mode: explicitly pick one pair (overrides --base_dir scan).
    parser.add_argument("--fpv_dir", type=str, default="", help="Run only this FPV directory if set.")
    parser.add_argument("--topview_dir", type=str, default="", help="Run only this Topview directory if set.")

    parser.add_argument("--stride", type=int, default=1, help="Use every Nth frame from each folder.")
    parser.add_argument("--max_frames", type=int, default=0, help="0 means no limit.")
    parser.add_argument("--debug", action="store_true", help="Print raw pipeline output.")
    return parser.parse_args()


def _list_images(dir_path: str) -> list[str]:
    patterns = ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp")
    files: list[str] = []
    for pat in patterns:
        files.extend(glob.glob(os.path.join(dir_path, pat)))
    return sorted(set(files))


def _chat_turn_to_text(turn: Any) -> str:
    if isinstance(turn, dict) and "content" in turn:
        content = turn["content"]
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for seg in content:
                if isinstance(seg, str):
                    parts.append(seg)
                elif isinstance(seg, dict) and "text" in seg and isinstance(seg["text"], str):
                    parts.append(seg["text"])
            return "\n".join(parts).strip()
    return ""


def _chat_turns_to_text(turns: Any) -> str:
    if not isinstance(turns, list):
        return ""
    assistant_texts = []
    any_texts = []
    for t in turns:
        txt = _chat_turn_to_text(t)
        if txt:
            any_texts.append(txt)
            if isinstance(t, dict) and t.get("role") == "assistant":
                assistant_texts.append(txt)
    if assistant_texts:
        return "\n".join(assistant_texts).strip()
    return "\n".join(any_texts).strip()


def _response_to_text(response: Any) -> str:
    if response is None:
        return ""
    if isinstance(response, str):
        return response
    if isinstance(response, dict):
        if "role" in response and "content" in response:
            return _chat_turn_to_text(response).strip()
        for k in ("generated_text", "text", "output_text"):
            if k in response:
                return _response_to_text(response[k])
        return json.dumps(response, ensure_ascii=False)
    if isinstance(response, list):
        if not response:
            return ""
        if isinstance(response[0], dict):
            # Either list[turn] or list[{"generated_text": ...}]
            if "role" in response[0] and "content" in response[0]:
                chat_txt = _chat_turns_to_text(response)
                if chat_txt:
                    return chat_txt
            for k in ("generated_text", "text", "output_text"):
                if k in response[0]:
                    return _response_to_text(response[0][k])
            return json.dumps(response[0], ensure_ascii=False)
        return "\n".join(_response_to_text(x) for x in response)
    if isinstance(response, tuple):
        return "\n".join(_response_to_text(x) for x in response)
    return str(response)


def extract_answer(response: Any) -> str:
    text = _response_to_text(response).strip()
    m = re.search(r"(?is)\bAnswer:\s*(.+?)\s*$", text)
    if m:
        return m.group(1).strip()
    m2 = re.search(r"(?is)\bAnswer:\s*(.+)", text)
    if m2:
        return m2.group(1).strip()
    return text


def _parse_int(s: str) -> Optional[int]:
    try:
        return int(s)
    except Exception:
        return None


def _row_range(a: str, b: str) -> list[str]:
    rows = "ABCDEFGH"
    ia = rows.find(a)
    ib = rows.find(b)
    if ia == -1 or ib == -1:
        return []
    if ia <= ib:
        return list(rows[ia : ib + 1])
    return list(rows[ib : ia + 1])


def parse_wells(text: str) -> set[tuple[str, int]]:
    wells: set[tuple[str, int]] = set()
    if not text:
        return wells

    t = text.upper()

    # ROW A ... COLUMNS 1-8
    for m in re.finditer(r"\bROW\s*([A-H])\b.*?\bCOLUMNS?\s*(\d{1,2})\s*(?:-|TO|THROUGH)\s*(\d{1,2})\b", t):
        row = m.group(1)
        c1 = _parse_int(m.group(2))
        c2 = _parse_int(m.group(3))
        if c1 is None or c2 is None:
            continue
        lo, hi = sorted((c1, c2))
        for c in range(lo, hi + 1):
            if 1 <= c <= 12:
                wells.add((row, c))

    # ROWS A-H ... COLUMN 1
    for m in re.finditer(r"\bROWS?\s*([A-H])\s*(?:-|TO|THROUGH)\s*([A-H])\b.*?\bCOLUMN\s*(\d{1,2})\b", t):
        r1, r2 = m.group(1), m.group(2)
        col = _parse_int(m.group(3))
        if col is None or not (1 <= col <= 12):
            continue
        for r in _row_range(r1, r2):
            wells.add((r, col))

    # Ranges like A1-A8 or A1 to A8
    for m in re.finditer(r"\b([A-H])\s*0?(\d{1,2})\s*(?:-|TO|THROUGH)\s*([A-H])\s*0?(\d{1,2})\b", t):
        r1, r2 = m.group(1), m.group(3)
        c1 = _parse_int(m.group(2))
        c2 = _parse_int(m.group(4))
        if c1 is None or c2 is None:
            continue
        if r1 == r2:
            lo, hi = sorted((c1, c2))
            for c in range(lo, hi + 1):
                if 1 <= c <= 12:
                    wells.add((r1, c))
        elif c1 == c2:
            if 1 <= c1 <= 12:
                for r in _row_range(r1, r2):
                    wells.add((r, c1))
        else:
            if 1 <= c1 <= 12:
                wells.add((r1, c1))
            if 1 <= c2 <= 12:
                wells.add((r2, c2))

    # Explicit wells like A1, B12 (also catches those inside longer text)
    for m in re.finditer(r"\b([A-H])\s*0?(1[0-2]|[1-9])\b", t):
        r = m.group(1)
        c = _parse_int(m.group(2))
        if c is not None and 1 <= c <= 12:
            wells.add((r, c))

    return wells


def format_wells(wells: set[tuple[str, int]]) -> str:
    rows = "ABCDEFGH"
    return ",".join(f"{r}{c}" for (r, c) in sorted(wells, key=lambda x: (rows.index(x[0]), x[1])))


def load_ground_truth_map(labels_path: str) -> dict[str, set[tuple[str, int]]]:
    with open(labels_path, "r") as f:
        items = json.load(f)
    gt: dict[str, set[tuple[str, int]]] = {}
    for it in items:
        clip_id = it.get("clip_id_FPV")
        wells = set()
        for w in it.get("wells_ground_truth", []):
            row = str(w.get("well_row", "")).upper()
            col = _parse_int(str(w.get("well_column", "")).strip())
            if row in "ABCDEFGH" and col is not None and 1 <= col <= 12:
                wells.add((row, col))
        if clip_id:
            gt[str(clip_id)] = wells
    return gt


def iter_clip_pairs(base_dir: str) -> list[tuple[str, str, str]]:
    pairs: list[tuple[str, str, str]] = []
    if not os.path.isdir(base_dir):
        return pairs
    for name in sorted(os.listdir(base_dir)):
        if not name.endswith("_FPV"):
            continue
        fpv_dir = os.path.join(base_dir, name)
        if not os.path.isdir(fpv_dir):
            continue
        top_name = name[: -len("_FPV")] + "_Topview"
        top_dir = os.path.join(base_dir, top_name)
        if not os.path.isdir(top_dir):
            continue
        pairs.append((name, fpv_dir, top_dir))
    return pairs


def main():
    args = parse_args()
    cache_dir = os.path.expandvars(args.cache_dir)
    pipe = pipeline("image-text-to-text", model=args.model, cache_dir=cache_dir)

    gt_map = load_ground_truth_map("pipette_well_dataset/pipette_well_dataset/labels.json")

    prompt = """Which well or wells is the pipette dispensing into in this 96-well plate? Provide the well row (A-H) and column (1-12) for each well. First identify single vs multi-channel, then determine row/column position. If its a single channel, provide the well row and column. If its a multi-channel, provide the well row and column for each channel. Return the final answer in \"Answer: <answer>\" format."""

    if args.fpv_dir and args.topview_dir:
        clip_jobs = [(os.path.basename(os.path.normpath(args.fpv_dir)), args.fpv_dir, args.topview_dir)]
    else:
        clip_jobs = iter_clip_pairs(args.base_dir)
        if args.limit_clips and args.limit_clips > 0:
            clip_jobs = clip_jobs[: int(args.limit_clips)]

    if not clip_jobs:
        raise RuntimeError(
            "No clip pairs found. Provide --fpv_dir/--topview_dir or ensure --base_dir contains *_FPV and *_Topview folders."
        )

    total = 0
    exact = 0
    sum_jacc = 0.0
    missing_gt = 0

    for clip_id_fpv, fpv_dir, top_dir in clip_jobs:
        fpv_paths = _list_images(fpv_dir)
        top_paths = _list_images(top_dir)
        if not fpv_paths or not top_paths:
            continue

        fpv_by_name = {os.path.basename(p): p for p in fpv_paths}
        top_by_name = {os.path.basename(p): p for p in top_paths}
        common_names = sorted(fpv_by_name.keys() & top_by_name.keys())
        if not common_names:
            continue

        stride = max(1, int(args.stride))
        selected_names = common_names[::stride]
        if args.max_frames and args.max_frames > 0:
            selected_names = selected_names[: int(args.max_frames)]

        content = []
        for name in selected_names:
            content.append({"type": "image", "path": fpv_by_name[name]})
            content.append({"type": "image", "path": top_by_name[name]})
        content.append({"type": "text", "text": prompt})

        messages = [{"role": "user", "content": content}]
        response = pipe(text=messages, max_new_tokens=16384)
        if args.debug:
            print({"clip": clip_id_fpv, "raw": response})

        extracted = extract_answer(response)
        pred_wells = parse_wells(extracted)
        pred_fmt = format_wells(pred_wells)

        gt_wells = gt_map.get(clip_id_fpv)
        if gt_wells is None:
            missing_gt += 1
            gt_wells = set()
        gt_fmt = format_wells(gt_wells)

        total += 1
        is_exact = pred_wells == gt_wells
        if is_exact:
            exact += 1

        union = pred_wells | gt_wells
        inter = pred_wells & gt_wells
        jacc = (len(inter) / len(union)) if union else (1.0 if not pred_wells and not gt_wells else 0.0)
        sum_jacc += jacc

        print(f"{clip_id_fpv} | pred={pred_fmt} | gt={gt_fmt} | exact={is_exact} | jaccard={jacc:.3f}")

    if total == 0:
        raise RuntimeError("No clips were processed (no images found?).")

    print(
        f"SUMMARY | clips={total} | exact={exact}/{total} ({(exact/total):.3f}) | "
        f"mean_jaccard={(sum_jacc/total):.3f} | missing_gt={missing_gt}"
    )



if __name__ == "__main__":
    main()