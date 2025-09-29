def ask_question(question: str, context: str) -> str:
    messages = [
        {"role": "system", "content": "You are a helpful assistant that answers questions based only on the provided OCR text."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}\nAnswer concisely."},
    ]
    text_inputs = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=[text_inputs], return_tensors="pt", padding=True).to("cuda", dtype=torch.bfloat16)

    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=512, temperature=0.0)
    return processor.batch_decode(output, skip_special_tokens=True)[0]

# Example multi-turn
print(ask_question("What company name appears in the header?", context_text))
print(ask_question("List all dates mentioned.", context_text))
print(ask_question("Summarize this page in 3 sentences.", context_text))

# pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# pip install transformers bitsandbytes accelerate pillow numpy opencv-python

import os, json, math, re
from typing import List, Tuple
import numpy as np
from PIL import Image
import torch
import cv2

from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    BitsAndBytesConfig,
)

MODEL_ID = "meta-llama/Llama-4-Scout-17B-16E"  # adjust if your local repo path differs
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---- Load model in 4-bit (GPU) ----
bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_cfg,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)

model.eval()
torch.set_grad_enabled(False)

# ---- Tiling helpers ----
def tile_image(img: Image.Image, tile_px: int = 1400, overlap: int = 128) -> Tuple[List[Image.Image], List[Tuple[int,int,int,int]], Tuple[int,int]]:
    """
    Split image into overlapping tiles. Returns (tiles, bboxes_pixel, orig_size).
    bbox format: (x1, y1, x2, y2) in original pixel coords.
    """
    w, h = img.size
    tiles, boxes = [], []
    step = tile_px - overlap
    for y in range(0, h, step):
        for x in range(0, w, step):
            x2, y2 = min(x + tile_px, w), min(y + tile_px, h)
            crop = img.crop((x, y, x2, y2))
            tiles.append(crop)
            boxes.append((x, y, x2, y2))
        # avoid extra narrow last column duplications
    return tiles, boxes, (w, h)

def norm_bbox(px_box: Tuple[int,int,int,int], orig_wh: Tuple[int,int]) -> List[float]:
    x1, y1, x2, y2 = px_box
    W, H = orig_wh
    return [round(x1/W, 6), round(y1/H, 6), round(x2/W, 6), round(y2/H, 6)]

# ---- Prompt ----
SYSTEM_PROMPT = (
    "You are an OCR+layout extractor. Return ONLY valid JSON (no code fences). "
    "Schema: {\"full_text\": str, \"blocks\": [{\"type\": \"heading|paragraph|table|figure|caption\", "
    "\"text\": str, \"bbox\": [x1,y1,x2,y2]}], \"tables\": [{\"bbox\": [x1,y1,x2,y2], "
    "\"cells\": [{\"row\": int, \"col\": int, \"text\": str, \"bbox\": [x1,y1,x2,y2]}]}]} . "
    "Coordinates must be normalized to the ORIGINAL full page width/height. "
    "Reading order top-to-bottom, left-to-right. Do not include anything except the JSON."
)

USER_INSTR = (
    "Extract text blocks with reading order. For each block, provide: text, normalized bbox, and type. "
    "Also detect tables with cell-level text and bboxes if visible. Return only the JSON."
)

def _run_scout_on_image(pil_img: Image.Image, max_new_tokens: int = 2048) -> str:
    """
    Calls the model once on one image with the fixed prompts and returns raw text output.
    Works with HF 'image + text' processors for Llama-Scout family.
    """
    # Many Scout processors accept messages via apply_chat_template; fallback to images+text if needed.
    try:
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": SYSTEM_PROMPT}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_img},
                    {"type": "text",  "text": USER_INSTR},
                ],
            },
        ]
        text_inputs = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=[text_inputs], images=[pil_img], return_tensors="pt", padding=True).to(DEVICE, dtype=torch.bfloat16)
    except Exception:
        # Fallback: older processors
        inputs = processor(images=pil_img, text=f"{SYSTEM_PROMPT}\n\n{USER_INSTR}", return_tensors="pt").to(DEVICE, dtype=torch.bfloat16)

    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            do_sample=False,
        )

    out = processor.batch_decode(generated, skip_special_tokens=True)[0]
    return out.strip()

_json_start = re.compile(r"\{", re.DOTALL)
_json_end = re.compile(r"\}\s*$", re.DOTALL)

def force_json(text: str) -> dict:
    """
    Best-effort: extract the outermost JSON object from the model’s text.
    """
    # Trim any leading junk before first '{'
    m = _json_start.search(text)
    if not m:
        raise ValueError("No JSON object start found.")
    json_str = text[m.start():]
    # Try to balance braces
    depth = 0
    end_idx = None
    for i, ch in enumerate(json_str):
        if ch == "{": depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end_idx = i + 1
                break
    if end_idx is None:
        raise ValueError("Unbalanced JSON braces.")
    json_str = json_str[:end_idx]
    return json.loads(json_str)

def merge_blocks(all_blocks: List[dict]) -> List[dict]:
    """
    Very light post-process: sort by y, then x; join identical-adjacent paragraphs.
    """
    def sort_key(b): 
        x1,y1,x2,y2 = b["bbox"]
        return (round(y1,4), round(x1,4))
    all_blocks = sorted(all_blocks, key=sort_key)

    merged = []
    for b in all_blocks:
        if merged and b["type"] == "paragraph" and merged[-1]["type"] == "paragraph":
            # Join if vertically adjacent and overlapping in x-range (naive heuristic)
            x1,y1,x2,y2 = b["bbox"]; X1,Y1,X2,Y2 = merged[-1]["bbox"]
            overlap = max(0.0, min(x2,X2) - max(x1,X1)) / max(1e-6, (x2 - x1))
            if overlap > 0.5 and abs(y1 - Y2) < 0.02:  # 2% of page height
                merged[-1]["text"] = (merged[-1]["text"] + " " + b["text"]).strip()
                merged[-1]["bbox"][2] = max(X2, x2)
                merged[-1]["bbox"][3] = max(Y2, y2)
                continue
        merged.append(b)
    return merged

def extract_image_to_json(image_path: str, tile_px: int = 1400, overlap: int = 128, max_new_tokens_tile: int = 1200) -> dict:
    """
    Full pipeline:
      1) load image
      2) tile
      3) run Scout per tile (JSON each)
      4) remap tile bboxes to full-page normalized coords
      5) light merge
    """
    img = Image.open(image_path).convert("RGB")
    tiles, boxes, orig_wh = tile_image(img, tile_px=tile_px, overlap=overlap)

    all_blocks = []
    all_tables = []
    full_text_parts = []

    for tile, px_box in zip(tiles, boxes):
        raw = _run_scout_on_image(tile, max_new_tokens=max_new_tokens_tile)
        try:
            j = force_json(raw)
        except Exception:
            # Try one repair prompt pass (cheap)
            repair_inp = f"Reformat this as valid JSON only, same schema, no extra text:\n{raw}"
            jraw = _run_scout_on_image(tile, max_new_tokens=max_new_tokens_tile)
            j = force_json(jraw)

        # Remap tile-normalized bboxes to full-page normalized coords
        tx1, ty1, tx2, ty2 = px_box
        tw, th = (tx2 - tx1), (ty2 - ty1)

        for b in j.get("blocks", []):
            x1,y1,x2,y2 = b["bbox"]
            # convert tile-normalized -> tile px
            X1 = tx1 + x1 * tw; Y1 = ty1 + y1 * th
            X2 = tx1 + x2 * tw; Y2 = ty1 + y2 * th
            b["bbox"] = norm_bbox((int(X1), int(Y1), int(X2), int(Y2)), orig_wh)
            all_blocks.append(b)

        for t in j.get("tables", []):
            bx1,by1,bx2,by2 = t["bbox"]
            BX1 = tx1 + bx1 * tw; BY1 = ty1 + by1 * th
            BX2 = tx1 + bx2 * tw; BY2 = ty1 + by2 * th
            t["bbox"] = norm_bbox((int(BX1), int(BY1), int(BX2), int(BY2)), orig_wh)

            for c in t.get("cells", []):
                cx1,cy1,cx2,cy2 = c["bbox"]
                CX1 = tx1 + cx1 * tw; CY1 = ty1 + cy1 * th
                CX2 = tx1 + cx2 * tw; CY2 = ty1 + cy2 * th
                c["bbox"] = norm_bbox((int(CX1), int(CY1), int(CX2), int(CY2)), orig_wh)
            all_tables.append(t)

        ft = j.get("full_text")
        if isinstance(ft, str) and ft.strip():
            full_text_parts.append(ft.strip())

    # Light merge/sort
    all_blocks = merge_blocks(all_blocks)
    full_text = "\n".join(full_text_parts).strip()

    return {
        "full_text": full_text,
        "blocks": all_blocks,
        "tables": all_tables,
    }

if __name__ == "__main__":
    import argparse, pathlib, sys
    p = argparse.ArgumentParser()
    p.add_argument("image_path", help="Path to a page image (e.g., PNG/JPG).")
    p.add_argument("--tile_px", type=int, default=1400)
    p.add_argument("--overlap", type=int, default=128)
    p.add_argument("--out", type=str, default="")
    args = p.parse_args()

    out = extract_image_to_json(args.image_path, tile_px=args.tile_px, overlap=args.overlap)
    js = json.dumps(out, ensure_ascii=False)
    if args.out:
        pathlib.Path(args.out).write_text(js, encoding="utf-8")
        print(f"Wrote: {args.out}")
    else:
        print(js)