import os, json, tempfile, math, shutil
import multiprocessing as mp
from pathlib import Path
import fitz  # PyMuPDF

# ---------- CONFIG ----------
DPI = 180
BATCH_SIZE = 8           # per worker loop batch (EasyOCR has no true batch; keep >1 for IO overlap)
LANGS = ['en']           # adjust
USE_GPU = True
# ----------------------------

def pre_render_pages(pdf_path, page_indices, out_dir, dpi=DPI):
    """Render selected pages to PNG once; return [(page_idx, png_path), ...]."""
    out = []
    doc = fitz.open(pdf_path)
    try:
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        for i in page_indices:
            page = doc.load_page(i)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            p = Path(out_dir) / f"page_{i:06d}.png"
            pix.save(p.as_posix())
            out.append((i, p.as_posix()))
    finally:
        doc.close()
    return out

def split_even(items, n):
    """Split list into n near-equal chunks."""
    k = math.ceil(len(items) / n) if n > 0 else 0
    return [items[i:i+k] for i in range(0, len(items), k)]

def worker_easyocr(gpu_id, items, out_json_path):
    """
    items: list[(page_idx, image_path)]
    Writes {page_idx: {"text": "...", "boxes": [[x1,y1,x2,y2,...], ...]}} to out_json_path
    """
    # Set GPU visibility BEFORE importing heavy libs in this process
    if USE_GPU:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    # Import here so it respects CUDA_VISIBLE_DEVICES for this subprocess
    import easyocr

    reader = easyocr.Reader(LANGS, gpu=USE_GPU)  # load once per process

    results = {}
    # Process in small batches to amortize overhead
    for start in range(0, len(items), BATCH_SIZE):
        batch = items[start:start+BATCH_SIZE]
        for page_idx, img_path in batch:
            det = reader.readtext(img_path, detail=1, paragraph=True)
            # det: list of (bbox, text, conf)
            lines = []
            boxes = []
            for bbox, txt, conf in det:
                if txt is None or txt.strip() == "":
                    continue
                lines.append(txt.strip())
                # flatten bbox
                flat = [float(v) for pt in bbox for v in pt]
                boxes.append(flat)
            results[page_idx] = {
                "text": "\n".join(lines),
                "boxes": boxes
            }

    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False)

def run_multi_gpu_ocr(pdf_path, page_indices=None, num_gpus=1, tmp_root=None):
    """
    Returns dict: {page_idx: {...}} merged from all workers.
    """
    if page_indices is None:
        # 0-based indices
        with fitz.open(pdf_path) as d:
            page_indices = list(range(len(d)))

    # Temp workspace
    workdir = Path(tmp_root or tempfile.mkdtemp(prefix="ocr_run_"))
    img_dir = workdir / "imgs"
    img_dir.mkdir(parents=True, exist_ok=True)

    # 1) Pre-render once
    items = pre_render_pages(pdf_path, page_indices, img_dir)

    # 2) Split across GPUs
    chunks = split_even(items, num_gpus) if num_gpus > 1 else [items]

    ctx = mp.get_context("spawn")
    procs = []
    json_paths = []

    for gpu_id, chunk in enumerate(chunks):
        if not chunk:
            continue
        out_json = workdir / f"out_gpu{gpu_id}.json"
        json_paths.append(out_json)
        p = ctx.Process(target=worker_easyocr, args=(gpu_id, chunk, out_json.as_posix()))
        procs.append(p)

    # 3) Start & join
    for p in procs:
        p.start()
    for p in procs:
        p.join()

    # 4) Merge outputs
    merged = {}
    for jp in json_paths:
        with open(jp, "r", encoding="utf-8") as f:
            part = json.load(f)
            merged.update({int(k): v for k, v in part.items()})

    # Optional: cleanup workspace
    shutil.rmtree(workdir, ignore_errors=True)

    return dict(sorted(merged.items(), key=lambda kv: kv[0]))

# ----------------- Example usage -----------------
if __name__ == "__main__":
    PDF = "/path/to/your.pdf"
    NUM_GPUS = 4              # set to your available GPUs
    # Example: run all pages
    out = run_multi_gpu_ocr(PDF, page_indices=None, num_gpus=NUM_GPUS)
    # `out` is {page_index: {"text": "...", "boxes": [...]}}
    # Persist if needed:
    with open("ocr_result.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)