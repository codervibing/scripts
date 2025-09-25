
# pip install -U "transformers>=4.41" bitsandbytes accelerate pillow torch --extra-index-url https://download.pytorch.org/whl/cu121
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig

MODEL_ID = "meta-llama/Llama-4-Scout-17B-16E"
IMAGE_PATH = "sample_page.jpg"  # local image

# Make sure only the two GPUs you want are visible
# export CUDA_VISIBLE_DEVICES=0,1

bnb = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
    llm_int8_has_fp16_weight=True,
)

processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID,
    quantization_config=bnb,               # 8-bit weights
    device_map="auto",                     # shard across both GPUs
    max_memory={"cuda:0":"38GiB","cuda:1":"38GiB","cpu":"96GiB"},
    offload_folder="offload_scout",        # disk spillover, prevents load-time OOM
    low_cpu_mem_usage=True,
    trust_remote_code=True
).eval()

# ---- Minimal OCR inference ----
img = Image.open(IMAGE_PATH).convert("RGB")
img.thumbnail((1280, 1280), Image.LANCZOS)   # cap resolution to keep seq length sane

messages = [{
    "role": "user",
    "content": [
        {"type":"image","image": img},
        {"type":"text","text":"Transcribe all visible text. Output plain text only."}
    ]
}]

inputs = processor.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=True, return_tensors="pt", return_dict=True
)
inputs = {k: (v.to(model.device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}

with torch.inference_mode():
    out = model.generate(**inputs, max_new_tokens=600, temperature=0.0, do_sample=False, use_cache=True)

text = processor.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
print(text)