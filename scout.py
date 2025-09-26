import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

MODEL_ID = "meta-llama/Llama-4-Scout-17B-16E-Instruct"  # adjust to your exact repo

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# Optional: cap per-GPU to help HF balance layers
max_memory = {0: "78GiB", 1: "78GiB"}

tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    quantization_config=bnb,
    device_map="auto",          # or use `max_memory=max_memory, device_map="auto"`
    low_cpu_mem_usage=True,
)

prompt = "Summarize why MoE models inflate VRAM even in 4-bit."
inputs = tok(prompt, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=256)
print(tok.decode(out[0], skip_special_tokens=True))