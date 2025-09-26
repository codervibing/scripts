# build inputs
inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
)

# pick the model's (first) device, works with device_map="auto"
device = next(model.parameters()).device

# move all tensors in the dict
inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}

# generate
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=5000)