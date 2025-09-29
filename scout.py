from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-3.2-11B-Vision-Instruct", trust_remote_code=True)
prompts = [{
  "prompt": [
    {"type": "text", "text": "Summarize the image."},
    {"type": "image", "image_url": "file:///abs/path/to/image.jpg"}
  ]
}]
out = llm.generate(prompts, SamplingParams(max_tokens=128, temperature=0))
print(out[0].outputs[0].text)