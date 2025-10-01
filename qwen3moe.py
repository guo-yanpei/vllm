from vllm import LLM, SamplingParams
import random
import numpy as np
import torch

def set_seed(seed):
    """Set seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)
MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"
print("A) building LLM")
llm = LLM(
    model=MODEL,
    trust_remote_code=True,
    tensor_parallel_size=1,          # single GPU first
    # Remove kv_cache_dtype until confirmed stable
    kv_cache_dtype="fp8_e4m3",
    gpu_memory_utilization=0.85,     # give headroom
    max_model_len=8192,              # start modest, scale later
    download_dir="/home/guest/data/guo_yanpei",
    enforce_eager=True               # for debugging
)

print("B) sampling params")
sp = SamplingParams(temperature=0.2, top_p=0.9, max_tokens=1024)

print("C) generate")
prompt = ["Where is capital of France?"]
out, additional_out = llm.generate(prompt, sp)
print("output:\n", out[0].outputs[0].text)
# print("additional output:\n", additional_out)