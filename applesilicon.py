import torch
from diffusers import ZImagePipeline

assert torch.backends.mps.is_available(), "MPS not available"

# 1) bf16 tends to be more stable than fp16 on MPS for some models
dtype = torch.bfloat16

pipe = ZImagePipeline.from_pretrained(
    "Tongyi-MAI/Z-Image-Turbo",
    dtype=dtype,
    low_cpu_mem_usage=False,
).to("mps")

# 2) keep VAE in fp32
pipe.vae.to(dtype=torch.float32)

# 3) warmup pass (recommended for MPS)
_ = pipe(
    prompt="warmup",
    height=512,
    width=512,
    num_inference_steps=1,
    guidance_scale=0.0,
).images[0]

prompt = """
make a high resolution realistic image of a beautiful german shepherd dog
"""

# 4) avoid MPS generator weirdness
torch.manual_seed(42)

image = pipe(
    prompt=prompt,
    height=768,
    width=768,
    num_inference_steps=9,
    guidance_scale=0.0,
).images[0]

image.save("out.png")
print("Saved image")
