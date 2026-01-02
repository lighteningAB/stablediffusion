import torch
from diffusers.pipelines.z_image.pipeline_z_image import ZImagePipeline

assert torch.cuda.is_available(), "CUDA not available"

device = "cuda"

# fp16 is typically best on NVIDIA (unless you hit stability issues)
torch_dtype = torch.float16

pipe = ZImagePipeline.from_pretrained(
    "Tongyi-MAI/Z-Image-Turbo",
    torch_dtype=torch_dtype,
    low_cpu_mem_usage=True,
)

pipe = pipe.to("cuda")
pipe.enable_attention_slicing()

pipe.vae.to(device="cuda", dtype=torch.float32)

g = torch.Generator(device=device).manual_seed(42)

prompt = "make a high resolution realistic image of a beautiful german shepherd dog"

# Use a CUDA generator (don’t rely on global seed)
g = torch.Generator(device=device).manual_seed(42)

image = pipe(
    prompt=prompt,
    height=128,
    width=128,
    num_inference_steps=20,
    guidance_scale=1.0,
    generator=g,
).images[0]

image.save("out.png")
print("Saved image")
