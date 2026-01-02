import torch
from diffusers import StableVideoDiffusionPipeline
from PIL import Image
import os

MODEL_PATH = "models/stable-video-diffusion-img2vid-xt"
INPUT_DIR = "keyframes"
OUTPUT_DIR = "output_videos"

os.makedirs(OUTPUT_DIR, exist_ok=True)

pipe = StableVideoDiffusionPipeline.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    variant="fp16"
).to("cuda")

pipe.enable_model_cpu_offload()

for img_name in sorted(os.listdir(INPUT_DIR)):
    if not img_name.lower().endswith((".png", ".jpg", ".jpeg")):
        continue

    image = Image.open(os.path.join(INPUT_DIR, img_name)).convert("RGB")

    result = pipe(
        image,
        num_frames=72,          # ~3s at 24fps
        fps=24,
        motion_bucket_id=40,    # LOW motion (critical)
        noise_aug_strength=0.02 # keeps image identity stable
    )

    video = result.frames
    output_path = os.path.join(
        OUTPUT_DIR,
        img_name.replace(".png", ".mp4")
    )

    pipe.export_to_video(video, output_path)
    print(f"Saved {output_path}")
