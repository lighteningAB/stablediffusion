import os, gc
import torch
from diffusers import StableVideoDiffusionPipeline
from PIL import Image
from diffusers.utils import export_to_video

MODEL_PATH = "models/stable-video-diffusion-img2vid-xt"
INPUT_DIR = "keyframes"
OUTPUT_DIR = "output_videos"

os.makedirs(OUTPUT_DIR, exist_ok=True)

pipe = StableVideoDiffusionPipeline.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    variant="fp16"
).to("cuda")

pipe.enable_attention_slicing("max")

for img_name in sorted(os.listdir(INPUT_DIR)):
    if not img_name.lower().endswith((".png", ".jpg", ".jpeg")):
        continue

    image = Image.open(os.path.join(INPUT_DIR, img_name)).convert("RGB")
    image = image.resize((1024, 576), Image.LANCZOS)

    with torch.inference_mode():
        result = pipe(
            image,
            num_frames=50,
            fps=10,
            motion_bucket_id=40,
            noise_aug_strength=0.02,
            decode_chunk_size=4,
        )

    output_path = os.path.join(OUTPUT_DIR, os.path.splitext(img_name)[0] + ".mp4")
    
    # Just pass frames directly - export_to_video handles the conversion
    export_to_video(result.frames[0], output_path, fps=10)
    print(f"Saved {output_path}")

    del result
    gc.collect()
    torch.cuda.empty_cache()