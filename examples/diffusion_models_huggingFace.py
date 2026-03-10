import torch
import imageio
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler

# -----------------------------
# Check if GPU is available
# -----------------------------
if torch.cuda.is_available():
    device = "cuda"
    print("✅ GPU detected. Using CUDA for faster generation.")
else:
    device = "cpu"
    print("⚠️ GPU not detected. Running on CPU will be much slower.")

# -----------------------------
# Load the text-to-video model
# -----------------------------
pipe = DiffusionPipeline.from_pretrained(
    "damo-vilab/text-to-video-ms-1.7b",
    torch_dtype=torch.float16 if device=="cuda" else torch.float32
)

pipe = pipe.to(device)

# -----------------------------
# Replace scheduler
# -----------------------------
pipe.scheduler = DPMSolverMultistepScheduler.from_config(
    pipe.scheduler.config
)

# -----------------------------
# Prompts list
# -----------------------------
prompts = [
    "A futuristic city with flying cars",
    "A dragon flying over snowy mountains",
    "A robot walking in a neon cyberpunk street"
]

# -----------------------------
# Loop through prompts
# -----------------------------
for i, prompt in enumerate(prompts):

    print(f"Generating video {i+1}: {prompt}")

    result = pipe(
        prompt,
        num_inference_steps=40,
        num_frames=24
    )

    frames = result.frames

    # Save video
    output_path = f"video_{i}.mp4"

    imageio.mimsave(
        output_path,
        frames,
        fps=8
    )

    print(f"Saved: {output_path}")