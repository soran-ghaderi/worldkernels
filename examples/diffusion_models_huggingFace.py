import torch
import imageio
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler

def main() -> None:
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"


    pipe = DiffusionPipeline.from_pretrained(
        "damo-vilab/text-to-video-ms-1.7b",
        torch_dtype=torch.float16 if device=="cuda" else torch.float32
    )

    pipe = pipe.to(device)

    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config
    )

    prompts = [
        "A futuristic city with flying cars",
        "A dragon flying over snowy mountains",
        "A robot walking in a neon cyberpunk street"
    ]

    for i, prompt in enumerate(prompts):
        result = pipe(
            prompt,
            num_inference_steps=40,
            num_frames=24
        )

        frames = result.frames

        output_path = f"video_{i}.mp4"

        imageio.mimsave(
            output_path,
            frames,
            fps=8
        )

if __name__ == '__main__':
    main()