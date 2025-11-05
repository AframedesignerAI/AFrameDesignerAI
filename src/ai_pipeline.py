# src/ai_pipeline.py
from __future__ import annotations
import os
import torch
from typing import Optional
from PIL import Image
from diffusers import AutoPipelineForText2Image

class AIDesigner:
    """
    Minimal text→image pipeline backed by an open model.
    Default: SDXL-Turbo (fast & light for demos).
    """

    def __init__(
        self,
        model_id: Optional[str] = None,
        dtype: str = "auto",
        device: Optional[str] = None,
    ) -> None:
        self.model_id = model_id or os.getenv("AFRAME_MODEL_ID", "stabilityai/sdxl-turbo")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # dtype auto: float16 on CUDA, float32 on CPU
        if dtype == "auto":
            torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
        else:
            torch_dtype = getattr(torch, dtype)

        # Load pipeline
        self.pipe = AutoPipelineForText2Image.from_pretrained(
            self.model_id,
            torch_dtype=torch_dtype,
            safety_checker=None,  # démo simple; à remplacer si besoin d’un safety checker
        )

        # Move to device
        self.pipe = self.pipe.to(self.device)

        # SDXL-Turbo is optimized for very few steps; enable compile if available
        try:
            if hasattr(torch, "compile") and self.device == "cuda":
                self.pipe = torch.compile(self.pipe)
        except Exception:
            pass

    @torch.inference_mode()
    def generate_image(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        steps: int = 2,
        guidance: float = 0.0,
        seed: Optional[int] = None,
        width: int = 768,
        height: int = 512,
    ) -> Image.Image:
        if not prompt or not prompt.strip():
            prompt = "a cozy modern A-frame cabin in a lush garden, soft daylight, design magazine render, high detail"

        generator = torch.Generator(device=self.device)
        if seed is not None:
            generator = generator.manual_seed(int(seed))

        image = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=max(1, int(steps)),
            guidance_scale=float(guidance),
            width=int(width),
            height=int(height),
            generator=generator,
        ).images[0]

        return image
