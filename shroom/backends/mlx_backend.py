from mlx_vlm import load, generate as mlx_generate
from mlx_vlm.prompt_utils import apply_chat_template

from .base import VLMBackend


class MLXBackend(VLMBackend):
    def __init__(self, model_id="mlx-community/Qwen2.5-VL-3B-Instruct-4bit"):
        self.model, self.processor = load(model_id)
        self.config = self.model.config

    def generate(self, image, text, n=1, temperature=0.5):
        prompt = apply_chat_template(self.processor, self.config, text, num_images=1)
        outs = []
        for _ in range(n):
            out = mlx_generate(
                self.model, self.processor, prompt, image=[image],
                temperature=temperature, max_tokens=512, verbose=False)
            outs.append((out.text if hasattr(out, "text") else str(out)).strip())
        return outs
