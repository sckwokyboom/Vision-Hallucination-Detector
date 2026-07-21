import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from .base import VLMBackend


class HFBackend(VLMBackend):
    def __init__(self, model_id="Qwen/Qwen2.5-VL-7B-Instruct",
                 max_pixels=1024 * 1024, dtype=torch.bfloat16):
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id, device_map="auto", dtype=dtype,
            attn_implementation="sdpa", trust_remote_code=True,
        )
        self.model.eval()
        self.max_pixels = max_pixels

    def generate(self, image, text, n=1, temperature=0.5):
        messages = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": text}]}]
        templ = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(
            text=templ, images=image, return_tensors="pt",
            max_pixels=self.max_pixels).to(self.model.device)
        plen = inputs.input_ids.shape[-1]
        with torch.no_grad():
            gen = self.model.generate(
                **inputs, max_new_tokens=512,
                do_sample=temperature > 0, temperature=temperature,
                top_p=0.95, num_return_sequences=n)
        return [self.processor.decode(seq[plen:], skip_special_tokens=True).strip()
                for seq in gen]
