from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from transformers import pipeline

class LLM:
    def __init__(self):
        self.model_name = "Qwen/Qwen3-0.6B"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_length=512,
            do_sample=False
        )
