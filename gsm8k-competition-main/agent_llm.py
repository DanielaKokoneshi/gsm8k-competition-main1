"""
GSM8K Competition Agent — starter template.

Implement `answer(questions)`: it receives a list of question strings (a
batch from Env, up to 10 at a time) and must return a list of floats of
the same length — one numeric answer per question. Return `float('nan')`
(or any non-matching number) if your pipeline fails to produce one;
Env counts those as format_errors.

The default implementation loads HuggingFaceTB/SmolLM3-3B, prompts it to
solve each problem step by step ending with `#### N`, and parses N out
of the model's text. You are free to change the model, the prompt, the
decoding strategy, or replace the whole pipeline.
"""

import os
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer



_NUMBER_RE = re.compile(r"-?\d+(?:[.,]\d+)*")


def _parse_final_number(text):
    """Pull the number after the final `####` marker, or return NaN."""
    if "####" not in text:
        return float("nan")
    tail = text.rsplit("####", 1)[-1].strip()
    matches = _NUMBER_RE.findall(tail)
    if not matches:
        return float("nan")
    try:
        return float(matches[-1].replace(",", ""))
    except ValueError:
        return float("nan")


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

SYSTEM_PROMPT = """You are a careful math tutor. Solve the problem step by step, then write the final numeric answer on a new line prefixed by '####'."""


class Agent:
    def __init__(self):

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, clean_up_tokenization_spaces=False)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.model.eval()

        # Trigger CUDA kernel compilation now, before any timeout window opens.
        if torch.cuda.is_available():
            dummy = self.tokenizer("warm-up", return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                self.model.generate(**dummy, max_new_tokens=1,
                                    pad_token_id=self.tokenizer.eos_token_id)
            torch.cuda.synchronize()

    def answer(self, questions: list[str]) -> list[float]:
        prompts = []
        for q in questions:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": q},
            ]
            prompts.append(self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            ))

        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )


        replies = []
        prompt_len = inputs["input_ids"].shape[1]
        for i in range(output_ids.shape[0]):
            generated_ids = output_ids[i, prompt_len:]
            output = self.tokenizer.decode(
                generated_ids,
                skip_special_tokens=True,
            )
            replies.append(_parse_final_number(output))

        return replies
