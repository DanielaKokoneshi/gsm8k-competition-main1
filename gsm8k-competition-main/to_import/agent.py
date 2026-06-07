"""
GSM8K Competition Agent — single greedy pass, fits 60s batch timeout on T4.

Strategy: one batched greedy generation (proven fast on Colab T4) plus
GSM8K-native few-shot prompting and robust answer extraction.
"""

import ast
import operator
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_NEW_TOKENS = 224

SYSTEM_PROMPT = (
    "You are an expert at grade-school math word problems. "
    "Solve step by step using <<expression=result>> tags for each calculation. "
    "End with exactly one line: #### <final numeric answer>"
)

FEW_SHOT = [
    (
        "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
        "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\n"
        "Natalia sold 48+24 = <<48+24=72>>72 clips altogether.\n#### 72",
    ),
    (
        "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?",
        "Weng earns 12/60 = <<12/60=0.2>>0.2 dollars per minute.\n"
        "She earned 0.2*50 = <<0.2*50=10>>10 dollars.\n#### 10",
    ),
    (
        "Anaya wants to buy a telescope that costs 540 dollars. Her grandfather offers to pay two-fifths of the cost. She has 11 dollars saved toward the rest. How much more money does Anaya still need?",
        "Grandfather pays 540*2/5 = <<540*2/5=216>>216 dollars.\n"
        "Remaining: 540-216 = <<540-216=324>>324 dollars.\n"
        "Still needed: 324-11 = <<324-11=313>>313 dollars.\n#### 313",
    ),
]

_NUMBER_RE = re.compile(r"-?\d+(?:[.,]\d+)*")
_GSM8K_RESULT_RE = re.compile(r"<<[^=]*=\s*(-?\d+(?:\.\d+)?)\s*>>")
_EXPR_LINE_RE = re.compile(
    r"(?:^|[\s=])(-?\d+(?:\.\d+)?)\s*([+\-*/])\s*(-?\d+(?:\.\d+)?)"
    r"(?:\s*([+\-*/])\s*(-?\d+(?:\.\d+)?))?\s*=\s*(-?\d+(?:\.\d+)?)",
    re.MULTILINE,
)

_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}


def _safe_arith_eval(expr: str):
    expr = expr.strip().replace(",", "")
    if not expr or not re.fullmatch(r"[\d\s+\-*/().]+", expr):
        return None
    try:
        node = ast.parse(expr, mode="eval")
    except SyntaxError:
        return None

    def _eval(n):
        if isinstance(n, ast.Expression):
            return _eval(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return float(n.value)
        if isinstance(n, ast.UnaryOp) and type(n.op) in _BINOPS:
            return _BINOPS[type(n.op)](_eval(n.operand))
        if isinstance(n, ast.BinOp) and type(n.op) in _BINOPS:
            return _BINOPS[type(n.op)](_eval(n.left), _eval(n.right))
        raise ValueError("unsupported expression")

    try:
        return float(_eval(node))
    except (ValueError, TypeError, ZeroDivisionError, OverflowError):
        return None


def _parse_final_number(text: str) -> float:
    if not text or not str(text).strip():
        return float("nan")

    text = str(text)

    hash_answer = None
    if "####" in text:
        tail = text.rsplit("####", 1)[-1].strip()
        matches = _NUMBER_RE.findall(tail)
        if matches:
            try:
                hash_answer = float(matches[0].replace(",", ""))
            except ValueError:
                pass

    gsm_results = _GSM8K_RESULT_RE.findall(text)
    tag_answer = None
    if gsm_results:
        try:
            tag_answer = float(gsm_results[-1])
        except ValueError:
            pass

    if hash_answer is not None:
        if tag_answer is not None and abs(hash_answer - tag_answer) > 1e-3:
            intermediates = [float(t) for t in gsm_results[:-1]]
            if any(abs(hash_answer - x) < 1e-3 for x in intermediates):
                return tag_answer
        return hash_answer

    if tag_answer is not None:
        return tag_answer

    for m in _EXPR_LINE_RE.finditer(text):
        try:
            return float(m.group(6))
        except (ValueError, IndexError):
            continue

    candidates = []
    for m in re.finditer(r"([0-9][0-9\s+\-*/().]*[0-9)])\s*=", text):
        val = _safe_arith_eval(m.group(1))
        if val is not None:
            candidates.append(val)

    for m in re.finditer(
        r"(?:is|are|equals?|total|left|remaining|need|answer)\s+(-?\d+(?:\.\d+)?)",
        text,
        re.I,
    ):
        try:
            candidates.append(float(m.group(1).replace(",", "")))
        except ValueError:
            pass

    if candidates:
        return candidates[-1]

    all_nums = _NUMBER_RE.findall(text)
    if all_nums:
        try:
            return float(all_nums[-1].replace(",", ""))
        except ValueError:
            pass

    return float("nan")


def _build_messages(question: str) -> list[dict]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for q, a in FEW_SHOT:
        messages.append({"role": "user", "content": q})
        messages.append({"role": "assistant", "content": a})
    messages.append({"role": "user", "content": question})
    return messages


class Agent:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME, clean_up_tokenization_spaces=False
        )
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        load_kwargs = {
            "torch_dtype": torch.bfloat16,
            "device_map": "auto",
        }
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME, attn_implementation="sdpa", **load_kwargs
            )
        except (TypeError, ValueError):
            self.model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME, **load_kwargs
            )
        self.model.eval()

        if torch.cuda.is_available():
            dummy = self.tokenizer("warm-up", return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                self.model.generate(
                    **dummy,
                    max_new_tokens=1,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            torch.cuda.synchronize()

    def answer(self, questions: list[str]) -> tuple[list[float], list[str]]:
        prompts = [
            self.tokenizer.apply_chat_template(
                _build_messages(q),
                tokenize=False,
                add_generation_prompt=True,
            )
            for q in questions
        ]

        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096,
        ).to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.05,
            )

        prompt_len = inputs["input_ids"].shape[1]
        solutions = []
        traces = []
        for i in range(output_ids.shape[0]):
            output = self.tokenizer.decode(
                output_ids[i, prompt_len:],
                skip_special_tokens=True,
            )
            traces.append(output)
            solutions.append(_parse_final_number(output))

        return solutions, traces
