"""
GSM8K Competition Agent — target ≥7/10 on ML Arena private set.

Incremental improvements over the Colab 7/10 baseline:
- Few-shot covers 3 hard patterns: basic ops, fraction/savings, max-miss threshold
- Calculator-verified <<expr=result>> extraction (fixes wrong #### lines)
- Single greedy pass on slow GPUs (Colab T4); optional 2nd pass on fast GPUs only
"""

import ast
import operator
import re
import time
from collections import Counter

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_NEW_TOKENS = 256
SAMPLE_TEMPERATURE = 0.7

# If greedy pass finishes faster than this, run one sampled pass (ML Arena GPUs).
FAST_GPU_THRESHOLD_SEC = 22.0
BATCH_BUDGET_SEC = 55.0

SYSTEM_PROMPT = """You are an expert at grade-school math word problems.
Solve step by step. Write each calculation as <<expression=result>> (e.g. <<48/2=24>>24).
For "how much more still needed": subtract any savings from what remains owed.
For percent increases: new value = original + original * percent / 100.
End with exactly one line: #### <final numeric answer>"""

# Three shots aligned with common hard patterns in the competition set.
FEW_SHOT = [
    (
        "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
        "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\n"
        "Natalia sold 48+24 = <<48+24=72>>72 clips altogether.\n#### 72",
    ),
    (
        "Anaya wants to buy a telescope that costs 540 dollars. Her grandfather offers to pay two-fifths of the cost. She has 11 dollars saved toward the rest. How much more money does Anaya still need?",
        "Grandfather pays 540*2/5 = <<540*2/5=216>>216 dollars.\n"
        "Remaining: 540-216 = <<540-216=324>>324 dollars.\n"
        "Still needed: 324-11 = <<324-11=313>>313 dollars.\n#### 313",
    ),
    (
        "Eldar signs up for 400 art workshops for a total of 476 dollars. His scholarship is revoked if the cost per attended workshop rises above 4 dollars. What is the maximum number of workshops Eldar can miss before his scholarship is revoked?",
        "Minimum attended for $4 each: 476/4 = <<476/4=119>>119 workshops.\n"
        "Maximum misses: 400-119 = <<400-119=281>>281 workshops.\n#### 281",
    ),
]

_NUMBER_RE = re.compile(r"-?\d+(?:[.,]\d+)*")
_GSM8K_TAG_RE = re.compile(r"<<([^=]+)=([^>]*)>>")
_GSM8K_RESULT_RE = re.compile(r"<<[^=]*=\s*(-?\d+(?:\.\d+)?)\s*>>")
_EXPR_LINE_RE = re.compile(
    r"(?:^|[\s=])(-?\d+(?:\.\d+)?)\s*([+\-*/])\s*(-?\d+(?:\.\d+)?)"
    r"(?:\s*([+\-*/])\s*(-?\d+(?:\.\d+)?))?\s*=\s*(-?\d+(?:\.\d+)?)",
    re.MULTILINE,
)
_BOXED_RE = re.compile(r"\\boxed\{([^}]+)\}")
_ANSWER_IS_RE = re.compile(
    r"(?:the\s+)?(?:final\s+)?answer\s+is[:\s]*\$?(-?\d+(?:\.\d+)?)",
    re.I,
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


def _verified_tag_values(text: str) -> list[float]:
    """Re-evaluate each <<expr=result>> tag; trust Python math over the model."""
    values = []
    for m in _GSM8K_TAG_RE.finditer(text):
        expr = m.group(1).strip()
        stated = m.group(2).strip()
        computed = _safe_arith_eval(expr)
        if computed is not None:
            values.append(computed)
            continue
        try:
            values.append(float(stated.replace(",", "")))
        except ValueError:
            pass
    return values


def _parse_final_number(text: str) -> float:
    if not text or not str(text).strip():
        return float("nan")

    text = str(text)
    verified = _verified_tag_values(text)

    hash_answer = None
    if "####" in text:
        tail = text.rsplit("####", 1)[-1].strip()
        matches = _NUMBER_RE.findall(tail)
        if matches:
            try:
                hash_answer = float(matches[0].replace(",", ""))
            except ValueError:
                pass

    if hash_answer is not None and verified:
        last_verified = verified[-1]
        if abs(hash_answer - last_verified) > 1e-3:
            if any(abs(hash_answer - v) < 1e-3 for v in verified[:-1]):
                return last_verified
            if abs(hash_answer - last_verified) / max(abs(last_verified), 1.0) > 0.01:
                return last_verified
        return hash_answer

    if hash_answer is not None:
        return hash_answer

    if verified:
        return verified[-1]

    gsm_results = _GSM8K_RESULT_RE.findall(text)
    if gsm_results:
        try:
            return float(gsm_results[-1])
        except ValueError:
            pass

    for m in _BOXED_RE.finditer(text):
        nums = _NUMBER_RE.findall(m.group(1))
        if nums:
            try:
                return float(nums[-1].replace(",", ""))
            except ValueError:
                pass

    answer_matches = _ANSWER_IS_RE.findall(text)
    if answer_matches:
        try:
            return float(answer_matches[-1].replace(",", ""))
        except ValueError:
            pass

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
        r"(?:is|are|equals?|total|left|remaining|need)\s+(-?\d+(?:\.\d+)?)",
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


def _majority_vote(answers: list[float]) -> float:
    valid = [a for a in answers if a == a]
    if not valid:
        return float("nan")
    rounded = [round(a, 4) for a in valid]
    winner, count = Counter(rounded).most_common(1)[0]
    if count == 1 and len(valid) > 1:
        return valid[0]
    return winner


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

        load_kwargs = {"torch_dtype": torch.bfloat16, "device_map": "auto"}
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME, attn_implementation="sdpa", **load_kwargs
            )
        except (TypeError, ValueError):
            self.model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **load_kwargs)
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

    def _tokenize_batch(self, questions: list[str]):
        prompts = [
            self.tokenizer.apply_chat_template(
                _build_messages(q),
                tokenize=False,
                add_generation_prompt=True,
            )
            for q in questions
        ]
        return self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096,
        ).to(self.model.device)

    def _generate(self, inputs, *, do_sample: bool):
        kwargs = {
            "max_new_tokens": MAX_NEW_TOKENS,
            "pad_token_id": self.tokenizer.eos_token_id,
            "repetition_penalty": 1.03,
            "do_sample": do_sample,
        }
        if do_sample:
            kwargs.update(temperature=SAMPLE_TEMPERATURE, top_p=0.95)
        with torch.no_grad():
            return self.model.generate(**inputs, **kwargs)

    def answer(self, questions: list[str]) -> tuple[list[float], list[str]]:
        t0 = time.monotonic()
        inputs = self._tokenize_batch(questions)
        prompt_len = inputs["input_ids"].shape[1]

        output_ids = self._generate(inputs, do_sample=False)
        traces = [
            self.tokenizer.decode(output_ids[i, prompt_len:], skip_special_tokens=True)
            for i in range(output_ids.shape[0])
        ]
        answers = [_parse_final_number(t) for t in traces]

        elapsed = time.monotonic() - t0
        if (
            elapsed < FAST_GPU_THRESHOLD_SEC
            and BATCH_BUDGET_SEC - elapsed >= elapsed + 2.0
        ):
            output_ids2 = self._generate(inputs, do_sample=True)
            traces2 = [
                self.tokenizer.decode(
                    output_ids2[i, prompt_len:], skip_special_tokens=True
                )
                for i in range(output_ids2.shape[0])
            ]
            answers2 = [_parse_final_number(t) for t in traces2]

            solutions = []
            final_traces = []
            for i in range(len(questions)):
                sol = _majority_vote([answers[i], answers2[i]])
                solutions.append(sol)
                picked = traces[i]
                if sol == sol:
                    for text, ans in ((traces[i], answers[i]), (traces2[i], answers2[i])):
                        if ans == ans and abs(ans - sol) < 1e-4:
                            picked = text
                            break
                final_traces.append(picked)
            return solutions, final_traces

        return answers, traces
