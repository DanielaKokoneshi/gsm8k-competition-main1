"""
GSM8K Competition Agent — optimized for ML Arena (target ≥7/10 per batch).

Fits the 60s batch timeout (same on Colab T4 and ML Arena):
- One greedy batched pass always (~15–25s on T4)
- Optional 2nd sampled pass only if time budget allows (self-consistency)
- GSM8K-native few-shot with <<expr=result>> tags
- Multi-layer answer extraction with arithmetic cross-check

INTERFACE: Returns tuple[list[float], list[str]]
"""

import ast
import operator
import re
import time
from collections import Counter

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ============================================================================
# Constants
# ============================================================================

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_NEW_TOKENS = 256
SAMPLE_TEMPERATURE = 0.6

# Env enforces 60s per batch; keep a safety margin for tokenize/decode overhead.
BATCH_BUDGET_SEC = 52.0
MIN_TIME_FOR_EXTRA_PASS_SEC = 18.0

SYSTEM_PROMPT = """You are an expert at grade-school math word problems.
Solve step by step. After each calculation write the expression and result using <<expression=result>> tags.
End with exactly one line: #### <final numeric answer>
Do not add any text after the #### line."""

FEW_SHOT = [
    (
        "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
        "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\n"
        "Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\n#### 72",
    ),
    (
        "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?",
        "Weng earns 12/60 = <<12/60=0.2>>0.2 dollars per minute.\n"
        "For 50 minutes she earned 0.2*50 = <<0.2*50=10>>10 dollars.\n#### 10",
    ),
    (
        "Anaya wants to buy a telescope that costs 540 dollars. Her grandfather offers to pay two-fifths of the cost. She has 11 dollars saved toward the rest. How much more money does Anaya still need?",
        "The grandfather pays two-fifths: 540*2/5 = <<540*2/5=216>>216 dollars.\n"
        "Remaining cost: 540-216 = <<540-216=324>>324 dollars.\n"
        "After savings: 324-11 = <<324-11=313>>313 dollars still needed.\n#### 313",
    ),
]

# ============================================================================
# Answer extraction
# ============================================================================

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


# ============================================================================
# Agent
# ============================================================================

class Agent:
    """Batch GSM8K solver: greedy pass + optional 2nd pass if time allows."""

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME, clean_up_tokenization_spaces=False
        )
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.bfloat16,
            device_map="auto",
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

    def _generate_batch(self, inputs, *, do_sample: bool, temperature=None):
        gen_kwargs = {
            "max_new_tokens": MAX_NEW_TOKENS,
            "pad_token_id": self.tokenizer.eos_token_id,
            "repetition_penalty": 1.02,
            "use_cache": True,
        }
        if do_sample:
            gen_kwargs.update(do_sample=True, temperature=temperature, top_p=0.95)
        else:
            gen_kwargs.update(do_sample=False)

        with torch.no_grad():
            return self.model.generate(**inputs, **gen_kwargs)

    def _decode_outputs(self, output_ids, prompt_len: int) -> list[str]:
        return [
            self.tokenizer.decode(
                output_ids[i, prompt_len:],
                skip_special_tokens=True,
            )
            for i in range(output_ids.shape[0])
        ]

    def _run_pass(self, inputs, prompt_len: int, *, do_sample: bool, temperature=None):
        output_ids = self._generate_batch(
            inputs, do_sample=do_sample, temperature=temperature
        )
        texts = self._decode_outputs(output_ids, prompt_len)
        answers = [_parse_final_number(t) for t in texts]
        return texts, answers

    def answer(self, questions: list[str]) -> tuple[list[float], list[str]]:
        t0 = time.monotonic()
        inputs = self._tokenize_batch(questions)
        prompt_len = inputs["input_ids"].shape[1]

        traces, answers = self._run_pass(inputs, prompt_len, do_sample=False)

        elapsed = time.monotonic() - t0
        remaining = BATCH_BUDGET_SEC - elapsed
        if remaining >= MIN_TIME_FOR_EXTRA_PASS_SEC:
            traces2, answers2 = self._run_pass(
                inputs,
                prompt_len,
                do_sample=True,
                temperature=SAMPLE_TEMPERATURE,
            )
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
