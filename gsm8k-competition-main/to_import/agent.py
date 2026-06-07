"""
GSM8K Competition Agent — optimized for ML Arena (target ≥7/10 per batch).

STRATEGY (research-backed for small instruct models):
- GSM8K-native few-shot with <<expr=result>> tags (matches gold format)
- 3-shot examples covering easy arithmetic, rates, and hard multi-step patterns
- Self-consistency: 1 greedy + 2 temperature-sampled passes, majority vote
- Multi-layer answer extraction with arithmetic cross-check
- Batched generation across all 10 questions per pass (fits 60s timeout)

INTERFACE: Returns tuple[list[float], list[str]]
"""

import ast
import operator
import re
from collections import Counter

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ============================================================================
# Constants
# ============================================================================

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_NEW_TOKENS = 320
NUM_SAMPLES = 3          # 1 greedy + 2 sampled for self-consistency
SAMPLE_TEMPERATURE = 0.6

SYSTEM_PROMPT = """You are an expert at grade-school math word problems.
Solve step by step. After each calculation write the expression and result using <<expression=result>> tags.
End with exactly one line: #### <final numeric answer>
Do not add any text after the #### line."""

# Few-shot examples in native GSM8K format; third example mirrors common hard patterns.
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
    """Extract numeric answer; prefer #### then <<>> tags then expression lines."""
    if not text or not str(text).strip():
        return float("nan")

    text = str(text)

    # Layer 1: GSM8K #### marker (primary)
    hash_answer = None
    if "####" in text:
        tail = text.rsplit("####", 1)[-1].strip()
        matches = _NUMBER_RE.findall(tail)
        if matches:
            try:
                hash_answer = float(matches[0].replace(",", ""))
            except ValueError:
                pass

    # Layer 2: <<expr=result>> tags — cross-check or fallback
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

    # Layer 3: expression lines like "48+24 = 72"
    expr_answer = None
    for m in _EXPR_LINE_RE.finditer(text):
        try:
            expr_answer = float(m.group(6))
        except (ValueError, IndexError):
            continue
    if expr_answer is not None:
        return expr_answer

    # Layer 4: evaluate LHS of "expr = result" and standalone expressions
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
    """Self-consistency: pick the most frequent valid numeric answer."""
    valid = [a for a in answers if a == a]  # exclude NaN
    if not valid:
        return float("nan")
    rounded = [round(a, 4) for a in valid]
    winner, count = Counter(rounded).most_common(1)[0]
    if count == 1 and len(valid) > 1:
        # No consensus — prefer answer from greedy pass (index 0) if valid
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
    """
    Batch GSM8K solver: 3-shot GSM8K prompting + self-consistency voting.

    Each batch runs NUM_SAMPLES generation passes (1 greedy, rest sampled),
    extracts answers, and majority-votes per question. Target: ≥7/10.
    """

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
        }
        if do_sample:
            gen_kwargs.update(
                do_sample=True,
                temperature=temperature,
                top_p=0.95,
            )
        else:
            gen_kwargs.update(do_sample=False)

        with torch.no_grad():
            return self.model.generate(**inputs, **gen_kwargs)

    def _decode_outputs(self, output_ids, prompt_len: int) -> list[str]:
        texts = []
        for i in range(output_ids.shape[0]):
            generated = output_ids[i, prompt_len:]
            texts.append(
                self.tokenizer.decode(generated, skip_special_tokens=True)
            )
        return texts

    def answer(self, questions: list[str]) -> tuple[list[float], list[str]]:
        inputs = self._tokenize_batch(questions)
        prompt_len = inputs["input_ids"].shape[1]

        # Self-consistency: greedy pass + sampled passes, batched per pass.
        all_answers: list[list[float]] = [[] for _ in questions]
        all_traces: list[list[str]] = [[] for _ in questions]

        for sample_idx in range(NUM_SAMPLES):
            do_sample = sample_idx > 0
            output_ids = self._generate_batch(
                inputs,
                do_sample=do_sample,
                temperature=SAMPLE_TEMPERATURE if do_sample else None,
            )
            texts = self._decode_outputs(output_ids, prompt_len)

            for i, text in enumerate(texts):
                all_traces[i].append(text)
                all_answers[i].append(_parse_final_number(text))

        solutions = [_majority_vote(votes) for votes in all_answers]

        traces: list[str] = []
        for i, sol in enumerate(solutions):
            picked = all_traces[i][0]
            if sol == sol:
                for text, ans in zip(all_traces[i], all_answers[i]):
                    if ans == ans and abs(ans - sol) < 1e-4:
                        picked = text
                        break
            traces.append(picked)

        return solutions, traces
