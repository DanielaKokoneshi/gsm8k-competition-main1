"""
GSM8K Competition Agent — Qwen/Qwen3-1.7B, non-thinking mode + few-shot CoT

WHY THIS MODEL:
  Qwen3-1.7B is pre-cached and is the best available model in the list for math:
  - Newer architecture than Qwen2.5-1.5B-Instruct (the previous 7/10 baseline)
  - Qwen3-1.7B-Base outperforms Qwen2.5-3B on STEM benchmarks despite fewer params
  - Supports /no_think mode: full reasoning quality WITHOUT runaway think chains
  - Research finding: on GSM8K, thinking mode is WORSE than non-thinking mode
    because the model overthinks simple arithmetic (gets lost in loops)

WHY NOT DeepSeek-R1-Distill-Qwen-1.5B:
  - Think chains avg 370+ tokens/question → 3700 tokens for 10 questions → ~53s
    (you actually observed this: 53s batch time, nearly timed out)
  - Scored 6/10 vs Qwen2.5-1.5B's 7/10 despite being "math specialized"

STRATEGY:
  1. Qwen3-1.7B in /no_think mode — fast, deterministic, smarter baseline
  2. 3-shot few-shot CoT with #### answer format (matches GSM8K gold format)
  3. Single batched greedy pass, all 10 questions, max 256 tokens
  4. NaN-only retry: 2 sampled passes (temp=0.7, top_p=0.8) batched together
  5. Python arithmetic cross-check on extracted candidates (catches wrong RHS)

BUDGET (arena: ~24GB GPU, ~300+ tok/s for 1.7B bfloat16):
  Primary:  10q × 256 tok = 2560 tokens ÷ 300 tok/s ≈ 9s
  Retry:    ≤10q × 2 × 180 tok = 3600 tokens ÷ 300 tok/s ≈ 12s worst case
  Total: ~25s — well inside 60s with large safety margin.

INTERFACE: tuple[list[float], list[str]]
"""

import ast
import operator
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

MODEL_NAME = "Qwen/Qwen3-1.7B"

# /no_think tells Qwen3 to skip the <think> chain entirely.
# Research shows this is FASTER and BETTER on GSM8K-level problems.
SYSTEM_PROMPT = (
    "/no_think\n"
    "You are a math problem solver. "
    "Solve step by step and end your answer with exactly: #### <number>"
)

# 3-shot examples — high-quality, varied, matching GSM8K gold format
FEW_SHOT = [
    (
        "Natalia sold clips to 48 of her friends in April, and then she sold "
        "half as many clips in May. How many clips did she sell altogether?",
        "April: 48 clips.\nMay: 48/2 = 24 clips.\nTotal: 48 + 24 = 72.\n#### 72",
    ),
    (
        "Weng earns $12 an hour for babysitting. Yesterday she did 50 minutes "
        "of babysitting. How much did she earn?",
        "Rate: 12/60 = $0.20 per minute.\nEarnings: 0.20 × 50 = $10.\n#### 10",
    ),
    (
        "A robe takes 2 bolts of blue fiber and half that much white fiber. "
        "How many bolts in total does it take?",
        "Blue: 2 bolts.\nWhite: 2/2 = 1 bolt.\nTotal: 2 + 1 = 3 bolts.\n#### 3",
    ),
]

MAX_NEW_TOKENS       = 256   # enough for GSM8K CoT; avoids runaway generation
RETRY_MAX_NEW_TOKENS = 180
N_RETRIES            = 2
RETRY_TEMPERATURE    = 0.7
RETRY_TOP_P          = 0.8

# ─────────────────────────────────────────────────────────────────────────────
# Answer extraction
# ─────────────────────────────────────────────────────────────────────────────

_NUMBER_RE = re.compile(r"-?\d+(?:[.,]\d+)*")
_HASH_RE   = re.compile(r"####\s*(-?[\d,]+(?:\.\d+)?)")
_BOXED_RE  = re.compile(r"\\boxed\{([^}]*)\}")
_TAG_RE    = re.compile(r"<<[^=]*=\s*(-?\d+(?:\.\d+)?)\s*>>")

_BINOPS = {
    ast.Add: operator.add,  ast.Sub: operator.sub,
    ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv, ast.Pow: operator.pow,
    ast.USub: operator.neg,
}


def _safe_arith(expr: str):
    """Evaluate a pure arithmetic expression safely (no eval())."""
    expr = expr.strip().replace(",", "").replace(" ", "")
    if not expr or not re.fullmatch(r"[\d+\-*/().]+", expr):
        return None
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        return None

    def _ev(n):
        if isinstance(n, ast.Expression):  return _ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return float(n.value)
        if isinstance(n, ast.UnaryOp) and type(n.op) in _BINOPS:
            return _BINOPS[type(n.op)](_ev(n.operand))
        if isinstance(n, ast.BinOp) and type(n.op) in _BINOPS:
            return _BINOPS[type(n.op)](_ev(n.left), _ev(n.right))
        raise ValueError

    try:
        result = _ev(tree)
        return result if result == result else None
    except Exception:
        return None


def _extract(text: str) -> float:
    """
    Extract final numeric answer. Priority:
      1. #### N          — GSM8K gold format (primary target)
      2. \\boxed{N}      — fallback if model uses LaTeX style
      3. <<expr=N>>      — GSM8K annotation tags
      4. Arithmetic cross-check: find "expr = N" lines, verify with Python
      5. Keyword scan    — contextual anchor words
      6. Last number     — weakest fallback
    """
    if not text:
        return float("nan")

    # 1. #### N  (GSM8K gold format — what we trained the model to output)
    for m in reversed(list(_HASH_RE.finditer(text))):
        try:
            return float(m.group(1).replace(",", ""))
        except ValueError:
            pass

    # 2. \boxed{N}
    for m in reversed(list(_BOXED_RE.finditer(text))):
        inner = m.group(1).strip().replace(",", "").replace("$", "").strip("\\")
        val = _safe_arith(inner)
        if val is not None:
            return val
        nums = _NUMBER_RE.findall(inner)
        if nums:
            try:
                return float(nums[-1].replace(",", ""))
            except ValueError:
                pass

    # 3. <<expr=N>> tags
    tags = _TAG_RE.findall(text)
    if tags:
        try:
            return float(tags[-1])
        except ValueError:
            pass

    # 4. Arithmetic cross-check: "48 + 24 = 72" → verify 48+24==72, trust RHS
    candidates = []
    for m in re.finditer(
        r"(-?\d[\d\s+\-*/().]*\d)\s*=\s*(-?\d[\d,]*(?:\.\d+)?)",
        text
    ):
        lhs = _safe_arith(m.group(1))
        try:
            rhs = float(m.group(2).replace(",", ""))
        except ValueError:
            continue
        if lhs is not None and abs(lhs - rhs) < 0.02:
            candidates.append(rhs)

    # 5. Keyword scan
    for m in re.finditer(
        r"(?:is|are|equals?|total|left|remaining|need|answer|result|therefore|spend|cost|earn|have|get)"
        r"[^\d\n-]{0,12}(-?\d[\d,]*(?:\.\d+)?)",
        text, re.I
    ):
        try:
            candidates.append(float(m.group(1).replace(",", "")))
        except ValueError:
            pass

    if candidates:
        return candidates[-1]

    # 6. Last number
    nums = _NUMBER_RE.findall(text)
    if nums:
        try:
            return float(nums[-1].replace(",", ""))
        except ValueError:
            pass

    return float("nan")


def _is_nan(x: float) -> bool:
    return x != x


# ─────────────────────────────────────────────────────────────────────────────
# Prompt builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_prompt(tokenizer, question: str) -> str:
    """3-shot few-shot CoT prompt using Qwen3 chat template."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for q, a in FEW_SHOT:
        messages.append({"role": "user",      "content": q})
        messages.append({"role": "assistant", "content": a})
    messages.append({"role": "user", "content": question})
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


# ─────────────────────────────────────────────────────────────────────────────
# Agent
# ─────────────────────────────────────────────────────────────────────────────

class Agent:
    """
    GSM8K solver using Qwen3-1.7B in non-thinking mode with 3-shot CoT.

    Pass 1: greedy, all 10 questions batched, max 256 tokens.
    Pass 2: sampled (temp=0.7, top_p=0.8), NaN questions only, max 180 tokens.
            Repeated up to N_RETRIES=2 times.
    """

    def __init__(self) -> None:
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

        # Warm up to avoid CUDA cold-start inside the timed window
        if torch.cuda.is_available():
            _d = self.tokenizer("warm", return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                self.model.generate(
                    **_d, max_new_tokens=1,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            torch.cuda.synchronize()

        # Pre-build prompts (tokenizer is ready after __init__)
        self._build = lambda q: _build_prompt(self.tokenizer, q)

    # ── Public interface ────────────────────────────────────────────────────

    def answer(self, questions: list[str]) -> tuple[list[float], list[str]]:
        prompts = [self._build(q) for q in questions]

        # Pass 1: greedy, full batch
        outputs   = self._generate(prompts, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        solutions = [_extract(o) for o in outputs]
        traces    = list(outputs)

        # Pass 2: sampled retries for NaN results only
        for _ in range(N_RETRIES):
            nan_idx = [i for i, s in enumerate(solutions) if _is_nan(s)]
            if not nan_idx:
                break
            retry_out = self._generate(
                [prompts[i] for i in nan_idx],
                max_new_tokens=RETRY_MAX_NEW_TOKENS,
                do_sample=True,
                temperature=RETRY_TEMPERATURE,
                top_p=RETRY_TOP_P,
            )
            for i, out in zip(nan_idx, retry_out):
                val = _extract(out)
                if not _is_nan(val):
                    solutions[i] = val
                    traces[i]    = out

        return solutions, traces

    # ── Generation helper ───────────────────────────────────────────────────

    def _generate(
        self,
        prompts: list[str],
        *,
        max_new_tokens: int,
        do_sample: bool,
        temperature: float = None,
        top_p: float = None,
    ) -> list[str]:
        if not prompts:
            return []

        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=768,  # few-shot prompt fits easily; keeps KV cache small
        ).to(self.model.device)

        gen_kwargs: dict = dict(
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
            repetition_penalty=1.05,
        )
        if do_sample:
            gen_kwargs.update(do_sample=True, temperature=temperature, top_p=top_p)
        else:
            gen_kwargs.update(do_sample=False, temperature=None, top_p=None)

        with torch.no_grad():
            out_ids = self.model.generate(**inputs, **gen_kwargs)

        prompt_len = inputs["input_ids"].shape[1]
        return [
            self.tokenizer.decode(out_ids[i, prompt_len:], skip_special_tokens=True)
            for i in range(len(prompts))
        ]
