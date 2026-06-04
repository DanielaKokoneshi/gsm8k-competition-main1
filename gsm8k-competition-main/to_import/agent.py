"""
GSM8K Competition Agent v4 — Speed-safe greedy with math-specialized model.

ROOT CAUSE OF TIMEOUT: Batched sampling (10 questions × 3 samples × 512 tokens)
produced ~15k tokens total — ~150s at 100 tok/s on a typical eval GPU.

FIX STRATEGY:
  1. Greedy only (do_sample=False) — fastest possible, fully deterministic.
  2. Max 256 new tokens — GSM8K solutions rarely exceed ~200 tokens.
  3. Qwen/Qwen2.5-Math-1.5B-Instruct — math-finetuned, same size as before,
     dramatically better accuracy than the general Qwen2.5-1.5B-Instruct.
  4. Prompt format matches the model's training: system = "Please reason step
     by step, and put your final answer within \\boxed{}."
  5. One cheap temperature=0.5 retry (128 tokens budget) only for NaN parses.

BUDGET ESTIMATE (conservative):
  10 questions × ~150 tokens avg × 1 greedy pass ≈ 1500 tokens ≈ 15s
  + up to 3 NaN retries × 128 tokens ≈ ~4s worst case
  Total: ~20s — well inside 60s.

INTERFACE: Returns tuple[list[float], list[str]]
"""

import ast
import operator
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ============================================================================
# Constants
# ============================================================================

MODEL_NAME = "Qwen/Qwen2.5-Math-1.5B-Instruct"
MAX_NEW_TOKENS = 256       # enough for GSM8K; saves ~2× time vs 512
RETRY_MAX_TOKENS = 128     # for NaN-retry pass only

# Exact system prompt Qwen2.5-Math was trained with (CoT mode).
SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."

# ============================================================================
# Answer extraction
# ============================================================================

_NUMBER_RE  = re.compile(r"-?\d+(?:[.,]\d+)*")
_BOXED_RE   = re.compile(r"\\boxed\{([^}]*)\}")
_HASH_RE    = re.compile(r"####\s*(-?[\d,]+)")
_TAG_RE     = re.compile(r"<<[^=]*=\s*(-?\d+(?:\.\d+)?)\s*>>")

_BINOPS = {
    ast.Add: operator.add,  ast.Sub: operator.sub,
    ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv, ast.Pow: operator.pow,
    ast.USub: operator.neg,
}


def _safe_arith(expr: str):
    expr = expr.strip().replace(",", "")
    if not expr or not re.fullmatch(r"[\d\s+\-*/().]+", expr):
        return None
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        return None

    def _ev(n):
        if isinstance(n, ast.Expression):    return _ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return float(n.value)
        if isinstance(n, ast.UnaryOp) and type(n.op) in _BINOPS:
            return _BINOPS[type(n.op)](_ev(n.operand))
        if isinstance(n, ast.BinOp) and type(n.op) in _BINOPS:
            return _BINOPS[type(n.op)](_ev(n.left), _ev(n.right))
        raise ValueError

    try:
        return float(_ev(tree))
    except Exception:
        return None


def _extract(text: str) -> float:
    """
    Extract final numeric answer.  Priority:
      1. \\boxed{N}   — Qwen2.5-Math native output
      2. #### N       — GSM8K gold format
      3. <<expr=N>>   — GSM8K annotation tags
      4. Contextual keyword scan
      5. Last number fallback
    """
    if not text:
        return float("nan")

    # 1. \boxed{...} — take the last occurrence
    for m in reversed(list(_BOXED_RE.finditer(text))):
        inner = m.group(1).strip().replace(",", "").replace("$", "")
        nums = _NUMBER_RE.findall(inner)
        if nums:
            try:
                return float(nums[-1].replace(",", ""))
            except ValueError:
                pass

    # 2. #### N
    for m in reversed(list(_HASH_RE.finditer(text))):
        try:
            return float(m.group(1).replace(",", ""))
        except ValueError:
            pass

    # 3. <<expr=N>>
    tags = _TAG_RE.findall(text)
    if tags:
        try:
            return float(tags[-1])
        except ValueError:
            pass

    # 4. Contextual keywords
    candidates = []
    for m in re.finditer(
        r"(?:is|are|equals?|total|left|remaining|need|answer|result|therefore)"
        r"[^\d-]{0,10}(-?\d[\d,]*(?:\.\d+)?)",
        text, re.I
    ):
        try:
            candidates.append(float(m.group(1).replace(",", "")))
        except ValueError:
            pass
    if candidates:
        return candidates[-1]

    # 5. Last number
    nums = _NUMBER_RE.findall(text)
    if nums:
        try:
            return float(nums[-1].replace(",", ""))
        except ValueError:
            pass

    return float("nan")


# ============================================================================
# Prompt builder
# ============================================================================

def _build_prompt(tokenizer, question: str) -> str:
    messages = [
        {"role": "system",  "content": SYSTEM_PROMPT},
        {"role": "user",    "content": question},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


# ============================================================================
# Agent
# ============================================================================

class Agent:
    """
    Greedy-only GSM8K solver using the math-finetuned Qwen2.5-Math-1.5B-Instruct.

    All 10 questions are batched into a single forward pass (greedy).
    Questions that fail to parse (NaN) get one cheap individual retry.
    Total budget: ~15-25s for a typical batch of 10.
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

        # Warm up to avoid JIT/CUDA init penalty inside timed window.
        if torch.cuda.is_available():
            _dummy = self.tokenizer("warm", return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                self.model.generate(
                    **_dummy, max_new_tokens=1,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            torch.cuda.synchronize()

    # ------------------------------------------------------------------

    def answer(self, questions: list[str]) -> tuple[list[float], list[str]]:
        prompts = [_build_prompt(self.tokenizer, q) for q in questions]

        # ---- Single batched greedy pass ----------------------------------
        outputs = self._run(prompts, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        solutions = [_extract(o) for o in outputs]
        traces    = list(outputs)

        # ---- Individual retry for NaN parses only ------------------------
        for i, (sol, prompt) in enumerate(zip(solutions, prompts)):
            if sol != sol:   # NaN
                retry = self._run(
                    [prompt],
                    max_new_tokens=RETRY_MAX_TOKENS,
                    do_sample=True,
                    temperature=0.5,
                    top_p=0.9,
                )[0]
                val = _extract(retry)
                if val == val:   # not NaN → use it
                    solutions[i] = val
                    traces[i]    = retry

        return solutions, traces

    # ------------------------------------------------------------------

    def _run(
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
            max_length=1024,   # prompt + answer fits in 1024 for GSM8K
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
            output_ids = self.model.generate(**inputs, **gen_kwargs)

        prompt_len = inputs["input_ids"].shape[1]
        return [
            self.tokenizer.decode(output_ids[i, prompt_len:], skip_special_tokens=True)
            for i in range(len(prompts))
        ]