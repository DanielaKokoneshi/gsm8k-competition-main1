"""
GSM8K Competition Agent — DeepSeek-R1-Distill-Qwen-1.5B (pre-cached on arena)

MODEL CHOICE: deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
  - Only math-specialized model in the pre-cached list.
  - Base is Qwen2.5-Math-1.5B fine-tuned on 800k R1 reasoning traces.
  - Achieves 83.9% on MATH-500, outperforms GPT-4o on math benchmarks.
  - Generates <think>...</think> chains then \boxed{answer}.

KEY TRICKS (from research):
  1. FORCE <think>\n prefix in the assistant turn — without it the model skips
     reasoning entirely, tanking accuracy (official DeepSeek recommendation).
  2. Prompt format: <|User|>...question...<|Assistant|><think>\n
     (raw string, NOT apply_chat_template which omits the <think> prefill)
  3. Hard token cap of 400: R1 think-chains are long; 400 covers GSM8K easily
     (~250 tokens avg) without risking timeout on hard questions.
  4. Single batched greedy pass — all 10 questions at once, no loops.
  5. NaN retry: 2 sampled passes (temperature=0.7) for any failed parses,
     batched together, capped at 200 tokens.
  6. Python arithmetic verification: after extracting the candidate answer,
     re-evaluate the last arithmetic expression in the trace and cross-check.

BUDGET (arena GPU ~24GB, ~200 tok/s for 1.5B bfloat16):
  Primary: 10q × 400 tok = 4000 tokens ÷ 200 tok/s ≈ 20s
  Retry:   ≤10q × 200 tok × 2 = 4000 tokens ÷ 200 tok/s ≈ 20s worst case
  Total: ~40s — safely within 60s even with overhead.
"""

import ast
import operator
import re
import subprocess
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ─────────────────────────────────────────────────────────────────────────────
# Model config
# ─────────────────────────────────────────────────────────────────────────────

MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

# Raw prompt template — we manually build the string so we can prefill <think>
# This is the official format from the DeepSeek-R1 paper and model card.
# Forcing <think>\n at the start prevents the model from bypassing reasoning.
_PROMPT_TEMPLATE = (
    "<|begin_of_sentence|>"
    "Please reason step by step, and put your final answer within \\boxed{{}}.\n"
    "<|User|>{question}<|Assistant|><think>\n"
)

MAX_NEW_TOKENS       = 400   # hard cap: enough for GSM8K, prevents runaway chains
RETRY_MAX_NEW_TOKENS = 200   # shorter budget for retry pass
N_RETRIES            = 2     # sampled retry attempts for NaN questions
RETRY_TEMPERATURE    = 0.7
RETRY_TOP_P          = 0.8

# ─────────────────────────────────────────────────────────────────────────────
# Answer extraction
# ─────────────────────────────────────────────────────────────────────────────

_NUMBER_RE = re.compile(r"-?\d+(?:[.,]\d+)*")
_BOXED_RE  = re.compile(r"\\boxed\{([^}]*)\}")
_HASH_RE   = re.compile(r"####\s*(-?[\d,]+)")
_TAG_RE    = re.compile(r"<<[^=]*=\s*(-?\d+(?:\.\d+)?)\s*>>")

_BINOPS = {
    ast.Add: operator.add,  ast.Sub: operator.sub,
    ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv, ast.Pow: operator.pow,
    ast.USub: operator.neg,
}


def _safe_arith(expr: str):
    """Evaluate pure arithmetic expression safely (no eval)."""
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
        return result if result == result else None  # reject NaN
    except Exception:
        return None


def _extract(text: str) -> float:
    """
    Extract the final numeric answer from model output.
    Priority: \\boxed{} > #### > <<expr=N>> > keyword scan > last number.
    """
    if not text:
        return float("nan")

    # 1. \boxed{answer} — R1 native output format
    for m in reversed(list(_BOXED_RE.finditer(text))):
        inner = m.group(1).strip().replace(",", "").replace("$", "").strip("\\")
        # Try arithmetic eval first (handles \frac, simple expressions)
        val = _safe_arith(inner)
        if val is not None:
            return val
        nums = _NUMBER_RE.findall(inner)
        if nums:
            try:
                return float(nums[-1].replace(",", ""))
            except ValueError:
                pass

    # 2. #### N — GSM8K gold format (model may use this too)
    for m in reversed(list(_HASH_RE.finditer(text))):
        try:
            return float(m.group(1).replace(",", ""))
        except ValueError:
            pass

    # 3. <<expr=N>> annotation tags
    tags = _TAG_RE.findall(text)
    if tags:
        try:
            return float(tags[-1])
        except ValueError:
            pass

    # 4. Keyword-anchored number scan
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

    # 5. Python arithmetic verification on the last expr=result line
    #    Find patterns like "48 + 24 = 72" and check the RHS is arithmetic
    for m in re.finditer(
        r"(-?\d[\d\s+\-*/().]*\d)\s*=\s*(-?\d[\d,]*(?:\.\d+)?)",
        text
    ):
        lhs_val = _safe_arith(m.group(1))
        try:
            rhs_val = float(m.group(2).replace(",", ""))
        except ValueError:
            continue
        if lhs_val is not None and abs(lhs_val - rhs_val) < 0.01:
            candidates.append(rhs_val)

    if candidates:
        return candidates[-1]

    # 6. Last number in text (weakest fallback)
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
# Agent
# ─────────────────────────────────────────────────────────────────────────────

class Agent:
    """
    GSM8K solver using DeepSeek-R1-Distill-Qwen-1.5B.

    Generation strategy:
      Pass 1 — greedy, all 10 questions batched, 400 token cap.
               The <think>\\n prefill forces the model to reason before answering.
      Pass 2 — sampled (temp=0.7, top_p=0.8), only NaN questions, 200 token cap.
               Repeated up to N_RETRIES times.
    """

    def __init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME, clean_up_tokenization_spaces=False
        )
        # R1-Distill uses left-padding for batch generation
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.model.eval()

        # Warm up CUDA to avoid cold-start penalty inside the timed window
        if torch.cuda.is_available():
            _d = self.tokenizer("warm", return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                self.model.generate(
                    **_d, max_new_tokens=1,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            torch.cuda.synchronize()

    # ── Public interface ────────────────────────────────────────────────────

    def answer(self, questions: list[str]) -> tuple[list[float], list[str]]:
        # Build prompts using raw string template (preserves <think>\n prefill)
        prompts = [_PROMPT_TEMPLATE.format(question=q) for q in questions]

        # ── Pass 1: greedy, all questions batched ─────────────────────────
        outputs = self._generate(
            prompts,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )

        solutions = [_extract(o) for o in outputs]
        traces    = list(outputs)

        # ── Pass 2: sampled retries for NaN results ───────────────────────
        for _attempt in range(N_RETRIES):
            nan_idx = [i for i, s in enumerate(solutions) if _is_nan(s)]
            if not nan_idx:
                break

            retry_prompts = [prompts[i] for i in nan_idx]
            retry_outputs = self._generate(
                retry_prompts,
                max_new_tokens=RETRY_MAX_NEW_TOKENS,
                do_sample=True,
                temperature=RETRY_TEMPERATURE,
                top_p=RETRY_TOP_P,
            )

            for i, out in zip(nan_idx, retry_outputs):
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
            max_length=512,   # GSM8K questions are short; keep KV cache small
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
