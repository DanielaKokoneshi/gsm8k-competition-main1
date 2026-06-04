"""
GSM8K Competition Agent — optimized for ML Arena (target ≥7/10 per batch).

INTERFACE: Returns tuple[list[float], list[str]]
- solutions: numeric answers (or NaN for parse failures)
- traces: full model outputs for rendering

STRATEGY:
- 3-shot GSM8K-style prompting with clear arithmetic examples
- Greedy generation (temperature=0) for deterministic, consistent results
- Qwen2.5-1.5B-Instruct: best speed/accuracy tradeoff on platform cache
- 6-layer fallback answer extraction: ####, tags, expressions, keywords, etc.
- Batch processing with padding for throughput (fits 60s timeout for 10q)
"""

import ast
import operator
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ============================================================================
# Constants: Model & Prompting
# ============================================================================

# Qwen2.5-1.5B-Instruct is pre-cached on ML Arena platform and optimized
# for 60s batches of ~10 questions each.
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_NEW_TOKENS = 128

SYSTEM_PROMPT = """You are an expert at grade-school math word problems.
Solve using short arithmetic steps.
Use at most 5 lines.
End with exactly one line: #### <final integer answer>
Do not add text after the #### line."""

# 3-shot examples train the model to follow the format reliably.
FEW_SHOT = [
    (
"Question:",
"A store sold 30 apples on Monday and twice as many on Tuesday. If 10 apples were returned, how many apples were sold in total?",
"Answer:",
"Tuesday sales = 30 * 2 = 60",
"andTotal before returns = 30 + 60 = 90",
"After returns = 90 - 10 = 80",
"#### 80",
    )
]

# ============================================================================
# Answer Extraction: Multi-layer fallback parsing
# ============================================================================

_NUMBER_RE = re.compile(r"-?\d+(?:[.,]\d+)*")
_GSM8K_RESULT_RE = re.compile(r"<<[^=]*=\s*(-?\d+(?:\.\d+)?)\s*>>")
_EXPR_LINE_RE = re.compile(
    r"(?:^|[\s=])(-?\d+(?:\.\d+)?)\s*([+\-*/])\s*(-?\d+(?:\.\d+)?)"
    r"(?:\s*([+\-*/])\s*(-?\d+(?:\.\d+)?))?\s*=\s*(-?\d+(?:\.\d+)?)",
    re.MULTILINE,
)

# Safe arithmetic evaluator: only supports +, -, *, /, //, ** (no eval())
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
    """Safely evaluate a pure arithmetic expression via AST."""
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
    """
    Extract the best numeric answer from model output with 6-level fallback.
    
    Layer 1: Gold GSM8K format (#### answer)
    Layer 2: Training-style tags (<<expr=result>>)
    Layer 3: Expression lines (48+24 = 72)
    Layer 4: Standalone arithmetic evaluation
    Layer 5: Contextual keywords (is, equals, total, etc.)
    Layer 6: Last number in response (weak fallback)
    """
    if not text or not str(text).strip():
        return float("nan")

    text = str(text)

    # Layer 1: GSM8K gold format: #### answer
    if "####" in text:
        tail = text.rsplit("####", 1)[-1].strip()
        matches = _NUMBER_RE.findall(tail)
        if matches:
            try:
                return float(matches[0].replace(",", ""))
            except ValueError:
                pass

    # Layer 2: Training-style <<expr=result>> tags
    gsm_results = _GSM8K_RESULT_RE.findall(text)
    if gsm_results:
        try:
            return float(gsm_results[-1])
        except ValueError:
            pass

    # Layer 3: Lines like "48+24 = 72" — trust the RHS
    for m in _EXPR_LINE_RE.finditer(text):
        try:
            return float(m.group(6))
        except (ValueError, IndexError):
            continue

    # Layer 4: Evaluate standalone arithmetic expressions
    candidates = []
    for m in re.finditer(r"([0-9][0-9\s+\-*/().]*[0-9)])\s*=", text):
        val = _safe_arith_eval(m.group(1))
        if val is not None:
            candidates.append(val)

    # Layer 5: Look for contextual keywords before numbers
    for m in re.finditer(r"(?:is|are|equals?|total|left|remaining|need)\s+(-?\d+(?:\.\d+)?)", text, re.I):
        try:
            candidates.append(float(m.group(1).replace(",", "")))
        except ValueError:
            pass

    if candidates:
        return candidates[-1]

    # Layer 6: Last number in response (weak fallback)
    all_nums = _NUMBER_RE.findall(text)
    if all_nums:
        try:
            return float(all_nums[-1].replace(",", ""))
        except ValueError:
            pass

    return float("nan")


def _build_messages(question: str) -> list[dict]:
    """Build chat template: system prompt + 3-shot examples + question."""
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
    Batch GSM8K solver using 3-shot prompting and greedy decoding.
    
    Performance target: ≥7/10 correct on each 10-question batch.
    Timeout: 60 seconds per batch.
    """

    def __init__(self):
        """Load model, tokenizer, and warm up GPU."""
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

        # Warm up CUDA to avoid JIT compilation delays during timed batches.
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
        """
        Solve a batch of GSM8K questions.

        Args:
            questions: List of question strings (up to 10).

        Returns:
            (solutions, traces): tuple of
            - solutions: list[float] — numeric answers or NaN if parsing fails
            - traces: list[str] — full model outputs (for rendering)
        """
        # Step 1: Build prompts using chat template + few-shot examples.
        prompts = []
        for q in questions:
            messages = _build_messages(q)
            prompts.append(
                self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )

        # Step 2: Tokenize and batch.
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096,
        ).to(self.model.device)

        # Step 3: Generate with greedy decoding (temperature=0 for consistency).
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                temperature=None,
                top_p=None,
                repetition_penalty=1.05,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Step 4: Decode only the generated part (skip prompt).
        input_len = inputs["attention_mask"][i].sum()
        solutions = []
        traces = []

        for i in range(output_ids.shape[0]):
            generated_ids = output_ids[i, input_len:]
            output = self.tokenizer.decode(
                generated_ids,
                skip_special_tokens=True,
            )
            traces.append(output)
            solutions.append(_parse_final_number(output))

        return solutions, traces
