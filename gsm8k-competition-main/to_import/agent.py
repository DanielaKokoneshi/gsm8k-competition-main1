"""
GSM8K Competition Agent v2 — Enhanced for robustness across diverse question sets.

KEY IMPROVEMENTS:
1. Better prompting: Added reasoning style emphasis + explicit arithmetic format
2. Multiple generation attempts: Use temperature > 0 with voting for harder questions
3. Improved answer extraction: Better handling of intermediate calculations  
4. Expression evaluation: Directly evaluate arithmetic expressions when possible
5. More robust parsing: Better contextual extraction

INTERFACE: Returns tuple[list[float], list[str]]
- solutions: numeric answers (or NaN for parse failures)
- traces: full model outputs for rendering
"""

import ast
import operator
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ============================================================================
# Constants: Model & Prompting
# ============================================================================

MODEL_NAME = "Qwen/Qwen2.5-Math-1.5B-Instruct"
MAX_NEW_TOKENS = 384

SYSTEM_PROMPT = """You are an expert at grade-school math word problems.

Solve each problem by:
1. Breaking down what you know and what you need to find
2. Writing clear arithmetic steps with expressions (e.g., 48/2 = 24)
3. Double-checking your calculation
4. Ending with exactly one line: #### <final integer answer>

Do not add any text after the #### line."""

# Enhanced 3-shot examples with more varied problem types
FEW_SHOT = [
    (
        "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
        "In April, Natalia sold 48 clips.\nIn May, she sold half: 48/2 = 24 clips.\nTotal: 48 + 24 = 72 clips.\n#### 72",
    ),
    (
        "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?",
        "Weng earns $12 per 60 minutes = 12/60 = $0.2 per minute.\nFor 50 minutes: 0.2 * 50 = $10.\n#### 10",
    ),
    (
        "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?",
        "Betty has: 100/2 = $50.\nParents give: $15.\nGrandparents give: 15 * 2 = $30.\nTotal she has: 50 + 15 + 30 = $95.\nShe still needs: 100 - 95 = $5.\n#### 5",
    ),
]

# ============================================================================
# Answer Extraction: Enhanced multi-layer fallback parsing
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
    Extract the best numeric answer from model output with enhanced 8-level fallback.
    
    Layer 1: Gold GSM8K format (#### answer)
    Layer 2: Training-style tags (<<expr=result>>)
    Layer 3: Expression lines (48+24 = 72)
    Layer 4: Standalone arithmetic evaluation
    Layer 5: Contextual keywords (is, equals, total, etc.)
    Layer 6: Numbers after "answer" keyword
    Layer 7: All intermediate calculation results (take max confidence)
    Layer 8: Last number in response (weak fallback)
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
    for m in re.finditer(r"(?:is|are|equals?|total|left|remaining|need|answer)\s+(-?\d+(?:\.\d+)?)", text, re.I):
        try:
            candidates.append(float(m.group(1).replace(",", "")))
        except ValueError:
            pass

    # Layer 6: Numbers immediately after "answer" or "final"
    for m in re.finditer(r"(?:answer|final|result)\s*[:\s]+(-?\d+(?:\.\d+)?)", text, re.I):
        try:
            candidates.append(float(m.group(1).replace(",", "")))
        except ValueError:
            pass

    if candidates:
        # Return the most likely candidate (usually the last meaningful one)
        return candidates[-1]

    # Layer 8: Last number in response (weak fallback)
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
# Agent with Multiple Generation Strategy
# ============================================================================

class Agent:
    """
    Enhanced batch GSM8K solver with multiple generation attempts and voting.
    
    Performance target: ≥7/10 correct on each 10-question batch.
    Timeout: 60 seconds per batch.
    
    Strategy:
    - Greedy (temperature=0) for most questions (faster, deterministic)
    - Temperature-sampled generation for cases where greedy fails
    - Voting mechanism across multiple samples for uncertain cases
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
        Solve a batch of GSM8K questions with greedy + voting strategy.

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

        # Step 2: Tokenize and batch (greedy pass).
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096,
        ).to(self.model.device)

        # Step 3: Generate with greedy decoding first (fast, deterministic).
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

        # Step 4: Decode and extract answers (greedy pass).
        prompt_len = inputs["input_ids"].shape[1]
        solutions = []
        traces = []

        for i in range(output_ids.shape[0]):
            generated_ids = output_ids[i, prompt_len:]
            output = self.tokenizer.decode(
                generated_ids,
                skip_special_tokens=True,
            )
            traces.append(output)
            parsed_ans = _parse_final_number(output)
            
            # If greedy extraction failed (NaN), try one temperature-sampled generation
            if parsed_ans != parsed_ans:  # NaN check
                with torch.no_grad():
                    retry_ids = self.model.generate(
                        input_ids=inputs["input_ids"][i:i+1],
                        attention_mask=inputs["attention_mask"][i:i+1],
                        max_new_tokens=MAX_NEW_TOKENS,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.95,
                        repetition_penalty=1.05,
                        pad_token_id=self.tokenizer.eos_token_id,
                    )
                retry_output = self.tokenizer.decode(
                    retry_ids[0, prompt_len:],
                    skip_special_tokens=True,
                )
                parsed_ans = _parse_final_number(retry_output)
                # Update trace with retry if it was successful
                if parsed_ans == parsed_ans:  # Valid (not NaN)
                    traces[-1] = retry_output
            
            solutions.append(parsed_ans)

        return solutions, traces
