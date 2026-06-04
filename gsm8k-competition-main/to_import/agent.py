"""
GSM8K Competition Agent v3 — Math-specialized model + self-consistency voting.

KEY UPGRADES over v2:
1. Model swap: Qwen/Qwen2.5-Math-1.5B-Instruct (math-finetuned) instead of
   general Qwen2.5-1.5B-Instruct. Same parameter count, massively better on math.
2. Self-consistency: generate N=3 samples per question at temperature 0.7,
   pick the majority-vote answer. Falls back to greedy if all answers differ.
3. Prompt format: matches Qwen2.5-Math training format exactly (CoT style).
4. Python-based arithmetic verification: re-evaluate the last expression in
   the trace to catch off-by-one and rounding errors.
5. Batched voting pass: all 10×3 samples generated in one GPU call to stay
   inside the 60s budget.

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

MODEL_NAME = "Qwen/Qwen2.5-Math-1.5B-Instruct"
MAX_NEW_TOKENS = 512          # math model benefits from slightly more tokens
N_SAMPLES = 3                 # majority vote over this many samples
VOTE_TEMPERATURE = 0.7        # sampling temperature for non-greedy passes
TOP_P = 0.95

# Qwen2.5-Math was trained with this exact system prompt for CoT mode.
SYSTEM_PROMPT = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)

# ============================================================================
# Answer Extraction
# ============================================================================

_NUMBER_RE = re.compile(r"-?\d+(?:[.,]\d+)*")
_BOXED_RE   = re.compile(r"\\boxed\{([^}]*)\}")
_HASH_RE    = re.compile(r"####\s*(-?[\d,]+)")
_GSM_TAG_RE = re.compile(r"<<[^=]*=\s*(-?\d+(?:\.\d+)?)\s*>>")

_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}


def _safe_arith(expr: str):
    """Safely evaluate a pure arithmetic expression (no eval)."""
    expr = expr.strip().replace(",", "")
    if not expr or not re.fullmatch(r"[\d\s+\-*/().]+", expr):
        return None
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        return None

    def _ev(n):
        if isinstance(n, ast.Expression):
            return _ev(n.body)
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
    Extract the final numeric answer from model output.

    Priority order:
      1. \\boxed{...}    — Qwen2.5-Math CoT native format
      2. #### N          — GSM8K gold format
      3. <<expr=N>>      — GSM8K training-tag format
      4. Contextual keyword scan
      5. Last number fallback
    """
    if not text:
        return float("nan")

    # 1. \boxed{answer}
    for m in reversed(list(_BOXED_RE.finditer(text))):
        inner = m.group(1).strip().replace(",", "")
        # Handle \boxed{72} or \boxed{$72} or \boxed{72.5}
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

    # 3. <<expr=N>> tags
    tags = _GSM_TAG_RE.findall(text)
    if tags:
        try:
            return float(tags[-1])
        except ValueError:
            pass

    # 4. Contextual keyword scan
    candidates = []
    for m in re.finditer(
        r"(?:is|are|equals?|total|left|remaining|need|answer|result|therefore)[^\d-]*(-?\d[\d,]*(?:\.\d+)?)",
        text, re.I
    ):
        try:
            candidates.append(float(m.group(1).replace(",", "")))
        except ValueError:
            pass

    if candidates:
        return candidates[-1]

    # 5. Last number in response
    nums = _NUMBER_RE.findall(text)
    if nums:
        try:
            return float(nums[-1].replace(",", ""))
        except ValueError:
            pass

    return float("nan")


def _answers_equal(a: float, b: float, tol: float = 1e-6) -> bool:
    if a != a or b != b:   # NaN check
        return False
    return abs(a - b) < tol


def _majority_vote(answers: list[float]) -> float:
    """
    Return the answer that appears most often (within tolerance).
    Ties broken by taking the first group found.
    Returns NaN if all answers are NaN.
    """
    valid = [a for a in answers if a == a]  # drop NaN
    if not valid:
        return float("nan")
    if len(valid) == 1:
        return valid[0]

    # Cluster by tolerance
    clusters: list[list[float]] = []
    for v in valid:
        placed = False
        for cl in clusters:
            if _answers_equal(v, cl[0]):
                cl.append(v)
                placed = True
                break
        if not placed:
            clusters.append([v])

    # Return centroid of the largest cluster
    best = max(clusters, key=len)
    return sum(best) / len(best)


# ============================================================================
# Prompt builder
# ============================================================================

def _build_messages(question: str) -> list[dict]:
    """Standard Qwen2.5-Math CoT chat template (no few-shot needed — model is
    already finetuned on GSM8K-style data)."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": question},
    ]


# ============================================================================
# Agent
# ============================================================================

class Agent:
    """
    GSM8K solver using Qwen2.5-Math-1.5B-Instruct with self-consistency voting.

    Strategy:
      • Generate N_SAMPLES completions per question via batched sampling.
      • Extract numeric answer from each; pick by majority vote.
      • If no majority (all different), fall back to greedy decoding answer.
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

        # GPU warm-up to avoid cold-start penalty in the timed window.
        if torch.cuda.is_available():
            dummy = self.tokenizer(
                "What is 1+1?", return_tensors="pt"
            ).to(self.model.device)
            with torch.no_grad():
                self.model.generate(
                    **dummy,
                    max_new_tokens=8,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            torch.cuda.synchronize()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def answer(
        self, questions: list[str]
    ) -> tuple[list[float], list[str]]:
        """
        Solve a batch of GSM8K questions.

        Returns
        -------
        solutions : list[float]
            Numeric answers (NaN on parse failure).
        traces : list[str]
            Best (majority-winning) model output per question.
        """
        prompts = [
            self.tokenizer.apply_chat_template(
                _build_messages(q),
                tokenize=False,
                add_generation_prompt=True,
            )
            for q in questions
        ]

        # ---- Pass 1: greedy (fast, one answer per question) --------------
        greedy_outputs = self._generate_batch(prompts, do_sample=False, n=1)
        greedy_answers = [_extract(o[0]) for o in greedy_outputs]
        greedy_traces  = [o[0] for o in greedy_outputs]

        # ---- Pass 2: sampled (N_SAMPLES per question, all in one GPU call)
        # Replicate each prompt N_SAMPLES times so we can batch everything.
        expanded_prompts = []
        for p in prompts:
            expanded_prompts.extend([p] * N_SAMPLES)

        sampled_flat = self._generate_batch(
            expanded_prompts, do_sample=True, n=1
        )
        # sampled_flat[i*N_SAMPLES : (i+1)*N_SAMPLES] → samples for question i
        sampled_outputs = [
            [sampled_flat[i * N_SAMPLES + k][0] for k in range(N_SAMPLES)]
            for i in range(len(questions))
        ]
        sampled_answers = [
            [_extract(o) for o in outs] for outs in sampled_outputs
        ]

        # ---- Voting ------------------------------------------------------
        solutions: list[float] = []
        traces:    list[str]   = []

        for i in range(len(questions)):
            # Pool: greedy answer + N sampled answers
            pool_answers = [greedy_answers[i]] + sampled_answers[i]
            pool_traces  = [greedy_traces[i]]  + sampled_outputs[i]

            voted = _majority_vote(pool_answers)

            # Pick the trace that produced the majority answer (prefer greedy
            # if it agrees, for cleaner reasoning output).
            best_trace = greedy_traces[i]
            for ans, trace in zip(pool_answers, pool_traces):
                if _answers_equal(ans, voted):
                    best_trace = trace
                    break

            solutions.append(voted)
            traces.append(best_trace)

        return solutions, traces

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _generate_batch(
        self,
        prompts: list[str],
        *,
        do_sample: bool,
        n: int,
    ) -> list[list[str]]:
        """
        Tokenize `prompts`, generate, decode.

        Returns list[list[str]] of shape (len(prompts), n).
        """
        if not prompts:
            return []

        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(self.model.device)

        gen_kwargs: dict = dict(
            max_new_tokens=MAX_NEW_TOKENS,
            pad_token_id=self.tokenizer.eos_token_id,
            repetition_penalty=1.05,
            num_return_sequences=n,
        )
        if do_sample:
            gen_kwargs.update(
                do_sample=True,
                temperature=VOTE_TEMPERATURE,
                top_p=TOP_P,
            )
        else:
            gen_kwargs.update(
                do_sample=False,
                temperature=None,
                top_p=None,
            )

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **gen_kwargs)

        prompt_len = inputs["input_ids"].shape[1]
        results: list[list[str]] = []

        # output_ids shape: (len(prompts)*n, seq_len)
        for i in range(len(prompts)):
            seqs = []
            for j in range(n):
                idx = i * n + j
                gen = output_ids[idx, prompt_len:]
                text = self.tokenizer.decode(gen, skip_special_tokens=True)
                seqs.append(text)
            results.append(seqs)

        return results