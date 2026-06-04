"""GSM8K Competition Environment (standalone).

At test time we draw K disjoint subsets of 10 questions each. Every subset
contains 5 easy questions (drawn from items with difficulty "easy" or
"medium") and 5 hard questions, shuffled. Each subset is sent to the agent
as a single batch via
`agent.answer(questions: list[str]) -> (list[float], list[str])`:
the agent returns one numeric solution and one thinking-trace string per
question. A legacy `list[float]` return is still accepted; empty traces
are synthesized in that case. A subset has
BATCH_TIMEOUT seconds to complete; if it fails or times out, no further
subsets are sent.

Questions and gold answers come from `question_set.json` next to this file.

For each agent we return:
  - score:               total correct across all delivered subsets
  - format_errors:       agent replies that were not coercible to float
  - questions_received:  number of questions actually sent
"""

import concurrent.futures
import json
import math
import os
import random
import re
import textwrap
import time
import traceback

import numpy as np
from PIL import Image, ImageDraw, ImageFont


DATA_PATH = os.path.join(os.path.dirname(__file__), "question_set.json")
DATA_PATH_PRIVATE = os.path.join(os.path.dirname(__file__), "question_set_private.json")

K = 10                       # number of subsets per agent
EASY_PER_SUBSET = 5         # easy/medium picks per subset
HARD_PER_SUBSET = 5         # hard picks per subset
BATCH_TIMEOUT = 60.0        # seconds per subset
SEED = 554284                    # fixed sample is the same for every agent

_NUMBER_RE = re.compile(r"-?\d+(?:[.,]\d+)*")


class _AgentProxy:
    """Mimics the platform's AgentProxy: adds `.call(method, ..., timeout=, default=, catch_errors=)`.

    Timeouts run the call in a worker thread; on timeout the parent abandons it
    (the thread keeps running until the process exits), since `torch.generate`
    can't be interrupted from another thread.
    """

    def __init__(self, agent):
        self._agent = agent
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.last_error = None

    def call(self, method, *args, timeout=None, default=None,
             catch_errors=False, **kwargs):
        self.last_error = None
        fn = getattr(self._agent, method)
        try:
            if timeout is None:
                return fn(*args, **kwargs)
            future = self._executor.submit(fn, *args, **kwargs)
            try:
                return future.result(timeout=timeout)
            except concurrent.futures.TimeoutError as e:
                raise TimeoutError(
                    f"agent.{method} exceeded timeout={timeout}s"
                ) from e
        except Exception as e:
            self.last_error = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
            if default is not None or catch_errors:
                return default
            raise


def wrap(agent):
    """Wrap a local agent so `agent.call(...)` mirrors the platform proxy."""
    return _AgentProxy(agent)


def _gold_from_answer(answer_field):
    """Parse the gold number from a GSM8K answer string ending in `#### N`."""
    if answer_field is None or "####" not in str(answer_field):
        return None
    tail = str(answer_field).rsplit("####", 1)[-1].strip()
    matches = _NUMBER_RE.findall(tail)
    if not matches:
        return None
    try:
        return float(matches[-1].replace(",", ""))
    except ValueError:
        return None


def _to_float(x):
    """Coerce an agent reply to float, or return None if it cannot be parsed."""
    if x is None:
        return None
    if isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        return None if (isinstance(x, float) and math.isnan(x)) else float(x)
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _get_font(size: int):
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ):
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _render_frame(question: str, thinking_trace: str, answer) -> np.ndarray:
    """Black-on-white frame showing one (question, thinking_trace, answer) triple."""
    W, H = 900, 600
    img = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(img)
    header = _get_font(20)
    body = _get_font(16)

    y = 16
    wrap_w = 90

    def block(label, text):
        nonlocal y
        d.text((20, y), label, fill=(0, 0, 0), font=header)
        y += 28
        for line in textwrap.wrap(str(text), width=wrap_w) or [""]:
            d.text((20, y), line, fill=(0, 0, 0), font=body)
            y += 20
            if y > H - 40:
                return
        y += 12

    block("Question:", question)
    block("Thinking trace:", thinking_trace)
    block("Answer:", answer)

    return np.array(img, dtype=np.uint8)


class Env:
    def __init__(self, is_evaluation=False):
        data_path = DATA_PATH_PRIVATE if is_evaluation else DATA_PATH
        # Fall back to the public set if the private one isn't present.
        if is_evaluation and not os.path.exists(DATA_PATH_PRIVATE):
            data_path = DATA_PATH

        try:
            with open(data_path) as f:
                items = json.load(f)
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Question set not found at {data_path!r}. "
                f"Place question_set.json next to env.py."
            ) from e
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Question set at {data_path!r} is not valid JSON: {e}"
            ) from e

        if not isinstance(items, list):
            raise TypeError(
                f"Question set at {data_path!r} must be a JSON list; "
                f"got {type(items).__name__}."
            )

        easy = [it for it in items if it.get("difficulty", "easy") != "hard"]
        hard = [it for it in items if it.get("difficulty") == "hard"]
        # At evaluation we draw K subsets; locally we only run a single batch.
        n_subsets = K if is_evaluation else 1
        need_easy = n_subsets * EASY_PER_SUBSET
        need_hard = n_subsets * HARD_PER_SUBSET
        if len(easy) < need_easy:
            raise ValueError(
                f"Question set has {len(easy)} easy/medium items; "
                f"need at least n_subsets * EASY_PER_SUBSET = {need_easy}."
            )
        if len(hard) < need_hard:
            raise ValueError(
                f"Question set has {len(hard)} hard items; "
                f"need at least n_subsets * HARD_PER_SUBSET = {need_hard}."
            )

        rng = random.Random(SEED)
        rng.shuffle(easy)
        rng.shuffle(hard)

        self.subsets = []
        for k in range(n_subsets):
            chunk = (
                easy[k * EASY_PER_SUBSET:(k + 1) * EASY_PER_SUBSET]
                + hard[k * HARD_PER_SUBSET:(k + 1) * HARD_PER_SUBSET]
            )
            rng.shuffle(chunk)
            self.subsets.append(chunk)

        # Flat view for callers that want all questions in delivery order.
        self.questions = [it["question"] for s in self.subsets for it in s]
        self.gold = [_gold_from_answer(it["answer"])
                     for s in self.subsets for it in s]

    def save_render(self, frame: np.ndarray) -> None:
        """Hook for subclasses / external loggers; default is no-op."""
        pass

    def evaluate(self, agents, agent_infos):
        results = []
        for i, agent in enumerate(agents):
            correct = 0
            received = 0
            format_errors = 0
            first_error = None
            batch_times = []
            subsets_run = 0

            for subset in self.subsets:
                q_batch = [it["question"] for it in subset]
                g_batch = [_gold_from_answer(it["answer"]) for it in subset]

                t0 = time.time()
                replies = agent.call(
                    "answer", q_batch,
                    timeout=BATCH_TIMEOUT,
                    catch_errors=True,
                )
                batch_times.append(time.time() - t0)

                if agent.last_error is not None:
                    # Proxy already typed the exception (e.g. "TimeoutError: ...");
                    # surface verbatim and stop sending more subsets.
                    first_error = first_error or agent.last_error
                    break

                # New API: (solutions: list[float], thinking_traces: list[str]).
                # JSON transport coerces tuples to lists, so accept a list of
                # two lists as the tuple form. Legacy: a flat list[float].
                if (
                    isinstance(replies, (list, tuple))
                    and len(replies) == 2
                    and all(isinstance(r, list) for r in replies)
                ):
                    solutions, thinking_traces = replies
                elif isinstance(replies, list):
                    solutions = replies
                    thinking_traces = ["" for _ in solutions]
                else:
                    first_error = first_error or (
                        f"TypeError: agent.answer must return list[float] or "
                        f"(list[float], list[str]); got {type(replies).__name__}."
                    )
                    break

                if not isinstance(solutions, list) or not isinstance(thinking_traces, list):
                    first_error = first_error or (
                        "TypeError: agent.answer must return list[float] or "
                        "(list[float], list[str])."
                    )
                    break
                if len(solutions) != len(q_batch):
                    first_error = first_error or (
                        f"ValueError: agent.answer returned {len(solutions)} solutions "
                        f"for a batch of {len(q_batch)} questions."
                    )
                    break
                if len(thinking_traces) != len(q_batch):
                    first_error = first_error or (
                        f"ValueError: agent.answer returned {len(thinking_traces)} thinking_traces "
                        f"for a batch of {len(q_batch)} questions."
                    )
                    break

                for q, trace, reply, gold in zip(q_batch, thinking_traces, solutions, g_batch):
                    received += 1
                    pred = _to_float(reply)
                    self.save_render(_render_frame(q, trace, reply))
                    if pred is None:
                        format_errors += 1
                        continue
                    if gold is not None and abs(pred - gold) < 1e-6:
                        correct += 1

                subsets_run += 1
                # Stage 1 → 2: after 2 subsets, need avg > 2/10 to continue
                if subsets_run == 2 and correct / subsets_run <= 2.0:
                    break
                # Stage 2 → 3: after 5 subsets, need avg > 8.5/10 to continue
                if subsets_run == 5 and correct / subsets_run <= 8.5:
                    break

            n_subsets = subsets_run if subsets_run > 0 else 1
            total = n_subsets * (EASY_PER_SUBSET + HARD_PER_SUBSET)
            avg_batch_time = (sum(batch_times) / len(batch_times)) if batch_times else 0.0
            entry = {
                "agent_index": i,
                "score": correct / n_subsets,
                "format_errors": format_errors,
                "questions_received": received,
                "avg_batch_time": avg_batch_time,
                "info_message": (
                    f"{correct} correct, {format_errors} formatting errors, "
                    f"{received}/{total} received"
                ),
            }
            if first_error is not None:
                entry["agent_code_error_message"] = first_error
                entry["is_agent_code_error"] = (correct == 0)
            results.append(entry)

        return {"agent_results": results}
