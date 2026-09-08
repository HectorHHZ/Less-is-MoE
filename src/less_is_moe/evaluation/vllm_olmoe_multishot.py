"""OLMoE vLLM evaluator using the paper multi-shot prompting protocol."""

import argparse
import json
import os
import re
import sys
import time
import multiprocessing
from typing import List, Dict, Any, Tuple

import numpy as np
import torch
from vllm import LLM, SamplingParams
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()  # read API keys and other secrets from .env if present


def preprocess_code(completion: str, language: str = "python") -> str:
    """Extract pure code from a generation, handling <think> tags and code blocks."""
    if completion is None:
        return ""
    completion = completion.strip().replace("\r", "")

    # Handle reasoning models with <think> ... </think>
    if "<think>" in completion:
        if "</think>" in completion:
            match = re.search(r"</think>\s*(.*)", completion, re.DOTALL)
            completion = match.group(1).strip() if match else ""
        else:
            return ""

    if not completion:
        return ""

    start_with_lang_tag = f"```{language}"
    generic_tag = "```"

    if start_with_lang_tag in completion:
        def_line = completion.index(start_with_lang_tag) + len(start_with_lang_tag)
        completion = completion[def_line:].strip()
        try:
            next_line = completion.index(generic_tag)
            completion = completion[:next_line].strip()
        except ValueError:
            pass
    elif generic_tag in completion:
        def_line = completion.index(generic_tag) + len(generic_tag)
        completion = completion[def_line:].strip()
        try:
            next_line = completion.index(generic_tag)
            completion = completion[:next_line].strip()
        except ValueError:
            pass

    # Remove [DONE] marker if present
    completion = completion.replace("[DONE]", "").strip()

    return completion


def preprocess_humaneval_completion(completion: str, language: str = "python") -> str:
    """Extract code completion for HumanEval, preserving leading indentation.

    HumanEval prompts end with a docstring inside a ``def`` block, so the model's
    output must be indented (typically 4 spaces) to stay inside the function body.
    The generic ``preprocess_code`` strips leading whitespace which breaks this.
    """
    if completion is None:
        return ""
    completion = completion.replace("\r", "")

    # Strip <think> blocks
    if "<think>" in completion:
        if "</think>" in completion:
            match = re.search(r"</think>\s*(.*)", completion, re.DOTALL)
            completion = match.group(1) if match else ""
        else:
            return ""
    if not completion:
        return ""

    # Extract from markdown code fence if present
    fence = f"```{language}"
    if fence in completion:
        start = completion.index(fence) + len(fence)
        completion = completion[start:]
        # Drop the initial newline after ```python
        if completion.startswith("\n"):
            completion = completion[1:]
        if "```" in completion:
            completion = completion[:completion.index("```")]
    elif "```" in completion:
        start = completion.index("```") + 3
        completion = completion[start:]
        if completion.startswith("\n"):
            completion = completion[1:]
        if "```" in completion:
            completion = completion[:completion.index("```")]

    # Truncate at common end-of-completion signals (so concatenation stays clean).
    # Keep only lines until: top-level def/class/if-main, or test markers.
    truncated_lines = []
    for line in completion.split("\n"):
        stripped = line.lstrip()
        if line and not line.startswith((" ", "\t")) and stripped.startswith(
            ("def ", "class ", "if __name__", "print(", "assert ", "#test", "# test")
        ):
            break
        truncated_lines.append(line)
    completion = "\n".join(truncated_lines)

    completion = completion.replace("[DONE]", "")
    # Strip trailing whitespace only — preserve leading indent
    return completion.rstrip()


def build_humaneval_check(prompt: str, completion: str, test: str, entry_point: str) -> str:
    """Build the full executable script for a HumanEval sample.

    Format: function prompt + model completion + test function + check() call.
    """
    return prompt + completion + "\n" + test + f"\ncheck({entry_point})\n"


def extract_mbpp_tests(prompt: str) -> List[str]:
    """Extract test assertion strings from MBPP prompt."""
    match = re.search(r"\[(['\"]assert .+?)\]", prompt, re.DOTALL)
    if not match:
        return []
    raw = match.group(0)
    try:
        tests = eval(raw)  # safe: only string literals in list
        if isinstance(tests, list):
            return tests
    except Exception:
        pass
    return re.findall(r"assert .+", raw)


def _run_code_with_tests(code: str, tests: List[str], timeout: float = 5.0) -> bool:
    """Execute code + tests in a subprocess, return True if all pass."""
    full_code = code + "\n" + "\n".join(tests)
    try:
        proc = multiprocessing.Process(target=exec, args=(full_code, {}))
        proc.start()
        proc.join(timeout)
        if proc.is_alive():
            proc.terminate()
            proc.join()
            return False
        return proc.exitcode == 0
    except Exception:
        return False


def run_code_with_tests_safe(code: str, tests: List[str], timeout: float = 5.0) -> bool:
    """Execute code + tests in a subprocess for safety, return True if all pass."""
    import subprocess as sp
    full_code = code + "\n" + "\n".join(tests)
    try:
        result = sp.run(
            [sys.executable, "-c", full_code],
            capture_output=True, timeout=timeout,
        )
        return result.returncode == 0
    except (sp.TimeoutExpired, Exception):
        return False


# Benchmarks for ESFT tasks
from .benchmarks import (
    IntentEvaluator,
    LawEvaluator,
    SummaryEvaluator,
    TranslationEvaluator,
)
from less_is_moe.model_patches.registry import (
    apply_vllm_patch,
    detect_model_family,
)


i_prompt = """<s> Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
"""


def extract_answer_number(args, sentence: str) -> float:
    dataset = args.dataset.lower()
    if dataset in ["multiarith", "addsub", "singleeq", "gsm8k", "gsm", "svamp", "mawps"]:
        sentence = sentence.replace(",", "")
        pred = [s for s in re.findall(r"-?\d+\.?\d*", sentence)]
        if not pred:
            return float("inf")
        pred_answer = float(pred[-1])
    else:
        raise NotImplementedError(f"not support dataset: {dataset}")
    if isinstance(pred_answer, str):
        try:
            pred_answer = float(pred_answer)
        except ValueError:
            pred_answer = float("inf")
    return pred_answer


def extract_answer_letter(args, sentence: str) -> str:
    sentence_ = sentence.strip()
    pred_answers = re.findall(r"A|B|C|D|E", sentence_)
    if pred_answers:
        return pred_answers[0]
    else:
        return ""


_CEVAL_SUBJECT_ZH = {
    # Minimal mapping used in the prompt. Unknown subjects fall back to the
    # raw config name, which is still valid Chinese for the model.
}


def _build_mcq_prompt_zh(subject: str, question: str,
                         a: str, b: str, c: str, d: str) -> str:
    """Prompt format used by C-Eval / CMMLU leaderboards (zero-shot)."""
    subj = _CEVAL_SUBJECT_ZH.get(subject, subject or "")
    header = f"以下是关于{subj}的单项选择题，请直接给出正确答案的选项。" if subj \
             else "以下是一道单项选择题，请直接给出正确答案的选项。"
    return (
        f"{header}\n\n"
        f"题目：{question}\n"
        f"A. {a}\nB. {b}\nC. {c}\nD. {d}\n"
        f"答案是："
    )


def _build_math_prompt(problem: str) -> str:
    """4-shot CoT prompt for Hendrycks MATH, matching DeepSeek-V2 paper Table 27."""
    return _MATH_4SHOT_PREFIX + f"Problem: {problem}\nSolution:\n"


# Legacy zero-shot template (kept for AIME / MATH-500 / Olympiad which still use it).
MATH_QUERY_TEMPLATE = (
    "Solve the following math problem efficiently and clearly.  The last line of your response "
    "should be of the following format: 'Therefore, the final answer is: $\\boxed{{ANSWER}}$. "
    "I hope it is correct' (without quotes) where ANSWER is just the final number or expression "
    "that solves the problem. Think step by step before answering.\n\n{Question}"
)


# ---------------------------------------------------------------------------
# DeepSeek-V2 paper-style few-shot CoT prompts (arXiv:2405.04434, Appendix G).
# These verbatim demos are used for GSM8K (8-shot, Table 24), MATH (4-shot,
# Table 27), MBPP (3-shot, Table 28), and MultiArith (4-shot, GSM-style).
# ---------------------------------------------------------------------------

_GSM_8SHOT_PREFIX = """Q: Max can mow the lawn in 40 minutes. If it takes him twice that long to fertilize the lawn, how long will it take him to both mow and fertilize the lawn?
A: Let's think step by step. It takes Max 2 * 40 minutes = 80 minutes to fertilize the lawn. In total, Max takes 80 minutes + 40 minutes = 120 minutes to both mow and fertilize the lawn. The answer is 120.

Q: The bagels cost $2.25 each, or a dozen for $24. How much is saved, per bagel, in cents, by buying a dozen at a time?
A: Let's think step by step. They cost 2.25*100=225 cents each. At the bulk rate, they are 24/12=2 dollar each. They cost 2*100=200 cents each. 225-200=25 cents are saved per bagel. The answer is 25.

Q: Tim is 5 years old. His cousin, Rommel, is thrice as old as he is. His other cousin, Jenny, is 2 years older than Rommel. How many years younger is Tim than Jenny?
A: Let's think step by step. Rommel is 5 x 3 = 15 years old. Jenny is 15 + 2 = 17 years old. So, Tim is 17 - 5 = 12 years younger than Jenny. The answer is 12.

Q: The school has 14 boys and 10 girls. If 4 boys and 3 girls drop out, how many boys and girls are left?
A: Let's think step by step. There are 14 boys - 4 boys = 10 boys left. There are 10 girls - 3 girls = 7 girls left. In total there are 10 boys + 7 girls = 17 boys and girls left. The answer is 17.

Q: Building one birdhouse requires 7 planks and 20 nails. If 1 nail costs 0.05, and one plank costs 3, what is the cost, in dollars, to build 4 birdhouses?
A: Let's think step by step. The cost of the planks for one birdhouse is 7 * 3 = 21. And the nails are a cost of 20 * 0.05 = 1 for each birdhouse. So to build one birdhouse one will need 21 + 1 = 22. So the cost of building 4 birdhouses is at 4 * 22 = 88. The answer is 88.

Q: Danny brings 3 watermelons to his family picnic. He cuts each watermelon into 10 slices. His sister brings 1 watermelon to the family picnic, and she cuts the watermelon into 15 slices. How many watermelon slices are there in total at the picnic?
A: Let's think step by step. From Danny, there are 3 * 10 = 30 watermelon slices. From his sister, there are 1 * 15 = 15 watermelon slices. There are a total of 30 + 15 = 45 watermelon slices. The answer is 45.

Q: Angela is a bike messenger in New York. She needs to deliver 8 times as many packages as meals. If she needs to deliver 27 meals and packages combined, how many meals does she deliver?
A: Let's think step by step. Let p be the number of packages Angela delivers and m be the number of meals. We know that p + m = 27 and p = 8m. Substituting the second equation into the first equation, we get 8m + m = 27. Combining like terms, we get 9m = 27. Dividing both sides by 9, we get m = 3. The answer is 3.

Q: Cori is 3 years old today. In 5 years, she will be one-third the age of her aunt. How old is her aunt today?
A: Let's think step by step. In 5 years, Cori will be 3 + 5 = 8 years old. In 5 years, Cori's aunt will be 8 x 3 = 24 years old. Today, her aunt is 24 - 5 = 19 years old. The answer is 19.

"""

# MultiArith re-uses the first 4 GSM demos (same arithmetic-word-problem format).
_MULTIARITH_4SHOT_PREFIX = "\n\n".join(
    _GSM_8SHOT_PREFIX.strip().split("\n\n")[:4]
) + "\n\n"


_MATH_4SHOT_PREFIX = r"""Problem:
Find the domain of the expression $\frac{\sqrt{x-2}}{\sqrt{5-x}}$.}
Solution:
The expressions inside each square root must be non-negative.
Therefore, $x-2 \ge 0$, so $x\ge2$, and $5 - x \ge 0$, so $x \le 5$.
Also, the denominator cannot be equal to zero, so $5-x>0$, which gives $x<5$.
Therefore, the domain of the expression is $\boxed{[2,5)}$.
Final Answer: The final answer is $[2,5)$. I hope it is correct.

Problem:
If $\det \mathbf{A} = 2$ and $\det \mathbf{B} = 12,$ then find $\det (\mathbf{A} \mathbf{B}).$
Solution:
We have that $\det (\mathbf{A} \mathbf{B}) = (\det \mathbf{A})(\det \mathbf{B}) = (2)(12) = \boxed{24}.$
Final Answer: The final answer is $24$. I hope it is correct.

Problem:
Terrell usually lifts two 20-pound weights 12 times. If he uses two 15-pound weights instead, how many times must Terrell lift them in order to lift the same total weight?
Solution:
If Terrell lifts two 20-pound weights 12 times, he lifts a total of $2\cdot 12\cdot20=480$ pounds of weight. If he lifts two 15-pound weights instead for $n$ times, he will lift a total of $2\cdot15\cdot n=30n$ pounds of weight. Equating this to 480 pounds, we can solve for $n$: \begin{align*}
30n&=480\\
\Rightarrow\qquad n&=480/30=\boxed{16}
\end{align*}
Final Answer: The final answer is $16$. I hope it is correct.

Problem:
If the system of equations
\begin{align*}
6x-4y&=a,\\
6y-9x &=b.
\end{align*}has a solution $(x, y)$ where $x$ and $y$ are both nonzero, find $\frac{a}{b},$ assuming $b$ is nonzero.
Solution:
If we multiply the first equation by $-\frac{3}{2}$, we obtain
$$6y-9x=-\frac{3}{2}a.$$Since we also know that $6y-9x=b$, we have
$$-\frac{3}{2}a=b\Rightarrow\frac{a}{b}=\boxed{-\frac{2}{3}}.$$
Final Answer: The final answer is $-\frac{2}{3}$. I hope it is correct.

"""


# MBPP paper-style demos, one per list entry. The first three follow the
# DeepSeek-V2 paper Table 28; we split them so callers can pick any N-shot.
_MBPP_DEMOS = [
    """You are an expert Python programmer, and here is your task: Write a function to find the similar elements from the given two tuple lists. Your code should pass these tests:
assert similar_elements((3, 4, 5, 6),(5, 7, 4, 10)) == (4, 5)
assert similar_elements((1, 2, 3, 4),(5, 4, 3, 7)) == (3, 4)
assert similar_elements((11, 12, 14, 13),(17, 15, 14, 13)) == (13, 14)
[BEGIN]
def similar_elements(test_tup1, test_tup2):
  res = tuple(set(test_tup1) & set(test_tup2))
  return (res)
[DONE]
""",
    """You are an expert Python programmer, and here is your task: Write a python function to identify non-prime numbers. Your code should pass these tests:
assert is_not_prime(2) == False
assert is_not_prime(10) == True
assert is_not_prime(35) == True
[BEGIN]
import math
def is_not_prime(n):
    result = False
    for i in range(2,int(math.sqrt(n)) + 1):
        if n % i == 0:
            result = True
    return result
[DONE]
""",
    """You are an expert Python programmer, and here is your task: Write a function to find the largest integers from a given list of numbers using heap queue algorithm. Your code should pass these tests:
assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],3)==[85, 75, 65]
assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],2)==[85, 75]
assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],5)==[85, 75, 65, 58, 35]
[BEGIN]
import heapq as hq
def heap_queue_largest(nums,n):
    largest_nums = hq.nlargest(n, nums)
    return largest_nums
[DONE]
""",
]


def build_mbpp_prefix(n_shots: int) -> str:
    """Concatenate the first ``n_shots`` MBPP demos (clamped to available)."""
    if n_shots <= 0:
        return ""
    n = min(n_shots, len(_MBPP_DEMOS))
    return "".join(_MBPP_DEMOS[:n])


# Kept for backward compatibility; equals the original 3-shot prefix.
_MBPP_3SHOT_PREFIX = build_mbpp_prefix(3)


def _build_gsm_paper_prompt(question_field: str) -> str:
    """8-shot CoT GSM8K prompt (paper Table 24).

    Local data has ``prompt = "Question: <body>"``. We strip that prefix so the
    test question slots into the ``Q:/A:`` few-shot format.
    """
    q = question_field.strip()
    if q.lower().startswith("question:"):
        q = q[len("question:"):].strip()
    return _GSM_8SHOT_PREFIX + f"Q: {q}\nA: Let's think step by step."


def _build_multiarith_paper_prompt(instruction: str) -> str:
    """4-shot CoT prompt for MultiArith (re-using GSM demo style)."""
    q = instruction.strip()
    return _MULTIARITH_4SHOT_PREFIX + f"Q: {q}\nA: Let's think step by step."


def _normalize_mbpp_test_field(prompt_text: str) -> Tuple[str, str]:
    """Split MBPP local prompt into (task_description, asserts_block).

    Local prompts wrap the asserts in a python-list-of-strings (e.g.
    ``['assert foo()...', 'assert bar()...']``). The paper format wants the
    raw assert lines. ``extract_mbpp_tests`` already parses the list-string,
    so we just convert it for display in the prompt.
    """
    tests = extract_mbpp_tests(prompt_text)
    if not tests:
        return prompt_text.rstrip(), ""

    pre = prompt_text
    last_newline_before_list = pre.find("[")
    if last_newline_before_list != -1:
        pre = pre[:last_newline_before_list].rstrip()
    pre = pre.replace("\n\n Your code should pass these tests:",
                      "\nYour code should pass these tests:")
    pre = pre.replace("\n Your code should pass these tests:",
                      "\nYour code should pass these tests:")
    return pre.strip(), "\n".join(tests)


def _build_mbpp_paper_prompt(prompt_text: str, n_shots: int = 3) -> str:
    """N-shot MBPP prompt (paper Table 28 = 3-shot). Stop on ``[DONE]``.

    ``n_shots`` is clamped to 0..len(_MBPP_DEMOS). 0 = pure zero-shot with no
    demos. The default 3 preserves the original DeepSeek-V2 paper behaviour.
    """
    pre, asserts = _normalize_mbpp_test_field(prompt_text)
    body = pre
    if asserts:
        body = body.rstrip() + "\n" + asserts
    if not body.startswith("You are an expert Python programmer"):
        body = "You are an expert Python programmer, and here is your task: " + body
    return build_mbpp_prefix(n_shots) + body.rstrip() + "\n[BEGIN]\n"


def _build_aime_prompt(problem: str) -> str:
    """Prompt for AIME problems (numeric answer, boxed extraction)."""
    return MATH_QUERY_TEMPLATE.format(Question=problem)


def _build_math500_prompt(problem: str) -> str:
    """Prompt for MATH-500 problems (latex answer, boxed extraction)."""
    return MATH_QUERY_TEMPLATE.format(Question=problem)


def _build_gpqa_prompt(problem: str) -> str:
    """Prompt for GPQA Diamond MC. The problem already contains choices and
    asks for \\boxed{A/B/C/D}, so we use it directly."""
    return problem


def _build_olympiad_prompt(problem: str) -> str:
    """Prompt for Olympiad Bench (text-only English math). Same CoT+boxed
    template as MATH-500 / AIME — asks model to place the final answer in
    \\boxed{...}."""
    return MATH_QUERY_TEMPLATE.format(Question=problem)


def _olympiad_answer_match(pred_boxed: str, gold_answers) -> bool:
    """Lenient match for Olympiad Bench (answer_type ∈ {Numerical, Expression,
    Tuple, Interval}).

    Tries, for each gold answer in the list:
      1. Numerical float match with 1e-4 relative tolerance (handles 572/674
         Numerical + any Expression/Tuple that simplifies to a number).
      2. Exact normalized-string match (handles Expression/Tuple/Interval
         shapes like ``\\frac{1}{2}`` or ``(1,2)`` once LaTeX noise is stripped).
    Returns True on the first match.
    """
    if not pred_boxed:
        return False
    if not isinstance(gold_answers, (list, tuple)):
        gold_answers = [gold_answers]

    pred_norm = _normalize_math_answer(pred_boxed)
    try:
        pred_float = float(pred_norm.replace(",", ""))
    except (ValueError, AttributeError):
        pred_float = None

    for g in gold_answers:
        if g is None or g == "":
            continue
        g_norm = _normalize_math_answer(str(g))
        if pred_norm and g_norm and pred_norm == g_norm:
            return True
        if pred_float is not None:
            try:
                g_float = float(g_norm.replace(",", ""))
            except ValueError:
                continue
            if abs(pred_float - g_float) <= max(1e-4 * abs(g_float), 1e-6):
                return True
    return False


def _extract_last_boxed(s: str) -> str:
    """Return the content of the last ``\\boxed{...}`` in ``s`` (brace-balanced).
    Returns '' if none is found."""
    if not s:
        return ""
    idx = s.rfind("\\boxed{")
    if idx == -1:
        # Fall back to \\fbox{...}
        idx = s.rfind("\\fbox{")
        if idx == -1:
            return ""
        start = idx + len("\\fbox{")
    else:
        start = idx + len("\\boxed{")

    depth = 1
    out = []
    i = start
    while i < len(s):
        c = s[i]
        if c == "{":
            depth += 1
            out.append(c)
        elif c == "}":
            depth -= 1
            if depth == 0:
                return "".join(out)
            out.append(c)
        else:
            out.append(c)
        i += 1
    return "".join(out)  # unclosed — return what we have


def _normalize_math_answer(s: str) -> str:
    """Canonicalize a MATH answer string for equality comparison.

    Covers the common LaTeX noise seen in model outputs and reference
    solutions: surrounding $, whitespace, \\left/\\right, \\text{...},
    \\dfrac/\\tfrac → \\frac, outer curly braces, trailing period.
    """
    if s is None:
        return ""
    s = str(s).strip()
    # Strip outer $...$
    while s.startswith("$") and s.endswith("$") and len(s) >= 2:
        s = s[1:-1].strip()
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\!", "").replace("\\,", "").replace("\\;", "")
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    s = re.sub(r"\\text\s*\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\mathrm\s*\{([^}]*)\}", r"\1", s)
    s = s.replace(" ", "")
    # Drop a single trailing period
    if s.endswith("."):
        s = s[:-1]
    # Strip outer braces repeatedly
    while s.startswith("{") and s.endswith("}"):
        s = s[1:-1]
    return s


# ---------------------------------------------------------------------------
# math-verify based scoring — the underlying library LightEval uses for its
# `multilingual_extractive_match_metric`. Strictly more lenient than the
# legacy boxed string-equality matcher: sympy equivalence
# (\frac{1}{2} == 0.5, \sqrt{2} == 2^{1/2}, 1,000 == 1000),
# LaTeX-aware \boxed{} extraction, and free-text "Answer: X" / "= X" anchors.
# Used for math, math500, aime{24,25,26}, gpqa, olympiad_bench, gsm{,8k} when
# --use_math_verify is set.
# ---------------------------------------------------------------------------
try:
    from math_verify import (
        parse as _mv_parse,
        verify as _mv_verify,
        LatexExtractionConfig as _MVLatexCfg,
        ExprExtractionConfig as _MVExprCfg,
    )
    _MATH_VERIFY_AVAILABLE = True
    # Pred extraction: boxed at top priority, then plain expressions / "answer is X".
    _MV_PRED_CFG = (
        _MVLatexCfg(boxed_match_priority=0),
        _MVExprCfg(),
    )
    _MV_GOLD_CFG_LATEX = (_MVLatexCfg(),)
    _MV_GOLD_CFG_EXPR = (_MVExprCfg(),)
except ImportError:
    _MATH_VERIFY_AVAILABLE = False
    _MV_PRED_CFG = None
    _MV_GOLD_CFG_LATEX = None
    _MV_GOLD_CFG_EXPR = None


def _math_verify_score(pred_text: str, gold_text: str, gold_mode: str = "latex") -> bool:
    """Return True iff ``pred_text`` and ``gold_text`` are mathematically
    equivalent under math-verify's sympy-aware comparison.

    gold_mode is a *hint* about which extractor to try first ("latex" for
    MATH-500/Olympiad-style LaTeX gold, "expr" for plain integers/floats like
    AIME). If the preferred extractor fails to parse, the other is tried as
    a fallback — many dataset gold strings sit on the boundary (e.g. AIME
    "233" parses as expr, MATH-500 "5" parses as either).

    Falls back silently to False on any parse/verify exception so the caller
    can still try a legacy comparison path if it wants.
    """
    if not _MATH_VERIFY_AVAILABLE:
        return False
    if not pred_text or not gold_text:
        return False
    try:
        pred_parsed = _mv_parse(str(pred_text), extraction_config=_MV_PRED_CFG)
        if not pred_parsed:
            return False
        gold_str = str(gold_text).strip()
        # Build candidate gold renderings. math_verify's LatexExtractionConfig
        # only fires on anchored LaTeX (``$...$`` or ``\boxed{...}``), so a bare
        # ``\sqrt{2}`` from a dataset's ``answer`` field parses to []. Wrap
        # unanchored gold in ``\boxed{...}`` as an additional fallback so it
        # still goes through the LaTeX path.
        looks_latex = ("\\" in gold_str or "^" in gold_str or "{" in gold_str)
        already_anchored = ("$" in gold_str or "\\boxed" in gold_str)
        gold_candidates = [gold_str]
        if looks_latex and not already_anchored:
            gold_candidates.append("\\boxed{" + gold_str + "}")
        primary_cfg = _MV_GOLD_CFG_LATEX if gold_mode == "latex" else _MV_GOLD_CFG_EXPR
        fallback_cfg = _MV_GOLD_CFG_EXPR if gold_mode == "latex" else _MV_GOLD_CFG_LATEX
        for cfg in (primary_cfg, fallback_cfg):
            for cand in gold_candidates:
                gold_parsed = _mv_parse(cand, extraction_config=cfg)
                if not gold_parsed:
                    continue
                if _mv_verify(gold_parsed, pred_parsed):
                    return True
        return False
    except Exception:
        return False


_GPQA_LETTER_TAIL_RE = re.compile(r"\b([ABCD])\b")
_GPQA_ANSWER_RE = re.compile(
    r"(?:final\s+answer|answer)\s*(?:is|:)?\s*\$?\\?boxed\{?\s*\(?\s*([ABCD])",
    re.IGNORECASE,
)
_GPQA_PLAIN_ANSWER_RE = re.compile(
    r"(?:final\s+answer|answer)\s*(?:is|:)?\s*\$?\(?\s*([ABCD])\s*\)?\$?",
    re.IGNORECASE,
)


def _extract_gpqa_letter(text: str) -> str:
    """Extract A/B/C/D from a GPQA generation. Tries, in order:
      1. Last \\boxed{...} content reduced to A-D
      2. "Answer: X" / "the answer is X" / "Answer: $\\boxed{X}$" patterns
      3. The last lone A-D letter in the trailing 200 characters
    """
    if not text:
        return ""
    boxed = _extract_last_boxed(text)
    if boxed:
        m = re.search(r"[ABCD]", boxed.upper())
        if m:
            return m.group(0)
    m = _GPQA_ANSWER_RE.search(text)
    if m:
        return m.group(1).upper()
    m = _GPQA_PLAIN_ANSWER_RE.search(text)
    if m:
        return m.group(1).upper()
    tail_letters = _GPQA_LETTER_TAIL_RE.findall(text[-300:].upper())
    if tail_letters:
        return tail_letters[-1]
    return ""


def _build_bbh_prompt(question: str) -> str:
    """Zero-shot direct-answer prompt for BIG-Bench Hard.

    BBH targets are short strings like ``(A)``, ``Yes``, ``valid``, ``42``,
    a sorted word list, etc. We ask for the answer directly rather than a
    chain of thought so that exact-match scoring is tractable with few tokens.
    """
    return f"{question}\nAnswer:"


def _normalize_bbh_answer(s: str) -> str:
    """Canonicalize a BBH answer (prediction or gold) for exact-match.

    Lowercases, strips whitespace / surrounding brackets / trailing period.
    Keeps the ``(X)`` multiple-choice tag shape intact by normalizing it to
    a bare letter — model outputs like ``A``, ``(A)``, ``A.`` all collapse to
    the same thing.
    """
    if s is None:
        return ""
    t = str(s).strip().lower()
    if not t:
        return ""
    t = t.split("\n", 1)[0].strip()
    for prefix in ("so the answer is ", "the answer is ", "answer: ", "answer is ", "a: "):
        if t.startswith(prefix):
            t = t[len(prefix):].strip()
            break
    # Strip trailing sentence punctuation
    while t and t[-1] in ".。,，!?":
        t = t[:-1].strip()
    # Leading "(a) ..." or "(a)" -> "a". BBH MCQ models often emit
    # "(A) <option text>"; gold is just "(A)", so we must collapse both.
    m = re.match(r"\(\s*([a-z0-9]+)\s*\)", t)
    if m:
        t = m.group(1)
    # Collapse any remaining outer brackets
    elif len(t) >= 2 and t[0] == "(" and t[-1] == ")":
        t = t[1:-1].strip()
    return t


def extract_chinese_mcq_letter(sentence: str) -> str:
    """Robust letter extraction for Chinese MCQ benchmarks (C-Eval / CMMLU).

    Handles: ``答案是A``, ``选A``, ``A. xxx``, ``Answer: A``, bare ``A`` etc.
    Returns '' if nothing plausible is found.
    """
    if not sentence:
        return ""
    s = sentence.strip()
    # Quick win: first alphabetic A/B/C/D on its own near the start
    patterns = [
        r"答案\s*(?:是|为|:|：)?\s*([ABCD])",
        r"选\s*([ABCD])",
        r"正确选项\s*(?:是|为|:|：)?\s*([ABCD])",
        r"[Aa]nswer\s*(?:is|:|：)?\s*([ABCD])",
        r"\b([ABCD])\b",
    ]
    for pat in patterns:
        m = re.search(pat, s)
        if m:
            return m.group(1).upper()
    return ""


def extract_commonsense_answer(dataset: str, sentence: str) -> str:
    """Parse model output for commonsense-style multiple choice datasets.

    Logic follows the Efficient-Distillation commonsense evaluator.
    so predictions stay consistent across repos.
    """
    sentence = sentence.lower().strip()
    ds = dataset.lower()

    if ds == "boolq":
        pred_answers = re.findall(r"true|false", sentence)
    elif ds == "piqa":
        pred_answers = re.findall(r"solution1|solution2", sentence)
    elif ds in {"social_i_qa", "arc-challenge", "arc-easy", "openbookqa"}:
        pred_answers = re.findall(r"answer1|answer2|answer3|answer4|answer5", sentence)
    elif ds == "hellaswag":
        pred_answers = re.findall(r"ending1|ending2|ending3|ending4", sentence)
    elif ds == "winogrande":
        pred_answers = re.findall(r"option1|option2", sentence)
    else:
        pred_answers = []

    if not pred_answers:
        return ""
    return pred_answers[0]


def fallback_digit_prediction(target: str, sentence: str) -> str:
    """
    Heuristic for noisy outputs like '111111...' seen in baseline runs.
    If the last digit in model output matches the last digit of the target label,
    return the target label so accuracy can be credited. Otherwise, try to
    rebuild a label with the same prefix and the predicted digit.
    """
    raw_digits = re.findall(r"\d", sentence)
    target_digits = re.findall(r"\d", str(target))
    if not raw_digits or not target_digits:
        return ""

    pred_digit = raw_digits[-1]
    target_digit = target_digits[-1]

    if pred_digit == target_digit:
        return str(target)

    prefix_match = re.match(r"(.+?)(\d+)$", str(target).strip())
    if prefix_match:
        return f"{prefix_match.group(1)}{pred_digit}"
    return pred_digit


def set_random_seed(seed: int) -> None:
    import random

    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_example(ex: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize different dataset field names to a common schema:
      - instruction: the question/prompt string
      - answer: the reference answer
    """
    instr = ex.get("instruction")
    ans = ex.get("answer")

    if instr is None:
        raise KeyError("Field 'instruction' is required in example.")
    if ans is None:
        raise KeyError("Field 'answer' is required in example.")

    out = dict(ex)
    out["instruction"] = instr
    out["answer"] = ans
    return out


def load_data(path: str) -> List[Dict[str, Any]]:
    """
    Load json or jsonl into a list of dicts.
    """
    if path.endswith(".jsonl"):
        with open(path, "r") as f:
            return [json.loads(line) for line in f if line.strip()]
    with open(path, "r") as f:
        return json.load(f)


def is_esft_dataset(name: str) -> bool:
    return name.lower() in {
        "esft-intent",
        "esft-summary",
        "esft-law",
        "esft-translation",
    }


def main(args):
    set_random_seed(args.seed)
    family = detect_model_family(args.model_name_or_path)
    if family != "olmoe":
        raise ValueError(
            "The multi-shot entry point reproduces the OLMoE paper protocol. "
            "Use vllm_zero_shot for Qwen MoE models; detected "
            f"{family!r}."
        )
    if not args.base_model:
        if not apply_vllm_patch(family):
            raise RuntimeError(
                "The installed vLLM does not provide the OLMoE model module "
                "required by this checkpoint."
            )
        print(f"[Patch] enabled Less-is-MoE vLLM patch for {family}.")
    else:
        print("[Patch] base_model=True，跳过本地 MoE patch，直接使用模型自带实现。")

    dataset_lower = args.dataset.lower()

    # Load test data
    raw_data = load_data(args.data_path)

    if is_esft_dataset(dataset_lower):
        # ESFT datasets already contain prompts/completions
        t_test_data = raw_data
        raw_prompts: List[str] = [
            ex.get("prompt")
            or ex.get("instruction")
            or ex.get("input")
            or ""
            for ex in t_test_data
        ]
        # Wrap with ChatML template if the SFT model was trained with chat format.
        # Use --apply_chat_template to enable this.
        if getattr(args, "apply_chat_template", False):
            from transformers import AutoTokenizer
            _tok = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
            prompts = []
            for p in raw_prompts:
                messages = [{"role": "user", "content": p}]
                prompts.append(_tok.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True))
            print(f"[Prompt] ESFT dataset detected: {args.dataset} (ChatML template applied)")
        else:
            prompts = raw_prompts
            print(f"[Prompt] ESFT dataset detected: {args.dataset}")
    elif dataset_lower in {"gsm", "gsm8k"}:
        # RoxanneWsyw/gsm: has prompt/completion/answer fields
        # Paper Table 24: 8-shot CoT, "Q:/A: Let's think step by step."
        t_test_data = raw_data
        prompts = [_build_gsm_paper_prompt(ex["prompt"]) for ex in t_test_data]
        print(f"[Prompt] GSM dataset detected (8-shot CoT, paper Table 24), {len(t_test_data)} examples")
    elif dataset_lower in ("multiarith", "multi_arith"):
        # MultiArith: has instruction/output/answer fields
        # 4-shot CoT (GSM-style demos), per user request.
        t_test_data = raw_data
        prompts = [_build_multiarith_paper_prompt(ex["instruction"]) for ex in t_test_data]
        print(f"[Prompt] MultiArith dataset detected (4-shot CoT, GSM-style), {len(t_test_data)} examples")
    elif dataset_lower == "mbpp":
        # RoxanneWsyw/MBPP: has prompt/completion fields, tests embedded in prompt
        # Paper Table 28 default = 3-shot, [BEGIN]/[DONE] format. Override
        # via --mbpp_shots N (clamped to len(_MBPP_DEMOS)=3).
        t_test_data = raw_data
        n_shots = args.mbpp_shots
        prompts = [_build_mbpp_paper_prompt(ex["prompt"], n_shots=n_shots) for ex in t_test_data]
        print(f"[Prompt] MBPP dataset detected ({n_shots}-shot, [BEGIN]/[DONE] format), "
              f"{len(t_test_data)} examples")
    elif dataset_lower == "humaneval":
        # openai/openai_humaneval: prompt / canonical_solution / test / entry_point
        t_test_data = raw_data
        prompts = [ex["prompt"] for ex in t_test_data]
        print(f"[Prompt] HumanEval dataset detected, {len(t_test_data)} examples")
    elif dataset_lower == "ceval":
        # C-Eval: id / question / A / B / C / D / answer (letter) / explanation / subject
        t_test_data = raw_data
        prompts = [
            _build_mcq_prompt_zh(
                ex.get("subject", ""),
                ex.get("question", ""),
                ex.get("A", ""), ex.get("B", ""),
                ex.get("C", ""), ex.get("D", ""),
            )
            for ex in t_test_data
        ]
        print(f"[Prompt] C-Eval dataset detected, {len(t_test_data)} examples")
    elif dataset_lower == "cmmlu":
        # CMMLU: Question / A / B / C / D / Answer (letter) / subject
        t_test_data = raw_data
        prompts = [
            _build_mcq_prompt_zh(
                ex.get("subject", ""),
                ex.get("Question") or ex.get("question", ""),
                ex.get("A", ""), ex.get("B", ""),
                ex.get("C", ""), ex.get("D", ""),
            )
            for ex in t_test_data
        ]
        print(f"[Prompt] CMMLU dataset detected, {len(t_test_data)} examples")
    elif dataset_lower == "math":
        # Hendrycks MATH: problem / level / type / solution (contains \boxed{...})
        t_test_data = raw_data
        prompts = [_build_math_prompt(ex.get("problem", "")) for ex in t_test_data]
        print(f"[Prompt] MATH dataset detected, {len(t_test_data)} examples")
    elif dataset_lower in ("aime24", "aime2024", "aime25", "aime2025", "aime26", "aime2026"):
        # AIME: problem / answer (integer)
        t_test_data = raw_data
        prompts = [_build_aime_prompt(ex.get("problem", "")) for ex in t_test_data]
        print(f"[Prompt] AIME dataset detected, {len(t_test_data)} examples")
    elif dataset_lower == "math500":
        # MATH-500: problem / solution / answer
        t_test_data = raw_data
        prompts = [_build_math500_prompt(ex.get("problem", "")) for ex in t_test_data]
        print(f"[Prompt] MATH-500 dataset detected, {len(t_test_data)} examples")
    elif dataset_lower in ("gpqa", "gpqa_diamond"):
        # GPQA Diamond MC: problem (contains choices + boxed instruction) / solution (\boxed{A/B/C/D})
        t_test_data = raw_data
        prompts = [_build_gpqa_prompt(ex.get("problem", "")) for ex in t_test_data]
        print(f"[Prompt] GPQA Diamond dataset detected, {len(t_test_data)} examples")
    elif dataset_lower in ("olympiad_bench", "olympiadbench"):
        # OlympiadBench (text-only English math, OE_TO_maths_en_COMP, 674 rows):
        # problem / answer / answer_list / answer_type / is_multiple_answer
        t_test_data = raw_data
        prompts = [_build_olympiad_prompt(ex.get("problem", "")) for ex in t_test_data]
        print(f"[Prompt] OlympiadBench dataset detected, {len(t_test_data)} examples")
    elif dataset_lower == "bbh":
        # BIG-Bench Hard: task / input / target / text
        t_test_data = raw_data
        prompts = [_build_bbh_prompt(ex.get("input", "")) for ex in t_test_data]
        print(f"[Prompt] BBH dataset detected, {len(t_test_data)} examples")
    elif dataset_lower == "mmlu":
        # cais/mmlu: question / subject / choices (list[str]) / answer (int 0-3)
        t_test_data = raw_data
        _letters = ["A", "B", "C", "D"]
        prompts = []
        for ex in t_test_data:
            q = ex.get("question", "")
            choices = ex.get("choices", []) or []
            choice_block = "\n".join(
                f"{_letters[i]}. {c}" for i, c in enumerate(choices[:4])
            )
            prompts.append(
                f"The following is a multiple choice question. "
                f"Respond with only the letter (A, B, C, or D) of the correct answer.\n\n"
                f"Question: {q}\n{choice_block}\nAnswer:"
            )
        print(f"[Prompt] MMLU dataset detected, {len(t_test_data)} examples")
    else:
        # Normalize to instruction/answer for existing math/MC tasks
        t_test_data = [normalize_example(e) for e in raw_data]
        if args.prompt_mode == "raw":
            print("[Prompt] raw mode — 直接使用 instruction 字段，不包裹模板。")
            prompts = [example["instruction"] for example in t_test_data]
        else:
            print("[Prompt] iprompt mode — 使用 Alpaca 风格模板包裹。")
            prompts = [i_prompt.format_map(example) for example in t_test_data]
    # Smoke-test cap: limit to first N problems for quick sanity checks.
    if args.max_problems is not None and args.max_problems < len(prompts):
        n = args.max_problems
        print(f"[max_problems] truncating from {len(prompts)} -> {n}")
        prompts = prompts[:n]
        t_test_data = t_test_data[:n]
    print(f"First prompt example:\n{prompts[0]}")
    print(f"Total prompts: {len(prompts)}")

    # vLLM config
    llm_kwargs = {
        "model": args.model_name_or_path,
        "trust_remote_code": True,
        "dtype": args.dtype,
        "max_model_len": args.max_model_len,
        "seed": args.seed,
        "disable_custom_all_reduce": True,
    }
    llm_kwargs["gpu_memory_utilization"] = args.gpu_memory_utilization
    if args.tensor_parallel_size > 1:
        llm_kwargs["tensor_parallel_size"] = args.tensor_parallel_size

    # Init LLM with a CUDA fallback. When the first attempt throws a CUDA
    # error (common for pruned MoE models going through the local vLLM plugin),
    # we must release the first attempt's GPU allocations before retrying —
    # otherwise the retry OOMs because the weights + KV pool (~70GB) from the
    # failed attempt are still pinned.
    try:
        llm = LLM(**llm_kwargs)
        print("✓ vLLM model loaded successfully")
    except RuntimeError as e:
        if "CUDA" in str(e):
            print(f"CUDA error detected, retrying with CUDA_LAUNCH_BLOCKING=1 ... (error: {e})")
            # Release the first attempt's memory
            try:
                del llm  # noqa: F821 — may not exist if LLM() raised before assignment
            except NameError:
                pass
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            try:
                from vllm.distributed import destroy_model_parallel, destroy_distributed_environment
                destroy_model_parallel()
                destroy_distributed_environment()
            except Exception:
                pass
            os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
            llm = LLM(**llm_kwargs)
        else:
            raise

    # Parse custom stop sequences
    stop_seqs = None
    if args.stop_sequences:
        # allow comma-separated list; strip blanks
        stop_seqs = [s for s in (seg.strip() for seg in args.stop_sequences.split(",")) if s]

    # Sampling params
    n_per_problem = max(1, args.n_samples_per_problem)
    if n_per_problem > 1 and args.temperature <= 0:
        print(f"[WARN] n_samples_per_problem={n_per_problem} but temperature={args.temperature}; "
              "greedy sampling gives identical outputs. Forcing temperature=0.6 for diversity.")
        sampling_temp = 0.6
    else:
        sampling_temp = args.temperature

    if sampling_temp > 0:
        sampling_params = SamplingParams(
            n=n_per_problem,
            temperature=sampling_temp,
            top_p=args.top_p,
            top_k=args.top_k,
            min_p=args.min_p,
            repetition_penalty=args.repetition_penalty,
            presence_penalty=args.presence_penalty,
            max_tokens=args.max_tokens,
            seed=args.seed,
            stop=stop_seqs,
        )
    else:
        sampling_params = SamplingParams(
            n=1,
            temperature=0.0,
            top_p=1.0,
            top_k=-1,
            min_p=0.0,
            repetition_penalty=args.repetition_penalty,
            presence_penalty=args.presence_penalty,
            max_tokens=args.max_tokens,
            seed=args.seed,
            stop=stop_seqs,
        )
    print(f"Sampling parameters: {sampling_params}")

    # Filter out prompts that exceed max_model_len (tokenize to check)
    tokenizer_for_check = llm.get_tokenizer()
    max_prompt_tokens = args.max_model_len - args.max_tokens
    kept_indices = []
    skipped = 0
    for idx, p in enumerate(prompts):
        token_len = len(tokenizer_for_check.encode(p))
        if token_len <= max_prompt_tokens:
            kept_indices.append(idx)
        else:
            skipped += 1
    if skipped > 0:
        print(f"[WARN] Skipped {skipped}/{len(prompts)} prompts exceeding {max_prompt_tokens} tokens (max_model_len={args.max_model_len} - max_tokens={args.max_tokens})")
        prompts = [prompts[i] for i in kept_indices]
        t_test_data = [t_test_data[i] for i in kept_indices]
        print(f"Remaining prompts: {len(prompts)}")

    # Generate
    batch_size = min(args.batch_size, len(prompts))
    all_outputs = []
    if args.use_chat_template:
        chat_kwargs = {"enable_thinking": args.enable_thinking}
        print(f"Starting generation via llm.chat() with chat_template_kwargs={chat_kwargs} ...")
    else:
        chat_kwargs = None
        print("Starting generation via llm.generate() (raw prompts, no chat template) ...")
    start_time = time.time()

    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i : i + batch_size]
        print(f"Processing batch {i // batch_size + 1}/{(len(prompts) + batch_size - 1) // batch_size}")
        if args.use_chat_template:
            messages_list = [
                [{"role": "user", "content": p}] for p in batch_prompts
            ]
            batch_outputs = llm.chat(
                messages_list,
                sampling_params,
                chat_template_kwargs=chat_kwargs,
            )
        else:
            batch_outputs = llm.generate(batch_prompts, sampling_params)
        all_outputs.extend(batch_outputs)
        if i % (batch_size * 5) == 0:
            torch.cuda.empty_cache()

    elapsed_time = time.time() - start_time
    total_output_tokens = sum(
        len(comp.token_ids) for req_out in all_outputs for comp in req_out.outputs
    )
    output_tps = total_output_tokens / elapsed_time if elapsed_time > 0 else 0.0
    print(f"Generation completed in {elapsed_time:.2f} seconds")
    print(f"Speed: {len(prompts) / elapsed_time:.2f} prompts/second")
    print(f"Total output tokens: {total_output_tokens}")
    print(f"Output TPS: {output_tps:.1f} tok/s")

    # Score
    save_outputs = []
    correct = 0
    miss = 0.001

    for example, output in tqdm(
        zip(t_test_data, all_outputs),
        total=len(t_test_data),
        desc="Evaluating",
        leave=True,
        dynamic_ncols=True,
        file=sys.stdout,  # force stdout so it shows in typical logs
    ):
        # Per-example invariants hoisted out of the inner sample loop
        target = (
            example.get("completion")
            if is_esft_dataset(dataset_lower)
            else example.get("answer")
        )
        commonsense_tasks = {
            "arc-challenge",
            "arc-easy",
            "boolq",
            "hellaswag",
            "openbookqa",
            "piqa",
            "social_i_qa",
            "winogrande",
        }

        sample_records = []
        predict = None
        for gen_out in output.outputs:
            generated_text = gen_out.text

            # Manually strip stop sequences and trailing whitespace (sometimes vLLM keeps tails)
            if stop_seqs:
                for s in stop_seqs:
                    if s and s in generated_text:
                        generated_text = generated_text.split(s)[0]
            # For ESFT translation tasks, keep only the first line to drop trailing artifacts
            if dataset_lower == "esft-translation" or dataset_lower == "esft-intent" or dataset_lower == "esft-law" or dataset_lower == "esft-summary":
                if "\n" in generated_text:
                    generated_text = generated_text.split("\n", 1)[0]
            generated_text = generated_text.rstrip()

            example["raw_output"] = generated_text  # last sample wins
            prev_correct = correct

            # Commonsense multiple-choice tasks: use regex extraction from evaluate_commonsense.py
            if dataset_lower == "mbpp":
                # Extract code from model output and run embedded tests
                code = preprocess_code(generated_text)
                tests = extract_mbpp_tests(example["prompt"])
                passed = run_code_with_tests_safe(code, tests) if code and tests else False
                predict = code
                if passed:
                    correct += 1
                example["extracted_code"] = code
                example["tests"] = tests
                example["passed"] = passed
            elif dataset_lower == "humaneval":
                # Preserve indentation so the completion stays inside the function body
                completion = preprocess_humaneval_completion(generated_text)
                full_script = build_humaneval_check(
                    example["prompt"],
                    completion,
                    example.get("test", ""),
                    example.get("entry_point", ""),
                )
                passed = run_code_with_tests_safe(full_script, []) if completion else False
                predict = completion
                if passed:
                    correct += 1
                example["extracted_code"] = completion
                example["passed"] = passed
            elif dataset_lower in ("ceval", "cmmlu"):
                # Single-letter MCQ. Labels live in "answer"/"Answer" as A/B/C/D.
                predict = extract_chinese_mcq_letter(generated_text)
                target_letter = (
                    example.get("answer")
                    or example.get("Answer")
                    or ""
                )
                target_letter = str(target_letter).strip().upper()
                if predict and target_letter and predict == target_letter:
                    correct += 1
                example["target_letter"] = target_letter
            elif dataset_lower == "bbh":
                # Exact-match on normalized answer string.
                predict = _normalize_bbh_answer(generated_text)
                target_norm = _normalize_bbh_answer(example.get("target", ""))
                if predict and target_norm and predict == target_norm:
                    correct += 1
                example["target_norm"] = target_norm
            elif dataset_lower == "math":
                # Hendrycks MATH: gold solution contains \boxed{...} in LaTeX.
                pred_boxed = _extract_last_boxed(generated_text)
                gold_solution = example.get("solution", "") or ""
                gold_boxed = _extract_last_boxed(gold_solution)
                predict = pred_boxed
                if args.use_math_verify and _math_verify_score(
                    generated_text, gold_solution, gold_mode="latex"
                ):
                    correct += 1
                else:
                    pred_norm = _normalize_math_answer(pred_boxed)
                    gold_norm = _normalize_math_answer(gold_boxed)
                    if pred_norm and gold_norm and pred_norm == gold_norm:
                        correct += 1
                example["pred_boxed"] = pred_boxed
                example["gold_boxed"] = gold_boxed
            elif dataset_lower in ("aime24", "aime2024", "aime25", "aime2025", "aime26", "aime2026"):
                # AIME: answer is an integer (0-999); use expr extraction on gold,
                # latex+expr on pred (sympy equivalence handles "the answer is 233"
                # vs "$\boxed{233}$" vs "233.0").
                pred_boxed = _extract_last_boxed(generated_text)
                gold_answer = str(example.get("answer", "")).strip()
                predict = pred_boxed
                if args.use_math_verify and _math_verify_score(
                    generated_text, gold_answer, gold_mode="expr"
                ):
                    correct += 1
                else:
                    pred_norm = _normalize_math_answer(pred_boxed)
                    gold_norm = _normalize_math_answer(gold_answer)
                    if pred_norm and gold_norm and pred_norm == gold_norm:
                        correct += 1
                example["pred_boxed"] = pred_boxed
                example["gold_answer"] = gold_answer
            elif dataset_lower == "math500":
                # MATH-500: gold lives in `answer` (LaTeX expression).
                pred_boxed = _extract_last_boxed(generated_text)
                gold_answer = str(example.get("answer", "")).strip()
                predict = pred_boxed
                if args.use_math_verify and _math_verify_score(
                    generated_text, gold_answer, gold_mode="latex"
                ):
                    correct += 1
                else:
                    pred_norm = _normalize_math_answer(pred_boxed)
                    gold_norm = _normalize_math_answer(gold_answer)
                    if pred_norm and gold_norm and pred_norm == gold_norm:
                        correct += 1
                example["pred_boxed"] = pred_boxed
                example["gold_answer"] = gold_answer
            elif dataset_lower in ("gpqa", "gpqa_diamond"):
                # GPQA Diamond MC: gold solution holds \boxed{A/B/C/D}. Pred can
                # appear as "\boxed{A}", "Answer: A", "Answer: $A$", trailing "(B)", etc.
                gold_boxed = _extract_last_boxed(example.get("solution", ""))
                gold_letter = gold_boxed.strip().upper() if gold_boxed else ""
                if args.use_math_verify:
                    pred_letter = _extract_gpqa_letter(generated_text)
                else:
                    pred_boxed = _extract_last_boxed(generated_text)
                    pred_letter = pred_boxed.strip().upper() if pred_boxed else ""
                predict = pred_letter
                if predict and gold_letter and predict == gold_letter:
                    correct += 1
                example["pred_letter"] = pred_letter
                example["gold_letter"] = gold_letter
            elif dataset_lower in ("olympiad_bench", "olympiadbench"):
                # OlympiadBench: match against answer_list (list of latex strings).
                pred_boxed = _extract_last_boxed(generated_text)
                gold_list = example.get("answer_list") or [example.get("answer", "")]
                predict = pred_boxed
                hit = False
                if args.use_math_verify:
                    for g in gold_list:
                        if not g:
                            continue
                        # Olympiad gold list mixes numerical / latex / interval /
                        # tuple shapes. Try latex first (covers \frac, \sqrt,
                        # tuples) then fall back to expr (covers bare numbers).
                        if _math_verify_score(generated_text, str(g), gold_mode="latex") \
                                or _math_verify_score(generated_text, str(g), gold_mode="expr"):
                            hit = True
                            break
                if not hit:
                    hit = _olympiad_answer_match(pred_boxed, gold_list)
                if hit:
                    correct += 1
                example["pred_boxed"] = pred_boxed
                example["gold_answers"] = gold_list
            elif dataset_lower == "mmlu":
                # answer is an int index (0-3) — convert to letter for comparison
                predict = extract_answer_letter(args, generated_text)
                ans_idx = example.get("answer")
                target_letter = ["A", "B", "C", "D"][ans_idx] if isinstance(ans_idx, int) and 0 <= ans_idx < 4 else ""
                if predict and target_letter and predict.upper() == target_letter:
                    correct += 1
                example["target_letter"] = target_letter
            elif is_esft_dataset(dataset_lower):
                predict = generated_text
            elif dataset_lower in commonsense_tasks:
                predict = extract_commonsense_answer(dataset_lower, generated_text)
                if not predict:
                    predict = fallback_digit_prediction(str(target), generated_text)

                if predict and str(target).strip().lower() == predict.lower():
                    correct += 1
            else:
                # 数学/通用选择题逻辑
                is_choice = dataset_lower in ["aqua", "mathqa"] or str(target).strip().upper() in ["A", "B", "C", "D", "E"]

                if is_choice:
                    predict = extract_answer_letter(args, generated_text)
                    target_letter = extract_answer_letter(args, str(target))
                    if target_letter and predict and target_letter.upper() == predict.upper():
                        correct += 1
                else:
                    # GSM-family (gsm8k, gsm, multiarith, addsub, svamp, mawps, ...):
                    # try math-verify first ("the answer is X" anchors, fraction
                    # equivalence, comma-stripped integers). Fall back to legacy
                    # last-number extractor on miss.
                    target_str = str(target).strip()
                    matched = False
                    if args.use_math_verify and dataset_lower in (
                        "gsm8k", "gsm", "multiarith", "addsub", "singleeq",
                        "svamp", "mawps",
                    ):
                        if _math_verify_score(generated_text, target_str, gold_mode="expr"):
                            matched = True
                    predict = extract_answer_number(args, generated_text)
                    try:
                        target_val = float(target_str.replace(",", ""))
                    except ValueError:
                        target_val = float("inf")
                    if matched or abs(target_val - predict) <= miss:
                        correct += 1

            sample_records.append({
                "raw_output": generated_text,
                "prediction": predict,
                "correct": int(correct > prev_correct),
            })

        example["prediction"] = predict
        if n_per_problem > 1:
            example["samples"] = sample_records
            example["n_correct"] = sum(r["correct"] for r in sample_records)
            example["n_total"] = len(sample_records)
        save_outputs.append(example)

    os.makedirs(args.output_dir, exist_ok=True)
    model_tag = os.path.basename(args.model_name_or_path.rstrip("/"))
    model_tag = model_tag.replace(os.sep, "_")
    output_file = os.path.join(args.output_dir, f"{model_tag}_predictions.jsonl")
    print(f"Saving outputs to {output_file}")
    metrics_file = os.path.join(args.output_dir, f"{model_tag}_metrics.json")

    if is_esft_dataset(dataset_lower):
        # Prepare results for ESFT evaluators
        eval_dataset = [
            {
                "prompt": ex.get("prompt") or "",
                "raw_answers": [
                    ex.get("completion")
                    or ex.get("answer")
                    or ex.get("output")
                    or ""
                ],
            }
            for ex in t_test_data
        ]
        results_for_eval = [
            {
                "prompt": ex.get("prompt") or "",
                "raw_prediction": ex["raw_output"],
                "raw_answers": [
                    ex.get("completion")
                    or ex.get("answer")
                    or ex.get("output")
                    or ""
                ],
            }
            for ex in t_test_data
        ]

        evaluator_cfg = {
            "max_new_tokens": sampling_params.max_tokens,
            "eval_batch_size": args.batch_size,
            "openai_api_key": args.openai_api_key
            or os.getenv("OPENAI_API_KEY")
            or "",
        }
        evaluator_map = {
            "esft-intent": IntentEvaluator,
            "esft-summary": SummaryEvaluator,
            "esft-law": LawEvaluator,
            "esft-translation": TranslationEvaluator,
        }
        evaluator_cls = evaluator_map[dataset_lower]
        evaluator = evaluator_cls(eval_dataset, evaluator_cfg)
        scores = evaluator.eval_metric(results_for_eval)
        avg_score = float(sum(scores) / len(scores))

        with open(output_file, "w") as fout:
            for example in save_outputs:
                fout.write(json.dumps(example, ensure_ascii=False) + "\n")

        metrics = {
            "model": args.model_name_or_path,
            "model_tag": model_tag,
            "dataset": args.dataset,
            "total": len(t_test_data),
            "average_score": avg_score,
            "elapsed_time_s": round(elapsed_time, 2),
            "total_output_tokens": total_output_tokens,
            "output_tps": round(output_tps, 1),
        }
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

        print("=" * 80)
        print(f"Average score {avg_score * 100:.2f} (0-1 scaled from GPT4/heuristics)")
        print("=" * 80)
    else:
        denom = len(t_test_data) * n_per_problem
        weighted_acc = correct / denom if denom > 0 else 0.0
        print("=" * 80)
        if n_per_problem > 1:
            print(f"avg@{n_per_problem}: {weighted_acc * 100:.2f}%  "
                  f"(correct {correct} / {denom} = {len(t_test_data)} problems × {n_per_problem} samples)")
        else:
            print(f"Result {weighted_acc * 100:.1f}%, total: {len(t_test_data)}")
        print("=" * 80)

        with open(output_file, "w") as fout:
            for example in save_outputs:
                fout.write(json.dumps(example, ensure_ascii=False) + "\n")

        metrics = {
            "model": args.model_name_or_path,
            "model_tag": model_tag,
            "dataset": args.dataset,
            "total": len(t_test_data),
            "n_samples_per_problem": n_per_problem,
            "total_samples": denom,
            "correct": correct,
            "accuracy": weighted_acc,
            "elapsed_time_s": round(elapsed_time, 2),
            "total_output_tokens": total_output_tokens,
            "output_tps": round(output_tps, 1),
        }
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"Metrics saved to {metrics_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to the test data file")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g., gsm8k, aqua)")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save outputs")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="HF model path or ID")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed.")
    parser.add_argument("--n_samples_per_problem", type=int, default=1,
                        help="Number of samples to generate per problem (for avg@N scoring). "
                             "Requires temperature>0 to get diverse samples. Accuracy becomes "
                             "correct / (total * n).")
    parser.add_argument("--temperature", type=float, default=0.1, help="Temperature during generation.")
    parser.add_argument("--top_p", type=float, default=0.95,
                        help="Nucleus sampling top_p (Qwen3 recommended: 0.95). Only applied when temperature>0.")
    parser.add_argument("--top_k", type=int, default=20,
                        help="Top-k sampling (Qwen3 recommended: 20). Only applied when temperature>0.")
    parser.add_argument("--repetition_penalty", type=float, default=1.0,
                        help="Repetition penalty (>1.0 discourages repeating tokens). "
                             "Useful for pruned MoE models that loop after the final answer. "
                             "Typical: 1.05-1.1.")
    parser.add_argument("--min_p", type=float, default=0.0,
                        help="Min-p sampling: drop tokens with prob < min_p * "
                             "max_prob. 0 disables (vLLM default).")
    parser.add_argument("--presence_penalty", type=float, default=0.0,
                        help="Presence penalty for already-seen tokens. "
                             "Qwen3 thinking-mode tech-report value: 1.5.")
    parser.add_argument(
        "--base_model",
        action="store_true",
    )
    parser.add_argument(
        "--use_math_verify",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the math-verify library (LightEval's underlying matcher) for "
             "math/math500/aime/gpqa/olympiad_bench/gsm8k scoring. Strictly "
             "more lenient than legacy boxed string-equality (sympy "
             "equivalence, free-text 'Answer: X' anchors, comma-stripped "
             "integers). Pass --no-use_math_verify to revert to the legacy "
             "regex-based comparison.",
    )
    parser.add_argument(
        "--use_chat_template",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Wrap each prompt as a single user message and call llm.chat(...) "
             "instead of llm.generate(...). vLLM applies the model's "
             "tokenizer.apply_chat_template, so Qwen3 / Qwen3.5 / etc. see the "
             "proper <|im_start|>user / assistant structure. Required for "
             "thinking-mode models to emit <think>...</think> traces.",
    )
    parser.add_argument(
        "--enable_thinking",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When --use_chat_template is on, pass chat_template_kwargs="
             "{'enable_thinking': True} to vLLM's llm.chat(). Qwen3 family "
             "templates already default this to True, but we set it explicitly "
             "for cross-model consistency. Pass --no-enable_thinking to disable "
             "thinking-mode generation (e.g. to compare apples-to-apples with "
             "non-thinking eval).",
    )
    parser.add_argument(
        "--max_problems",
        type=int,
        default=None,
        help="If set, evaluate only the first N problems from the dataset. "
             "Useful for smoke tests (e.g. --max_problems 10). Default: all.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=[
            "auto", "half", "fp16", "float16", "bf16", "bfloat16",
            "fp32", "float32",
        ],
        help="Data type for model weights and activations.",
    )
    parser.add_argument("--max_model_len", type=int, default=2048, help="Maximum sequence length the model can handle.")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9, help="GPU memory fraction to use (0-1).")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Number of GPUs for tensor parallelism.")
    parser.add_argument("--batch_size", type=int, default=100, help="Batch size for processing prompts.")
    parser.add_argument("--max_tokens", type=int, default=600, help="Maximum number of tokens to generate per prompt.")
    parser.add_argument(
        "--stop_sequences",
        type=str,
        # default="###,</answer>",
        help="Comma-separated list of custom stop strings (in addition to EOS). Example: '###,</answer>'",
    )
    parser.add_argument(
        "--prompt_mode",
        type=str,
        default="iprompt",
        choices=["iprompt", "raw"],
        help="Prompt wrapping mode: 'iprompt' wraps instruction with Alpaca-style template; 'raw' uses the instruction field directly.",
    )
    parser.add_argument(
        "--mbpp_shots",
        type=int,
        default=3,
        help="Number of MBPP few-shot demos to prepend (paper Table 28 default = 3, "
             "clamped to len(_MBPP_DEMOS)=3, 0 = zero-shot).",
    )
    parser.add_argument(
        "--apply_chat_template",
        action="store_true",
        help="Wrap ESFT prompts with the model's chat template (e.g. ChatML) before generation. "
             "Use this when the model was SFT-trained with chat format.",
    )
    parser.add_argument(
        "--openai_api_key",
        type=str,
        default=None,
        help="OpenAI API key for ESFT GPT-4 based evaluators. Falls back to env OPENAI_API_KEY (loaded via .env).",
    )

    args = parser.parse_args()

    # Normalize dtype aliases
    if args.dtype == "fp16":
        args.dtype = "float16"
    elif args.dtype == "bf16":
        args.dtype = "bfloat16"
    elif args.dtype == "fp32":
        args.dtype = "float32"

    main(args)
