# Evaluate Qwen1.5-MoE (or similar) with vLLM using local Qwen2-MoE modeling.
#
# Strict zero-shot variant: GSM / MBPP / HumanEval / MMLU / CEval / CMMLU /
# BBH / MATH / MultiArith all use raw prompts or the standard 0-shot CoT
# template (no few-shot demos). Use this module for Qwen MoE evaluations;
# use vllm_olmoe_multishot for OLMoE / few-shot CoT runs.
import argparse
import json
import os
import re
import sys
import time
from typing import List, Dict, Any

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

    completion = completion.replace("[DONE]", "").strip()

    # MBPP-style prompts use "[BEGIN]\n" as code-start marker; models reply with
    # the function body and an "END" sentinel, then sometimes drift into natural
    # language. Trim at the END token plus any trailing prose so the executor
    # doesn't choke on SyntaxError before reaching the assertions.
    end_match = re.search(r"\nEND\b", completion)
    if end_match:
        completion = completion[:end_match.start()]
    code_kw = (
        "def ", "import ", "from ", "class ", "@", "if ", "for ", "while ",
        "return ", "assert ", "print(", "try", "except", "raise ", "with ",
        "elif ", "else", "#", "yield ", "global ", "nonlocal ",
        "pass", "break", "continue", "lambda ", "async ", "await ",
    )
    out_lines = []
    for line in completion.split("\n"):
        s = line.lstrip()
        if (not line) or line.startswith((" ", "\t")) or s.startswith(code_kw) \
                or re.match(r"^[A-Za-z_]\w*\s*[=(]", s):
            out_lines.append(line)
        else:
            break
    return "\n".join(out_lines).rstrip()


def preprocess_humaneval_completion(completion: str, language: str = "python") -> str:
    """Extract code completion for HumanEval, preserving leading indentation."""
    if completion is None:
        return ""
    completion = completion.replace("\r", "")

    # Strip thinking trace. vLLM's chat_template_kwargs={'enable_thinking': True}
    # prefills <think> at the prompt boundary, so the model often emits only
    # </think> (no opening <think>) before the final answer. Anchor on </think>
    # alone so we always grab the post-thinking section.
    if "</think>" in completion:
        match = re.search(r"</think>\s*(.*)", completion, re.DOTALL)
        completion = match.group(1) if match else ""
    elif "<think>" in completion:
        # Open but not closed → output truncated mid-thinking, nothing usable.
        return ""
    if not completion:
        return ""

    # Pick the LAST fenced block — when the model writes intermediate code
    # examples and a final solution, the last block is the answer.
    fence = f"```{language}"
    if fence in completion:
        start = completion.rindex(fence) + len(fence)
        completion = completion[start:]
        if completion.startswith("\n"):
            completion = completion[1:]
        if "```" in completion:
            completion = completion[:completion.index("```")]
    elif "```" in completion:
        # Fallback to last generic fence.
        start = completion.rindex("```") + 3
        # If this ``` happened to be the closing of the previous block, search
        # backwards for the opening fence instead.
        before = completion[:completion.rindex("```")]
        if "```" in before:
            open_idx = before.rindex("```")
            completion = completion[open_idx + 3 : completion.rindex("```")]
        else:
            completion = completion[start:]
        if completion.startswith("\n"):
            completion = completion[1:]
        if "```" in completion:
            completion = completion[:completion.index("```")]

    # Truncate at top-level statements that look like test scaffolding or
    # extraneous extra functions leaking past the answer. HumanEval prompts end
    # inside the docstring (the function signature is in the prompt), so the
    # model's first output line is at indent>=4. Any column-0 def/class is
    # therefore an UNRELATED second function, often truncated mid-stream by
    # max_tokens — keeping it produces SyntaxError and fails the test.
    truncated_lines = []
    for line in completion.split("\n"):
        stripped = line.lstrip()
        if line and not line.startswith((" ", "\t")) and stripped.startswith(
            ("if __name__", "print(", "assert ", "#test", "# test",
             "def ", "class ")
        ):
            break
        truncated_lines.append(line)
    completion = "\n".join(truncated_lines)

    completion = completion.replace("[DONE]", "")
    return completion.rstrip()


def build_humaneval_check(prompt: str, completion: str, test: str, entry_point: str) -> str:
    return prompt + completion + "\n" + test + f"\ncheck({entry_point})\n"


def extract_mbpp_tests(prompt: str) -> List[str]:
    match = re.search(r"\[(['\"]assert .+?)\]", prompt, re.DOTALL)
    if not match:
        return []
    raw = match.group(0)
    try:
        tests = eval(raw)
        if isinstance(tests, list):
            return tests
    except Exception:
        pass
    return re.findall(r"assert .+", raw)


def run_code_with_tests_safe(code: str, tests: List[str], timeout: float = 5.0) -> bool:
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


def _truncate_at_followup_question(text: str) -> str:
    """Cut model continuation at the first sign of a new Q&A turn.

    Zero-shot prompts like "Question: ... Answer:" don't constrain the model
    from echoing more questions after answering the first one. With T=0 and
    max_tokens=500 the model often writes 2-3 follow-up questions, and
    re.findall(r"-?\\d+\\.?\\d*", ...)[-1] then grabs an unrelated number.
    We anchor on common follow-up markers and keep only the first segment.
    """
    patterns = [
        r"\n\[Question\]", r"\n\s*Question:",
        r"\n\[Q:\]", r"\n\[Q\]", r"\n\s*Q:",
        r"\n\[Problem\]", r"\n\s*Problem:",
    ]
    earliest = len(text)
    for p in patterns:
        m = re.search(p, text)
        if m and m.start() < earliest:
            earliest = m.start()
    return text[:earliest]


def extract_answer_number(args, sentence: str) -> float:
    dataset = args.dataset.lower()
    if dataset in ["multiarith", "addsub", "singleeq", "gsm8k", "gsm", "svamp", "mawps"]:
        # Truncate at first follow-up question marker (zero-shot continuation).
        sentence = _truncate_at_followup_question(sentence)
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


_CEVAL_SUBJECT_ZH = {}


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


# Prompt template adapted from simple-evals / open-r1 evaluate.py
MATH_QUERY_TEMPLATE = (
    "Solve the following math problem efficiently and clearly.  The last line of your response "
    "should be of the following format: 'Therefore, the final answer is: $\\boxed{{ANSWER}}$. "
    "I hope it is correct' (without quotes) where ANSWER is just the final number or expression "
    "that solves the problem. Think step by step before answering.\n\n{Question}"
)


def _build_math_prompt(problem: str) -> str:
    """Zero-shot CoT prompt for Hendrycks MATH."""
    return MATH_QUERY_TEMPLATE.format(Question=problem)


def _build_aime_prompt(problem: str) -> str:
    return MATH_QUERY_TEMPLATE.format(Question=problem)


def _build_math500_prompt(problem: str) -> str:
    return MATH_QUERY_TEMPLATE.format(Question=problem)


def _build_gpqa_prompt(problem: str) -> str:
    return problem


def _build_olympiad_prompt(problem: str) -> str:
    return MATH_QUERY_TEMPLATE.format(Question=problem)


def _olympiad_answer_match(pred_boxed: str, gold_answers) -> bool:
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
    if not s:
        return ""
    idx = s.rfind("\\boxed{")
    if idx == -1:
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
    return "".join(out)


def _extract_math_pred(s: str) -> str:
    """Extract a math answer from a generation. Tries (in order):
    1. The last \\boxed{} or \\fbox{} (preferred — prompt explicitly asks for it)
    2. "answer is/=/: X" patterns in the final sentence
    3. The last $...$ inline-math segment
    4. The last numeric / fractional token
    """
    if not s:
        return ""
    boxed = _extract_last_boxed(s)
    if boxed:
        return boxed

    tail = s.strip().split("\n")[-1] if s.strip() else ""
    pat = re.search(
        r"(?:final\s+)?answer(?:\s+is)?\s*[:=]?\s*\$?([^\n$.]+?)\$?\s*\.?\s*$",
        tail, re.IGNORECASE,
    )
    if pat:
        cand = pat.group(1).strip().rstrip(".")
        if cand:
            return cand

    dollar = re.findall(r"\$([^$]+)\$", s)
    if dollar:
        return dollar[-1].strip()

    nums = re.findall(
        r"-?\d+(?:\\?[/.]\d+)?(?:\^\{?-?\d+\}?)?|\\frac\s*\{[^}]+\}\s*\{[^}]+\}",
        s,
    )
    if nums:
        return nums[-1].strip()
    return ""


def _normalize_math_answer(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip()
    while s.startswith("$") and s.endswith("$") and len(s) >= 2:
        s = s[1:-1].strip()
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\!", "").replace("\\,", "").replace("\\;", "")
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    s = re.sub(r"\\text\s*\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\mathrm\s*\{([^}]*)\}", r"\1", s)
    s = s.replace(" ", "")
    if s.endswith("."):
        s = s[:-1]
    while s.startswith("{") and s.endswith("}"):
        s = s[1:-1]
    return s


def _build_bbh_prompt(question: str) -> str:
    return f"{question}\nAnswer:"


def _normalize_bbh_answer(s: str) -> str:
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
    while t and t[-1] in ".。,，!?":
        t = t[:-1].strip()
    m = re.match(r"\(\s*([a-z0-9]+)\s*\)", t)
    if m:
        t = m.group(1)
    elif len(t) >= 2 and t[0] == "(" and t[-1] == ")":
        t = t[1:-1].strip()
    return t


def extract_chinese_mcq_letter(sentence: str) -> str:
    if not sentence:
        return ""
    s = sentence.strip()
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
    supported_families = {"qwen2_moe", "qwen3_moe", "qwen3_5_moe"}
    if family not in supported_families:
        raise ValueError(
            "The zero-shot entry point is for Qwen1.5/Qwen2-MoE, Qwen3-MoE, "
            "and Qwen3.5-MoE. Use vllm_olmoe_multishot for OLMoE; detected "
            f"{family!r}."
        )
    if not args.base_model:
        if not apply_vllm_patch(family):
            raise RuntimeError(
                f"The installed vLLM does not provide the {family} model module "
                "required by this checkpoint."
            )
        print(f"[Patch] enabled Less-is-MoE vLLM patch for {family}.")
    else:
        print("[Patch] base_model=True，跳过本地 MoE patch，直接使用模型自带实现。")

    dataset_lower = args.dataset.lower()

    raw_data = load_data(args.data_path)

    if is_esft_dataset(dataset_lower):
        t_test_data = raw_data
        raw_prompts: List[str] = [
            ex.get("prompt")
            or ex.get("instruction")
            or ex.get("input")
            or ""
            for ex in t_test_data
        ]
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
        # Zero-shot: raw prompt field
        t_test_data = raw_data
        prompts = [ex["prompt"] for ex in t_test_data]
        print(f"[Prompt] GSM dataset detected (zero-shot), {len(t_test_data)} examples")
    elif dataset_lower == "mbpp":
        # Zero-shot: raw prompt field (tests embedded)
        t_test_data = raw_data
        prompts = [ex["prompt"] for ex in t_test_data]
        print(f"[Prompt] MBPP dataset detected (zero-shot), {len(t_test_data)} examples")
    elif dataset_lower == "humaneval":
        t_test_data = raw_data
        prompts = [ex["prompt"] for ex in t_test_data]
        print(f"[Prompt] HumanEval dataset detected, {len(t_test_data)} examples")
    elif dataset_lower == "ceval":
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
        t_test_data = raw_data
        prompts = [_build_math_prompt(ex.get("problem", "")) for ex in t_test_data]
        print(f"[Prompt] MATH dataset detected (zero-shot CoT), {len(t_test_data)} examples")
    elif dataset_lower in ("aime24", "aime2024", "aime25", "aime2025", "aime26", "aime2026"):
        t_test_data = raw_data
        prompts = [_build_aime_prompt(ex.get("problem", "")) for ex in t_test_data]
        print(f"[Prompt] AIME dataset detected, {len(t_test_data)} examples")
    elif dataset_lower == "math500":
        t_test_data = raw_data
        prompts = [_build_math500_prompt(ex.get("problem", "")) for ex in t_test_data]
        print(f"[Prompt] MATH-500 dataset detected, {len(t_test_data)} examples")
    elif dataset_lower in ("gpqa", "gpqa_diamond"):
        t_test_data = raw_data
        prompts = [_build_gpqa_prompt(ex.get("problem", "")) for ex in t_test_data]
        print(f"[Prompt] GPQA Diamond dataset detected, {len(t_test_data)} examples")
    elif dataset_lower in ("olympiad_bench", "olympiadbench"):
        t_test_data = raw_data
        prompts = [_build_olympiad_prompt(ex.get("problem", "")) for ex in t_test_data]
        print(f"[Prompt] OlympiadBench dataset detected, {len(t_test_data)} examples")
    elif dataset_lower == "bbh":
        t_test_data = raw_data
        prompts = [_build_bbh_prompt(ex.get("input", "")) for ex in t_test_data]
        print(f"[Prompt] BBH dataset detected, {len(t_test_data)} examples")
    elif dataset_lower == "mmlu":
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
    elif dataset_lower in ("multiarith", "multi_arith"):
        # Zero-shot: raw instruction field
        t_test_data = raw_data
        prompts = [ex.get("instruction", "") for ex in t_test_data]
        print(f"[Prompt] MultiArith dataset detected (zero-shot), {len(t_test_data)} examples")
    else:
        t_test_data = [normalize_example(e) for e in raw_data]
        if args.prompt_mode == "raw":
            print("[Prompt] raw mode — 直接使用 instruction 字段，不包裹模板。")
            prompts = [example["instruction"] for example in t_test_data]
        else:
            print("[Prompt] iprompt mode — 使用 Alpaca 风格模板包裹。")
            prompts = [i_prompt.format_map(example) for example in t_test_data]
    print(f"First prompt example:\n{prompts[0]}")
    print(f"Total prompts: {len(prompts)}")

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
    if args.quantization:
        llm_kwargs["quantization"] = args.quantization

    if family == "qwen3_5_moe" and not os.environ.get(
        "VLLM_SKIP_QWEN35_ARCH_OVERRIDE"
    ):
        llm_kwargs["hf_overrides"] = {
            "architectures": ["Qwen3_5MoeForCausalLM"],
        }
        print(
            "[Patch] Qwen3.5-MoE detected — overriding config.architectures "
            "to ['Qwen3_5MoeForCausalLM'] to bypass vLLM's multimodal "
            "preprocessor load."
        )

    try:
        llm = LLM(**llm_kwargs)
        print("✓ vLLM model loaded successfully")
    except RuntimeError as e:
        if "CUDA" in str(e):
            print(f"CUDA error detected, retrying with CUDA_LAUNCH_BLOCKING=1 ... (error: {e})")
            try:
                del llm
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

    stop_seqs = None
    if args.stop_sequences:
        stop_seqs = [s for s in (seg.strip() for seg in args.stop_sequences.split(",")) if s]

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

    batch_size = min(args.batch_size, len(prompts))
    all_outputs = []
    if getattr(args, "use_chat_template", False):
        chat_kwargs = {"enable_thinking": getattr(args, "enable_thinking", True)}
        print(f"Starting generation via llm.chat() with chat_template_kwargs={chat_kwargs} ...")
    else:
        chat_kwargs = None
        print("Starting generation via llm.generate() (raw prompts, no chat template) ...")
    start_time = time.time()

    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i : i + batch_size]
        print(f"Processing batch {i // batch_size + 1}/{(len(prompts) + batch_size - 1) // batch_size}")
        if getattr(args, "use_chat_template", False):
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
    print(f"Generation completed in {elapsed_time:.2f} seconds")
    print(f"Speed: {len(prompts) / elapsed_time:.2f} prompts/second")

    save_outputs = []
    correct = 0
    miss = 0.001

    for example, output in tqdm(
        zip(t_test_data, all_outputs),
        total=len(t_test_data),
        desc="Evaluating",
        leave=True,
        dynamic_ncols=True,
        file=sys.stdout,
    ):
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

            if stop_seqs:
                for s in stop_seqs:
                    if s and s in generated_text:
                        generated_text = generated_text.split(s)[0]
            if dataset_lower == "esft-translation" or dataset_lower == "esft-intent" or dataset_lower == "esft-law" or dataset_lower == "esft-summary":
                if "\n" in generated_text:
                    generated_text = generated_text.split("\n", 1)[0]
            generated_text = generated_text.rstrip()

            example["raw_output"] = generated_text
            prev_correct = correct

            if dataset_lower == "mbpp":
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
                predict = _normalize_bbh_answer(generated_text)
                target_norm = _normalize_bbh_answer(example.get("target", ""))
                if predict and target_norm and predict == target_norm:
                    correct += 1
                example["target_norm"] = target_norm
            elif dataset_lower == "math":
                pred_boxed = _extract_math_pred(generated_text)
                gold_boxed = _extract_last_boxed(example.get("solution", ""))
                predict = pred_boxed
                pred_norm = _normalize_math_answer(pred_boxed)
                gold_norm = _normalize_math_answer(gold_boxed)
                if pred_norm and gold_norm and pred_norm == gold_norm:
                    correct += 1
                example["pred_boxed"] = pred_boxed
                example["gold_boxed"] = gold_boxed
            elif dataset_lower in ("aime24", "aime2024", "aime25", "aime2025", "aime26", "aime2026"):
                pred_boxed = _extract_last_boxed(generated_text)
                gold_answer = str(example.get("answer", "")).strip()
                predict = pred_boxed
                pred_norm = _normalize_math_answer(pred_boxed)
                gold_norm = _normalize_math_answer(gold_answer)
                if pred_norm and gold_norm and pred_norm == gold_norm:
                    correct += 1
                example["pred_boxed"] = pred_boxed
                example["gold_answer"] = gold_answer
            elif dataset_lower == "math500":
                pred_boxed = _extract_last_boxed(generated_text)
                gold_answer = str(example.get("answer", "")).strip()
                predict = pred_boxed
                pred_norm = _normalize_math_answer(pred_boxed)
                gold_norm = _normalize_math_answer(gold_answer)
                if pred_norm and gold_norm and pred_norm == gold_norm:
                    correct += 1
                example["pred_boxed"] = pred_boxed
                example["gold_answer"] = gold_answer
            elif dataset_lower in ("gpqa", "gpqa_diamond"):
                pred_boxed = _extract_last_boxed(generated_text)
                gold_boxed = _extract_last_boxed(example.get("solution", ""))
                predict = pred_boxed.strip().upper() if pred_boxed else ""
                gold_letter = gold_boxed.strip().upper() if gold_boxed else ""
                if predict and gold_letter and predict == gold_letter:
                    correct += 1
                example["pred_boxed"] = pred_boxed
                example["gold_letter"] = gold_letter
            elif dataset_lower in ("olympiad_bench", "olympiadbench"):
                pred_boxed = _extract_last_boxed(generated_text)
                gold_list = example.get("answer_list") or [example.get("answer", "")]
                predict = pred_boxed
                if _olympiad_answer_match(pred_boxed, gold_list):
                    correct += 1
                example["pred_boxed"] = pred_boxed
                example["gold_answers"] = gold_list
            elif dataset_lower == "mmlu":
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
                is_choice = dataset_lower in ["aqua", "mathqa"] or str(target).strip().upper() in ["A", "B", "C", "D", "E"]

                if is_choice:
                    predict = extract_answer_letter(args, generated_text)
                    target_letter = extract_answer_letter(args, str(target))
                    if target_letter and predict and target_letter.upper() == predict.upper():
                        correct += 1
                else:
                    predict = extract_answer_number(args, generated_text)
                    try:
                        target_val = float(str(target).replace(",", ""))
                    except ValueError:
                        target_val = float("inf")
                    if abs(target_val - predict) <= miss:
                        correct += 1

            sample_records.append({
                "raw_output": generated_text,
                "prediction": predict,
                "correct": int(correct > prev_correct),
            })

        example["prediction"] = predict
        example["correct"] = int(sample_records[-1]["correct"]) if sample_records else 0
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
        }
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"Metrics saved to {metrics_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--n_samples_per_problem", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--min_p", type=float, default=0.0,
                        help="Min-p sampling: drop tokens with prob < min_p * "
                             "max_prob. 0 disables (vLLM default).")
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--presence_penalty", type=float, default=0.0,
                        help="Presence penalty for already-seen tokens. "
                             "Qwen3 non-thinking-mode tech-report value: 2.0.")
    parser.add_argument("--base_model", action="store_true")
    parser.add_argument(
        "--dtype", type=str, default="auto",
        choices=[
            "auto", "half", "fp16", "float16", "bf16", "bfloat16",
            "fp32", "float32",
        ],
    )
    parser.add_argument("--max_model_len", type=int, default=2048)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument(
        "--quantization",
        type=str,
        default=None,
        help="Optional vLLM quantization backend, for example awq or awq_marlin.",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--stop_sequences", type=str, default=None)
    parser.add_argument(
        "--prompt_mode", type=str, default="iprompt",
        choices=["iprompt", "raw"],
    )
    parser.add_argument(
        "--apply_chat_template", action="store_true",
    )
    parser.add_argument(
        "--use_chat_template",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Wrap each prompt as a single user message and call llm.chat(...) "
             "instead of llm.generate(...). vLLM applies the model's "
             "tokenizer.apply_chat_template, so Qwen3 / Qwen3.5 see the "
             "proper <|im_start|>user / assistant structure. Required for "
             "thinking-mode models to emit <think>...</think> traces.",
    )
    parser.add_argument(
        "--enable_thinking",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When --use_chat_template is on, pass chat_template_kwargs="
             "{'enable_thinking': True} to vLLM's llm.chat(). Pass "
             "--no-enable_thinking to disable thinking-mode generation.",
    )
    parser.add_argument("--openai_api_key", type=str, default=None)

    args = parser.parse_args()

    if args.dtype == "fp16":
        args.dtype = "float16"
    elif args.dtype == "bf16":
        args.dtype = "bfloat16"
    elif args.dtype == "fp32":
        args.dtype = "float32"

    main(args)
