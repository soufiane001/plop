# GSM8K 8-shot evaluation.
#
# Adapted from QwenLM/Qwen, eval/evaluate_chat_gsm8k.py. Unchanged from upstream:
# the `Question: ...\nLet's think step by step` prompt format, the last-number
# `extract_answer` regex, and `is_correct` (math.isclose, rel_tol=0, abs_tol=1e-4).
# Changed here: the few-shot examples live in src/gsm8k_prompt.txt with a
# --num-fewshot flag (upstream hardcodes four inline), --adapter merges a LoRA
# adapter into the base model before generating, and --use-vllm adds a vLLM
# generation path alongside the HuggingFace one.

import json
import os
import re
from pathlib import Path
import argparse
import math
import numpy as np
import tqdm
import torch
from datasets import load_from_disk, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteriaList, StoppingCriteria
from transformers.generation import GenerationConfig
from datetime import datetime

"""
python eval/evaluate_chat_gsm8k.py [--num-fewshot] [--use-vllm]
"""

INVALID_ANS = "[invalid]"
DEVICE = "cuda:0"


PROMPT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gsm8k_prompt.txt")


def load_examples():
    with open(PROMPT_FILE, "r") as f:
        return f.read()


EXAMPLES = load_examples()

class StopOnSubsequence(StoppingCriteria):
    def __init__(self, stop_ids):
        super().__init__()
        self.stop_ids = stop_ids

    def __call__(self, input_ids, scores, **kwargs):
        seq = input_ids[0]
        if seq.size(0) < len(self.stop_ids):
            return False
        # if the tail of seq equals stop_ids, stop
        if (seq[-len(self.stop_ids):] == 
            torch.tensor(self.stop_ids, device=seq.device)
           ).all():
            return True
        return False


def doc_to_text(doc, num_fewshot: int):
    if num_fewshot > 0:
        # Split on double newlines, drop any empty strings
        examples = [e.strip() for e in EXAMPLES.strip().split("\n\n") if e.strip()]
        # Take as many as requested (or all, if num_fewshot > len)
        chosen = examples[:num_fewshot]
        context = "\n\n".join(chosen)
        context += f"\n\nQuestion: {doc['question']}\nLet's think step by step"
    else:
        # Zero-shot format still follows the same structure
        context = f"Question: {doc['question']}\nLet's think step by step"
    return context


def generate_sample(
    model, tokenizer, question, max_new_tokens=512, temperature=0.0, verbose=True
):
    # Apply chat template if the tokenizer supports one
    template = getattr(tokenizer, "chat_template", None)

    if template is not None:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": question}],
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        # Fallback to raw prompt if no template is defined
        prompt = question

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=temperature,
    )

    output_text = tokenizer.decode(
        output_ids[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True
    )

    if verbose:
        print(prompt)
        print("-------------")
        print(output_text)
        print("=============")

    return output_text


def extract_answer(s):
    _PAT_LAST_DIGIT = re.compile(
        r"([+-])?(?=([0-9]|\.[0-9]))(0|([1-9](\d{0,2}(,\d{3})*)|\d*))?(\.\d*)?(?=\D|$)"
    )
    match = list(_PAT_LAST_DIGIT.finditer(s))
    if match:
        last_digit = match[-1].group().replace(",", "").replace("+", "").strip()
        # print(f"The last digit in {s} is {last_digit}")
    else:
        last_digit = None
        print(f"No digits found in {s!r}", flush=True)
    return last_digit


def is_correct(completion, answer):
    gold = extract_answer(answer)
    assert gold is not None, "No ground truth answer found in the document."

    def number_equal(answer, pred):
        if pred is None:
            return False
        try:
            return math.isclose(eval(answer), eval(pred), rel_tol=0, abs_tol=1e-4)
        except:
            print(
                f"cannot compare two numbers: answer={answer}, pred={pred}", flush=True
            )
            return False

    return number_equal(gold, extract_answer(completion))


def add_timestamp_to_filename(filename):
    """Add timestamp to filename before the extension."""
    path = Path(filename)
    timestamp = datetime.now().strftime('%d%b%Y_%H-%M-%S')  # e.g., 21Mar2024_14-30-22
    return str(path.parent / f"{path.stem}_{timestamp}{path.suffix}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test HF checkpoint.")
    parser.add_argument(
        "-c",
        "--checkpoint-path",
        type=Path,
        help="Model to evaluate; a plain HuggingFace model or a merged checkpoint",
        default="Qwen/Qwen3-1.7B",
    )
    parser.add_argument(
        "--adapter",
        type=str,
        default=None,
        help="LoRA adapter directory from sft_metamath.py. The adapter is merged "
             "into its base model before generating; overrides --checkpoint-path.",
    )
    parser.add_argument("-f", "--sample-input-file", type=str, default=None)
    parser.add_argument(
        "-o", "--sample-output-file", type=str, default="gsm8k_res.jsonl"
    )
    parser.add_argument(
        "--result-file", type=str, default="results.json",
        help="JSON file to write the evaluation results to"
    )
    parser.add_argument(
        "--num-fewshot",
        type=int,
        default=4,
        help="How many few-shot examples to prepend; 0 for zero-shot",
    )
    parser.add_argument(
        "--use-vllm",
        action="store_true",
        help="Run inference via in-process vLLM engine",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum number of tokens to generate",
    )

    args = parser.parse_args()

    if args.adapter is not None:
        import tempfile

        from peft import PeftConfig, PeftModel

        peft_config = PeftConfig.from_pretrained(args.adapter)
        print(f"Merging adapter {args.adapter} into {peft_config.base_model_name_or_path} ...")
        base = AutoModelForCausalLM.from_pretrained(
            peft_config.base_model_name_or_path, torch_dtype=torch.bfloat16
        )
        merged = PeftModel.from_pretrained(base, args.adapter).merge_and_unload()
        merged_dir = tempfile.mkdtemp(prefix="merged_")
        merged.save_pretrained(merged_dir)
        AutoTokenizer.from_pretrained(args.adapter).save_pretrained(merged_dir)
        del base, merged
        args.checkpoint_path = Path(merged_dir)
        print(f"Merged model written to {merged_dir}")

    # Add timestamps to output filenames
    args.sample_output_file = add_timestamp_to_filename(args.sample_output_file)
    args.result_file = add_timestamp_to_filename(args.result_file)

    if args.sample_input_file is not None:
        dataset = load_from_disk(args.sample_input_file)  # or:
    else:
        dataset = load_dataset("gsm8k", "main")

    print("Loading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint_path, trust_remote_code=True, bf16=True, use_flash_attn=True
    )

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Shared sampling settings
    temperature = 0.0
    batch_size = 8

    print("Loading model ...")

    stop_seqs = ["\nQuestion:", "\n\nQuestion:", "Question:"]

    # Convert each into token-ID sequences
    stop_ids_list = [
        tokenizer.encode(seq, add_special_tokens=False)
        for seq in stop_seqs
    ]

    stopping_criteria = StoppingCriteriaList([
        StopOnSubsequence(ids) for ids in stop_ids_list
    ])

    if args.use_vllm:
        from vllm import LLM, SamplingParams

        llm_engine = LLM(
            model=str(args.checkpoint_path),
            dtype="auto",
            gpu_memory_utilization=0.8,
        )
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=args.max_tokens,
            stop=stop_seqs,
        )
        model = None
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.checkpoint_path,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        ).eval()
        model.generation_config = GenerationConfig.from_pretrained(
            args.checkpoint_path, trust_remote_code=True
        )
        model.generation_config.do_sample = False  # use greedy decoding
        model.generation_config.repetition_penalty = 1.0  # disable repetition penalty
        model.generation_config.top_k = None
        model.generation_config.top_p = None
        model.generation_config.temperature = None

        llm_engine = None
        sampling_params = None

    test = list(dataset["test"])
    contexts = [doc_to_text(doc, args.num_fewshot) for doc in test]
    # right after you build `contexts`:
    if getattr(tokenizer, "chat_template", None):
        contexts = [
            tokenizer.apply_chat_template(
                [{"role":"user", "content":ctx}],
                tokenize=False,
                add_generation_prompt=True
            )
            for ctx in contexts
        ]


    answers = [doc["answer"] for doc in test]

    if args.use_vllm:
        outputs = llm_engine.generate(contexts, sampling_params)
        completions = [o.outputs[0].text for o in outputs]
    else:
        completions = []
        for i in tqdm.tqdm(range(0, len(contexts), batch_size), desc="HF batches"):
            batch_ctx = contexts[i : i + batch_size]
            with torch.inference_mode():
                toks = tokenizer(
                    batch_ctx,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                ).to(model.device)
                outs = model.generate(
                    **toks,
                    max_new_tokens=args.max_tokens,
                    do_sample=False,
                    temperature=temperature,
                    stopping_criteria=stopping_criteria,
                    eos_token_id=tokenizer.eos_token_id,
                )
            # how many tokens were in each prompt?
            input_lens = toks["attention_mask"].sum(dim=1)
            for seq, length in zip(outs, input_lens):
                # drop exactly `length` tokens of prompt, decode the rest
                gen_ids = seq[length:]
                completions.append(tokenizer.decode(gen_ids, skip_special_tokens=True))

    acc_res = []
    with open(args.sample_output_file, "w", encoding="utf-8") as fout:
        for doc, pred, ans in zip(test, completions, answers):
            ok = is_correct(pred, ans)
            doc["completion"] = pred
            doc["acc"] = ok
            fout.write(json.dumps(doc, ensure_ascii=False) + "\n")
            acc_res.append(ok)

    accuracy = np.mean(acc_res)
    print(
        f"{args.num_fewshot}-shot Acc: " if args.num_fewshot > 0 else "Zero-shot Acc",
        accuracy,
    )

    # Write results to a JSON file
    results = {
        "accuracy": float(accuracy),
        "num_fewshot": args.num_fewshot,
        "model": str(args.checkpoint_path),
        "max_tokens": args.max_tokens,
        "temperature": temperature,
    }
    with open(args.result_file, "w") as f:
        json.dump(results, f, indent=2)
