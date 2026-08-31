# !/usr/bin/env python
# coding=utf-8
# Copyright 2024 AllenAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
import logging
import math
import re
import numpy as np
import os
import shutil
import time
from dataclasses import dataclass, field
from datetime import timedelta
from typing import List, Literal, Optional, Union, Dict
from collections import defaultdict
import datasets
import deepspeed
import torch
import transformers
from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.logging import get_logger
from accelerate.utils import InitProcessGroupKwargs, set_seed
from huggingface_hub import HfApi
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from rich.pretty import pprint
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    get_scheduler,
)
from transformers.tokenization_utils_base import (
    PreTrainedTokenizerBase,
    PaddingStrategy,
)
from dataset_transformation import (
    INPUT_IDS_KEY,
    TOKENIZED_SFT_DATASET_KEYS,
    TokenizerConfig,
    get_cached_dataset_tulu,
    visualize_token,
)
from model_utils import push_folder_to_hub, save_with_accelerate
from utils import (
    ArgumentParserPlus,
    clean_last_n_checkpoints,
    get_last_checkpoint_path,
    get_wandb_tags,
    is_beaker_job,
    launch_ai2_evals_on_weka,
    maybe_get_beaker_config,
    maybe_use_ai2_hf_entity,
    maybe_use_ai2_wandb_entity,
)


logger = get_logger(__name__)

@dataclass
class FlatArguments:
    """
    Full arguments class for all fine-tuning jobs.
    """
    alignment_metrics_sample_size: int = field(
        default=100,
        metadata={"help": "Number of samples to use for alignment metrics calculation"},
    )
    alignment_metrics_max_length: int = field(
        default=256,
        metadata={"help": "Maximum sequence length for alignment metrics calculation"},
    )
    exp_name: str = field(
        default=os.path.basename(__file__)[: -len(".py")],
        metadata={"help": "The name of this experiment"},
    )

    tags: str = field(
        default="",
        metadata={"help": "The tag name of this experiment"},
    )
    
    run_name: Optional[str] = field(
        default=None,
        metadata={"help": "A unique name for this run"},
    )
    
    """A unique name of this run"""
    model_name_or_path: Optional[str] = field(
        default="meta-llama/Llama-3.2-1B",
        metadata={
            "help": (
                "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
            )
        },
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained config name or path if not the same as model_name"},
    )
    use_flash_attn: bool = field(
        default=False,
        metadata={"help": "Whether to use flash attention in the model training"},
    )
    model_revision: Optional[str] = field(
        default=None,
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    low_cpu_mem_usage: bool = field(
        default=False,
        metadata={
            "help": (
                "It is an option to create the model as an empty shell, "
                "then only materialize its parameters when the pretrained weights are loaded. "
                "set True will benefit LLM loading time and RAM consumption."
            )
        },
    )
    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the dataset to use (via the datasets library)."},
    )
    dataset_mixer: Optional[dict] = field(
        default=None,
        metadata={"help": "A dictionary of datasets (local or HF) to sample from."},
    )
    dataset_mixer_list: List[str] = field(default_factory=lambda: ["allenai/tulu-3-sft-personas-algebra", "1.0"])
    """A list of datasets (local or HF) to sample from."""
    dataset_mixer_list_splits: List[str] = field(default_factory=lambda: ["train"])
    """The dataset splits to use for training"""
    dataset_transform_fn: list[str] = field(
        default_factory=lambda: ["sft_tulu_tokenize_and_truncate_v1", "sft_tulu_filter_v1"]
    )
    """The list of transform functions to apply to the dataset."""
    dataset_target_columns: List[str] = field(default_factory=lambda: TOKENIZED_SFT_DATASET_KEYS)
    """The columns to use for the dataset."""
    dataset_cache_mode: Literal["hf", "local"] = "local"
    """The mode to use for caching the dataset."""
    dataset_local_cache_dir: str = "local_dataset_cache"
    """The directory to save the local dataset cache to."""
    dataset_config_hash: Optional[str] = None
    """The hash of the dataset configuration."""
    dataset_skip_cache: bool = False
    """Whether to skip the cache."""
    dataset_mix_dir: Optional[str] = field(
        default=None,
        metadata={"help": "The directory to save the mixed dataset to disk."},
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={"help": "The configuration name of the dataset to use (via the datasets library)."},
    )
    train_file: Optional[str] = field(
        default=None,
        metadata={"help": "The input training data file (a json/jsonl file)."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    dataloader_num_workers: int = field(
        default=4,
        metadata={"help": "Number of subprocesses for data loading"},
    )
    max_seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. "
                "Sequences longer than this will be truncated,"
            )
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    clip_grad_norm: float = field(
        default=-1,
        metadata={"help": "Clip gradient norm. Not compatible with deepspeed (use deepspeed config instead)."},
    )
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass."},
    )
    learning_rate: float = field(
        default=2e-5,
        metadata={"help": "The initial learning rate for AdamW optimizer."},
    )
    logging_steps: Optional[int] = field(
        default=None,
        metadata={"help": "Log the training loss and learning rate every logging_steps steps."},
    )
    lora_select_strategy: str = field(
        default="increasing",
        metadata={
            "help": (
                "The strategy to select layers for LoRA. "
                "Options are 'increasing', 'decreasing', 'first', 'last', or 'every'."
            ),
        },
    )
    lora_module_frac: float = field(
        default=1.0,
        metadata={
            "help": (
                "The fraction of layers to use for LoRA. "
                "If 1.0, all layers are used. If 0.5, half of the layers are used."
            )
        },
    )
    lora_rank: int = field(
        default=64,
        metadata={"help": "The rank of lora."},
    )
    lora_alpha: float = field(
        default=16,
        metadata={"help": "The alpha parameter of lora."},
    )
    lora_dropout: float = field(
        default=0.1,
        metadata={"help": "The dropout rate of lora modules."},
    )
    lr_scheduler_type: str = field(
        default="linear",
        metadata={
            "help": "The scheduler type to use for learning rate adjustment.",
            "choices": [
                "linear",
                "cosine",
                "cosine_with_restarts",
                "polynomial",
                "constant",
                "constant_with_warmup",
            ],
        },
    )
    num_train_epochs: int = field(
        default=2,
        metadata={"help": "Total number of training epochs to perform."},
    )
    output_dir: str = field(
        default="output/",
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )
    per_device_train_batch_size: int = field(
        default=8,
        metadata={"help": "Batch size per GPU/TPU core/CPU for training."},
    )
    use_lora: bool = field(
        default=False,
        metadata={"help": "If True, will use LORA (low-rank parameter-efficient training) to train the model."},
    )
    use_qlora: bool = field(
        default=False,
        metadata={"help": "Use qLoRA training - initializes model in quantized form. Not compatible with deepspeed."},
    )
    use_8bit_optimizer: bool = field(
        default=False,
        metadata={"help": "Use 8bit optimizer from bitsandbytes. Not compatible with deepspeed."},
    )
    warmup_ratio: float = field(
        default=0.03,
        metadata={"help": "Linear warmup over warmup_ratio fraction of total steps."},
    )
    weight_decay: float = field(
        default=0.0,
        metadata={"help": "Weight decay for AdamW if we apply some."},
    )
    timeout: int = field(
        default=1800,
        metadata={
            "help": "Timeout for the training process in seconds."
            "Useful if tokenization process is long. Default is 1800 seconds (30 minutes)."
        },
    )
    reduce_loss: str = field(
        default="mean",
        metadata={
            "help": "How to reduce loss over tokens. Options are 'mean' or 'sum'."
            "Using 'sum' can improve chat model performance."
        },
    )
    resume_from_checkpoint: Optional[str] = field(
        default=None,
        metadata={"help": "If the training should continue from a checkpoint folder."},
    )
    report_to: Union[str, List[str]] = field(
        default="all",
        metadata={
            "help": "The integration(s) to report results and logs to. "
            "Can be a single string or a list of strings. "
            "Options are 'tensorboard', 'wandb', 'comet_ml', 'clearml', or 'all'. "
            "Specify multiple by listing them: e.g., ['tensorboard', 'wandb']"
        },
    )
    save_to_hub: Optional[str] = field(
        default=None,
        metadata={"help": "Save the model to the Hub under this name. E.g allenai/your-model"},
    )
    gradient_checkpointing: bool = field(
        default=False,
        metadata={"help": "Turn on gradient checkpointing. Saves memory but slows training."},
    )
    use_liger_kernel: bool = field(
        default=False,
        metadata={"help": "Whether to use LigerKernel for training."},
    )
    max_train_steps: Optional[int] = field(
        default=None,
        metadata={"help": "If set, overrides the number of training steps. Otherwise, num_train_epochs is used."},
    )
    seed: int = field(
        default=42,
        metadata={"help": "Random seed for initialization and dataset shuffling."},
    )
    checkpointing_steps: Optional[str] = field(
        default=None,
        metadata={
            "help": "Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch."  # noqa
        },
    )
    keep_last_n_checkpoints: int = field(
        default=3,
        metadata={"help": "How many checkpoints to keep in the output directory. -1 for all."},
    )
    fused_optimizer: bool = field(
        default=True,
        metadata={
            "help": "Whether to use fused AdamW or not.",
        },
    )
    load_balancing_loss: bool = field(
        default=False,
        metadata={
            "help": "Whether to include a load balancing loss (for OLMoE) or not.",
        },
    )
    load_balancing_weight: float = field(
        default=0.5,
        metadata={"help": "Weight for load balancing loss if applicable."},
    )

    # Experiment tracking
    with_tracking: bool = False
    """If toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "open_instruct_internal"
    """The wandb's project name"""
    wandb_entity: Optional[str] = None
    """The entity (team) of wandb's project"""
    push_to_hub: bool = False
    """Whether to upload the saved model to huggingface"""
    hf_entity: Optional[str] = None
    """The user or org name of the model repository from the Hugging Face Hub"""
    hf_repo_id: Optional[str] = None
    """The id of the saved model in the Hugging Face Hub (can be autoset if not given)"""
    hf_repo_revision: Optional[str] = None
    """The revision of the saved model in the Hugging Face Hub (can be autoset if not given)"""
    hf_repo_url: Optional[str] = None
    """The url of the saved model in the Hugging Face Hub (will be autoset)"""
    try_launch_beaker_eval_jobs: bool = False
    """Whether to launch beaker evaluation jobs after training"""
    hf_metadata_dataset: Optional[str] = "allenai/tulu-3-evals"
    """What dataset to upload the metadata to. If unset, don't upload metadata"""
    cache_dataset_only: bool = False
    """Immediately exit after caching the dataset"""

    # Ai2 specific settings
    try_auto_save_to_beaker: bool = True
    """Whether to try to save the model to Beaker dataset `/output` after training"""
    gs_bucket_path: Optional[str] = None
    """The path to the gs bucket to save the model to"""
    oe_eval_tasks: Optional[List[str]] = None
    """The beaker evaluation tasks to launch"""
    oe_eval_max_length: int = 4096
    """the max generation length for evaluation for oe-eval"""

    def __post_init__(self):
        if self.reduce_loss not in ["mean", "sum"]:
            raise ValueError("reduce_loss must be either 'mean' or 'sum'")
        if (
            self.dataset_name is None
            and self.train_file is None
            and self.dataset_mixer is None
            and self.dataset_mixer_list is None
        ):
            raise ValueError("Need either a dataset name, dataset mixer, or a training file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["json", "jsonl"], "`train_file` should be a json or a jsonl file."
        if (
            (self.dataset_name is not None and (self.dataset_mixer is not None or self.dataset_mixer_list is not None))
            or (self.dataset_name is not None and self.train_file is not None)
            or (
                (self.dataset_mixer is not None or self.dataset_mixer_list is not None) and self.train_file is not None
            )
            or (self.dataset_mixer is not None and self.dataset_mixer_list is not None)
        ):
            raise ValueError("Cannot provide two dataset selection mechanisms.")
        if self.try_launch_beaker_eval_jobs and not self.push_to_hub:
            raise ValueError("Cannot launch Beaker evaluation jobs without pushing to the Hub.")


# Add this function as well
def get_standardized_output_dir(model_name):
    """
    Get a standardized output directory name based on the model name.
    Preserves version numbers in the model name.
    """
    # Extract model name without organization prefix
    clean_name = model_name.split('/')[-1]
    # Keep model version numbers intact (don't replace dots in version numbers)
    # But replace other special characters with underscores
    clean_name = re.sub(r'[^\w\-\.]', '_', clean_name)
    return f"alignment_metrics_{clean_name}"

# Add this function for the alignment metrics computation
def compute_and_save_alignment_metrics(model, tokenizer, dataset, dataset_name, output_dir, sample_size, max_length):
    """
    Compute alignment metrics on a dataset and save them to disk.
    Skip computation if metrics already exist.
    """
    metrics_dir = os.path.join(output_dir, "raw_metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    
    metrics_file = os.path.join(metrics_dir, f"{dataset_name}_metrics.json")
    
    # Skip if metrics already exist
    if os.path.exists(metrics_file):
        logger.info(f"Alignment metrics for {dataset_name} already exist at {metrics_file}. Skipping computation.")
        return
    
    logger.info(f"Computing alignment metrics for {dataset_name} (sample_size={sample_size}, max_length={max_length})...")
    
    # Extract text samples from the dataset
    # Limit to the specified sample size
    sample_size = min(sample_size, len(dataset))
    
    # Get text samples from the dataset
    samples = []
    for i in range(sample_size):
        # Extract text from dataset based on its structure
        if isinstance(dataset[i], dict) and "input_ids" in dataset[i]:
            # If dataset has input_ids, decode them to text
            samples.append(tokenizer.decode(dataset[i]["input_ids"], skip_special_tokens=True))
        elif isinstance(dataset[i], dict) and any(isinstance(dataset[i].get(k), str) for k in dataset[i]):
            # Find the first string value in the dictionary
            for k, v in dataset[i].items():
                if isinstance(v, str):
                    samples.append(v)
                    break
        else:
            # Fallback: try to convert the item to string
            samples.append(str(dataset[i]))
    
    logger.info(f"Prepared {len(samples)} samples for alignment metrics calculation")
    
    # Layer-level adapter selection (--lora_module_frac < 1.0) is not part of the
    # published SFT path; its NFN scoring lives in the repo root (main.py).
    from data import prepare_batch
    from metrics import calculate_alignment_metrics

    batch = prepare_batch(samples, tokenizer, max_length=max_length)
    metrics = calculate_alignment_metrics(model, batch)
    
    # Save metrics
    with open(metrics_file, 'w') as f:
        # Convert defaultdict to regular dict for JSON serialization
        serializable_metrics = {k: dict(v) for k, v in metrics.items()}
        json.dump(serializable_metrics, f, indent=2)
    
    logger.info(f"Saved alignment metrics to {metrics_file}")

def select_layers_by_ratio(
    metrics: Dict[str, Dict[str, float]],
    target_modules: List[str],
    fraction: float,
    strategy: str = "increasing",
) -> Dict[str, List[str]]:
    """
    Returns a dict mapping each suffix in `target_modules` to its selected module names.

    Args:
        metrics: map from full module-name -> {"actual": float, "random": float}.
        target_modules: list of suffixes (e.g. ["q_proj","v_proj"]).
        fraction: fraction in (0,1] to select.
        strategy: one of
            - "increasing"/"decreasing"
            - "first"/"last"/"every"
            - "layer_increasing"/"layer_decreasing"
            - "module_type_increasing"/"module_type_decreasing"
            - a single module type (e.g. "q_proj")
            - comma-separated list of module types (e.g. "q_proj,v_proj")

    Returns:
        Dict[suffix, List[module-names]]
    """
    if not (0 < fraction <= 1):
        raise ValueError("`fraction` must be in (0, 1].")

    # First check if strategy is a module type or list of module types
    module_types = [m.strip() for m in strategy.split(",")]
    # Validate that all module types are in target_modules
    invalid_types = [m for m in module_types if m not in target_modules]
    if not invalid_types:
        # If all module types are valid, use them directly
        return {mod: [name for name in metrics.keys() if name.endswith(mod)] for mod in module_types}

    # If there were invalid types, check if it's a predefined strategy
    allowed = {
        "increasing", "decreasing",
        "first", "last", "every",
        "layer_increasing", "layer_decreasing",
        "module_type_increasing", "module_type_decreasing",
    }
    if strategy in allowed:
        # Handle predefined strategies
        def _layer_index(name: str) -> Optional[int]:
            m = re.search(r"layers\.(\d+)", name)
            return int(m.group(1)) if m else None

        # 1) Bucket all entries by suffix
        entries_by_mod: Dict[str, List[tuple]] = {mod: [] for mod in target_modules}
        all_entries: List[tuple] = []  # (name, ratio, layer_idx)

        for name, scores in metrics.items():
            for mod in target_modules:
                if name.endswith(mod):
                    a = scores.get("actual", 0.0)
                    r = scores.get("random", 0.0)
                    ratio = a / r if r else float("inf")
                    idx = _layer_index(name)
                    entries_by_mod[mod].append((name, ratio, idx))
                    all_entries.append((name, ratio, idx))
                    break

        selection: Dict[str, List[str]] = {mod: [] for mod in target_modules}

        # 2) module_type_*: pick whole suffixes by avg ratio
        if strategy.startswith("module_type_"):
            reverse = strategy.endswith("decreasing")
            # compute avg per suffix
            avg_ratios = []
            for mod, ents in entries_by_mod.items():
                if not ents:
                    continue
                avg = sum(r for _, r, _ in ents) / len(ents)
                avg_ratios.append((mod, avg))
            # sort and pick top fraction
            avg_ratios.sort(key=lambda x: x[1], reverse=reverse)
            to_pick = math.ceil(fraction * len(avg_ratios)) or 1
            picked_mods = {mod for mod, _ in avg_ratios[:to_pick]}
            # collect all their names
            for mod in target_modules:
                if mod in picked_mods:
                    selection[mod] = [name for name, *_ in entries_by_mod[mod]]
            return selection

        # 3) layer_*: pick whole layers across suffixes
        if strategy.startswith("layer_"):
            reverse = strategy.endswith("decreasing")
            # group ratios by layer idx
            layer_map: Dict[int, List[float]] = defaultdict(list)
            for _, ratio, idx in all_entries:
                if idx is not None:
                    layer_map[idx].append(ratio)
            # avg per layer
            layer_avgs = [(idx, sum(vals)/len(vals)) for idx, vals in layer_map.items()]
            layer_avgs.sort(key=lambda x: x[1], reverse=reverse)
            to_pick = math.ceil(fraction * len(layer_avgs)) or 1
            picked_layers = {idx for idx, _ in layer_avgs[:to_pick]}
            # collect names in those layers
            for mod, ents in entries_by_mod.items():
                selection[mod] = [name for name, _, idx in ents if idx in picked_layers]
            return selection

        # 4) other strategies: operate per-suffix
        for mod, ents in entries_by_mod.items():
            if not ents:
                continue

            total = len(ents)
            # sort by index for index-based strategies
            by_idx = sorted(ents, key=lambda x: (x[2] is None, x[2] if x[2] is not None else -1))

            if strategy in ("increasing", "decreasing"):
                rev = (strategy == "decreasing")
                sorted_by_ratio = sorted(ents, key=lambda x: x[1], reverse=rev)
                k = math.ceil(fraction * total) or 1
                chosen = sorted_by_ratio[:k]

            elif strategy == "first":
                k = math.ceil(fraction * total) or 1
                chosen = by_idx[:k]

            elif strategy == "last":
                k = math.ceil(fraction * total) or 1
                chosen = by_idx[-k:]

            elif strategy == "every":
                step = max(1, round(1 / fraction))
                chosen = by_idx[::step] or by_idx[:1]

            else:
                # should never happen
                raise ValueError(f"Unhandled strategy {strategy!r}")

            selection[mod] = [name for name, *_ in chosen]

        return selection

    # If we get here, the strategy is neither a valid module type nor a predefined strategy
    raise ValueError(f"Unknown lora_select_strategy: {strategy}")


@dataclass
class FastDataCollatorForSeq2Seq(DataCollatorForSeq2Seq):  # ← inherits here
    """
    Fully compatible drop-in replacement for HF `DataCollatorForSeq2Seq`
    that avoids the slow `torch.tensor(list_of_ndarrays)` conversion by
    doing a single `np.array(...)` + `torch.from_numpy(...)`.
    All other behaviour (padding strategies, left/right pad, decoder ids)
    is identical.
    """

    # (All the same fields are inherited from the base dataclass)

    def __call__(self, features: List[dict], return_tensors: Optional[str] = None) -> dict:
        # -------------------------------------------------------------
        # 1) Split off labels
        label_name = "label" if "label" in features[0] else "labels"
        labels = [f[label_name] for f in features] if label_name in features[0] else None
        if labels is not None and all(l is None for l in labels):
            labels = None
        feats_no_label = [{k: v for k, v in f.items() if k != label_name} for f in features]

        # -------------------------------------------------------------
        # 2) Pad inputs exactly the same way HF does
        batch = self.tokenizer.pad(
            feats_no_label,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors or self.return_tensors,
        )

        # -------------------------------------------------------------
        # 3) Pad labels into a list of equal-length lists
        if labels is not None:
            no_pad = (self.padding is False) or (self.padding == PaddingStrategy.DO_NOT_PAD)
            if no_pad:
                batch["labels"] = list(labels)
            else:
                # decide target length
                if self.padding == PaddingStrategy.MAX_LENGTH and self.max_length:
                    tgt_len = self.max_length
                else:
                    tgt_len = max(len(l) for l in labels)
                if self.pad_to_multiple_of:
                    m = self.pad_to_multiple_of
                    tgt_len = ((tgt_len + m - 1) // m) * m

                pad_id = self.label_pad_token_id
                side = self.tokenizer.padding_side
                padded = []
                for lbl in labels:
                    arr = lbl.tolist() if hasattr(lbl, "tolist") else list(lbl)
                    pad_len = tgt_len - len(arr)
                    if pad_len:
                        if side == "right":
                            arr += [pad_id] * pad_len
                        else:
                            arr = [pad_id] * pad_len + arr
                    padded.append(arr)
                batch["labels"] = padded
        else:
            batch["labels"] = None

        # -------------------------------------------------------------
        # 4) Fast tensor conversion (PyTorch / NumPy)
        rt = return_tensors or self.return_tensors
        if batch["labels"] is not None:
            if rt == "pt":
                batch["labels"] = torch.from_numpy(np.array(batch["labels"], dtype=np.int64))
            elif rt == "np":
                batch["labels"] = np.array(batch["labels"], dtype=np.int64)
            # "tf" path just falls back to base behaviour → leave as list

        # -------------------------------------------------------------
        # 5) Optional decoder_input_ids
        if (
            batch.get("labels") is not None
            and self.model is not None
            and hasattr(self.model, "prepare_decoder_input_ids_from_labels")
        ):
            batch["decoder_input_ids"] = self.model.prepare_decoder_input_ids_from_labels(
                labels=batch["labels"]
            )

        return batch


def main(args: FlatArguments, tc: TokenizerConfig):
    logging.getLogger().setLevel(logging.INFO)

    # ------------------------------------------------------------
    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    accelerator_log_kwargs = {}
    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["project_dir"] = args.output_dir
    # if you get timeouts (e.g. due to long tokenization) increase this.
    timeout_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=args.timeout))
    dataloader_config = DataLoaderConfiguration(use_seedable_sampler=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        dataloader_config=dataloader_config,
        **accelerator_log_kwargs,
        kwargs_handlers=[timeout_kwargs],
    )

    # ------------------------------------------------------------
    # Setup tokenizer
    tc.tokenizer_revision = args.model_revision if tc.tokenizer_revision is None else tc.tokenizer_revision
    tc.tokenizer_name_or_path = (
        args.model_name_or_path if tc.tokenizer_name_or_path is None else tc.tokenizer_name_or_path
    )
    if tc.tokenizer_revision != args.model_revision and tc.tokenizer_name_or_path != args.model_name_or_path:
        # Warn user if tokenizer and model use different revisions; this is an unusual
        # use case.
        warning = f"""Requested tokenizer revision `{tc.tokenizer_revision=}` is different
                   from the model revision `{args.model_revision=}` or the tokenizer name `{tc.tokenizer_name_or_path=}`
                   is different from the model name `{args.model_name_or_path=}`."""
        logger.warning(warning)
    tokenizer = tc.tokenizer

    # ------------------------------------------------------------
    # Set up runtime variables
    if args.run_name is None:
        args.run_name = f"{args.exp_name}__{args.seed}__{int(time.time())}"
    args.output_dir = os.path.join(args.output_dir, args.run_name)
    args.dataset_local_cache_dir = os.path.abspath(args.dataset_local_cache_dir)
    if is_beaker_job():
        args.dataset_local_cache_dir = "/weka/oe-adapt-default/allennlp/deletable_open_instruct_dataset_cache"
    if args.push_to_hub and accelerator.is_main_process:
        if args.hf_repo_id is None:  # auto-generate one
            args.hf_repo_id = "open_instruct_dev"
        if args.hf_entity is None:  # first try to use AI2 entity
            args.hf_entity = maybe_use_ai2_hf_entity()
        if args.hf_entity is None:  # then try to use the user's entity
            args.hf_entity = HfApi().whoami()["name"]
        args.hf_repo_id = f"{args.hf_entity}/{args.hf_repo_id}"
        if args.hf_repo_revision is None:
            args.hf_repo_revision = args.run_name
        args.hf_repo_url = f"https://huggingface.co/{args.hf_repo_id}/tree/{args.hf_repo_revision}"
        if is_beaker_job():
            beaker_config = maybe_get_beaker_config()

    # ------------------------------------------------------------
    # Initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"]

        # (Optional) Ai2 internal tracking
        if args.wandb_entity is None:
            args.wandb_entity = maybe_use_ai2_wandb_entity()
        if accelerator.is_main_process and is_beaker_job():
            experiment_config.update(vars(beaker_config))
        experiment_config.update(vars(tc))
        accelerator.init_trackers(
            args.wandb_project_name,
            experiment_config,
            init_kwargs={
                "wandb": {
                    "name": (args.exp_name + "_" + args.run_name).replace("/", "_"),
                    "entity": args.wandb_entity,
                    "tags": [args.tags] + get_wandb_tags(),
                }
            },
        )
        wandb_tracker = accelerator.get_tracker("wandb")

    if accelerator.is_main_process:
        pprint([args, tc])

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    accelerator.wait_for_everyone()

    if args.dataset_mixer is not None:
        args.dataset_mixer_list = [item for pair in args.dataset_mixer.items() for item in pair]
    with accelerator.main_process_first():
        transform_fn_args = [
            {"max_seq_length": args.max_seq_length},
            {},
        ]
        train_dataset = get_cached_dataset_tulu(
            dataset_mixer_list=args.dataset_mixer_list,
            dataset_mixer_list_splits=args.dataset_mixer_list_splits,
            tc=tc,
            dataset_transform_fn=args.dataset_transform_fn,
            transform_fn_args=transform_fn_args,
            target_columns=args.dataset_target_columns,
            dataset_cache_mode=args.dataset_cache_mode,
            dataset_config_hash=args.dataset_config_hash,
            hf_entity=args.hf_entity,
            dataset_local_cache_dir=args.dataset_local_cache_dir,
            dataset_skip_cache=args.dataset_skip_cache,
        )
        train_dataset = train_dataset.shuffle(seed=args.seed)
        train_dataset.set_format(type="pt")
    if accelerator.is_main_process:
        visualize_token(train_dataset[0][INPUT_IDS_KEY], tokenizer)

    if args.cache_dataset_only:
        return

    # Load pretrained model and tokenizer
    if args.config_name:
        config = AutoConfig.from_pretrained(
            args.config_name,
            revision=args.model_revision,
            trust_remote_code=tc.trust_remote_code,
        )
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(
            args.model_name_or_path,
            revision=args.model_revision,
            trust_remote_code=tc.trust_remote_code,
        )
    else:
        raise ValueError(
            "You are instantiating a new config instance from scratch. This is not supported by this script."
        )

    if args.model_name_or_path:
        if args.use_qlora:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            device_index = accelerator.local_process_index
            device_map = {"": device_index}  # force data-parallel training.
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name_or_path,
                revision=args.model_revision,
                from_tf=bool(".ckpt" in args.model_name_or_path),
                config=config,
                trust_remote_code=tc.trust_remote_code,
                quantization_config=bnb_config,
                device_map=device_map,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2" if args.use_flash_attn else "eager",
            )
        elif args.use_liger_kernel:
            raise NotImplementedError("LigerKernel is not yet supported.")
            # from liger_kernel.transformers import AutoLigerKernelForCausalLM

            #     fused_linear_cross_entropy = args.reduce_loss == "mean"
            #     logger.info(f"Attempting to apply liger-kernel. {fused_linear_cross_entropy=}")

            #     # Supported models: https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/transformers/monkey_patch.py#L948
            #     model = AutoLigerKernelForCausalLM.from_pretrained(
            #         args.model_name_or_path,
            #         revision=args.model_revision,
            #         from_tf=bool(".ckpt" in args.model_name_or_path),
            #         config=config,
            #         trust_remote_code=tc.trust_remote_code,
            #         low_cpu_mem_usage=args.low_cpu_mem_usage,
            #         use_flash_attention_2=True if args.use_flash_attn else False,
            #         # liger-kernel specific args
            #         fused_linear_cross_entropy=fused_linear_cross_entropy,
            #     )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name_or_path,
                revision=args.model_revision,
                from_tf=bool(".ckpt" in args.model_name_or_path),
                config=config,
                trust_remote_code=tc.trust_remote_code,
                low_cpu_mem_usage=args.low_cpu_mem_usage,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2" if args.use_flash_attn else "eager",
            )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForCausalLM.from_config(config)

    # Explicitly move the model to CUDA
    if torch.cuda.is_available():
        model = model.to("cuda")

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    # gather deepspeed to get "real" embedding size
    embeddings = model.get_input_embeddings()
    with deepspeed.zero.GatheredParameters(embeddings.weight, modifier_rank=None):
        embedding_size = embeddings.weight.shape[0]
    # resize does its own gather
    if len(tokenizer) > embedding_size:
        # pad to multiple for tensor cores.
        model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)
    # update embedding size after resizing for sum loss
    embeddings = model.get_input_embeddings()
    with deepspeed.zero.GatheredParameters(embeddings.weight, modifier_rank=None):
        embedding_size = embeddings.weight.shape[0]

    compute_alignment_metrics = args.use_lora and args.lora_module_frac < 1.0
    if compute_alignment_metrics:
        alignment_output_dir = get_standardized_output_dir(args.model_name_or_path)

        # Get the dataset name for the metrics file
        dataset_key = "_".join(
            [name.split('/')[-1] for name in args.dataset_mixer_list if not name.replace(".", "").isdigit()][:1]
        )
        if not dataset_key:
            dataset_key = "train_dataset"
    
        # For metrics calculation, we need the model on the main process
        if accelerator.is_main_process:
            logger.info(f"Computing alignment metrics before training...")
            
            # Create a copy of the model for metrics calculation to avoid modifying the training model
            # We'll use the already loaded model
            metrics_model = model
            metrics_model.eval()
            
            # Compute and save metrics
            compute_and_save_alignment_metrics(
                metrics_model, 
                tokenizer, 
                train_dataset, 
                dataset_key, 
                alignment_output_dir,
                sample_size=args.alignment_metrics_sample_size,
                max_length=args.alignment_metrics_max_length
            )
            
            # Don't delete the model since we're using the same one for training
            metrics_model.train()  # Set back to training mode
            
            logger.info(f"Alignment metrics computation completed.")
        
        # Wait for all processes to complete before continuing
        accelerator.wait_for_everyone()

    if args.use_lora:
        if args.use_qlora:
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)
        
        # Define the default target modules for LoRA.
        default_target_module_types = [
            "q_proj",
            "o_proj",
            "v_proj",
            "k_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]

        selected_layer_names = default_target_module_types

        # First check if strategy is a module type or list of module types
        module_types = [m.strip() for m in args.lora_select_strategy.split(",")]
        # Validate that all module types are in default_target_module_types
        invalid_types = [m for m in module_types if m not in default_target_module_types]
        
        if not invalid_types:
            # If all module types are valid, use them directly
            selected_layer_names = module_types
        elif args.lora_module_frac < 1.0:
            # Path to your computed metrics JSON file.
            # Here, 'dataset_key' should have already been sanitized (e.g. "MetaMathQA")
            metrics_file = os.path.join(alignment_output_dir, "raw_metrics", f"{dataset_key}_metrics.json")

            # Map abbreviated strategy names to full names.
            strategy_map = {
                "inc": "increasing",
                "incr": "increasing",
                "increasing": "increasing",
                "dec": "decreasing",
                "decr": "decreasing",
                "decreasing": "decreasing",
                "layer_inc": "layer_increasing",
                "layer_incr": "layer_increasing",
                "layer_increasing": "layer_increasing",
                "layer_dec": "layer_decreasing",
                "layer_decr": "layer_decreasing",
                "layer_decreasing": "layer_decreasing",
                "mod_type_inc": "module_type_increasing",
                "mod_type_incr": "module_type_increasing",
                "mod_type_increasing": "module_type_increasing",
                "mod_type_dec": "module_type_decreasing",
                "mod_type_decr": "module_type_decreasing",
                "mod_type_decreasing": "module_type_decreasing",
                "ev": "every",
                "every": "every",
                "fst": "first",
                "first": "first",
                "lst": "last",
                "last": "last",
            }
            
            # Otherwise try to map to a predefined strategy
            lora_strategy = strategy_map.get(args.lora_select_strategy.lower())
            if lora_strategy is None:
                raise ValueError(f"Unknown lora_select_strategy: {args.lora_select_strategy}")
            
            # Read metrics from the JSON file
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
            
            # Use command-line arguments for LoRA module fraction and selection strategy.
            selected_layers_dict = select_layers_by_ratio(
                metrics, 
                default_target_module_types, 
                args.lora_module_frac, 
                strategy=lora_strategy
            )
            
            # Flatten the selected layers into a single list of layer names.
            selected_layer_names = [layer for layers in selected_layers_dict.values() for layer in layers]

        
        logger.info("Initializing LORA model...")
        logger.info(f"Selected LoRA target modules: {selected_layer_names}")

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=selected_layer_names,
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    elif args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # DataLoaders creation:
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=FastDataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding="longest"),
        batch_size=args.per_device_train_batch_size,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    if args.use_qlora:
        raise NotImplementedError("QLoRA is not yet supported.")
        # from bitsandbytes.optim import AdamW

        # optimizer = AdamW(
        #     optimizer_grouped_parameters,
        #     lr=args.learning_rate,
        #     optim_bits=8 if args.use_8bit_optimizer else 32,
        #     is_paged=True,
        # )
    else:
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=args.learning_rate,
            fused=args.fused_optimizer,
        )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    # Create the learning rate scheduler.
    # Note: the current accelerator.step() calls the .step() of the real scheduler
    # for the `num_processes` times. This is because they assume
    # the user initialize the scheduler with the entire training set.
    # In the case of data parallel training, each process only
    # sees a subset (1/num_processes) of the training set.
    # So each time the process needs to update the lr multiple times so that the total
    # number of updates in the end matches the num_training_steps here.
    # Here we need to set the num_training_steps to either using the
    # entire training set (when epochs is specified) or we need to multiply the
    # num_training_steps by num_processes so that the total number of
    # updates matches the num_training_steps.
    num_training_steps_for_scheduler = (
        args.max_train_steps if overrode_max_train_steps else args.max_train_steps * accelerator.num_processes
    )
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_training_steps=num_training_steps_for_scheduler,
        num_warmup_steps=int(num_training_steps_for_scheduler * args.warmup_ratio),
    )
    # Prepare everything with `accelerator`.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and str(checkpointing_steps).lower() != "epoch":
        checkpointing_steps = int(checkpointing_steps)

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    completed_steps = 0
    starting_epoch = 0

    # Potentially load in the weights and states from a previous save
    last_checkpoint_path = get_last_checkpoint_path(args)
    if last_checkpoint_path:
        accelerator.print(f"Resumed from checkpoint: {last_checkpoint_path}")
        accelerator.load_state(last_checkpoint_path)
        # Extract `epoch_{i}` or `step_{i}`
        last_checkpoint_path = os.path.basename(last_checkpoint_path)
        training_difference = os.path.splitext(last_checkpoint_path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
            completed_steps = starting_epoch * num_update_steps_per_epoch
        else:
            # need to multiply `gradient_accumulation_steps` to reflect real steps
            resume_step = int(training_difference.replace("step_", "")) * args.gradient_accumulation_steps
            starting_epoch = resume_step // len(train_dataloader)
            completed_steps = resume_step // args.gradient_accumulation_steps
            resume_step -= starting_epoch * len(train_dataloader)

    print(f"Starting from epoch {starting_epoch} and step {completed_steps}.")
    progress_bar = tqdm(total=args.max_train_steps, initial=completed_steps, disable=not accelerator.is_local_main_process)
    local_total_tokens = torch.tensor(0, dtype=torch.int64, device=accelerator.device)
    total_token_including_padding = torch.tensor(0, dtype=torch.int64, device=accelerator.device)
    start_time = time.time()
    last_logged_avg_loss = None


    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        train_dataloader.set_epoch(epoch)
        total_loss = 0
        total_aux_loss = 0
        if last_checkpoint_path and resume_step is not None:
            # We skip the first `n` batches in the dataloader when resuming from a checkpoint
            active_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)
        else:
            active_dataloader = train_dataloader
        for step, batch in enumerate(active_dataloader):
            local_total_tokens += batch["attention_mask"].sum()
            total_token_including_padding += batch["attention_mask"].numel()
            with accelerator.accumulate(model):
                if args.load_balancing_loss:
                    outputs = model(**batch, use_cache=False, output_router_logits=True)
                else:
                    # TODO: we have calculated the mean loss here anyway, so doubling the calculation
                    outputs = model(**batch, use_cache=False)
                if args.reduce_loss == "mean":
                    loss = outputs.loss
                else:
                    # reduce loss is sum
                    # this ensures that we weight all tokens in the dataset equally,
                    # rather than weighting each overall example equally when
                    # using high amounts of gradient accumulation.
                    # this can result in > 5 point improvements in AlpacaEval
                    # see https://github.com/huggingface/transformers/issues/24725 for
                    # more discussion and details.
                    logits = outputs.logits
                    labels = batch["labels"]
                    # Shift so that tokens < n predict n
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    # Flatten the tokens
                    loss_fct = torch.nn.CrossEntropyLoss(reduction="sum")
                    shift_logits = shift_logits.view(-1, embedding_size)
                    shift_labels = shift_labels.view(-1)
                    # Enable model parallelism
                    shift_labels = shift_labels.to(shift_logits.device)
                    loss = loss_fct(shift_logits, shift_labels)
                    if args.load_balancing_loss:
                        aux_loss = args.load_balancing_weight * outputs.aux_loss
                        loss += aux_loss
                # We keep track of the loss at each logged step
                total_loss += loss.detach().float()
                accelerator.backward(loss)
                if args.load_balancing_loss:
                    total_aux_loss += aux_loss.detach().float()
                # clip gradient norm. don't do this with deepspeed
                if accelerator.sync_gradients and args.clip_grad_norm > 0:
                    accelerator.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1
                if args.logging_steps and completed_steps % args.logging_steps == 0:
                    avg_loss = (
                        accelerator.gather(total_loss).mean().item()
                        / args.gradient_accumulation_steps
                        / args.logging_steps
                    )
                    last_logged_avg_loss = avg_loss
                    total_tokens = accelerator.gather(local_total_tokens).sum().item()
                    total_tokens_including_padding = accelerator.gather(total_token_including_padding).sum().item()
                    metrics_to_log = {
                        "learning_rate": lr_scheduler.get_last_lr()[0],
                        "train_loss": avg_loss,
                        "total_tokens": total_tokens,
                        "per_device_tps": total_tokens / accelerator.num_processes / (time.time() - start_time),
                        "total_tokens_including_padding": total_tokens_including_padding,
                        "per_device_tps_including_padding": total_tokens_including_padding
                        / accelerator.num_processes
                        / (time.time() - start_time),
                    }
                    if args.load_balancing_loss:
                        avg_aux_loss = (
                            accelerator.gather(total_aux_loss).mean().item()
                            / args.gradient_accumulation_steps
                            / args.logging_steps
                        )
                        logger.info(
                            f"  Step: {completed_steps}, LR: {lr_scheduler.get_last_lr()[0]}, Loss: {avg_loss}, Aux Loss: {avg_aux_loss}, TPS: {total_tokens / (time.time() - start_time)}"
                        )
                        metrics_to_log["aux_loss"] = avg_aux_loss
                    else:
                        logger.info(
                            f"  Step: {completed_steps}, LR: {lr_scheduler.get_last_lr()[0]}, Loss: {avg_loss}, TPS: {total_tokens / (time.time() - start_time)}"
                        )
                    if args.with_tracking:
                        accelerator.log(
                            metrics_to_log,
                            step=completed_steps,
                        )
                    total_loss = 0
                    total_aux_loss = 0

                if isinstance(checkpointing_steps, int):
                    if completed_steps % checkpointing_steps == 0:
                        output_dir = f"step_{completed_steps}"
                        if args.output_dir is not None:
                            output_dir = os.path.join(args.output_dir, output_dir)
                        accelerator.save_state(output_dir, max_shard_size="2GB")
                        # use this to mark the checkpoint as completely saved, to avoid restoring from garbled checkpoints
                        with open(
                            os.path.join(
                                get_last_checkpoint_path(args, incomplete=True),
                                "COMPLETED",
                            ),
                            "w",
                        ) as f:
                            f.write("COMPLETED")  # annoyingly, empty files arent uploaded by beaker.
                        if (
                            accelerator.is_local_main_process
                        ):  # TODO: in mason local model this is gonna error out if using something like output/test; because mason used the same shared file ssytem.
                            clean_last_n_checkpoints(args.output_dir, args.keep_last_n_checkpoints)
                        accelerator.wait_for_everyone()

                if completed_steps >= args.max_train_steps:
                    break

        if checkpointing_steps == "epoch":
            output_dir = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
            accelerator.save_state(output_dir, max_shard_size="2GB")
            # use this to mark the checkpoint as completely saved, to avoid restoring from garbled checkpoints
            with open(
                os.path.join(get_last_checkpoint_path(args, incomplete=True), "COMPLETED"),
                "w",
            ) as f:
                f.write("COMPLETED")  # annoyingly, empty files arent uploaded by beaker.
            if accelerator.is_local_main_process:
                clean_last_n_checkpoints(args.output_dir, args.keep_last_n_checkpoints)
            accelerator.wait_for_everyone()

    # Save final training loss to JSON
    if accelerator.is_main_process and last_logged_avg_loss is not None:
        final_loss_path = os.path.join(args.output_dir, "final_train_loss.json")
        with open(final_loss_path, "w") as f:
            json.dump({"final_train_loss": last_logged_avg_loss}, f, indent=2)
        logger.info(f"Saved final logged training loss to {final_loss_path}")


    if args.output_dir is not None:
        save_with_accelerate(
            accelerator,
            model,
            tokenizer,
            args.output_dir,
            args.use_lora,
        )

    # remove all checkpoints to save space
    if accelerator.is_local_main_process:
        clean_last_n_checkpoints(args.output_dir, keep_last_n_checkpoints=0)

    if (
        args.try_auto_save_to_beaker
        and accelerator.is_main_process
        and is_beaker_job()
        and len(beaker_config.beaker_dataset_id_urls) > 0
        and args.output_dir.rstrip("/") != "/output"
    ):
        shutil.copytree(args.output_dir, "/output", dirs_exist_ok=True)

    if is_beaker_job() and accelerator.is_main_process and args.try_launch_beaker_eval_jobs:
        launch_ai2_evals_on_weka(
            path=args.output_dir,
            leaderboard_name=args.hf_repo_revision,
            oe_eval_max_length=args.oe_eval_max_length,
            wandb_url=wandb_tracker.run.get_url(),
            oe_eval_tasks=args.oe_eval_tasks,
            gs_bucket_path=args.gs_bucket_path,
        )
    if args.push_to_hub:
        push_folder_to_hub(
            accelerator,
            args.output_dir,
            args.hf_repo_id,
            args.hf_repo_revision,
        )
    accelerator.wait_for_everyone()
    if args.with_tracking:
        accelerator.end_training()


if __name__ == "__main__":
    parser = ArgumentParserPlus((FlatArguments, TokenizerConfig))
    args, tc = parser.parse_args_into_dataclasses()
    main(args, tc)