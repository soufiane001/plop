# PLoP: Precise LoRA Placement for Efficient Finetuning of Large Models (ICLR 2026)

This is the official code for "PLoP: Precise LoRA Placement for Efficient Finetuning of Large Models" (https://arxiv.org/abs/2506.20629). `main.py` computes the alignment metrics that PLoP uses to select module types; `sft/` reproduces the MetaMathQA → GSM8K finetuning results.

## Usage

Install dependencies:
```bash
pip install -r requirements.txt
```

Run the main script:
```bash
python main.py --model <huggingface-model-handle> --dataset <math|code|history|logic> --batchsize <BATCHSIZE> --nbsamples <N> --seqlen <SEQ_LEN> --aggregation <type|layer|None> --output_dir <RESULTS_DIR>
```

Example:
```bash
python main.py --model meta-llama/Llama-3.2-1B-Instruct --dataset math --batchsize 8 --nbsamples 100 --seqlen 256 --aggregation type --output_dir results/
```

## Arguments
- `--model`: HuggingFace model handle (e.g., `google/gemma-2b`)
- `--dataset`: Dataset name (`math`, `code`, `history`, `logic`)
- `--batchsize`: Batch size
- `--nbsamples`: Number of samples to use from the dataset
- `--seqlen`: Sequence length for tokenization
- `--aggregation`: How to aggregate results (`type`, `layer`, or `None`)
- `--output_dir`: Directory to save results

## Output
- Raw and aggregated metrics are saved as JSON files in the specified output directory.

## Finetuning and evaluation

`sft/sft_metamath.py` finetunes a model with LoRA adapters in the module types
selected by PLoP. `sft/eval_gsm8k.py` merges the adapters into the base model and
evaluates 8-shot GSM8K accuracy.

`sft/Qwen3_1.7B_sft.sh` runs both stages with the configuration used for the
paper. Run it from inside `sft/`:

```bash
bash Qwen3_1.7B_sft.sh
```
