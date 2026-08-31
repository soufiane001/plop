import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="GRPO finetuning on MetaMathQA.")
    parser.add_argument('--model_id', type=str, required=True, help='HuggingFace model handle')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save checkpoints')
    parser.add_argument('--target_modules', type=str, nargs='+', required=True,
                        help='Module types to place LoRA adapters in, e.g. q_proj k_proj gate_proj')
    parser.add_argument('--learning_rate', type=float, default=4e-6, help='Learning rate')
    parser.add_argument('--num_generations', type=int, default=8, help='Generations per prompt')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4, help='Gradient accumulation steps')
    parser.add_argument('--per_device_train_batch_size', type=int, default=16, help='Per-device batch size')
    parser.add_argument('--logging_steps', type=int, default=2, help='Logging interval in steps')
    return parser.parse_args()
