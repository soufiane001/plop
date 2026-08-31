
# Reproduces the PLoP row of Table 2. For the other placements, set
# --lora_select_strategy to one of:
#   PLoP          down_proj,o_proj,v_proj
#   PLoP-inverse  gate_proj,k_proj,q_proj
#   MLP           down_proj,gate_proj,up_proj
#   Attn          k_proj,q_proj,v_proj
# LoRA alpha is 2 * rank throughout.

accelerate launch --mixed_precision bf16 --num_processes 4 sft_metamath.py \
    --model_name_or_path Qwen/Qwen3-1.7B \
    --use_lora True \
    --lora_select_strategy down_proj,o_proj,v_proj \
    --lora_rank 64 \
    --lora_alpha 128 \
    --lora_dropout 0.0 \
    --dataset_mixer_list meta-math/MetaMathQA 1.0 \
    --dataset_mixer_list_splits train \
    --dataset_transform_fn sft_metamathqa_tokenize_and_truncate_v1 sft_metamathqa_filter_v1 \
    --max_seq_length 1024 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --learning_rate 4e-4 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --weight_decay 0.0 \
    --num_train_epochs 2 \
    --clip_grad_norm 1.0 \
    --use_flash_attn True \
    --output_dir ../results/qwen3_1.7b_r64_dov

python eval_gsm8k.py \
    --adapter ../results/qwen3_1.7b_r64_dov \
    --num-fewshot 8 \
    --use-vllm \
    --max-tokens 512 \
    --result-file ../results/qwen3_1.7b_r64_dov/gsm8k.json
