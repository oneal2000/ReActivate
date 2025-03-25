python src/warmup_lora.py \
    --model_name llama3.2-1b-instruct \
    --per_device_train_batch_size 1 \
    --num_train_epochs 1 \
    --learning_rate 3e-4  \
    --block_size 3000 \
    --lora_rank 2 \
    --lora_alpha 32 \
    --IA3
    # --with_cot \