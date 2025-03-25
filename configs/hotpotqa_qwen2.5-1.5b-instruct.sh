python3 src/encode.py \
    --model_name=qwen2.5-1.5b-instruct \
    --dataset=hotpotqa \
    --sample=300 \
    --per_device_train_batch_size=1 \
    --num_train_epochs=2 \
    --learning_rate=0.0003 \
    --lora_rank=2 \
    --lora_alpha=32 \
    --with_cot \
    --IA3 \
    # --warm_up

python3 src/inference.py \
    --model_name=qwen2.5-1.5b-instruct \
    --dataset=hotpotqa \
    --sample=300 \
    --num_train_epochs=2 \
    --learning_rate=0.0003 \
    --lora_rank=2 \
    --lora_alpha=32 \
    --max_new_tokens=128 \
    --inference_method=prag \
    --with_cot \
    --doc_num=3 \
    --IA3 \
    # --warm_up