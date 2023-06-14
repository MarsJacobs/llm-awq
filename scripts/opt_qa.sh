export HF_DATASETS_CACHE="/raid/ms/hf_cache"

# 125m 350m 1.3b 2.7b 6.7b 13b
size=$4
MODEL=facebook/opt-${size}
bit=$2
format=$3

CUDA_VISIBLE_DEVICES=$1 python -m awq.entry --model_path $MODEL \
    --w_bit ${bit} --q_group_size 128 --q_format ${format} --mse_range \
    --run_awq --dump_awq awq_cache/opt-${size}-hf-ms-w${bit}-g128-${format}.pt

# evaluate the AWQ quantize model (simulated pseudo quantization)
CUDA_VISIBLE_DEVICES=$1 python -m awq.entry --model_path $MODEL \
    --tasks wikitext \
    --w_bit ${bit} --q_group_size 128 --q_format ${format} \
    --load_awq awq_cache/opt-${size}-hf-ms-w${bit}-g128-${format}.pt \
    --q_backend fake \
    --output_path opt-${size}-${bit}bit-${format}-wiki
