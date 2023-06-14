export HF_DATASETS_CACHE="/home/ms/hf_cache"

size=$4
MODEL=llama_hf_ms/llama-${size}b-hf
bit=$2
format=$3

CUDA_VISIBLE_DEVICES=$1 python -m awq.entry --model_path $MODEL \
    --w_bit ${bit} --q_group_size 128 --q_format ${format} --mse_range \
    --run_awq --dump_awq awq_cache/llama-${size}b-hf-ms-w${bit}-g128-${format}.pt

# evaluate the AWQ quantize model (simulated pseudo quantization)
CUDA_VISIBLE_DEVICES=$1 python -m awq.entry --model_path $MODEL \
    --tasks piqa,hellaswag,winogrande,arc_easy \
    --w_bit ${bit} --q_group_size 128 --q_format ${format} \
    --load_awq awq_cache/llama-${size}b-hf-ms-w${bit}-g128-${format}.pt \
    --q_backend fake \
    --output_path llama-${size}b-${bit}bit-${format}-qa
