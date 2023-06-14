export HF_DATASETS_CACHE="/home/ms/hf_cache"

size=$3
MODEL=llama_hf_ms/llama-${size}b-hf
bit=$2

# evaluate the AWQ quantize model (simulated pseudo quantization)
CUDA_VISIBLE_DEVICES=$1 python -m awq.entry --model_path $MODEL \
    --mmlu_dir mmlu_data \
    --output_path llama-${size}b-fp-${format}-mmlu
