export HF_DATASETS_CACHE="/home/ms/hf_cache"

size=$2
MODEL=huggyllama/llama-${size}b

CUDA_VISIBLE_DEVICES=$1 python -m awq.entry --model_path $MODEL \
    --tasks wikitext \
    --output_path llama-${size}b-fp-qa
