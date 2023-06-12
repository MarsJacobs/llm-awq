export HF_DATASETS_CACHE="/home/ms/hf_cache"

MODEL=llama_hf_ms/llama-7b-hf

for bit in 4
do

# evaluate the AWQ quantize model (simulated pseudo quantization)
python -m awq.entry --model_path $MODEL \
    --mmlu_dir mmlu_data \
    --w_bit ${bit} --q_group_size 128 \
    --load_awq awq_cache/llama-7b-hf-ms-w${bit}-g128.pt \
    --q_backend fake
done
