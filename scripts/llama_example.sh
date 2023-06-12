export HF_DATASETS_CACHE="/home/ms/hf_cache"

MODEL=llama_hf_ms/llama-7b-hf

for bit in 3 4
do
# run AWQ search (optional; we provided the pre-computed results)
python -m awq.entry --model_path $MODEL \
    --w_bit ${bit} --q_group_size 128 \
    --run_awq --dump_awq awq_cache/llama-7b-hf-ms-w${bit}-g128.pt

# evaluate the AWQ quantize model (simulated pseudo quantization)
python -m awq.entry --model_path $MODEL \
    --tasks piqa,hellaswag,winogrande,arc_easy \
    --w_bit ${bit} --q_group_size 128 \
    --load_awq awq_cache/llama-7b-hf-ms-w${bit}-g128.pt \
    --q_backend fake
done

MODEL=llama_hf_ms/llama-13b-hf

for bit in 3 4
do
# run AWQ search (optional; we provided the pre-computed results)
python -m awq.entry --model_path $MODEL \
    --w_bit ${bit} --q_group_size 128 \
    --run_awq --dump_awq awq_cache/llama-13b-hf-ms-w${bit}-g128.pt

# evaluate the AWQ quantize model (simulated pseudo quantization)
python -m awq.entry --model_path $MODEL \
    --tasks piqa,hellaswag,winogrande,arc_easy \
    --w_bit ${bit} --q_group_size 128 \
    --load_awq awq_cache/llama-13b-hf-ms-w${bit}-g128.pt \
    --q_backend fake
done

MODEL=llama_hf_ms/llama-30b-hf

for bit in 3 4
do
# run AWQ search (optional; we provided the pre-computed results)
python -m awq.entry --model_path $MODEL \
    --w_bit ${bit} --q_group_size 128 \
    --run_awq --dump_awq awq_cache/llama-30b-hf-ms-w${bit}-g128.pt

# evaluate the AWQ quantize model (simulated pseudo quantization)
python -m awq.entry --model_path $MODEL \
    --tasks piqa,hellaswag,winogrande,arc_easy \
    --w_bit ${bit} --q_group_size 128 \
    --load_awq awq_cache/llama-30b-hf-ms-w${bit}-g128.pt \
    --q_backend fake
done

MODEL=llama_hf_ms/llama-65b-hf

for bit in 3 4
do
# run AWQ search (optional; we provided the pre-computed results)
python -m awq.entry --model_path $MODEL \
    --w_bit ${bit} --q_group_size 128 \
    --run_awq --dump_awq awq_cache/llama-65b-hf-ms-w${bit}-g128.pt

# evaluate the AWQ quantize model (simulated pseudo quantization)
python -m awq.entry --model_path $MODEL \
    --tasks piqa,hellaswag,winogrande,arc_easy \
    --w_bit ${bit} --q_group_size 128 \
    --load_awq awq_cache/llama-65b-hf-ms-w${bit}-g128.pt \
    --q_backend fake
done
