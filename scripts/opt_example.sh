# run AWQ search (optional; we provided the pre-computed results)

for MODEL in opt-13b
do
python -m awq.entry --model_path facebook/$MODEL \
    --w_bit 3 --q_group_size 128 \
    --run_awq --dump_awq awq_cache/$MODEL-w3-g128.pt
python -m awq.entry --model_path facebook/$MODEL \
    --tasks wikitext \
    --w_bit 3 --q_group_size 128 \
    --load_awq awq_cache/$MODEL-w3-g128.pt \
#     --q_backend fake
done
# evaluate the AWQ quantize model (simulated pseudo quantization)
# python -m awq.entry --model_path facebook/opt-6.7b \
#     --tasks wikitext \
#     --w_bit 4 --q_group_size 128 \
#     --load_awq awq_cache/$MODEL-w4-g128.pt \
#     --q_backend fake

# # generate real quantized weights (w4)
# python -m awq.entry --model_path /dataset/opt/$MODEL \
#     --w_bit 4 --q_group_size 128 \
#     --load_awq awq_cache/$MODEL-w4-g128.pt \
#     --q_backend real --dump_quant quant_cache/$MODEL-w4-g128-awq.pt

# # load and evaluate the real quantized model (smaller gpu memory usage)
# python -m awq.entry --model_path /dataset/opt/$MODEL \
#     --tasks wikitext \
#     --w_bit 4 --q_group_size 128 \
#     --load_quant quant_cache/$MODEL-w4-g128-awq.pt