
# 01 original
CUDA_VISIBLE_DEVICES=0 nohup uv run python -m cs336_basics.train --config configs/01_origin.json --checkpoint-dir checkpoints/01_original > logs/01_original.log 2>&1 &
# 02 mixed precision
CUDA_VISIBLE_DEVICES=1 nohup uv run python -m cs336_basics.train --config configs/02_mix_precision.json --use-mixed-precision --mixed-precision-dtype bfloat16 --checkpoint-dir checkpoints/02_mixed_precision > logs/02_mixed_precision.log 2>&1 &





# python -m cs336_basics.generate \
#     --checkpoint checkpoints/small_model/latest.pt \
#     --vocab-path bpe_result/tiny/vocab.pkl \
#     --merges-path bpe_result/tiny/merge.pkl \
#     --config configs/small_model_config.json \
#     --prompt "The story begins" \
#     --temperature 0.7 \
#     --top-p 0.8 \
#     --max-tokens 150

# # Creative generation with multiple samples
# python -m cs336_basics.generate \
#     --checkpoint checkpoints/small_model/latest.pt \
#     --vocab-path bpe_result/tiny/vocab.pkl \
#     --merges-path bpe_result/tiny/merge.pkl \
#     --config configs/small_model_config.json \
#     --prompt "In a distant galaxy" \
#     --temperature 1.3 \
#     --top-p 0.9 \
#     --num-samples 5 \
#     --output-file creative_samples.txt

# # Reproducible generation
# python -m cs336_basics.generate \
#     --checkpoint checkpoints/small_model/latest.pt \
#     --vocab-path bpe_result/tiny/vocab.pkl \
#     --merges-path bpe_result/tiny/merge.pkl \
#     --config configs/small_model_config.json \
#     --prompt "The mystery of" \
#     --seed 42 \
#     --temperature 1.0