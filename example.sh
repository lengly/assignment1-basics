python -m cs336_basics.train --config configs/small_model_config.json

python -m cs336_basics.generate \
    --checkpoint checkpoints/small_model/latest.pt \
    --vocab-path bpe_result/tiny/vocab.pkl \
    --merges-path bpe_result/tiny/merge.pkl \
    --config configs/small_model_config.json \
    --prompt "The story begins" \
    --temperature 0.7 \
    --top-p 0.8 \
    --max-tokens 150

# Creative generation with multiple samples
python -m cs336_basics.generate \
    --checkpoint checkpoints/small_model/latest.pt \
    --vocab-path bpe_result/tiny/vocab.pkl \
    --merges-path bpe_result/tiny/merge.pkl \
    --config configs/small_model_config.json \
    --prompt "In a distant galaxy" \
    --temperature 1.3 \
    --top-p 0.9 \
    --num-samples 5 \
    --output-file creative_samples.txt

# Reproducible generation
python -m cs336_basics.generate \
    --checkpoint checkpoints/small_model/latest.pt \
    --vocab-path bpe_result/tiny/vocab.pkl \
    --merges-path bpe_result/tiny/merge.pkl \
    --config configs/small_model_config.json \
    --prompt "The mystery of" \
    --seed 42 \
    --temperature 1.0