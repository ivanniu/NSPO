model_path=""
lm_eval --model hf \
    --model_args pretrained=$model_path,dtype=bfloat16 \
    --tasks mmlu,gsm8k \
    --batch_size auto \
    --device cuda:0 \


