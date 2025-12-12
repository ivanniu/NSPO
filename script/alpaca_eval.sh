
MODEL_NAME=""
MODEL_PATH=""

OUTPUT_PATH="SafeRL/results/alpaca/${MODEL_NAME}.json"
REFERENCE_PATH="" ## compare the output path
LLM_JUDGE_PATH="SafeRL/evaluation/evaluator_configs/configs.yaml"
SAVE_PATH="SafeRL/results/alpaca/alpaca_${MODEL_NAME}"

python SafeRL/evaluation/generate_alpaca.py \
    --model_path $MODEL_PATH \
    --output_path $OUTPUT_PATH \
    --batch_size 128 \
    --device_ids 6 \
    --model_name $MODEL_NAME \


export OPENAI_API_KEY=""
export OPENAI_API_BASE=""
export OPENAI_ENGINE=""
alpaca_eval --model_outputs $OUTPUT_PATH \
    --reference_outputs $REFERENCE_PATH \
    --annotators_config $LLM_JUDGE_PATH \
    --output_path $SAVE_PATH