export CUDA_VISIBLE_DEVICES=""

BATCH_SIZE=128
DEVICE_IDS=""
MODEL_NAME=""
MODEL_PATH="/path to your model/${MODEL_NAME}"
JUDGE_MODEL_PATH="Llama-Guard-3-8B"
export OPENAI_API_KEY=""
export OPENAI_API_BASE=""
export OPENAI_ENGINE=""

########################## sorry-bench ##############################

python sorry-bench/gen_model_answer_vllm.py --bench-name sorry_bench --model-path $MODEL_PATH --model-id $MODEL_NAME
python sorry-bench/gen_judgment_safety_vllm.py --model-list $MODEL_NAME 
python SafeRL/evaluation/eval_sorrybench.py --data_path "sorry-bench/data/sorry_bench/model_judgment/ft-mistral-7b-instruct-v0.2.jsonl" --model_name $MODEL_NAME

########################## sorry-bench ##############################

###################### JailbreakBench ######################
DATA_PATH="SafeRL/SafetyBench/JailbreakBench/jailbreakbench.jsonl"
OUTPUT_PATH="SafeRL/SafetyBench/JailbreakBench/${MODEL_NAME}/JailbreakBench_output_${MODEL_NAME}.json"
KEY="goal"
echo "Processing dataset: JailbreakBench"
python SafeRL/evaluation/evaluation.py\
    --data_path "$DATA_PATH" \
    --model_path "$MODEL_PATH" \
    --batch_size "$BATCH_SIZE" \
    --output_path "$OUTPUT_PATH" \
    --key "$KEY"\
    --device_ids $DEVICE_IDS \
    --dataset_type "JailbreakBench" \
    --type "generation"

DATA_PATH="SafeRL/SafetyBench/JailbreakBench/${MODEL_NAME}/JailbreakBench_output_${MODEL_NAME}.json"
OUTPUT_PATH="SafeRL/SafetyBench/JailbreakBench/${MODEL_NAME}/JailbreakBench_score_${MODEL_NAME}.json"
KEY="prompt"
echo "Processing dataset: JailbreakBench"
python SafeRL/evaluation/evaluation.py\
   --data_path "$DATA_PATH" \
    --batch_size "$BATCH_SIZE" \
    --output_path "$OUTPUT_PATH" \
    --device_ids $DEVICE_IDS \
    --dataset_type "JailbreakBench" \
    --guard_model_path $JUDGE_MODEL_PATH\
    --eval_type "Guard" \
    --type "eval" \
    --model_path "$MODEL_PATH" \
    --key "$KEY"\
###################### JailbreakBench ######################

###################### ALERT ######################
DATA_PATH="SafeRL/SafetyBench/ALERT/alert.jsonl"
OUTPUT_PATH="SafeRL/SafetyBench/ALERT/${MODEL_NAME}/ALERT_output_${MODEL_NAME}.json"
KEY="prompt"
echo "Processing dataset: ALERT"
python SafeRL/evaluation/evaluation.py\
    --data_path "$DATA_PATH" \
    --model_path "$MODEL_PATH" \
    --batch_size "$BATCH_SIZE" \
    --output_path "$OUTPUT_PATH" \
    --key "$KEY"\
    --device_ids $DEVICE_IDS \
    --dataset_type "ALERT" \
    --type "generation"

DATA_PATH="SafeRL/SafetyBench/ALERT/${MODEL_NAME}/ALERT_output_${MODEL_NAME}.json"
OUTPUT_PATH="SafeRL/SafetyBench/ALERT/${MODEL_NAME}/ALERT_score_${MODEL_NAME}.json"
KEY="prompt"
echo "Processing dataset: ALERT"
python SafeRL/evaluation/evaluation.py\
   --data_path "$DATA_PATH" \
    --batch_size "$BATCH_SIZE" \
    --output_path "$OUTPUT_PATH" \
    --device_ids $DEVICE_IDS \
    --dataset_type "ALERT" \
    --guard_model_path $JUDGE_MODEL_PATH\
    --eval_type "Guard" \
    --type "eval" \
    --model_path "$MODEL_PATH" \
    --key "$KEY"\
###################### ALERT ######################

###################### AdvBench ######################

DATA_PATH="SafeRL/SafetyBench/AdvBenchfull_safety.json"
OUTPUT_PATH="SafeRL/SafetyBench/AdvBench/${MODEL_NAME}/AdvBenchfull_output_${MODEL_NAME}.json"
KEY="prompt"
echo "Processing dataset: Advbench"
python SafeRL/evaluation/evaluation.py\
    --data_path "$DATA_PATH" \
    --model_path "$MODEL_PATH" \
    --batch_size "$BATCH_SIZE" \
    --output_path "$OUTPUT_PATH" \
    --key "$KEY"\
    --device_ids $DEVICE_IDS \
    --dataset_type "AdvBench" \
    --type "generation"

DATA_PATH="SafeRL/SafetyBench/AdvBench/${MODEL_NAME}/AdvBenchfull_output_${MODEL_NAME}.json"
OUTPUT_PATH="SafeRL/SafetyBench/AdvBench/${MODEL_NAME}/AdvBenchfull_score_${MODEL_NAME}.json"
KEY="prompt"
echo "Processing dataset: Advbench"
python SafeRL/evaluation/evaluation.py\
   --data_path "$DATA_PATH" \
    --model_path "$MODEL_PATH" \
    --batch_size 1 \
    --output_path "$OUTPUT_PATH" \
    --key "$KEY"\
    --device_ids $DEVICE_IDS \
    --dataset_type "AdvBench" \
    --type "eval"

###################### AdvBench ######################

###################### HarmBench ######################
DATA_PATH="SafeRL/SafetyBench/HarmBenchfull_safety.json"
OUTPUT_PATH="SafeRL/SafetyBench/HarmBench/${MODEL_NAME}/HarmBenchfull_output_${MODEL_NAME}.json"
KEY="prompt"
echo "Processing dataset: HarmBench"
python SafeRL/evaluation/evaluation.py\
    --data_path "$DATA_PATH" \
    --model_path "$MODEL_PATH" \
    --batch_size "$BATCH_SIZE" \
    --output_path "$OUTPUT_PATH" \
    --key "$KEY"\
    --device_ids $DEVICE_IDS \
    --dataset_type "HarmBench" \
    --type "generation"


DATA_PATH="SafeRL/SafetyBench/HarmBench/${MODEL_NAME}/HarmBenchfull_output_${MODEL_NAME}.json"
OUTPUT_PATH="SafeRL/SafetyBench/HarmBench/${MODEL_NAME}/HarmBenchfull_score_${MODEL_NAME}.json"
KEY="prompt"
echo "Processing dataset: HarmBench"
python SafeRL/evaluation/evaluation.py\
    --data_path "$DATA_PATH" \
    --model_path "$MODEL_PATH" \
    --batch_size 1 \
    --output_path "$OUTPUT_PATH" \
    --key "$KEY"\
    --device_ids $DEVICE_IDS \
    --dataset_type "HarmBench" \
    --type "eval"
###################### HarmBench ######################

###################### HarmfulQA ######################
DATA_PATH="SafeRL/SafetyBench/HarmfulQAfull_safety.json"
OUTPUT_PATH="SafeRL/SafetyBench/HarmfulQA/${MODEL_NAME}/HarmfulQAfull_output_${MODEL_NAME}.json"
KEY="question"
echo "Processing dataset: HarmfulQA"
python SafeRL/evaluation/evaluation.py\
    --data_path "$DATA_PATH" \
    --model_path "$MODEL_PATH" \
    --batch_size "$BATCH_SIZE" \
    --output_path "$OUTPUT_PATH" \
    --key "$KEY"\
    --device_ids $DEVICE_IDS \
    --dataset_type "HarmfulQA" \
    --type "generation"
    
DATA_PATH="SafeRL/SafetyBench/HarmfulQA/${MODEL_NAME}/HarmfulQAfull_output_${MODEL_NAME}.json"
OUTPUT_PATH="SafeRL/SafetyBench/HarmfulQA/${MODEL_NAME}/HarmfulQAfull_score_${MODEL_NAME}.json"
KEY="question"

echo "Processing dataset: HarmfulQA"
python SafeRL/evaluation/evaluation.py\
    --data_path "$DATA_PATH" \
    --model_path "$MODEL_PATH" \
    --batch_size 1 \
    --output_path "$OUTPUT_PATH" \
    --key "$KEY"\
    --device_ids $DEVICE_IDS \
    --dataset_type "HarmfulQA" \
    --type "eval"
###################### HarmfulQA ######################


###################### SafeRLHF ######################
DATA_PATH="SafeRL/SafetyBench/SafeRLHFfull_test_safety.json"
OUTPUT_PATH="SafeRL/SafetyBench/SafeRLHF/${MODEL_NAME}/SafeRLHF_output_${MODEL_NAME}.json"
KEY="prompt"
echo "Processing dataset: SafeRLHF"
python SafeRL/evaluation/evaluation.py\
    --data_path "$DATA_PATH" \
    --model_path "$MODEL_PATH" \
    --batch_size "$BATCH_SIZE" \
    --output_path "$OUTPUT_PATH" \
    --key "$KEY"\
    --device_ids $DEVICE_IDS \
    --dataset_type "SafeRLHF" \
    --type "generation"

DATA_PATH="SafeRL/SafetyBench/SafeRLHF/${MODEL_NAME}/SafeRLHF_output_${MODEL_NAME}.json"
OUTPUT_PATH="SafeRL/SafetyBench/SafeRLHF/${MODEL_NAME}/SafeRLHF_score_${MODEL_NAME}.json"
KEY="prompt"
echo "Processing dataset: SafeRLHF"
python SafeRL/evaluation/evaluation.py\
    --data_path "$DATA_PATH" \
    --model_path "$MODEL_PATH" \
    --batch_size 1 \
    --output_path "$OUTPUT_PATH" \
    --key "$KEY"\
    --device_ids $DEVICE_IDS \
    --dataset_type "SafeRLHF" \
    --type "eval"
###################### SafeRLHF ######################




