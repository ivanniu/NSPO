PORT=52001
HOST="127.0.0.1"
GPU_MEMORY_UTILIZATION=0.6
TENSOR_PARALLEL_SIZE=2
MAX_MODEL_LEN=3072
MODEL_PATH=models/Llama-Guard-4-12B


echo "Start vLLM Service..."
echo "Model Path: $MODEL_PATH"
echo "Port: $PORT"
echo "GPU Num: $TENSOR_PARALLEL_SIZE"

CUDA_VISIBLE_DEVICES=4,5 python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --host "$HOST" \
    --port "$PORT" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    --max-model-len "$MAX_MODEL_LEN" \
    --served-model-name "$(basename "$MODEL_PATH")" \
    --trust-remote-code \
    --disable-log-requests \.