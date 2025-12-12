export CUDA_VISIBLE_DEVICES=""
model_name=""
### Follow LiveCodeBench official repository
python -m lcb_runner.runner.main --model $model_name --scenario codegeneration --evaluate --release_version release_v6