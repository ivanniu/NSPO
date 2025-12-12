export CUDA_VISIBLE_DEVICES=
export PYTHONPATH=$(pwd)
model_name=""

### Follow SuperGPQA official repository
python infer/infer.py --config config/config_default.yaml --split SuperGPQA-all --mode zero-shot --model_name $model_name --output_dir results --batch_size 1000 --use_accel --index 0 --world_size 1
python eval/eval.py --evaluate_all --excel_output --json_output --output_dir results --save_dir results_with_status