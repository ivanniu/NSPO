# NSPO: Null-Space constrained Policy Optimization



## Project Structure

```
NSPO/
├── evaluation/
│   ├── eval_sorrybench.py
│   ├── evaluation.py
│   ├── evaluator_configs/
│   │   ├── configs.yaml
│   │   └── template_base.txt
│   └── generate_alpaca.py
├── script/
│   ├── alpaca_eval.sh
│   ├── livecodebench_eval.sh
│   ├── math_eval.sh
│   ├── merge_verl_ckpt.py
│   ├── mmlu_bench.sh
│   ├── safe_reward.py
│   ├── safe_rl_verl_rule_base.sh
│   ├── safety_eval.sh
│   ├── start_vllm_llama_guard.sh
│   └── superGPQA_eval.sh
└── verl/
```

---

## Quick Start

### Setup

1. **Download Required Assets**  
   - Get the 🔗 [**Qwen2.5-7B-Instruct**](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) model from Hugging Face.
   - Download the 🔗 [**PKU-SafeRLHF**](https://huggingface.co/datasets/PKU-Alignment/PKU-SafeRLHF) dataset from Hugging Face.

2. **Install Dependencies**
   ```bash
   cd verl
   pip install -r requirements.txt
   pip install .
   ```

3. **Configure Paths**  
   - Modify `model_path` and `dataset_path` in:
     ```
     NSPO/verl/verl/workers/fsdp_workers.py (lines 248–249)
     ```
   - In the script directory, update `DATA_PATH` and `NSPO_PATH` in:
     ```
     NSPO/script/nspo_verl_rule_base.sh
     ```
   - In the script directory, update `MODEL_PATH` in:
     ```
     NSPO/script/start_vllm_llama_guard.sh
     ```

4. **Launch Services and Training**
   ```bash
   cd script
   bash start_vllm_llama_guard.sh
   bash nspo_verl_rule_base.sh
   ```

---

## Evaluation

The `script/` and `evaluation/` directories contain benchmark evaluation scripts.  
**Note:** You need to manually configure the dataset paths and API keys in these scripts before running them.

---

## Pretrained Checkpoint

We provide a checkpoint trained with NSPO based on Qwen2.5-7B-Instruct:

🔗 [Qwen2.5-7B-Instruct-NSPO on Hugging Face](https://huggingface.co/ICLR2026NSPO/Qwen2.5-7B-Instruct-NSPO)
