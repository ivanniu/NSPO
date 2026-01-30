# NSPO: Null-Space constrained Policy Optimization for Safety Alignment


This is the **official implementation** of the [**paper**](https://arxiv.org/abs/2512.11391):  
**"Mitigating the Safety Alignment Tax with Null-Space Constrained Policy Optimization"** (ICLR 2026)

---

## 💡 Overview

Null-Space constrained Policy Optimization (NSPO) is a RL framework for LLM safety alignment while preserving their core abilities.  Notably, NSPO is data-efficient and only requires 40\% of public human-annotated safety data from PKU-SafeRLHF to achieve promising performance.

---

## 📂 Project Structure

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

## 🚀 Quick Start

### 1. Setup & Environment

**Download Assets:**
* **Base Model:** [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
* **Dataset:** [PKU-SafeRLHF](https://huggingface.co/datasets/PKU-Alignment/PKU-SafeRLHF)

**Installation:**
```bash
cd NSPO/verl
pip install -r requirements.txt
pip install .
```
### 2. Configuration
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

### 3. Launch Services and Training
   ```bash
   cd script
   bash start_vllm_llama_guard.sh
   bash nspo_verl_rule_base.sh
   ```

---

## 📊 Evaluation

The `script/` and `evaluation/` directories contain benchmark evaluation scripts.  
**Note:** You need to manually configure the dataset paths and API keys in these scripts before running them.

---

## 🤗 Models & Checkpoints

We have released our trained checkpoint on Hugging Face:

* [Qwen2.5-7B-Instruct-NSPO](https://huggingface.co/ICLR2026NSPO/Qwen2.5-7B-Instruct-NSPO): The policy model fine-tuned using NSPO on the Qwen2.5-7B-Instruct.

---

## ✍️ Citation

If you find this work helpful for your research, please cite our paper:
```
@article{niu2025mitigating,
  title={Mitigating the Safety Alignment Tax with Null-Space Constrained Policy Optimization},
  author={Niu, Yifan and Xiao, Han and Liu, Dongyi and Chen, Nuo and Li, Jia},
  journal={arXiv preprint arXiv:2512.11391},
  year={2025}
}
```
