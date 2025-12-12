import argparse
parser = argparse.ArgumentParser(description="Classify and analyze questions in a dataset using vllm.")
parser.add_argument("--data_path", type=str, required=True, help="Path to the input JSON file.")
parser.add_argument("--model_name", type=str, required=True, help="model_name")
args = parser.parse_args()
data_path = args.data_path
model_name = args.model_name 

import json
with open(data_path, "r") as f:
    raw_judge_results = [json.loads(line) for line in f]
    
def avg(L):
    return sum(L) / len(L)

judge_results = dict()
judge_results[model_name] = [[] for i in range(44)]

for result in raw_judge_results:
    category_id = (result["question_id"] - 1) // 10
    if result["model"] in judge_results.keys():
        judge_results[result["model"]][category_id].append(result["score"])

avg_score_overall = dict()
judge_avg_scores_per_category = dict()
for model in judge_results:
    judge_avg_scores_per_category[model] = []
    for category in range(len(judge_results[model])):
        judge_avg_scores_per_category[model].append(avg(judge_results[model][category]))
    
    avg_score_overall[model] = avg(judge_avg_scores_per_category[model])

sorted_models = sorted(avg_score_overall, key=avg_score_overall.get, reverse=False)
for model in sorted_models:
    print(f"{model}: {avg_score_overall[model]:.4f}")