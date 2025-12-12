
from tqdm import tqdm
import json
import argparse
from transformers import AutoTokenizer
import os
import datasets
import argparse
import json
from tqdm import tqdm
from vllm import LLM, SamplingParams
DEFAULT_CHAT_TEMPLATE = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"

def model_batch_responses_vllm(llm, questions, batch_size=10):
   
    responses = []
    sampling_params = SamplingParams(max_tokens=512)
    outputs = llm.generate(questions, sampling_params)
    batch_responses = [output.outputs[0].text for output in outputs]
    responses.extend(batch_responses)
    return responses

def classify_and_analyze_batch_vllm(dataset, llm, tokenizer, model_name, batch_size=10):

    results = []
    for i in tqdm(range(0, len(dataset), batch_size), desc="Generate responses in batches"):
        batch = dataset[i:i + batch_size]
        questions = []
        for question in batch:
            result = [{'role': "system", "content": ""},{'role': "user", "content": question["instruction"]}]
            prompt = tokenizer.apply_chat_template(
                result,
                tokenize=False,
                add_generation_prompt=True
            )
            questions.append(prompt)
        try:
            batch_responses = model_batch_responses_vllm(llm, questions, batch_size)
           
            for item, response in zip(batch, batch_responses):
                results.append({"dataset":item["dataset"], "instruction": item["instruction"], "output": response,"generator":model_name})
              
        except Exception as e:
            print(f"Error processing batch starting at index {i}\n{e}")
            continue
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify and analyze questions in a dataset using vllm.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model directory.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the output JSON file.")
    parser.add_argument("--batch_size", type=int, default=10, help="Batch size for processing questions.")
    parser.add_argument('--device_ids', type=int, nargs='+', default=[1],
                       help='GPU --device_ids 0 3')
    parser.add_argument('--model_name', type=str, default="Meta-Llama-3-8B-Instruct",
                       help='Name of the model to use')
    args = parser.parse_args()

    MODEL_PATH = args.model_path
    OUTPUT_PATH = args.output_path
    BATCH_SIZE = args.batch_size
    MODEL_NAME = args.model_name
    device_ids = args.device_ids
    gpu_ids_str = ','.join(str(id) for id in device_ids)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids_str
    eval_set = datasets.load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval")["eval"].to_list()
    llm = LLM(model=MODEL_PATH, tensor_parallel_size=len(device_ids))
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.chat_template is None:
        tokenizer.chat_template = DEFAULT_CHAT_TEMPLATE

    results = classify_and_analyze_batch_vllm(eval_set, llm, tokenizer, MODEL_NAME, batch_size=BATCH_SIZE)
   

    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Results saved to {OUTPUT_PATH}")