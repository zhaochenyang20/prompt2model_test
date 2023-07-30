from prompt2model.utils.openai_tools import ChatGPTAgent
from datasets import load_from_disk
import datasets
from pathlib import Path
import logging
import argparse
from prompt2model.model_evaluator import Seq2SeqEvaluator
from prompt2model.utils import OPENAI_ERRORS
import time
logging.basicConfig(level=logging.INFO)
from prompt2model.model_executor import ModelOutput

result_path = Path("/home/chenyan3/prompt2model_test/baseline/real_datasets/datasets")

def evaluate_with_gpt(task_name):
    test_dataset = load_from_disk(result_path / f"{task_name}_gpt_model")
    chat_api = ChatGPTAgent()
    outputs = []
    evaluate_length = 10
    # for idx in range(len(test_dataset)):
    for idx in range(evaluate_length):
        api_call_counter = 0
        model_input = test_dataset[idx]["model_input"]
        input_col = test_dataset[idx]["input_col"]
        while True:
            try:
                api_call_counter += 1
                if api_call_counter >= 4:
                    logging.info(f"index: {idx}")
                    logging.info(f"input: {input_col}")
                    logging.info(f"output: None")
                    outputs.append("")
                    break
                response = chat_api.generate_one_openai_chat_completion(model_input)
                output = response.choices[0]["message"]["content"]
                outputs.append(output)
                logging.info(f"index: {idx}")
                logging.info(f"input: {input_col}")
                logging.info(f"output: {output}")
                break
            except OPENAI_ERRORS as e:
                logging.error(e)
                time.sleep(1)
    result_dataset = datasets.Dataset.from_dict(
        {
            "input_col": test_dataset["input_col"][:evaluate_length],
            "model_input": test_dataset["model_input"][:evaluate_length],
            "output_col": test_dataset["output_col"][:evaluate_length],
            "model_output": test_dataset["model_output"][:evaluate_length],
            "output": outputs,
        }
    )
    result_root = Path("results")
    result_dataset.save_to_disk(result_root / task_name)
    GPT_PREDICTIONS = [
    ModelOutput(
        f"{example['output']}", auxiliary_info={}
    ) for example in result_dataset
]
    evaluator = Seq2SeqEvaluator()
    metric_values = evaluator.evaluate_model(
        test_dataset, "model_output", GPT_PREDICTIONS, encoder_model_name="xlm-roberta-base"
    )
    print(metric_values)
    RESULT_PATH = Path("result")
    RESULT_PATH.mkdir(parents=True, exist_ok=True)
    with open(RESULT_PATH / f"{task_name}.txt", "w") as result_file:
        result_file.write(f"task_name: {task_name}\n")
        for metric_name, metric_value in metric_values.items():
            result_file.write(f"{metric_name}: {metric_value}\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate GPT model.")
    parser.add_argument(
        "--task_name", type=str, required=True, help="Name of the task."
    )
    args = parser.parse_args()

    evaluate_with_gpt(args.task_name)


if __name__ == "__main__":
    main()