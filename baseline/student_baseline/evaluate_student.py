from datasets import load_from_disk
from pathlib import Path
import torch
import logging
import argparse
import transformers
from prompt2model.model_evaluator import Seq2SeqEvaluator
from prompt2model.model_executor import GenerationModelExecutor
from prompt2model.utils.path import TEST_DATA_ROOT

logging.basicConfig(level=logging.INFO)


def evaluate(model_name, task_name):
    logging.info(f"model: {model_name}, task: {task_name}")
    model_store_name = model_name.split("/")[-1]
    TRAINED_MODEL_ROOT = Path("/home/chenyan3/beta-test/train/result/trained_model")
    TRAINED_TOKENIZER_ROOT = Path(
        "/home/chenyan3/beta-test/train/result/trained_tokenizer"
    )
    DATASET_DICTS_STORE_ROOT = Path(
        TEST_DATA_ROOT+"/prompt2model_test/baseline/real_datasets/datasets"
    )
    RESULT_PATH = Path(f"./result")
    TRAINED_MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    TRAINED_TOKENIZER_ROOT.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.mkdir(parents=True, exist_ok=True)

    test_dataset = load_from_disk(
        DATASET_DICTS_STORE_ROOT / f"{task_name}_student_model"
    )["test"]
    t5_model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
        TRAINED_MODEL_ROOT / f"{model_store_name}_{task_name}"
    )
    t5_tokenizer = transformers.AutoTokenizer.from_pretrained(
        TRAINED_TOKENIZER_ROOT / f"{model_store_name}_{task_name}"
    )
    BATCH_SIZE = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_executor = GenerationModelExecutor(t5_model.to(device), t5_tokenizer, BATCH_SIZE)
    t5_outputs = model_executor.make_prediction(
        test_set=test_dataset, input_column="model_input"
    )
    evaluator = Seq2SeqEvaluator()
    metric_values = evaluator.evaluate_model(
        test_dataset, "model_output", t5_outputs, encoder_model_name="xlm-roberta-base"
    )
    print(metric_values)
    with open(RESULT_PATH / f"{model_store_name}_{task_name}.txt", "w") as result_file:
        result_file.write(f"model_name: {model_store_name}\n")
        result_file.write(f"task_name: {task_name}\n")
        result_file.write(f"batch_size: {BATCH_SIZE}\n")
        for metric_name, metric_value in metric_values.items():
            result_file.write(f"{metric_name}: {metric_value}\n")


def main():
    parser = argparse.ArgumentParser(description="Train the generation model.")
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Name or path of the pre-trained model to use.",
    )
    parser.add_argument(
        "--task_name", type=str, required=True, help="Name of the task."
    )
    args = parser.parse_args()

    evaluate(args.model_name, args.task_name)


if __name__ == "__main__":
    main()
