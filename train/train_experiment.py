from datasets import load_from_disk
from pathlib import Path
import datasets
from prompt2model.model_trainer.generate import GenerationModelTrainer
from prompt2model.dataset_processor.textualize import TextualizeProcessor
from datasets import DatasetDict
from pathlib import Path
import logging
import torch
import argparse
import transformers
from prompt2model.model_evaluator import Seq2SeqEvaluator
from prompt2model.model_executor import GenerationModelExecutor, ModelOutput
from prompt2model.utils.path import TEST_DATA_ROOT

logging.basicConfig(level=logging.INFO)


def train(model_name, task_name, evaluate=True, realistic=True):
    # Read the CSV file using pandas
    logging.info(f"model: {model_name}, task: {task_name}")
    model_store_name = model_name.split("/")[-1]
    dataset_root = Path("../generation/generated_dataset/")
    assert dataset_root.exists()
    DATASET_DICTS_STORE_ROOT = dataset_root / f"{model_store_name}_{task_name}"
    TRAINED_MODEL_ROOT = Path("/home/chenyan3/result/trained_model")
    TRAINED_TOKENIZER_ROOT = Path("/home/chenyan3/result/trained_tokenizer")
    RESULT_PATH = Path(f"/home/chenyan3/result/{model_store_name}_{task_name}")
    DATASET_DICTS_STORE_ROOT.mkdir(parents=True, exist_ok=True)
    TRAINED_MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    TRAINED_TOKENIZER_ROOT.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.mkdir(parents=True, exist_ok=True)
    dataset = load_from_disk(dataset_root / task_name)
    train_dataset = datasets.Dataset.from_dict(dataset[:3000])
    val_dataset = datasets.Dataset.from_dict(dataset[3000:4000])
    test_dataset = datasets.Dataset.from_dict(dataset[4000:5000])

    # Create the DatasetDict
    dataset_dict = DatasetDict(
        {"train": train_dataset, "val": val_dataset, "test": test_dataset}
    )

    DATASET_DICTS = [dataset_dict]

    if "normalization" in task_name:
        INSTRUCTION = """Temporal date expressions are commonly used to refer to specific time periods. Your task is to identify these temporal date expressions and provide the exact dates they refer to.

For this task, the input is a string containing two specific elements: a posted date in the format "[Posted: YYYY-MM-DD]" and a sentence or statement that contains various temporal date references (e.g., early December, the end of the year, today, August, last Christmas, next Month, etc).

Your program should output a string that maps the time period references mentioned in the input to their corresponding dates, following these strict rules:

1. If temporal date references are found, the output should use either "YYYY-MM-DD", "YYYY-MM", or "YYYY" to represent the exact date.
- If multiple time period references are found, separate them using '|'.
2. If no temporal date reference is found or the referred date is ambiguous, the output should just be 'N/A', i.e., output="N/A".
"""
    elif task_name == "NQ":
        INSTRUCTION = """
    Your task is to generate an answer to a natural question. In this task, the input is a question string. and the output is the corresponding answer string. The question can range from Math, Cultural, Social, Geometry, Biology, History, Sports, Technology, Science, and so on.
    """
    elif task_name == "Chinese2SQL":
        INSTRUCTION = """Chinese2SQL is an NLP task that involves converting natural language queries written in Chinese into SQL queries for querying relational databases.

For this task, the input is a Chinese string that describes a natural language query. The output is the corresponding SQL query.
"""
    elif "SQuAD" in task_name:
        INSTRUCTION = """Your task is to generate an answer to a natural question. In this task, the input is a string that consists of both a question and a context passage. The context is a descriptive passage related to the question and contains the answer. And the question can range from Math, Cultural, Social, Geometry, Biology, History, Sports, Technology, Science, and so on."""
    elif task_name == "jp2python":
        INSTRUCTION = """Pythonで1行のコードを生成し、StackOverflowの日本語の質問を解決してください。コメントや式は含めないでください。インポート文も不要です。

このタスクでは、入力は日本語のテキストで、変数名や操作が記述されています。出力は、そのタスクを達成するためのPythonの1行のコードです。コメントや式は含めないでください。インポート文も不要です。
"""
    t5_processor = TextualizeProcessor(has_encoder=True)
    t5_modified_dataset_dicts = t5_processor.process_dataset_dict(
        INSTRUCTION, DATASET_DICTS
    )
    t5_modified_dataset_dicts[0].save_to_disk(DATASET_DICTS_STORE_ROOT)
    training_datasets = [t5_modified_dataset_dicts[0]["train"]]
    validation_datasets = [t5_modified_dataset_dicts[0]["val"]]
    trainer = GenerationModelTrainer(
        model_name,
        has_encoder=True,
        executor_batch_size=4,
        tokenizer_max_length=1024,
        sequence_max_length=1280,
    )
    # model_max_length 会限制 sentence 的长度，可能会丢失一些特征
    args_output_root = Path(
        f"/home/chenyan3/result/training_output/{model_store_name}_{task_name}"
    )
    args_output_root.mkdir(parents=True, exist_ok=True)
    trained_model, trained_tokenizer = trainer.train_model(
        hyperparameter_choices={
            "output_dir": str(args_output_root),
            "num_train_epochs": 10,
            "per_device_train_batch_size": 4,
            "evaluation_strategy": "epoch",
        },
        training_datasets=training_datasets,
        validation_datasets=validation_datasets,
    )

    trained_model.save_pretrained(
        TRAINED_MODEL_ROOT / f"{model_store_name}_{task_name}"
    )
    trained_tokenizer.save_pretrained(
        TRAINED_TOKENIZER_ROOT / f"{model_store_name}_{task_name}"
    )
    if evaluate:
        dataset_dict = load_from_disk(DATASET_DICTS_STORE_ROOT)
        test_dataset = dataset_dict["test"]
        BATCH_SIZE = 4
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        t5_model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
            TRAINED_MODEL_ROOT / f"{model_store_name}_{task_name}"
        ).to(device)
        t5_tokenizer = transformers.AutoTokenizer.from_pretrained(
            TRAINED_TOKENIZER_ROOT / f"{model_store_name}_{task_name}"
        )
        model_executor = GenerationModelExecutor(
            t5_model,
            t5_tokenizer,
            BATCH_SIZE,
            tokenizer_max_length=1024,
            sequence_max_length=1280,
        )

        # No post-filter
        t5_outputs = model_executor.make_prediction(
            test_set=test_dataset, input_column="model_input"
        )
        test_dataset = datasets.Dataset.from_dict(
            {
                "input_col": test_dataset["input_col"],
                "output_col": test_dataset["output_col"],
                "model_input": test_dataset["model_input"],
                "model_output": test_dataset["model_output"],
                "output": [each.prediction for each in t5_outputs],
            }
        )
        test_dataset.save_to_disk(
            f"{str(DATASET_DICTS_STORE_ROOT)}_generated_test_dataset_without_post_filter"
        )
        evaluator = Seq2SeqEvaluator()
        metric_values = evaluator.evaluate_model(
            test_dataset,
            "model_output",
            t5_outputs,
            encoder_model_name="xlm-roberta-base",
        )
        print("generated test set")
        print(metric_values)
        with open(
            RESULT_PATH
            / f"{model_store_name}_{task_name}_generated_dataset_without_post_filter.txt",
            "w",
        ) as result_file:
            result_file.write(f"model_name: {model_store_name}\n")
            result_file.write(f"task_name: {task_name}\n")
            result_file.write(f"batch_size: {BATCH_SIZE}\n")
            for metric_name, metric_value in metric_values.items():
                result_file.write(f"{metric_name}: {metric_value}\n")

    if realistic:
        realistic_dataset_root = Path(
            TEST_DATA_ROOT + "/prompt2model_test/baseline/real_datasets/datasets"
        )
        print(str(realistic_dataset_root))
        if "SQuAD" in task_name:
            real_task_name = "SQuAD"
        elif "normalization" in task_name:
            real_task_name = "normalization"
        elif "jp2python" in task_name:
            real_task_name = "jp2python"
        test_dataset = load_from_disk(
            realistic_dataset_root / f"{real_task_name}_student_model"
        )["test"]
        BATCH_SIZE = 4
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        t5_model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
            TRAINED_MODEL_ROOT / f"{model_store_name}_{task_name}"
        ).to(device)
        t5_tokenizer = transformers.AutoTokenizer.from_pretrained(
            TRAINED_TOKENIZER_ROOT / f"{model_store_name}_{task_name}"
        )
        model_executor = GenerationModelExecutor(
            t5_model,
            t5_tokenizer,
            BATCH_SIZE,
            tokenizer_max_length=1024,
            sequence_max_length=1280,
        )

        # No post-filter
        t5_outputs = model_executor.make_prediction(
            test_set=test_dataset, input_column="model_input"
        )
        test_dataset = datasets.Dataset.from_dict(
            {
                "input_col": test_dataset["input_col"],
                "output_col": test_dataset["output_col"],
                "model_input": test_dataset["model_input"],
                "model_output": test_dataset["model_output"],
                "output": [each.prediction for each in t5_outputs],
            }
        )
        test_dataset.save_to_disk(
            f"{str(DATASET_DICTS_STORE_ROOT)}_real_dataset_without_post_filter"
        )
        evaluator = Seq2SeqEvaluator()
        metric_values = evaluator.evaluate_model(
            test_dataset,
            "model_output",
            t5_outputs,
            encoder_model_name="xlm-roberta-base",
        )
        print("real evaluation:")
        print(metric_values)
        if "SQuAD" in task_name:
            from datasets import load_dataset

            original_dataset = load_dataset("squad", split="validation")
            counter = 0
            for idx, each in enumerate(test_dataset["output"]):
                if each in original_dataset[idx]["answers"]["text"]:
                    counter += 1
            exact_match = counter / len(original_dataset)
            print(f"exact_match: {exact_match}")


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

    train(args.model_name, args.task_name)


if __name__ == "__main__":
    main()
