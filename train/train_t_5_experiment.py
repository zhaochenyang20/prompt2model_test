from datasets import load_from_disk
from pathlib import Path
import datasets
from prompt2model.model_trainer.generate import GenerationModelTrainer
from prompt2model.dataset_processor.textualize import TextualizeProcessor
from datasets import DatasetDict
from pathlib import Path
import logging
import argparse
import transformers
from prompt2model.model_evaluator import Seq2SeqEvaluator
from prompt2model.model_executor import GenerationModelExecutor
logging.basicConfig(level=logging.INFO)


def train(model_name, task_name, evaluate=True):
    # Read the CSV file using pandas
    logging.info(f"model: {model_name}, task: {task_name}")
    model_store_name = model_name.split("/")[-1]
    DATASET_DICTS_STORE_ROOT = Path(f"./datasets/{model_store_name}_{task_name}")
    TRAINED_MODEL_ROOT = Path("./result/trained_model")
    TRAINED_TOKENIZER_ROOT = Path("./result/trained_tokenizer")
    RESULT_PATH = Path(f"./result/{model_store_name}_{task_name}")
    DATASET_DICTS_STORE_ROOT.mkdir(parents=True, exist_ok=True)
    TRAINED_MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    TRAINED_TOKENIZER_ROOT.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.mkdir(parents=True, exist_ok=True)
    dataset = load_from_disk(f"./datasets/{task_name}")
    train_dataset = datasets.Dataset.from_dict(dataset[:3000])
    val_dataset = datasets.Dataset.from_dict(dataset[3000:4000])
    test_dataset = datasets.Dataset.from_dict(dataset[4000:5000])

    # Create the DatasetDict
    dataset_dict = DatasetDict(
        {"train": train_dataset, "val": val_dataset, "test": test_dataset}
    )

    DATASET_DICTS = [dataset_dict]

    if task_name == "normalization":
        INSTRUCTION = """People often uses some temporal date expression in dalegies. I want to know the exact date of all the temporal date expression in some sentences.

    For this task, the input is a string contains two specific elements: a posted date as "[Posted: YYYY-MM-DD]" and a sentence or statement with a temporal date reference to a time period (e.g., early December, the end of the year, July, August, last Christmas, next Month, etc).

    The output is a string that provides a mapping between the time period references mentioned in the input and the corresponding dates. The output uses the "==" symbol to show the relationship, with the time period reference on the left and the corresponding date on the right. The date is formatted as "YYYY-MM" to represent the year and month.
    """
    elif task_name == "NQ":
        INSTRUCTION = """
    Your task is to generate an answer to a natural question. In this task, the input is a question string. and the output is the corresponding answer string. The question can range from Math, Cultural, Social, Geometry, Biology, History, Sports, Technology, Science, and so on.
    """
    elif task_name == "Chinese2SQL":
        INSTRUCTION = """Chinese2SQL is an NLP task that involves converting natural language queries written in Chinese into SQL queries for querying relational databases.

For this task, the input is a Chinese string that describes a natural language query. The output is the corresponding SQL query.
"""
    t5_processor = TextualizeProcessor(has_encoder=True)
    t5_modified_dataset_dicts = t5_processor.process_dataset_dict(
        INSTRUCTION, DATASET_DICTS
    )
    t5_modified_dataset_dicts[0].save_to_disk(DATASET_DICTS_STORE_ROOT)
    training_datasets = [t5_modified_dataset_dicts[0]["train"]]
    validation_datasets = [t5_modified_dataset_dicts[0]["val"]]
    trainer = GenerationModelTrainer(
        model_name, has_encoder=True, tokenizer_max_length=1024
    )
    # model_max_length 会限制 sentence 的长度，可能会丢失一些特征
    args_output_root = Path(f"./result/training_output/{model_store_name}_{task_name}")
    args_output_root.mkdir(parents=True, exist_ok=True)
    trained_model, trained_tokenizer = trainer.train_model(
        {
            "output_dir": str(args_output_root),
            "num_train_epochs": 10,
            "per_device_train_batch_size": 8,
            "evaluation_strategy": "epoch",
        },
        training_datasets,
        validation_datasets,
    )

    trained_model.save_pretrained(
        TRAINED_MODEL_ROOT / f"{model_store_name}_{task_name}"
    )
    trained_tokenizer.save_pretrained(
        TRAINED_TOKENIZER_ROOT / f"{model_store_name}_{task_name}"
    )
    if evaluate:
        test_dataset = load_from_disk(DATASET_DICTS_STORE_ROOT)["test"]
        t5_model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
TRAINED_MODEL_ROOT / f"{model_store_name}_{task_name}"
        )
        BATCH_SIZE = 4
        t5_tokenizer = transformers.AutoTokenizer.from_pretrained(TRAINED_TOKENIZER_ROOT / f"{model_store_name}_{task_name}")
        model_executor = GenerationModelExecutor(
            t5_model, t5_tokenizer, BATCH_SIZE
        )
        t5_outputs = model_executor.make_prediction(test_set=test_dataset, input_column="model_input")
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

    train(args.model_name, args.task_name)


if __name__ == "__main__":
    main()
