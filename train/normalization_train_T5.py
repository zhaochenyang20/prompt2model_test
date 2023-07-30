from datasets import load_from_disk
from pathlib import Path
import datasets
from os import listdir
from prompt2model.model_trainer.generate import GenerationModelTrainer
from transformers import T5ForConditionalGeneration, T5Tokenizer
from prompt2model.model_executor import GenerationModelExecutor
from prompt2model.model_evaluator import Seq2SeqEvaluator
from prompt2model.prompt_parser import MockPromptSpec, TaskType
from prompt2model.demo_creator import create_gradio
from prompt2model.dataset_processor.textualize import TextualizeProcessor
from datasets import DatasetDict, Dataset
from sklearn.model_selection import train_test_split
import pandas as pd
from pathlib import Path
from transformers import T5Tokenizer
import logging

logging.basicConfig(level=logging.INFO)


DATASET_DICTS_STORE_ROOT = Path("./normalization_T5_dataset_dict")
TRAINED_MODEL_ROOT = Path("./model/trained_model")
TRAINED_TOKENIZER_ROOT = Path("./model/trained_tokenizer")
RESULT_PATH = Path("./result_T5")
DATASET_DICTS_STORE_ROOT.mkdir(parents=True, exist_ok=True)
TRAINED_MODEL_ROOT.mkdir(parents=True, exist_ok=True)
TRAINED_TOKENIZER_ROOT.mkdir(parents=True, exist_ok=True)
RESULT_PATH.mkdir(parents=True, exist_ok=True)


def prepare_data():
    # Read the CSV file using pandas
    dataset = load_from_disk(
        "/home/chenyan3/beta-test/normalization/cached_genrated_dataset/normalization"
    )
    train_dataset = datasets.Dataset.from_dict(dataset[:1500])
    val_dataset = datasets.Dataset.from_dict(dataset[1500:2000])
    test_dataset = datasets.Dataset.from_dict(dataset[2000:])

    # Create the DatasetDict
    dataset_dict = DatasetDict(
        {"train": train_dataset, "val": val_dataset, "test": test_dataset}
    )

    # Access the data in the DatasetDict
    print(
        dataset_dict["train"]["input_col"][0]
    )  # Print the input of the first example in the training set
    print(
        dataset_dict["train"]["output_col"][0]
    )  # Print the output of the first example in the training set

    DATASET_DICTS = [dataset_dict]

    INSTRUCTION = """People often uses some temporal date expression in dalegies. I want to know the exact date of all the temporal date expression in some sentences.

For this task, the input is a string contains two specific elements: a posted date as "[Posted: YYYY-MM-DD]" and a sentence or statement with a temporal date reference to a time period (e.g., early December, the end of the year, July, August, last Christmas, next Month, etc).

The output is a string that provides a mapping between the time period references mentioned in the input and the corresponding dates. The output uses the "==" symbol to show the relationship, with the time period reference on the left and the corresponding date on the right. The date is formatted as "YYYY-MM" to represent the year and month.
"""
    t5_tokenizer = T5Tokenizer.from_pretrained("t5-base")
    t5_processor = TextualizeProcessor(
        has_encoder=True, eos_token=t5_tokenizer.eos_token
    )
    t5_modified_dataset_dicts = t5_processor.process_dataset_dict(
        INSTRUCTION, DATASET_DICTS
    )

    for index, dataset_dict in enumerate(t5_modified_dataset_dicts):
        store_path = DATASET_DICTS_STORE_ROOT / f"{index}"
        dataset_dict.save_to_disk(str(store_path))


def train(model_name):
    directories_paths = [
        DATASET_DICTS_STORE_ROOT / d
        for d in listdir(DATASET_DICTS_STORE_ROOT)
        if (DATASET_DICTS_STORE_ROOT / d).is_dir()
    ]
    training_datasets = [load_from_disk(directories_paths[0])["train"]]
    validation_datasets = [load_from_disk(directories_paths[0])["val"]]
    pretrained_model_name = "t5-base"
    trainer = GenerationModelTrainer(
        pretrained_model_name, has_encoder=True, tokenizer_max_length=1024
    )
    # model_max_length 会限制 sentence 的长度，可能会丢失一些特征
    args_output_root = Path("./training_output")
    args_output_root.mkdir(parents=True, exist_ok=True)
    trained_model, trained_tokenizer = trainer.train_model(
        {
            "output_dir": str(args_output_root / f"{model_name}"),
            "num_train_epochs": 3,
            "per_device_train_batch_size": BATCH_SIZE,
            "evaluation_strategy": "epoch",
        },
        training_datasets,
        validation_datasets,
    )

    trained_model.save_pretrained(TRAINED_MODEL_ROOT / f"{model_name}")
    trained_tokenizer.save_pretrained(TRAINED_TOKENIZER_ROOT / f"{model_name}")


def evaluate(model_name="t5-base"):
    DATASET_DICTS_STORE_ROOT = Path("./dataset_dicts")
    directories_paths = [
        DATASET_DICTS_STORE_ROOT / d
        for d in listdir(DATASET_DICTS_STORE_ROOT)
        if (DATASET_DICTS_STORE_ROOT / d).is_dir()
    ]
    training_datasets = [load_from_disk(dir_path) for dir_path in directories_paths]
    t5_model = T5ForConditionalGeneration.from_pretrained(
        TRAINED_MODEL_ROOT / f"{model_name}"
    )
    t5_tokenizer = T5Tokenizer.from_pretrained(TRAINED_TOKENIZER_ROOT / f"{model_name}")
    test_dataset = training_datasets[0]["test"]
    model_executor = GenerationModelExecutor(
        t5_model, t5_tokenizer, test_dataset, "model_input", BATCH_SIZE
    )
    t5_outputs = model_executor.make_prediction()
    evaluator = Seq2SeqEvaluator()
    metric_values = evaluator.evaluate_model(
        test_dataset, "model_output", t5_outputs, encoder_model_name="xlm-roberta-base"
    )
    print(metric_values)
    with open(RESULT_PATH / f"{model_name}.txt", "w") as result_file:
        result_file.write(f"model_name: {model_name}\n")
        result_file.write(f"batch_size: {BATCH_SIZE}\n")
        for metric_name, metric_value in metric_values.items():
            result_file.write(f"{metric_name}: {metric_value}\n")


def make_demo(model_name):
    t5_model = T5ForConditionalGeneration.from_pretrained(
        TRAINED_MODEL_ROOT / f"{model_name}"
    )
    t5_tokenizer = T5Tokenizer.from_pretrained(TRAINED_TOKENIZER_ROOT / f"{model_name}")
    t5_executor = GenerationModelExecutor(
        model=t5_model,
        tokenizer=t5_tokenizer,
    )

    # Create OpenAIInstructionParser.
    t5_prompt_parser = MockPromptSpec(task_type=TaskType.CLASSIFICATION)

    # Create Gradio interface.
    interface_t5 = create_gradio(t5_executor, t5_prompt_parser)
    return interface_t5


if __name__ == "__main__":
    prepare_data()
    BATCH_SIZE = 8
    model_name = "t5-base"
    train(model_name)
