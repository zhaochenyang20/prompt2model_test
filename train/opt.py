from datasets import load_from_disk
from pathlib import Path
from os import listdir
import datasets
from prompt2model.model_trainer.generate import GenerationModelTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
from prompt2model.model_executor import GenerationModelExecutor
from prompt2model.model_evaluator import Seq2SeqEvaluator
from prompt2model.model_executor import GenerationModelExecutor
from prompt2model.dataset_processor.textualize import TextualizeProcessor
from datasets import DatasetDict, Dataset
from sklearn.model_selection import train_test_split
import pandas as pd
from pathlib import Path
import logging
from prompt2model.prompt_parser import MockPromptSpec, TaskType
from prompt2model.demo_creator import create_gradio
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO)

DATASET_DICTS_STORE_ROOT = Path("./dataset_dicts_opt")
TRAINED_MODEL_ROOT = Path("./model/trained_model_opt")
TRAINED_TOKENIZER_ROOT = Path("./model/trained_tokenizer_opt")
RESULT_PATH = Path("./result_opt")
CKPT_PATH = Path("./ckpt_opt")
DATASET_DICTS_STORE_ROOT.mkdir(parents=True, exist_ok=True)
TRAINED_MODEL_ROOT.mkdir(parents=True, exist_ok=True)
TRAINED_TOKENIZER_ROOT.mkdir(parents=True, exist_ok=True)
RESULT_PATH.mkdir(parents=True, exist_ok=True)
CKPT_PATH.mkdir(parents=True, exist_ok=True)
PRETRAINER_MODEL_NAME = "facebook/opt-iml-1.3b"
model_name = PRETRAINER_MODEL_NAME


def prepare_data():
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

    opt_tokenizer = AutoTokenizer.from_pretrained(
        PRETRAINER_MODEL_NAME, padding_side="left"
    )
    opt_processor = TextualizeProcessor(
        has_encoder=False, eos_token=opt_tokenizer.eos_token
    )
    opt_modified_dataset_dicts = opt_processor.process_dataset_dict(
        INSTRUCTION, DATASET_DICTS
    )

    for index, dataset_dict in enumerate(opt_modified_dataset_dicts):
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
    pretrained_model_name = "facebook/opt-iml-1.3b"
    trainer = GenerationModelTrainer(
        pretrained_model_name, has_encoder=False, tokenizer_max_length=1024
    )
    trained_model, trained_tokenizer = trainer.train_model(
        {
            "output_dir": CKPT_PATH,
            "num_train_epochs": 3,
            "per_device_train_batch_size": BATCH_SIZE,
            "save_strategy": "epoch",
        },
        training_datasets,
        validation_datasets,
    )

    trained_model.save_pretrained(TRAINED_MODEL_ROOT / f"{model_name}")
    trained_tokenizer.save_pretrained(TRAINED_TOKENIZER_ROOT / f"{model_name}")


def evaluate(model_name, toy_test=False):
    directories_paths = [
        DATASET_DICTS_STORE_ROOT / d
        for d in listdir(DATASET_DICTS_STORE_ROOT)
        if (DATASET_DICTS_STORE_ROOT / d).is_dir()
    ]
    training_datasets = [load_from_disk(dir_path) for dir_path in directories_paths]
    gpt2_model = AutoModelForCausalLM.from_pretrained(
        TRAINED_MODEL_ROOT / f"{model_name}"
    )
    gpt2_tokenizer = AutoTokenizer.from_pretrained(
        TRAINED_TOKENIZER_ROOT / f"{model_name}"
    )
    test_dataset = (
        Dataset.from_dict(training_datasets[0]["test"][:200])
        if toy_test
        else training_datasets[0]["test"]
    )
    model_executor = GenerationModelExecutor(
        gpt2_model, gpt2_tokenizer, test_dataset, "model_input", BATCH_SIZE
    )
    gpt2_outputs = model_executor.make_prediction()
    evaluator = Seq2SeqEvaluator()
    metric_values = evaluator.evaluate_model(
        test_dataset, "output_col", gpt2_outputs, encoder_model_name="xlm-roberta-base"
    )
    with open(RESULT_PATH / f"{model_name}.txt", "w") as result_file:
        result_file.write(f"model_name: {model_name}\n")
        result_file.write(f"batch_size: {BATCH_SIZE}\n")
        for metric_name, metric_value in metric_values.items():
            result_file.write(f"{metric_name}: {metric_value}\n")


def make_demo(model_name):
    opt_model = AutoModelForCausalLM.from_pretrained(
        TRAINED_MODEL_ROOT / f"{model_name}"
    )
    opt_tokenizer = AutoTokenizer.from_pretrained(
        TRAINED_TOKENIZER_ROOT / f"{model_name}"
    )
    opt_executor = GenerationModelExecutor(
        model=opt_model,
        tokenizer=opt_tokenizer,
        batch_size=1,
    )
    # Create OpenAIInstructionParser.
    opt_prompt_parser = MockPromptSpec(task_type=TaskType.CLASSIFICATION)

    # Create Gradio interface.
    interface_opt = create_gradio(opt_executor, opt_prompt_parser)
    interface_opt.launch(share=True)
    return interface_opt


if __name__ == "__main__":
    prepare_data()
    BATCH_SIZE = 2
    model_name = "facebook/opt-iml-1.3b"
    train(model_name, toy_test=True)
