from datasets import load_from_disk
from pathlib import Path
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


DATASET_DICTS_STORE_ROOT = Path("./dataset_dicts")
TRAINED_MODEL_ROOT = Path("./model/trained_model")
TRAINED_TOKENIZER_ROOT = Path("./model/trained_tokenizer")
RESULT_PATH = Path("./result")
DATASET_DICTS_STORE_ROOT.mkdir(parents=True, exist_ok=True)
TRAINED_MODEL_ROOT.mkdir(parents=True, exist_ok=True)
TRAINED_TOKENIZER_ROOT.mkdir(parents=True, exist_ok=True)
RESULT_PATH.mkdir(parents=True, exist_ok=True)


def prepare_data():
    # Read the CSV file using pandas
    df = pd.read_csv("reviews.csv")
    # Convert all columns to string
    df = df.astype(str)
    # Shuffle the dataset and reset the index
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    # Split the dataset into train, validation, and test sets
    train_data, test_data, train_labels, test_labels = train_test_split(
        df["review_text"], df["rating"], test_size=0.05, random_state=42
    )
    train_data, val_data, train_labels, val_labels = train_test_split(
        train_data, train_labels, test_size=0.1, random_state=42
    )

    # Create the Hugging Face dataset for each split
    train_dataset = Dataset.from_dict(
        {"output_col": train_labels.tolist(), "input_col": train_data.tolist()}
    )
    val_dataset = Dataset.from_dict(
        {"output_col": val_labels.tolist(), "input_col": val_data.tolist()}
    )
    test_dataset = Dataset.from_dict(
        {"output_col": test_labels.tolist(), "input_col": test_data.tolist()}
    )

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
    dataset_dict.save_to_disk("./review")

    DATASET_DICTS = [dataset_dict]

    INSTRUCTION = "Given a product review, predict the sentiment score associated with it. The sentiment score label ranges from 1 to 5. Just give me the label."
    t5_tokenizer = T5Tokenizer.from_pretrained("t5-small")
    t5_processor = TextualizeProcessor(
        has_encoder=True, eos_token=t5_tokenizer.eos_token
    )
    t5_modified_dataset_dicts = t5_processor.process_dataset_dict(
        INSTRUCTION, DATASET_DICTS
    )

    for index, dataset_dict in enumerate(t5_modified_dataset_dicts):
        store_path = DATASET_DICTS_STORE_ROOT / f"{index}"
        dataset_dict.save_to_disk(str(store_path))


def train(model_name, toy_test=False):
    directories_paths = [
        DATASET_DICTS_STORE_ROOT / d
        for d in listdir(DATASET_DICTS_STORE_ROOT)
        if (DATASET_DICTS_STORE_ROOT / d).is_dir()
    ]
    training_dataset = load_from_disk(directories_paths[0])["train"]
    training_datasets = (
        [Dataset.from_dict(training_dataset[:5000])] if toy_test else [training_dataset]
    )
    pretrained_model_name = f"{model_name}"
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
    )

    trained_model.save_pretrained(TRAINED_MODEL_ROOT / f"{model_name}")
    trained_tokenizer.save_pretrained(TRAINED_TOKENIZER_ROOT / f"{model_name}")


def remove_eos_token(dataset):
    def map_function(example):
        example["output_col"] = example["output_col"][:1]
        return example

    return dataset.map(map_function)


def evaluate(model_name, toy_test=False):
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
    test_dataset = (
        Dataset.from_dict(training_datasets[0]["test"][:3000])
        if toy_test
        else training_datasets[0]["test"]
    )
    model_executor = GenerationModelExecutor(
        t5_model, t5_tokenizer, test_dataset, "model_input", BATCH_SIZE
    )
    t5_outputs = model_executor.make_prediction()
    evaluator = Seq2SeqEvaluator()
    metric_values = evaluator.evaluate_model(
        test_dataset, "output_col", t5_outputs, encoder_model_name="xlm-roberta-base"
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
    model_name = "t5-small"
    train(model_name, toy_test=True)
