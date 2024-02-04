from datasets import load_from_disk
from pathlib import Path
import datasets
from prompt2model.model_trainer.generate import GenerationModelTrainer
from prompt2model.dataset_processor.textualize import TextualizeProcessor
from datasets import DatasetDict, concatenate_datasets
from pathlib import Path
import logging
import torch
import argparse
import transformers
from prompt2model.model_evaluator import Seq2SeqEvaluator
from prompt2model.model_executor import GenerationModelExecutor, ModelOutput
from prompt2model.utils.path import TEST_DATA_ROOT

logging.basicConfig(level=logging.INFO)


def train(evaluate=True, realistic=True):
    model_name = "google/mt5-base"
    task_name = "cbq"
    logging.info(f"model: {model_name}, task: {task_name}")
    model_store_name = model_name.split("/")[-1]
    dataset_root = Path("../generation/generated_dataset/")
    assert dataset_root.exists()
    TRAINED_MODEL_ROOT = Path("/home/chenyan3/result/trained_model")
    TRAINED_TOKENIZER_ROOT = Path("/home/chenyan3/result/trained_tokenizer")
    RESULT_PATH = Path(f"/home/chenyan3/result/{model_store_name}_{task_name}")
    TRAINED_MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    TRAINED_TOKENIZER_ROOT.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.mkdir(parents=True, exist_ok=True)
    dataset = load_from_disk(dataset_root / task_name)
    train_dataset = dataset["train"]
    val_dataset = dataset["val"]
    test_dataset = dataset["test"]

    # Create the DatasetDict
    dataset_dict = DatasetDict(
        {"train": train_dataset, "val": val_dataset, "test": test_dataset}
    )

    DATASET_DICTS = [dataset_dict]

    INSTRUCTION = """Your task is to generate an answer to a natural question. In this task, the input is a string that consists of both a question and a context passage. The context is a descriptive passage related to the question and contains the answer. And the question can range from Math, Cultural, Social, Geometry, Biology, History, Sports, Technology, Science, and so on."""
    t5_processor = TextualizeProcessor(has_encoder=True)
    t5_modified_dataset_dicts = t5_processor.process_dataset_dict(
        INSTRUCTION, DATASET_DICTS
    )
    training_datasets = [
        t5_modified_dataset_dicts[0]["train"],
    ]
    validation_datasets = [
        t5_modified_dataset_dicts[0]["val"],
    ]
    test_datasets = [
        t5_modified_dataset_dicts[0]["test"],
    ]
    trainer = GenerationModelTrainer(
        model_name,
        has_encoder=True,
        executor_batch_size=8,
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
            "per_device_train_batch_size": 8,
            "evaluation_strategy": "no",
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
        test_dataset = concatenate_datasets(test_datasets)
        BATCH_SIZE = 8
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
            "cbq_on_testset"
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
            / f"{task_name}_generated_dataset_without_post_filter.txt",
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
        real_task_name = "SQuAD"
        test_dataset = load_from_disk(
            realistic_dataset_root / f"{real_task_name}_student_model"
        )["test"]
        BATCH_SIZE = 8
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
            f"cbq_on_real_dataset"
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


if __name__ == "__main__":
    train()
