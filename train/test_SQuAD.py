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
logging.basicConfig(level=logging.INFO)


model_name = "google/flan-t5-base"
task_name = "NQ"
# Read the CSV file using pandas
logging.info(f"model: {model_name}, task: {task_name}")
model_store_name = model_name.split("/")[-1]
dataset_root = Path("../generation/generated_dataset/")
assert dataset_root.exists()
DATASET_DICTS_STORE_ROOT = dataset_root/f"{model_store_name}_{task_name}"
TRAINED_MODEL_ROOT = Path("/home/chenyan3/result/trained_model")
TRAINED_TOKENIZER_ROOT = Path("/home/chenyan3/result/trained_tokenizer")
RESULT_PATH = Path(f"/home/chenyan3/result/{model_store_name}_{task_name}")
DATASET_DICTS_STORE_ROOT.mkdir(parents=True, exist_ok=True)
TRAINED_MODEL_ROOT.mkdir(parents=True, exist_ok=True)
TRAINED_TOKENIZER_ROOT.mkdir(parents=True, exist_ok=True)
RESULT_PATH.mkdir(parents=True, exist_ok=True)
realistic_dataset_root = Path(
    "/home/chenyan3/prompt2model_test/baseline/real_datasets/datasets"
)
test_dataset = load_from_disk(
realistic_dataset_root / f"SQuAD_student_model"
)["test"]
BATCH_SIZE = 200
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
t5_model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
TRAINED_MODEL_ROOT / f"{model_store_name}_{task_name}"
).to(device)
t5_tokenizer = transformers.AutoTokenizer.from_pretrained(TRAINED_TOKENIZER_ROOT / f"{model_store_name}_{task_name}")
model_executor = GenerationModelExecutor(
    t5_model, t5_tokenizer, BATCH_SIZE, tokenizer_max_length=512, sequence_max_length=680
)

# No post-filter
t5_outputs = model_executor.make_prediction(test_set=datasets.Dataset.from_dict(test_dataset[:3000]), input_column="model_input")
test_dataset = datasets.Dataset.from_dict(
    {
        'input_col': test_dataset['input_col'][:3000],
        'output_col': test_dataset['output_col'][:3000],
        'model_input': test_dataset['model_input'][:3000],
        'model_output': test_dataset['model_output'][:3000],
        'output': [each.prediction for each in t5_outputs],
    }
)
test_dataset.save_to_disk(f"squad")
test_dataset.save_to_disk(f"../generation/generated_dataset/flan-t5-base_SQuAD_real_dataset_without_post_filter")
evaluator = Seq2SeqEvaluator()
metric_values = evaluator.evaluate_model(
    test_dataset, "model_output", t5_outputs, encoder_model_name="xlm-roberta-base"
)
print(metric_values)
with open(RESULT_PATH / f"{model_store_name}_SQuAD_real_dataset_without_post_filter.txt", "w") as result_file:
    result_file.write(f"model_name: {model_store_name}\n")
    result_file.write(f"task_name: {task_name}\n")
    result_file.write(f"batch_size: {BATCH_SIZE}\n")
    for metric_name, metric_value in metric_values.items():
        result_file.write(f"{metric_name}: {metric_value}\n")
