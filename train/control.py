from itertools import product
import os
from pathlib import Path


logging_root = Path("logs")
logging_root.mkdir(parents=True, exist_ok=True)
model_name_list = ["google/flan-t5-base", "google/mt5-base", "facebook/bart-large-cnn"]
model_name_list = ["facebook/bart-large-cnn", "bigscience/bloomz-1b1", "facebook/opt-iml-max-1.3b"]
task_name_list = ["normalization", "SQuAD", "jp2python"]

# domenicrosati/QA2D-t5-base, lmqg/flan-t5-base-squad-qag for NQ

for model_name, task_name in product(["facebook/bart-large-cnn"], ["normalization"]):
    model_store_name = model_name.split("/")[-1]
    logging_path = logging_root / f"{model_store_name}_{task_name}_real_test.log"
    command = f"sbatch --output={str(logging_path)} --gres=gpu:A6000:1 --mem=32G --time 20:00:00 main.sh {model_name} {task_name}"
    os.system(command)
