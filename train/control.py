from itertools import product
import os
from pathlib import Path


logging_root = Path("logs")
logging_root.mkdir(parents=True, exist_ok=True)
model_name_list = ["google/flan-t5-base", "google/mt5-base"]
# task_name_list = ["normalization", "NQ", "Chinese2SQL"]
# task_name_list = ["normalization", "normalization_filter_NA"]
task_name_list = ["jp2python"]
# NQ_model_name_list = ["domenicrosati/QA2D-t5-base", "lmqg/flan-t5-base-squad-qag"]

# domenicrosati/QA2D-t5-base, lmqg/flan-t5-base-squad-qag for NQ

for model_name, task_name in product(model_name_list, task_name_list):
    model_store_name = model_name.split("/")[-1]
    logging_path = logging_root / f"{model_store_name}_{task_name}_real_test.log"
    command = f"sbatch --output={str(logging_path)} --gres=gpu:A6000:1 --mem=32G --time 20:00:00 main.sh {model_name} {task_name}"
    os.system(command)

# for model_name, task_name in product(NQ_model_name_list, ["NQ"]):
#     model_store_name = model_name.split("/")[-1]
#     logging_path = logging_root / f"{model_store_name}_{task_name}.log"
#     command = f"sbatch --output={str(logging_path)} --gres=gpu:A6000:1 --mem=32G --time 20:00:00 main.sh {model_name} {task_name}"
#     os.system(command)
