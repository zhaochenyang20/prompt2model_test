from itertools import product
import os
from pathlib import Path


logging_root = Path("logs")
logging_root.mkdir(parents=True, exist_ok=True)
model_name_list = [# 'facebook/bart-large-cnn',
 # 'pszemraj/led-base-book-summary',
'pszemraj/long-t5-tglobal-base-16384-book-summary',
'pszemraj/led-large-book-summary',
]
task_name_list = ["normalization_0.3_1.4_with_filtering", "SQuAD_0.3_1.4_with_filtering", "jp2python"]

# domenicrosati/QA2D-t5-base, lmqg/flan-t5-base-squad-qag for NQ

for model_name, task_name in product(model_name_list, task_name_list):
    model_store_name = model_name.split("/")[-1]
    logging_path = logging_root / f"{model_store_name}_{task_name}.log"
    command = f"sbatch --output={str(logging_path)} --gres=gpu:A6000:1 --mem=32G --time 20:00:00 main.sh {model_name} {task_name}"
    os.system(command)
