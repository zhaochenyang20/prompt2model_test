import os
from pathlib import Path


logging_root = Path("logs")
logging_root.mkdir(parents=True, exist_ok=True)
task_name_list = ["normalization", "NQ", "Chinese2SQL"]

# domenicrosati/QA2D-t5-base, lmqg/flan-t5-base-squad-qag for NQ

for task_name in ["normalization"]:
    logging_path = logging_root / f"{task_name}.log"
    command = f"sbatch --output={str(logging_path)} --mem=32G --time 20:00:00 main.sh {task_name}"
    os.system(command)
