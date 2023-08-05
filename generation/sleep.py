import time

print(f"sleep for {60 * 60 * 2} seconds")

for i in range(60 * 60 * 2):
    time.sleep(1)

import os
print("os")
os.system("sbatch --mem=32G --time 20:00:00 main.sh SQuAD.py")