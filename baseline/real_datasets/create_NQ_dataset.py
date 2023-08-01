import json
import datasets
from prompt2model.dataset_processor.textualize import TextualizeProcessor
from datasets import load_from_disk

import json

# Load the input file
# import json

# # Load data from the .jsonl file
# data = []
# with open('/dev.jsonl', 'r') as file:
#     for line in file:
#         data.append(json.loads(line))

# # Process data
# processed_data = [{'input_col': item['question'], 'output_col': " ".join(item['answer'])} for item in data]

# from datasets import Dataset
# import pandas as pd

# # Convert the processed_data list of dictionaries into a DataFrame
# df = pd.DataFrame(processed_data)

# # Convert the DataFrame into a Hugging Face Dataset
# hf_dataset = Dataset.from_pandas(df)


# hf_dataset.save_to_disk("./datasets/NQ")

real_test_set = load_from_disk("./datasets/NQ")
print("orginal")
print(real_test_set["input_col"][1])

dataset_dict = datasets.DatasetDict(
    {"test": real_test_set}
)
DATASET_DICTS = [dataset_dict]

INSTRUCTION = """Your task is to generate an answer to a natural question. In this task, the input is a question string. and the output is the corresponding answer string. The question can range from Math, Cultural, Social, Geometry, Biology, History, Sports, Technology, Science, and so on."""

t5_processor = TextualizeProcessor(has_encoder=True)
t5_modified_dataset_dicts = t5_processor.process_dataset_dict(
    INSTRUCTION, DATASET_DICTS
)
t5_modified_dataset_dicts[0].save_to_disk("./datasets/NQ_student_model")
dataset = load_from_disk("./datasets/NQ_student_model")
print("student model")
print(dataset["test"]["model_input"][1])

INSTRUCTION = """Your task is to generate an answer to a natural question. In this task, the input is a question string. and the output is the corresponding answer string. The question can range from Math, Cultural, Social, Geometry, Biology, History, Sports, Technology, Science, and so on.

Here are examples with input questions along with their expected outputs:

input=“Question: What is the capital city of France?"
output=“Paris”

input="Question: When was the American Declaration of Independence signed?"
output="July 4, 1776"

input="Question: What is the tallest mountain in the world?"
output="Mount Everest"

Now, complete the following input, by continuing after "Label:". Please just provide the answer, without any additional explanation or special formatting.
"""

t5_processor = TextualizeProcessor(has_encoder=True)
t5_modified_dataset_dicts = t5_processor.process_dataset_dict(
    INSTRUCTION, DATASET_DICTS
)
t5_modified_dataset_dicts[0].save_to_disk("./datasets/NQ_gpt_model")
dataset = load_from_disk("./datasets/NQ_gpt_model")
print("gpt model")
print(dataset["test"]["model_input"][1])