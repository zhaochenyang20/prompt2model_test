import datasets
from prompt2model.dataset_processor.textualize import TextualizeProcessor
from datasets import load_from_disk
from datasets import load_dataset

original_dataset = load_dataset("squad", split="validation")

def join_function(example):
    new_example = {}
    new_example["input_col"] = example["question"]
    new_example["output_col"] = " ".join(example["answers"]["text"])
    return new_example

joined_dataset = original_dataset.map(join_function)
joined_dataset.remove_columns(["question", "answers"])
joined_dataset.save_to_disk("./datasets/SQuAD")
joined_dataset = load_from_disk("./datasets/SQuAD")
print("orginal")
print(joined_dataset["input_col"][1])

dataset_dict = datasets.DatasetDict(
    {"test": joined_dataset}
)
DATASET_DICTS = [dataset_dict]

INSTRUCTION = """Your task is to generate an answer to a natural question. In this task, the input is a question string. and the output is the corresponding answer string. The question can range from Math, Cultural, Social, Geometry, Biology, History, Sports, Technology, Science, and so on."""

t5_processor = TextualizeProcessor(has_encoder=True)
t5_modified_dataset_dicts = t5_processor.process_dataset_dict(
    INSTRUCTION, DATASET_DICTS
)
t5_modified_dataset_dicts[0].save_to_disk("./datasets/SQuAD_student_model")
dataset = load_from_disk("./datasets/SQuAD_student_model")
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
t5_modified_dataset_dicts[0].save_to_disk("./datasets/SQuAD_gpt_model")
dataset = load_from_disk("./datasets/SQuAD_gpt_model")
print("gpt model")
print(dataset["test"]["model_input"][1])