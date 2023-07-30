from prompt2model.prompt_parser import OpenAIInstructionParser, TaskType
from prompt2model.dataset_generator.base import DatasetSplit
from prompt2model.dataset_generator.openai_gpt import OpenAIDatasetGenerator
import os
import openai
import logging

logging.basicConfig(level=logging.INFO)

prompt = """
Your task is to generate an answer to a natural question. In this task, the input is a question string. and the output is the corresponding answer string. The question can range from Math, Cultural, Social, Geometry, Biology, History, Sports, Technology, Science, and so on.

Here are examples with input questions along with their expected outputs:

input=“Question: What is the capital city of France?"
output=“Paris”

input="Question: When was the American Declaration of Independence signed?"
output="July 4, 1776"

input="Question: What is the tallest mountain in the world?"
output="Mount Everest"
"""
prompt_spec = OpenAIInstructionParser(task_type=TaskType.TEXT_GENERATION)
prompt_spec.parse_from_prompt(prompt)


def generate_and_save_dataset(
    temp: float = 1,
    presence_pen: float = 0,
    freq_pen: float = 0,
    filename: str = "./QA_dataset",
):
    openai.api_key = os.environ["OPENAI_API_KEY"]
    logging.basicConfig(level=logging.INFO)
    dataset_generator = OpenAIDatasetGenerator(
        filter_duplicated_examples=True,
        temperature=temp,
        presence_penalty=presence_pen,
        frequency_penalty=freq_pen,
        batch_size=5,
        responses_per_request=5,
    )
    QA_dataset = dataset_generator.generate_dataset_split(
        prompt_spec, 200, split=DatasetSplit.TRAIN
    )
    print(len(set(QA_dataset["output_col"])))
    QA_dataset.save_to_disk(filename)
    return dataset_generator, QA_dataset


test_configs = [
    (1.7, 0, 0, "./test_filter"),
]

for test_config in test_configs:
    logging.info(test_config)
    dataset_generator, QA_dataset = generate_and_save_dataset(*test_config)
