from prompt2model.utils.openai_tools import ChatGPTAgent
from datasets import load_from_disk
import datasets
from prompt2model.dataset_processor.textualize import TextualizeProcessor
from datasets import DatasetDict
from pathlib import Path
import logging
import argparse
from prompt2model.model_evaluator import Seq2SeqEvaluator
from prompt2model.utils import OPENAI_ERRORS
import time
logging.basicConfig(level=logging.INFO)

result_path = Path("/home/chenyan3/beta-test/train/datasets")
DATASET_DICTS_STORE_ROOT = Path("./datasets")
task_name_list = ["normalization", "NQ", "Chinese2SQL"]


def evaluate_with_gpt(task_name):
    # Read the CSV file using pandas
    dataset = load_from_disk(result_path / task_name)
    test_dataset = datasets.Dataset.from_dict(dataset[4000:5000])

    # Create the DatasetDict
    dataset_dict = DatasetDict({"test": test_dataset})

    DATASET_DICTS = [dataset_dict]

    if task_name == "normalization":
        INSTRUCTION = """People often uses some temporal date expression in dalegies. I want to know the exact date of all the temporal date expression in some sentences.

For this task, the input is a string contains two specific elements: a posted date as "[Posted: YYYY-MM-DD]" and a sentence or statement with a temporal date reference to a time period (e.g., early December, the end of the year, July, August, last Christmas, next Month, etc).

The output is a string that provides a mapping between the time period references mentioned in the input and the corresponding dates. The output uses the "==" symbol to show the relationship, with the time period reference on the left and the corresponding date on the right. The date is formatted as "YYYY-MM" to represent the year and month.

Here are some example:

input="[Posted: 2013-03-22] This flu season started in early December, a month earlier than usual, and peaked by the end of year."
output="early December == 2012-12 | the end of year == 2012"

input="[Posted: 2017-04-21] she thought her husband devised the plan after he was fired from his job in July."
output="July == 2016-07"

input="[Posted: 2022-01-02] Raymond Roth's attorney, Brian Davis, denied in August that Roth had involved his son in the scheme."
output="August == 2021-08"

Now, complete the following input, by continuing after "Label:". Please just provide the answer, without any additional explanation or special formatting.

"""
    elif task_name == "NQ":
        INSTRUCTION = """
Your task is to generate an answer to a natural question. In this task, the input is a question string. and the output is the corresponding answer string. The question can range from Math, Cultural, Social, Geometry, Biology, History, Sports, Technology, Science, and so on.

Here are examples with input questions along with their expected outputs:

input=“Question: What is the capital city of France?"
output=“Paris”

input="Question: When was the American Declaration of Independence signed?"
output="July 4, 1776"

input="Question: What is the tallest mountain in the world?"
output="Mount Everest"

Now, complete the following input, by continuing after "Label:". Please just provide the answer, without any additional explanation or special formatting.

"""
    elif task_name == "Chinese2SQL":
        INSTRUCTION = """Chinese2SQL is an NLP task that involves converting natural language queries written in Chinese into SQL queries for querying relational databases.

For this task, the input is a Chinese string that describes a natural language query. The output is the corresponding SQL query.

Here are some example:

input="北京市的人口是多少？"
output="SELECT population FROM cities WHERE city_name = '北京市'"

input="查询销售额大于10000的产品。"
output="SELECT * FROM products WHERE sales > 10000"

input="显示2019年至今的订单数量。"
output="SELECT COUNT(*) FROM orders WHERE order_date >= '2019-01-01'"

Now, complete the following input, by continuing after "Label:". Please just provide the answer, without any additional explanation or special formatting.

"""
    processor = TextualizeProcessor(has_encoder=True)
    modified_dataset_dicts = processor.process_dataset_dict(INSTRUCTION, DATASET_DICTS)
    modified_dataset_dicts[0].save_to_disk(DATASET_DICTS_STORE_ROOT / task_name)
    test_dataset = modified_dataset_dicts[0]["test"]
    chat_api = ChatGPTAgent()
    outputs = []
    evaluate_length = 10
    # for idx in range(len(test_dataset)):
    for idx in range(evaluate_length):
        api_call_counter = 0
        model_input = test_dataset[idx]["model_input"]
        input_col = test_dataset[idx]["input_col"]
        while True:
            try:
                api_call_counter += 1
                if api_call_counter >= 4:
                    logging.info(f"index: {idx}")
                    logging.info(f"input: {input_col}")
                    logging.info(f"output: None")
                    outputs.append("")
                    break
                response = chat_api.generate_one_openai_chat_completion(model_input)
                output = response.choices[0]["message"]["content"]
                outputs.append(output)
                logging.info(f"index: {idx}")
                logging.info(f"input: {input_col}")
                logging.info(f"output: {output}")
                break
            except OPENAI_ERRORS as e:
                logging.error(e)
                time.sleep(1)
    result_dataset = datasets.Dataset.from_dict(
        {
            "input_col": test_dataset["input_col"][:evaluate_length],
            "model_input": test_dataset["model_input"][:evaluate_length],
            "output_col": test_dataset["output_col"][:evaluate_length],
            "model_output": test_dataset["model_output"][:evaluate_length],
            "output": outputs,
        }
    )
    result_root = Path("results")
    result_dataset.save_to_disk(result_root / task_name)

def main():
    parser = argparse.ArgumentParser(description="Evaluate GPT model.")
    parser.add_argument(
        "--task_name", type=str, required=True, help="Name of the task."
    )
    args = parser.parse_args()

    evaluate_with_gpt(args.task_name)


if __name__ == "__main__":
    main()