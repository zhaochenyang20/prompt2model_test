from prompt2model.prompt_parser import OpenAIInstructionParser, TaskType
import os
import openai
from prompt2model.dataset_generator.base import DatasetSplit
from prompt2model.dataset_generator.openai_gpt import OpenAIDatasetGenerator
import logging

openai.api_key = os.environ["OPENAI_API_KEY"]
# logging.basicConfig(level=logging.INFO)

prompt = """People often uses some temporal date expression in dalegies. I want to know the exact date of all the temporal date expression in some sentences.

For this task, the input is a string contains two specific elements: a posted date as "[Posted: YYYY-MM-DD]" and a sentence or statement with a temporal date reference to a time period (e.g., early December, the end of the year, July, August, last Christmas, next Month, etc).

The output is a string that provides a mapping between the time period references mentioned in the input and the corresponding dates. The output uses the "==" symbol to show the relationship, with the time period reference on the left and the corresponding date on the right. The date is formatted as "YYYY-MM" to represent the year and month.

Here are some example:

input="[Posted: 2013-03-22] This flu season started in early December, a month earlier than usual, and peaked by the end of year."
output="early December == 2012-12 | the end of year == 2012"

input="[Posted: 2017-04-21] she thought her husband devised the plan after he was fired from his job in July."
output="July == 2016-07"

input="[Posted: 2022-01-02] Raymond Roth's attorney, Brian Davis, denied in August that Roth had involved his son in the scheme."
output="August == 2021-08"
"""
prompt_spec = OpenAIInstructionParser(task_type=TaskType.TEXT_GENERATION)
prompt_spec.parse_from_prompt(prompt)
unlimited_dataset_generator = OpenAIDatasetGenerator(
    temperature=1.4, requests_per_minute=90, responses_per_request=5
)
normalized_dataset = unlimited_dataset_generator.generate_dataset_split(
    prompt_spec, 2000, split=DatasetSplit.TRAIN
)
normalized_dataset.save_to_disk("normalized_dataset_2000")
