import json
import datasets
from prompt2model.dataset_processor.textualize import TextualizeProcessor
from datasets import load_from_disk

with open("./temporal.json") as f:
    data = json.load(f)


inputs = []
outputs = []
for (input, output, _) in data:
    inputs.append(input)
    outputs.append(output)

real_test_set = datasets.Dataset.from_dict({
    "input_col": inputs,
    "output_col": outputs,
})

real_test_set.save_to_disk("./datasets/normalization")
dataset = load_from_disk("./datasets/normalization")
print("orginal")
print(dataset["input_col"][1])

dataset_dict = datasets.DatasetDict(
    {"test": real_test_set}
)
DATASET_DICTS = [dataset_dict]

INSTRUCTION = """Temporal date expressions are commonly used to refer to specific time periods. Your task is to identify these temporal date expressions and provide the exact dates they refer to.

For this task, the input is a string containing two specific elements: a posted date in the format "[Posted: YYYY-MM-DD]" and a sentence or statement that contains various temporal date references (e.g., early December, the end of the year, today, August, last Christmas, next Month, etc).

Your program should output a string that maps the time period references mentioned in the input to their corresponding dates, following these strict rules:

1. If temporal date references are found, the output should use either "YYYY-MM-DD", "YYYY-MM", or "YYYY" to represent the exact date.
- If multiple time period references are found, separate them using '|'.
2. If no temporal date reference is found or the referred date is ambiguous, the output should just be 'N/A', i.e., output="N/A".
"""

t5_processor = TextualizeProcessor(has_encoder=True)
t5_modified_dataset_dicts = t5_processor.process_dataset_dict(
    INSTRUCTION, DATASET_DICTS
)
t5_modified_dataset_dicts[0].save_to_disk("./datasets/normalization_student_model")
dataset = load_from_disk("./datasets/normalization_student_model")
print("student model")
print(dataset["test"]["model_input"][1])

INSTRUCTION = """Temporal date expressions are commonly used to refer to specific time periods. Your task is to identify these temporal date expressions and provide the exact dates they refer to.

For this task, the input is a string containing two specific elements: a posted date in the format "[Posted: YYYY-MM-DD]" and a sentence or statement that contains various temporal date references (e.g., early December, the end of the year, today, August, last Christmas, next Month, etc).

Your program should output a string that maps the time period references mentioned in the input to their corresponding dates, following these strict rules:

1. If temporal date references are found, the output should use either "YYYY-MM-DD", "YYYY-MM", or "YYYY" to represent the exact date.
- If multiple time period references are found, separate them using '|'.
2. If no temporal date reference is found or the referred date is ambiguous, the output should just be 'N/A', i.e., output="N/A".

Here are some examples:

input="[Posted: 1998-09-07] Tourism industry revenues reportedly dropped to $300 million last year, down from $450 million the year before."
output="last year == 1997"

input="[Posted: 2013-09-27] Eat! @mepangilinan"
output="N/A"

input="[Posted: 1989-10-30] Rated single-B-1 by Moody's Investors Service Inc. and single-B-plus by Standard amp Poor's Corp., the issue will be sold through underwriters led by Goldman, Sachs amp Co. Hertz Corp. -- $100 million of senior notes due Nov. 1, 2009, priced at par to yield 9%."
output="Nov. 1, 2009 == 2009-11-01"

input="[Posted: 2014-07-11] So out of place with this early transfer business."
output="N/A"

input="[Posted: 2013-10-25] Quote of the Day: '#Yoga is what you learn on your way down!"
output="the Day == 2013-10-25"

input="[Posted: 2021-06-15] Google plans to develop PALM 2 model in the first quarter of next year."
output="N/A"

input="[Posted: 2013-03-22] We will release a new github repository in the next three months."
output="the next three month == 2013-04"

input="[Posted: 2013-03-22] We will release a new Github repository in the next three months."
output="N/A"

input="[Posted: 2022-05-17] The company's fiscal year starts on July 1st and ends on June 30th."
output="July 1st == 2022-07-01 | June 30th == 2022-06-30"

input="[Posted: 2013-03-22] This flu season started in early December, a month earlier than usual, and peaked by the end of year."
output="N/A"

input="[Posted: 1989-10-30] The issue, which is puttable back to the company in 1999, was priced at a spread of 110 basis points above the Treasury's 10-year note."
output="1999 == 1999"

input="[Posted: 2022-04-15] The company announced that they will release their new product at the end of next month."
output="the end of next month == 2022-05-31"

input="[Posted: 2022-03-15] The teacher is going to release a new assignment in a few days."
output="N/A"

Now, complete the following input, by continuing after "Label:". Please just provide the answer, without any additional explanation or special formatting.
"""

t5_processor = TextualizeProcessor(has_encoder=True)
t5_modified_dataset_dicts = t5_processor.process_dataset_dict(
    INSTRUCTION, DATASET_DICTS
)
t5_modified_dataset_dicts[0].save_to_disk("./datasets/normalization_gpt_model")
dataset = load_from_disk("./datasets/normalization_gpt_model")
print("gpt model")
print(dataset["test"]["model_input"][1])