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

INSTRUCTION = """People often uses some temporal date expression in dalegies. I want to know the exact date of all the temporal date expression in some sentences.

For this task, the input is a string contains two specific elements: a posted date as "[Posted: YYYY-MM-DD]" and a sentence or statement with a temporal date reference to a time period (e.g., early December, the end of the year, July, August, last Christmas, next Month, etc).

The output is a string that provides a mapping between the time period references mentioned in the input and the corresponding dates. The output uses the "==" symbol to show the relationship, with the time period reference on the left and the corresponding date on the right. The date is formatted as "YYYY-MM" to represent the year and month.
"""

t5_processor = TextualizeProcessor(has_encoder=True)
t5_modified_dataset_dicts = t5_processor.process_dataset_dict(
    INSTRUCTION, DATASET_DICTS
)
t5_modified_dataset_dicts[0].save_to_disk("./datasets/normalization_student_model")
dataset = load_from_disk("./datasets/normalization_student_model")
print("student model")
print(dataset["test"]["model_input"][1])

INSTRUCTION = """People often uses some temporal date expression in dalegies. I want to know the exact date of all the temporal date expression in some sentences.

For this task, the input is a string contains two specific elements: a posted date as "[Posted: YYYY-MM-DD]" and a sentence or statement with a temporal date reference to a time period (e.g., early December, the end of the year, July, August, last Christmas, next Month, etc).

The output is a string that provides a mapping between the time period references mentioned in the input and the corresponding dates. The output uses the "==" symbol to show the relationship, with the time period reference on the left and the corresponding date on the right. The date is formatted as "YYYY-MM" to represent the year and month.

Here are some examples:

input="[Posted: 2013-03-22] This flu season started in early December, a month earlier than usual, and peaked by the end of year."
output="early December == 2012-12 | the end of year == 2012"

input="[Posted: 2017-04-21] she thought her husband devised the plan after he was fired from his job in July."
output="July == 2016-07"

input="[Posted: 2022-01-02] Raymond Roth's attorney, Brian Davis, denied in August that Roth had involved his son in the scheme."
output="August == 2021-08"

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