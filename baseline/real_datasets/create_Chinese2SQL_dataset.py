import json
import datasets
from prompt2model.dataset_processor.textualize import TextualizeProcessor
from datasets import load_from_disk

import json

# Load the input file
with open('dev.json', 'r') as f:
    data = json.load(f)

# Extract all question-query pairs
question_query_pairs = [{'question': item['question'], 'query': item['query']} for item in data]

# Write the pairs into a new JSON file
with open('question_query_pairs.json', 'w') as f:
    json.dump(question_query_pairs, f, ensure_ascii=False, indent=4)

from datasets import Dataset
import pandas as pd

# Convert the list of dictionaries into a DataFrame
df = pd.DataFrame(question_query_pairs)

# Convert the DataFrame into a Dataset
dataset = Dataset.from_pandas(df)

# Set the input and output columns
dataset = dataset.rename_column("question", "input_col")
dataset = dataset.rename_column("query", "output_col")

dataset.save_to_disk("./datasets/Chinese2SQL")

real_test_set = load_from_disk("./datasets/Chinese2SQL")
print("orginal")
print(real_test_set["input_col"][1])

dataset_dict = datasets.DatasetDict(
    {"test": real_test_set}
)
DATASET_DICTS = [dataset_dict]

INSTRUCTION = """Chinese2SQL is an NLP task that involves converting natural language queries written in Chinese into SQL queries for querying relational databases.

For this task, the input is a Chinese string that describes a natural language query. The output is the corresponding SQL query.
"""

t5_processor = TextualizeProcessor(has_encoder=True)
t5_modified_dataset_dicts = t5_processor.process_dataset_dict(
    INSTRUCTION, DATASET_DICTS
)
t5_modified_dataset_dicts[0].save_to_disk("./datasets/Chinese2SQL_student_model")
dataset = load_from_disk("./datasets/Chinese2SQL_student_model")
print("student model")
print(dataset["test"]["model_input"][1])

INSTRUCTION = """Chinese2SQL is an NLP task that involves converting natural language queries written in Chinese into SQL queries for querying relational databases.

For this task, the input is a Chinese string that describes a natural language query. The output is the corresponding SQL query.

Here are some examples:

input="北京市的人口是多少？"
output="SELECT population FROM cities WHERE city_name = '北京市'"

input="查询销售额大于10000的产品。"
output="SELECT * FROM products WHERE sales > 10000"

input="显示2019年至今的订单数量。"
output="SELECT COUNT(*) FROM orders WHERE order_date >= '2019-01-01'"

Now, complete the following input, by continuing after "Label:". Please just provide the answer, without any additional explanation or special formatting.
"""

t5_processor = TextualizeProcessor(has_encoder=True)
t5_modified_dataset_dicts = t5_processor.process_dataset_dict(
    INSTRUCTION, DATASET_DICTS
)
t5_modified_dataset_dicts[0].save_to_disk("./datasets/Chinese2SQL_gpt_model")
dataset = load_from_disk("./datasets/Chinese2SQL_gpt_model")
print("gpt model")
print(dataset["test"]["model_input"][1])