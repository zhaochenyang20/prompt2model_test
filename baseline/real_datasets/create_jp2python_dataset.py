import datasets
from prompt2model.dataset_processor.textualize import TextualizeProcessor
from datasets import load_from_disk
from datasets import load_dataset

original_dataset = load_from_disk("./datasets/jp2python")
print("orginal")
print(original_dataset["input_col"][1])

dataset_dict = datasets.DatasetDict(
    {"test": original_dataset}
)
DATASET_DICTS = [dataset_dict]

INSTRUCTION = """Pythonで1行のコードを生成し、StackOverflowの日本語の質問を解決してください。コメントや式は含めないでください。インポート文も不要です。

このタスクでは、入力は日本語のテキストで、変数名や操作が記述されています。出力は、そのタスクを達成するためのPythonの1行のコードです。コメントや式は含めないでください。インポート文も不要です。
"""

t5_processor = TextualizeProcessor(has_encoder=True)
t5_modified_dataset_dicts = t5_processor.process_dataset_dict(
    INSTRUCTION, DATASET_DICTS
)
t5_modified_dataset_dicts[0].save_to_disk("./datasets/jp2python_student_model")
dataset = load_from_disk("./datasets/jp2python_student_model")
print("student model")
print(dataset["test"]["model_input"][1])

INSTRUCTION = """Pythonで1行のコードを生成し、StackOverflowの日本語の質問を解決してください。コメントや式は含めないでください。インポート文も不要です。

このタスクでは、入力は日本語のテキストで、変数名や操作が記述されています。出力は、そのタスクを達成するためのPythonの1行のコードです。コメントや式は含めないでください。インポート文も不要です。

input="スペースで区切られた入力`stdin`を変数に格納して表示する"
output="for line in stdin: a = line.rstrip().split(' ') print(a)"

input="リスト`word_list'内に出現する単語を数える"
output="Counter(word_list)"

input="tweepyインスタンス`api`を使い、文字列`word`を含んだツイートを検索し、結果をリストとして得る"
output="search = api.search(q=word)"

input="データベースの設定を表示する"
output="print(settings.DATABASES)"

input="ネストされているリスト`li`を見やすく表示する"
output="pprint.pprint(li)"

input="HTMLファイル'test.html'を開き、テキストオブジェクト'text'をutf-8で保存する"
output="f = open('test.html', 'w') f.write(text.encode('utf-8'))"

Now, complete the following input, by continuing after "Label:". Please just provide the answer, without any additional explanation or special formatting.
"""

t5_processor = TextualizeProcessor(has_encoder=True)
t5_modified_dataset_dicts = t5_processor.process_dataset_dict(
    INSTRUCTION, DATASET_DICTS
)
t5_modified_dataset_dicts[0].save_to_disk("./datasets/jp2python_gpt_model")
dataset = load_from_disk("./datasets/jp2python_gpt_model")
print("gpt model")
print(dataset["test"]["model_input"][1])