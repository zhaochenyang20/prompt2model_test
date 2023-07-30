from prompt2model.prompt_parser import OpenAIInstructionParser, TaskType
import os
import openai

openai.api_key = os.environ["OPENAI_API_KEY"]


prompt = """Translate some traditional Chinese Poem to English."""
prompt_spec = OpenAIInstructionParser(task_type=TaskType.TEXT_GENERATION)
prompt_spec.parse_from_prompt(prompt)
assert prompt_spec.task_type == TaskType.TEXT_GENERATION

prompt = """People often uses some temporal expressions in dalegies. I want to know the exact meaning of these expressions. Here is one example:

ASAP - As Soon As Possible
FYI - For Your Information
ETA - Estimated Time of Arrival
EOD - End of Day
TBA - To Be Announced
TBD - To Be Determined
BTW - By the Way
AFAIK - As Far As I Know
IMO - In My Opinion
TTYL - Talk to You Later

I'll send you the report ASAP so you can review it.
Meaning: I will send you the report as soon as possible so that you can review it.

Please submit the project update by EOD today.
"""
prompt_spec = OpenAIInstructionParser(task_type=TaskType.TEXT_GENERATION)
prompt_spec.parse_from_prompt(prompt)
print(prompt_spec.examples)
print(prompt_spec.instruction)


prompt = """Translate some traditional Chinese Poem to English. Here is one example:
茅屋为秋风所破歌

八月秋高风怒号，
卷我屋上三重茅。
茅飞渡江洒江郊，
高者挂罥长林梢，
下者飘转沉塘坳。

南村群童欺我老无力，
忍能对面为盗贼。
公然抱茅入竹去，
唇焦口燥呼不得，
归来倚杖自叹息。

俄顷风定云墨色，
秋天漠漠向昏黑。
布衾多年冷似铁，
娇儿恶卧踏里裂。
床头屋漏无干处，
雨脚如麻未断绝。
自经丧乱少睡眠，
长夜沾湿何由彻。

安得广厦千万间，
大庇天下寒士俱欢颜！
风雨不动安如山。
呜呼！何时眼前突兀见此屋，
吾庐独破受冻死亦足！

Song of the Thatched Cottage Ravaged by Autumn Wind

In August, when autumn's high winds howl,
They sweep across my thatched cottage with fury.
Thatch flies, crossing the river, scattering in the outskirts,
The tall ones hang on the tips of distant trees,
The lower ones drift and sink in marshy hollows.

The village children in the south mock my feeble old age,
Yet, how could I lower myself to be a thief in response?
Boldly, I carry my thatch and venture into the bamboo grove,
My lips parched, my throat dry, unable to call out,
Returning, I lean on my staff and sigh.

In an instant, the wind subsides, the clouds turn dark,
Autumn spreads dimly, approaching dusk.
My quilt, cold as iron after years of use,
My delicate child, unwillingly sleeps on a torn mat.
The roof leaks by my pillow, with no dry spot,
The raindrops incessant, like countless fibers.
Through troubled times, I've had little sleep,
The long night dampens, how can it ever clear?

If only I had a grand mansion with a thousand rooms,
Providing shelter and happiness to all the destitute!
Unmoved by wind and rain, stable as a mountain.
Alas! When will I suddenly see such a house before me,
My humble dwelling alone, shattered and freezing, would be enough to die!
"""
prompt_spec = OpenAIInstructionParser(task_type=TaskType.TEXT_GENERATION)
prompt_spec.parse_from_prompt(prompt)
print(prompt_spec.examples)
print(prompt_spec.instruction)


prompt = """Translate some traditional Chinese Poem to English. Here is one example:
《静夜思》 - 李白
床前明月光，疑是地上霜。
举头望明月，低头思故乡。
Translation:
Beside my bed, the moonlight shines so bright,
I wonder if it's frost upon the ground.
I raise my head to gaze at the moon's light,
Then lower it, missing my hometown profound.

茅屋为秋风所破歌

八月秋高风怒号，
卷我屋上三重茅。
茅飞渡江洒江郊，
高者挂罥长林梢，
下者飘转沉塘坳。

南村群童欺我老无力，
忍能对面为盗贼。
公然抱茅入竹去，
唇焦口燥呼不得，
归来倚杖自叹息。

俄顷风定云墨色，
秋天漠漠向昏黑。
布衾多年冷似铁，
娇儿恶卧踏里裂。
床头屋漏无干处，
雨脚如麻未断绝。
自经丧乱少睡眠，
长夜沾湿何由彻。

安得广厦千万间，
大庇天下寒士俱欢颜！
风雨不动安如山。
呜呼！何时眼前突兀见此屋，
吾庐独破受冻死亦足！

Song of the Thatched Cottage Ravaged by Autumn Wind

In August, when autumn's high winds howl,
They sweep across my thatched cottage with fury.
Thatch flies, crossing the river, scattering in the outskirts,
The tall ones hang on the tips of distant trees,
The lower ones drift and sink in marshy hollows.

The village children in the south mock my feeble old age,
Yet, how could I lower myself to be a thief in response?
Boldly, I carry my thatch and venture into the bamboo grove,
My lips parched, my throat dry, unable to call out,
Returning, I lean on my staff and sigh.

In an instant, the wind subsides, the clouds turn dark,
Autumn spreads dimly, approaching dusk.
My quilt, cold as iron after years of use,
My delicate child, unwillingly sleeps on a torn mat.
The roof leaks by my pillow, with no dry spot,
The raindrops incessant, like countless fibers.
Through troubled times, I've had little sleep,
The long night dampens, how can it ever clear?

If only I had a grand mansion with a thousand rooms,
Providing shelter and happiness to all the destitute!
Unmoved by wind and rain, stable as a mountain.
Alas! When will I suddenly see such a house before me,
My humble dwelling alone, shattered and freezing, would be enough to die!
"""
prompt_spec = OpenAIInstructionParser(task_type=TaskType.TEXT_GENERATION)
prompt_spec.parse_from_prompt(prompt)
print(prompt_spec.examples)
print(prompt_spec.instruction)


prompt = """People often uses some temporal date expression in dalegies. I want to know the exact date. Here are some example:

[Posted: 2013-03-22] This flu season started in early December, a month earlier than usual, and peaked by the end of year.
early December == 2012-12 | the end of year == 2012

[Posted: 2013-03-22] she thought her husband devised the plan after he was fired from his job in July.
July == 2012-07

[Posted: 2013-03-22] Raymond Roth's attorney, Brian Davis, denied in August that Roth had involved his son in the scheme.
August == 2012-08
"""
prompt_spec = OpenAIInstructionParser(task_type=TaskType.TEXT_GENERATION)
prompt_spec.parse_from_prompt(prompt)
print(prompt_spec.examples)
print(prompt_spec.instruction)


prompt = """People often uses some temporal date expression in dialogues. I want to know the exact date, i.e. normalization. Here are some example:

Input: [Posted: 2013-03-22] This flu season started in early December, a month earlier than usual, and peaked by the end of year.
Output: early December == 2012-12 | the end of year == 2012

Input: [Posted: 2013-03-22] she thought her husband devised the plan after he was fired from his job in July.
Output:

Input: [Posted: 2013-03-22] Raymond Roth's attorney, Brian Davis, denied in August that Roth had involved his son in the scheme.
Output:
"""
prompt_spec = OpenAIInstructionParser(task_type=TaskType.TEXT_GENERATION)
prompt_spec.parse_from_prompt(prompt)
print(prompt_spec.instruction)
print(prompt_spec.examples)


prompt = """QA is a type of question-answer pair. I want to learn some Chinese cultures from Chinese QA pairs. PleaseHere are some example:

Question: 四川省的省会是？
Answer: 成都市

Question: 清华大学建立于哪一年？
Answer: 1911 年

Question: 中国最好的大学是？
Answer:
"""
prompt_spec = OpenAIInstructionParser(task_type=TaskType.TEXT_GENERATION)
prompt_spec.parse_from_prompt(prompt)
print(prompt_spec.instruction)
print(prompt_spec.examples)
