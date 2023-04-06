# import json


# data_list = []

# with open('/home/oogundep/african_qa/runs/run.xqa.bem.test.en.human_translation.mdpr.json') as f:
#     data = json.load(f)

# for key, value in data.items():
#     cl = []
#     for context in value['contexts']:
#         dd = {}
#         dd['id'] = context['docid']
#         dd['text'] = context['text']
#         dd["title"] = ""
#         dd["score"] = context['score']
#         dd["has_answer"] = context['has_answer']
#         cl.append(dd)
#     value['ctxs'] = cl
#     data_list.append(value)

# with open('/home/oogundep/african_qa/runs/reader/run.xqa.bem.test.en.human_translation.mdpr.json', 'w') as f:
#     json.dump(data_list, f)

import jsonlines
import json
from datasets import load_dataset


with open("queries/gold_passages/data_split_gold_passages_bem_test.json") as f:
    data = json.load(f)

written_set = set()
with jsonlines.open('queries/json_lines_gold/data_split_gold_passages_bem_test.json', mode='w') as writer:
    for i, json_dict in enumerate(data):
        if type(json_dict['answer_pivot']['answer_start']) == int:
            json_dict['answer_pivot']['answer_start'] = [json_dict['answer_pivot']['answer_start']]

        if type(json_dict['answer_lang']) == int:
            json_dict['answer_lang'] = str(json_dict['answer_lang'])
        
        if type(json_dict['answer_lang']) == None:
            json_dict['answer_lang'] = ""

        if json_dict['id'] not in written_set:
            writer.write(json_dict)
        
        written_set.add(json_dict['id'])

        # if i == 372:
        #     break



ds = load_dataset("json", data_files='queries/json_lines_gold/data_split_gold_passages_bem_test.json')
print(ds)



# python3 scripts/generate_translation_gold_span_file.py --input_gold_data_split queries/eval_gold_span/test.wol.human_translation.json --input_translation_directory queries/nllb_topics_new --output_directory queries/eval_gold_span --lang wol --split test --translation google_machine_translation