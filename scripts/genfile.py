import json


data_list = []

with open('/home/oogundep/african_qa/runs/run.xqa.bem.test.en.human_translation.mdpr.json') as f:
    data = json.load(f)

for key, value in data.items():
    cl = []
    for context in value['contexts']:
        dd = {}
        dd['id'] = context['docid']
        dd['text'] = context['text']
        dd["title"] = ""
        dd["score"] = context['score']
        dd["has_answer"] = context['has_answer']
        cl.append(dd)
    value['ctxs'] = cl
    data_list.append(value)

with open('/home/oogundep/african_qa/runs/reader/run.xqa.bem.test.en.human_translation.mdpr.json', 'w') as f:
    json.dump(data_list, f)