import json
import difflib
import os
import glob


# hotpot_train_v1.1_preprocessed.json add two more additional fields: 'numbered_context' and 'numbered_supporting_facts'
hotpot_train_file = "hotpot_train_v1.1_preprocessed.json"
with open(hotpot_train_file, "r", encoding = 'utf-8') as handle:
    hotpot_train = json.load(handle)


raw_dir = "raw_annotations/"
data = []
for file in glob.glob(raw_dir + "*.json"):
    with open(file, 'r') as fp:
        annotation = json.load(fp)
        data.extend(annotation['data'])

print(len(data))

#For each question in raw_annotations, replace its answer and supporting facts in hotpot_train_v1.1.json with annotation get new file mini_train_human.json
mini_train_human = []
for question in data:
    if (not question['answer'] or not question['supportingFacts']):
        continue
    hotpot_train_idx = question["idx"]   # the index in the order of hotpot_train_v1.1.json
    hotpot_data = hotpot_train[hotpot_train_idx]

    # replace gold answer with answer annotation
    hotpot_data['answer'] = question['answer']

    numbered_context = hotpot_data['numbered_context']  # 10 element, each is [title, [ [id1, sent1], [id2, sent2], ... ] ]
    sp_annotation = []
    for fact_id in question['supportingFacts']:
        for p in range(len(numbered_context)):
            para =  numbered_context[len(numbered_context) - 1 - p]  # enumerate in reverse order
            if (fact_id >= para[1][0][0]): # id of the first sent in a para
                title = para[0]
                para_first_sent_id = para[1][0][0]
                fact_id_in_para = fact_id - para_first_sent_id
                sp_annotation.append([title, fact_id_in_para])
                break

    # replace gold supporting facts with sp_annotation
    hotpot_data['supporting_facts'] = sp_annotation

    mini_train_human.append(hotpot_data)

print(len(mini_train_human))

with open("mini_train_human.json", 'w') as fp:
    json.dump(mini_train_human, fp)