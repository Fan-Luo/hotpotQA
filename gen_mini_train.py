import json

data_index_file = "../hotpotanalyze/answered_questions.json"
hotpot_train_file = "hotpot_train_v1.1.json"
raw_dir = "raw_annotations/"

with open(data_index_file, "r", encoding = 'utf-8') as handle:
    data_index = json.load(handle)
with open(hotpot_train_file, "r", encoding = 'utf-8') as handle:
    hotpot_train = json.load(handle)

mini_train = []
for index in data_index:
	mini_train.append(hotpot_train[index])

with open("mini_train.json", 'w') as fp:
    json.dump(mini_train, fp)