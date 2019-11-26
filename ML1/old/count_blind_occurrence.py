import jsonlines
TRAIN_PATH = 'data/train_dataset.jsonl'
TEST_PATH = 'data/test_dataset_blind.jsonl'

train_set = set()
test_set = set()
with jsonlines.open(TRAIN_PATH) as reader:
    for elem in reader:
        riga = ""
        for instruction in elem['instructions']:
            first = instruction.split(" ")[0]
            train_set.add(first)


with jsonlines.open(TEST_PATH) as reader:
    for elem in reader:
        riga = ""
        for instruction in elem['instructions']:
            first = instruction.split(" ")[0]
            test_set.add(first)

print(len(train_set), len(test_set))
for elem in test_set:
    if elem not in train_set:
        print(elem, "not in train_set")
