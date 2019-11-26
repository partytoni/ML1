import jsonlines
import pandas as pd

TRAIN_PATH = "data/train_dataset.jsonl"
TEST_PATH = "data/test_dataset_blind.jsonl"
COLUMNS = ['instructions','opt','compiler']

dataframe = pd.DataFrame(columns=COLUMNS)
#instructions, opt, compiler
with jsonlines.open(TRAIN_PATH) as reader:
    i=0
    for elem in reader:
        dataframe.loc[i] = [elem['instructions'], elem['opt'],elem['compiler']]
    
    #dataframe.to_pickle("train-lollo.pkl")

print()

